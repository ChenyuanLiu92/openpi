"""Pretrain pi0.5 from a heterogeneous sample-level RLDS mixture.

Run with the optional RLDS dependencies installed:

    uv run --group rlds scripts/pretrain.py configs/pretraining/pi05/template.yaml
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import platform
import shutil
import signal
import sys
import threading
import time
import traceback

from etils import epath
from flax.training import common_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm

from openpi.shared import array_typing as at
from openpi.training import checkpoints
from openpi.training import gpu_collectives
from openpi.training import observability
from openpi.training import pretrain_config
from openpi.training import pretrain_config_loader
from openpi.training import rlds_mixture
from openpi.training import sharding
from openpi.training import trainer
from openpi.training import utils as training_utils

_DISTRIBUTED_INITIALIZED = False


def _init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    root.handlers[0].setFormatter(formatter)


def _initialize_distributed(config: pretrain_config.DistributedConfig) -> None:
    global _DISTRIBUTED_INITIALIZED  # noqa: PLW0603
    if not config.initialize:
        return
    jax.distributed.initialize(
        coordinator_address=config.coordinator_address,
        coordinator_bind_address=config.coordinator_bind_address,
        num_processes=config.num_processes,
        process_id=config.process_id,
        local_device_ids=None if config.local_device_ids is None else list(config.local_device_ids),
        cluster_detection_method=config.cluster_detection_method,
        initialization_timeout=config.initialization_timeout,
    )
    _DISTRIBUTED_INITIALIZED = True


def _configure_jax_runtime(config: pretrain_config.RuntimeConfig) -> None:
    """Apply YAML defaults while allowing standard JAX environment overrides."""
    cache = config.compilation_cache
    settings = {
        "jax_enable_compilation_cache": cache.enabled,
        "jax_compilation_cache_dir": str(epath.Path(cache.directory).expanduser()),
        "jax_persistent_cache_min_compile_time_secs": cache.minimum_compile_time_seconds,
        "jax_explain_cache_misses": cache.explain_misses,
    }
    for name, value in settings.items():
        if name.upper() not in os.environ:
            jax.config.update(name, value)


def _shard_batch_start(shard) -> int:
    index = shard.index[1] if np.ndim(shard.data) >= 5 and len(shard.index) > 1 else shard.index[0]
    if isinstance(index, slice):
        return 0 if index.start is None else index.start
    return int(index)


def _addressable_shard_prefix(shards, *, limit: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    remaining = limit
    for shard in sorted(shards, key=_shard_batch_start):
        chunk = np.asarray(jax.device_get(shard.data))
        if chunk.ndim >= 5:
            chunk = chunk.reshape(-1, *chunk.shape[2:])
        chunk = chunk[:remaining]
        if len(chunk):
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            break
    if not chunks:
        raise RuntimeError("The primary process has no addressable examples in the first global batch")
    return np.concatenate(chunks, axis=0)


def _process_local_prefix(array: jax.Array | np.ndarray, *, limit: int) -> np.ndarray:
    """Copy at most ``limit`` process-local batch elements to the host."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not isinstance(array, jax.Array) or array.is_fully_addressable:
        value = np.asarray(jax.device_get(array))
        if value.ndim >= 5:
            value = value.reshape(-1, *value.shape[2:])
        return value[:limit]
    return _addressable_shard_prefix(array.addressable_shards, limit=limit)


def _log_first_batch(observer: observability.RunObserver, batch: rlds_mixture.PretrainBatch, *, step: int) -> None:
    if jax.process_index() != 0:
        return
    images = jax.tree.map(lambda image: _process_local_prefix(image, limit=5), batch.observation.images)
    batch_size = len(next(iter(images.values())))
    camera_views = [
        np.concatenate([np.asarray(image[index]) for image in images.values()], axis=1)
        for index in range(min(5, batch_size))
    ]
    observer.log_images("camera_views", camera_views, step=step)


_ACTIVE_OBSERVERS: list[observability.RunObserver] = []


def _mean_metrics(infos: list[dict]) -> dict[str, np.ndarray]:
    stacked = common_utils.stack_forest(infos)
    return jax.device_get(jax.tree.map(jnp.mean, stacked))


def _reduce_train_metrics(config: pretrain_config.PretrainConfig, infos: list[dict]) -> dict[str, np.ndarray]:
    stacked = common_utils.stack_forest(infos)
    metrics = {
        key: jnp.mean(value) for key, value in stacked.items() if not key.endswith(("/loss_sum", "/valid_count"))
    }
    for source in config.data.sources:
        numerator = jnp.sum(stacked[f"source/{source.id}/loss_sum"])
        denominator = jnp.sum(stacked[f"source/{source.id}/valid_count"])
        metrics[f"source/{source.id}/loss"] = numerator / jnp.clip(denominator, 1)
    return jax.device_get(metrics)


def _run_validation(
    config: pretrain_config.PretrainConfig,
    state: training_utils.TrainState,
    validation_loaders: dict[str, rlds_mixture.RldsMixtureDataLoader],
    validation_step,
    rng: jax.Array,
    mesh: jax.sharding.Mesh,
) -> dict[str, float]:
    source_losses: dict[str, float] = {}
    for source_index, source in enumerate(config.data.sources):
        infos = []
        iterator = iter(validation_loaders[source.id])
        for batch_index in range(config.validation.batches_per_source):
            batch = _next_synchronized_validation_batch(
                iterator,
                source_id=source.id,
                split=source.validation_split,
                batch_index=batch_index,
            )
            validation_rng = jax.random.fold_in(jax.random.fold_in(rng, source_index), batch_index)
            with sharding.set_mesh(mesh):
                infos.append(validation_step(validation_rng, state, batch))
        source_losses[source.id] = float(_mean_metrics(infos)["loss"])

    probabilities = config.data.effective_probabilities()
    result = {f"validation/source/{source_id}/loss": loss for source_id, loss in source_losses.items()}
    result["validation/macro_loss"] = float(np.mean(list(source_losses.values())))
    result["validation/mixture_loss"] = float(
        sum(
            probability * source_losses[source.id]
            for source, probability in zip(config.data.sources, probabilities, strict=True)
        )
    )
    return result


def _next_synchronized_validation_batch(iterator, *, source_id: str, split: str, batch_index: int):
    """Keep ranks from entering a validation collective when another rank exhausted its local shard."""
    try:
        batch = next(iterator)
        local_available = 1
    except StopIteration:
        batch = None
        local_available = 0

    availability = np.asarray([local_available], dtype=np.int32)
    if jax.process_count() > 1:
        availability = np.asarray(multihost_utils.process_allgather(availability)).reshape(-1)
    available_ranks = int(availability.sum())
    if available_ranks != jax.process_count():
        raise RuntimeError(
            f"Validation split {split!r} for source {source_id!r} cannot provide batch {batch_index + 1} "
            f"on every process ({available_ranks}/{jax.process_count()} ranks have data)"
        )
    assert batch is not None
    return batch


def main(resolved: pretrain_config_loader.ResolvedPretrainConfig) -> None:
    _init_logging()
    config = resolved.config
    _configure_jax_runtime(config.runtime)
    _initialize_distributed(config.distributed)
    logging.info(
        "Running on %s with process %d/%d and %d global devices",
        platform.node(),
        jax.process_index(),
        jax.process_count(),
        jax.device_count(),
    )
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Global batch size {config.batch_size} must be divisible by global device count {jax.device_count()}"
        )
    if config.micro_batch_size is None:
        if config.gradient_accumulation_steps != 1:
            raise ValueError("micro_batch_size may be null only when gradient_accumulation_steps is 1")
    else:
        expected_batch_size = config.micro_batch_size * jax.device_count() * config.gradient_accumulation_steps
        if config.batch_size != expected_batch_size:
            raise ValueError(
                "training batch invariant failed: batch_size must equal micro_batch_size * global_device_count * "
                f"gradient_accumulation_steps ({config.batch_size} != {config.micro_batch_size} * "
                f"{jax.device_count()} * {config.gradient_accumulation_steps} = {expected_batch_size})"
            )

    logging.info(
        "JAX compilation cache: enabled=%s dir=%s min_compile_seconds=%s explain_misses=%s",
        jax.config.jax_enable_compilation_cache,
        jax.config.jax_compilation_cache_dir,
        jax.config.jax_persistent_cache_min_compile_time_secs,
        jax.config.jax_explain_cache_misses,
    )
    rng = jax.random.key(config.seed)
    train_rng, init_rng, validation_rng = jax.random.split(rng, 3)
    mesh = sharding.make_mesh(config.fsdp_devices)
    if config.distributed.diagnostics.topology_check:
        gpu_collectives.log_topology_diagnostics()
    gpu_collectives.warmup_collectives(enabled=config.distributed.warmup_collectives, mesh=mesh)
    collective_benchmarks = ()
    if config.distributed.warmup_collectives and jax.device_count() > 1:
        diagnostics = config.distributed.diagnostics
        collective_benchmarks = gpu_collectives.benchmark_global_collectives(
            mesh,
            tensor_sizes_mib=diagnostics.tensor_sizes_mib,
            warmup_iterations=diagnostics.warmup_iterations,
            measure_iterations=diagnostics.measure_iterations,
        )
        if diagnostics.collective_baseline_path is not None:
            gpu_collectives.validate_collective_baseline(
                diagnostics.collective_baseline_path,
                collective_benchmarks,
                minimum_fraction=diagnostics.minimum_baseline_fraction,
                policy=diagnostics.bandwidth_regression_policy,
            )
    # The leading axis is the replicated accumulation dimension; DATA_AXIS shards each microbatch.
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    snapshot = resolved.snapshot()
    lineage = rlds_mixture.build_lineage(config, snapshot)
    run_id_path = config.checkpoint_dir / "wandb_id.txt"
    if jax.process_index() == 0:
        observability.ensure_run_id(run_id_path)
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("observability_run_id_initialized")
    observer = observability.RunObserver(
        observability.options_from_pretrain_config(config, job_type="training"),
        manifest=snapshot,
        lineage=lineage,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        run_id_path=run_id_path,
        resume=resuming,
    )
    _ACTIVE_OBSERVERS.append(observer)
    observer.log_artifact_metadata("training-inputs", lineage, artifact_type="training-lineage")
    observer.log_code(epath.Path(__file__).parent.parent)
    for result in collective_benchmarks:
        prefix = f"communication/{result.operation}/{result.payload_mib:g}mib"
        observer.log_metrics(
            {
                f"{prefix}/median_seconds": result.median_seconds,
                f"{prefix}/p95_seconds": result.p95_seconds,
                f"{prefix}/algorithm_gib_per_second": result.algorithm_gib_per_second,
                f"{prefix}/bus_gib_per_second": result.bus_gib_per_second,
                f"{prefix}/rank_straggler_ratio": result.rank_straggler_ratio,
            },
            step=0,
        )

    signal_count = 0

    def handle_signal(signum, _frame) -> None:
        nonlocal signal_count
        signal_count += 1
        name = signal.Signals(signum).name
        observer.alert("termination_signal", f"Received {name}; checkpoint requested before exit")
        observer.request_stop(name)
        if signal_count > 1:
            raise KeyboardInterrupt(f"Received a second {name}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    observer.set_phase("initializing")
    state, state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(checkpoint_manager, state, None)
    jax.block_until_ready(state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(state.params))

    start_step = int(jax.device_get(state.step))
    previous_data_state = (
        checkpoints.load_extra_metadata(config.checkpoint_dir, checkpoint_manager.latest_step()) if resuming else None
    )
    initial_counts = None
    if resuming and previous_data_state is None:
        logging.warning("Checkpoint has no data_state.json; resuming the RLDS mixture with zero consumption counters")
    elif previous_data_state is not None:
        if previous_data_state.get("seed") != config.seed:
            raise ValueError("Cannot statistically resume with a different training seed")
        previous_topology = previous_data_state.get("topology")
        current_topology = {
            "process_count": jax.process_count(),
            "global_device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
        }
        if config.data_resume_mode == "exact" and previous_topology != current_topology:
            message = f"Exact data resume topology mismatch: checkpoint={previous_topology}, current={current_topology}"
            if not config.cluster.allow_topology_change or config.on_topology_change == "error":
                raise ValueError(message)
            logging.warning("%s; explicitly falling back to statistical data resume", message)
            config = dataclasses.replace(config, data_resume_mode="statistical")
        initial_counts = previous_data_state.get("consumed_examples_per_source")
        if not isinstance(initial_counts, dict) or set(initial_counts) != set(config.source_indices):
            raise ValueError("Checkpoint data source IDs do not match the current pretraining config")
    iterator_state_dir = None
    if resuming and config.data_resume_mode == "exact":
        latest_step = checkpoint_manager.latest_step()
        candidate = config.checkpoint_dir / str(latest_step) / "data_iterator" / f"rank-{jax.process_index():05d}"
        local_available = np.asarray([int(candidate.exists())], dtype=np.int32)
        availability = (
            np.asarray(multihost_utils.process_allgather(local_available)).reshape(-1)
            if jax.process_count() > 1
            else local_available
        )
        if not np.all(availability):
            message = (
                f"Exact resume requires one iterator sidecar per process at step {latest_step}; "
                f"availability={availability.tolist()}"
            )
            if config.on_missing_iterator_state == "error":
                raise FileNotFoundError(message)
            logging.warning("%s; explicitly falling back to statistical data resume", message)
            config = dataclasses.replace(config, data_resume_mode="statistical")
        else:
            iterator_state_dir = candidate
    train_loader = rlds_mixture.create_train_loader(
        config,
        sharding=data_sharding,
        start_step=start_step,
        initial_counts=initial_counts,
        iterator_state_dir=iterator_state_dir,
    )
    train_iterator = iter(train_loader)
    validation_loaders = rlds_mixture.create_validation_loaders(config, sharding=data_sharding)
    batch = next(train_iterator)
    logging.info("Initialized RLDS mixture:\n%s", training_utils.array_tree_to_info(batch))
    _log_first_batch(observer, batch, step=start_step)

    compiled_train_step = jax.jit(
        functools.partial(trainer.pretrain_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    compiled_validation_step = jax.jit(
        functools.partial(trainer.validation_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    observer.set_phase("compiling", step=start_step)
    compile_started = time.monotonic()
    # ``lower`` reconstructs dataclass pytrees with internal JAX ArgInfo leaves. Those are not runtime values and are
    # intentionally outside TrainState's public annotations, so disable the repository's runtime type checker only for
    # this tracing boundary.
    with at.disable_typechecking(), sharding.set_mesh(mesh):
        compiled_train_step = compiled_train_step.lower(train_rng, state, batch).compile()
    train_compile_seconds = time.monotonic() - compile_started
    observer.log_metrics({"performance/train_compile_seconds": train_compile_seconds}, step=start_step)
    observer.event("train_step_compiled", step=start_step, seconds=train_compile_seconds)
    logging.info("Compiled train step in %.3f seconds", train_compile_seconds)

    progress = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=jax.process_index() != 0,
    )
    infos = []
    log_window_start = time.monotonic()
    data_wait_seconds = 0.0
    observer.mark_progress(start_step)
    profiler_active = False
    profiler_stopped = False
    latest_rank_straggler_ratio = 1.0
    for _ in progress:
        observer.set_phase("training")
        current_step = int(jax.device_get(state.step))
        diagnostics = config.distributed.diagnostics
        if diagnostics.profile_start_step == current_step and not profiler_active and not profiler_stopped:
            trace_dir = config.observability_root / "jax-traces" / f"rank-{jax.process_index():05d}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(trace_dir))
            profiler_active = True
        step_started = time.monotonic()
        with sharding.set_mesh(mesh):
            state, info = compiled_train_step(train_rng, state, batch)
        completed_step = int(jax.device_get(state.step))
        step_seconds = time.monotonic() - step_started
        if jax.process_count() > 1 and (
            completed_step % config.log_interval == 0 or completed_step == config.num_train_steps
        ):
            rank_times = np.asarray(multihost_utils.process_allgather(np.asarray([step_seconds]))).reshape(-1)
            latest_rank_straggler_ratio = float(np.max(rank_times) / max(np.median(rank_times), 1e-12))
            if latest_rank_straggler_ratio > diagnostics.straggler_ratio_threshold:
                observer.alert(
                    "rank_straggler",
                    f"Step {completed_step} rank time ratio {latest_rank_straggler_ratio:.2f} exceeds "
                    f"{diagnostics.straggler_ratio_threshold:.2f}; times={rank_times.tolist()}",
                )
        if (
            profiler_active
            and diagnostics.profile_start_step is not None
            and completed_step >= diagnostics.profile_start_step + diagnostics.profile_num_steps
        ):
            jax.profiler.stop_trace()
            profiler_active = False
            profiler_stopped = True
        observer.set_phase("training", step=completed_step)
        observer.mark_progress(completed_step)
        infos.append(info)

        safety_metrics = jax.device_get({key: info[key] for key in ("loss", "grad_norm")})
        if not all(np.isfinite(float(value)) for value in safety_metrics.values()):
            observer.alert(
                "non_finite_training_metric",
                f"Non-finite metric at step {completed_step}: "
                + ", ".join(f"{key}={float(value)}" for key, value in safety_metrics.items()),
            )
            observer.request_stop("non_finite_training_metric")

        if completed_step % config.log_interval == 0 or completed_step == config.num_train_steps:
            metrics = {f"train/{key}": value for key, value in _reduce_train_metrics(config, infos).items()}
            elapsed = max(time.monotonic() - log_window_start, 1e-6)
            metrics.update(
                {
                    "train/learning_rate": float(jax.device_get(config.lr_schedule.create()(completed_step))),
                    "performance/step_seconds": step_seconds,
                    "performance/data_wait_seconds": data_wait_seconds,
                    "performance/samples_per_second": config.batch_size * len(infos) / elapsed,
                    "performance/rank_straggler_ratio": latest_rank_straggler_ratio,
                }
            )
            metrics.update({f"input/{key}": value for key, value in train_loader.metrics(reset_interval=True).items()})
            valid_fraction = float(metrics["train/valid_action_fraction"])
            metrics["performance/valid_actions_per_second"] = (
                valid_fraction
                * config.batch_size
                * config.model.action_horizon
                * config.model.action_dim
                * len(infos)
                / elapsed
            )
            data_state = train_loader.data_state()
            for source_id, count in data_state["consumed_examples_per_source"].items():
                metrics[f"data/source/{source_id}/consumed_examples"] = count
            if jax.process_index() == 0:
                progress.write(
                    f"Step {completed_step}: "
                    + ", ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
                )
            observer.log_metrics(metrics, step=completed_step)
            infos = []
            log_window_start = time.monotonic()
            data_wait_seconds = 0.0

        if completed_step % config.validation.interval_steps == 0 or completed_step == config.num_train_steps:
            observer.set_phase("validation", step=completed_step)
            validation_started = time.monotonic()
            metrics = _run_validation(
                config,
                state,
                validation_loaders,
                compiled_validation_step,
                jax.random.fold_in(validation_rng, completed_step),
                mesh,
            )
            metrics["performance/validation_seconds"] = time.monotonic() - validation_started
            if jax.process_index() == 0:
                progress.write(
                    f"Validation {completed_step}: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                )
            observer.log_metrics(metrics, step=completed_step)
            observer.set_phase("training", step=completed_step)

        should_save = (
            completed_step % config.save_interval == 0
            or completed_step == config.num_train_steps
            or observer.stop_requested
        )
        if should_save:
            observer.set_phase("checkpoint", step=completed_step)
            checkpoint_started = time.monotonic()
            try:
                iterator_snapshot_dir = None
                if config.data_resume_mode == "exact":
                    iterator_snapshot_dir = (
                        config.checkpoint_dir
                        / ".iterator_staging"
                        / str(completed_step)
                        / f"rank-{jax.process_index():05d}"
                    )
                    if iterator_snapshot_dir.exists():
                        shutil.rmtree(iterator_snapshot_dir)
                    snapshot_started = time.monotonic()
                    train_loader.snapshot_iterator(iterator_snapshot_dir)
                    snapshot_seconds = time.monotonic() - snapshot_started
                    if snapshot_seconds > config.iterator_snapshot_timeout_seconds:
                        raise TimeoutError(
                            f"Iterator snapshot took {snapshot_seconds:.1f}s, exceeding "
                            f"checkpoint.iterator_snapshot_timeout_seconds={config.iterator_snapshot_timeout_seconds}"
                        )
                    observer.log_metrics(
                        {"checkpoint/iterator_snapshot_seconds": snapshot_seconds}, step=completed_step
                    )
                    if jax.process_count() > 1:
                        multihost_utils.sync_global_devices(f"iterator_snapshot_{completed_step}_ready")
                checkpoints.save_state(
                    checkpoint_manager,
                    state,
                    None,
                    completed_step,
                    config_snapshot=snapshot,
                    extra_assets=train_loader.all_norm_stats(),
                    extra_metadata=train_loader.data_state(),
                    iterator_snapshot_dir=iterator_snapshot_dir,
                )
            except BaseException as exc:
                observer.alert("checkpoint_failed", f"Checkpoint enqueue failed: {type(exc).__name__}: {exc}")
                raise
            observer.log_metrics(
                {
                    "checkpoint/enqueue_seconds": time.monotonic() - checkpoint_started,
                    "checkpoint/latest_step": completed_step,
                },
                step=completed_step,
            )
            observer.event(
                "checkpoint_enqueued", step=completed_step, uri=str(config.checkpoint_dir / str(completed_step))
            )
            observer.set_phase("training", step=completed_step)
        if observer.stop_requested:
            break
        if completed_step < config.num_train_steps:
            data_started = time.monotonic()
            batch = next(train_iterator)
            data_wait_seconds += time.monotonic() - data_started

    if profiler_active:
        jax.profiler.stop_trace()
    logging.info("Waiting for checkpoint manager to finish")
    observer.set_phase("checkpoint", step=int(jax.device_get(state.step)))
    finalize_started = time.monotonic()
    try:
        checkpoint_manager.wait_until_finished()
    except BaseException as exc:
        observer.alert("checkpoint_failed", f"Checkpoint finalization failed: {type(exc).__name__}: {exc}")
        raise
    final_step = int(jax.device_get(state.step))
    if final_step >= config.num_train_steps:
        final_counts = train_loader.data_state()["consumed_examples_per_source"]
        shortfalls = {
            source_id: limit.min_samples - final_counts[source_id]
            for source_id, limit in config.data.mixing.source_limits.items()
            if limit.min_samples is not None and final_counts[source_id] < limit.min_samples
        }
        if shortfalls:
            raise RuntimeError(f"Training completed without satisfying source min_samples: {shortfalls}")
    observer.log_metrics(
        {
            "checkpoint/finalize_seconds": time.monotonic() - finalize_started,
            "checkpoint/latest_step": final_step,
        },
        step=final_step,
    )
    checkpoint_pointer = {
        "uri": str(config.checkpoint_dir / str(final_step)),
        "checkpoint_root": str(config.checkpoint_dir),
        "step": final_step,
        "lineage_id": observer.lineage_id,
    }
    observer.log_artifact_metadata("checkpoint-index", checkpoint_pointer, artifact_type="checkpoint-reference")
    train_loader.close()
    for loader in validation_loaders.values():
        loader.close()
    staging_root = config.checkpoint_dir / ".iterator_staging"
    if jax.process_index() == 0 and staging_root.exists():
        shutil.rmtree(staging_root)
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("iterator_staging_cleaned")
    observer.finish(status="stopped" if observer.stop_requested else "completed")
    _ACTIVE_OBSERVERS.clear()


def _fatal_exit_code(exc: BaseException) -> int:
    return 130 if isinstance(exc, KeyboardInterrupt) else 1


def _requires_fatal_cleanup(exc: BaseException) -> bool:
    return not (isinstance(exc, SystemExit) and exc.code in (None, 0))


def _start_fatal_watchdog(timeout_seconds: float, exit_code: int) -> threading.Timer:
    def force_exit() -> None:
        with contextlib.suppress(OSError):
            os.write(2, f"Fatal cleanup exceeded {timeout_seconds:.1f}s; forcing rank exit\n".encode())
        os._exit(exit_code)

    watchdog = threading.Timer(timeout_seconds, force_exit)
    watchdog.name = "openpi-fatal-exit"
    watchdog.daemon = True
    watchdog.start()
    return watchdog


def _cleanup_after_uncaught_exception(exc: BaseException, *, timeout_seconds: float) -> None:
    """Best-effort bounded cleanup; the watchdog survives an interpreter shutdown hang."""
    exit_code = _fatal_exit_code(exc)
    _start_fatal_watchdog(timeout_seconds, exit_code)
    traceback.print_exception(exc, file=sys.stderr)
    if _ACTIVE_OBSERVERS:
        observer = _ACTIVE_OBSERVERS[-1]
        with contextlib.suppress(BaseException):
            observer.alert("uncaught_exception", f"{type(exc).__name__}: {exc}", deduplicate_seconds=0)
        with contextlib.suppress(BaseException):
            observer.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        _ACTIVE_OBSERVERS.clear()
    with contextlib.suppress(BaseException):
        if _DISTRIBUTED_INITIALIZED:
            jax.distributed.shutdown()
    with contextlib.suppress(BaseException):
        logging.shutdown()
    with contextlib.suppress(BaseException):
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    resolved = None
    try:
        resolved = pretrain_config_loader.parse_cli()
        main(resolved)
    except BaseException as exc:
        if not _requires_fatal_cleanup(exc):
            raise
        timeout_seconds = 15.0 if resolved is None else resolved.config.runtime.fatal_cleanup_timeout_seconds
        _cleanup_after_uncaught_exception(exc, timeout_seconds=timeout_seconds)
        raise SystemExit(_fatal_exit_code(exc)) from exc
