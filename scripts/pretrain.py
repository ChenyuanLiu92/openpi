"""Pretrain pi0.5 from a heterogeneous sample-level RLDS mixture.

Run with the optional RLDS dependencies installed:

    uv run --group rlds scripts/pretrain.py configs/pretraining/pi05/template.yaml
"""

from __future__ import annotations

import functools
import logging
import platform

from etils import epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

from openpi.training import checkpoints
from openpi.training import pretrain_config
from openpi.training import pretrain_config_loader
from openpi.training import rlds_mixture
from openpi.training import sharding
from openpi.training import trainer
from openpi.training import utils as training_utils


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
    if not config.initialize:
        return
    jax.distributed.initialize(
        coordinator_address=config.coordinator_address,
        num_processes=config.num_processes,
        process_id=config.process_id,
        local_device_ids=None if config.local_device_ids is None else list(config.local_device_ids),
        cluster_detection_method=config.cluster_detection_method,
        initialization_timeout=config.initialization_timeout,
    )


def _init_wandb(
    config: pretrain_config.PretrainConfig,
    snapshot: dict,
    *,
    resuming: bool,
) -> None:
    enabled = config.wandb_enabled and jax.process_index() == 0
    if not enabled:
        wandb.init(mode="disabled")
        return
    if resuming:
        run_id = (config.checkpoint_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=snapshot,
            project=config.project_name,
        )
        (config.checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)
    wandb.run.log_code(epath.Path(__file__).parent.parent)


def _log_first_batch(batch: rlds_mixture.PretrainBatch, *, step: int) -> None:
    if jax.process_index() != 0:
        return
    images = jax.device_get(batch.observation.images)
    batch_size = len(next(iter(images.values())))
    camera_views = [
        wandb.Image(np.concatenate([np.asarray(image[index]) for image in images.values()], axis=1))
        for index in range(min(5, batch_size))
    ]
    wandb.log({"camera_views": camera_views}, step=step)


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
        for batch_index, batch in enumerate(validation_loaders[source.id]):
            validation_rng = jax.random.fold_in(jax.random.fold_in(rng, source_index), batch_index)
            with sharding.set_mesh(mesh):
                infos.append(validation_step(validation_rng, state, batch))
        if not infos:
            raise RuntimeError(
                f"Validation split {source.validation_split!r} for source {source.id!r} "
                f"does not contain one complete global batch of {config.batch_size} examples"
            )
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


def main(resolved: pretrain_config_loader.ResolvedPretrainConfig) -> None:
    _init_logging()
    config = resolved.config
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

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, init_rng, validation_rng = jax.random.split(rng, 3)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    snapshot = resolved.snapshot()
    _init_wandb(config, snapshot, resuming=resuming)

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
        initial_counts = previous_data_state.get("consumed_examples_per_source")
        if not isinstance(initial_counts, dict) or set(initial_counts) != set(config.source_indices):
            raise ValueError("Checkpoint data source IDs do not match the current pretraining config")
    train_loader = rlds_mixture.create_train_loader(
        config,
        sharding=data_sharding,
        start_step=start_step,
        initial_counts=initial_counts,
    )
    train_iterator = iter(train_loader)
    validation_loaders = rlds_mixture.create_validation_loaders(config, sharding=data_sharding)
    batch = next(train_iterator)
    logging.info("Initialized RLDS mixture:\n%s", training_utils.array_tree_to_info(batch))
    _log_first_batch(batch, step=start_step)

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

    progress = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=jax.process_index() != 0,
    )
    infos = []
    for _ in progress:
        with sharding.set_mesh(mesh):
            state, info = compiled_train_step(train_rng, state, batch)
        completed_step = int(jax.device_get(state.step))
        infos.append(info)

        if completed_step % config.log_interval == 0 or completed_step == config.num_train_steps:
            metrics = {f"train/{key}": value for key, value in _reduce_train_metrics(config, infos).items()}
            if jax.process_index() == 0:
                progress.write(
                    f"Step {completed_step}: "
                    + ", ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
                )
                wandb.log(metrics, step=completed_step)
            infos = []

        if completed_step % config.validation.interval_steps == 0 or completed_step == config.num_train_steps:
            metrics = _run_validation(
                config,
                state,
                validation_loaders,
                compiled_validation_step,
                jax.random.fold_in(validation_rng, completed_step),
                mesh,
            )
            if jax.process_index() == 0:
                wandb.log(metrics, step=completed_step)
                progress.write(
                    f"Validation {completed_step}: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                )

        should_save = completed_step % config.save_interval == 0 or completed_step == config.num_train_steps
        if should_save:
            checkpoints.save_state(
                checkpoint_manager,
                state,
                None,
                completed_step,
                config_snapshot=snapshot,
                extra_assets=train_loader.all_norm_stats(),
                extra_metadata=train_loader.data_state(),
            )
        if completed_step < config.num_train_steps:
            batch = next(train_iterator)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(pretrain_config_loader.parse_cli())
