"""High-throughput, source-aware RLDS mixture loading for pi0.5 pre-training."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import queue
import threading
import time
import traceback
from typing import Any

from flax import struct
import jax
from jax.experimental import multihost_utils
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import normalize
from openpi.training import observability
from openpi.training import pretrain_config
from openpi.training import rlds_adapters

_STATS_MANIFEST = "stats_manifest.json"


class _MixtureRuntime:
    """Thread-safe source health shared by the host converter and tf.data sampler."""

    def __init__(self, config: pretrain_config.PretrainConfig, initial_counts: Mapping[str, int] | None):
        self._config = config
        self._lock = threading.Lock()
        self._active = np.ones(len(config.data.sources), dtype=bool)
        self._failures = np.zeros(len(config.data.sources), dtype=np.int64)
        self._active_variable: Any | None = None
        for index, source in enumerate(config.data.sources):
            limit = config.data.mixing.source_limits.get(source.id)
            if limit is not None and limit.max_samples is not None:
                self._active[index] = (
                    _local_remaining_quota(
                        limit.max_samples,
                        int((initial_counts or {}).get(source.id, 0)),
                    )
                    > 0
                )
        if not self._active.any():
            raise RuntimeError("All RLDS sources have already reached their configured max_samples")

    def attach_tensorflow(self, tf: Any) -> Any:
        with self._lock:
            self._active_variable = tf.Variable(self._active, trainable=False, dtype=tf.bool)
            return self._active_variable

    def record_success(self, source_ids: np.ndarray) -> None:
        with self._lock:
            self._failures[np.unique(source_ids)] = 0

    def record_failure(self, source_ids: np.ndarray) -> tuple[str, ...]:
        if self._config.data.mixing.source_failure_policy != "degrade":
            return ()
        degraded = []
        with self._lock:
            for index in np.unique(source_ids):
                self._failures[index] += 1
                if self._failures[index] >= self._config.data.mixing.consecutive_failure_threshold:
                    self._active[index] = False
                    degraded.append(self._config.data.sources[index].id)
            if not self._active.any():
                raise RuntimeError("All RLDS sources have been degraded after consecutive failures")
            if self._active_variable is not None:
                self._active_variable.assign(self._active)
        return tuple(degraded)

    def record_counts(self, session_counts: Mapping[str, int], initial_counts: Mapping[str, int]) -> None:
        changed = False
        with self._lock:
            for index, source in enumerate(self._config.data.sources):
                limit = self._config.data.mixing.source_limits.get(source.id)
                if (
                    limit is not None
                    and limit.max_samples is not None
                    and session_counts[source.id]
                    >= _local_remaining_quota(limit.max_samples, initial_counts[source.id])
                ):
                    changed |= bool(self._active[index])
                    self._active[index] = False
            if changed and self._active_variable is not None:
                self._active_variable.assign(self._active)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._active.copy(), self._failures.copy()


def _local_remaining_quota(global_limit: int, globally_consumed: int) -> int:
    """Assign a deterministic disjoint share of a remaining global sample limit to this rank."""
    remaining = max(global_limit - globally_consumed, 0)
    quotient, remainder = divmod(remaining, jax.process_count())
    return quotient + int(jax.process_index() < remainder)


@struct.dataclass
class PretrainBatch:
    """A model batch plus masks and source metadata used only by the trainer."""

    observation: _model.Observation
    actions: Any
    action_mask: Any
    source_id: Any


def source_fingerprint(source: pretrain_config.RldsSourceConfig) -> str:
    payload = dataclasses.asdict(source)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def build_lineage(config: pretrain_config.PretrainConfig, snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build stable data lineage shared by normalization and training jobs."""
    datasets = {}
    source_lineage_ids = []
    for source in config.data.sources:
        manifest_path = pathlib.Path(source.data_dir) / source.tfds_name / source.version / "conversion_manifest.json"
        conversion_lineage_id = None
        if manifest_path.is_file():
            try:
                conversion_lineage_id = json.loads(manifest_path.read_text()).get("lineage_id")
            except (json.JSONDecodeError, OSError):
                conversion_lineage_id = None
        if conversion_lineage_id:
            source_lineage_ids.append(str(conversion_lineage_id))
        datasets[source.id] = {
            "tfds_name": source.tfds_name,
            "version": source.version,
            "uri": str(manifest_path.parent),
            "source_config_sha256": source_fingerprint(source),
            "manifest_sha256": observability.file_digest(manifest_path) if manifest_path.is_file() else None,
            "conversion_lineage_id": conversion_lineage_id,
        }
    identity = {"schema_version": 1, "datasets": datasets}
    normalizations = {}
    for source in config.data.sources:
        path = config.assets_dirs / source.normalization_id / _STATS_MANIFEST
        normalizations[source.normalization_id] = {
            "uri": str(path.parent),
            "manifest_sha256": observability.file_digest(path) if path.is_file() else None,
        }
    lineage_id = (
        source_lineage_ids[0]
        if len(datasets) == 1 and len(source_lineage_ids) == 1
        else observability.stable_digest(identity)[:16]
    )
    return {
        **identity,
        "lineage_id": lineage_id,
        "normalizations": normalizations,
        "config_sha256": observability.stable_digest(snapshot) if snapshot is not None else None,
    }


def expected_stats_manifest(
    config: pretrain_config.PretrainConfig, normalization_id: str, *, sample_count: int | None = None
) -> dict[str, Any]:
    sources = [source for source in config.data.sources if source.normalization_id == normalization_id]
    logits = [math.log(source.weight) / config.data.temperature for source in sources]
    maximum = max(logits)
    scaled = [math.exp(logit - maximum) for logit in logits]
    total = sum(scaled)
    return {
        "schema_version": 1,
        "normalization_id": normalization_id,
        "model_action_dim": config.model.action_dim,
        "sources": {source.id: source_fingerprint(source) for source in sources},
        "sampling": {
            "temperature": config.data.temperature,
            "conditional_probabilities": {
                source.id: probability / total for source, probability in zip(sources, scaled, strict=True)
            },
        },
        "sample_count": sample_count,
    }


def save_stats(
    config: pretrain_config.PretrainConfig,
    normalization_id: str,
    stats: dict[str, normalize.NormStats],
    *,
    sample_count: int,
) -> pathlib.Path:
    output_dir = config.assets_dirs / normalization_id
    normalize.save(output_dir, stats)
    manifest = expected_stats_manifest(config, normalization_id, sample_count=sample_count)
    (output_dir / _STATS_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output_dir


def load_stats(
    config: pretrain_config.PretrainConfig, source: pretrain_config.RldsSourceConfig
) -> dict[str, normalize.NormStats]:
    stats_dir = config.assets_dirs / source.normalization_id
    stats = normalize.load(stats_dir)
    manifest_path = stats_dir / _STATS_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Normalization manifest not found: {manifest_path}")
    actual = json.loads(manifest_path.read_text())
    expected = expected_stats_manifest(config, source.normalization_id)
    for field in ("schema_version", "normalization_id", "model_action_dim", "sources", "sampling"):
        if actual.get(field) != expected[field]:
            raise ValueError(
                f"Normalization manifest mismatch for {source.normalization_id!r} at field {field!r}; "
                "re-run scripts/compute_pretrain_norm_stats.py"
            )
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"Normalization stats {stats_dir} are missing {key!r}")
    return stats


class RldsMixtureDataLoader:
    """Converts batched TensorFlow RLDS samples into sharded OpenPI batches."""

    def __init__(
        self,
        config: pretrain_config.PretrainConfig,
        dataset: Any,
        *,
        sharding: jax.sharding.Sharding | None,
        num_batches: int | None = None,
        initial_counts: Mapping[str, int] | None = None,
        initial_batches: int = 0,
        mixture_runtime: _MixtureRuntime | None = None,
    ):
        self._config = config
        self._dataset = dataset
        self._sharding = sharding
        self._num_batches = num_batches
        self._tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
        self._base_counts = {source.id: int((initial_counts or {}).get(source.id, 0)) for source in config.data.sources}
        self._session_counts = {source.id: 0 for source in config.data.sources}
        self._base_batches = initial_batches
        self._session_batches = 0
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "processed_batches": 0.0,
            "consumed_batches": 0.0,
            "bad_batches": 0.0,
            "tokenizer_seconds": 0.0,
            "host_queue_depth": 0.0,
            "device_queue_depth": 0.0,
            "host_queue_wait_seconds": 0.0,
            **{f"source/{source.id}/tokenizer_seconds": 0.0 for source in config.data.sources},
        }
        self._recent_bad = deque(maxlen=config.data.pipeline.bad_sample_window)
        self._last_seen_batch = {source.id: 0 for source in config.data.sources}
        self._recent_prompt_hashes: deque[str] = deque(maxlen=10000)
        self._repeated_examples = 0
        self._processed_examples = 0
        self._mixture_runtime = mixture_runtime

    def __iter__(self) -> Iterator[PretrainBatch]:
        batches = self._dataset.as_numpy_iterator()
        if self._num_batches is not None:
            batches = itertools.islice(batches, self._num_batches)
        host_batches = self._host_prefetch(batches)
        depth = self._config.data.pipeline.device_prefetch_batches
        if depth <= 0:
            for batch in host_batches:
                self._mark_consumed(batch)
                yield self._to_device(batch)
            return
        pending: deque[tuple[PretrainBatch, PretrainBatch]] = deque()
        for batch in host_batches:
            pending.append((batch, self._to_device(batch)))
            self._set_metric("device_queue_depth", len(pending))
            if len(pending) > depth:
                host_batch, ready = pending.popleft()
                self._mark_consumed(host_batch)
                yield ready
        while pending:
            self._set_metric("device_queue_depth", len(pending))
            host_batch, ready = pending.popleft()
            self._mark_consumed(host_batch)
            yield ready

    def _host_prefetch(self, batches: Iterator[Any]) -> Iterator[PretrainBatch]:
        depth = self._config.data.pipeline.host_prefetch_batches
        if depth <= 0:
            for batch in batches:
                converted = self._convert_or_handle(batch)
                if converted is not None:
                    yield converted
            return

        output: queue.Queue[Any] = queue.Queue(maxsize=depth)
        sentinel = object()

        def produce() -> None:
            try:
                for batch in batches:
                    converted = self._convert_or_handle(batch)
                    if converted is not None:
                        output.put(converted)
                        self._set_metric("host_queue_depth", output.qsize())
            except BaseException as exc:  # propagate worker exceptions on the training thread
                output.put(exc)
            finally:
                output.put(sentinel)

        worker = threading.Thread(target=produce, name="rlds-host-prefetch", daemon=True)
        worker.start()
        while True:
            started = time.monotonic()
            item = output.get()
            self._add_metric("host_queue_wait_seconds", time.monotonic() - started)
            self._set_metric("host_queue_depth", output.qsize())
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _convert_or_handle(self, batch: Mapping[str, Any]) -> PretrainBatch | None:
        try:
            converted = self._convert_batch(batch)
        except Exception as exc:
            self._record_bad_batch(batch, exc)
            source_ids = np.asarray(batch.get("source_id", []), dtype=np.int32)
            if self._mixture_runtime is not None and source_ids.size:
                degraded = self._mixture_runtime.record_failure(source_ids)
                for source_id in degraded:
                    self._set_metric(f"source/{source_id}/degraded", 1)
            if (
                self._config.data.pipeline.bad_sample_policy == "fail"
                and self._config.data.mixing.source_failure_policy == "fail"
            ):
                raise
            self._check_circuit_breaker()
            return None
        if self._mixture_runtime is not None:
            self._mixture_runtime.record_success(np.asarray(batch["source_id"], dtype=np.int32))
        self._recent_bad.append(False)
        self._add_metric("processed_batches", 1)
        return converted

    def _mark_consumed(self, converted: PretrainBatch) -> None:
        """Advance resume/cap counters only when a prefetched batch is handed to training."""
        self._add_metric("consumed_batches", 1)
        source_ids = np.asarray(converted.source_id, dtype=np.int32).reshape(-1)
        bincount = np.bincount(source_ids, minlength=len(self._config.data.sources))
        with self._metrics_lock:
            for index, source in enumerate(self._config.data.sources):
                self._session_counts[source.id] += int(bincount[index])
                if bincount[index]:
                    self._last_seen_batch[source.id] = self._base_batches + self._session_batches + 1
                limit = self._config.data.mixing.source_limits.get(source.id)
                if limit is not None and limit.max_samples is not None:
                    quota = _local_remaining_quota(limit.max_samples, self._base_counts[source.id])
                    if self._session_counts[source.id] > quota:
                        raise RuntimeError(
                            f"Source {source.id!r} exceeded this rank's remaining quota {quota} for global "
                            f"max_samples={limit.max_samples}: {self._session_counts[source.id]}"
                        )
            self._session_batches += 1
        if self._mixture_runtime is not None:
            self._mixture_runtime.record_counts(self._session_counts, self._base_counts)

    def _convert_batch(self, batch: Mapping[str, Any]) -> PretrainBatch:
        source_ids = np.asarray(batch["source_id"], dtype=np.int32)
        prompts = np.asarray(batch["prompt"])
        states = np.asarray(batch["state"])
        prompt_texts = [prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt) for prompt in prompts]
        for prompt_text in prompt_texts:
            digest = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
            if digest in self._recent_prompt_hashes:
                self._repeated_examples += 1
            self._recent_prompt_hashes.append(digest)
            self._processed_examples += 1
        token_states = [
            states[index, : self._config.data.sources[int(source_index)].state_dim]
            if self._config.model.discrete_state_input
            else None
            for index, source_index in enumerate(source_ids)
        ]
        started = time.monotonic()
        token_values, token_masks = self._tokenizer.tokenize_batch(
            prompt_texts,
            token_states,
            num_threads=self._config.data.pipeline.tokenizer_threads,
        )
        tokenizer_seconds = time.monotonic() - started
        self._add_metric("tokenizer_seconds", tokenizer_seconds)
        source_counts = np.bincount(source_ids, minlength=len(self._config.data.sources))
        for index, source in enumerate(self._config.data.sources):
            self._add_metric(
                f"source/{source.id}/tokenizer_seconds",
                tokenizer_seconds * float(source_counts[index]) / max(len(source_ids), 1),
            )
        observation = _model.Observation(
            images={key: np.asarray(value) for key, value in batch["image"].items()},
            image_masks={key: np.asarray(value) for key, value in batch["image_mask"].items()},
            state=states.astype(np.float32),
            tokenized_prompt=token_values,
            tokenized_prompt_mask=token_masks,
        )
        result = PretrainBatch(
            observation=observation,
            actions=np.asarray(batch["actions"], dtype=np.float32),
            action_mask=np.asarray(batch["action_mask"], dtype=bool),
            source_id=source_ids,
        )
        accumulation_steps = self._config.gradient_accumulation_steps
        local_batch_size = len(source_ids)
        if local_batch_size % accumulation_steps:
            raise ValueError(
                f"Local batch size {local_batch_size} is not divisible by gradient_accumulation_steps "
                f"{accumulation_steps}"
            )
        micro_batch_size = local_batch_size // accumulation_steps
        return jax.tree.map(
            lambda value: np.asarray(value).reshape(accumulation_steps, micro_batch_size, *value.shape[1:]),
            result,
        )

    def _to_device(self, batch: PretrainBatch) -> PretrainBatch:
        if self._sharding is None:
            return batch
        return jax.tree.map(lambda value: jax.make_array_from_process_local_data(self._sharding, value), batch)

    def _record_bad_batch(self, batch: Mapping[str, Any], exc: Exception) -> None:
        self._recent_bad.append(True)
        self._add_metric("bad_batches", 1)
        if self._config.data.pipeline.bad_sample_policy != "quarantine":
            return
        directory = pathlib.Path(self._config.data.pipeline.quarantine_dir or "")
        directory.mkdir(parents=True, exist_ok=True)
        prompts = np.asarray(batch.get("prompt", []))
        prompt_hash = hashlib.sha256(repr(prompts[:1]).encode()).hexdigest()[:16]
        record = {
            "time": time.time(),
            "process_index": jax.process_index(),
            "source_ids": np.asarray(batch.get("source_id", []), dtype=np.int32).tolist(),
            "prompt_hash": prompt_hash,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(exc))[-4000:],
        }
        path = directory / f"rank-{jax.process_index():05d}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _check_circuit_breaker(self) -> None:
        bad = sum(self._recent_bad)
        total = len(self._recent_bad)
        pipeline = self._config.data.pipeline
        fraction_exceeded = total == self._recent_bad.maxlen and bad / max(total, 1) > pipeline.max_bad_fraction
        if bad > pipeline.max_bad_samples or fraction_exceeded:
            raise RuntimeError(
                f"RLDS bad-sample circuit breaker opened on rank {jax.process_index()}: {bad}/{total} bad batches"
            )

    def _add_metric(self, name: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[name] += float(value)

    def _set_metric(self, name: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[name] = float(value)

    def metrics(self, *, reset_interval: bool = False) -> dict[str, float]:
        with self._metrics_lock:
            result = dict(self._metrics)
            result.update({f"source/{key}/examples": float(value) for key, value in self._session_counts.items()})
            total = sum(self._session_counts.values())
            probabilities = self._config.data.effective_probabilities(self._base_batches + self._session_batches)
            drift = 0.0
            for source, target in zip(self._config.data.sources, probabilities, strict=True):
                observed = self._session_counts[source.id] / max(total, 1)
                result[f"source/{source.id}/observed_probability"] = observed
                result[f"source/{source.id}/target_probability"] = target
                result[f"source/{source.id}/starvation_batches"] = float(
                    self._base_batches + self._session_batches - self._last_seen_batch[source.id]
                )
                limit = self._config.data.mixing.source_limits.get(source.id)
                minimum_quota = (
                    None
                    if limit is None or limit.min_samples is None
                    else _local_remaining_quota(limit.min_samples, self._base_counts[source.id])
                )
                maximum_quota = (
                    None
                    if limit is None or limit.max_samples is None
                    else _local_remaining_quota(limit.max_samples, self._base_counts[source.id])
                )
                result[f"source/{source.id}/minimum_sample_shortfall"] = float(
                    0 if minimum_quota is None else max(minimum_quota - self._session_counts[source.id], 0)
                )
                result[f"source/{source.id}/maximum_sample_remaining"] = float(
                    -1 if maximum_quota is None else max(maximum_quota - self._session_counts[source.id], 0)
                )
                if observed > 0:
                    drift += observed * math.log(observed / max(target, 1e-12))
            result["mixture_kl_divergence"] = drift
            result["repeat_fraction"] = self._repeated_examples / max(self._processed_examples, 1)
            if self._mixture_runtime is not None:
                active, failures = self._mixture_runtime.snapshot()
                for index, source in enumerate(self._config.data.sources):
                    result[f"source/{source.id}/active"] = float(active[index])
                    result[f"source/{source.id}/consecutive_failures"] = float(failures[index])
            if reset_interval:
                for key in self._metrics:
                    if key in {"tokenizer_seconds", "host_queue_wait_seconds"} or key.endswith("/tokenizer_seconds"):
                        self._metrics[key] = 0.0
        return result

    def data_state(self) -> dict[str, Any]:
        local_counts = np.asarray(
            [self._session_counts[source.id] for source in self._config.data.sources], dtype=np.int64
        )
        if jax.process_count() > 1:
            local_counts = np.asarray(multihost_utils.process_allgather(local_counts)).sum(axis=0)
        global_counts = {
            source.id: self._base_counts[source.id] + int(local_counts[index])
            for index, source in enumerate(self._config.data.sources)
        }
        return {
            "seed": self._config.seed,
            "consumed_examples_per_source": global_counts,
            "consumed_batches_per_rank": self._base_batches + self._session_batches,
            "resume_semantics": self._config.data_resume_mode,
            "topology": {
                "process_count": jax.process_count(),
                "global_device_count": jax.device_count(),
                "local_device_count": jax.local_device_count(),
            },
        }

    def all_norm_stats(self) -> dict[str, dict[str, normalize.NormStats]]:
        result = {}
        for source in self._config.data.sources:
            if source.normalization_id not in result:
                result[source.normalization_id] = load_stats(self._config, source)
        return result


def create_train_loader(
    config: pretrain_config.PretrainConfig,
    *,
    sharding: jax.sharding.Sharding | None,
    start_step: int = 0,
    initial_counts: Mapping[str, int] | None = None,
) -> RldsMixtureDataLoader:
    local_batch_size = _local_batch_size(config.batch_size)
    mixture_runtime = _MixtureRuntime(config, initial_counts)
    datasets = [
        _create_source_dataset(config, source, split=source.train_split, training=True, normalize_data=True)
        for source in config.data.sources
    ]
    probabilities = config.data.effective_probabilities()
    exact_resume = config.data_resume_mode == "exact"
    seed = config.seed + jax.process_index() if exact_resume else config.seed + start_step * 9973 + jax.process_index()
    dataset = _sample_and_batch(
        datasets,
        probabilities,
        batch_size=local_batch_size,
        seed=seed,
        prefetch_batches=config.data.prefetch_batches,
        training=True,
        config=config,
        start_step=0 if exact_resume else start_step,
        mixture_runtime=mixture_runtime,
    )
    if exact_resume and start_step:
        # In exact mode every source transform and the mixer are deterministic; replaying the batch ordinal restores
        # the same next batch without serializing large tf.data buffers into the model checkpoint.
        dataset = dataset.skip(start_step)
    return RldsMixtureDataLoader(
        config,
        dataset,
        sharding=sharding,
        initial_counts=initial_counts,
        initial_batches=start_step,
        mixture_runtime=mixture_runtime,
    )


def create_validation_loaders(
    config: pretrain_config.PretrainConfig,
    *,
    sharding: jax.sharding.Sharding | None,
) -> dict[str, RldsMixtureDataLoader]:
    local_batch_size = _local_batch_size(config.batch_size)
    result = {}
    for source in config.data.sources:
        source_dataset = _create_source_dataset(
            config, source, split=source.validation_split, training=False, normalize_data=True
        )
        dataset = _sample_and_batch(
            [source_dataset],
            [1.0],
            batch_size=local_batch_size,
            seed=config.seed,
            prefetch_batches=config.data.prefetch_batches,
            training=False,
        )
        result[source.id] = RldsMixtureDataLoader(
            config,
            dataset,
            sharding=sharding,
            num_batches=config.validation.batches_per_source,
        )
    return result


def create_source_frames(
    config: pretrain_config.PretrainConfig,
    source: pretrain_config.RldsSourceConfig,
    *,
    split: str,
) -> Any:
    """Create a finite, unnormalized source stream for normalization statistics."""
    return _create_source_dataset(
        config,
        source,
        split=split,
        training=False,
        normalize_data=False,
        statistics_only=True,
    )


def _local_batch_size(global_batch_size: int) -> int:
    if global_batch_size % jax.process_count() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by process count {jax.process_count()}"
        )
    return global_batch_size // jax.process_count()


def _create_source_dataset(
    config: pretrain_config.PretrainConfig,
    source: pretrain_config.RldsSourceConfig,
    *,
    split: str,
    training: bool,
    normalize_data: bool,
    statistics_only: bool = False,
) -> Any:
    # Keep TensorFlow and DLimP optional for users that only fine-tune with LeRobot.
    try:
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RLDS pretraining dependencies are missing; run `uv sync --group rlds` with Python 3.11"
        ) from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("TensorFlow initialized a GPU before the RLDS loader could disable it") from exc
    parallel_reads = _autotune(config.data.num_parallel_reads, tf)
    parallel_calls = _autotune(config.data.num_parallel_calls, tf)
    builder = _create_tfds_builder(source, tfds)
    available_splits = set(builder.info.splits)
    split_name = split.split("[")[0]
    if split_name not in available_splits:
        raise ValueError(f"Source {source.id!r} requests split {split!r}, but TFDS provides {sorted(available_splits)}")
    dataset = dl.DLataset.from_rlds(
        builder,
        split=split,
        shuffle=training or statistics_only,
        num_parallel_reads=parallel_reads,
    )
    if jax.process_count() > 1:
        dataset = dataset.shard(jax.process_count(), jax.process_index())

    adapter = rlds_adapters.create_adapter(source)
    stats = load_stats(config, source) if normalize_data else None

    def prepare_trajectory(trajectory: Mapping[str, Any]) -> dict[str, Any]:
        canonical = adapter.adapt_trajectory(trajectory)
        if statistics_only:
            return {
                "state": tf.ensure_shape(tf.cast(canonical["state"], tf.float32), [None, source.state_dim]),
                "actions": tf.ensure_shape(tf.cast(canonical["actions"], tf.float32), [None, source.action_dim]),
            }
        return _chunk_and_prepare(config, source, canonical, stats)

    dataset = dataset.traj_map(prepare_trajectory, num_parallel_calls=parallel_calls)
    dataset = dataset.flatten(num_parallel_calls=parallel_calls)
    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(config.data.shuffle_buffer_size, seed=config.seed + jax.process_index())
    elif statistics_only:
        dataset = dataset.shuffle(config.data.shuffle_buffer_size, seed=config.seed)
    options = tf.data.Options()
    options.deterministic = config.data_resume_mode == "exact" or not (training or statistics_only)
    return dataset.with_options(options)


def _create_tfds_builder(source: pretrain_config.RldsSourceConfig, tfds: Any) -> Any:
    """Load registered TFDS builders or externally generated folder datasets."""
    external_dir = pathlib.Path(source.data_dir) / source.tfds_name / source.version
    if (external_dir / "dataset_info.json").is_file() and (external_dir / "features.json").is_file():
        return tfds.builder_from_directory(external_dir)
    return tfds.builder(source.tfds_name, data_dir=source.data_dir, version=source.version)


def _chunk_and_prepare(
    config: pretrain_config.PretrainConfig,
    source: pretrain_config.RldsSourceConfig,
    trajectory: Mapping[str, Any],
    stats: dict[str, normalize.NormStats] | None,
) -> dict[str, Any]:
    import tensorflow as tf

    state = tf.ensure_shape(tf.cast(trajectory["state"], tf.float32), [None, source.state_dim])
    actions = tf.ensure_shape(tf.cast(trajectory["actions"], tf.float32), [None, source.action_dim])
    length = tf.shape(actions)[0]
    horizon = config.model.action_horizon
    offsets = tf.range(horizon, dtype=tf.int32) * source.action_stride
    indices = tf.range(length, dtype=tf.int32)[:, None] + offsets[None, :]
    valid_time = indices < length
    clipped = tf.minimum(indices, tf.maximum(length - 1, 0))
    action_chunks = tf.gather(actions, clipped)

    if stats is not None:
        state = _normalize_tensor(state, stats["state"], use_quantiles=True)
        action_chunks = _normalize_tensor(action_chunks, stats["actions"], use_quantiles=True)

    state_padding = config.model.action_dim - source.state_dim
    action_padding = config.model.action_dim - source.action_dim
    state = tf.pad(state, [[0, 0], [0, state_padding]])
    action_chunks = tf.pad(action_chunks, [[0, 0], [0, 0], [0, action_padding]])
    dimension_mask = tf.sequence_mask(source.action_dim, config.model.action_dim)
    action_mask = valid_time[..., None] & dimension_mask[None, None, :]
    source_id = config.source_indices[source.id]

    return {
        "image": {
            key: tf.ensure_shape(tf.cast(trajectory["image"][key], tf.float32), [None, 224, 224, 3])
            for key in rlds_adapters.CANONICAL_IMAGE_KEYS
        },
        "image_mask": {
            key: tf.ensure_shape(tf.cast(trajectory["image_mask"][key], tf.bool), [None])
            for key in rlds_adapters.CANONICAL_IMAGE_KEYS
        },
        "state": tf.ensure_shape(state, [None, config.model.action_dim]),
        "actions": tf.ensure_shape(action_chunks, [None, horizon, config.model.action_dim]),
        "action_mask": tf.ensure_shape(action_mask, [None, horizon, config.model.action_dim]),
        "prompt": tf.ensure_shape(trajectory["prompt"], [None]),
        "source_id": tf.fill([length], tf.cast(source_id, tf.int32)),
    }


def _normalize_tensor(value: Any, stats: normalize.NormStats, *, use_quantiles: bool) -> Any:
    import tensorflow as tf

    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("pi0.5 pre-training requires q01/q99 normalization statistics")
        low = tf.constant(np.asarray(stats.q01), dtype=tf.float32)
        high = tf.constant(np.asarray(stats.q99), dtype=tf.float32)
        return (value - low) / (high - low + 1e-6) * 2.0 - 1.0
    mean = tf.constant(np.asarray(stats.mean), dtype=tf.float32)
    std = tf.constant(np.asarray(stats.std), dtype=tf.float32)
    return (value - mean) / (std + 1e-6)


def _sample_and_batch(
    datasets: list[Any],
    probabilities: Any,
    *,
    batch_size: int,
    seed: int,
    prefetch_batches: int,
    training: bool,
    config: pretrain_config.PretrainConfig | None = None,
    start_step: int = 0,
    mixture_runtime: _MixtureRuntime | None = None,
) -> Any:
    import dlimp as dl
    import tensorflow as tf

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        weights: Any = list(probabilities)
        if (
            training
            and config is not None
            and (
                config.data.mixing.schedule
                or config.data.mixing.source_limits
                or config.data.mixing.source_failure_policy == "degrade"
            )
        ):
            weights = _dynamic_weight_dataset(
                config,
                local_batch_size=batch_size,
                start_step=start_step,
                tf=tf,
                mixture_runtime=mixture_runtime,
            )
        dataset = dl.DLataset.sample_from_datasets(datasets, weights=weights, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    options = tf.data.Options()
    options.deterministic = (config is not None and config.data_resume_mode == "exact") or not training
    dataset = dataset.with_options(options)
    return dataset.prefetch(prefetch_batches)


def _dynamic_weight_dataset(
    config: pretrain_config.PretrainConfig,
    *,
    local_batch_size: int,
    start_step: int,
    tf: Any,
    mixture_runtime: _MixtureRuntime | None = None,
) -> Any:
    """Create a deterministic per-sample probability stream evaluated at optimizer-step boundaries."""
    source_ids = [source.id for source in config.data.sources]
    base_weights = {source.id: source.weight for source in config.data.sources}
    points = list(config.data.mixing.schedule)
    if not points or points[0].step != 0:
        points.insert(0, pretrain_config.MixingSchedulePoint(0, config.data.temperature, {}))
    steps = tf.constant([point.step for point in points], dtype=tf.int64)
    temperatures = tf.constant([point.temperature for point in points], dtype=tf.float32)
    weights = tf.constant(
        [[{**base_weights, **point.weights}[source_id] for source_id in source_ids] for point in points],
        dtype=tf.float32,
    )
    minimums = tf.constant(
        [
            config.data.mixing.source_limits.get(
                source_id, pretrain_config.SourceLimit(0, 1, None, None)
            ).min_probability
            for source_id in source_ids
        ],
        dtype=tf.float32,
    )
    maximums = tf.constant(
        [
            config.data.mixing.source_limits.get(
                source_id, pretrain_config.SourceLimit(0, 1, None, None)
            ).max_probability
            for source_id in source_ids
        ],
        dtype=tf.float32,
    )
    active = (
        mixture_runtime.attach_tensorflow(tf)
        if mixture_runtime is not None
        else tf.constant([True] * len(source_ids), dtype=tf.bool)
    )

    def probabilities(sample_index: Any) -> Any:
        optimizer_step = tf.cast(start_step, tf.int64) + sample_index // tf.cast(local_batch_size, tf.int64)
        left = tf.maximum(tf.searchsorted(steps, optimizer_step[None], side="right")[0] - 1, 0)
        left_weights = tf.gather(weights, left)
        left_temperature = tf.gather(temperatures, left)
        if config.data.mixing.interpolation == "linear":
            right = tf.minimum(left + 1, tf.cast(len(points) - 1, left.dtype))
            denominator = tf.maximum(tf.gather(steps, right) - tf.gather(steps, left), 1)
            ratio = tf.cast(optimizer_step - tf.gather(steps, left), tf.float32) / tf.cast(denominator, tf.float32)
            selected_weights = left_weights + ratio * (tf.gather(weights, right) - left_weights)
            selected_temperature = left_temperature + ratio * (tf.gather(temperatures, right) - left_temperature)
        else:
            selected_weights = left_weights
            selected_temperature = left_temperature
        logits = tf.math.log(selected_weights) / selected_temperature
        result = tf.nn.softmax(logits)
        result = tf.where(active, result, 0.0)
        effective_minimums = tf.where(active, minimums, 0.0)
        effective_maximums = tf.where(active, maximums, 0.0)
        # Iteratively clamp violated coordinates and redistribute the remaining mass among free sources.
        free = tf.ones_like(result, dtype=tf.bool)
        fixed = tf.zeros_like(result)
        remaining = tf.constant(1.0, dtype=tf.float32)
        for _ in source_ids:
            free_weights = tf.where(free, result, 0.0)
            proposed = tf.where(
                free,
                remaining * free_weights / tf.maximum(tf.reduce_sum(free_weights), 1e-12),
                fixed,
            )
            below = free & (proposed < effective_minimums)
            above = free & (proposed > effective_maximums)
            newly_fixed = below | above
            bounded = tf.where(below, effective_minimums, tf.where(above, effective_maximums, fixed))
            fixed = tf.where(newly_fixed, bounded, fixed)
            free = free & ~newly_fixed
            remaining = 1.0 - tf.reduce_sum(fixed)
        free_weights = tf.where(free, result, 0.0)
        free_values = remaining * free_weights / tf.maximum(tf.reduce_sum(free_weights), 1e-12)
        return tf.where(free, free_values, fixed)

    return tf.data.Dataset.counter().map(probabilities, num_parallel_calls=1, deterministic=True)


def _autotune(value: int, tf: Any) -> int:
    return tf.data.AUTOTUNE if value == -1 else value
