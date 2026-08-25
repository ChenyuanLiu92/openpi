"""High-throughput, source-aware RLDS mixture loading for pi0.5 pre-training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
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
    ):
        self._config = config
        self._dataset = dataset
        self._sharding = sharding
        self._num_batches = num_batches
        self._tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
        self._base_counts = {source.id: int((initial_counts or {}).get(source.id, 0)) for source in config.data.sources}
        self._session_counts = {source.id: 0 for source in config.data.sources}

    def __iter__(self) -> Iterator[PretrainBatch]:
        batches = self._dataset.as_numpy_iterator()
        if self._num_batches is not None:
            batches = itertools.islice(batches, self._num_batches)
        for batch in batches:
            source_ids = np.asarray(batch["source_id"], dtype=np.int32)
            bincount = np.bincount(source_ids, minlength=len(self._config.data.sources))
            for index, source in enumerate(self._config.data.sources):
                self._session_counts[source.id] += int(bincount[index])

            prompts = np.asarray(batch.pop("prompt"))
            states = np.asarray(batch["state"])
            token_values = []
            token_masks = []
            for prompt, state, source_index in zip(prompts, states, source_ids, strict=True):
                prompt_text = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
                source = self._config.data.sources[int(source_index)]
                tokens, mask = self._tokenizer.tokenize(
                    prompt_text,
                    state[: source.state_dim] if self._config.model.discrete_state_input else None,
                )
                token_values.append(tokens)
                token_masks.append(mask)

            observation = _model.Observation(
                images={key: np.asarray(value) for key, value in batch["image"].items()},
                image_masks={key: np.asarray(value) for key, value in batch["image_mask"].items()},
                state=states.astype(np.float32),
                tokenized_prompt=np.stack(token_values).astype(np.int32),
                tokenized_prompt_mask=np.stack(token_masks).astype(bool),
            )
            pretrain_batch = PretrainBatch(
                observation=observation,
                actions=np.asarray(batch["actions"], dtype=np.float32),
                action_mask=np.asarray(batch["action_mask"], dtype=bool),
                source_id=source_ids,
            )
            if self._sharding is not None:
                pretrain_batch = jax.tree.map(
                    lambda value: jax.make_array_from_process_local_data(self._sharding, value), pretrain_batch
                )
            yield pretrain_batch

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
            "resume_semantics": "statistical",
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
    datasets = [
        _create_source_dataset(config, source, split=source.train_split, training=True, normalize_data=True)
        for source in config.data.sources
    ]
    probabilities = config.data.effective_probabilities()
    seed = config.seed + start_step * 9973 + jax.process_index()
    dataset = _sample_and_batch(
        datasets,
        probabilities,
        batch_size=local_batch_size,
        seed=seed,
        prefetch_batches=config.data.prefetch_batches,
        training=True,
    )
    return RldsMixtureDataLoader(config, dataset, sharding=sharding, initial_counts=initial_counts)


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
    options.deterministic = not (training or statistics_only)
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
) -> Any:
    import dlimp as dl
    import tensorflow as tf

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = dl.DLataset.sample_from_datasets(datasets, weights=list(probabilities), seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    options = tf.data.Options()
    options.deterministic = not training
    dataset = dataset.with_options(options)
    return dataset.prefetch(prefetch_batches)


def _autotune(value: int, tf: Any) -> int:
    return tf.data.AUTOTUNE if value == -1 else value
