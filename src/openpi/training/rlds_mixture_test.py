import dataclasses
import json
import pathlib
import types

import numpy as np
import pytest

from openpi.training import pretrain_config
from openpi.training import pretrain_config_loader
from openpi.training import rlds_adapters
from openpi.training import rlds_mixture

tf = pytest.importorskip("tensorflow")


def _small_config():
    path = pathlib.Path(__file__).parents[3] / "configs" / "pretraining" / "pi05" / "template.yaml"
    config = pretrain_config_loader.load(path).config
    source = dataclasses.replace(config.data.sources[0], state_dim=2, action_dim=2)
    model = dataclasses.replace(config.model, action_dim=5, action_horizon=3)
    return dataclasses.replace(config, model=model, data=dataclasses.replace(config.data, sources=(source,))), source


def test_field_map_and_chunk_create_padding_masks():
    config, source = _small_config()
    trajectory = {
        "observation": {
            "image": tf.zeros([2, 16, 20, 3], dtype=tf.uint8),
            "wrist_image": tf.ones([2, 8, 12, 3], dtype=tf.uint8) * 255,
            "state": tf.constant([[1.0, 2.0], [3.0, 4.0]]),
        },
        "action": tf.constant([[5.0, 6.0], [7.0, 8.0]]),
        "task": {"language_instruction": tf.constant("move")},
    }

    canonical = rlds_adapters.create_adapter(source).adapt_trajectory(trajectory)
    batch = rlds_mixture._chunk_and_prepare(config, source, canonical, None)  # noqa: SLF001

    assert batch["state"].shape == (2, 5)
    assert batch["actions"].shape == (2, 3, 5)
    assert batch["image"]["base_0_rgb"].shape == (2, 224, 224, 3)
    np.testing.assert_array_equal(batch["state"].numpy()[:, 2:], 0)
    np.testing.assert_array_equal(batch["image_mask"]["right_wrist_0_rgb"].numpy(), [False, False])
    np.testing.assert_array_equal(batch["prompt"].numpy(), [b"move", b"move"])
    np.testing.assert_array_equal(batch["action_mask"].numpy().sum(axis=(1, 2)), [4, 2])


def test_create_tfds_builder_loads_external_folder_dataset(tmp_path):
    _, source = _small_config()
    source = dataclasses.replace(source, data_dir=str(tmp_path), tfds_name="external", version="1.0.0")
    external_dir = tmp_path / "external" / "1.0.0"
    external_dir.mkdir(parents=True)
    (external_dir / "dataset_info.json").write_text("{}")
    (external_dir / "features.json").write_text("{}")
    calls = []
    fake_tfds = types.SimpleNamespace(
        builder_from_directory=lambda path: calls.append(("folder", path)) or "external-builder",
        builder=lambda *args, **kwargs: calls.append(("registered", args, kwargs)) or "registered-builder",
    )

    builder = rlds_mixture._create_tfds_builder(source, fake_tfds)  # noqa: SLF001

    assert builder == "external-builder"
    assert calls == [("folder", external_dir)]


def test_build_lineage_inherits_conversion_lineage_id(tmp_path):
    config, source = _small_config()
    source = dataclasses.replace(source, data_dir=str(tmp_path), tfds_name="external", version="1.0.0")
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sources=(source,)))
    version_dir = tmp_path / "external" / "1.0.0"
    version_dir.mkdir(parents=True)
    (version_dir / "conversion_manifest.json").write_text(json.dumps({"lineage_id": "conversion-123"}))

    lineage = rlds_mixture.build_lineage(config, {"config": "snapshot"})

    assert lineage["lineage_id"] == "conversion-123"
    assert lineage["datasets"][source.id]["conversion_lineage_id"] == "conversion-123"


class _FakeDataset:
    def __init__(self, batches):
        self._batches = batches

    def as_numpy_iterator(self):
        return iter(self._batches)


class _FakeTokenizer:
    def __init__(self, max_len):
        self._max_len = max_len

    def tokenize_batch(self, prompts, states, *, num_threads):
        del states, num_threads
        return (
            np.ones((len(prompts), self._max_len), dtype=np.int32),
            np.ones((len(prompts), self._max_len), dtype=bool),
        )


def _numpy_batch(batch_size: int, action_dim: int, horizon: int):
    return {
        "image": {"base_0_rgb": np.zeros((batch_size, 4, 4, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.ones(batch_size, dtype=bool)},
        "state": np.zeros((batch_size, action_dim), dtype=np.float32),
        "actions": np.zeros((batch_size, horizon, action_dim), dtype=np.float32),
        "action_mask": np.ones((batch_size, horizon, action_dim), dtype=bool),
        "prompt": np.asarray([b"move"] * batch_size),
        "source_id": np.zeros(batch_size, dtype=np.int32),
    }


def test_loader_prefetches_batch_tokenization_and_reshapes_microbatches(monkeypatch):
    config, _ = _small_config()
    pipeline = dataclasses.replace(config.data.pipeline, host_prefetch_batches=2, device_prefetch_batches=1)
    config = dataclasses.replace(
        config,
        batch_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        data=dataclasses.replace(config.data, pipeline=pipeline),
    )
    monkeypatch.setattr(rlds_mixture._tokenizer, "PaligemmaTokenizer", _FakeTokenizer)  # noqa: SLF001
    loader = rlds_mixture.RldsMixtureDataLoader(
        config,
        _FakeDataset([_numpy_batch(4, config.model.action_dim, config.model.action_horizon)] * 4),
        sharding=None,
    )

    batch = next(iter(loader))

    assert batch.actions.shape == (2, 2, 3, 5)
    assert batch.observation.tokenized_prompt.shape == (2, 2, config.model.max_token_len)
    assert loader.data_state()["consumed_examples_per_source"] == {"example_robot": 4}
    assert loader.metrics()["consumed_batches"] == 1


def test_exact_iterator_snapshot_restores_next_batch_without_replay(monkeypatch, tmp_path):
    config, _ = _small_config()
    config = dataclasses.replace(config, batch_size=1, micro_batch_size=1, gradient_accumulation_steps=1)
    monkeypatch.setattr(rlds_mixture._tokenizer, "PaligemmaTokenizer", _FakeTokenizer)  # noqa: SLF001
    batches = [_numpy_batch(1, config.model.action_dim, config.model.action_horizon) for _ in range(3)]
    for index, batch in enumerate(batches):
        batch["state"].fill(index)

    def stack(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: stack([value[key] for value in values]) for key in first}
        return np.stack(values)

    dataset = tf.data.Dataset.from_tensor_slices(stack(batches))
    loader = rlds_mixture.RldsMixtureDataLoader(config, dataset, sharding=None)
    iterator = iter(loader)
    next(iterator)
    snapshot = tmp_path / "rank-00000"
    loader.snapshot_iterator(snapshot)
    expected = next(iterator)

    restored = rlds_mixture.RldsMixtureDataLoader(
        config,
        tf.data.Dataset.from_tensor_slices(stack(batches)),
        sharding=None,
        iterator_state_dir=snapshot,
    )
    actual = next(iter(restored))

    np.testing.assert_array_equal(actual.observation.state, expected.observation.state)


def test_dynamic_weight_stream_changes_at_optimizer_step():
    config, source = _small_config()
    second = dataclasses.replace(source, id="second", weight=1.0)
    mixing = dataclasses.replace(
        config.data.mixing,
        schedule=(pretrain_config.MixingSchedulePoint(step=1, temperature=1.0, weights={source.id: 9.0}),),
    )
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sources=(source, second), mixing=mixing))

    stream = rlds_mixture._dynamic_weight_dataset(  # noqa: SLF001
        config, local_batch_size=1, start_step=0, tf=tf
    ).take(2)
    values = list(stream.as_numpy_iterator())

    np.testing.assert_allclose(values[0], [0.5, 0.5])
    np.testing.assert_allclose(values[1], [0.9, 0.1])


def test_source_probability_bounds_are_projected_without_schedule():
    config, source = _small_config()
    second = dataclasses.replace(source, id="second", weight=9.0)
    limits = {
        source.id: pretrain_config.SourceLimit(0.3, 0.4, None, None),
        second.id: pretrain_config.SourceLimit(0.0, 0.7, None, None),
    }
    mixing = dataclasses.replace(config.data.mixing, source_limits=limits)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sources=(source, second), mixing=mixing))

    assert config.data.effective_probabilities() == pytest.approx((0.3, 0.7))
    values = list(
        rlds_mixture._dynamic_weight_dataset(config, local_batch_size=1, start_step=0, tf=tf)  # noqa: SLF001
        .take(1)
        .as_numpy_iterator()
    )
    np.testing.assert_allclose(values[0], [0.3, 0.7], atol=1e-6)


def test_degraded_source_is_removed_from_dynamic_weights():
    config, source = _small_config()
    second = dataclasses.replace(source, id="second")
    mixing = dataclasses.replace(
        config.data.mixing,
        source_failure_policy="degrade",
        consecutive_failure_threshold=1,
    )
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sources=(source, second), mixing=mixing))
    runtime = rlds_mixture._MixtureRuntime(config, None)  # noqa: SLF001
    weights = rlds_mixture._dynamic_weight_dataset(  # noqa: SLF001
        config,
        local_batch_size=1,
        start_step=0,
        tf=tf,
        mixture_runtime=runtime,
    )

    assert runtime.record_failure(np.asarray([0], dtype=np.int32)) == (source.id,)
    value = next(iter(weights.as_numpy_iterator()))

    np.testing.assert_allclose(value, [0.0, 1.0], atol=1e-6)


def test_global_sample_limit_is_partitioned_across_ranks(monkeypatch):
    monkeypatch.setattr(rlds_mixture.jax, "process_count", lambda: 3)
    monkeypatch.setattr(rlds_mixture.jax, "process_index", lambda: 1)

    # Seven samples remain globally: rank quotas are [3, 2, 2].
    assert rlds_mixture._local_remaining_quota(12, 5) == 2  # noqa: SLF001


def test_minimum_sample_quota_forces_source_before_run_ends():
    config, source = _small_config()
    second = dataclasses.replace(source, id="second", weight=99.0)
    limits = {source.id: pretrain_config.SourceLimit(0.0, 1.0, 1, None)}
    mixing = dataclasses.replace(config.data.mixing, source_limits=limits)
    config = dataclasses.replace(
        config,
        num_train_steps=2,
        data=dataclasses.replace(config.data, sources=(dataclasses.replace(source, weight=1.0), second), mixing=mixing),
    )
    runtime = rlds_mixture._MixtureRuntime(config, None)  # noqa: SLF001
    values = rlds_mixture._dynamic_weight_dataset(  # noqa: SLF001
        config, local_batch_size=1, start_step=1, tf=tf, mixture_runtime=runtime
    )

    np.testing.assert_allclose(next(iter(values.as_numpy_iterator())), [1.0, 0.0], atol=1e-6)


def test_source_with_unmet_minimum_cannot_degrade():
    config, source = _small_config()
    second = dataclasses.replace(source, id="second")
    mixing = dataclasses.replace(
        config.data.mixing,
        source_limits={source.id: pretrain_config.SourceLimit(0.0, 1.0, 1, None)},
        source_failure_policy="degrade",
        consecutive_failure_threshold=1,
    )
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, sources=(source, second), mixing=mixing))
    runtime = rlds_mixture._MixtureRuntime(config, None)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="Cannot degrade source"):
        runtime.record_failure(np.asarray([0], dtype=np.int32))
