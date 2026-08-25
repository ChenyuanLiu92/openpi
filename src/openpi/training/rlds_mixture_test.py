import dataclasses
import json
import pathlib
import types

import numpy as np
import pytest

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
