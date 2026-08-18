import dataclasses
import pathlib

import numpy as np
import pytest
import yaml

from openpi.shared import normalize
from openpi.training import pretrain_config_loader
from openpi.training import rlds_adapters
from openpi.training import rlds_mixture


def _template_path() -> pathlib.Path:
    return pathlib.Path(__file__).parents[3] / "configs" / "pretraining" / "pi05" / "template.yaml"


def test_complete_pi05_pretrain_template_is_valid():
    resolved = pretrain_config_loader.load(_template_path())

    assert resolved.config.name == "pi05_rlds_pretrain"
    assert resolved.config.model.pi05 is True
    assert resolved.config.exp_name == "baseline"
    assert resolved.config.distributed.warmup_collectives is True
    assert resolved.config.data.effective_probabilities() == (1.0,)
    assert (
        resolved.config.checkpoint_dir == pathlib.Path("checkpoints/pretraining/pi05_rlds_pretrain/baseline").resolve()
    )
    assert rlds_adapters.registered_adapters() == ("field_map",)
    adapter = rlds_adapters.create_adapter(resolved.config.data.sources[0])
    assert isinstance(adapter, rlds_adapters.FieldMapAdapter)


def test_pretrain_config_supports_cli_overrides():
    resolved = pretrain_config_loader.parse_cli(
        [
            str(_template_path()),
            "--exp-name",
            "lr_test",
            "--batch-size",
            "8",
            "--data.temperature",
            "2.0",
            "--distributed.no-warmup-collectives",
        ]
    )

    assert resolved.config.exp_name == "lr_test"
    assert resolved.config.batch_size == 8
    assert resolved.config.data.temperature == 2.0
    assert resolved.config.distributed.warmup_collectives is False
    assert resolved.manifest["checkpoint"]["exp_name"] == "baseline"


def test_pretrain_snapshot_is_safe_yaml(tmp_path: pathlib.Path):
    resolved = pretrain_config_loader.load(_template_path())
    path = tmp_path / "metadata" / "train_config.yaml"

    pretrain_config_loader.write_snapshot(path, resolved.snapshot())
    snapshot = yaml.safe_load(path.read_text())

    assert snapshot["kind"] == "pi05_pretrain"
    assert snapshot["manifest"]["data"]["sources"][0]["id"] == "example_robot"
    assert snapshot["resolved_mixture_probabilities"] == {"example_robot": 1.0}
    assert "commit" in snapshot["git"]


def test_pretrain_config_rejects_missing_and_duplicate_fields(tmp_path: pathlib.Path):
    contents = yaml.safe_load(_template_path().read_text())
    del contents["model"]["action_dim"]
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text(yaml.safe_dump(contents))

    with pytest.raises(pretrain_config_loader.ConfigError, match="Missing required fields at model.*action_dim"):
        pretrain_config_loader.load(missing_path)

    duplicate_path = tmp_path / "duplicate.yaml"
    duplicate_path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(pretrain_config_loader.ConfigError, match="Duplicate YAML key"):
        pretrain_config_loader.load(duplicate_path)


def test_pretrain_config_rejects_unsafe_yaml(tmp_path: pathlib.Path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("schema_version: !!python/object/apply:os.system ['echo unsafe']\n")

    with pytest.raises(pretrain_config_loader.ConfigError, match="Invalid YAML"):
        pretrain_config_loader.load(path)


def test_temperature_adjusts_source_probabilities():
    config = pretrain_config_loader.load(_template_path()).config
    source = config.data.sources[0]
    data = dataclasses.replace(
        config.data,
        temperature=2.0,
        sources=(source, dataclasses.replace(source, id="second", weight=4.0)),
    )

    assert data.effective_probabilities() == pytest.approx((1 / 3, 2 / 3))


def test_field_map_rejects_unknown_options():
    config = pretrain_config_loader.load(_template_path()).config
    source = config.data.sources[0]
    adapter = dataclasses.replace(source.adapter, options={**source.adapter.options, "python_callable": "bad"})

    with pytest.raises(rlds_adapters.AdapterError, match="Unknown field_map options"):
        rlds_adapters.create_adapter(dataclasses.replace(source, adapter=adapter))


def test_stats_manifest_detects_source_changes(tmp_path: pathlib.Path):
    config = pretrain_config_loader.load(_template_path()).config
    config = dataclasses.replace(config, assets_base_dir=str(tmp_path))
    stats = {
        "state": normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=-np.ones(8), q99=np.ones(8)),
        "actions": normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=-np.ones(8), q99=np.ones(8)),
    }
    source = config.data.sources[0]
    rlds_mixture.save_stats(config, source.normalization_id, stats, sample_count=100)

    loaded = rlds_mixture.load_stats(config, source)
    assert np.array_equal(loaded["state"].mean, np.zeros(8))

    changed_source = dataclasses.replace(source, action_stride=2)
    changed_config = dataclasses.replace(config, data=dataclasses.replace(config.data, sources=(changed_source,)))
    with pytest.raises(ValueError, match="manifest mismatch"):
        rlds_mixture.load_stats(changed_config, changed_source)
