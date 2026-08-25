import dataclasses
import pathlib

import numpy as np
import pytest
import yaml

from openpi.shared import normalize
from openpi.training import pretrain_config_loader
from openpi.training import rlds_adapters
from openpi.training import rlds_mixture
from openpi.training import weight_loaders


def _template_path() -> pathlib.Path:
    return pathlib.Path(__file__).parents[3] / "configs" / "pretraining" / "pi05" / "template.yaml"


def test_complete_pi05_pretrain_template_is_valid():
    resolved = pretrain_config_loader.load(_template_path())

    assert resolved.config.name == "pi05_rlds_pretrain"
    assert resolved.config.model.pi05 is True
    assert resolved.config.exp_name == "baseline"
    assert resolved.config.distributed.warmup_collectives is True
    assert resolved.config.distributed.coordinator_bind_address is None
    assert resolved.config.data.effective_probabilities() == (1.0,)
    assert (
        resolved.config.checkpoint_dir == pathlib.Path("checkpoints/pretraining/pi05_rlds_pretrain/baseline").resolve()
    )
    assert rlds_adapters.registered_adapters() == ("field_map",)
    adapter = rlds_adapters.create_adapter(resolved.config.data.sources[0])
    assert isinstance(adapter, rlds_adapters.FieldMapAdapter)
    assert resolved.config.wandb_mode == "online"
    assert resolved.config.system_interval_seconds == 10
    assert resolved.config.runtime.compilation_cache.enabled is True
    assert resolved.config.runtime.fatal_cleanup_timeout_seconds == 15.0
    assert resolved.config.micro_batch_size == 4
    assert resolved.config.gradient_accumulation_steps == 8
    assert resolved.config.data.pipeline.tokenizer_threads == 8
    assert resolved.config.data_resume_mode == "exact"
    assert resolved.config.distributed.diagnostics.tensor_sizes_mib == (1.0, 16.0, 64.0, 256.0)


def test_schema_v1_is_migrated_with_compatible_defaults(tmp_path: pathlib.Path):
    contents = yaml.safe_load(_template_path().read_text())
    contents["schema_version"] = 1
    del contents["data"]["pipeline"]
    del contents["data"]["mixing"]
    del contents["training"]["micro_batch_size"]
    del contents["training"]["gradient_accumulation_steps"]
    del contents["checkpoint"]["data_resume_mode"]
    del contents["checkpoint"]["on_topology_change"]
    del contents["distributed"]["diagnostics"]
    path = tmp_path / "v1.yaml"
    path.write_text(yaml.safe_dump(contents))

    resolved = pretrain_config_loader.load(path)

    assert resolved.config.micro_batch_size is None
    assert resolved.config.gradient_accumulation_steps == 1
    assert resolved.config.data.pipeline.host_prefetch_batches == 0
    assert resolved.config.data_resume_mode == "statistical"
    assert resolved.snapshot()["source_schema_version"] == 1


def test_old_logging_section_receives_observability_defaults(tmp_path: pathlib.Path):
    contents = yaml.safe_load(_template_path().read_text())
    contents["logging"] = {
        "project_name": "legacy-project",
        "wandb_enabled": False,
        "log_interval": 20,
    }
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(contents))

    config = pretrain_config_loader.load(path).config

    assert config.wandb_mode == "online"
    assert config.observability_local_root is None
    assert config.stall_timeout_seconds == 600


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
            "--runtime.compilation-cache.explain-misses",
        ]
    )

    assert resolved.config.exp_name == "lr_test"
    assert resolved.config.batch_size == 8
    assert resolved.config.data.temperature == 2.0
    assert resolved.config.distributed.warmup_collectives is False
    assert resolved.config.runtime.compilation_cache.explain_misses is True
    assert resolved.manifest["checkpoint"]["exp_name"] == "baseline"


def test_pretrain_config_supports_random_initialization(tmp_path: pathlib.Path):
    contents = yaml.safe_load(_template_path().read_text())
    contents["initialization"] = {"type": "random", "params_path": None}
    path = tmp_path / "random.yaml"
    path.write_text(yaml.safe_dump(contents))

    config = pretrain_config_loader.load(path).config

    assert isinstance(config.weight_loader, weight_loaders.NoOpWeightLoader)


def test_random_initialization_rejects_params_path(tmp_path: pathlib.Path):
    contents = yaml.safe_load(_template_path().read_text())
    contents["initialization"] = {"type": "random", "params_path": "/tmp/params"}
    path = tmp_path / "invalid-random.yaml"
    path.write_text(yaml.safe_dump(contents))

    with pytest.raises(pretrain_config_loader.ConfigError, match="must be null for random"):
        pretrain_config_loader.load(path)


def test_pretrain_config_validates_explicit_distributed_topology():
    config = pretrain_config_loader.load(_template_path()).config
    incomplete = dataclasses.replace(config.distributed, initialize=True, coordinator_address="node0:12345")

    with pytest.raises(ValueError, match="requires coordinator_address"):
        dataclasses.replace(config, distributed=incomplete)

    complete = dataclasses.replace(
        config.distributed,
        initialize=True,
        coordinator_address="node0:12345",
        coordinator_bind_address="[::]:12345",
        num_processes=2,
        process_id=0,
        local_device_ids=(0, 1, 2, 3),
    )
    assert dataclasses.replace(config, distributed=complete).distributed.num_processes == 2


def test_pretrain_config_rejects_duplicate_local_devices():
    config = pretrain_config_loader.load(_template_path()).config
    distributed = dataclasses.replace(
        config.distributed,
        initialize=True,
        coordinator_address="node0:12345",
        num_processes=2,
        process_id=0,
        local_device_ids=(0, 0),
    )

    with pytest.raises(ValueError, match="must be unique"):
        dataclasses.replace(config, distributed=distributed)


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
