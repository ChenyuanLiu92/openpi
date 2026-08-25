import pathlib

import pytest
import yaml

from openpi.training import config_loader


def _write_config(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text)
    return path


def test_load_yaml_config_and_nested_overrides(tmp_path: pathlib.Path):
    path = _write_config(
        tmp_path / "experiment.yaml",
        """
schema_version: 1
name: yaml_debug
base: debug
overrides:
  batch_size: 4
  model:
    action_horizon: 12
""",
    )

    resolved = config_loader.load_yaml_config(path)

    assert resolved.config.name == "yaml_debug"
    assert resolved.config.batch_size == 4
    assert resolved.config.model.action_horizon == 12
    assert resolved.base == "debug"


def test_yaml_config_supports_tyro_overrides(tmp_path: pathlib.Path):
    path = _write_config(
        tmp_path / "experiment.yaml",
        """
schema_version: 1
name: yaml_debug
base: debug
overrides:
  batch_size: 4
""",
    )

    resolved = config_loader.parse_cli(
        [str(path), "--exp-name", "cli-run", "--batch-size", "8", "--model.action-horizon", "10"]
    )

    assert resolved.config.exp_name == "cli-run"
    assert resolved.config.batch_size == 8
    assert resolved.config.model.action_horizon == 10
    assert resolved.cli_args == (
        "--exp-name",
        "cli-run",
        "--batch-size",
        "8",
        "--model.action-horizon",
        "10",
    )


def test_load_standalone_pi05_config():
    config_path = pathlib.Path(__file__).parents[3] / "configs" / "experiments" / "pi05_template.yaml"

    resolved = config_loader.load_yaml_config(config_path)

    assert resolved.base is None
    assert resolved.manifest is not None
    assert resolved.config.name == "pi05_libero_standalone"
    assert resolved.config.exp_name == "baseline"
    assert resolved.config.model.pi05 is True
    assert repr(resolved.config.freeze_filter) == "Nothing()"
    assert resolved.config.data.repo_id == "physical-intelligence/libero"
    assert resolved.config.batch_size == 256
    assert resolved.config.checkpoint_dir == pathlib.Path("checkpoints/pi05_libero_standalone/baseline").resolve()


def test_standalone_config_supports_tyro_overrides():
    config_path = pathlib.Path(__file__).parents[3] / "configs" / "experiments" / "pi05_template.yaml"

    resolved = config_loader.parse_cli([str(config_path), "--exp-name", "override", "--batch-size", "8"])

    assert resolved.config.exp_name == "override"
    assert resolved.config.batch_size == 8
    assert resolved.manifest["checkpoint"]["exp_name"] == "baseline"
    assert resolved.cli_args == ("--exp-name", "override", "--batch-size", "8")


def test_standalone_config_requires_all_sections(tmp_path: pathlib.Path):
    path = _write_config(
        tmp_path / "incomplete.yaml",
        """
schema_version: 1
name: incomplete
model:
  type: pi0
""",
    )

    with pytest.raises(config_loader.ConfigError, match="Missing required fields at config"):
        config_loader.load_yaml_config(path)


def test_standalone_config_rejects_unknown_component_type(tmp_path: pathlib.Path):
    template_path = pathlib.Path(__file__).parents[3] / "configs" / "experiments" / "pi05_template.yaml"
    contents = template_path.read_text().replace("type: pi0\n", "type: unknown_model\n", 1)
    path = _write_config(tmp_path / "unknown.yaml", contents)

    with pytest.raises(config_loader.ConfigError, match="Unknown model.type"):
        config_loader.load_yaml_config(path)


def test_standalone_config_rejects_missing_component_field(tmp_path: pathlib.Path):
    template_path = pathlib.Path(__file__).parents[3] / "configs" / "experiments" / "pi05_template.yaml"
    contents = yaml.safe_load(template_path.read_text())
    del contents["model"]["action_dim"]
    path = _write_config(tmp_path / "missing-model-field.yaml", yaml.safe_dump(contents))

    with pytest.raises(config_loader.ConfigError, match="Missing required fields at model.*action_dim"):
        config_loader.load_yaml_config(path)


def test_standalone_fake_data_config_builds_nested_base_config(tmp_path: pathlib.Path):
    template_path = pathlib.Path(__file__).parents[3] / "configs" / "experiments" / "pi05_template.yaml"
    contents = yaml.safe_load(template_path.read_text())
    contents["data"] = {
        "type": "fake",
        "repo_id": "fake",
        "assets": {"assets_dir": None, "asset_id": None},
        "base_config": {"prompt_from_task": True, "action_sequence_keys": ["actions"]},
    }
    path = _write_config(tmp_path / "fake.yaml", yaml.safe_dump(contents))

    resolved = config_loader.load_yaml_config(path)

    assert resolved.config.data.base_config.prompt_from_task is True
    assert resolved.config.data.base_config.action_sequence_keys == ("actions",)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ("unknown_field: true", "Unknown fields"),
        ("batch_size: many", "Expected an integer"),
        ("exp_name: committed-run", "runtime-only"),
        ("data:\n    rlds_data_dir: /mnt/data", "runtime-only"),
    ],
)
def test_yaml_config_rejects_invalid_overrides(tmp_path: pathlib.Path, overrides: str, message: str):
    path = _write_config(
        tmp_path / "experiment.yaml",
        f"""
schema_version: 1
name: invalid
base: pi05_full_droid_finetune
overrides:
  {overrides}
""",
    )

    with pytest.raises(config_loader.ConfigError, match=message):
        config_loader.load_yaml_config(path)


def test_yaml_config_rejects_duplicate_keys(tmp_path: pathlib.Path):
    path = _write_config(
        tmp_path / "experiment.yaml",
        """
schema_version: 1
name: duplicate
name: duplicate-again
base: debug
""",
    )

    with pytest.raises(config_loader.ConfigError, match="Duplicate YAML key"):
        config_loader.load_yaml_config(path)


def test_yaml_config_rejects_unsafe_tags(tmp_path: pathlib.Path):
    path = _write_config(
        tmp_path / "experiment.yaml",
        """
schema_version: 1
name: unsafe
base: debug
overrides: !!python/object/apply:os.system [[echo, unsafe]]
""",
    )

    with pytest.raises(config_loader.ConfigError, match="Invalid YAML"):
        config_loader.load_yaml_config(path)


def test_snapshot_is_safe_yaml(tmp_path: pathlib.Path):
    resolved = config_loader.resolve_config("debug")
    output_path = tmp_path / "metadata" / "train_config.yaml"

    config_loader.write_snapshot(output_path, resolved.snapshot())
    snapshot = yaml.safe_load(output_path.read_text())

    assert snapshot["base"] == "debug"
    assert snapshot["resolved_config"]["name"] == "debug"
    assert "commit" in snapshot["git"]


def test_standalone_snapshot_contains_manifest(tmp_path: pathlib.Path):
    config_path = pathlib.Path(__file__).parents[3] / "configs" / "experiments" / "pi05_template.yaml"
    resolved = config_loader.load_yaml_config(config_path)
    output_path = tmp_path / "train_config.yaml"

    config_loader.write_snapshot(output_path, resolved.snapshot())
    snapshot = yaml.safe_load(output_path.read_text())

    assert snapshot["mode"] == "standalone"
    assert snapshot["manifest"]["model"]["type"] == "pi0"
    assert "base" not in snapshot


def test_all_versioned_experiment_configs_are_valid():
    config_dir = pathlib.Path(__file__).parents[3] / "configs" / "experiments"
    configs = [config_loader.load_yaml_config(path) for path in sorted(config_dir.rglob("*.yaml"))]
    names = [resolved.config.name for resolved in configs]

    assert configs
    assert len(names) == len(set(names))
