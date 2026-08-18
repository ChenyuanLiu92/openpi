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


def test_all_versioned_experiment_configs_are_valid():
    config_dir = pathlib.Path(__file__).parents[3] / "configs" / "experiments"
    configs = [config_loader.load_yaml_config(path) for path in sorted(config_dir.glob("*.yaml"))]
    names = [resolved.config.name for resolved in configs]

    assert configs
    assert len(names) == len(set(names))
