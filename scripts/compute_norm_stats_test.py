import dataclasses
import pathlib

import pytest

from openpi.training import config as _config

from . import compute_norm_stats


def test_norm_stats_output_path_uses_asset_id(tmp_path: pathlib.Path):
    config = dataclasses.replace(_config.get_config("debug"), assets_base_dir=str(tmp_path))
    data_config = _config.DataConfig(repo_id="org/dataset", asset_id="robot_v2")

    output = compute_norm_stats._norm_stats_output_path(config, data_config)  # noqa: SLF001

    assert output == config.assets_dirs / "robot_v2"


def test_norm_stats_output_path_falls_back_to_repo_id(tmp_path: pathlib.Path):
    config = dataclasses.replace(_config.get_config("debug"), assets_base_dir=str(tmp_path))
    data_config = _config.DataConfig(repo_id="org/dataset")

    output = compute_norm_stats._norm_stats_output_path(config, data_config)  # noqa: SLF001

    assert output == config.assets_dirs / "org/dataset"


def test_norm_stats_output_path_requires_identifier(tmp_path: pathlib.Path):
    config = dataclasses.replace(_config.get_config("debug"), assets_base_dir=str(tmp_path))

    with pytest.raises(ValueError, match="asset_id or repo_id"):
        compute_norm_stats._norm_stats_output_path(config, _config.DataConfig())  # noqa: SLF001
