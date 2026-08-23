from __future__ import annotations

import json
import pathlib
import sys
import time
import types

import pytest

from . import launch_pretrain


def _config(tmp_path: pathlib.Path, *, batch_size: int = 8, fsdp_devices: int = 8):
    return types.SimpleNamespace(
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        exp_name="test",
        checkpoint_dir=tmp_path / "checkpoints" / "name" / "test",
        distributed=types.SimpleNamespace(initialize=False),
    )


def test_rank_specs_reject_overlapping_and_missing_devices():
    with pytest.raises(ValueError, match="must not overlap"):
        launch_pretrain._rank_specs(["0,1", "1,2"], {0, 1, 2, 3})  # noqa: SLF001
    with pytest.raises(ValueError, match="not available"):
        launch_pretrain._rank_specs(["0,4"], {0, 1, 2, 3})  # noqa: SLF001


def test_rank_command_injects_runtime_topology(tmp_path: pathlib.Path):
    command = launch_pretrain._rank_command(  # noqa: SLF001
        config_path=tmp_path / "config.yaml",
        config=_config(tmp_path),
        spec=launch_pretrain.RankSpec(1, (4, 5, 6, 7)),
        num_processes=2,
        coordinator_address="127.0.0.1:12345",
        coordinator_bind_address="[::]:12345",
        probe_only=False,
        overrides=("--batch-size", "8"),
    )

    assert command[:3] == [sys.executable, str(launch_pretrain._PRETRAIN_SCRIPT), str(tmp_path / "config.yaml")]  # noqa: SLF001
    assert command[command.index("--distributed.process-id") + 1] == "1"
    local_ids = command.index("--distributed.local-device-ids")
    assert command[local_ids + 1 : local_ids + 5] == ["4", "5", "6", "7"]
    assert command[-2:] == ["--distributed.coordinator-bind-address", "[::]:12345"]


def test_local_dry_run_builds_two_ranks(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys):
    monkeypatch.setattr(launch_pretrain, "_available_gpu_ids", lambda: set(range(8)))
    monkeypatch.setattr(launch_pretrain, "_free_loopback_address", lambda: "127.0.0.1:12345")
    monkeypatch.setattr(launch_pretrain, "_resolved_config", lambda *_: _config(tmp_path))

    result = launch_pretrain.main(
        [
            "local",
            str(tmp_path / "config.yaml"),
            "--device-group",
            "0,1,2,3",
            "--device-group",
            "4,5,6,7",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["coordinator_address"] == "127.0.0.1:12345"
    assert len(payload["commands"]) == 2


def test_local_launcher_stops_sibling_after_rank_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(launch_pretrain, "_available_gpu_ids", lambda: {0, 1})
    monkeypatch.setattr(launch_pretrain, "_free_loopback_address", lambda: "127.0.0.1:12345")
    monkeypatch.setattr(launch_pretrain, "_resolved_config", lambda *_: _config(tmp_path, batch_size=2, fsdp_devices=2))

    def command(**kwargs):
        if kwargs["spec"].process_id == 0:
            return [sys.executable, "-c", "raise SystemExit(7)"]
        return [sys.executable, "-c", "import time; time.sleep(30)"]

    monkeypatch.setattr(launch_pretrain, "_rank_command", command)
    started = time.monotonic()
    result = launch_pretrain.main(
        [
            "local",
            str(tmp_path / "config.yaml"),
            "--device-group",
            "0",
            "--device-group",
            "1",
            "--log-dir",
            str(tmp_path / "logs"),
            "--shutdown-grace-seconds",
            "1",
        ]
    )

    assert result == 7
    assert time.monotonic() - started < 5


def test_launcher_rejects_distributed_passthrough_override():
    with pytest.raises(ValueError, match="launcher owns"):
        launch_pretrain._normalize_overrides(["--", "--distributed.process-id", "1"])  # noqa: SLF001


def test_launcher_validates_global_batch_and_fsdp(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="batch size"):
        launch_pretrain._validate_launcher_config(  # noqa: SLF001
            _config(tmp_path, batch_size=7), global_device_count=8
        )
    with pytest.raises(ValueError, match="fsdp_devices"):
        launch_pretrain._validate_launcher_config(  # noqa: SLF001
            _config(tmp_path, fsdp_devices=3), global_device_count=8
        )
