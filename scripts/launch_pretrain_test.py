from __future__ import annotations

import json
import pathlib
import sys
import time
import types

import pytest

from . import launch_pretrain


def _config(tmp_path: pathlib.Path, *, batch_size: int = 8, fsdp_devices: int = 8):
    diagnostics = types.SimpleNamespace(
        tensor_sizes_mib=(1, 16), warmup_iterations=1, measure_iterations=2, topology_check=True
    )
    return types.SimpleNamespace(
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        exp_name="test",
        checkpoint_dir=tmp_path / "checkpoints" / "name" / "test",
        distributed=types.SimpleNamespace(initialize=False, diagnostics=diagnostics),
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


def test_probe_rank_command_forwards_bandwidth_baseline(tmp_path: pathlib.Path):
    command = launch_pretrain._rank_command(  # noqa: SLF001
        config_path=tmp_path / "config.yaml",
        config=_config(tmp_path),
        spec=launch_pretrain.RankSpec(0, (0, 1, 2, 3)),
        num_processes=2,
        coordinator_address="127.0.0.1:12345",
        coordinator_bind_address="[::]:12345",
        probe_only=True,
        overrides=(),
        write_baseline=tmp_path / "baseline.json",
        minimum_baseline_fraction=0.85,
        bandwidth_regression_policy="warn",
    )

    assert command[command.index("--write-baseline") + 1] == str(tmp_path / "baseline.json")
    assert command[command.index("--minimum-baseline-fraction") + 1] == "0.85"
    assert command[command.index("--bandwidth-regression-policy") + 1] == "warn"


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


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_local_launcher_stops_sibling_after_rank_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, failed_rank: int
):
    monkeypatch.setattr(launch_pretrain, "_available_gpu_ids", lambda: {0, 1})
    monkeypatch.setattr(launch_pretrain, "_free_loopback_address", lambda: "127.0.0.1:12345")
    monkeypatch.setattr(launch_pretrain, "_resolved_config", lambda *_: _config(tmp_path, batch_size=2, fsdp_devices=2))

    def command(**kwargs):
        if kwargs["spec"].process_id == failed_rank:
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


def test_cgroup_working_set_excludes_inactive_file(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    current = tmp_path / "memory.current"
    maximum = tmp_path / "memory.max"
    stat = tmp_path / "memory.stat"
    current.write_text(str(170 * 2**30))
    maximum.write_text(str(180 * 2**30))
    stat.write_text(f"anon {120 * 2**30}\ninactive_file {50 * 2**30}\n")
    monkeypatch.setattr(launch_pretrain, "_CGROUP_MEMORY_CURRENT", current)
    monkeypatch.setattr(launch_pretrain, "_CGROUP_MEMORY_MAX", maximum)
    monkeypatch.setattr(launch_pretrain, "_CGROUP_MEMORY_STAT", stat)

    memory = launch_pretrain._read_cgroup_memory()  # noqa: SLF001

    assert memory is not None
    assert memory.working_set == 120 * 2**30
    launch_pretrain._wait_for_memory(50, 0, accounting="working-set")  # noqa: SLF001
    with pytest.raises(RuntimeError, match="10.0 GiB"):
        launch_pretrain._wait_for_memory(50, 0, accounting="current")  # noqa: SLF001


def test_cgroup_working_set_falls_back_to_current(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys):
    current = tmp_path / "memory.current"
    maximum = tmp_path / "memory.max"
    current.write_text(str(100 * 2**30))
    maximum.write_text(str(180 * 2**30))
    monkeypatch.setattr(launch_pretrain, "_CGROUP_MEMORY_CURRENT", current)
    monkeypatch.setattr(launch_pretrain, "_CGROUP_MEMORY_MAX", maximum)
    monkeypatch.setattr(launch_pretrain, "_CGROUP_MEMORY_STAT", tmp_path / "missing.stat")

    launch_pretrain._wait_for_memory(50, 0, accounting="working-set")  # noqa: SLF001

    assert "falling back to raw memory.current" in capsys.readouterr().out
