"""Production preflight for the GPU portion of OpenPI pre-training.

The passive phase is safe on a busy host. Active phases are guarded by cgroup
working-set headroom, CPU load/pressure, free PFS space, and exclusive GPU use.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import contextlib
import dataclasses
import datetime
import json
import math
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BURN_SCRIPT = _REPO_ROOT / "scripts" / "gpu_burn.py"
_COLLECTIVE_SCRIPT = _REPO_ROOT / "scripts" / "check_gpu_collectives.py"
_LAUNCHER = _REPO_ROOT / "scripts" / "launch_pretrain.py"
_MODEL_SMOKE = _REPO_ROOT / "scripts" / "multinode_pretrain_smoke.py"
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "pretraining" / "pi05" / "template.yaml"
_CGROUP_MEMORY_CURRENT = pathlib.Path("/sys/fs/cgroup/memory.current")
_CGROUP_MEMORY_MAX = pathlib.Path("/sys/fs/cgroup/memory.max")
_CGROUP_MEMORY_STAT = pathlib.Path("/sys/fs/cgroup/memory.stat")
_CPU_PRESSURE = pathlib.Path("/sys/fs/cgroup/cpu.pressure")

_TOPOLOGIES: dict[str, tuple[str, ...]] = {
    "1x8": ("0,1,2,3,4,5,6,7",),
    "2x4": ("0,1,2,3", "4,5,6,7"),
    "4x2": ("0,1", "2,3", "4,5", "6,7"),
    "8x1": ("0", "1", "2", "3", "4", "5", "6", "7"),
}
_ACTIVE_PHASES = {"burn", "collective", "model"}


@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
    load_one: float
    cpu_psi_full_avg60: float
    memory_current_gib: float
    inactive_file_gib: float
    working_set_gib: float
    memory_limit_gib: float
    working_set_headroom_gib: float
    pfs_free_gib: float
    gpu_processes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class GateConfig:
    minimum_memory_headroom_gib: float
    maximum_load_one: float
    maximum_cpu_psi_full_avg60: float
    minimum_pfs_free_gib: float


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--phase",
        action="append",
        choices=("all", "passive", "burn", "collective", "model"),
        help="Repeat to select phases; defaults to all.",
    )
    parser.add_argument("--config", type=pathlib.Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--burn-seconds", type=float, default=600.0)
    parser.add_argument("--memory-gib-per-gpu", type=float, default=75.0)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--min-memory-headroom-gib", type=float, default=140.0)
    parser.add_argument("--max-load-one", type=float, default=45.0)
    parser.add_argument("--max-cpu-psi-full-avg60", type=float, default=1.0)
    parser.add_argument("--min-pfs-free-gib", type=float, default=100.0)
    parser.add_argument("--wait-for-resources-seconds", type=float, default=0.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--retain-success-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _selected_phases(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values or "all" in values:
        return ("passive", "burn", "collective", "model")
    order = ("passive", "burn", "collective", "model")
    selected = set(values)
    return tuple(phase for phase in order if phase in selected)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "expected-gpus": args.expected_gpus,
        "burn-seconds": args.burn_seconds,
        "memory-gib-per-gpu": args.memory_gib_per_gpu,
        "matrix-size": args.matrix_size,
        "min-memory-headroom-gib": args.min_memory_headroom_gib,
        "max-load-one": args.max_load_one,
        "max-cpu-psi-full-avg60": args.max_cpu_psi_full_avg60,
        "min-pfs-free-gib": args.min_pfs_free_gib,
        "command-timeout-seconds": args.command_timeout_seconds,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.wait_for_resources_seconds < 0:
        raise ValueError(f"GPU validation arguments must be positive (wait may be zero); invalid={invalid}")


def _read_number(path: pathlib.Path) -> int:
    return int(path.read_text().strip())


def _cgroup_working_set_percent() -> float:
    current = _read_number(_CGROUP_MEMORY_CURRENT)
    maximum_text = _CGROUP_MEMORY_MAX.read_text().strip()
    if maximum_text == "max":
        raise RuntimeError("GPU production validation requires a finite cgroup v2 memory limit")
    memory_stat = {
        key: int(value)
        for line in _CGROUP_MEMORY_STAT.read_text().splitlines()
        for key, value in [line.split(maxsplit=1)]
    }
    working_set = max(current - memory_stat.get("inactive_file", 0), 0)
    return working_set / int(maximum_text) * 100


def _parse_pressure(path: pathlib.Path, category: str, field: str) -> float:
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == category:
            values = dict(item.split("=", 1) for item in parts[1:])
            return float(values[field])
    raise RuntimeError(f"Missing {category}/{field} in {path}")


def _gpu_processes() -> tuple[str, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def resource_snapshot(work_dir: pathlib.Path) -> ResourceSnapshot:
    current = _read_number(_CGROUP_MEMORY_CURRENT)
    maximum_text = _CGROUP_MEMORY_MAX.read_text().strip()
    if maximum_text == "max":
        raise RuntimeError("GPU production validation requires a finite cgroup v2 memory limit")
    maximum = int(maximum_text)
    memory_stat = {
        key: int(value)
        for line in _CGROUP_MEMORY_STAT.read_text().splitlines()
        for key, value in [line.split(maxsplit=1)]
    }
    inactive_file = memory_stat.get("inactive_file", 0)
    working_set = max(current - inactive_file, 0)
    free_bytes = shutil.disk_usage(work_dir.parent).free
    gib = 2**30
    return ResourceSnapshot(
        load_one=os.getloadavg()[0],
        cpu_psi_full_avg60=_parse_pressure(_CPU_PRESSURE, "full", "avg60"),
        memory_current_gib=current / gib,
        inactive_file_gib=inactive_file / gib,
        working_set_gib=working_set / gib,
        memory_limit_gib=maximum / gib,
        working_set_headroom_gib=(maximum - working_set) / gib,
        pfs_free_gib=free_bytes / gib,
        gpu_processes=_gpu_processes(),
    )


def resource_gate_failures(snapshot: ResourceSnapshot, config: GateConfig) -> tuple[str, ...]:
    failures = []
    if snapshot.working_set_headroom_gib < config.minimum_memory_headroom_gib:
        failures.append(
            f"working-set memory headroom {snapshot.working_set_headroom_gib:.1f} GiB "
            f"< {config.minimum_memory_headroom_gib:.1f} GiB"
        )
    if snapshot.load_one > config.maximum_load_one:
        failures.append(f"one-minute load {snapshot.load_one:.1f} > {config.maximum_load_one:.1f}")
    if snapshot.cpu_psi_full_avg60 > config.maximum_cpu_psi_full_avg60:
        failures.append(
            f"CPU PSI full avg60 {snapshot.cpu_psi_full_avg60:.2f}% > {config.maximum_cpu_psi_full_avg60:.2f}%"
        )
    if snapshot.pfs_free_gib < config.minimum_pfs_free_gib:
        failures.append(f"PFS free space {snapshot.pfs_free_gib:.1f} GiB < {config.minimum_pfs_free_gib:.1f} GiB")
    if snapshot.gpu_processes:
        failures.append(f"GPUs are not exclusive: {list(snapshot.gpu_processes)}")
    return tuple(failures)


def wait_for_resources(
    work_dir: pathlib.Path,
    config: GateConfig,
    *,
    timeout_seconds: float,
) -> ResourceSnapshot:
    deadline = time.monotonic() + timeout_seconds
    while True:
        snapshot = resource_snapshot(work_dir)
        failures = resource_gate_failures(snapshot, config)
        if not failures:
            return snapshot
        if timeout_seconds == 0 or time.monotonic() >= deadline:
            raise RuntimeError("Active GPU validation resource gate failed: " + "; ".join(failures))
        print("Waiting for GPU validation resources: " + "; ".join(failures), flush=True)
        time.sleep(min(30.0, max(deadline - time.monotonic(), 0.0)))


def _xml_text(node: ET.Element, path: str, default: str = "N/A") -> str:
    value = node.findtext(path)
    return default if value is None else value.strip()


def _integer_or_none(value: str) -> int | None:
    try:
        return int(value.split()[0])
    except (ValueError, IndexError):
        return None


def parse_nvidia_snapshot(xml_text: str, *, expected_gpus: int) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    gpus = []
    failures = []
    warnings = []
    for index, node in enumerate(root.findall("gpu")):
        volatile_paths = (
            "ecc_errors/volatile/sram_uncorrectable_parity",
            "ecc_errors/volatile/sram_uncorrectable_secded",
            "ecc_errors/volatile/dram_uncorrectable",
        )
        corrected_paths = (
            "ecc_errors/volatile/sram_correctable",
            "ecc_errors/volatile/dram_correctable",
        )
        uncorrected = sum(_integer_or_none(_xml_text(node, path)) or 0 for path in volatile_paths)
        corrected = sum(_integer_or_none(_xml_text(node, path)) or 0 for path in corrected_paths)
        temperature = _integer_or_none(_xml_text(node, "temperature/gpu_temp"))
        record = {
            "index": index,
            "uuid": _xml_text(node, "uuid"),
            "product_name": _xml_text(node, "product_name"),
            "pci_bus_id": _xml_text(node, "pci/pci_bus_id"),
            "ecc_mode": _xml_text(node, "ecc_mode/current_ecc"),
            "volatile_corrected_ecc": corrected,
            "volatile_uncorrected_ecc": uncorrected,
            "unrepairable_memory": _xml_text(node, "ecc_errors/unrepairable_memory"),
            "row_remap_pending": _xml_text(node, "remapped_rows/remapped_row_pending"),
            "row_remap_failure": _xml_text(node, "remapped_rows/remapped_row_failure"),
            "retired_single_bit_pages": _integer_or_none(
                _xml_text(node, "retired_pages/multiple_single_bit_retirement/retired_count")
            ),
            "retired_double_bit_pages": _integer_or_none(
                _xml_text(node, "retired_pages/double_bit_retirement/retired_count")
            ),
            "pending_page_retirement": _xml_text(node, "retired_pages/pending_retirement"),
            "recovery_action": _xml_text(node, "gpu_recovery_action"),
            "temperature_c": temperature,
            "memory_used_mib": _integer_or_none(_xml_text(node, "fb_memory_usage/used")),
            "pcie_max_gen": _integer_or_none(_xml_text(node, "pci/pci_gpu_link_info/pcie_gen/max_link_gen")),
            "pcie_current_gen": _integer_or_none(
                _xml_text(node, "pci/pci_gpu_link_info/pcie_gen/current_link_gen")
            ),
            "pcie_max_width": _xml_text(node, "pci/pci_gpu_link_info/link_widths/max_link_width"),
            "pcie_current_width": _xml_text(node, "pci/pci_gpu_link_info/link_widths/current_link_width"),
            "hw_thermal_slowdown_us": _integer_or_none(
                _xml_text(
                    node,
                    "clocks_event_reasons_counters/clocks_event_reasons_counters_hw_therm_slowdown",
                )
            )
            or 0,
            "hw_power_brake_us": _integer_or_none(
                _xml_text(node, "clocks_event_reasons_counters/clocks_event_reasons_counters_hw_power_brake")
            )
            or 0,
            "sw_thermal_slowdown_us": _integer_or_none(
                _xml_text(node, "clocks_event_reasons_counters/clocks_event_reasons_counters_sw_therm_slowdown")
            )
            or 0,
        }
        gpus.append(record)
        prefix = f"GPU {index} ({record['uuid']})"
        if record["ecc_mode"] != "Enabled":
            failures.append(f"{prefix} ECC is not enabled")
        if uncorrected:
            failures.append(f"{prefix} has {uncorrected} volatile uncorrected ECC errors")
        if record["unrepairable_memory"] not in {"No", "N/A"}:
            failures.append(f"{prefix} reports unrepairable memory")
        if record["row_remap_pending"] == "Yes" or record["row_remap_failure"] == "Yes":
            failures.append(f"{prefix} reports pending/failed row remapping")
        if (record["retired_single_bit_pages"] or 0) > 0 or (record["retired_double_bit_pages"] or 0) > 0:
            failures.append(f"{prefix} reports retired framebuffer pages")
        if record["pending_page_retirement"] == "Yes":
            failures.append(f"{prefix} reports pending framebuffer page retirement")
        if record["recovery_action"] not in {"None", "N/A"}:
            failures.append(f"{prefix} requires recovery action {record['recovery_action']}")
        if temperature is not None and temperature >= 85:
            failures.append(f"{prefix} temperature is {temperature} C")
        if record["pcie_current_width"] != record["pcie_max_width"]:
            failures.append(
                f"{prefix} PCIe width is {record['pcie_current_width']}, expected {record['pcie_max_width']}"
            )
        if corrected:
            warnings.append(f"{prefix} starts with {corrected} volatile corrected ECC errors")
    if len(gpus) != expected_gpus:
        failures.append(f"nvidia-smi found {len(gpus)} GPUs, expected {expected_gpus}")
    return {
        "driver_version": _xml_text(root, "driver_version"),
        "cuda_version": _xml_text(root, "cuda_version"),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "failures": failures,
        "warnings": warnings,
    }


def capture_passive_snapshot(work_dir: pathlib.Path, *, expected_gpus: int, label: str) -> dict[str, Any]:
    xml_result = subprocess.run(
        ["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True, timeout=30
    )
    snapshot = parse_nvidia_snapshot(xml_result.stdout, expected_gpus=expected_gpus)
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True, timeout=30
    ).stdout
    xid_result = subprocess.run(
        ["dmesg", "--level=err,crit,alert,emerg"], check=False, capture_output=True, text=True, timeout=30
    )
    xid_lines = [line for line in xid_result.stdout.splitlines() if "NVRM: Xid" in line]
    if xid_lines:
        snapshot["failures"].append(f"Kernel log contains {len(xid_lines)} NVIDIA Xid events")
    elif xid_result.returncode != 0:
        snapshot["warnings"].append("Kernel Xid history is unavailable to this user")
    snapshot.update({"label": label, "captured_at": _utc_now(), "topology": topology, "xid_lines": xid_lines})
    (work_dir / f"gpu-{label}.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    (work_dir / f"gpu-topology-{label}.txt").write_text(topology)
    return snapshot


def compare_health(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, ...]:
    failures = list(after["failures"])
    initial = {gpu["uuid"]: gpu for gpu in before["gpus"]}
    for gpu in after["gpus"]:
        previous = initial.get(gpu["uuid"])
        if previous is None:
            failures.append(f"Unexpected GPU UUID after validation: {gpu['uuid']}")
            continue
        if gpu["volatile_corrected_ecc"] > previous["volatile_corrected_ecc"]:
            failures.append(f"GPU {gpu['uuid']} corrected ECC counter increased during validation")
        if gpu["volatile_uncorrected_ecc"] > previous["volatile_uncorrected_ecc"]:
            failures.append(f"GPU {gpu['uuid']} uncorrected ECC counter increased during validation")
        failures.extend(
            f"GPU {gpu['uuid']} {counter} increased during validation"
            for counter in ("hw_thermal_slowdown_us", "hw_power_brake_us", "sw_thermal_slowdown_us")
            if gpu[counter] > previous[counter]
        )
    return tuple(failures)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _telemetry_row() -> list[dict[str, Any]]:
    fields = (
        "index,uuid,temperature.gpu,power.draw,utilization.gpu,memory.used,"
        "clocks.current.sm,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current"
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    names = (
        "index",
        "uuid",
        "temperature_c",
        "power_w",
        "utilization_percent",
        "memory_used_mib",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "pcie_gen",
        "pcie_width",
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        row = dict(zip(names, values, strict=True))
        rows.append(row)
    return rows


def _sample_telemetry(path: pathlib.Path, stop: threading.Event, interval_seconds: float = 1.0) -> None:
    with path.open("a", encoding="utf-8") as stream:
        while not stop.is_set():
            try:
                payload = {"time": _utc_now(), "gpus": _telemetry_row()}
            except Exception as exc:  # telemetry failure is retained in-band; the workload remains authoritative
                payload = {"time": _utc_now(), "error": f"{type(exc).__name__}: {exc}"}
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            stop.wait(interval_seconds)


def validate_telemetry(
    paths: Sequence[pathlib.Path],
    *,
    expected_gpus: int,
    require_compute_saturation: bool,
    require_model_allocation: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    per_gpu: dict[str, dict[str, float]] = {}
    failures = []
    telemetry_errors = 0
    for path in paths:
        for line in path.read_text().splitlines():
            payload = json.loads(line)
            if "error" in payload:
                telemetry_errors += 1
                continue
            for row in payload["gpus"]:
                uuid = row["uuid"]
                metrics = per_gpu.setdefault(
                    uuid,
                    {
                        "index": float(row["index"]),
                        "max_temperature_c": 0.0,
                        "max_utilization_percent": 0.0,
                        "max_memory_used_mib": 0.0,
                        "minimum_active_pcie_gen": math.inf,
                        "minimum_pcie_width": math.inf,
                    },
                )
                temperature = float(row["temperature_c"])
                utilization = float(row["utilization_percent"])
                metrics["max_temperature_c"] = max(metrics["max_temperature_c"], temperature)
                metrics["max_utilization_percent"] = max(metrics["max_utilization_percent"], utilization)
                metrics["max_memory_used_mib"] = max(metrics["max_memory_used_mib"], float(row["memory_used_mib"]))
                metrics["minimum_pcie_width"] = min(metrics["minimum_pcie_width"], float(row["pcie_width"]))
                if utilization >= 50:
                    metrics["minimum_active_pcie_gen"] = min(
                        metrics["minimum_active_pcie_gen"], float(row["pcie_gen"])
                    )
    if len(per_gpu) != expected_gpus:
        failures.append(f"Telemetry observed {len(per_gpu)} GPU UUIDs, expected {expected_gpus}")
    for uuid, metrics in per_gpu.items():
        if metrics["max_temperature_c"] >= 85:
            failures.append(f"GPU {uuid} reached {metrics['max_temperature_c']:.0f} C")
        if metrics["minimum_pcie_width"] < 16:
            failures.append(f"GPU {uuid} PCIe width dropped below x16")
        if metrics["minimum_active_pcie_gen"] != math.inf and metrics["minimum_active_pcie_gen"] < 5:
            failures.append(f"GPU {uuid} PCIe generation stayed below Gen5 under load")
        if require_compute_saturation and metrics["max_utilization_percent"] < 80:
            failures.append(f"GPU {uuid} never reached 80% utilization during compute burn")
        if require_model_allocation and metrics["max_memory_used_mib"] < 1024:
            failures.append(f"GPU {uuid} never received a model-sized device allocation")
        if metrics["minimum_active_pcie_gen"] == math.inf:
            metrics["minimum_active_pcie_gen"] = None
    if telemetry_errors:
        failures.append(f"GPU telemetry sampling failed {telemetry_errors} times")
    summary = {
        "telemetry_files": len(paths),
        "telemetry_errors": telemetry_errors,
        "gpus": per_gpu,
    }
    return summary, tuple(failures)


def run_command(
    command: Sequence[str],
    *,
    log_path: pathlib.Path,
    telemetry_path: pathlib.Path,
    environment: dict[str, str],
    timeout_seconds: float,
    maximum_cgroup_working_set_percent: float = 95.0,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_telemetry, args=(telemetry_path, stop), name="gpu-telemetry", daemon=True
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        sampler.start()
        try:
            deadline = time.monotonic() + timeout_seconds
            while (returncode := process.poll()) is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"GPU validation command timed out after {timeout_seconds}s: {command}")
                working_set_percent = _cgroup_working_set_percent()
                if working_set_percent >= maximum_cgroup_working_set_percent:
                    raise MemoryError(
                        f"Cgroup working set reached {working_set_percent:.1f}% during GPU validation; "
                        f"limit={maximum_cgroup_working_set_percent:.1f}%"
                    )
                time.sleep(1)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            stop.set()
            sampler.join(timeout=5)
    if returncode:
        raise RuntimeError(f"GPU validation command failed with exit code {returncode}: {command}; log={log_path}")


def build_burn_commands(args: argparse.Namespace, work_dir: pathlib.Path) -> tuple[list[str], ...]:
    common = [
        sys.executable,
        str(_BURN_SCRIPT),
        "--expected-device-count",
        str(args.expected_gpus),
    ]
    return (
        [
            *common,
            "--workload",
            "memory",
            "--memory-gib-per-device",
            str(args.memory_gib_per_gpu),
            "--output",
            str(work_dir / "burn-memory.json"),
        ],
        [
            *common,
            "--workload",
            "compute",
            "--duration-seconds",
            str(args.burn_seconds),
            "--matrix-size",
            str(args.matrix_size),
            "--output",
            str(work_dir / "burn-compute.json"),
        ],
    )


def _write_probe_config(source: pathlib.Path, output: pathlib.Path, work_dir: pathlib.Path) -> None:
    contents = yaml.safe_load(source.read_text())
    contents["initialization"] = {"type": "random", "params_path": None}
    contents["paths"] = {
        "assets_base_dir": str(work_dir / "probe-assets"),
        "checkpoint_base_dir": str(work_dir / "probe-checkpoints"),
    }
    contents["logging"]["wandb_enabled"] = False
    contents["training"]["batch_size"] = 16
    contents["distributed"]["fsdp_devices"] = 1
    contents["distributed"]["initialize"] = False
    contents["distributed"]["diagnostics"].update(
        {
            "topology_check": True,
            "tensor_sizes_mib": [1, 16, 64, 256, 1024],
            "warmup_iterations": 2,
            "measure_iterations": 10,
        }
    )
    output.write_text(yaml.safe_dump(contents, sort_keys=False, allow_unicode=True))


def build_collective_command(
    config: pathlib.Path,
    *,
    topology: str,
    log_dir: pathlib.Path,
    baseline: pathlib.Path,
    minimum_memory_headroom_gib: float,
) -> list[str]:
    command = [sys.executable, str(_LAUNCHER), "local", str(config)]
    for group in _TOPOLOGIES[topology]:
        command.extend(["--device-group", group])
    command.extend(
        [
            "--probe-only",
            "--log-dir",
            str(log_dir),
            "--write-baseline",
            str(baseline),
            "--min-memory-headroom-gib",
            str(minimum_memory_headroom_gib),
            "--max-cgroup-memory-percent",
            "95",
        ]
    )
    return command


def aggregate_baselines(paths: Sequence[pathlib.Path], output: pathlib.Path, *, maximum_round_spread: float = 0.15) -> None:
    payloads = [json.loads(path.read_text()) for path in paths]
    keyed = [
        {(item["operation"], round(float(item["payload_mib"]), 6)): item for item in payload["results"]}
        for payload in payloads
    ]
    if any(set(items) != set(keyed[0]) for items in keyed[1:]):
        raise RuntimeError("Collective certification rounds produced different operation/payload keys")
    results = []
    for key in sorted(keyed[0]):
        rounds = [items[key] for items in keyed]
        bandwidths = [float(item["algorithm_gib_per_second"]) for item in rounds]
        median_bandwidth = statistics.median(bandwidths)
        spread = (max(bandwidths) - min(bandwidths)) / max(median_bandwidth, 1e-12)
        if spread > maximum_round_spread:
            raise RuntimeError(f"Collective {key} round spread {spread:.1%} exceeds {maximum_round_spread:.1%}")
        if max(float(item["rank_straggler_ratio"]) for item in rounds) > 1.5:
            raise RuntimeError(f"Collective {key} rank straggler ratio exceeds 1.5")
        results.append(
            {
                "operation": key[0],
                "payload_mib": float(key[1]),
                "median_seconds": statistics.median(float(item["median_seconds"]) for item in rounds),
                "p95_seconds": max(float(item["p95_seconds"]) for item in rounds),
                "algorithm_gib_per_second": median_bandwidth,
                "bus_gib_per_second": statistics.median(float(item["bus_gib_per_second"]) for item in rounds),
                "rank_straggler_ratio": max(float(item["rank_straggler_ratio"]) for item in rounds),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "process_count": payloads[0]["process_count"],
                "global_device_count": payloads[0]["global_device_count"],
                "topology": payloads[0].get("topology"),
                "certification": {"rounds": len(paths), "maximum_round_spread": maximum_round_spread},
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_model_commands(args: argparse.Namespace, work_dir: pathlib.Path) -> tuple[tuple[str, list[str], pathlib.Path], ...]:
    def command(name: str, groups: Iterable[str], *, fsdp_devices: int, dummy: bool) -> tuple[str, list[str], pathlib.Path]:
        output = work_dir / name
        result = [
            sys.executable,
            str(_MODEL_SMOKE),
            "--work-dir",
            str(output),
            "--fsdp-devices",
            str(fsdp_devices),
            "--min-memory-headroom-gib",
            str(args.min_memory_headroom_gib),
            "--max-cgroup-memory-percent",
            "95",
        ]
        for group in groups:
            result.extend(["--device-group", group])
        if dummy:
            result.append("--dummy-model")
        return name, result, output

    return (
        command("model-full-1x8-dp", _TOPOLOGIES["1x8"], fsdp_devices=1, dummy=False),
        command("model-full-2x4-dp", _TOPOLOGIES["2x4"], fsdp_devices=1, dummy=False),
        command("model-dummy-1x8-fsdp2", _TOPOLOGIES["1x8"], fsdp_devices=2, dummy=True),
        command("model-dummy-1x8-fsdp4", _TOPOLOGIES["1x8"], fsdp_devices=4, dummy=True),
    )


def _validation_environment(work_dir: pathlib.Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "TMPDIR": str(work_dir / "tmp"),
            "CUDA_CACHE_PATH": str(work_dir / "cuda-cache"),
            "JAX_COMPILATION_CACHE_DIR": str(work_dir / "jax-cache"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


def _write_summary(work_dir: pathlib.Path, report: dict[str, Any]) -> None:
    phases = "\n".join(f"- {name}: {status}" for name, status in report["phases"].items())
    (work_dir / "summary.md").write_text(
        f"# OpenPI GPU production validation\n\n"
        f"- Status: {report['status']}\n"
        f"- Started: {report['started_at']}\n"
        f"- Finished: {report.get('finished_at', 'incomplete')}\n\n"
        f"## Phases\n\n{phases}\n"
    )
    (work_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _cleanup_success_artifacts(paths: Iterable[pathlib.Path]) -> None:
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def _dry_run_payload(args: argparse.Namespace, phases: Sequence[str]) -> dict[str, Any]:
    placeholder = args.work_dir.resolve()
    probe_config = placeholder / "probe-config.yaml"
    collectives = [
        build_collective_command(
            probe_config,
            topology=topology,
            log_dir=placeholder / "collective-logs" / f"{topology}-round-1",
            baseline=placeholder / "baseline-rounds" / f"{topology}-round-1.json",
            minimum_memory_headroom_gib=args.min_memory_headroom_gib,
        )
        for topology in _TOPOLOGIES
    ]
    return {
        "phases": list(phases),
        "resource_gate": dataclasses.asdict(
            GateConfig(
                args.min_memory_headroom_gib,
                args.max_load_one,
                args.max_cpu_psi_full_avg60,
                args.min_pfs_free_gib,
            )
        ),
        "burn_commands": build_burn_commands(args, placeholder),
        "collective_round_one_commands": collectives,
        "model_commands": [command for _, command, _ in build_model_commands(args, placeholder)],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    phases = _selected_phases(args.phase)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args, phases), indent=2))
        return 0

    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("tmp", "cuda-cache", "jax-cache", "logs", "telemetry", "baselines"):
        (work_dir / directory).mkdir()
    environment = _validation_environment(work_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "phases": dict.fromkeys(phases, "pending"),
        "commands": [],
    }
    cleanup_paths: list[pathlib.Path] = [work_dir / "tmp", work_dir / "cuda-cache", work_dir / "jax-cache"]
    before = None
    try:
        before = capture_passive_snapshot(work_dir, expected_gpus=args.expected_gpus, label="before")
        if before["failures"]:
            raise RuntimeError("Passive GPU health check failed: " + "; ".join(before["failures"]))
        if "passive" in phases:
            report["phases"]["passive"] = "passed"

        if set(phases) & _ACTIVE_PHASES:
            gate = GateConfig(
                args.min_memory_headroom_gib,
                args.max_load_one,
                args.max_cpu_psi_full_avg60,
                args.min_pfs_free_gib,
            )
            snapshot = wait_for_resources(
                work_dir, gate, timeout_seconds=args.wait_for_resources_seconds
            )
            report["resource_snapshot"] = dataclasses.asdict(snapshot)

        if "burn" in phases:
            for index, command in enumerate(build_burn_commands(args, work_dir), start=1):
                report["commands"].append(command)
                run_command(
                    command,
                    log_path=work_dir / "logs" / f"burn-{index}.log",
                    telemetry_path=work_dir / "telemetry" / f"burn-{index}.jsonl",
                    environment=environment,
                    timeout_seconds=args.command_timeout_seconds,
                )
            report["phases"]["burn"] = "passed"

        if "collective" in phases:
            probe_config = work_dir / "probe-config.yaml"
            _write_probe_config(args.config.resolve(), probe_config, work_dir)
            cleanup_paths.extend([work_dir / "probe-assets", work_dir / "probe-checkpoints"])
            for topology in _TOPOLOGIES:
                rounds = []
                for round_index in (1, 2):
                    baseline = work_dir / "baseline-rounds" / f"{topology}-round-{round_index}.json"
                    baseline.parent.mkdir(exist_ok=True)
                    rounds.append(baseline)
                    command = build_collective_command(
                        probe_config,
                        topology=topology,
                        log_dir=work_dir / "collective-logs" / f"{topology}-round-{round_index}",
                        baseline=baseline,
                        minimum_memory_headroom_gib=args.min_memory_headroom_gib,
                    )
                    report["commands"].append(command)
                    run_command(
                        command,
                        log_path=work_dir / "logs" / f"collective-{topology}-round-{round_index}.log",
                        telemetry_path=work_dir / "telemetry" / f"collective-{topology}-round-{round_index}.jsonl",
                        environment=environment,
                        timeout_seconds=args.command_timeout_seconds,
                    )
                aggregate_baselines(rounds, work_dir / "baselines" / f"{topology}.json")
            for fsdp_devices in (2, 4):
                command = [
                    sys.executable,
                    str(_COLLECTIVE_SCRIPT),
                    "--visible-devices",
                    "0,1,2,3,4,5,6,7",
                    "--expected-device-count",
                    str(args.expected_gpus),
                    "--fsdp-devices",
                    str(fsdp_devices),
                    "--skip-bandwidth",
                ]
                report["commands"].append(command)
                run_command(
                    command,
                    log_path=work_dir / "logs" / f"collective-fsdp{fsdp_devices}.log",
                    telemetry_path=work_dir / "telemetry" / f"collective-fsdp{fsdp_devices}.jsonl",
                    environment=environment,
                    timeout_seconds=args.command_timeout_seconds,
                )
            report["phases"]["collective"] = "passed"

        if "model" in phases:
            for name, command, output in build_model_commands(args, work_dir):
                cleanup_paths.append(output)
                report["commands"].append(command)
                run_command(
                    command,
                    log_path=work_dir / "logs" / f"{name}.log",
                    telemetry_path=work_dir / "telemetry" / f"{name}.jsonl",
                    environment=environment,
                    timeout_seconds=args.command_timeout_seconds,
                )
            report["phases"]["model"] = "passed"

        telemetry_paths = sorted((work_dir / "telemetry").glob("*.jsonl"))
        if telemetry_paths:
            telemetry_summary, telemetry_failures = validate_telemetry(
                telemetry_paths,
                expected_gpus=args.expected_gpus,
                require_compute_saturation=False,
                require_model_allocation=False,
            )
            report["telemetry"] = telemetry_summary
            if telemetry_failures:
                raise RuntimeError("GPU telemetry validation failed: " + "; ".join(telemetry_failures))
        if report["phases"].get("burn") == "passed":
            _, burn_failures = validate_telemetry(
                sorted((work_dir / "telemetry").glob("burn-*.jsonl")),
                expected_gpus=args.expected_gpus,
                require_compute_saturation=True,
                require_model_allocation=False,
            )
            if burn_failures:
                raise RuntimeError("GPU burn telemetry validation failed: " + "; ".join(burn_failures))
        if report["phases"].get("model") == "passed":
            _, model_failures = validate_telemetry(
                sorted((work_dir / "telemetry").glob("model-*.jsonl")),
                expected_gpus=args.expected_gpus,
                require_compute_saturation=False,
                require_model_allocation=True,
            )
            if model_failures:
                raise RuntimeError("GPU model telemetry validation failed: " + "; ".join(model_failures))

        after = capture_passive_snapshot(work_dir, expected_gpus=args.expected_gpus, label="after")
        health_failures = compare_health(before, after)
        if health_failures:
            raise RuntimeError("Post-validation GPU health check failed: " + "; ".join(health_failures))
        report["status"] = "passed"
        report["finished_at"] = _utc_now()
        if not args.retain_success_artifacts:
            _cleanup_success_artifacts(cleanup_paths)
    except BaseException as exc:
        report["status"] = "failed"
        report["finished_at"] = _utc_now()
        report["error"] = f"{type(exc).__name__}: {exc}"
        for phase, status in report["phases"].items():
            if status == "pending":
                report["phases"][phase] = "not-run"
        _write_summary(work_dir, report)
        raise
    _write_summary(work_dir, report)
    print(f"GPU production validation passed: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
