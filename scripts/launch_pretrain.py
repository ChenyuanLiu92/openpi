"""Launch OpenPI pre-training ranks locally or under an external node launcher.

Examples:
    uv run --group rlds scripts/launch_pretrain.py local config.yaml \
        --device-group 0,1,2,3 --device-group 4,5,6,7
    uv run --group rlds scripts/launch_pretrain.py rank config.yaml \
        --coordinator-address 10.0.0.1:12345 --num-processes 2 \
        --process-id 0 --local-device-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import contextlib
import dataclasses
import datetime
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PRETRAIN_SCRIPT = _REPO_ROOT / "scripts" / "pretrain.py"
_PROBE_SCRIPT = _REPO_ROOT / "scripts" / "check_gpu_collectives.py"
_CGROUP_MEMORY_CURRENT = pathlib.Path("/sys/fs/cgroup/memory.current")
_CGROUP_MEMORY_MAX = pathlib.Path("/sys/fs/cgroup/memory.max")


@dataclasses.dataclass(frozen=True)
class RankSpec:
    process_id: int
    local_device_ids: tuple[int, ...]


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if args is None else args)
    if "--" in raw_args:
        separator = raw_args.index("--")
        launcher_args = raw_args[:separator]
        overrides = raw_args[separator + 1 :]
    else:
        launcher_args = raw_args
        overrides = []
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    local = subparsers.add_parser("local", help="Spawn several JAX ranks on this machine.")
    local.add_argument("config", type=pathlib.Path)
    local.add_argument(
        "--device-group",
        action="append",
        required=True,
        help="Comma-separated physical GPU IDs for one rank; repeat once per rank.",
    )
    local.add_argument("--coordinator-address", help="Defaults to a free 127.0.0.1 port.")
    local.add_argument("--log-dir", type=pathlib.Path)
    local.add_argument("--probe-only", action="store_true")
    local.add_argument("--dry-run", action="store_true")
    local.add_argument("--shutdown-grace-seconds", type=float, default=15.0)
    local.add_argument("--min-memory-headroom-gib", type=float, default=0.0)
    local.add_argument("--wait-for-memory-seconds", type=float, default=0.0)
    local.add_argument(
        "--max-cgroup-memory-percent",
        type=float,
        default=0.0,
        help="Stop only the launched ranks if cgroup usage reaches this percentage; 0 disables the watchdog.",
    )

    rank = subparsers.add_parser("rank", help="Run one rank started by Slurm, SSH, Kubernetes, or another launcher.")
    rank.add_argument("config", type=pathlib.Path)
    rank.add_argument("--coordinator-address", required=True)
    rank.add_argument("--coordinator-bind-address")
    rank.add_argument("--num-processes", required=True, type=int)
    rank.add_argument("--process-id", required=True, type=int)
    rank.add_argument("--local-device-ids", required=True)
    rank.add_argument("--probe-only", action="store_true")
    rank.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(launcher_args)
    parsed.overrides = overrides
    return parsed


def _parse_device_group(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid device group {value!r}; expected comma-separated integers") from exc
    if not result:
        raise ValueError("Device groups must not be empty")
    if any(device_id < 0 for device_id in result):
        raise ValueError("Device IDs must be non-negative")
    if len(result) != len(set(result)):
        raise ValueError(f"Device group contains duplicate IDs: {value}")
    return result


def _rank_specs(device_groups: Sequence[str], available_devices: set[int]) -> tuple[RankSpec, ...]:
    groups = tuple(_parse_device_group(value) for value in device_groups)
    flattened = [device_id for group in groups for device_id in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Device groups must not overlap")
    missing = sorted(set(flattened) - available_devices)
    if missing:
        raise ValueError(f"Requested GPU IDs are not available: {missing}; available={sorted(available_devices)}")
    return tuple(RankSpec(index, group) for index, group in enumerate(groups))


def _available_gpu_ids() -> set[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi could not enumerate GPUs") from exc
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def _validate_address(value: str) -> None:
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid coordinator address {value!r}; expected host:port") from exc
    if not host or not 1 <= port <= 65535:
        raise ValueError(f"Invalid coordinator address {value!r}; expected host:port")


def _free_loopback_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def _normalize_overrides(overrides: Sequence[str]) -> tuple[str, ...]:
    result = tuple(overrides[1:] if overrides and overrides[0] == "--" else overrides)
    if any(argument.startswith("--distributed.") for argument in result):
        raise ValueError("The launcher owns all --distributed.* overrides; set topology with launcher arguments")
    return result


def _resolved_config(path: pathlib.Path, overrides: Sequence[str]):
    from openpi.training import pretrain_config_loader

    return pretrain_config_loader.parse_cli([str(path), *overrides]).config


def _validate_launcher_config(config, *, global_device_count: int) -> None:
    if config.distributed.initialize:
        raise ValueError("Launcher-managed configs must set distributed.initialize: false")
    if config.batch_size % global_device_count != 0:
        raise ValueError(
            f"Global batch size {config.batch_size} is not divisible by {global_device_count} launched devices"
        )
    if global_device_count % config.fsdp_devices != 0:
        raise ValueError(
            f"Launched device count {global_device_count} is not divisible by fsdp_devices={config.fsdp_devices}"
        )


def _rank_command(
    *,
    config_path: pathlib.Path,
    config,
    spec: RankSpec,
    num_processes: int,
    coordinator_address: str,
    coordinator_bind_address: str | None,
    probe_only: bool,
    overrides: Sequence[str],
) -> list[str]:
    local_ids = ",".join(str(device_id) for device_id in spec.local_device_ids)
    if probe_only:
        command = [
            sys.executable,
            str(_PROBE_SCRIPT),
            "--coordinator-address",
            coordinator_address,
            "--num-processes",
            str(num_processes),
            "--process-id",
            str(spec.process_id),
            "--local-device-ids",
            local_ids,
            "--expected-device-count",
            str(len(spec.local_device_ids)),
            "--fsdp-devices",
            str(config.fsdp_devices),
        ]
        if coordinator_bind_address is not None:
            command.extend(["--coordinator-bind-address", coordinator_bind_address])
        return command

    command = [sys.executable, str(_PRETRAIN_SCRIPT), str(config_path), *overrides]
    command.extend(
        [
            "--distributed.initialize",
            "--distributed.coordinator-address",
            coordinator_address,
            "--distributed.num-processes",
            str(num_processes),
            "--distributed.process-id",
            str(spec.process_id),
            "--distributed.local-device-ids",
            *(str(device_id) for device_id in spec.local_device_ids),
        ]
    )
    if coordinator_bind_address is not None:
        command.extend(["--distributed.coordinator-bind-address", coordinator_bind_address])
    return command


def _read_cgroup_memory() -> tuple[int, int] | None:
    try:
        current = int(_CGROUP_MEMORY_CURRENT.read_text().strip())
        maximum_text = _CGROUP_MEMORY_MAX.read_text().strip()
        if maximum_text == "max":
            return None
        return current, int(maximum_text)
    except (OSError, ValueError):
        return None


def _wait_for_memory(minimum_headroom_gib: float, timeout_seconds: float) -> None:
    if minimum_headroom_gib < 0 or timeout_seconds < 0:
        raise ValueError("Memory headroom and wait timeout must be non-negative")
    if minimum_headroom_gib == 0:
        return
    required = int(minimum_headroom_gib * 2**30)
    deadline = time.monotonic() + timeout_seconds
    while True:
        memory = _read_cgroup_memory()
        if memory is None:
            raise RuntimeError("Cannot enforce memory headroom because this process has no finite cgroup v2 limit")
        current, maximum = memory
        headroom = maximum - current
        if headroom >= required:
            return
        if timeout_seconds == 0 or time.monotonic() >= deadline:
            raise RuntimeError(
                f"Only {headroom / 2**30:.1f} GiB cgroup memory headroom is available; "
                f"{minimum_headroom_gib:.1f} GiB is required"
            )
        print(
            f"Waiting for cgroup memory: {headroom / 2**30:.1f} GiB available, {minimum_headroom_gib:.1f} GiB required",
            flush=True,
        )
        time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))


def _default_log_dir(config) -> pathlib.Path:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return config.checkpoint_dir.parent / ".launcher_logs" / f"{config.exp_name}-{timestamp}"


def _terminate_processes(processes: Sequence[subprocess.Popen[str]], grace_seconds: float) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while live and time.monotonic() < deadline:
        live = [process for process in live if process.poll() is None]
        if live:
            time.sleep(0.1)
    for process in live:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def _stream_rank_output(rank: int, stream, destination, lock: threading.Lock) -> None:
    for line in iter(stream.readline, ""):
        destination.write(line)
        destination.flush()
        with lock:
            print(f"[rank {rank}] {line}", end="", flush=True)
    stream.close()


def _run_local(args: argparse.Namespace) -> int:
    overrides = _normalize_overrides(args.overrides)
    config_path = args.config.expanduser().resolve()
    config = _resolved_config(config_path, overrides)
    specs = _rank_specs(args.device_group, _available_gpu_ids())
    global_device_count = sum(len(spec.local_device_ids) for spec in specs)
    _validate_launcher_config(config, global_device_count=global_device_count)
    if args.shutdown_grace_seconds < 0:
        raise ValueError("shutdown-grace-seconds must be non-negative")
    if args.max_cgroup_memory_percent != 0 and not 0 < args.max_cgroup_memory_percent < 100:
        raise ValueError("max-cgroup-memory-percent must be 0 or between 0 and 100")
    _wait_for_memory(args.min_memory_headroom_gib, args.wait_for_memory_seconds)

    coordinator_address = args.coordinator_address or _free_loopback_address()
    _validate_address(coordinator_address)
    port = coordinator_address.rsplit(":", 1)[1]
    coordinator_bind_address = f"[::]:{port}"
    commands = [
        _rank_command(
            config_path=config_path,
            config=config,
            spec=spec,
            num_processes=len(specs),
            coordinator_address=coordinator_address,
            coordinator_bind_address=coordinator_bind_address,
            probe_only=args.probe_only,
            overrides=overrides,
        )
        for spec in specs
    ]
    if args.dry_run:
        print(json.dumps({"coordinator_address": coordinator_address, "commands": commands}, indent=2))
        return 0

    log_dir = (args.log_dir or _default_log_dir(config)).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "launcher.json").write_text(
        json.dumps(
            {
                "coordinator_address": coordinator_address,
                "global_device_count": global_device_count,
                "ranks": [dataclasses.asdict(spec) for spec in specs],
                "commands": commands,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Launcher logs: {log_dir}", flush=True)

    processes: list[subprocess.Popen[str]] = []
    outputs = []
    threads: list[threading.Thread] = []
    output_lock = threading.Lock()
    stop_requested = False

    def _handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Launcher received {signal.Signals(signum).name}; stopping ranks", flush=True)

    previous_handlers = {signum: signal.signal(signum, _handle_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for spec, command in zip(specs, commands, strict=True):
            destination = (log_dir / f"rank-{spec.process_id:05d}.log").open("w")
            outputs.append(destination)
            process = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            processes.append(process)
            thread = threading.Thread(
                target=_stream_rank_output,
                args=(spec.process_id, process.stdout, destination, output_lock),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        failure_code = 0
        while any(process.poll() is None for process in processes):
            failed = next(
                (process.returncode for process in processes if process.returncode not in (None, 0)),
                None,
            )
            if failed is not None:
                failure_code = failed
                break
            if stop_requested:
                failure_code = 130
                break
            if args.max_cgroup_memory_percent:
                memory = _read_cgroup_memory()
                if memory is not None and memory[0] / memory[1] * 100 >= args.max_cgroup_memory_percent:
                    print(
                        f"Cgroup memory reached {memory[0] / memory[1] * 100:.1f}%; stopping launched ranks",
                        flush=True,
                    )
                    failure_code = 75
                    break
            time.sleep(0.2)

        if failure_code:
            _terminate_processes(processes, args.shutdown_grace_seconds)
        return_codes = [process.wait() for process in processes]
        if failure_code:
            return failure_code
        return next((code for code in return_codes if code != 0), 0)
    except BaseException:
        _terminate_processes(processes, args.shutdown_grace_seconds)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=5)
        for output in outputs:
            output.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _run_rank(args: argparse.Namespace) -> int:
    overrides = _normalize_overrides(args.overrides)
    config_path = args.config.expanduser().resolve()
    config = _resolved_config(config_path, overrides)
    if config.distributed.initialize:
        raise ValueError("Launcher-managed configs must set distributed.initialize: false")
    local_device_ids = _parse_device_group(args.local_device_ids)
    if args.num_processes <= 0 or not 0 <= args.process_id < args.num_processes:
        raise ValueError("process-id must be in [0, num-processes)")
    _validate_address(args.coordinator_address)
    if args.coordinator_bind_address is not None:
        _validate_address(args.coordinator_bind_address)
    command = _rank_command(
        config_path=config_path,
        config=config,
        spec=RankSpec(args.process_id, local_device_ids),
        num_processes=args.num_processes,
        coordinator_address=args.coordinator_address,
        coordinator_bind_address=args.coordinator_bind_address,
        probe_only=args.probe_only,
        overrides=overrides,
    )
    if args.dry_run:
        print(json.dumps({"command": command}, indent=2))
        return 0
    os.execvpe(command[0], command, os.environ.copy())
    raise AssertionError("os.execvpe returned unexpectedly")


def main(args: Sequence[str] | None = None) -> int:
    parsed = _parse_args(args)
    return _run_local(parsed) if parsed.mode == "local" else _run_rank(parsed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
