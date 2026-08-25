"""Submit one process per node for OpenPI pre-training under Slurm."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

from openpi.training import pretrain_config_loader

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SBATCH_SCRIPT = _REPO_ROOT / "scripts" / "slurm" / "pretrain.sbatch"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    separator = raw.index("--") if "--" in raw else len(raw)
    launcher_args, overrides = raw[:separator], raw[separator + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--cpus-per-task", type=int, default=64)
    parser.add_argument("--partition")
    parser.add_argument("--account")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--job-name")
    parser.add_argument("--dry-run", action="store_true")
    result = parser.parse_args(launcher_args)
    result.overrides = overrides
    return result


def build_command(args: argparse.Namespace) -> list[str]:
    resolved = pretrain_config_loader.parse_cli([str(args.config), *args.overrides])
    config = resolved.config
    if config.cluster.platform != "slurm":
        raise ValueError("Slurm submission requires cluster.platform: slurm (or --cluster.platform slurm)")
    if args.nodes <= 0 or args.gpus_per_node <= 0 or args.cpus_per_task <= 0:
        raise ValueError("nodes, gpus-per-node, and cpus-per-task must be positive")
    global_devices = args.nodes * args.gpus_per_node
    if config.batch_size % global_devices or global_devices % config.fsdp_devices:
        raise ValueError(
            f"Topology {args.nodes}x{args.gpus_per_node} is incompatible with batch_size={config.batch_size} "
            f"or fsdp_devices={config.fsdp_devices}"
        )
    command = [
        "sbatch",
        "--nodes",
        str(args.nodes),
        "--ntasks-per-node",
        "1",
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--time",
        args.time,
        "--job-name",
        args.job_name or config.exp_name,
        f"--signal=B:USR1@{config.cluster.preemption_grace_seconds}",
        "--requeue",
        "--chdir",
        str(_REPO_ROOT),
    ]
    if args.partition:
        command.extend(["--partition", args.partition])
    if args.account:
        command.extend(["--account", args.account])
    command.extend(
        [
            str(_SBATCH_SCRIPT),
            str(args.config.resolve()),
            str(args.gpus_per_node),
            str(config.cluster.max_restarts),
            json.dumps(args.overrides),
        ]
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
