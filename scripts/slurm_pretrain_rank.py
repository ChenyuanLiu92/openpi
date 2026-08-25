"""Translate Slurm task environment into one OpenPI JAX rank."""

from __future__ import annotations

import json
import os
import pathlib
import sys


def rank_command(environ: dict[str, str]) -> list[str]:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    gpu_count = int(environ["OPENPI_GPUS_PER_NODE"])
    command = [
        sys.executable,
        str(repo_root / "scripts" / "launch_pretrain.py"),
        "rank",
        environ["OPENPI_CONFIG"],
        "--coordinator-address",
        environ["OPENPI_COORDINATOR_ADDRESS"],
        "--num-processes",
        environ["SLURM_NTASKS"],
        "--process-id",
        environ["SLURM_PROCID"],
        "--local-device-ids",
        ",".join(str(index) for index in range(gpu_count)),
    ]
    if environ.get("SLURM_PROCID") == "0":
        command.extend(["--coordinator-bind-address", f"[::]:{environ['OPENPI_COORDINATOR_PORT']}"])
    overrides = json.loads(environ.get("OPENPI_OVERRIDES_JSON", "[]"))
    if overrides:
        command.extend(["--", *overrides])
    return command


if __name__ == "__main__":
    os.execv(sys.executable, rank_command(dict(os.environ)))
