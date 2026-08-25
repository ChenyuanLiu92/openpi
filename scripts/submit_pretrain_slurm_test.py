from __future__ import annotations

import pathlib

from . import slurm_pretrain_rank
from . import submit_pretrain_slurm


def _template() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1] / "configs" / "pretraining" / "pi05" / "template.yaml"


def test_slurm_command_contains_restart_and_topology_options():
    args = submit_pretrain_slurm._parse_args(  # noqa: SLF001
        [str(_template()), "--nodes", "2", "--gpus-per-node", "4", "--dry-run", "--", "--cluster.platform", "slurm"]
    )
    command = submit_pretrain_slurm.build_command(args)

    assert command[0] == "sbatch"
    assert command[command.index("--nodes") + 1] == "2"
    assert "--signal=B:USR1@120" in command
    assert command[-1] == '["--cluster.platform", "slurm"]'


def test_slurm_rank_command_uses_one_process_per_node():
    command = slurm_pretrain_rank.rank_command(
        {
            "OPENPI_GPUS_PER_NODE": "4",
            "OPENPI_CONFIG": "/shared/config.yaml",
            "OPENPI_COORDINATOR_ADDRESS": "node0:23456",
            "OPENPI_COORDINATOR_PORT": "23456",
            "OPENPI_OVERRIDES_JSON": "[]",
            "SLURM_NTASKS": "2",
            "SLURM_PROCID": "1",
        }
    )

    assert command[command.index("--num-processes") + 1] == "2"
    assert command[command.index("--process-id") + 1] == "1"
    assert command[command.index("--local-device-ids") + 1] == "0,1,2,3"
