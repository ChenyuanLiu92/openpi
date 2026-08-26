from __future__ import annotations

import pytest

from . import multinode_pretrain_smoke


@pytest.mark.parametrize(
    "metrics",
    [
        "train/grad_norm=4.2, train/loss=1.7",
        "train/loss=1.7, train/grad_norm=4.2",
    ],
)
def test_finite_metric_check_is_order_independent(tmp_path, metrics: str):
    (tmp_path / "rank-00000.log").write_text(f"Step 1: {metrics}\n")

    multinode_pretrain_smoke._assert_finite_metrics(tmp_path)  # noqa: SLF001


def test_finite_metric_check_rejects_non_finite_values(tmp_path):
    (tmp_path / "rank-00000.log").write_text("Step 1: train/grad_norm=inf, train/loss=1.7\n")

    with pytest.raises(RuntimeError, match="No finite"):
        multinode_pretrain_smoke._assert_finite_metrics(tmp_path)  # noqa: SLF001


def test_rank_log_check_accepts_single_process_local_probe(tmp_path):
    (tmp_path / "rank-00000.log").write_text("Collective check passed: devices=8\n")

    multinode_pretrain_smoke._assert_rank_logs(tmp_path, probe=True, expected_ranks=1)  # noqa: SLF001


def test_rank_log_check_requires_global_probe_for_multiple_processes(tmp_path):
    (tmp_path / "rank-00000.log").write_text("Global collective check passed\n")
    (tmp_path / "rank-00001.log").write_text("local probe only\n")

    with pytest.raises(RuntimeError, match="Not all ranks"):
        multinode_pretrain_smoke._assert_rank_logs(tmp_path, probe=True, expected_ranks=2)  # noqa: SLF001
