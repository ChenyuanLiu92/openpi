import json
import os
import pathlib
import shutil

import pytest

from openpi.training import observability


def _options(root: pathlib.Path) -> observability.Options:
    return observability.Options(
        project="test-project",
        experiment="test-run",
        job_type="training",
        enabled=False,
        wandb_mode="disabled",
        local_root=root,
        system_interval_seconds=60,
        min_free_space_gib=0,
    )


def test_run_observer_persists_and_resumes_run(tmp_path: pathlib.Path):
    run_id_path = tmp_path / "checkpoint" / "wandb_id.txt"
    first = observability.RunObserver(
        _options(tmp_path / "logs"),
        manifest={"config": "one"},
        lineage={"dataset": "rlds-v1"},
        run_id_path=run_id_path,
    )
    first.log_metrics({"train/loss": 1.5}, step=3)
    first.alert("test_alert", "testing", deduplicate_seconds=0)
    run_id = first.run_id
    run_dir = first.run_dir
    first.finish()

    second = observability.RunObserver(
        _options(tmp_path / "logs"),
        manifest={"config": "one"},
        lineage={"dataset": "rlds-v1"},
        run_id_path=run_id_path,
        resume=True,
    )
    second.log_metrics({"train/loss": 1.0}, step=4)
    second.finish()

    assert second.run_id == run_id
    assert second.run_dir == run_dir
    metric_rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metric_rows] == [3, 4]
    assert json.loads((run_dir / "lineage.json").read_text())["lineage_id"] == first.lineage_id
    assert (run_dir / "logs" / "process-00000.log").is_file()


def test_non_primary_process_uses_separate_event_files(tmp_path: pathlib.Path):
    run_id_path = tmp_path / "run_id.txt"
    observability.ensure_run_id(run_id_path)
    observer = observability.RunObserver(
        _options(tmp_path / "logs"),
        manifest={},
        lineage={"lineage_id": "shared"},
        process_index=1,
        process_count=2,
        run_id_path=run_id_path,
        resume=True,
    )
    observer.event("worker_ready")
    run_dir = observer.run_dir
    observer.finish()

    assert (run_dir / "events" / "process-00001.jsonl").is_file()
    assert not (run_dir / "metrics.jsonl").exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd executable is required")
def test_archive_and_prune_system_logs(tmp_path: pathlib.Path):
    raw = tmp_path / "project" / "run" / "system" / "process-00000.20200101.000.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"metric":1}\n')
    old = 1_600_000_000
    os.utime(raw, (old, old))

    archived = observability.archive_system_logs(tmp_path, raw_retention_days=90, prune=False, now=old + 100 * 86400)
    assert archived == {"compressed": 1, "pruned": 0, "skipped_active": 0}
    assert raw.with_suffix(".jsonl.zst").is_file()

    pruned = observability.archive_system_logs(tmp_path, raw_retention_days=90, prune=True, now=old + 100 * 86400)
    assert pruned == {"compressed": 0, "pruned": 1, "skipped_active": 0}
    assert not raw.exists()


def test_archive_skips_active_system_log(tmp_path: pathlib.Path):
    raw = tmp_path / "run" / "system" / "process-00000.20200101.000.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("{}\n")

    result = observability.archive_system_logs(
        tmp_path,
        prune=False,
        active_grace_seconds=300,
        now=raw.stat().st_mtime + 10,
    )

    assert result == {"compressed": 0, "pruned": 0, "skipped_active": 1}
    assert not raw.with_suffix(".jsonl.zst").exists()
