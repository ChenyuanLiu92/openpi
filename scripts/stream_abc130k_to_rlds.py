#!/usr/bin/env python3
"""Stream ABC-130K MCAP episodes from BOS through PFS into sharded RLDS.

Single-episode smoke test:
  uv run --group rlds scripts/stream_abc130k_to_rlds.py run \
    --episode-path /mnt/bos/dataset/abc-130k/data/train/fold_and_stack_the_t_shirts/episode_001005fe-c6ed-4e3c-b6ce-6beb4e8ce0cf \
    --dataset-name abc_130k_smoke --episodes-per-shard 1 --convert-workers 1 --no-publish --no-wandb

Full production conversion:
  uv run --group rlds scripts/stream_abc130k_to_rlds.py run \
    --stream-readers 4 --spool-limit-gib 512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import threading
import time

from openpi.data import abc130k
from openpi.training import observability


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/bos/dataset/abc-130k"))
    parser.add_argument(
        "--pfs-work-root",
        type=Path,
        default=Path("/mnt/pfs/rhos-vla/chenyuan/abc-130k-rlds-work"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/bos/dataset/RLDS"))
    parser.add_argument("--dataset-name", default="abc_130k")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, dest="wandb_enabled")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="openpi-data")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--lineage-id")
    parser.add_argument("--observability-local-root", type=Path)
    parser.add_argument("--observability-interval-seconds", type=int, default=10)
    parser.add_argument("--webhook-url-env", default="OPENPI_TRAIN_ALERT_WEBHOOK_URL")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run or resume the streaming conversion.")
    _common_arguments(run)
    run.add_argument("--split", choices=("train", "val"), action="append", dest="splits")
    run.add_argument("--task-name", action="append", default=[], dest="task_names")
    run.add_argument("--episode-id", action="append", default=[], dest="episode_ids")
    run.add_argument(
        "--episode-path",
        type=Path,
        action="append",
        default=[],
        dest="episode_paths",
        help="Episode directory or episode.mcap path; may be repeated.",
    )
    run.add_argument("--max-episodes", type=int)
    run.add_argument("--episodes-per-shard", type=int, default=8)
    run.add_argument(
        "--discovery-workers",
        type=int,
        default=32,
        help="Concurrent BOS metadata scanners used before conversion starts.",
    )
    run.add_argument("--stream-readers", type=int, default=4)
    run.add_argument("--convert-workers", type=int, default=0, help="0 uses all physical CPU cores.")
    run.add_argument("--spool-limit-gib", type=int, default=512)
    run.add_argument("--target-fps", type=int, default=30)
    run.add_argument("--image-height", type=int, default=224)
    run.add_argument("--image-width", type=int, default=224)
    run.add_argument("--jpeg-quality", type=int, default=90)
    run.add_argument("--decoder-threads", type=int, default=1)
    run.add_argument("--skip-bad-episodes", action="store_true")
    run.add_argument("--keep-spool", action="store_true")
    run.add_argument("--no-publish", action="store_false", dest="publish")
    run.set_defaults(publish=True)

    status = subparsers.add_parser("status", help="Print durable conversion state without starting work.")
    _common_arguments(status)

    validate = subparsers.add_parser("validate", help="Read back an existing PFS or BOS TFDS directory.")
    _common_arguments(validate)
    validate.add_argument("--location", choices=("pfs", "bos"), default="bos")
    validate.add_argument("--read-examples", type=int, default=1)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> abc130k.ConversionConfig:
    excluded = {"command", "location", "read_examples"}
    values = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(abc130k.ConversionConfig)
        if field.name not in excluded and hasattr(args, field.name)
    }
    for key in ("task_names", "episode_ids", "episode_paths"):
        if key in values:
            values[key] = tuple(values[key])
    if values.get("splits") is None:
        values["splits"] = ("train", "val")
    else:
        values["splits"] = tuple(dict.fromkeys(values["splits"]))
    return abc130k.ConversionConfig(**values)


def _conversion_lineage(config: abc130k.ConversionConfig, explicit_id: str | None) -> dict:
    identity = dict(config.semantic_identity)
    return {**identity, "lineage_id": explicit_id or observability.stable_digest(identity)[:16]}


def _status_metrics(config: abc130k.ConversionConfig) -> dict[str, float | int]:
    current = abc130k.status(config)
    metrics: dict[str, float | int] = {
        "conversion/source_bytes_total": int(current.get("source_bytes_total", 0)),
        "conversion/source_bytes_copied": int(current.get("source_bytes_copied", 0)),
        "conversion/spool_bytes": int(current.get("spool_bytes", 0)),
    }
    for group in ("episodes", "fragments", "uploads"):
        for status_name, count in current.get(group, {}).items():
            metrics[f"conversion/{group}/{status_name}"] = int(count)
    return metrics


def _report_status(
    observer: observability.RunObserver,
    config: abc130k.ConversionConfig,
    stop: threading.Event,
    interval_seconds: int,
) -> None:
    started = time.monotonic()
    previous_bytes = 0
    while not stop.is_set():
        try:
            metrics = _status_metrics(config)
            copied = int(metrics["conversion/source_bytes_copied"])
            elapsed = max(time.monotonic() - started, 1e-6)
            metrics["conversion/read_bytes_per_second"] = max(0, copied - previous_bytes) / interval_seconds
            metrics["conversion/average_read_bytes_per_second"] = copied / elapsed
            previous_bytes = copied
            observer.mark_progress(copied)
            # A copied-byte counter can remain unchanged between completed
            # episodes. Let W&B own its monotonic report step and retain bytes as
            # a metric/local progress value instead of forcing it into `_step`.
            observer.log_metrics(metrics)
        except Exception as exc:
            logging.warning("Could not collect ABC conversion status: %s", exc)
        stop.wait(interval_seconds)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = _config(args)
    if args.command == "status":
        print(json.dumps(abc130k.status(config), indent=2, sort_keys=True))
        return
    if args.command == "validate":
        path = config.local_version_dir if args.location == "pfs" else config.bos_version_dir
        result = abc130k.validate_tfds_directory(path, read_examples=args.read_examples)
        print(json.dumps({"path": str(path), **result}, indent=2, sort_keys=True))
        return
    if args.observability_interval_seconds <= 0:
        raise ValueError("--observability-interval-seconds must be positive")

    lineage = _conversion_lineage(config, args.lineage_id)
    observer = observability.RunObserver(
        observability.Options(
            project=args.wandb_project,
            experiment=f"{config.dataset_name}-{config.version}",
            job_type="data_conversion",
            enabled=args.wandb_enabled,
            wandb_mode=args.wandb_mode,
            wandb_entity=args.wandb_entity,
            local_root=args.observability_local_root or config.pfs_work_root / "observability",
            system_interval_seconds=args.observability_interval_seconds,
            heartbeat_interval_seconds=max(60, args.observability_interval_seconds),
            webhook_url_env=args.webhook_url_env,
        ),
        manifest={"conversion_config": dataclasses.asdict(config)},
        lineage=lineage,
        run_id_path=config.pfs_work_root / "state" / "observability_run_id.txt",
        resume=config.state_path.exists(),
    )
    observer.log_artifact_metadata("conversion-inputs", lineage, artifact_type="dataset-lineage")
    observer.set_phase("data_conversion")
    report_stop = threading.Event()
    reporter = threading.Thread(
        target=_report_status,
        args=(observer, config, report_stop, args.observability_interval_seconds),
        name="abc-conversion-status-reporter",
        daemon=True,
    )
    reporter.start()
    try:
        manifest = abc130k.run_pipeline(config, lineage_id=observer.lineage_id)
    except BaseException as exc:
        observer.alert("conversion_failed", f"{type(exc).__name__}: {exc}", deduplicate_seconds=0)
        observer.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_stop.set()
        reporter.join(timeout=args.observability_interval_seconds + 5)
    destination = config.bos_version_dir if config.publish else config.local_version_dir
    observer.log_artifact_metadata(
        "dataset-manifest",
        {
            "uri": str(destination),
            "manifest": manifest,
            "manifest_sha256": observability.file_digest(destination / "conversion_manifest.json"),
        },
        artifact_type="dataset-reference",
    )
    observer.finish(status="completed")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
