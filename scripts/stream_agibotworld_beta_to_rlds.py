#!/usr/bin/env python3
"""Sequentially stream AgiBotWorld Beta from BOS, convert on PFS, and publish RLDS.

Real single-episode smoke test:
  uv run --group rlds scripts/stream_agibotworld_beta_to_rlds.py run \
    --observation-archive /mnt/bos/dataset/agibotworld-beta/observations/389/653277-674627.tar \
    --task-id 389 --episode-id 655660 \
    --dataset-name agibotworld_beta_stream_smoke --no-publish

Full production conversion:
  uv run --group rlds scripts/stream_agibotworld_beta_to_rlds.py run \
    --stream-readers 2 --spool-limit-gib 1024
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import threading
import time

from openpi.data import agibotworld_beta
from openpi.data import agibotworld_beta_stream
from openpi.training import observability


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/bos/dataset/agibotworld-beta"))
    parser.add_argument(
        "--pfs-work-root",
        type=Path,
        default=Path("/mnt/pfs/rhos-vla/chenyuan/agibotworld-beta-rlds-work"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/bos/dataset/RLDS"))
    parser.add_argument("--dataset-name", default="agibotworld_beta")
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

    run = subparsers.add_parser("run", help="Run or resume the complete streaming conversion pipeline.")
    _common_arguments(run)
    run.add_argument("--task-id", type=int, action="append", default=[], dest="task_ids")
    run.add_argument("--episode-id", type=int, action="append", default=[], dest="episode_ids")
    run.add_argument(
        "--observation-archive",
        type=Path,
        action="append",
        default=[],
        dest="observation_archives",
        help="Read only this observation tar. May be repeated.",
    )
    run.add_argument("--max-episodes", type=int)
    run.add_argument("--episodes-per-shard", type=int, default=8)
    run.add_argument("--stream-readers", type=int, default=2)
    run.add_argument("--convert-workers", type=int, default=0, help="0 uses all physical CPU cores or GPU policy.")
    run.add_argument("--spool-limit-gib", type=int, default=1024)
    run.add_argument(
        "--decoder",
        choices=[decoder.value for decoder in agibotworld_beta.Decoder],
        default=agibotworld_beta.Decoder.AUTO.value,
    )
    run.add_argument("--decoder-threads", type=int, default=1)
    run.add_argument("--gpu-workers-per-device", type=int, default=4)
    run.add_argument("--image-height", type=int, default=224)
    run.add_argument("--image-width", type=int, default=224)
    run.add_argument("--jpeg-quality", type=int, default=90)
    run.add_argument("--alignment-tolerance", type=int, default=2)
    run.add_argument("--allow-incomplete", action="store_true")
    run.add_argument("--no-publish", action="store_false", dest="publish")
    run.set_defaults(publish=True)

    status = subparsers.add_parser("status", help="Print durable pipeline state without starting work.")
    _common_arguments(status)

    validate = subparsers.add_parser("validate", help="Read back an existing PFS or BOS TFDS directory.")
    _common_arguments(validate)
    validate.add_argument("--location", choices=("pfs", "bos"), default="bos")
    validate.add_argument("--expected-episodes", type=int)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> agibotworld_beta_stream.StreamConfig:
    excluded = {"decoder", "command", "location", "expected_episodes"}
    values = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(agibotworld_beta_stream.StreamConfig)
        if field.name not in excluded and hasattr(args, field.name)
    }
    if hasattr(args, "decoder"):
        values["decoder"] = agibotworld_beta.Decoder(args.decoder)
    for key in ("task_ids", "episode_ids", "observation_archives"):
        if key in values:
            values[key] = tuple(values[key])
    return agibotworld_beta_stream.StreamConfig(**values)


def _conversion_lineage(config: agibotworld_beta_stream.StreamConfig, explicit_id: str | None) -> dict:
    identity = {
        "schema_version": 1,
        "dataset_name": config.dataset_name,
        "version": config.version,
        "input_root": str(config.input_root.resolve()),
        "task_ids": config.task_ids,
        "episode_ids": config.episode_ids,
        "observation_archives": [str(path.resolve()) for path in config.observation_archives],
        "image_height": config.image_height,
        "image_width": config.image_width,
        "jpeg_quality": config.jpeg_quality,
        "alignment_tolerance": config.alignment_tolerance,
        "allow_incomplete": config.allow_incomplete,
    }
    return {**identity, "lineage_id": explicit_id or observability.stable_digest(identity)[:16]}


def _status_metrics(config: agibotworld_beta_stream.StreamConfig) -> dict[str, float | int]:
    current = agibotworld_beta_stream.status(config)
    metrics: dict[str, float | int] = {
        "conversion/archive_bytes_total": int(current.get("archive_bytes_total", 0)),
        "conversion/archive_bytes_processed": int(current.get("archive_bytes_processed", 0)),
        "conversion/spool_bytes": int(current.get("spool_bytes", 0)),
    }
    for group in ("archives", "episodes", "fragments", "copies", "uploads"):
        for status_name, count in current.get(group, {}).items():
            metrics[f"conversion/{group}/{status_name}"] = int(count)
    return metrics


def _report_status(
    observer: observability.RunObserver,
    config: agibotworld_beta_stream.StreamConfig,
    stop: threading.Event,
    interval_seconds: int,
) -> None:
    started = time.monotonic()
    previous_bytes = 0
    while not stop.is_set():
        try:
            metrics = _status_metrics(config)
            processed = int(metrics["conversion/archive_bytes_processed"])
            elapsed = max(time.monotonic() - started, 1e-6)
            metrics["conversion/read_bytes_per_second"] = max(0, processed - previous_bytes) / interval_seconds
            metrics["conversion/average_read_bytes_per_second"] = processed / elapsed
            previous_bytes = processed
            observer.mark_progress(processed)
            # Keep processed bytes as the durable progress value, but let W&B
            # advance its own monotonically increasing report step. Archive
            # offsets only move at tar-member boundaries, so using `processed`
            # as W&B's global step produces repeated/out-of-order step warnings
            # while a large member is being streamed or spool backpressure pauses
            # scanning.
            observer.log_metrics(metrics)
        except Exception as exc:
            logging.warning("Could not collect conversion status: %s", exc)
        stop.wait(interval_seconds)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = _config(args)
    if args.command == "status":
        print(json.dumps(agibotworld_beta_stream.status(config), indent=2, sort_keys=True))
        return
    if args.command == "validate":
        path = config.local_version_dir if args.location == "pfs" else config.bos_version_dir
        result = agibotworld_beta_stream.validate_tfds_directory(path, expected_episodes=args.expected_episodes)
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
        manifest={"stream_config": dataclasses.asdict(config)},
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
        name="conversion-status-reporter",
        daemon=True,
    )
    reporter.start()
    try:
        manifest = agibotworld_beta_stream.run_pipeline(config, lineage_id=observer.lineage_id)
    except BaseException as exc:
        observer.alert("conversion_failed", f"{type(exc).__name__}: {exc}", deduplicate_seconds=0)
        observer.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_stop.set()
        reporter.join(timeout=args.observability_interval_seconds + 5)
    destination = config.bos_version_dir if config.publish else config.local_version_dir
    metadata = {
        "uri": str(destination),
        "manifest": manifest,
        "manifest_sha256": observability.file_digest(destination / "conversion_manifest.json"),
    }
    observer.log_artifact_metadata("dataset-manifest", metadata, artifact_type="dataset-reference")
    observer.finish(status="completed")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
