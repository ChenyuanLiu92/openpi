"""Durable, low-overhead observability for data and training jobs.

The local JSONL files are the source of truth. W&B and webhooks are best-effort
sinks and must never make a healthy job fail.
"""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
from typing import Any, Literal
import urllib.request
import uuid

import numpy as np
import wandb

LOGGER = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def stable_digest(value: Any) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_run_id(path: Path) -> str:
    """Return a durable run ID, atomically creating it when absent."""
    if path.is_file() and (existing := path.read_text().strip()):
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:16]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(run_id + "\n")
    os.replace(temporary, path)
    return run_id


@dataclasses.dataclass(frozen=True)
class Options:
    project: str
    experiment: str
    job_type: Literal["data_conversion", "normalization", "training"]
    enabled: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_entity: str | None = None
    local_root: Path | None = None
    tags: tuple[str, ...] = ()
    system_interval_seconds: int = 10
    heartbeat_interval_seconds: int = 60
    stall_timeout_seconds: int = 600
    emergency_checkpoint_timeout_seconds: int = 120
    webhook_url_env: str = "OPENPI_TRAIN_ALERT_WEBHOOK_URL"
    min_free_space_gib: int = 500
    raw_retention_days: int = 90

    def validate(self) -> None:
        if not self.project or not self.experiment:
            raise ValueError("Observability project and experiment must be non-empty")
        for name, value in {
            "system_interval_seconds": self.system_interval_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "emergency_checkpoint_timeout_seconds": self.emergency_checkpoint_timeout_seconds,
            "raw_retention_days": self.raw_retention_days,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_free_space_gib < 0:
            raise ValueError("min_free_space_gib must be non-negative")


def options_from_pretrain_config(
    config: Any, *, job_type: Literal["normalization", "training"], experiment: str | None = None
) -> Options:
    return Options(
        project=config.project_name,
        experiment=experiment or config.exp_name,
        job_type=job_type,
        enabled=config.wandb_enabled,
        wandb_mode=config.wandb_mode,
        wandb_entity=config.wandb_entity,
        local_root=config.observability_root,
        tags=config.wandb_tags,
        system_interval_seconds=config.system_interval_seconds,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        stall_timeout_seconds=config.stall_timeout_seconds,
        emergency_checkpoint_timeout_seconds=config.emergency_checkpoint_timeout_seconds,
        webhook_url_env=config.webhook_url_env,
        min_free_space_gib=config.min_free_space_gib,
        raw_retention_days=config.raw_retention_days,
    )


class _JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()


class _RotatingJsonlWriter:
    def __init__(self, directory: Path, stem: str, *, max_bytes: int = 512 * 2**20):
        self.directory = directory
        self.stem = stem
        self.max_bytes = max_bytes
        self._day = ""
        self._sequence = -1
        self._writer: _JsonlWriter | None = None
        self._lock = threading.Lock()

    def write(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            day = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
            if self._writer is None or day != self._day or self._writer.path.stat().st_size >= self.max_bytes:
                if self._writer is not None:
                    self._writer.close()
                self._sequence = self._sequence + 1 if day == self._day else 0
                self._day = day
                self._writer = _JsonlWriter(self.directory / f"{self.stem}.{day}.{self._sequence:03d}.jsonl")
            self._writer.write(payload)

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.close()


class RunObserver:
    """Owns durable logs and optional W&B/webhook delivery for one process."""

    def __init__(
        self,
        options: Options,
        *,
        manifest: Mapping[str, Any],
        lineage: Mapping[str, Any],
        process_index: int = 0,
        process_count: int = 1,
        run_id_path: Path | None = None,
        resume: bool = False,
    ):
        options.validate()
        self.options = options
        self.process_index = process_index
        self.process_count = process_count
        self.is_primary = process_index == 0
        self.lineage_id = str(lineage.get("lineage_id") or stable_digest(lineage)[:16])
        self.run_id = self._resolve_run_id(run_id_path, resume=resume)
        local_root = options.local_root or Path.cwd() / "observability"
        self.run_dir = local_root / options.project / options.experiment / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = _JsonlWriter(self.run_dir / "metrics.jsonl") if self.is_primary else None
        events_path = (
            self.run_dir / "events.jsonl"
            if self.is_primary
            else self.run_dir / "events" / f"process-{process_index:05d}.jsonl"
        )
        self._events = _JsonlWriter(events_path)
        self._alerts = _JsonlWriter(self.run_dir / "alerts.jsonl") if self.is_primary else None
        self._system = _RotatingJsonlWriter(self.run_dir / "system", f"process-{process_index:05d}")
        self._shutdown = threading.Event()
        self._stop_requested = threading.Event()
        self._phase_lock = threading.Lock()
        self._phase = "initializing"
        self._step = 0
        self._last_progress = time.monotonic()
        self._stall_reported = False
        self._forced_exit_deadline: float | None = None
        self._alert_times: dict[str, float] = {}
        self._wandb_run: Any | None = None
        self._previous_io = _read_proc_io()
        self._previous_io_time = time.monotonic()
        self._previous_process_cpu = _read_process_cpu_seconds()
        self._previous_host_cpu = _read_host_cpu_ticks()
        manifest_path = (
            self.run_dir / "run_manifest.json"
            if self.is_primary
            else self.run_dir / f"run_manifest.process-{process_index:05d}.json"
        )
        self._write_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "lineage_id": self.lineage_id,
                "project": options.project,
                "experiment": options.experiment,
                "job_type": options.job_type,
                "host": platform.node(),
                "process_index": process_index,
                "process_count": process_count,
                "created_at": utc_now(),
                "manifest": manifest,
            },
        )
        if self.is_primary:
            self._write_json_atomic(self.run_dir / "lineage.json", {**dict(lineage), "lineage_id": self.lineage_id})
        self._attach_file_logging()
        if self.is_primary:
            self._init_wandb(manifest, lineage, resume=resume)
        self.event("run_started", resume=resume)
        self._monitor = threading.Thread(target=self._monitor_loop, name="openpi-observability", daemon=True)
        self._monitor.start()

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    @property
    def wandb_active(self) -> bool:
        return self._wandb_run is not None

    def _resolve_run_id(self, path: Path | None, *, resume: bool) -> str:
        if path is not None and path.is_file():
            run_id = path.read_text().strip()
            if run_id:
                return run_id
        if resume:
            LOGGER.warning("Resume requested without a persisted observability run ID; creating a new run")
        return ensure_run_id(path) if path is not None and self.is_primary else uuid.uuid4().hex[:16]

    def _attach_file_logging(self) -> None:
        path = self.run_dir / "logs" / f"process-{self.process_index:05d}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )
        logging.getLogger().addHandler(handler)
        self._file_handler = handler

    def _init_wandb(self, manifest: Mapping[str, Any], lineage: Mapping[str, Any], *, resume: bool) -> None:
        if not self.options.enabled or self.options.wandb_mode == "disabled":
            return
        kwargs = {
            "id": self.run_id,
            "name": self.options.experiment,
            "project": self.options.project,
            "entity": self.options.wandb_entity,
            "job_type": self.options.job_type,
            "group": self.lineage_id,
            "tags": list(self.options.tags),
            "config": {"manifest": _json_value(manifest), "lineage": _json_value(lineage)},
            "dir": str(self.run_dir),
            "resume": "allow" if resume else None,
            "mode": self.options.wandb_mode,
            "settings": wandb.Settings(init_timeout=30),
        }
        try:
            self._wandb_run = wandb.init(**kwargs)
        except Exception as exc:  # W&B is deliberately a non-critical sink.
            LOGGER.exception("W&B initialization failed; continuing with durable PFS logs: %s", exc)
            if self.options.wandb_mode == "online":
                with contextlib.suppress(Exception):
                    self._wandb_run = wandb.init(**{**kwargs, "mode": "offline", "resume": None})

    def event(self, name: str, *, step: int | None = None, **details: Any) -> None:
        payload = self._base_payload(step)
        payload.update({"event": name, "details": details})
        self._events.write(payload)

    def set_phase(self, phase: str, *, step: int | None = None) -> None:
        with self._phase_lock:
            changed = phase != self._phase
            self._phase = phase
            if step is not None:
                self._step = step
        if changed:
            self.event("phase_changed", step=step, phase=phase)

    def mark_progress(self, step: int, **details: Any) -> None:
        with self._phase_lock:
            self._step = step
            self._last_progress = time.monotonic()
            self._stall_reported = False
        if details:
            self.event("progress", step=step, **details)

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        payload = self._base_payload(step)
        clean = {str(key): _json_value(value) for key, value in metrics.items()}
        payload["metrics"] = clean
        if self._metrics is not None:
            self._metrics.write(payload)
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(clean, step=step, commit=commit)
            except Exception as exc:
                LOGGER.warning("W&B metric delivery failed: %s", exc)

    def log_artifact_metadata(self, name: str, payload: Mapping[str, Any], *, artifact_type: str) -> None:
        if not self.is_primary:
            return
        destination = self.run_dir / "artifacts" / f"{name}.json"
        self._write_json_atomic(destination, payload)
        if self._wandb_run is None:
            return
        try:
            artifact = wandb.Artifact(
                name=f"{name}-{self.lineage_id}", type=artifact_type, metadata=_json_value(payload)
            )
            artifact.add_file(str(destination), name=destination.name)
            self._wandb_run.log_artifact(artifact)
        except Exception as exc:
            LOGGER.warning("W&B artifact metadata delivery failed: %s", exc)

    def log_images(self, key: str, images: list[np.ndarray], *, step: int) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log({key: [wandb.Image(image) for image in images]}, step=step)
        except Exception as exc:
            LOGGER.warning("W&B image delivery failed: %s", exc)

    def log_code(self, root: Path) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log_code(str(root))
        except Exception as exc:
            LOGGER.warning("W&B code snapshot delivery failed: %s", exc)

    def alert(self, kind: str, message: str, *, severity: str = "error", deduplicate_seconds: int = 900) -> bool:
        if not self.is_primary:
            return False
        now = time.monotonic()
        if now - self._alert_times.get(kind, -math.inf) < deduplicate_seconds:
            return False
        self._alert_times[kind] = now
        payload = {**self._base_payload(None), "kind": kind, "severity": severity, "message": message}
        assert self._alerts is not None
        self._alerts.write(payload)
        LOGGER.error("OpenPI alert [%s]: %s", kind, message)
        if self._wandb_run is not None:
            with contextlib.suppress(Exception):
                level = getattr(wandb.AlertLevel, severity.upper(), wandb.AlertLevel.ERROR)
                self._wandb_run.alert(title=f"OpenPI: {kind}", text=message, level=level)
        self._send_webhook(payload)
        return True

    def request_stop(self, reason: str) -> None:
        self.event("stop_requested", reason=reason)
        self._stop_requested.set()
        if self._forced_exit_deadline is None:
            self._forced_exit_deadline = time.monotonic() + self.options.emergency_checkpoint_timeout_seconds

    def finish(
        self, *, status: Literal["completed", "failed", "stopped"] = "completed", error: str | None = None
    ) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.event("run_finished", status=status, error=error)
        self._shutdown.set()
        self._monitor.join(timeout=max(1, min(self.options.system_interval_seconds + 1, 15)))
        if self._wandb_run is not None:
            with contextlib.suppress(Exception):
                self._wandb_run.finish(exit_code=0 if status == "completed" else 1)
        self._system.close()
        self._events.close()
        if self._metrics is not None:
            self._metrics.close()
        if self._alerts is not None:
            self._alerts.close()
        logging.getLogger().removeHandler(self._file_handler)
        self._file_handler.close()

    def _monitor_loop(self) -> None:
        next_heartbeat = time.monotonic()
        while not self._shutdown.wait(self.options.system_interval_seconds):
            try:
                self._system.write({**self._base_payload(None), "metrics": self._system_metrics()})
                now = time.monotonic()
                if now >= next_heartbeat:
                    self._write_json_atomic(
                        self.run_dir / f"heartbeat.process-{self.process_index:05d}.json",
                        self._base_payload(None),
                    )
                    next_heartbeat = now + self.options.heartbeat_interval_seconds
                with self._phase_lock:
                    stalled_for = now - self._last_progress
                    phase = self._phase
                    stalled = phase == "training" and stalled_for >= self.options.stall_timeout_seconds
                    already_reported = self._stall_reported
                    if stalled:
                        self._stall_reported = True
                if stalled and not already_reported:
                    self.alert("training_stalled", f"No optimizer step completed for {stalled_for:.0f}s")
                    self.request_stop("training_stalled")
                    self._forced_exit_deadline = now + self.options.emergency_checkpoint_timeout_seconds
                if self._forced_exit_deadline is not None and now >= self._forced_exit_deadline:
                    self.alert(
                        "emergency_checkpoint_timeout",
                        "Training did not return to a checkpoint-safe boundary before the emergency timeout",
                        deduplicate_seconds=0,
                    )
                    os._exit(70)
                free = shutil.disk_usage(self.run_dir).free
                threshold = self.options.min_free_space_gib * 2**30
                if threshold and free < threshold:
                    self.alert("low_pfs_space", f"Only {free / 2**30:.1f} GiB free at {self.run_dir}")
            except Exception:
                LOGGER.exception("Observability system sampler failed")

    def _system_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "system/load_1m": os.getloadavg()[0],
            "system/load_5m": os.getloadavg()[1],
            "system/load_15m": os.getloadavg()[2],
        }
        memory = _read_meminfo()
        metrics.update({f"system/memory_{key}_bytes": value for key, value in memory.items()})
        io_now = _read_proc_io()
        now = time.monotonic()
        elapsed = max(now - self._previous_io_time, 1e-6)
        for key, value in io_now.items():
            metrics[f"process/{key}_bytes"] = value
            metrics[f"process/{key}_bytes_per_second"] = max(0, value - self._previous_io.get(key, value)) / elapsed
        self._previous_io = io_now
        self._previous_io_time = now
        process_cpu = _read_process_cpu_seconds()
        metrics["process/cpu_percent"] = max(0.0, process_cpu - self._previous_process_cpu) / elapsed * 100
        self._previous_process_cpu = process_cpu
        host_cpu = _read_host_cpu_ticks()
        previous_total, previous_idle = self._previous_host_cpu
        total_delta = host_cpu[0] - previous_total
        idle_delta = host_cpu[1] - previous_idle
        if total_delta > 0:
            metrics["system/cpu_percent"] = max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100))
        self._previous_host_cpu = host_cpu
        with contextlib.suppress(OSError):
            metrics["process/rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
        usage = shutil.disk_usage(self.run_dir)
        metrics.update(
            {
                "filesystem/total_bytes": usage.total,
                "filesystem/used_bytes": usage.used,
                "filesystem/free_bytes": usage.free,
            }
        )
        metrics.update(_nvidia_smi_metrics())
        return metrics

    def _send_webhook(self, payload: Mapping[str, Any]) -> None:
        url = os.environ.get(self.options.webhook_url_env)
        if not url:
            return
        body = json.dumps(_json_value(payload)).encode()
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    if 200 <= response.status < 300:
                        return
            except Exception as exc:
                if attempt == 2:
                    LOGGER.warning("Webhook delivery failed after 3 attempts: %s", exc)
                else:
                    time.sleep(2**attempt)

    def _base_payload(self, step: int | None) -> dict[str, Any]:
        with self._phase_lock:
            phase = self._phase
            current_step = self._step
        return {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "lineage_id": self.lineage_id,
            "job_type": self.options.job_type,
            "process_index": self.process_index,
            "phase": phase,
            "step": current_step if step is None else step,
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)


def archive_system_logs(
    root: Path,
    *,
    raw_retention_days: int = 90,
    prune: bool = True,
    active_grace_seconds: int = 300,
    now: float | None = None,
) -> dict[str, int]:
    """Compress closed system JSONL files and prune old raw files only after verification."""
    if raw_retention_days <= 0:
        raise ValueError("raw_retention_days must be positive")
    if active_grace_seconds < 0:
        raise ValueError("active_grace_seconds must be non-negative")
    now = time.time() if now is None else now
    compressed = 0
    pruned = 0
    skipped = 0
    for path in root.rglob("system/process-*.jsonl"):
        if now - path.stat().st_mtime < active_grace_seconds:
            skipped += 1
            continue
        archive = path.with_suffix(path.suffix + ".zst")
        if not archive.exists() or archive.stat().st_mtime < path.stat().st_mtime:
            subprocess.run(["zstd", "-T0", "-q", "-f", "-k", str(path), "-o", str(archive)], check=True)
            subprocess.run(["zstd", "-q", "-t", str(archive)], check=True)
            compressed += 1
        age_days = (now - path.stat().st_mtime) / 86400
        if prune and age_days >= raw_retention_days:
            subprocess.run(["zstd", "-q", "-t", str(archive)], check=True)
            path.unlink()
            pruned += 1
    return {"compressed": compressed, "pruned": pruned, "skipped_active": skipped}


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                result[key.lower()] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return result


def _read_proc_io() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"read_bytes", "write_bytes"}:
                result[key] = int(value)
    except OSError:
        pass
    return result


def _read_process_cpu_seconds() -> float:
    try:
        fields = Path("/proc/self/stat").read_text().split()
        return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return 0.0


def _read_host_cpu_ticks() -> tuple[int, int]:
    try:
        values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        return sum(values), values[3] + (values[4] if len(values) > 4 else 0)
    except (OSError, ValueError, IndexError):
        return 0, 0


def _nvidia_smi_metrics() -> dict[str, float]:
    fields = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    metrics: dict[str, float] = {}
    names = ("utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c", "power_watts")
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6:
            continue
        index = values[0]
        for name, value in zip(names, values[1:], strict=True):
            with contextlib.suppress(ValueError):
                metrics[f"gpu/{index}/{name}"] = float(value)
    return metrics
