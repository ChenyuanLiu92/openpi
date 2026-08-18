"""Small GPU collective probes used before expensive model initialization."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import importlib.metadata
import logging
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class CollectiveProbeResult:
    """Result of one or more numerically checked local all-reduces."""

    device_count: int
    expected_sum: float
    elapsed_seconds: tuple[float, ...]


def log_cuda_cache_configuration(devices: Sequence[jax.Device] | None = None) -> None:
    """Log CUDA JIT-cache settings and warn about settings that make cold starts repeat."""
    devices = tuple(jax.local_devices() if devices is None else devices)
    cache_disabled = os.environ.get("CUDA_CACHE_DISABLE", "0") == "1"
    cache_path = pathlib.Path(os.environ.get("CUDA_CACHE_PATH", "~/.nv/ComputeCache")).expanduser()
    cache_size = os.environ.get("CUDA_CACHE_MAXSIZE", "<driver default>")
    logging.info(
        "CUDA driver cache: path=%s, max_size=%s, disabled=%s",
        cache_path,
        cache_size,
        cache_disabled,
    )

    if cache_disabled:
        logging.warning(
            "CUDA_CACHE_DISABLE=1 disables the driver JIT cache; every job may repeat slow PTX compilation. "
            "Unset it for normal training."
        )
    if not _path_is_writable(cache_path):
        logging.warning(
            "CUDA cache path %s is not writable (or cannot be created by this user); cold PTX compilation may repeat. "
            "Set CUDA_CACHE_PATH to a writable persistent directory before starting Python.",
            cache_path,
        )

    capabilities = sorted(
        {
            str(capability)
            for device in devices
            if (capability := getattr(device, "compute_capability", None)) is not None
        }
    )
    logging.info(
        "JAX %s with nvidia-nccl-cu12 %s; local GPU devices: %s",
        jax.__version__,
        _package_version("nvidia-nccl-cu12"),
        ", ".join(
            f"{device} ({device.device_kind}, compute capability {getattr(device, 'compute_capability', 'unknown')})"
            for device in devices
        ),
    )
    if any(capability.startswith("12.") for capability in capabilities):
        logging.warning(
            "Detected compute capability %s. On a cold cache, this CUDA/JAX stack may spend tens of seconds "
            "JIT-compiling collective kernels. XLA's 'rendezvous ... may be stuck' message after 10 seconds is a "
            "progress warning; wait for this checked probe to finish before diagnosing a deadlock.",
            ", ".join(capabilities),
        )
        if _version_tuple(_package_version("nvidia-nccl-cu12")) < (2, 28, 9):
            logging.warning(
                "NCCL older than 2.28.9 is not supported on this sm_120 setup: NCCL 2.26.2 AllGather was "
                "observed to crash with CUDA_ERROR_ILLEGAL_ADDRESS. Run `uv sync` to install the locked override."
            )


def run_local_fsdp_collective_probe(
    *,
    devices: Sequence[jax.Device] | None = None,
    repetitions: int = 1,
    require_multiple_devices: bool = False,
) -> CollectiveProbeResult:
    """Run and verify AllReduce, AllGather, and ReduceScatter across local devices.

    The first repetition includes compilation and is intentionally representative of a
    cold startup. Later repetitions show the steady-state launch time. In a multi-host
    job this warms each host's local CUDA/NCCL kernels; the normal training executable
    still performs the first cross-host rendezvous.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    devices = tuple(jax.local_devices() if devices is None else devices)
    if not devices:
        raise RuntimeError("No local JAX devices are available")
    if len(devices) == 1:
        if require_multiple_devices:
            raise RuntimeError(
                "The collective probe requires at least two visible JAX devices; check CUDA_VISIBLE_DEVICES"
            )
        logging.info("Skipping collective warmup because only one local JAX device is visible")
        return CollectiveProbeResult(device_count=1, expected_sum=0.0, elapsed_seconds=())

    device_count = len(devices)
    expected_sum = float(sum(range(device_count)))
    payload_rows = 1024
    payload_columns = 256

    def _fsdp_collectives(rank):
        payload = jnp.full((payload_rows, payload_columns), rank, dtype=jnp.float32)
        return (
            jax.lax.psum(rank, "local_devices"),
            jax.lax.all_gather(payload, "local_devices", axis=0, tiled=True),
            jax.lax.psum_scatter(payload, "local_devices", scatter_dimension=0, tiled=True),
        )

    collective = jax.pmap(_fsdp_collectives, axis_name="local_devices", devices=devices)
    inputs = np.arange(device_count, dtype=np.float32)
    elapsed_seconds: list[float] = []
    for repetition in range(repetitions):
        start = time.monotonic()
        all_reduce, all_gather, reduce_scatter = jax.device_get(collective(inputs))
        elapsed = time.monotonic() - start
        _validate_collective_results(
            np.asarray(all_reduce),
            np.asarray(all_gather),
            np.asarray(reduce_scatter),
            expected_sum=expected_sum,
            device_count=device_count,
            payload_rows=payload_rows,
            payload_columns=payload_columns,
        )
        elapsed_seconds.append(elapsed)
        logging.info(
            "Local FSDP collective probe %d/%d passed on %d devices in %.3f seconds "
            "(AllReduce=%s, AllGather and ReduceScatter verified)",
            repetition + 1,
            repetitions,
            device_count,
            elapsed,
            np.asarray(all_reduce).tolist(),
        )
        if elapsed >= 10:
            logging.info(
                "The checked collective completed successfully after %.1f seconds; any earlier XLA 10-second "
                "rendezvous warning was a cold-start warning, not a communication failure.",
                elapsed,
            )

    return CollectiveProbeResult(
        device_count=device_count, expected_sum=expected_sum, elapsed_seconds=tuple(elapsed_seconds)
    )


def warmup_local_collectives(*, enabled: bool, devices: Sequence[jax.Device] | None = None) -> None:
    """Warm local collectives when enabled, otherwise log the explicit opt-out."""
    devices = tuple(jax.local_devices() if devices is None else devices)
    log_cuda_cache_configuration(devices)
    if not enabled:
        logging.info("GPU collective warmup is disabled by distributed.warmup_collectives")
        return
    run_local_fsdp_collective_probe(devices=devices)


def _validate_collective_results(
    all_reduce: np.ndarray,
    all_gather: np.ndarray,
    reduce_scatter: np.ndarray,
    *,
    expected_sum: float,
    device_count: int,
    payload_rows: int,
    payload_columns: int,
) -> None:
    expected_all_reduce = np.full((device_count,), expected_sum, dtype=np.float32)
    if all_reduce.shape != expected_all_reduce.shape or not np.array_equal(all_reduce, expected_all_reduce):
        raise RuntimeError(f"AllReduce produced {all_reduce!r}; expected {expected_all_reduce!r}")

    gathered_once = np.repeat(np.arange(device_count, dtype=np.float32), payload_rows * payload_columns).reshape(
        device_count * payload_rows, payload_columns
    )
    expected_all_gather = np.broadcast_to(gathered_once, (device_count, *gathered_once.shape))
    if all_gather.shape != expected_all_gather.shape or not np.array_equal(all_gather, expected_all_gather):
        raise RuntimeError(
            f"AllGather produced shape/value mismatch: shape={all_gather.shape}, expected={expected_all_gather.shape}"
        )

    expected_reduce_scatter = np.full(
        (device_count, payload_rows // device_count, payload_columns), expected_sum, dtype=np.float32
    )
    if reduce_scatter.shape != expected_reduce_scatter.shape or not np.array_equal(
        reduce_scatter, expected_reduce_scatter
    ):
        raise RuntimeError(
            "ReduceScatter produced shape/value mismatch: "
            f"shape={reduce_scatter.shape}, expected={expected_reduce_scatter.shape}"
        )


def _path_is_writable(path: pathlib.Path) -> bool:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(component) for component in version.split(".")[:3])
    except ValueError:
        return ()
