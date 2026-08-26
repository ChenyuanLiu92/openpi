"""Run bounded JAX GPU compute or framebuffer-memory validation.

This is a worker for ``validate_pretrain_gpu.py``. It deliberately uses one
host process for all local GPUs so that GPU validation does not multiply host
memory usage by the device count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import time
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("compute", "memory"), required=True)
    parser.add_argument("--expected-device-count", type=int, default=8)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--memory-gib-per-device", type=float, default=75.0)
    parser.add_argument("--memory-chunk-mib", type=int, default=1000)
    parser.add_argument("--max-throughput-spread", type=float, default=0.10)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.expected_device_count <= 0:
        raise ValueError("expected-device-count must be positive")
    if args.duration_seconds <= 0 or args.matrix_size <= 0:
        raise ValueError("duration-seconds and matrix-size must be positive")
    if args.memory_gib_per_device <= 0 or args.memory_chunk_mib <= 0:
        raise ValueError("GPU memory target and chunk size must be positive")
    if not 0 <= args.max_throughput_spread < 1:
        raise ValueError("max-throughput-spread must be in [0, 1)")


def _write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _run_memory(args: argparse.Namespace, jax, jnp, devices) -> dict[str, Any]:
    chunk_elements = args.memory_chunk_mib * 2**20 // 4
    chunks_per_device = math.ceil(args.memory_gib_per_device * 2**30 / (chunk_elements * 4))
    allocated_gib = chunks_per_device * chunk_elements * 4 / 2**30
    arrays: dict[int, list[Any]] = {device.id: [] for device in devices}
    makers = {
        device.id: jax.jit(
            lambda value, elements=chunk_elements: jnp.full((elements,), value, dtype=jnp.uint32),
            device=device,
        )
        for device in devices
    }
    checksums = {
        device.id: jax.jit(lambda value: jnp.sum(value, dtype=jnp.uint32), device=device) for device in devices
    }
    started = time.monotonic()
    for chunk_index in range(chunks_per_device):
        for device in devices:
            value = (device.id + 1) * 1009 + chunk_index * 9176 + 1
            array = makers[device.id](jnp.asarray(value, dtype=jnp.uint32))
            jax.block_until_ready(array)
            arrays[device.id].append(array)

    validation = []
    for device in devices:
        for chunk_index, array in enumerate(arrays[device.id]):
            value = (device.id + 1) * 1009 + chunk_index * 9176 + 1
            expected = (value * chunk_elements) % 2**32
            actual = int(jax.device_get(checksums[device.id](array)))
            if actual != expected:
                raise RuntimeError(
                    f"GPU {device.id} memory checksum mismatch for chunk {chunk_index}: {actual} != {expected}"
                )
        validation.append({"device_id": device.id, "chunks": len(arrays[device.id]), "checksum": "passed"})
    return {
        "status": "pass",
        "workload": "memory",
        "device_count": len(devices),
        "target_gib_per_device": args.memory_gib_per_device,
        "allocated_gib_per_device": allocated_gib,
        "chunk_mib": args.memory_chunk_mib,
        "elapsed_seconds": time.monotonic() - started,
        "devices": validation,
    }


def _compute_on_device(args: argparse.Namespace, jax, jnp, device) -> dict[str, Any]:
    initialize = jax.jit(
        lambda key: (
            jax.random.normal(key, (args.matrix_size, args.matrix_size), dtype=jnp.bfloat16),
            jax.random.normal(jax.random.fold_in(key, 1), (args.matrix_size, args.matrix_size), dtype=jnp.bfloat16),
        ),
        device=device,
    )
    step = jax.jit(lambda left, right: jnp.mean((left @ right).astype(jnp.float32)), device=device)
    left, right = initialize(jax.random.PRNGKey(device.id + 1729))
    warmup = step(left, right)
    jax.block_until_ready(warmup)
    if not math.isfinite(float(jax.device_get(warmup))):
        raise RuntimeError(f"GPU {device.id} produced a non-finite BF16 GEMM warmup result")

    iterations = 0
    started = time.monotonic()
    deadline = started + args.duration_seconds
    last_value = warmup
    while time.monotonic() < deadline:
        last_value = step(left, right)
        jax.block_until_ready(last_value)
        iterations += 1
    elapsed = time.monotonic() - started
    value = float(jax.device_get(last_value))
    if iterations == 0 or not math.isfinite(value):
        raise RuntimeError(f"GPU {device.id} produced no finite BF16 GEMM result")
    operations = 2 * args.matrix_size**3 * iterations
    return {
        "device_id": device.id,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "tflops": operations / elapsed / 1e12,
        "result": value,
    }


def _run_compute(args: argparse.Namespace, jax, jnp, devices) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="gpu-burn") as executor:
        futures = [executor.submit(_compute_on_device, args, jax, jnp, device) for device in devices]
        results = [future.result() for future in futures]
    throughputs = [result["tflops"] for result in results]
    median = sorted(throughputs)[len(throughputs) // 2]
    spread = (max(throughputs) - min(throughputs)) / max(median, 1e-12)
    if spread > args.max_throughput_spread:
        raise RuntimeError(
            f"Per-GPU BF16 throughput spread {spread:.1%} exceeds {args.max_throughput_spread:.1%}: {throughputs}"
        )
    return {
        "status": "pass",
        "workload": "compute",
        "device_count": len(devices),
        "matrix_size": args.matrix_size,
        "duration_seconds": args.duration_seconds,
        "throughput_spread": spread,
        "devices": sorted(results, key=lambda result: result["device_id"]),
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    devices = tuple(jax.local_devices())
    if len(devices) != args.expected_device_count:
        raise RuntimeError(f"Expected {args.expected_device_count} JAX GPUs, found {len(devices)}: {devices}")
    if any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU burn refuses non-GPU devices: {devices}")
    report = _run_memory(args, jax, jnp, devices) if args.workload == "memory" else _run_compute(
        args, jax, jnp, devices
    )
    _write_report(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
