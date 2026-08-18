"""Verify JAX GPU collectives before launching an expensive training job.

Examples:
    uv run scripts/check_gpu_collectives.py --visible-devices 4,5,6,7
    CUDA_VISIBLE_DEVICES=4,5 uv run scripts/check_gpu_collectives.py
"""

from __future__ import annotations

import argparse
import logging
import os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visible-devices",
        help="Comma-separated physical GPU IDs. Sets CUDA_VISIBLE_DEVICES before importing JAX.",
    )
    parser.add_argument("--expected-device-count", type=int, help="Fail unless JAX sees exactly this many devices.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=2,
        help="Number of checked all-reduces. The first includes cold compilation (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    # This probe transfers only a few bytes and should not reserve training-sized GPU pools.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # Import only after CUDA_VISIBLE_DEVICES is finalized.
    import jax

    from openpi.training import gpu_collectives

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    devices = tuple(jax.local_devices())
    if args.expected_device_count is not None and len(devices) != args.expected_device_count:
        raise RuntimeError(f"Expected {args.expected_device_count} JAX devices, but found {len(devices)}: {devices}")

    gpu_collectives.log_cuda_cache_configuration(devices)
    result = gpu_collectives.run_local_fsdp_collective_probe(
        devices=devices,
        repetitions=args.repetitions,
        require_multiple_devices=True,
    )
    logging.info(
        "Collective check passed: devices=%d, expected_sum=%.1f, timings=%s",
        result.device_count,
        result.expected_sum,
        ", ".join(f"{elapsed:.3f}s" for elapsed in result.elapsed_seconds),
    )


if __name__ == "__main__":
    main()
