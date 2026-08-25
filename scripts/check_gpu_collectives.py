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
    parser.add_argument("--coordinator-address", help="JAX coordinator host:port for a multi-process probe.")
    parser.add_argument("--coordinator-bind-address", help="Optional process-0 coordinator bind address.")
    parser.add_argument("--num-processes", type=int, help="Total JAX process count.")
    parser.add_argument("--process-id", type=int, help="This process's dense zero-based rank.")
    parser.add_argument("--local-device-ids", help="Comma-separated physical GPU IDs assigned to this rank.")
    parser.add_argument("--fsdp-devices", type=int, help="FSDP mesh width; defaults to all global devices.")
    parser.add_argument("--initialization-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    distributed_values = (args.coordinator_address, args.num_processes, args.process_id, args.local_device_ids)
    if any(value is not None for value in distributed_values) and not all(
        value is not None for value in distributed_values
    ):
        raise ValueError(
            "Distributed probing requires coordinator-address, num-processes, process-id, and local-device-ids"
        )
    if args.visible_devices and args.local_device_ids:
        raise ValueError("Use either --visible-devices or --local-device-ids, not both")
    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    # This probe transfers only a few bytes and should not reserve training-sized GPU pools.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # Import only after CUDA_VISIBLE_DEVICES is finalized.
    import jax

    from openpi.training import gpu_collectives
    from openpi.training import sharding

    if all(value is not None for value in distributed_values):
        local_device_ids = [int(value) for value in args.local_device_ids.split(",")]
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            coordinator_bind_address=args.coordinator_bind_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=local_device_ids,
            initialization_timeout=args.initialization_timeout,
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    devices = tuple(jax.local_devices())
    if args.expected_device_count is not None and len(devices) != args.expected_device_count:
        raise RuntimeError(f"Expected {args.expected_device_count} JAX devices, but found {len(devices)}: {devices}")

    gpu_collectives.log_cuda_cache_configuration(devices)
    result = gpu_collectives.run_local_fsdp_collective_probe(
        devices=devices,
        repetitions=args.repetitions,
        # A one-device rank has no meaningful local collective, but a multi-process job still has a global
        # collective to validate. Keep rejecting a single-process/single-device invocation because it tests neither.
        require_multiple_devices=jax.process_count() == 1,
    )
    logging.info(
        "Collective check passed: devices=%d, expected_sum=%.1f, timings=%s",
        result.device_count,
        result.expected_sum,
        ", ".join(f"{elapsed:.3f}s" for elapsed in result.elapsed_seconds),
    )
    if jax.process_count() > 1:
        fsdp_devices = args.fsdp_devices or jax.device_count()
        mesh = sharding.make_mesh(fsdp_devices)
        global_result = gpu_collectives.run_global_collective_probe(
            mesh,
            repetitions=args.repetitions,
            require_multiple_processes=True,
        )
        logging.info(
            "Global collective check passed: processes=%d, devices=%d, expected_sum=%.1f, timings=%s",
            global_result.process_count,
            global_result.global_device_count,
            global_result.expected_sum,
            ", ".join(f"{elapsed:.3f}s" for elapsed in global_result.elapsed_seconds),
        )


if __name__ == "__main__":
    main()
