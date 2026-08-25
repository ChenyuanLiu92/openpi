#!/usr/bin/env python3
"""Convert AgiBotWorld Beta tar archives directly to resumable sharded TFDS/RLDS.

Example pilot:
  uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py \
    --task-id 389 --episode-id 655660 --dataset-name agibotworld_beta_pilot

Full conversion (source tar files are retained by default):
  uv run --group rlds scripts/convert_agibotworld_beta_to_rlds.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from openpi.data import agibotworld_beta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/bos/dataset/agibotworld-beta"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/bos/dataset/RLDS"))
    parser.add_argument("--dataset-name", default="agibotworld_beta")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--task-id", type=int, action="append", default=[], dest="task_ids")
    parser.add_argument("--episode-id", type=int, action="append", default=[], dest="episode_ids")
    parser.add_argument(
        "--observation-archive",
        type=Path,
        action="append",
        default=[],
        dest="observation_archives",
        help="Convert only these observation tar files. May be repeated.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--episodes-per-shard", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0, help="Episode/shard workers; 0 uses hardware-aware sizing.")
    parser.add_argument(
        "--index-workers", type=int, default=0, help="Tar-header index workers; 0 selects automatically."
    )
    parser.add_argument(
        "--decoder",
        choices=[decoder.value for decoder in agibotworld_beta.Decoder],
        default=agibotworld_beta.Decoder.AUTO.value,
    )
    parser.add_argument("--decoder-threads", type=int, default=1)
    parser.add_argument("--gpu-workers-per-device", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--alignment-tolerance", type=int, default=2)
    parser.add_argument("--skip-bad-episodes", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument(
        "--delete-source-archives-after-success",
        action="store_true",
        help="Delete only tar archives whose every indexed episode is present in the verified output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    explicitly_converted = {"task_ids", "episode_ids", "observation_archives", "decoder"}
    config = agibotworld_beta.ConverterConfig(
        **{
            field.name: getattr(args, field.name)
            for field in dataclasses.fields(agibotworld_beta.ConverterConfig)
            if hasattr(args, field.name) and field.name not in explicitly_converted
        },
        task_ids=tuple(args.task_ids),
        episode_ids=tuple(args.episode_ids),
        observation_archives=tuple(args.observation_archives),
        decoder=agibotworld_beta.Decoder(args.decoder),
    )
    manifest = agibotworld_beta.run_conversion(config)
    if manifest is not None:
        logging.info(
            "Conversion complete: episodes=%d frames=%d shards=%d bytes=%d",
            manifest["episodes"],
            manifest["frames"],
            manifest["shards"],
            manifest["bytes"],
        )


if __name__ == "__main__":
    main()
