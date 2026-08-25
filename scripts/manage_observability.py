#!/usr/bin/env python3
"""Archive high-frequency OpenPI system logs according to retention policy."""

from pathlib import Path
from typing import Literal

import tyro

from openpi.training import observability


def main(
    root: Path,
    *,
    action: Literal["archive", "prune"] = "archive",
    raw_retention_days: int = 90,
    active_grace_seconds: int = 300,
) -> None:
    result = observability.archive_system_logs(
        root,
        raw_retention_days=raw_retention_days,
        prune=action == "prune",
        active_grace_seconds=active_grace_seconds,
    )
    print(
        f"Compressed {result['compressed']} files; pruned {result['pruned']} expired raw files; "
        f"skipped {result['skipped_active']} active files"
    )


if __name__ == "__main__":
    tyro.cli(main)
