"""Run a reproducible two-rank pi0.5 pre-training smoke test with mock RLDS data."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = _REPO_ROOT / "configs" / "pretraining" / "pi05" / "template.yaml"
_LAUNCHER = _REPO_ROOT / "scripts" / "launch_pretrain.py"
_MOCK_RLDS_BUILDER = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", required=True, type=pathlib.Path, help="New PFS directory for all smoke artifacts."
    )
    parser.add_argument("--device-group", action="append")
    parser.add_argument("--min-memory-headroom-gib", type=float, default=140.0)
    parser.add_argument("--wait-for-memory-seconds", type=float, default=0.0)
    parser.add_argument("--max-cgroup-memory-percent", type=float, default=95.0)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Generate and validate mock inputs without JAX ranks."
    )
    return parser.parse_args()


def _write_mock_rlds(data_dir: pathlib.Path, *, source_offset: float) -> tuple[str, str]:
    global _MOCK_RLDS_BUILDER  # noqa: PLW0603

    import tensorflow_datasets as tfds

    if _MOCK_RLDS_BUILDER is None:

        class MockRldsBuilder(tfds.core.GeneratorBasedBuilder):
            VERSION = tfds.core.Version("1.0.0")

            def __init__(self, *, source_offset: float, **kwargs):
                self._source_offset = source_offset
                super().__init__(**kwargs)

            def _info(self):
                step = tfds.features.FeaturesDict(
                    {
                        "observation": tfds.features.FeaturesDict(
                            {
                                "image": tfds.features.Image(shape=(224, 224, 3), encoding_format="png"),
                                "wrist_image": tfds.features.Image(shape=(224, 224, 3), encoding_format="png"),
                                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "action": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        "discount": np.float32,
                        "is_first": np.bool_,
                        "is_last": np.bool_,
                        "is_terminal": np.bool_,
                        "reward": np.float32,
                    }
                )
                return tfds.core.DatasetInfo(
                    builder=self,
                    features=tfds.features.FeaturesDict({"steps": tfds.features.Dataset(step)}),
                )

            def _split_generators(self, dl_manager):
                del dl_manager
                return {"train": self._generate_examples()}

            def _generate_examples(self):
                for episode_index in range(16):
                    steps = []
                    for step_index in range(8):
                        value = self._source_offset + episode_index / 100.0 + step_index / 1000.0
                        image_value = np.uint8((episode_index * 13 + step_index * 7) % 255)
                        steps.append(
                            {
                                "observation": {
                                    "image": np.full((224, 224, 3), image_value, dtype=np.uint8),
                                    "wrist_image": np.full((224, 224, 3), 255 - image_value, dtype=np.uint8),
                                    "state": np.full(8, value, dtype=np.float32),
                                    "language_instruction": f"mock task {self._source_offset:g}",
                                },
                                "action": np.full(8, value / 2, dtype=np.float32),
                                "discount": np.float32(1.0),
                                "is_first": step_index == 0,
                                "is_last": step_index == 7,
                                "is_terminal": step_index == 7,
                                "reward": np.float32(0.0),
                            }
                        )
                    yield f"episode-{episode_index:05d}", {"steps": steps}

        _MOCK_RLDS_BUILDER = MockRldsBuilder

    builder = _MOCK_RLDS_BUILDER(data_dir=data_dir, source_offset=source_offset)
    builder.download_and_prepare()
    return builder.name, str(builder.version)


def _source(source_id: str, tfds_name: str, version: str, data_dir: pathlib.Path, weight: float) -> dict[str, Any]:
    return {
        "id": source_id,
        "tfds_name": tfds_name,
        "version": version,
        "data_dir": str(data_dir),
        "train_split": "train[:75%]",
        "validation_split": "train[75%:]",
        "weight": weight,
        "normalization_id": f"{source_id}_stats",
        "action_stride": 1,
        "state_dim": 8,
        "action_dim": 8,
        "adapter": {
            "type": "field_map",
            "options": {
                "images": {
                    "base_0_rgb": "observation/image",
                    "left_wrist_0_rgb": "observation/wrist_image",
                    "right_wrist_0_rgb": None,
                },
                "state": ["observation/state"],
                "actions": ["action"],
                "prompt": "observation/language_instruction",
                "image_range": "uint8",
            },
        },
    }


def _write_config(work_dir: pathlib.Path, sources: list[dict[str, Any]]) -> pathlib.Path:
    contents = yaml.safe_load(_TEMPLATE.read_text())
    contents["name"] = "pi05_multinode_smoke"
    contents["data"].update(
        {
            "temperature": 1.0,
            "shuffle_buffer_size": 64,
            "num_parallel_reads": 1,
            "num_parallel_calls": 1,
            "prefetch_batches": 1,
            "sources": sources,
        }
    )
    contents["training"].update({"batch_size": 8, "num_train_steps": 1, "ema_decay": None})
    contents["paths"] = {
        "assets_base_dir": str(work_dir / "assets"),
        "checkpoint_base_dir": str(work_dir / "checkpoints"),
    }
    contents["checkpoint"].update(
        {"exp_name": "two-rank-full-model", "save_interval": 1, "keep_period": None, "overwrite": True, "resume": False}
    )
    contents["logging"] = {
        "project_name": "openpi-multinode-smoke",
        "wandb_enabled": False,
        "log_interval": 1,
    }
    contents["distributed"].update(
        {
            "fsdp_devices": 8,
            "warmup_collectives": True,
            "initialize": False,
            "coordinator_address": None,
            "coordinator_bind_address": None,
            "num_processes": None,
            "process_id": None,
            "local_device_ids": None,
            "cluster_detection_method": None,
        }
    )
    contents["validation"] = {"interval_steps": 1, "batches_per_source": 1}
    config_path = work_dir / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(contents, sort_keys=False, allow_unicode=True))
    return config_path


def _write_norm_stats(config_path: pathlib.Path) -> None:
    from openpi.shared import normalize
    from openpi.training import pretrain_config_loader
    from openpi.training import rlds_mixture

    config = pretrain_config_loader.load(config_path).config
    stats = {
        "state": normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=-np.ones(8), q99=np.ones(8)),
        "actions": normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=-np.ones(8), q99=np.ones(8)),
    }
    for source in config.data.sources:
        rlds_mixture.save_stats(config, source.normalization_id, stats, sample_count=128)


def _launcher_command(args: argparse.Namespace, config_path: pathlib.Path, log_dir: pathlib.Path) -> list[str]:
    command = [sys.executable, str(_LAUNCHER), "local", str(config_path)]
    for group in args.device_group or ["0,1,2,3", "4,5,6,7"]:
        command.extend(["--device-group", group])
    command.extend(
        [
            "--log-dir",
            str(log_dir),
            "--min-memory-headroom-gib",
            str(args.min_memory_headroom_gib),
            "--wait-for-memory-seconds",
            str(args.wait_for_memory_seconds),
            "--max-cgroup-memory-percent",
            str(args.max_cgroup_memory_percent),
        ]
    )
    return command


def _assert_checkpoint(config_path: pathlib.Path, expected_step: int, expected_examples: int) -> None:
    from openpi.training import pretrain_config_loader

    config = pretrain_config_loader.load(config_path).config
    checkpoint = config.checkpoint_dir / str(expected_step)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Expected checkpoint was not created: {checkpoint}")
    state = json.loads((checkpoint / "metadata" / "data_state.json").read_text())
    consumed = sum(state["consumed_examples_per_source"].values())
    if consumed != expected_examples:
        raise RuntimeError(f"Checkpoint {expected_step} consumed {consumed} examples; expected {expected_examples}")


def _assert_rank_logs(log_dir: pathlib.Path, *, probe: bool) -> None:
    logs = sorted(log_dir.glob("rank-*.log"))
    if len(logs) != 2:
        raise RuntimeError(f"Expected two rank logs in {log_dir}; found {logs}")
    combined = "\n".join(path.read_text() for path in logs)
    if probe:
        if combined.count("Global collective check passed") < 2:
            raise RuntimeError("Both ranks did not report a passing global collective probe")
    elif "process 0/2" not in combined or "process 1/2" not in combined:
        raise RuntimeError("Training logs do not contain both JAX process identities")


def _assert_finite_metrics(log_dir: pathlib.Path) -> None:
    pattern = re.compile(r"train/loss=([^, ]+).*train/grad_norm=([^, ]+)")
    for path in log_dir.glob("rank-*.log"):
        for match in pattern.finditer(path.read_text()):
            if math.isfinite(float(match.group(1))) and math.isfinite(float(match.group(2))):
                return
    raise RuntimeError("No finite train/loss and train/grad_norm pair was recorded")


def main() -> None:
    args = _parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    source_a_root = work_dir / "rlds-a"
    source_b_root = work_dir / "rlds-b"
    name_a, version_a = _write_mock_rlds(source_a_root, source_offset=0.0)
    name_b, version_b = _write_mock_rlds(source_b_root, source_offset=0.5)
    config_path = _write_config(
        work_dir,
        [
            _source("mock_a", name_a, version_a, source_a_root, 1.0),
            _source("mock_b", name_b, version_b, source_b_root, 3.0),
        ],
    )
    _write_norm_stats(config_path)
    if args.prepare_only:
        print(f"Prepared mock RLDS smoke inputs: {config_path}")
        return

    probe_command = [*_launcher_command(args, config_path, work_dir / "logs-probe"), "--probe-only"]
    subprocess.run(probe_command, cwd=_REPO_ROOT, check=True)
    _assert_rank_logs(work_dir / "logs-probe", probe=True)
    subprocess.run(_launcher_command(args, config_path, work_dir / "logs-step-1"), cwd=_REPO_ROOT, check=True)
    _assert_rank_logs(work_dir / "logs-step-1", probe=False)
    _assert_checkpoint(config_path, expected_step=1, expected_examples=8)
    _assert_finite_metrics(work_dir / "logs-step-1")

    contents = yaml.safe_load(config_path.read_text())
    contents["training"]["num_train_steps"] = 2
    contents["checkpoint"].update({"overwrite": False, "resume": True})
    config_path.write_text(yaml.safe_dump(contents, sort_keys=False, allow_unicode=True))
    subprocess.run(_launcher_command(args, config_path, work_dir / "logs-step-2"), cwd=_REPO_ROOT, check=True)
    _assert_rank_logs(work_dir / "logs-step-2", probe=False)
    _assert_checkpoint(config_path, expected_step=2, expected_examples=16)
    _assert_finite_metrics(work_dir / "logs-step-2")
    print(f"Full-model multi-rank smoke test passed: {work_dir}")


if __name__ == "__main__":
    main()
