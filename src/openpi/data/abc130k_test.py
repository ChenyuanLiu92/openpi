from pathlib import Path
import time

import numpy as np
import pytest

from openpi.data import abc130k


def _fake_episode(root: Path, split: str, task: str, episode_id: str, size: int = 16) -> Path:
    directory = root / "data" / split / task / f"episode_{episode_id}"
    directory.mkdir(parents=True)
    path = directory / "episode.mcap"
    path.write_bytes(b"x" * size)
    return path


def _minimal_mcap(path: Path) -> None:
    from mcap.writer import Writer

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        writer.finish()


def test_floor_indices_are_causal() -> None:
    source = np.asarray([10, 20, 30], dtype=np.int64)
    targets = np.asarray([10, 19, 20, 29, 31], dtype=np.int64)

    np.testing.assert_array_equal(abc130k.floor_indices(source, targets), [0, 0, 1, 1, 2])


def test_deterministic_top_topic_supports_stereo_and_mono() -> None:
    stereo = {"/top-left-camera", "/top-right-camera", "/top-camera"}
    first = abc130k.deterministic_top_topic("abc", stereo)

    assert first in {"/top-left-camera", "/top-right-camera"}
    assert abc130k.deterministic_top_topic("abc", stereo) == first
    assert abc130k.deterministic_top_topic("abc", {"/top-camera"}) == "/top-camera"
    with pytest.raises(ValueError, match="top camera"):
        abc130k.deterministic_top_topic("abc", set())


def test_discover_and_plan_preserve_train_validation_splits(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _fake_episode(input_root, "train", "fold_shirt", "002")
    _fake_episode(input_root, "train", "fold_shirt", "001")
    _fake_episode(input_root, "val", "stack_cups", "003")
    config = abc130k.ConversionConfig(
        input_root=input_root,
        pfs_work_root=tmp_path / "work",
        output_root=tmp_path / "bos",
        episodes_per_shard=1,
    )
    config.validate()

    episodes = abc130k.discover_episodes(config)
    jobs = abc130k.build_plan(config, episodes, "fingerprint")

    assert [(episode.split, episode.episode_id) for episode in episodes] == [
        ("train", "001"),
        ("train", "002"),
        ("val", "003"),
    ]
    assert [job.key for job in jobs] == ["train-00000", "train-00001", "val-00000"]
    assert abc130k.status(config)["episodes"] == {"pending": 3}


def test_explicit_episode_path_and_filters(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    selected = _fake_episode(input_root, "train", "fold_shirt", "selected")
    _fake_episode(input_root, "train", "fold_shirt", "ignored")
    config = abc130k.ConversionConfig(
        input_root=input_root,
        pfs_work_root=tmp_path / "work",
        output_root=tmp_path / "bos",
        splits=("train",),
        episode_paths=(selected.parent,),
    )

    episodes = abc130k.discover_episodes(config)

    assert len(episodes) == 1
    assert episodes[0].episode_id == "selected"


def test_parallel_discovery_is_sorted_and_applies_max_episodes(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _fake_episode(input_root, "train", "task_b", "002")
    _fake_episode(input_root, "train", "task_a", "003")
    _fake_episode(input_root, "train", "task_a", "001")
    config = abc130k.ConversionConfig(
        input_root=input_root,
        pfs_work_root=tmp_path / "work",
        output_root=tmp_path / "bos",
        splits=("train",),
        discovery_workers=2,
        max_episodes=2,
    )

    episodes = abc130k.discover_episodes(config)

    assert [(episode.task_name, episode.episode_id) for episode in episodes] == [
        ("task_a", "001"),
        ("task_a", "003"),
    ]


def test_staged_file_is_reused_only_for_same_source_version(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    destination = tmp_path / "spool" / "episode.mcap"
    _minimal_mcap(source)

    copied = abc130k._stage_file(source, destination, None, require_summary=True)  # noqa: SLF001

    assert copied == source.stat().st_size
    assert abc130k._staged_file_is_current(source, destination, require_summary=True)  # noqa: SLF001
    old_metadata = abc130k._staging_metadata_path(destination).read_text()  # noqa: SLF001

    time.sleep(0.001)
    _minimal_mcap(source)
    source.touch()

    assert not abc130k._staged_file_is_current(source, destination, require_summary=True)  # noqa: SLF001
    abc130k._stage_file(source, destination, None, require_summary=True)  # noqa: SLF001
    assert abc130k._staged_file_is_current(source, destination, require_summary=True)  # noqa: SLF001
    assert abc130k._staging_metadata_path(destination).read_text() != old_metadata  # noqa: SLF001


def test_staged_mcap_rejects_missing_footer(tmp_path: Path) -> None:
    path = tmp_path / "truncated.mcap"
    path.write_bytes(abc130k._MCAP_MAGIC + b"\x00" * 32)  # noqa: SLF001

    with pytest.raises(ValueError, match="footer magic"):
        abc130k._validate_staged_mcap(path, require_summary=False)  # noqa: SLF001


def test_semantic_identity_changes_for_output_affecting_options(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    common = {
        "input_root": input_root,
        "pfs_work_root": tmp_path / "work",
        "output_root": tmp_path / "bos",
    }

    strict = abc130k.ConversionConfig(**common)
    permissive = abc130k.ConversionConfig(**common, skip_bad_episodes=True)

    assert strict.semantic_identity != permissive.semantic_identity


def test_strict_work_root_can_migrate_to_skip_bad_policy(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    common = {
        "input_root": input_root,
        "pfs_work_root": tmp_path / "work",
        "output_root": tmp_path / "bos",
    }
    strict = abc130k.ConversionConfig(**common)
    strict.validate()
    original_fingerprint = abc130k._assert_identity(strict)  # noqa: SLF001

    permissive = abc130k.ConversionConfig(**common, skip_bad_episodes=True)
    migrated_fingerprint = abc130k._assert_identity(permissive)  # noqa: SLF001

    assert migrated_fingerprint == original_fingerprint
