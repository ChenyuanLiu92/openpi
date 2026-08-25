from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from openpi.data import agibotworld_beta as converter


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_scan_archive_indexes_only_training_cameras_and_bounded_read(tmp_path: Path):
    archive = tmp_path / "389" / "1-2.tar"
    archive.parent.mkdir()
    _write_tar(
        archive,
        {
            "1/videos/head_color.mp4": b"head",
            "1/videos/hand_left_color.mp4": b"left",
            "1/videos/hand_right_color.mp4": b"right",
            "1/videos/back_left_fisheye_color.mp4": b"ignored",
            "1/depth/head_depth_000000.png": b"ignored",
        },
    )

    result = converter._scan_archive((str(archive), "observation", 389))  # noqa: SLF001

    assert {row[2] for row in result.observations} == set(converter.CAMERA_FILES)
    head = next(row for row in result.observations if row[2] == "base_0_rgb")
    reference = converter.MemberRef(result.archive, head[3], head[4], head[5])
    with converter.TarMemberIO(reference) as stream:
        assert stream.read(2) == b"he"
        assert stream.seek(-2, io.SEEK_END) == 2
        assert stream.read() == b"ad"


def test_scan_proprio_archive(tmp_path: Path):
    archive = tmp_path / "1-2.tar"
    _write_tar(
        archive,
        {
            "389/1/proprio_stats.h5": b"one",
            "389/2/proprio_stats.h5": b"two",
            "389/2/parameters.h5": b"ignored",
        },
    )

    result = converter._scan_archive((str(archive), "proprio", None))  # noqa: SLF001

    assert [(row[0], row[1], row[2]) for row in result.proprio] == [
        (389, 1, "389/1/proprio_stats.h5"),
        (389, 2, "389/2/proprio_stats.h5"),
    ]


def test_index_preserves_duplicate_episode_members_from_overlapping_archives(tmp_path: Path):
    first = tmp_path / "389" / "1-10.tar"
    second = tmp_path / "389" / "1-20.tar"
    first.parent.mkdir()
    members = {f"1/videos/{filename}": filename.encode() for filename in converter.CAMERA_FILES.values()}
    _write_tar(first, members)
    _write_tar(second, members)
    connection = converter._connect_index(tmp_path / "index.sqlite3")  # noqa: SLF001
    try:
        converter._store_archive_scan(  # noqa: SLF001
            connection,
            converter._scan_archive((str(first), "observation", 389)),  # noqa: SLF001
        )
        converter._store_archive_scan(  # noqa: SLF001
            connection,
            converter._scan_archive((str(second), "observation", 389)),  # noqa: SLF001
        )
        count = connection.execute("SELECT COUNT(*) FROM observation_members").fetchone()[0]
    finally:
        connection.close()

    assert count == 2 * len(converter.CAMERA_FILES)


def test_annotation_prompts_clip_segments_and_fall_back_to_task():
    annotation = converter.EpisodeAnnotation(
        task_name="fallback",
        init_scene_text="scene",
        action_segments=((-2, 2, "first"), (3, 20, "last")),
    )

    assert annotation.prompts(5) == ["first", "first", "fallback", "last", "last"]


def test_load_proprio_uses_canonical_20d_state_and_22d_action(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "proprio.h5"
    length = 3
    with h5py.File(h5_path, "w") as file:
        for path, width in zip(converter.STATE_PATHS, (14, 2, 2, 2), strict=True):
            file.create_dataset(path, data=np.ones((length, width)))
        for path, width in zip(converter.ACTION_PATHS, (14, 2, 2, 2, 2), strict=True):
            file.create_dataset(path, data=np.ones((length, width)) * 2)
        file.create_dataset("timestamp", data=np.arange(length, dtype=np.int64))
    archive = tmp_path / "stats.tar"
    _write_tar(archive, {"389/1/proprio_stats.h5": h5_path.read_bytes()})
    scan = converter._scan_archive((str(archive), "proprio", None))  # noqa: SLF001
    row = scan.proprio[0]
    member = converter.MemberRef(scan.archive, row[2], row[3], row[4])

    state, action, timestamp = converter._load_proprio(member)  # noqa: SLF001

    assert state.shape == (length, converter.STATE_DIM)
    assert action.shape == (length, converter.ACTION_DIM)
    np.testing.assert_array_equal(timestamp, np.arange(length))


def test_plan_fingerprint_changes_with_member_identity(tmp_path: Path):
    config = converter.ConverterConfig(input_root=tmp_path)
    annotation = converter.EpisodeAnnotation("task", "scene", ())
    member = converter.MemberRef("a.tar", "video.mp4", 512, 100)
    episode = converter.EpisodeRef(
        task_id=1,
        episode_id=2,
        cameras=dict.fromkeys(converter.CAMERA_FILES, member),
        proprio=converter.MemberRef("p.tar", "stats.h5", 512, 10),
        annotation=annotation,
    )
    changed = converter.EpisodeRef(
        task_id=1,
        episode_id=2,
        cameras={key: dataclasses.replace(member, offset=1024) for key in converter.CAMERA_FILES},
        proprio=episode.proprio,
        annotation=annotation,
    )

    assert converter.plan_fingerprint(config, [episode]) != converter.plan_fingerprint(config, [changed])


def test_auto_decoder_uses_projected_machine_throughput(monkeypatch, tmp_path: Path):
    member = converter.MemberRef("a.tar", "video.mp4", 512, 100)
    episode = converter.EpisodeRef(
        task_id=1,
        episode_id=2,
        cameras=dict.fromkeys(converter.CAMERA_FILES, member),
        proprio=converter.MemberRef("p.tar", "stats.h5", 512, 10),
        annotation=converter.EpisodeAnnotation("task", "scene", ()),
    )
    monkeypatch.setattr(converter, "physical_core_count", lambda: 90)
    monkeypatch.setattr(converter, "nvidia_gpu_count", lambda: 8)
    monkeypatch.setattr(
        converter,
        "_benchmark_member",
        lambda _member, decoder, _threads: (100, 5.0 if decoder == converter.Decoder.CPU else 4.0),
    )

    selected = converter.select_decoder(converter.ConverterConfig(input_root=tmp_path), episode)

    assert selected == converter.Decoder.CPU


def test_resolve_decoder_reuses_existing_auto_plan(monkeypatch, tmp_path: Path):
    config = converter.ConverterConfig(input_root=tmp_path, output_root=tmp_path / "out", dataset_name="pilot")
    config.version_dir.mkdir(parents=True)
    (config.version_dir / "conversion_plan.json").write_text(
        json.dumps({"plan_fingerprint": "fingerprint", "decoder": "nvidia"})
    )
    monkeypatch.setattr(
        converter,
        "select_decoder",
        lambda *_args: pytest.fail("resume should not benchmark the decoder"),
    )

    selected = converter._resolve_decoder(config, object(), "fingerprint", available_jobs=1)  # noqa: SLF001

    assert selected == converter.Decoder.NVIDIA
