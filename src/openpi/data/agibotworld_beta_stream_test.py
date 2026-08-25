from __future__ import annotations

import dataclasses
import hashlib
import io
from pathlib import Path
import sqlite3
import tarfile
import threading
import time

import numpy as np
import pytest

from openpi.data import agibotworld_beta as base
from openpi.data import agibotworld_beta_stream as stream


def _write_observation_tar(
    path: Path,
    episode_id: int,
    *,
    tar_format: int = tarfile.DEFAULT_FORMAT,
    ignored_size: int = 4097,
) -> dict[str, bytes]:
    payloads = {
        "head_color.mp4": b"head-video",
        "hand_left_color.mp4": b"left-video",
        "hand_right_color.mp4": b"right-video",
    }
    with tarfile.open(path, "w", format=tar_format) as archive:
        ignored_name = f"{episode_id}/depth/head_depth_000000.png"
        ignored = tarfile.TarInfo("ignored-placeholder" if tar_format == tarfile.PAX_FORMAT else ignored_name)
        if tar_format == tarfile.PAX_FORMAT:
            ignored.pax_headers = {"path": ignored_name}
        ignored.size = ignored_size
        archive.addfile(ignored, io.BytesIO(b"x" * ignored_size))
        for filename, payload in payloads.items():
            name = f"{episode_id}/videos/{filename}"
            info = tarfile.TarInfo("video-placeholder" if tar_format == tarfile.PAX_FORMAT else name)
            if tar_format == tarfile.PAX_FORMAT:
                info.pax_headers = {"path": name}
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return payloads


def _config(tmp_path: Path, archive: Path, *, episode_id: int = 7) -> stream.StreamConfig:
    input_root = tmp_path / "input"
    input_root.mkdir(exist_ok=True)
    return stream.StreamConfig(
        input_root=input_root,
        pfs_work_root=tmp_path / "work",
        output_root=tmp_path / "output",
        dataset_name="test_stream",
        episode_ids=(episode_id,),
        observation_archives=(archive,),
        publish=False,
    )


@pytest.mark.parametrize("tar_format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_sequential_scanner_extracts_only_training_videos_and_consumes_archive(tmp_path: Path, tar_format: int):
    archive = tmp_path / "input" / "observations" / "3" / "1-9.tar"
    archive.parent.mkdir(parents=True)
    payloads = _write_observation_tar(archive, 7, tar_format=tar_format)
    config = dataclasses.replace(_config(tmp_path, archive), episode_ids=())
    progress: list[int] = []

    result = stream.scan_observation_archive(
        config, stream.ObservationArchive(archive, task_id=3, ordinal=0), progress.append
    )

    assert sum(progress) == archive.stat().st_size
    assert result.selected_members == 3
    assert result.ready_episodes == (7,)
    for filename, camera in {value: key for key, value in base.CAMERA_FILES.items()}.items():
        path = config.pfs_work_root / "spool" / "3" / "7" / f"{camera}.mp4"
        assert path.read_bytes() == payloads[filename]
    assert not list((config.pfs_work_root / "spool").rglob("*depth*"))


def test_scanner_resumes_at_last_completed_member_boundary(monkeypatch, tmp_path: Path):
    archive = tmp_path / "input" / "observations" / "3" / "1-9.tar"
    archive.parent.mkdir(parents=True)
    _write_observation_tar(archive, 7, tar_format=tarfile.GNU_FORMAT)
    config = _config(tmp_path, archive)
    original = stream._copy_exact  # noqa: SLF001
    calls = 0

    def fail_on_first_video(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(stream, "_copy_exact", fail_on_first_video)
    with pytest.raises(OSError, match="injected interruption"):
        stream.scan_observation_archive(config, stream.ObservationArchive(archive, 3, 0))
    connection = stream._connect_state(config.state_path)  # noqa: SLF001
    try:
        status, offset = connection.execute(
            "SELECT status, offset FROM archives WHERE source_path=?", (str(archive.resolve()),)
        ).fetchone()
    finally:
        connection.close()
    assert status == "failed"
    assert 0 < offset < archive.stat().st_size

    monkeypatch.setattr(stream, "_copy_exact", original)
    result = stream.scan_observation_archive(config, stream.ObservationArchive(archive, 3, 0))

    assert result.ready_episodes == (7,)
    assert not list(config.pfs_work_root.rglob("*.partial"))


def test_incomplete_camera_set_is_quarantined_and_blocks_publish(tmp_path: Path):
    archive = tmp_path / "input" / "observations" / "3" / "1-9.tar"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as tar:
        payload = b"only-one-camera"
        info = tarfile.TarInfo("7/videos/head_color.mp4")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    config = dataclasses.replace(_config(tmp_path, archive), episode_ids=())

    stream.scan_observation_archive(config, stream.ObservationArchive(archive, 3, 0))

    state = stream._connect_state(config.state_path)  # noqa: SLF001
    try:
        status, error = state.execute("SELECT status, error FROM episodes WHERE task_id=3 AND episode_id=7").fetchone()
        assert status == "failed"
        assert "1/3" in error
        assert state.execute("SELECT COUNT(*) FROM spool_members").fetchone()[0] == 0
    finally:
        state.close()
    with pytest.raises(RuntimeError, match="failed episodes=1"):
        stream.finalize_local_dataset(config)


def test_resumable_copy_reuses_durable_partial(tmp_path: Path):
    source = tmp_path / "source.bin"
    payload = bytes(range(251)) * 1000
    source.write_bytes(payload)
    destination = tmp_path / "dest" / "copy.bin"
    destination.parent.mkdir()
    partial = destination.with_name(f".{destination.name}.partial")
    partial.write_bytes(payload[:12345])
    progress: list[int] = []

    size, digest = stream.resumable_copy(source, destination, tmp_path / "state.sqlite3", progress.append)

    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert sum(progress) == len(payload) - 12345
    assert not partial.exists()


def test_fragment_acceptance_is_the_gate_for_spool_cleanup(tmp_path: Path):
    archive = tmp_path / "input" / "observations" / "3" / "1-9.tar"
    archive.parent.mkdir(parents=True)
    _write_observation_tar(archive, 7)
    config = _config(tmp_path, archive)
    stream.scan_observation_archive(config, stream.ObservationArchive(archive, 3, 0))
    connection = stream._connect_state(config.state_path)  # noqa: SLF001
    try:
        spool_paths = [Path(row[0]) for row in connection.execute("SELECT path FROM spool_members")]
        assert spool_paths
        assert all(path.exists() for path in spool_paths)
        connection.execute(
            """
            INSERT INTO fragments(fragment_key, source_archive, fragment_index, path, status,
                                  episode_keys, fingerprint, updated_at)
            VALUES('fragment', ?, 0, ?, 'writing', '["3/7"]', 'fingerprint', 'now')
            """,
            (str(archive.resolve()), str(tmp_path / "fragment.tfrecord")),
        )
        connection.commit()
        assert all(path.exists() for path in spool_paths)

        result = stream.FragmentResult(
            key="fragment",
            path=str(tmp_path / "fragment.tfrecord"),
            episode_keys=("3/7",),
            frames=10,
            bytes_written=100,
            sha256="digest",
            elapsed_seconds=1.0,
            errors=(),
        )
        stream._accept_fragment_result(connection, result)  # noqa: SLF001

        assert not any(path.exists() for path in spool_paths)
        assert connection.execute("SELECT status FROM episodes").fetchone()[0] == "converted"
        assert connection.execute("SELECT COUNT(*) FROM spool_members").fetchone()[0] == 0
    finally:
        connection.close()


def test_standalone_spool_source_is_preserved_in_episode_metadata(monkeypatch, tmp_path: Path):
    pytest.importorskip("tensorflow_datasets")
    config = base.ConverterConfig(input_root=tmp_path)
    member = base.MemberRef(str(tmp_path / "spooled.mp4"), "7/videos/head_color.mp4", 0, 10)
    episode = base.EpisodeRef(
        task_id=3,
        episode_id=7,
        cameras=dict.fromkeys(base.CAMERA_FILES, member),
        proprio=base.MemberRef(str(tmp_path / "local.tar"), "3/7/proprio_stats.h5", 512, 10),
        annotation=base.EpisodeAnnotation("task", "scene", ()),
        source_observation_archive="/mnt/bos/source.tar",
        source_proprio_archive="/mnt/bos/proprio.tar",
    )
    monkeypatch.setattr(
        base,
        "_load_proprio",
        lambda _member: (
            __import__("numpy").zeros((1, base.STATE_DIM), dtype="float32"),
            __import__("numpy").zeros((1, base.ACTION_DIM), dtype="float32"),
            __import__("numpy").zeros((1,), dtype="int64"),
        ),
    )
    monkeypatch.setattr(base, "_decode_camera", lambda *_args, **_kwargs: (b"jpeg",))
    captured = {}

    class Features:
        def serialize_example(self, payload):
            captured.update(payload)
            return b"serialized"

    serialized, frames = base._serialize_episode(  # noqa: SLF001
        episode, config, base.Decoder.CPU, Features()
    )

    assert serialized == b"serialized"
    assert frames == 1
    assert captured["episode_metadata"]["source_observation_archive"] == "/mnt/bos/source.tar"
    assert captured["episode_metadata"]["source_proprio_archive"] == "/mnt/bos/proprio.tar"


def test_verified_fragment_finalizes_to_readable_tfds_directory(tmp_path: Path):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_datasets")
    archive = tmp_path / "input" / "observations" / "3" / "1-9.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"source identity")
    config = _config(tmp_path, archive)
    fragment = config.fragment_dir / "fragment-00000.tfrecord"
    fragment.parent.mkdir(parents=True)
    features = base.make_tfds_features(config.converter_config())
    image = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)
    example = features.serialize_example(
        {
            "steps": [
                {
                    "observation": {
                        "base_0_rgb": image,
                        "left_wrist_0_rgb": image,
                        "right_wrist_0_rgb": image,
                        "state": np.zeros(base.STATE_DIM, dtype=np.float32),
                    },
                    "action": np.zeros(base.ACTION_DIM, dtype=np.float32),
                    "language_instruction": "test",
                    "timestamp_ns": np.int64(0),
                    "is_first": True,
                    "is_last": True,
                    "is_terminal": True,
                    "reward": np.float32(0),
                    "discount": np.float32(0),
                }
            ],
            "episode_metadata": {
                "episode_id": 7,
                "task_id": 3,
                "task_name": "test",
                "init_scene_text": "scene",
                "source_observation_archive": str(archive),
                "source_proprio_archive": "proprio.tar",
            },
        }
    )
    with tf.io.TFRecordWriter(str(fragment)) as writer:
        writer.write(example)
    digest = stream._sha256_file(fragment)  # noqa: SLF001
    connection = stream._connect_state(config.state_path)  # noqa: SLF001
    try:
        connection.execute(
            """
            INSERT INTO archives(source_path, task_id, ordinal, size, mtime_ns, status, offset,
                                 selected_members, updated_at)
            VALUES(?, 3, 0, ?, ?, 'complete', ?, 3, 'now')
            """,
            (
                str(archive.resolve()),
                archive.stat().st_size,
                archive.stat().st_mtime_ns,
                archive.stat().st_size,
            ),
        )
        connection.execute(
            """
            INSERT INTO episodes(task_id, episode_id, status, source_archive, updated_at)
            VALUES(3, 7, 'converted', ?, 'now')
            """,
            (str(archive.resolve()),),
        )
        connection.execute(
            """
            INSERT INTO fragments(fragment_key, source_archive, fragment_index, path, status,
                                  episode_keys, frames, bytes_written, sha256, fingerprint, updated_at)
            VALUES('fragment', ?, 0, ?, 'complete', '["3/7"]', 5, ?, ?, 'fingerprint', 'now')
            """,
            (str(archive.resolve()), str(fragment), fragment.stat().st_size, digest),
        )
        connection.commit()
    finally:
        connection.close()

    manifest = stream.finalize_local_dataset(config)
    result = stream.validate_tfds_directory(config.local_version_dir, expected_episodes=1)

    assert manifest["episodes"] == 1
    assert manifest["frames"] == 5
    assert result == {"episodes": 1, "shards": 1}


def test_status_is_read_only_for_missing_state(tmp_path: Path):
    config = dataclasses.replace(_config(tmp_path, tmp_path / "missing.tar"), pfs_work_root=tmp_path / "new")

    result = stream.status(config)

    assert result["state"] == "not_started"
    assert not config.state_path.exists()


def test_status_reports_durable_archive_byte_progress(tmp_path: Path):
    archive = tmp_path / "input" / "observations" / "3" / "1-9.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"x" * 100)
    config = _config(tmp_path, archive)
    connection = stream._connect_state(config.state_path)  # noqa: SLF001
    try:
        connection.execute(
            """
            INSERT INTO archives(source_path, task_id, ordinal, size, mtime_ns, status, offset,
                                 selected_members, updated_at)
            VALUES(?, 3, 0, 100, ?, 'scanning', 40, 0, 'now')
            """,
            (str(archive.resolve()), archive.stat().st_mtime_ns),
        )
        connection.commit()
    finally:
        connection.close()

    result = stream.status(config)

    assert result["archive_bytes_total"] == 100
    assert result["archive_bytes_processed"] == 40


def test_state_connection_retries_concurrent_writer_lock(tmp_path: Path):
    state_path = tmp_path / "state.sqlite3"
    holder = stream._connect_state(state_path)  # noqa: SLF001
    worker_ready = threading.Event()
    worker_start = threading.Event()
    errors: list[BaseException] = []

    def write_from_thread() -> None:
        connection = stream._connect_state(state_path)  # noqa: SLF001
        try:
            connection.execute("PRAGMA busy_timeout=10")
            worker_ready.set()
            worker_start.wait()
            with connection:
                connection.execute("INSERT OR REPLACE INTO metadata VALUES('worker', 'complete')")
        except BaseException as exc:
            errors.append(exc)
        finally:
            connection.close()

    thread = threading.Thread(target=write_from_thread)
    thread.start()
    assert worker_ready.wait(timeout=5)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT OR REPLACE INTO metadata VALUES('holder', 'active')")
    worker_start.set()
    time.sleep(0.2)
    holder.commit()
    holder.close()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    connection = sqlite3.connect(state_path)
    try:
        assert connection.execute("SELECT value FROM metadata WHERE key='worker'").fetchone() == ("complete",)
    finally:
        connection.close()


def test_state_connection_recovers_busy_snapshot(tmp_path: Path):
    state_path = tmp_path / "state.sqlite3"
    reader = stream._connect_state(state_path)  # noqa: SLF001
    writer = stream._connect_state(state_path)  # noqa: SLF001
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] >= 1
        with writer:
            writer.execute("INSERT OR REPLACE INTO metadata VALUES('other_writer', 'complete')")

        with reader:
            reader.execute("INSERT OR REPLACE INTO metadata VALUES('snapshot_reader', 'recovered')")

        assert reader.execute("SELECT value FROM metadata WHERE key='snapshot_reader'").fetchone() == ("recovered",)
    finally:
        reader.close()
        writer.close()
