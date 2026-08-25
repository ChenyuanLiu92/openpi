"""Production conversion of AgiBotWorld Beta WebDataset tar files to TFDS/RLDS.

The converter deliberately separates cheap tar-header indexing from expensive video
decoding. Uncompressed tar members are read with bounded seekable views, so source
archives are never expanded to disk and are scanned at most once per fingerprint.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
import concurrent.futures
import contextlib
import dataclasses
import datetime
import enum
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sqlite3
import subprocess
import tarfile
import time
from typing import Any, BinaryIO

import numpy as np
import tqdm_loggable.auto as tqdm

LOGGER = logging.getLogger(__name__)

DATASET_SCHEMA_VERSION = 1
CAMERA_FILES = {
    "base_0_rgb": "head_color.mp4",
    "left_wrist_0_rgb": "hand_left_color.mp4",
    "right_wrist_0_rgb": "hand_right_color.mp4",
}
STATE_PATHS = (
    "state/joint/position",
    "state/effector/position",
    "state/head/position",
    "state/waist/position",
)
ACTION_PATHS = (
    "action/joint/position",
    "action/effector/position",
    "action/head/position",
    "action/waist/position",
    "action/robot/velocity",
)
STATE_DIM = 20
ACTION_DIM = 22
_ARCHIVE_RANGE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)\.tar$")
_INDEX_SCHEMA_VERSION = 2


class Decoder(enum.StrEnum):
    """Available AV1 decoder policies."""

    AUTO = "auto"
    CPU = "cpu"
    NVIDIA = "nvidia"


@dataclasses.dataclass(frozen=True)
class ConverterConfig:
    input_root: Path = Path("/mnt/bos/dataset/agibotworld-beta")
    output_root: Path = Path("/mnt/bos/dataset/RLDS")
    dataset_name: str = "agibotworld_beta"
    version: str = "1.0.0"
    image_height: int = 224
    image_width: int = 224
    jpeg_quality: int = 90
    episodes_per_shard: int = 8
    workers: int = 0
    index_workers: int = 0
    decoder: Decoder = Decoder.AUTO
    decoder_threads: int = 1
    gpu_workers_per_device: int = 4
    alignment_tolerance: int = 2
    skip_bad_episodes: bool = False
    task_ids: tuple[int, ...] = ()
    episode_ids: tuple[int, ...] = ()
    observation_archives: tuple[Path, ...] = ()
    max_episodes: int | None = None
    plan_only: bool = False
    index_only: bool = False
    delete_source_archives_after_success: bool = False

    def validate(self) -> None:
        if not self.input_root.is_dir():
            raise FileNotFoundError(f"AgiBotWorld Beta input root not found: {self.input_root}")
        if not self.dataset_name or not re.fullmatch(r"[a-z0-9_]+", self.dataset_name):
            raise ValueError("dataset_name must contain only lowercase letters, digits, and underscores")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("version must have the form MAJOR.MINOR.PATCH")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.episodes_per_shard <= 0:
            raise ValueError("episodes_per_shard must be positive")
        if self.workers < 0 or self.index_workers < 0:
            raise ValueError("worker counts must be non-negative; use 0 for automatic sizing")
        if self.decoder_threads <= 0 or self.gpu_workers_per_device <= 0:
            raise ValueError("decoder_threads and gpu_workers_per_device must be positive")
        if self.alignment_tolerance < 0:
            raise ValueError("alignment_tolerance must be non-negative")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive or omitted")
        if self.plan_only and self.index_only:
            raise ValueError("plan_only and index_only cannot both be set")

    @property
    def version_dir(self) -> Path:
        return self.output_root / self.dataset_name / self.version

    @property
    def state_dir(self) -> Path:
        return self.output_root / ".agibotworld_beta_conversion"


@dataclasses.dataclass(frozen=True)
class MemberRef:
    archive: str
    name: str
    offset: int
    size: int


@dataclasses.dataclass(frozen=True)
class EpisodeAnnotation:
    task_name: str
    init_scene_text: str
    action_segments: tuple[tuple[int, int, str], ...]

    def prompts(self, length: int) -> list[str]:
        prompts = [self.task_name] * length
        for start, end, text in self.action_segments:
            clipped_start = max(0, min(start, length))
            clipped_end = max(clipped_start, min(end, length))
            prompts[clipped_start:clipped_end] = [text] * (clipped_end - clipped_start)
        return prompts


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    task_id: int
    episode_id: int
    cameras: dict[str, MemberRef]
    proprio: MemberRef
    annotation: EpisodeAnnotation
    source_observation_archive: str | None = None
    source_proprio_archive: str | None = None

    @property
    def key(self) -> str:
        return f"{self.task_id}/{self.episode_id}"


@dataclasses.dataclass(frozen=True)
class ArchiveScan:
    archive: str
    kind: str
    size: int
    mtime_ns: int
    observations: tuple[tuple[int, int, str, str, int, int], ...] = ()
    proprio: tuple[tuple[int, int, str, int, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class ShardJob:
    shard_index: int
    shard_count: int
    episodes: tuple[EpisodeRef, ...]
    config: ConverterConfig
    decoder: Decoder
    plan_fingerprint: str


@dataclasses.dataclass(frozen=True)
class ShardResult:
    shard_index: int
    path: str
    episode_keys: tuple[str, ...]
    frames: int
    bytes_written: int
    elapsed_seconds: float
    errors: tuple[str, ...]


class TarMemberIO(io.RawIOBase):
    """A seekable, bounded view into one regular file member of an uncompressed tar."""

    def __init__(self, member: MemberRef):
        super().__init__()
        self._file: BinaryIO = open(member.archive, "rb", buffering=0)  # noqa: SIM115
        self._start = member.offset
        self._size = member.size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unknown whence: {whence}")
        if position < 0:
            raise OSError("Cannot seek before the tar member")
        self._position = min(position, self._size)
        return self._position

    def readinto(self, buffer: Any) -> int:
        if self.closed or self._position >= self._size:
            return 0
        view = memoryview(buffer).cast("B")
        count = min(len(view), self._size - self._position)
        self._file.seek(self._start + self._position)
        data = self._file.read(count)
        view[: len(data)] = data
        self._position += len(data)
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if self.closed or self._position >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        size = min(size, self._size - self._position)
        self._file.seek(self._start + self._position)
        data = self._file.read(size)
        self._position += len(data)
        return data

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()


def physical_core_count() -> int:
    """Return physical cores when Linux topology is available, else a conservative estimate."""
    topology = Path("/sys/devices/system/cpu")
    cores: set[tuple[str, str]] = set()
    for cpu in topology.glob("cpu[0-9]*"):
        try:
            package = (cpu / "topology/physical_package_id").read_text().strip()
            core = (cpu / "topology/core_id").read_text().strip()
        except OSError:
            continue
        cores.add((package, core))
    if cores:
        return len(cores)
    return max(1, (os.cpu_count() or 1) // 2)


def nvidia_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    return sum(bool(line.strip()) for line in result.stdout.splitlines())


def _connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    previous = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
    if previous is not None:
        schema_row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if schema_row is None or int(schema_row[0]) != _INDEX_SCHEMA_VERSION:
            LOGGER.warning("Rebuilding derived tar index %s for schema version %d", path, _INDEX_SCHEMA_VERSION)
            connection.executescript(
                """
                DROP TABLE IF EXISTS observation_members;
                DROP TABLE IF EXISTS proprio_members;
                DROP TABLE IF EXISTS archives;
                DROP TABLE IF EXISTS metadata;
                """
            )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS archives (
            path TEXT PRIMARY KEY, kind TEXT NOT NULL, size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL, complete INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS observation_members (
            task_id INTEGER NOT NULL, episode_id INTEGER NOT NULL, camera TEXT NOT NULL,
            archive TEXT NOT NULL, name TEXT NOT NULL, offset INTEGER NOT NULL, size INTEGER NOT NULL,
            PRIMARY KEY (archive, task_id, episode_id, camera)
        );
        CREATE INDEX IF NOT EXISTS observation_archive_idx ON observation_members(archive);
        CREATE TABLE IF NOT EXISTS proprio_members (
            task_id INTEGER NOT NULL, episode_id INTEGER NOT NULL,
            archive TEXT NOT NULL, name TEXT NOT NULL, offset INTEGER NOT NULL, size INTEGER NOT NULL,
            PRIMARY KEY (archive, task_id, episode_id)
        );
        CREATE INDEX IF NOT EXISTS proprio_archive_idx ON proprio_members(archive);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)", (str(_INDEX_SCHEMA_VERSION),)
    )
    connection.commit()
    return connection


def _normalize_member_name(name: str) -> str:
    return name.removeprefix("./")


def _scan_archive(request: tuple[str, str, int | None]) -> ArchiveScan:
    archive, kind, task_hint = request
    path = Path(archive)
    stat = path.stat()
    observations: list[tuple[int, int, str, str, int, int]] = []
    proprio: list[tuple[int, int, str, int, int]] = []
    wanted_files = {filename: camera for camera, filename in CAMERA_FILES.items()}
    with tarfile.open(path, mode="r:") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = _normalize_member_name(member.name)
            parts = PurePosixPath(name).parts
            if kind == "observation":
                if len(parts) != 3 or parts[1] != "videos" or parts[2] not in wanted_files:
                    continue
                try:
                    episode_id = int(parts[0])
                except ValueError:
                    continue
                assert task_hint is not None
                observations.append(
                    (task_hint, episode_id, wanted_files[parts[2]], name, member.offset_data, member.size)
                )
            elif kind == "proprio":
                if len(parts) != 3 or parts[2] != "proprio_stats.h5":
                    continue
                try:
                    task_id, episode_id = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                proprio.append((task_id, episode_id, name, member.offset_data, member.size))
            else:
                raise ValueError(f"Unknown archive kind: {kind}")
    return ArchiveScan(
        archive=str(path.resolve()),
        kind=kind,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        observations=tuple(observations),
        proprio=tuple(proprio),
    )


def _archive_is_current(connection: sqlite3.Connection, path: Path) -> bool:
    resolved = str(path.resolve())
    stat = path.stat()
    row = connection.execute("SELECT size, mtime_ns, complete FROM archives WHERE path=?", (resolved,)).fetchone()
    return row == (stat.st_size, stat.st_mtime_ns, 1)


def _store_archive_scan(connection: sqlite3.Connection, scan: ArchiveScan) -> None:
    with connection:
        connection.execute("DELETE FROM observation_members WHERE archive=?", (scan.archive,))
        connection.execute("DELETE FROM proprio_members WHERE archive=?", (scan.archive,))
        connection.execute(
            "INSERT OR REPLACE INTO archives(path, kind, size, mtime_ns, complete) VALUES(?, ?, ?, ?, 0)",
            (scan.archive, scan.kind, scan.size, scan.mtime_ns),
        )
        connection.executemany(
            """
            INSERT INTO observation_members(task_id, episode_id, camera, archive, name, offset, size)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(archive, task_id, episode_id, camera) DO UPDATE SET
              name=excluded.name, offset=excluded.offset, size=excluded.size
            """,
            ((*row[:3], scan.archive, *row[3:]) for row in scan.observations),
        )
        connection.executemany(
            """
            INSERT INTO proprio_members(task_id, episode_id, archive, name, offset, size)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(archive, task_id, episode_id) DO UPDATE SET
              name=excluded.name, offset=excluded.offset, size=excluded.size
            """,
            ((row[0], row[1], scan.archive, *row[2:]) for row in scan.proprio),
        )
        connection.execute("UPDATE archives SET complete=1 WHERE path=?", (scan.archive,))


def _index_archives(
    connection: sqlite3.Connection,
    requests: Sequence[tuple[str, str, int | None]],
    *,
    workers: int,
) -> None:
    kind = requests[0][1] if requests else "tar"
    LOGGER.info("Checking persistent index state for %d %s archives", len(requests), kind)
    pending = [
        request
        for request in tqdm.tqdm(requests, desc=f"Checking {kind} index", unit="tar", dynamic_ncols=True)
        if not _archive_is_current(connection, Path(request[0]))
    ]
    if not pending:
        LOGGER.info("Tar index is current for all %d requested archives", len(requests))
        return
    LOGGER.info("Indexing %d/%d tar archives with %d workers", len(pending), len(requests), workers)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_scan_archive, request): request for request in pending}
        total_members = 0
        with tqdm.tqdm(
            total=len(pending), desc=f"Indexing {kind} archives", unit="tar", dynamic_ncols=True
        ) as progress:
            for future in concurrent.futures.as_completed(futures):
                scan = future.result()
                _store_archive_scan(connection, scan)
                total_members += len(scan.observations) + len(scan.proprio)
                progress.set_postfix(archive=Path(scan.archive).name, members=total_members, refresh=False)
                progress.update()
    LOGGER.info("Indexed %d %s archives with %d selected members", len(pending), kind, total_members)


def _observation_archives(config: ConverterConfig) -> tuple[Path, ...]:
    if config.observation_archives:
        archives = tuple(path.resolve() for path in config.observation_archives)
    else:
        task_dirs = (
            [config.input_root / "observations" / str(task_id) for task_id in config.task_ids]
            if config.task_ids
            else sorted((config.input_root / "observations").glob("*"))
        )
        archives = tuple(path.resolve() for task_dir in task_dirs for path in sorted(task_dir.glob("*.tar")))
    missing = [path for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Observation archives not found: {missing[:5]}")
    if not archives:
        raise FileNotFoundError("No observation tar archives matched the requested selection")
    for path in archives:
        try:
            int(path.parent.name)
        except ValueError as exc:
            raise ValueError(f"Cannot infer task ID from observation archive path: {path}") from exc
    return archives


def _episode_ids_in_observation_archives(
    connection: sqlite3.Connection,
    archives: Sequence[Path],
    selected_episode_ids: set[int],
) -> list[int]:
    episode_ids: set[int] = set()
    for archive in tqdm.tqdm(
        archives,
        desc="Collecting observation episodes",
        unit="tar",
        dynamic_ncols=True,
    ):
        rows = connection.execute(
            "SELECT DISTINCT episode_id FROM observation_members WHERE archive=?", (str(archive.resolve()),)
        )
        episode_ids.update(row[0] for row in rows if not selected_episode_ids or row[0] in selected_episode_ids)
    return sorted(episode_ids)


def _proprio_archives_for_episodes(input_root: Path, episode_ids: Sequence[int]) -> tuple[Path, ...]:
    archives: list[Path] = []
    for path in sorted((input_root / "proprio_stats").glob("*.tar")):
        match = _ARCHIVE_RANGE.fullmatch(path.name)
        if match is None:
            LOGGER.warning("Ignoring proprio archive with an unrecognized range name: %s", path)
            continue
        start, end = int(match["start"]), int(match["end"])
        index = bisect_left(episode_ids, start)
        if index < len(episode_ids) and episode_ids[index] <= end:
            archives.append(path.resolve())
    return tuple(archives)


def build_tar_index(config: ConverterConfig) -> tuple[Path, tuple[Path, ...]]:
    """Index selected observation archives and the minimum required proprio archives."""
    index_path = config.state_dir / "agibotworld_beta_index.sqlite3"
    connection = _connect_index(index_path)
    try:
        observation_archives = _observation_archives(config)
        index_workers = config.index_workers or min(16, max(1, physical_core_count() // 4))
        observation_requests = [(str(path), "observation", int(path.parent.name)) for path in observation_archives]
        _index_archives(connection, observation_requests, workers=index_workers)
        episode_ids = _episode_ids_in_observation_archives(connection, observation_archives, set(config.episode_ids))
        proprio_archives = _proprio_archives_for_episodes(config.input_root, episode_ids)
        if episode_ids and not proprio_archives:
            raise FileNotFoundError("No proprio_stats archive range covers the selected observation episodes")
        proprio_requests = [(str(path), "proprio", None) for path in proprio_archives]
        _index_archives(connection, proprio_requests, workers=min(index_workers, max(1, len(proprio_requests))))
    finally:
        connection.close()
    return index_path, observation_archives


def _load_annotations(input_root: Path, task_ids: set[int]) -> dict[tuple[int, int], EpisodeAnnotation]:
    annotations: dict[tuple[int, int], EpisodeAnnotation] = {}
    for task_id in sorted(task_ids):
        path = input_root / "task_info" / f"task_{task_id}.json"
        if not path.is_file():
            LOGGER.warning("Task annotation file not found: %s", path)
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of episodes in {path}")
        for item in payload:
            episode_id = int(item["episode_id"])
            segments = []
            for segment in item.get("label_info", {}).get("action_config", []):
                text = str(segment.get("action_text", "")).strip()
                if text:
                    segments.append((int(segment["start_frame"]), int(segment["end_frame"]), text))
            annotations[(task_id, episode_id)] = EpisodeAnnotation(
                task_name=str(item.get("task_name", f"task_{task_id}")),
                init_scene_text=str(item.get("init_scene_text", "")),
                action_segments=tuple(segments),
            )
    return annotations


def load_episode_plan(
    config: ConverterConfig,
    index_path: Path,
    observation_archives: Sequence[Path],
) -> tuple[EpisodeRef, ...]:
    connection = _connect_index(index_path)
    selected_episode_ids = set(config.episode_ids)
    cameras: dict[tuple[int, int], dict[str, MemberRef]] = {}
    try:
        for archive in tqdm.tqdm(
            observation_archives,
            desc="Loading observation members",
            unit="tar",
            dynamic_ncols=True,
        ):
            rows = connection.execute(
                """
                SELECT task_id, episode_id, camera, archive, name, offset, size
                FROM observation_members WHERE archive=?
                """,
                (str(archive.resolve()),),
            )
            for task_id, episode_id, camera, source, name, offset, size in rows:
                if selected_episode_ids and episode_id not in selected_episode_ids:
                    continue
                cameras.setdefault((task_id, episode_id), {}).setdefault(camera, MemberRef(source, name, offset, size))
        task_ids = {key[0] for key in cameras}
        annotations = _load_annotations(config.input_root, task_ids)
        episodes: list[EpisodeRef] = []
        missing_cameras = missing_proprio = missing_annotation = 0
        for key in tqdm.tqdm(
            sorted(cameras),
            desc="Joining episode metadata",
            unit="episode",
            dynamic_ncols=True,
        ):
            camera_refs = cameras[key]
            if set(camera_refs) != set(CAMERA_FILES):
                missing_cameras += 1
                continue
            row = connection.execute(
                """
                SELECT archive, name, offset, size FROM proprio_members
                WHERE task_id=? AND episode_id=?
                ORDER BY archive LIMIT 1
                """,
                key,
            ).fetchone()
            if row is None:
                missing_proprio += 1
                continue
            annotation = annotations.get(key)
            if annotation is None:
                missing_annotation += 1
                continue
            episodes.append(
                EpisodeRef(
                    task_id=key[0],
                    episode_id=key[1],
                    cameras=camera_refs,
                    proprio=MemberRef(*row),
                    annotation=annotation,
                )
            )
    finally:
        connection.close()
    LOGGER.info(
        "Eligible episodes=%d; skipped incomplete cameras=%d, missing proprio=%d, missing annotation=%d",
        len(episodes),
        missing_cameras,
        missing_proprio,
        missing_annotation,
    )
    if config.max_episodes is not None:
        episodes = episodes[: config.max_episodes]
    if not episodes:
        raise RuntimeError("No complete AgiBotWorld Beta episodes matched the selection")
    return tuple(episodes)


def _read_member(member: MemberRef) -> bytes:
    with TarMemberIO(member) as stream:
        payload = stream.read()
    if len(payload) != member.size:
        raise EOFError(f"Short read for {member.archive}:{member.name}: {len(payload)} != {member.size}")
    return payload


def _load_proprio(member: MemberRef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(io.BytesIO(_read_member(member)), "r") as file:
        state_parts = [np.asarray(file[path], dtype=np.float32) for path in STATE_PATHS]
        action_parts = [np.asarray(file[path], dtype=np.float32) for path in ACTION_PATHS]
        timestamps = np.asarray(file["timestamp"], dtype=np.int64)
    state = np.concatenate(state_parts, axis=-1)
    action = np.concatenate(action_parts, axis=-1)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"Expected state [T,{STATE_DIM}], got {state.shape}")
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action [T,{ACTION_DIM}], got {action.shape}")
    lengths = {len(state), len(action), len(timestamps)}
    if len(lengths) != 1:
        raise ValueError(
            f"Proprio trajectory lengths differ: state={len(state)}, action={len(action)}, timestamps={len(timestamps)}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action trajectory contains NaN or infinity")
    return state, action, timestamps


def _codec_frames(member: MemberRef, decoder: Decoder, *, threads: int) -> Iterator[np.ndarray]:
    import av

    codec_name = "av1_cuvid" if decoder == Decoder.NVIDIA else "libdav1d"
    with TarMemberIO(member) as stream, av.open(stream, mode="r", format="mp4") as container:
        source_stream = container.streams.video[0]
        codec = av.CodecContext.create(codec_name, "r")
        codec.extradata = source_stream.codec_context.extradata
        codec.thread_count = threads
        for packet in container.demux(source_stream):
            for frame in codec.decode(packet):
                yield frame.to_ndarray(format="rgb24")
        try:
            flushed = codec.decode(None)
        except av.error.EOFError:
            flushed = ()
        for frame in flushed:
            yield frame.to_ndarray(format="rgb24")


def _resize_and_encode(image: np.ndarray, height: int, width: int, quality: int) -> bytes:
    import cv2

    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("OpenCV failed to encode a JPEG frame")
    return encoded.tobytes()


def _decode_camera(
    member: MemberRef,
    decoder: Decoder,
    *,
    threads: int,
    expected_frames: int,
    tolerance: int,
    image_height: int,
    image_width: int,
    jpeg_quality: int,
) -> tuple[bytes, ...]:
    encoded: list[bytes] = []
    limit = expected_frames + tolerance + 1
    for image in _codec_frames(member, decoder, threads=threads):
        encoded.append(_resize_and_encode(image, image_height, image_width, jpeg_quality))
        if len(encoded) >= limit:
            break
    return tuple(encoded)


def make_tfds_features(config: ConverterConfig) -> Any:
    import tensorflow_datasets as tfds

    image = lambda: tfds.features.Image(  # noqa: E731
        shape=(config.image_height, config.image_width, 3), encoding_format="jpeg"
    )
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": {
                        "base_0_rgb": image(),
                        "left_wrist_0_rgb": image(),
                        "right_wrist_0_rgb": image(),
                        "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32),
                    },
                    "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "timestamp_ns": np.int64,
                    "is_first": np.bool_,
                    "is_last": np.bool_,
                    "is_terminal": np.bool_,
                    "reward": np.float32,
                    "discount": np.float32,
                }
            ),
            "episode_metadata": {
                "episode_id": np.int64,
                "task_id": np.int64,
                "task_name": tfds.features.Text(),
                "init_scene_text": tfds.features.Text(),
                "source_observation_archive": tfds.features.Text(),
                "source_proprio_archive": tfds.features.Text(),
            },
        }
    )


def _serialize_episode(
    episode: EpisodeRef, config: ConverterConfig, decoder: Decoder, features: Any
) -> tuple[bytes, int]:
    state, action, timestamps = _load_proprio(episode.proprio)
    expected_frames = len(state)
    images = {
        key: _decode_camera(
            member,
            decoder,
            threads=config.decoder_threads,
            expected_frames=expected_frames,
            tolerance=config.alignment_tolerance,
            image_height=config.image_height,
            image_width=config.image_width,
            jpeg_quality=config.jpeg_quality,
        )
        for key, member in episode.cameras.items()
    }
    lengths = {"state": expected_frames, **{key: len(value) for key, value in images.items()}}
    shortest, longest = min(lengths.values()), max(lengths.values())
    if longest - shortest > config.alignment_tolerance:
        raise ValueError(f"Episode {episode.key} frame lengths exceed tolerance: {lengths}")
    length = shortest
    if length <= 0:
        raise ValueError(f"Episode {episode.key} is empty")
    prompts = episode.annotation.prompts(length)
    steps = []
    for index in range(length):
        is_last = index == length - 1
        steps.append(
            {
                "observation": {
                    **{key: value[index] for key, value in images.items()},
                    "state": state[index],
                },
                "action": action[index],
                "language_instruction": prompts[index],
                "timestamp_ns": timestamps[index],
                "is_first": index == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "reward": np.float32(0.0),
                "discount": np.float32(0.0 if is_last else 1.0),
            }
        )
    observation_archives = sorted({member.archive for member in episode.cameras.values()})
    payload = {
        "steps": steps,
        "episode_metadata": {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "task_name": episode.annotation.task_name,
            "init_scene_text": episode.annotation.init_scene_text,
            "source_observation_archive": episode.source_observation_archive or ",".join(observation_archives),
            "source_proprio_archive": episode.source_proprio_archive or episode.proprio.archive,
        },
    }
    return features.serialize_example(payload), length


def _shard_filename(job: ShardJob) -> str:
    return f"{job.config.dataset_name}-train.tfrecord-{job.shard_index:05d}-of-{job.shard_count:05d}"


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _worker_initialize(decoder: Decoder, gpu_count: int) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if decoder == Decoder.NVIDIA and gpu_count:
        identity = multiprocessing.current_process()._identity  # noqa: SLF001
        worker_index = (identity[0] - 1) if identity else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index % gpu_count)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2

    cv2.setNumThreads(1)


def _write_shard(job: ShardJob) -> ShardResult:
    import tensorflow as tf

    with contextlib.suppress(RuntimeError):
        tf.config.set_visible_devices([], "GPU")
    start = time.monotonic()
    output_dir = job.config.version_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / _shard_filename(job)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    sidecar = destination.with_name(f"{destination.name}.conversion.json")
    features = make_tfds_features(job.config)
    episode_keys: list[str] = []
    errors: list[str] = []
    frames = 0
    try:
        with tf.io.TFRecordWriter(str(partial)) as writer:
            for episode in job.episodes:
                try:
                    serialized, episode_frames = _serialize_episode(episode, job.config, job.decoder, features)
                except Exception as exc:
                    message = f"{episode.key}: {type(exc).__name__}: {exc}"
                    if not job.config.skip_bad_episodes:
                        raise RuntimeError(message) from exc
                    errors.append(message)
                    continue
                writer.write(serialized)
                episode_keys.append(episode.key)
                frames += episode_frames
        if not episode_keys:
            raise RuntimeError(f"Shard {job.shard_index} contains no valid episodes")
        os.replace(partial, destination)
        result = ShardResult(
            shard_index=job.shard_index,
            path=str(destination),
            episode_keys=tuple(episode_keys),
            frames=frames,
            bytes_written=destination.stat().st_size,
            elapsed_seconds=time.monotonic() - start,
            errors=tuple(errors),
        )
        _atomic_write_json(
            sidecar,
            {
                **dataclasses.asdict(result),
                "plan_fingerprint": job.plan_fingerprint,
                "expected_episode_keys": [episode.key for episode in job.episodes],
                "decoder": job.decoder.value,
            },
        )
        return result
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _existing_shard(job: ShardJob) -> ShardResult | None:
    path = job.config.version_dir / _shard_filename(job)
    sidecar = path.with_name(f"{path.name}.conversion.json")
    if not path.is_file() or not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text())
    if payload.get("plan_fingerprint") != job.plan_fingerprint:
        return None
    if payload.get("decoder") != job.decoder.value:
        return None
    if payload.get("expected_episode_keys") != [episode.key for episode in job.episodes]:
        return None
    if int(payload.get("bytes_written", -1)) != path.stat().st_size:
        return None
    return ShardResult(
        shard_index=job.shard_index,
        path=str(path),
        episode_keys=tuple(payload["episode_keys"]),
        frames=int(payload["frames"]),
        bytes_written=int(payload["bytes_written"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        errors=tuple(payload["errors"]),
    )


def _config_payload(config: ConverterConfig) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_name": config.dataset_name,
        "version": config.version,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "jpeg_quality": config.jpeg_quality,
        "episodes_per_shard": config.episodes_per_shard,
        "decoder_threads": config.decoder_threads,
        "alignment_tolerance": config.alignment_tolerance,
        "skip_bad_episodes": config.skip_bad_episodes,
        "camera_files": CAMERA_FILES,
        "state_paths": STATE_PATHS,
        "action_paths": ACTION_PATHS,
    }


def plan_fingerprint(config: ConverterConfig, episodes: Sequence[EpisodeRef]) -> str:
    digest = hashlib.sha256(json.dumps(_config_payload(config), sort_keys=True).encode())
    for episode in episodes:
        digest.update(b"\0")
        digest.update(episode.key.encode())
        if episode.source_observation_archive is not None:
            digest.update(f"\0source_observation_archive\0{episode.source_observation_archive}".encode())
        if episode.source_proprio_archive is not None:
            digest.update(f"\0source_proprio_archive\0{episode.source_proprio_archive}".encode())
        for key in sorted(episode.cameras):
            member = episode.cameras[key]
            digest.update(f"\0{key}\0{member.archive}\0{member.offset}\0{member.size}".encode())
        digest.update(
            f"\0proprio\0{episode.proprio.archive}\0{episode.proprio.offset}\0{episode.proprio.size}".encode()
        )
    return digest.hexdigest()


def _benchmark_member(member: MemberRef, decoder: Decoder, threads: int) -> tuple[int, float]:
    start = time.monotonic()
    frames = sum(1 for _ in _codec_frames(member, decoder, threads=threads))
    return frames, time.monotonic() - start


def select_decoder(config: ConverterConfig, episode: EpisodeRef, *, available_jobs: int | None = None) -> Decoder:
    if config.decoder != Decoder.AUTO:
        return config.decoder
    cpu_frames, cpu_seconds = _benchmark_member(episode.cameras["base_0_rgb"], Decoder.CPU, config.decoder_threads)
    gpu_count = nvidia_gpu_count()
    if not gpu_count:
        LOGGER.info("Decoder benchmark: CPU %.2fs/%d frames; no NVIDIA GPU decoder available", cpu_seconds, cpu_frames)
        return Decoder.CPU
    try:
        gpu_frames, gpu_seconds = _benchmark_member(
            episode.cameras["base_0_rgb"], Decoder.NVIDIA, config.decoder_threads
        )
    except Exception as exc:
        LOGGER.warning("NVIDIA AV1 decoder probe failed; using CPU: %s", exc)
        return Decoder.CPU
    if gpu_frames != cpu_frames:
        LOGGER.warning("NVIDIA decoded %d frames but CPU decoded %d; using CPU", gpu_frames, cpu_frames)
        return Decoder.CPU
    cpu_workers = config.workers or physical_core_count()
    gpu_workers = config.workers or gpu_count * config.gpu_workers_per_device
    if available_jobs is not None:
        cpu_workers = min(cpu_workers, available_jobs)
        gpu_workers = min(gpu_workers, available_jobs)
    cpu_throughput = cpu_workers / cpu_seconds
    gpu_throughput = gpu_workers / gpu_seconds
    selected = Decoder.NVIDIA if gpu_throughput > cpu_throughput * 1.05 else Decoder.CPU
    LOGGER.info(
        "Decoder benchmark: CPU=%.2fs x %d workers (%.2f streams/s), "
        "NVIDIA=%.2fs x %d workers (%.2f streams/s) for %d frames; selected %s",
        cpu_seconds,
        cpu_workers,
        cpu_throughput,
        gpu_seconds,
        gpu_workers,
        gpu_throughput,
        cpu_frames,
        selected.value,
    )
    return selected


def _make_jobs(
    config: ConverterConfig,
    episodes: Sequence[EpisodeRef],
    decoder: Decoder,
    fingerprint: str,
) -> tuple[ShardJob, ...]:
    shard_count = math.ceil(len(episodes) / config.episodes_per_shard)
    return tuple(
        ShardJob(
            shard_index=index,
            shard_count=shard_count,
            episodes=tuple(episodes[index * config.episodes_per_shard : (index + 1) * config.episodes_per_shard]),
            config=config,
            decoder=decoder,
            plan_fingerprint=fingerprint,
        )
        for index in range(shard_count)
    )


def _write_plan(config: ConverterConfig, episodes: Sequence[EpisodeRef], fingerprint: str, decoder: Decoder) -> None:
    config.version_dir.mkdir(parents=True, exist_ok=True)
    path = config.version_dir / "conversion_plan.json"
    payload = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "plan_fingerprint": fingerprint,
        "episode_count": len(episodes),
        "first_episode": episodes[0].key,
        "last_episode": episodes[-1].key,
        "decoder": decoder.value,
        "config": _config_payload(config),
    }
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous.get("plan_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing conversion plan at {path} does not match this run. "
                "Use a different dataset_name/version to preserve the existing output."
            )
        return
    _atomic_write_json(path, payload)


def _resolve_decoder(
    config: ConverterConfig,
    episode: EpisodeRef,
    fingerprint: str,
    *,
    available_jobs: int,
) -> Decoder:
    """Keep auto-selected decoding stable across interrupted conversion runs."""
    plan_path = config.version_dir / "conversion_plan.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text())
        if previous.get("plan_fingerprint") == fingerprint:
            previous_decoder = Decoder(previous["decoder"])
            if config.decoder not in {Decoder.AUTO, previous_decoder}:
                raise RuntimeError(
                    f"Existing plan uses decoder {previous_decoder.value!r}, but this run forces "
                    f"{config.decoder.value!r}; use a different dataset_name/version"
                )
            LOGGER.info("Reusing decoder %s from the existing conversion plan", previous_decoder.value)
            return previous_decoder
    return select_decoder(config, episode, available_jobs=available_jobs)


def _auto_worker_count(config: ConverterConfig, decoder: Decoder, gpu_count: int) -> int:
    if config.workers:
        return config.workers
    if decoder == Decoder.NVIDIA:
        return max(1, gpu_count * config.gpu_workers_per_device)
    return physical_core_count()


def convert_shards(
    config: ConverterConfig,
    episodes: Sequence[EpisodeRef],
    decoder: Decoder,
    fingerprint: str,
) -> tuple[ShardResult, ...]:
    jobs = _make_jobs(config, episodes, decoder, fingerprint)
    results: dict[int, ShardResult] = {}
    pending: list[ShardJob] = []
    for job in tqdm.tqdm(jobs, desc="Checking RLDS shards", unit="shard", dynamic_ncols=True):
        existing = _existing_shard(job)
        if existing is None:
            pending.append(job)
        else:
            results[job.shard_index] = existing
    if not pending:
        LOGGER.info("All %d shards already exist and match the conversion plan", len(jobs))
        return tuple(results[index] for index in range(len(jobs)))
    gpu_count = nvidia_gpu_count() if decoder == Decoder.NVIDIA else 0
    workers = min(len(pending), _auto_worker_count(config, decoder, gpu_count))
    LOGGER.info(
        "Converting %d pending shards (%d total) with %d workers using %s",
        len(pending),
        len(jobs),
        workers,
        decoder.value,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_initialize,
        initargs=(decoder, gpu_count),
    ) as executor:
        futures = {executor.submit(_write_shard, job): job for job in pending}
        completed_episodes = completed_frames = completed_bytes = 0
        with tqdm.tqdm(
            total=len(pending),
            desc="Converting RLDS shards",
            unit="shard",
            dynamic_ncols=True,
        ) as progress:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results[result.shard_index] = result
                completed_episodes += len(result.episode_keys)
                completed_frames += result.frames
                completed_bytes += result.bytes_written
                progress.set_postfix(
                    episodes=completed_episodes,
                    frames=completed_frames,
                    gib=f"{completed_bytes / 2**30:.2f}",
                    refresh=False,
                )
                progress.update()
    LOGGER.info(
        "Converted %d shards: episodes=%d frames=%d size=%.2f GiB",
        len(pending),
        completed_episodes,
        completed_frames,
        completed_bytes / 2**30,
    )
    return tuple(results[index] for index in range(len(jobs)))


def finalize_dataset(config: ConverterConfig, results: Sequence[ShardResult], fingerprint: str) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    manifest_path = config.version_dir / "conversion_manifest.json"
    expected_episodes = sum(len(result.episode_keys) for result in results)
    expected_frames = sum(result.frames for result in results)
    expected_bytes = sum(result.bytes_written for result in results)
    if manifest_path.is_file() and (config.version_dir / "dataset_info.json").is_file():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("status") == "complete"
            and previous.get("plan_fingerprint") == fingerprint
            and previous.get("episodes") == expected_episodes
            and previous.get("frames") == expected_frames
            and previous.get("bytes") == expected_bytes
            and previous.get("shards") == len(results)
        ):
            builder = tfds.builder_from_directory(config.version_dir)
            if builder.info.splits["train"].num_examples == expected_episodes:
                LOGGER.info("Reusing verified final manifest and TFDS metadata")
                return previous
    tfds.folder_dataset.write_metadata(
        data_dir=config.version_dir,
        features=make_tfds_features(config),
        version=config.version,
        check_data=True,
        description=(
            "AgiBotWorld Beta converted directly from source WebDataset tar archives to sharded TFDS/RLDS. "
            "State is 20-D joint/effector/head/waist position; action is 22-D "
            "joint/effector/head/waist position plus base velocity."
        ),
        homepage="https://github.com/OpenDriveLab/AgiBot-World",
    )
    builder = tfds.builder_from_directory(config.version_dir)
    if builder.info.splits["train"].num_examples != expected_episodes:
        raise RuntimeError(
            f"TFDS metadata reports {builder.info.splits['train'].num_examples} episodes, expected {expected_episodes}"
        )
    manifest = {
        "status": "complete",
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "plan_fingerprint": fingerprint,
        "dataset_name": config.dataset_name,
        "version": config.version,
        "episodes": expected_episodes,
        "frames": expected_frames,
        "shards": len(results),
        "bytes": expected_bytes,
        "errors": [error for result in results for error in result.errors],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _deletable_archives(
    index_path: Path,
    results: Sequence[ShardResult],
    episodes: Sequence[EpisodeRef],
) -> tuple[Path, ...]:
    converted = {key for result in results for key in result.episode_keys}
    used_archives = {member.archive for episode in episodes for member in (*episode.cameras.values(), episode.proprio)}
    connection = _connect_index(index_path)
    deletable: list[Path] = []
    try:
        rows = connection.execute("SELECT path, kind FROM archives WHERE complete=1")
        for archive, kind in rows:
            if archive not in used_archives:
                continue
            table = "observation_members" if kind == "observation" else "proprio_members"
            keys = {
                f"{task_id}/{episode_id}"
                for task_id, episode_id in connection.execute(
                    f"SELECT DISTINCT task_id, episode_id FROM {table} WHERE archive=?",
                    (archive,),
                )
            }
            if keys and keys <= converted:
                deletable.append(Path(archive))
    finally:
        connection.close()
    return tuple(sorted(deletable))


def _handle_source_archives(
    config: ConverterConfig,
    index_path: Path,
    results: Sequence[ShardResult],
    episodes: Sequence[EpisodeRef],
) -> tuple[Path, ...]:
    deletable = _deletable_archives(index_path, results, episodes)
    list_path = config.version_dir / "verified_deletable_source_archives.txt"
    list_path.write_text("".join(f"{path}\n" for path in deletable))
    if config.delete_source_archives_after_success:
        for path in deletable:
            path.unlink()
        LOGGER.warning("Deleted %d fully converted and verified source tar archives", len(deletable))
    else:
        LOGGER.info("Retained source archives; %d fully covered archives are listed in %s", len(deletable), list_path)
    return deletable


def run_conversion(config: ConverterConfig) -> dict[str, Any] | None:
    """Build/resume an indexed, parallel and validated AgiBotWorld Beta conversion."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    config.validate()
    config.output_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Phase 1/5: build or resume persistent tar indexes")
    index_path, observation_archives = build_tar_index(config)
    if config.index_only:
        LOGGER.info("Index-only run completed: %s", index_path)
        return None
    LOGGER.info("Phase 2/5: assemble and validate the episode plan")
    episodes = load_episode_plan(config, index_path, observation_archives)
    fingerprint = plan_fingerprint(config, episodes)
    shard_count = math.ceil(len(episodes) / config.episodes_per_shard)
    LOGGER.info("Phase 3/5: select the AV1 decoder")
    decoder = _resolve_decoder(config, episodes[0], fingerprint, available_jobs=shard_count)
    _write_plan(config, episodes, fingerprint, decoder)
    LOGGER.info(
        "Conversion plan: episodes=%d shards=%d fingerprint=%s output=%s",
        len(episodes),
        shard_count,
        fingerprint[:12],
        config.version_dir,
    )
    if config.plan_only:
        return None
    LOGGER.info("Phase 4/5: convert episodes into atomic RLDS shards")
    results = convert_shards(config, episodes, decoder, fingerprint)
    LOGGER.info("Phase 5/5: write TFDS metadata and verify the output")
    manifest = finalize_dataset(config, results, fingerprint)
    _handle_source_archives(config, index_path, results, episodes)
    return manifest
