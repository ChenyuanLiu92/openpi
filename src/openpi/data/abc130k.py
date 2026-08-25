"""Streaming ABC-130K MCAP to sharded TFDS/RLDS conversion.

Raw MCAP files are copied sequentially from BOS into a bounded PFS spool. CPU
workers decode and causally align each episode at a fixed clock, write TFRecord
fragments, and only then remove the corresponding spool files. Source MCAPs are
always read-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import concurrent.futures
import contextlib
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from typing import Any

import numpy as np
import tqdm_loggable.auto as tqdm

LOGGER = logging.getLogger(__name__)

STATE_DIM = 14
ACTION_DIM = 14
STATE_TOPICS = (
    ("/left-arm-state", 6),
    ("/left-ee-state", 1),
    ("/right-arm-state", 6),
    ("/right-ee-state", 1),
)
ACTION_TOPICS = (
    ("/left-arm-action", 6),
    ("/left-ee-action", 1),
    ("/right-arm-action", 6),
    ("/right-ee-action", 1),
)
WRIST_TOPICS = {
    "left_wrist_0_rgb": "/left-wrist-camera",
    "right_wrist_0_rgb": "/right-wrist-camera",
}
TOP_TOPICS = ("/top-left-camera", "/top-right-camera", "/top-camera")
_COPY_BUFFER_SIZE = 16 * 2**20
_STATE_SCHEMA_VERSION = 2
_EPISODE_PATTERN = re.compile(r"^episode_(?P<id>.+)$")
_MCAP_MAGIC = b"\x89MCAP0\r\n"


@dataclasses.dataclass(frozen=True)
class ConversionConfig:
    input_root: Path = Path("/mnt/bos/dataset/abc-130k")
    pfs_work_root: Path = Path("/mnt/pfs/rhos-vla/chenyuan/abc-130k-rlds-work")
    output_root: Path = Path("/mnt/bos/dataset/RLDS")
    dataset_name: str = "abc_130k"
    version: str = "1.0.0"
    splits: tuple[str, ...] = ("train", "val")
    task_names: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()
    episode_paths: tuple[Path, ...] = ()
    max_episodes: int | None = None
    episodes_per_shard: int = 8
    discovery_workers: int = 32
    stream_readers: int = 4
    convert_workers: int = 0
    spool_limit_gib: int = 512
    target_fps: int = 30
    image_height: int = 224
    image_width: int = 224
    jpeg_quality: int = 90
    decoder_threads: int = 1
    skip_bad_episodes: bool = False
    keep_spool: bool = False
    publish: bool = True

    def validate(self) -> None:
        if not self.input_root.is_dir():
            raise FileNotFoundError(f"ABC-130K input root not found: {self.input_root}")
        if not re.fullmatch(r"[a-z0-9_]+", self.dataset_name):
            raise ValueError("dataset_name must contain only lowercase letters, digits, and underscores")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("version must have the form MAJOR.MINOR.PATCH")
        if not self.splits or set(self.splits) - {"train", "val"}:
            raise ValueError("splits must contain train and/or val")
        for name, value in {
            "episodes_per_shard": self.episodes_per_shard,
            "discovery_workers": self.discovery_workers,
            "stream_readers": self.stream_readers,
            "spool_limit_gib": self.spool_limit_gib,
            "target_fps": self.target_fps,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "jpeg_quality": self.jpeg_quality,
            "decoder_threads": self.decoder_threads,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.convert_workers < 0:
            raise ValueError("convert_workers must be non-negative")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive or omitted")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.pfs_work_root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.pfs_work_root / "state" / "pipeline.sqlite3"

    @property
    def spool_root(self) -> Path:
        return self.pfs_work_root / "spool"

    @property
    def local_version_dir(self) -> Path:
        return self.pfs_work_root / "output" / self.dataset_name / self.version

    @property
    def bos_version_dir(self) -> Path:
        return self.output_root / self.dataset_name / self.version

    @property
    def semantic_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "input_root": str(self.input_root.resolve()),
            "dataset_name": self.dataset_name,
            "version": self.version,
            "splits": self.splits,
            "task_names": self.task_names,
            "episode_ids": self.episode_ids,
            "episode_paths": tuple(str(path.resolve()) for path in self.episode_paths),
            "max_episodes": self.max_episodes,
            "episodes_per_shard": self.episodes_per_shard,
            "target_fps": self.target_fps,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "jpeg_quality": self.jpeg_quality,
            "skip_bad_episodes": self.skip_bad_episodes,
        }


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    split: str
    task_name: str
    episode_id: str
    source_mcap: Path
    source_annotation: Path | None
    size: int
    annotation_size: int
    mtime_ns: int

    @property
    def key(self) -> str:
        return f"{self.split}/{self.episode_id}"

    @property
    def total_size(self) -> int:
        return self.size + self.annotation_size

    def spool_dir(self, config: ConversionConfig) -> Path:
        return config.spool_root / self.split / f"episode_{self.episode_id}"

    def spool_mcap(self, config: ConversionConfig) -> Path:
        return self.spool_dir(config) / "episode.mcap"

    def spool_annotation(self, config: ConversionConfig) -> Path | None:
        return self.spool_dir(config) / "annotation.mcap" if self.source_annotation is not None else None

    def spool_error(self, config: ConversionConfig) -> Path:
        return self.spool_dir(config) / ".stage_error.json"


@dataclasses.dataclass(frozen=True)
class ShardJob:
    split: str
    shard_index: int
    shard_count: int
    episodes: tuple[EpisodeRef, ...]
    config: ConversionConfig
    plan_fingerprint: str

    @property
    def key(self) -> str:
        return f"{self.split}-{self.shard_index:05d}"


@dataclasses.dataclass(frozen=True)
class ShardResult:
    key: str
    split: str
    shard_index: int
    path: str
    episode_keys: tuple[str, ...]
    frames: int
    bytes_written: int
    sha256: str
    elapsed_seconds: float
    errors: tuple[str, ...]


@dataclasses.dataclass
class _DecodedCamera:
    timestamps: np.ndarray
    jpeg_frames: tuple[bytes, ...]
    codec: str


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def physical_core_count() -> int:
    topology = Path("/sys/devices/system/cpu")
    cores: set[tuple[str, str]] = set()
    for cpu in topology.glob("cpu[0-9]*"):
        try:
            package = (cpu / "topology/physical_package_id").read_text().strip()
            core = (cpu / "topology/core_id").read_text().strip()
        except OSError:
            continue
        cores.add((package, core))
    return len(cores) or max(1, (os.cpu_count() or 1) // 2)


def floor_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    if len(source_timestamps) == 0:
        raise ValueError("Cannot align an empty source stream")
    return np.clip(
        np.searchsorted(source_timestamps, target_timestamps, side="right") - 1, 0, len(source_timestamps) - 1
    )


def deterministic_top_topic(episode_id: str, topics: Iterable[str]) -> str:
    available = set(topics)
    if {"/top-left-camera", "/top-right-camera"} <= available:
        choice = hashlib.sha1(f"episode_{episode_id}".encode()).digest()[0] % 2
        return "/top-left-camera" if choice == 0 else "/top-right-camera"
    if "/top-camera" in available:
        return "/top-camera"
    raise ValueError("Episode has no supported top camera")


def make_tfds_features(config: ConversionConfig) -> Any:
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
                    "subtask_instruction": tfds.features.Text(),
                    "timestamp_ns": np.int64,
                    "is_first": np.bool_,
                    "is_last": np.bool_,
                    "is_terminal": np.bool_,
                    "reward": np.float32,
                    "discount": np.float32,
                }
            ),
            "episode_metadata": {
                "episode_id": tfds.features.Text(),
                "task_name": tfds.features.Text(),
                "source_mcap": tfds.features.Text(),
                "source_annotation": tfds.features.Text(),
                "source_split": tfds.features.Text(),
                "station_type": tfds.features.Text(),
                "annotated": np.bool_,
                "alignment": tfds.features.Text(),
            },
        }
    )


class _CameraDecoder:
    def __init__(self, config: ConversionConfig):
        self._config = config
        self._codec: Any | None = None
        self._codec_name = ""
        self._message_timestamps: list[int] = []
        self._frames: list[bytes] = []

    def push(self, timestamp: int, codec_name: str, data: bytes) -> None:
        import av

        normalized = "hevc" if codec_name.lower() in {"h265", "hevc"} else "h264"
        if self._codec is None:
            self._codec_name = normalized
            self._codec = av.CodecContext.create(normalized, "r")
            self._codec.thread_count = self._config.decoder_threads
            self._codec.thread_type = "FRAME"
        elif normalized != self._codec_name:
            raise ValueError(f"Camera codec changed from {self._codec_name} to {normalized}")
        self._message_timestamps.append(timestamp)
        for packet in self._codec.parse(data):
            for frame in self._codec.decode(packet):
                self._frames.append(_resize_and_encode(frame.to_ndarray(format="rgb24"), self._config))

    def finish(self) -> _DecodedCamera:
        if self._codec is None:
            raise ValueError("Camera stream is empty")
        for packet in self._codec.parse(b""):
            for frame in self._codec.decode(packet):
                self._frames.append(_resize_and_encode(frame.to_ndarray(format="rgb24"), self._config))
        for frame in self._codec.decode(None):
            self._frames.append(_resize_and_encode(frame.to_ndarray(format="rgb24"), self._config))
        if not self._frames:
            raise ValueError("Camera decoder produced no frames")
        source = np.asarray(self._message_timestamps, dtype=np.int64)
        if len(self._frames) != len(source):
            LOGGER.warning(
                "Camera produced %d decoded frames for %d MCAP messages; respacing timestamps",
                len(self._frames),
                len(source),
            )
            source = np.linspace(source[0], source[-1], len(self._frames), dtype=np.int64)
        return _DecodedCamera(source, tuple(self._frames), self._codec_name)


def _resize_and_encode(image: np.ndarray, config: ConversionConfig) -> bytes:
    import cv2

    source_height, source_width = image.shape[:2]
    scale = min(config.image_width / source_width, config.image_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)
    top = (config.image_height - resized_height) // 2
    left = (config.image_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    success, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    )
    if not success:
        raise RuntimeError("OpenCV failed to encode a JPEG frame")
    return encoded.tobytes()


def _read_annotations(path: Path | None) -> tuple[np.ndarray, tuple[str, ...]]:
    if path is None or not path.is_file():
        return np.empty(0, dtype=np.int64), ()
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    annotations: list[tuple[int, str]] = []
    # mcap wraps RawIOBase in a temporary BufferedReader whose finalizer closes
    # the underlying file. Pass an already-buffered stream so seeking remains
    # valid across summary and message iteration.
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _channel, message, decoded in reader.iter_decoded_messages(topics=["/subtask-annotation"]):
            value = str(decoded.data).strip()
            if value:
                annotations.append((message.log_time, value))
    annotations.sort(key=lambda item: item[0])
    return (
        np.asarray([item[0] for item in annotations], dtype=np.int64),
        tuple(item[1] for item in annotations),
    )


def _episode_payload(episode: EpisodeRef, config: ConversionConfig, features: Any) -> tuple[bytes, int]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    mcap_path = episode.spool_mcap(config)
    annotation_path = episode.spool_annotation(config)
    with mcap_path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
        if summary is None:
            raise ValueError("MCAP summary is missing")
        topics = {channel.topic for channel in summary.channels.values()}
    top_topic = deterministic_top_topic(episode.episode_id, topics)
    camera_topics = {
        "base_0_rgb": top_topic,
        **WRIST_TOPICS,
    }
    missing_cameras = set(camera_topics.values()) - topics
    if missing_cameras:
        raise ValueError(f"Missing camera topics: {sorted(missing_cameras)}")

    scalar_topics = dict(STATE_TOPICS + ACTION_TOPICS)
    scalars: dict[str, list[tuple[int, np.ndarray]]] = {topic: [] for topic in scalar_topics}
    cameras = {key: _CameraDecoder(config) for key in camera_topics}
    topic_to_key = {topic: key for key, topic in camera_topics.items()}
    task_name = episode.task_name.replace("_", " ")
    metadata: dict[str, str] = {}
    selected_topics = set(scalar_topics) | set(topic_to_key) | {"/instruction"}
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for record in reader.iter_metadata():
            if record.name in {"episode-metadata", "session-metadata"}:
                metadata.update(record.metadata)
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=selected_topics):
            topic = channel.topic
            if topic in topic_to_key:
                cameras[topic_to_key[topic]].push(message.log_time, str(decoded.format), bytes(decoded.data))
            elif topic in scalar_topics:
                value = np.asarray(decoded.position, dtype=np.float32)
                expected = scalar_topics[topic]
                if value.shape != (expected,):
                    raise ValueError(f"{topic} has shape {value.shape}, expected ({expected},)")
                scalars[topic].append((message.log_time, value))
            elif topic == "/instruction" and str(decoded.data).strip():
                task_name = str(decoded.data).strip()
    task_name = metadata.get("task_name") or metadata.get("instruction") or task_name

    decoded_cameras = {key: decoder.finish() for key, decoder in cameras.items()}
    scalar_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for topic, values in scalars.items():
        if not values:
            raise ValueError(f"Required scalar stream {topic} is empty")
        values.sort(key=lambda item: item[0])
        scalar_arrays[topic] = (
            np.asarray([item[0] for item in values], dtype=np.int64),
            np.stack([item[1] for item in values]),
        )

    starts = [camera.timestamps[0] for camera in decoded_cameras.values()]
    ends = [camera.timestamps[-1] for camera in decoded_cameras.values()]
    starts.extend(values[0][0] for values in scalar_arrays.values())
    ends.extend(values[0][-1] for values in scalar_arrays.values())
    start = max(starts)
    end = min(ends)
    tick_ns = round(1e9 / config.target_fps)
    ticks = np.arange(start + tick_ns, end + 1, tick_ns, dtype=np.int64)
    if len(ticks) < 10:
        raise ValueError(f"Episode overlap is too short: {len(ticks)} frames")

    aligned: list[np.ndarray] = []
    for topic, _ in STATE_TOPICS + ACTION_TOPICS:
        timestamps, values = scalar_arrays[topic]
        aligned.append(values[floor_indices(timestamps, ticks)])
    states = np.concatenate(aligned[: len(STATE_TOPICS)], axis=-1).astype(np.float32, copy=False)
    actions = np.concatenate(aligned[len(STATE_TOPICS) :], axis=-1).astype(np.float32, copy=False)
    if states.shape != (len(ticks), STATE_DIM) or actions.shape != (len(ticks), ACTION_DIM):
        raise RuntimeError(f"Unexpected aligned state/action shapes: {states.shape}, {actions.shape}")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("State/action contains NaN or infinite values")
    camera_indices = {key: floor_indices(camera.timestamps, ticks) for key, camera in decoded_cameras.items()}
    annotation_timestamps, annotation_values = _read_annotations(annotation_path)
    if len(annotation_timestamps):
        annotation_indices = np.searchsorted(annotation_timestamps, ticks, side="right") - 1
    else:
        annotation_indices = np.full(len(ticks), -1, dtype=np.int64)

    steps = []
    for index, timestamp in enumerate(ticks):
        is_last = index == len(ticks) - 1
        annotation_index = int(annotation_indices[index])
        subtask = annotation_values[annotation_index] if annotation_index >= 0 else task_name
        steps.append(
            {
                "observation": {
                    key: camera.jpeg_frames[int(camera_indices[key][index])] for key, camera in decoded_cameras.items()
                }
                | {"state": states[index]},
                "action": actions[index],
                "language_instruction": task_name,
                "subtask_instruction": subtask,
                "timestamp_ns": timestamp,
                "is_first": index == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "reward": np.float32(0.0),
                "discount": np.float32(0.0 if is_last else 1.0),
            }
        )
    payload = {
        "steps": steps,
        "episode_metadata": {
            "episode_id": episode.episode_id,
            "task_name": task_name,
            "source_mcap": str(episode.source_mcap),
            "source_annotation": str(episode.source_annotation or ""),
            "source_split": episode.split,
            "station_type": metadata.get("top_camera_type", "unknown"),
            "annotated": bool(annotation_values),
            "alignment": f"fixed_clock_{config.target_fps}hz_causal_floor",
        },
    }
    return features.serialize_example(payload), len(ticks)


def _worker_initialize() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2

    cv2.setNumThreads(1)


def _shard_filename(job: ShardJob) -> str:
    split = "validation" if job.split == "val" else "train"
    return f"{job.config.dataset_name}-{split}.tfrecord-{job.shard_index:05d}-of-{job.shard_count:05d}"


def _write_shard(job: ShardJob) -> ShardResult:
    import tensorflow as tf

    with contextlib.suppress(RuntimeError):
        tf.config.set_visible_devices([], "GPU")
    started = time.monotonic()
    job.config.local_version_dir.mkdir(parents=True, exist_ok=True)
    destination = job.config.local_version_dir / _shard_filename(job)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    features = make_tfds_features(job.config)
    episode_keys: list[str] = []
    errors: list[str] = []
    frames = 0
    try:
        with tf.io.TFRecordWriter(str(partial)) as writer:
            for episode in job.episodes:
                if episode.spool_error(job.config).is_file():
                    stage_error = json.loads(episode.spool_error(job.config).read_text()).get(
                        "error", "Unknown staging error"
                    )
                    errors.append(f"{episode.key}: {stage_error}")
                    continue
                try:
                    serialized, episode_frames = _episode_payload(episode, job.config, features)
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
            raise RuntimeError(f"Shard {job.key} contains no valid episodes")
        os.replace(partial, destination)
        digest = sha256_file(destination)
        result = ShardResult(
            key=job.key,
            split=job.split,
            shard_index=job.shard_index,
            path=str(destination),
            episode_keys=tuple(episode_keys),
            frames=frames,
            bytes_written=destination.stat().st_size,
            sha256=digest,
            elapsed_seconds=time.monotonic() - started,
            errors=tuple(errors),
        )
        atomic_write_json(
            destination.with_name(f"{destination.name}.conversion.json"),
            {
                **dataclasses.asdict(result),
                "plan_fingerprint": job.plan_fingerprint,
                "expected_episode_keys": [episode.key for episode in job.episodes],
            },
        )
        return result
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=_COPY_BUFFER_SIZE) as stream:
        while chunk := stream.read(_COPY_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS episodes (
            episode_key TEXT PRIMARY KEY, split TEXT NOT NULL, task_name TEXT NOT NULL,
            episode_id TEXT NOT NULL, source_mcap TEXT NOT NULL, source_annotation TEXT,
            size INTEGER NOT NULL, annotation_size INTEGER NOT NULL DEFAULT 0,
            mtime_ns INTEGER NOT NULL, fragment_key TEXT NOT NULL,
            status TEXT NOT NULL, bytes_copied INTEGER NOT NULL DEFAULT 0,
            error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fragments (
            fragment_key TEXT PRIMARY KEY, split TEXT NOT NULL, shard_index INTEGER NOT NULL,
            shard_count INTEGER NOT NULL, episode_keys TEXT NOT NULL, status TEXT NOT NULL,
            path TEXT, frames INTEGER NOT NULL DEFAULT 0, bytes_written INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT, errors TEXT NOT NULL DEFAULT '[]', updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uploads (
            source_path TEXT PRIMARY KEY, destination TEXT NOT NULL, size INTEGER NOT NULL,
            status TEXT NOT NULL, bytes_copied INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS episodes_status_idx ON episodes(status);
        CREATE INDEX IF NOT EXISTS episodes_fragment_idx ON episodes(fragment_key);
        CREATE INDEX IF NOT EXISTS fragments_status_idx ON fragments(status);
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if row is not None and int(row[0]) not in {1, _STATE_SCHEMA_VERSION}:
        raise RuntimeError(f"Unsupported ABC conversion state schema {row[0]}; expected {_STATE_SCHEMA_VERSION}")
    columns = {item[1] for item in connection.execute("PRAGMA table_info(episodes)")}
    if "annotation_size" not in columns:
        connection.execute("ALTER TABLE episodes ADD COLUMN annotation_size INTEGER NOT NULL DEFAULT 0")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(_STATE_SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def _recover_state(config: ConversionConfig) -> None:
    connection = _connect_state(config.state_path)
    try:
        with connection:
            connection.execute("UPDATE episodes SET status='pending' WHERE status='copying'")
            connection.execute("UPDATE fragments SET status='pending' WHERE status='converting'")
            connection.execute(
                "UPDATE episodes SET status='copied' WHERE status='converting' AND fragment_key IN "
                "(SELECT fragment_key FROM fragments WHERE status='pending')"
            )
    finally:
        connection.close()


def _assert_identity(config: ConversionConfig) -> str:
    encoded = json.dumps(config.semantic_identity, sort_keys=True, separators=(",", ":"))
    connection = _connect_state(config.state_path)
    try:
        previous = connection.execute("SELECT value FROM metadata WHERE key='pipeline_identity'").fetchone()
        stored_fingerprint = connection.execute(
            "SELECT value FROM metadata WHERE key='plan_fingerprint'"
        ).fetchone()
        if previous is not None:
            if previous[0] == encoded:
                return stored_fingerprint[0] if stored_fingerprint is not None else hashlib.sha256(encoded.encode()).hexdigest()
            previous_identity = json.loads(previous[0])
            current_identity = json.loads(encoded)
            previous_skip = bool(previous_identity.pop("skip_bad_episodes", False))
            current_skip = bool(current_identity.pop("skip_bad_episodes", False))
            if previous_identity != current_identity or previous_skip or not current_skip:
                raise RuntimeError(
                    f"PFS work root {config.pfs_work_root} belongs to a different ABC conversion configuration; "
                    "reuse the original arguments or choose a fresh --pfs-work-root"
                )
            # Strict -> skip-bad is a safe in-place policy relaxation. Keep the
            # original plan fingerprint so already verified shards remain reusable.
            fingerprint = (
                stored_fingerprint[0]
                if stored_fingerprint is not None
                else hashlib.sha256(previous[0].encode()).hexdigest()
            )
        else:
            fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        with connection:
            connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('pipeline_identity', ?)", (encoded,))
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('plan_fingerprint', ?)", (fingerprint,)
            )
    finally:
        connection.close()
    return fingerprint


def _discover_task_episodes(
    split: str,
    task_dir: Path,
    selected_ids: set[str],
) -> list[EpisodeRef]:
    """Discover one task with the minimum number of remote metadata calls.

    ABC lives on a FUSE-mounted object store. Path.resolve(), repeated is_file()
    checks, and serial glob traversal each turn into remote metadata requests.
    Scan task directories concurrently and stat each MCAP at most once instead.
    """
    results: list[EpisodeRef] = []
    with os.scandir(task_dir) as entries:
        episode_names = sorted(entry.name for entry in entries if _EPISODE_PATTERN.fullmatch(entry.name))
    for episode_name in episode_names:
        episode_id = _EPISODE_PATTERN.fullmatch(episode_name).group("id")  # type: ignore[union-attr]
        if selected_ids and episode_id not in selected_ids:
            continue
        episode_dir = task_dir / episode_name
        mcap = episode_dir / "episode.mcap"
        try:
            mcap_stat = mcap.stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        annotation = episode_dir / "annotation.mcap"
        try:
            annotation_stat = annotation.stat()
        except (FileNotFoundError, NotADirectoryError):
            annotation_path = None
            annotation_size = 0
        else:
            annotation_path = annotation.absolute()
            annotation_size = annotation_stat.st_size
        results.append(
            EpisodeRef(
                split=split,
                task_name=task_dir.name,
                episode_id=episode_id,
                source_mcap=mcap.absolute(),
                source_annotation=annotation_path,
                size=mcap_stat.st_size,
                annotation_size=annotation_size,
                mtime_ns=mcap_stat.st_mtime_ns,
            )
        )
    return results


def discover_episodes(config: ConversionConfig) -> tuple[EpisodeRef, ...]:
    selected_tasks = set(config.task_names)
    selected_ids = {value.removeprefix("episode_") for value in config.episode_ids}
    episodes: list[EpisodeRef] = []
    if config.episode_paths:
        candidates: list[Path] = []
        for value in config.episode_paths:
            path = value.resolve()
            candidates.append(path if path.name == "episode.mcap" else path / "episode.mcap")
        for path in candidates:
            match = _EPISODE_PATTERN.fullmatch(path.parent.name)
            if match is None:
                raise ValueError(f"Expected episode_<id>/episode.mcap, got {path}")
            episode_id = match.group("id")
            split = path.parent.parent.parent.name
            if split not in config.splits or (selected_ids and episode_id not in selected_ids):
                continue
            task_name = path.parent.parent.name
            if selected_tasks and task_name not in selected_tasks:
                continue
            stat = path.stat()
            annotation = path.with_name("annotation.mcap")
            try:
                annotation_stat = annotation.stat()
            except FileNotFoundError:
                annotation_path = None
                annotation_size = 0
            else:
                annotation_path = annotation.absolute()
                annotation_size = annotation_stat.st_size
            episodes.append(
                EpisodeRef(
                    split=split,
                    task_name=task_name,
                    episode_id=episode_id,
                    source_mcap=path.absolute(),
                    source_annotation=annotation_path,
                    size=stat.st_size,
                    annotation_size=annotation_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
    else:
        task_specs: list[tuple[str, Path]] = []
        for split in config.splits:
            split_root = config.input_root / "data" / split
            if not split_root.is_dir():
                raise FileNotFoundError(f"ABC split directory not found: {split_root}")
            for task_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                if selected_tasks and task_dir.name not in selected_tasks:
                    continue
                task_specs.append((split, task_dir))
        if not task_specs:
            raise ValueError("No ABC-130K task directories matched the requested selection")
        workers = min(config.discovery_workers, len(task_specs))
        with (
            tqdm.tqdm(desc="Discovering ABC episodes", unit="episode", dynamic_ncols=True) as progress,
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            futures = {
                executor.submit(_discover_task_episodes, split, task_dir, selected_ids): (split, task_dir)
                for split, task_dir in task_specs
            }
            for completed_tasks, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                discovered = future.result()
                episodes.extend(discovered)
                progress.update(len(discovered))
                progress.set_postfix(tasks=f"{completed_tasks}/{len(task_specs)}")
    if not episodes:
        raise ValueError("No ABC-130K episodes matched the requested selection")
    episodes.sort(key=lambda episode: (episode.split, episode.task_name, episode.episode_id))
    if config.max_episodes is not None:
        episodes = episodes[: config.max_episodes]
    return tuple(episodes)


def build_plan(config: ConversionConfig, episodes: Sequence[EpisodeRef], fingerprint: str) -> tuple[ShardJob, ...]:
    jobs: list[ShardJob] = []
    by_split = {split: [episode for episode in episodes if episode.split == split] for split in config.splits}
    for split, values in by_split.items():
        if not values:
            continue
        shard_count = math.ceil(len(values) / config.episodes_per_shard)
        for index in range(shard_count):
            start = index * config.episodes_per_shard
            jobs.append(
                ShardJob(
                    split,
                    index,
                    shard_count,
                    tuple(values[start : start + config.episodes_per_shard]),
                    config,
                    fingerprint,
                )
            )
    connection = _connect_state(config.state_path)
    try:
        with connection:
            for job in jobs:
                expected = json.dumps([episode.key for episode in job.episodes])
                row = connection.execute(
                    "SELECT split, shard_index, shard_count, episode_keys FROM fragments WHERE fragment_key=?",
                    (job.key,),
                ).fetchone()
                if row is not None and row != (job.split, job.shard_index, job.shard_count, expected):
                    raise RuntimeError(f"Fragment plan changed for {job.key}; use a fresh --pfs-work-root")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fragments(fragment_key, split, shard_index, shard_count,
                      episode_keys, status, updated_at) VALUES(?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (job.key, job.split, job.shard_index, job.shard_count, expected, utc_now()),
                )
                for episode in job.episodes:
                    connection.execute(
                        """
                        INSERT INTO episodes(episode_key, split, task_name, episode_id, source_mcap,
                          source_annotation, size, annotation_size, mtime_ns, fragment_key, status, updated_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                        ON CONFLICT(episode_key) DO UPDATE SET
                          source_mcap=excluded.source_mcap, source_annotation=excluded.source_annotation,
                          size=excluded.size, annotation_size=excluded.annotation_size,
                          mtime_ns=excluded.mtime_ns, fragment_key=excluded.fragment_key
                        """,
                        (
                            episode.key,
                            episode.split,
                            episode.task_name,
                            episode.episode_id,
                            str(episode.source_mcap),
                            str(episode.source_annotation) if episode.source_annotation else None,
                            episode.size,
                            episode.annotation_size,
                            episode.mtime_ns,
                            job.key,
                            utc_now(),
                        ),
                    )
    finally:
        connection.close()
    return tuple(jobs)


def _copy_stream(source: Path, destination: Path, progress: Callable[[int], None] | None = None) -> int:
    stat = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.is_file() and destination.stat().st_size == stat.st_size:
        if progress is not None:
            progress(stat.st_size)
        return stat.st_size
    copied = partial.stat().st_size if partial.is_file() else 0
    if copied > stat.st_size:
        partial.unlink()
        copied = 0
    elif copied and progress is not None:
        progress(copied)
    with source.open("rb", buffering=0) as raw_input:
        raw_input.seek(copied)
        with partial.open("ab", buffering=0) as raw_output:
            while chunk := raw_input.read(_COPY_BUFFER_SIZE):
                raw_output.write(chunk)
                copied += len(chunk)
                if progress is not None:
                    progress(len(chunk))
            os.fdatasync(raw_output.fileno())
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"Source MCAP changed while copying: {source}")
    if copied != stat.st_size:
        raise RuntimeError(f"Short copy for {source}: {copied}/{stat.st_size}")
    os.replace(partial, destination)
    return copied


def _staging_metadata_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.source.json")


def _source_identity(source: Path) -> dict[str, Any]:
    stat = source.stat()
    return {"source": str(source.absolute()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _validate_staged_mcap(path: Path, *, require_summary: bool) -> None:
    size = path.stat().st_size
    if size < 2 * len(_MCAP_MAGIC):
        raise ValueError(f"MCAP is too short: {path} ({size} bytes)")
    with path.open("rb") as stream:
        if stream.read(len(_MCAP_MAGIC)) != _MCAP_MAGIC:
            raise ValueError(f"MCAP header magic is missing: {path}")
        stream.seek(-len(_MCAP_MAGIC), os.SEEK_END)
        if stream.read(len(_MCAP_MAGIC)) != _MCAP_MAGIC:
            raise ValueError(f"MCAP footer magic is missing: {path}")
    if require_summary:
        from mcap.reader import make_reader

        with path.open("rb") as stream:
            if make_reader(stream).get_summary() is None:
                raise ValueError(f"MCAP summary is missing: {path}")


def _staged_file_is_current(source: Path, destination: Path, *, require_summary: bool) -> bool:
    metadata_path = _staging_metadata_path(destination)
    if not destination.is_file() or not metadata_path.is_file():
        return False
    try:
        expected = json.loads(metadata_path.read_text())
        if expected != _source_identity(source) or destination.stat().st_size != expected["size"]:
            return False
        _validate_staged_mcap(destination, require_summary=require_summary)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def _stage_file(
    source: Path,
    destination: Path,
    progress: Callable[[int], None] | None,
    *,
    require_summary: bool,
) -> int:
    if _staged_file_is_current(source, destination, require_summary=require_summary):
        size = destination.stat().st_size
        if progress is not None:
            progress(size)
        return size
    destination.unlink(missing_ok=True)
    _staging_metadata_path(destination).unlink(missing_ok=True)
    copied = _copy_stream(source, destination, progress)
    try:
        _validate_staged_mcap(destination, require_summary=require_summary)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    atomic_write_json(_staging_metadata_path(destination), _source_identity(source))
    return copied


def _staged_episode_is_current(config: ConversionConfig, episode: EpisodeRef) -> bool:
    if not _staged_file_is_current(episode.source_mcap, episode.spool_mcap(config), require_summary=True):
        return False
    annotation_destination = episode.spool_annotation(config)
    return episode.source_annotation is None or (
        annotation_destination is not None
        and _staged_file_is_current(episode.source_annotation, annotation_destination, require_summary=False)
    )


def _stage_episode(config: ConversionConfig, episode: EpisodeRef, progress: Callable[[int], None] | None) -> int:
    episode.spool_error(config).unlink(missing_ok=True)
    copied = _stage_file(
        episode.source_mcap,
        episode.spool_mcap(config),
        progress,
        require_summary=True,
    )
    annotation_destination = episode.spool_annotation(config)
    if episode.source_annotation is not None and annotation_destination is not None:
        copied += _stage_file(
            episode.source_annotation,
            annotation_destination,
            progress,
            require_summary=False,
        )
    return copied


def _existing_result(job: ShardJob) -> ShardResult | None:
    destination = job.config.local_version_dir / _shard_filename(job)
    sidecar = destination.with_name(f"{destination.name}.conversion.json")
    if not destination.is_file() or not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text())
    if payload.get("plan_fingerprint") != job.plan_fingerprint:
        return None
    if destination.stat().st_size != payload.get("bytes_written"):
        return None
    return ShardResult(
        key=payload["key"],
        split=payload["split"],
        shard_index=int(payload["shard_index"]),
        path=payload["path"],
        episode_keys=tuple(payload["episode_keys"]),
        frames=int(payload["frames"]),
        bytes_written=int(payload["bytes_written"]),
        sha256=payload["sha256"],
        elapsed_seconds=float(payload["elapsed_seconds"]),
        errors=tuple(payload["errors"]),
    )


def _accept_result(connection: sqlite3.Connection, result: ShardResult, job: ShardJob) -> None:
    with connection:
        connection.execute(
            """
            UPDATE fragments SET status='complete', path=?, frames=?, bytes_written=?, sha256=?,
              errors=?, updated_at=? WHERE fragment_key=?
            """,
            (
                result.path,
                result.frames,
                result.bytes_written,
                result.sha256,
                json.dumps(result.errors),
                utc_now(),
                result.key,
            ),
        )
        for episode_key in result.episode_keys:
            connection.execute(
                "UPDATE episodes SET status='converted', error=NULL, updated_at=? WHERE episode_key=?",
                (utc_now(), episode_key),
            )
        successful = set(result.episode_keys)
        errors_by_episode = {error.split(": ", 1)[0]: error for error in result.errors}
        for episode in job.episodes:
            if episode.key in successful:
                continue
            connection.execute(
                "UPDATE episodes SET status='failed', error=COALESCE(error, ?), updated_at=? WHERE episode_key=?",
                (errors_by_episode.get(episode.key, "Episode omitted from completed shard"), utc_now(), episode.key),
            )


def _cleanup_spool(config: ConversionConfig, episodes: Sequence[EpisodeRef]) -> None:
    if config.keep_spool:
        return
    for episode in episodes:
        shutil.rmtree(episode.spool_dir(config), ignore_errors=True)


def run_conversion(config: ConversionConfig, jobs: Sequence[ShardJob]) -> tuple[ShardResult, ...]:
    workers = config.convert_workers or physical_core_count()
    connection = _connect_state(config.state_path)
    results: dict[str, ShardResult] = {}
    pending_jobs: list[ShardJob] = []
    try:
        for job in jobs:
            existing = _existing_result(job)
            if existing is not None:
                _accept_result(connection, existing, job)
                _cleanup_spool(config, job.episodes)
                results[job.key] = existing
            else:
                pending_jobs.append(job)
        if not pending_jobs:
            return tuple(results[job.key] for job in jobs)

        LOGGER.info(
            "ABC conversion plan: pending_shards=%d copy_readers=%d workers=%d spool_limit=%d GiB",
            len(pending_jobs),
            config.stream_readers,
            workers,
            config.spool_limit_gib,
        )
        pending_episodes = [episode for job in pending_jobs for episode in job.episodes]
        copied_keys: set[str] = set()
        failed_staging_keys: set[str] = set()
        for episode in pending_episodes:
            if _staged_episode_is_current(config, episode):
                copied_keys.add(episode.key)
            elif episode.spool_dir(config).exists():
                # Legacy spool files had no source-version sidecar and were
                # accepted by size alone. Never reuse them: a BOS object can
                # keep its final size while its contents are still changing.
                shutil.rmtree(episode.spool_dir(config), ignore_errors=True)
        queued_jobs: set[str] = set()
        copy_futures: dict[concurrent.futures.Future[int], EpisodeRef] = {}
        convert_futures: dict[concurrent.futures.Future[ShardResult], ShardJob] = {}
        reserved_bytes = sum(episode.total_size for episode in pending_episodes if episode.key in copied_keys)
        progress_lock = threading.Lock()
        total_source_bytes = sum(episode.total_size for episode in pending_episodes)

        with (
            tqdm.tqdm(
                total=total_source_bytes,
                initial=min(reserved_bytes, total_source_bytes),
                desc="BOS -> PFS ABC MCAP stream",
                unit="B",
                unit_scale=True,
                dynamic_ncols=True,
            ) as byte_progress,
            tqdm.tqdm(
                total=len(pending_jobs), desc="ABC -> RLDS shards", unit="shard", dynamic_ncols=True
            ) as shard_progress,
            concurrent.futures.ThreadPoolExecutor(max_workers=config.stream_readers) as copy_executor,
            concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_initialize,
            ) as convert_executor,
        ):
            pending_index = 0

            def report_copy(delta: int) -> None:
                with progress_lock:
                    byte_progress.update(delta)

            while len(results) < len(jobs):
                for job in pending_jobs:
                    if job.key in queued_jobs or job.key in results:
                        continue
                    ready_keys = copied_keys | failed_staging_keys
                    if all(episode.key in ready_keys for episode in job.episodes):
                        queued_jobs.add(job.key)
                        with connection:
                            connection.execute(
                                "UPDATE fragments SET status='converting', updated_at=? WHERE fragment_key=?",
                                (utc_now(), job.key),
                            )
                            connection.executemany(
                                "UPDATE episodes SET status='converting', updated_at=? WHERE episode_key=?",
                                [(utc_now(), episode.key) for episode in job.episodes if episode.key in copied_keys],
                            )
                        convert_futures[convert_executor.submit(_write_shard, job)] = job

                spool_limit = config.spool_limit_gib * 2**30
                while pending_index < len(pending_episodes) and len(copy_futures) < config.stream_readers:
                    episode = pending_episodes[pending_index]
                    if episode.key in copied_keys:
                        pending_index += 1
                        continue
                    if reserved_bytes + episode.total_size > spool_limit and (copy_futures or convert_futures):
                        break
                    pending_index += 1
                    reserved_bytes += episode.total_size
                    with connection:
                        connection.execute(
                            "UPDATE episodes SET status='copying', updated_at=? WHERE episode_key=?",
                            (utc_now(), episode.key),
                        )
                    copy_futures[copy_executor.submit(_stage_episode, config, episode, report_copy)] = episode

                all_futures = set(copy_futures) | set(convert_futures)
                if not all_futures:
                    unresolved = [job.key for job in jobs if job.key not in results]
                    raise RuntimeError(f"Conversion scheduler stalled with unresolved shards: {unresolved[:10]}")
                done, _ = concurrent.futures.wait(
                    all_futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done & set(copy_futures):
                    episode = copy_futures.pop(future)
                    try:
                        copied = future.result()
                    except BaseException as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        with connection:
                            connection.execute(
                                "UPDATE episodes SET status='failed', error=?, updated_at=? WHERE episode_key=?",
                                (error, utc_now(), episode.key),
                            )
                        if not config.skip_bad_episodes:
                            raise
                        failed_staging_keys.add(episode.key)
                        reserved_bytes -= episode.total_size
                        atomic_write_json(episode.spool_error(config), {"error": error})
                        LOGGER.warning("Skipping invalid ABC source episode %s: %s", episode.key, error)
                        continue
                    copied_keys.add(episode.key)
                    with connection:
                        connection.execute(
                            "UPDATE episodes SET status='copied', bytes_copied=?, error=NULL, updated_at=? WHERE episode_key=?",
                            (copied, utc_now(), episode.key),
                        )
                for future in done & set(convert_futures):
                    job = convert_futures.pop(future)
                    try:
                        result = future.result()
                    except BaseException as exc:
                        with connection:
                            connection.execute(
                                "UPDATE fragments SET status='failed', errors=?, updated_at=? WHERE fragment_key=?",
                                (json.dumps([f"{type(exc).__name__}: {exc}"]), utc_now(), job.key),
                            )
                        raise
                    _accept_result(connection, result, job)
                    results[job.key] = result
                    _cleanup_spool(config, job.episodes)
                    reserved_bytes -= sum(episode.total_size for episode in job.episodes)
                    shard_progress.set_postfix(
                        episodes=sum(len(value.episode_keys) for value in results.values()),
                        frames=sum(value.frames for value in results.values()),
                        gib=f"{sum(value.bytes_written for value in results.values()) / 2**30:.1f}",
                        refresh=False,
                    )
                    shard_progress.update()
        return tuple(results[job.key] for job in jobs)
    finally:
        connection.close()


def finalize_dataset(
    config: ConversionConfig,
    results: Sequence[ShardResult],
    fingerprint: str,
    lineage_id: str,
) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    config.local_version_dir.mkdir(parents=True, exist_ok=True)
    tfds.folder_dataset.write_metadata(
        data_dir=config.local_version_dir,
        features=make_tfds_features(config),
        version=config.version,
        check_data=True,
        description=(
            "ABC-130K bimanual YAM episodes converted from MCAP to sharded TFDS/RLDS. "
            "State/action are 14-D left-arm/gripper/right-arm/gripper vectors aligned with three camera views "
            f"on a causal fixed {config.target_fps} Hz clock."
        ),
        homepage="https://abc.bot/",
    )
    builder = tfds.builder_from_directory(config.local_version_dir)
    split_names = {"train": "train", "val": "validation"}
    per_split: dict[str, Any] = {}
    for source_split in config.splits:
        split_results = [result for result in results if result.split == source_split]
        if not split_results:
            continue
        tfds_split = split_names[source_split]
        expected = sum(len(result.episode_keys) for result in split_results)
        actual = builder.info.splits[tfds_split].num_examples
        if actual != expected:
            raise RuntimeError(f"TFDS split {tfds_split} reports {actual} episodes, expected {expected}")
        per_split[tfds_split] = {
            "episodes": expected,
            "frames": sum(result.frames for result in split_results),
            "shards": len(split_results),
            "bytes": sum(result.bytes_written for result in split_results),
        }
    manifest = {
        "status": "complete",
        "completed_at": utc_now(),
        "pipeline": "abc130k_bos_pfs_mcap_rlds_v1",
        "plan_fingerprint": fingerprint,
        "lineage_id": lineage_id,
        "dataset_name": config.dataset_name,
        "version": config.version,
        "alignment": f"fixed_clock_{config.target_fps}hz_causal_floor",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "image_shape": [config.image_height, config.image_width, 3],
        "splits": per_split,
        "errors": [error for result in results for error in result.errors],
        "shard_sha256": {Path(result.path).name: result.sha256 for result in results},
    }
    atomic_write_json(config.local_version_dir / "conversion_manifest.json", manifest)
    return manifest


def validate_tfds_directory(path: Path, *, read_examples: int = 1) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    if not (path / "dataset_info.json").is_file() or not (path / "features.json").is_file():
        raise FileNotFoundError(f"Incomplete TFDS directory: {path}")
    builder = tfds.builder_from_directory(path)
    splits = {name: info.num_examples for name, info in builder.info.splits.items()}
    checked = 0
    if read_examples:
        for split in splits:
            for example in tfds.as_numpy(builder.as_dataset(split=split).take(read_examples)):
                try:
                    first_step = next(iter(example["steps"]))
                except StopIteration:
                    raise ValueError(f"TFDS split {split} contains an empty episode") from None
                if first_step["observation"]["state"].shape[-1] != STATE_DIM:
                    raise ValueError("Unexpected state dimension during TFDS readback")
                checked += 1
    return {"splits": splits, "examples_read": checked}


def _copy_publish_file(source: Path, destination: Path, progress: Callable[[int], None]) -> None:
    _copy_stream(source, destination, progress)


def publish_to_bos(config: ConversionConfig, manifest: dict[str, Any]) -> Path:
    final = config.bos_version_dir
    if final.is_dir():
        existing = validate_tfds_directory(final, read_examples=1)
        expected = {name: values["episodes"] for name, values in manifest["splits"].items()}
        if existing["splits"] != expected:
            raise RuntimeError(f"Existing BOS dataset has different split sizes: {existing['splits']} != {expected}")
        LOGGER.info("BOS output already exists and validates: %s", final)
        return final
    hidden = final.with_name(f".{final.name}.uploading")
    hidden.mkdir(parents=True, exist_ok=True)
    metadata_names = {"dataset_info.json", "features.json", "conversion_manifest.json", "LICENSE"}
    sources = sorted(path for path in config.local_version_dir.iterdir() if path.is_file())
    sources.sort(key=lambda path: (path.name in metadata_names, path.name))
    total = sum(path.stat().st_size for path in sources)
    connection = _connect_state(config.state_path)
    try:
        with connection:
            for source in sources:
                connection.execute(
                    """
                    INSERT INTO uploads(source_path, destination, size, status, bytes_copied, updated_at)
                    VALUES(?, ?, ?, 'pending', 0, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                      destination=excluded.destination, size=excluded.size, updated_at=excluded.updated_at
                    """,
                    (str(source), str(hidden / source.name), source.stat().st_size, utc_now()),
                )
        with tqdm.tqdm(
            total=total, desc="PFS -> BOS ABC publish", unit="B", unit_scale=True, dynamic_ncols=True
        ) as progress:
            for source in sources:
                with connection:
                    connection.execute(
                        "UPDATE uploads SET status='uploading', updated_at=? WHERE source_path=?",
                        (utc_now(), str(source)),
                    )
                try:
                    _copy_publish_file(source, hidden / source.name, progress.update)
                except BaseException:
                    with connection:
                        connection.execute(
                            "UPDATE uploads SET status='failed', updated_at=? WHERE source_path=?",
                            (utc_now(), str(source)),
                        )
                    raise
                with connection:
                    connection.execute(
                        "UPDATE uploads SET status='complete', bytes_copied=size, updated_at=? WHERE source_path=?",
                        (utc_now(), str(source)),
                    )
    finally:
        connection.close()
    validate_tfds_directory(hidden, read_examples=1)
    ready = hidden / "READY"
    ready.write_text(json.dumps({"validated_at": utc_now(), "manifest": manifest}, sort_keys=True) + "\n")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(hidden, final)
    validate_tfds_directory(final, read_examples=1)
    return final


def status(config: ConversionConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        return {"state": "not_started", "state_path": str(config.state_path)}
    connection = sqlite3.connect(f"file:{config.state_path}?mode=ro", uri=True, timeout=60)
    try:
        source = connection.execute(
            "SELECT COALESCE(SUM(size + annotation_size), 0), COALESCE(SUM(bytes_copied), 0) FROM episodes"
        ).fetchone()
        result = {
            "state_path": str(config.state_path),
            "source_bytes_total": int(source[0]),
            "source_bytes_copied": int(source[1]),
            "spool_bytes": sum(path.stat().st_size for path in config.spool_root.rglob("*.mcap"))
            if config.spool_root.is_dir()
            else 0,
            "episodes": dict(connection.execute("SELECT status, COUNT(*) FROM episodes GROUP BY status")),
            "fragments": dict(connection.execute("SELECT status, COUNT(*) FROM fragments GROUP BY status")),
            "uploads": dict(connection.execute("SELECT status, COUNT(*) FROM uploads GROUP BY status")),
        }
        errors = connection.execute(
            """
            SELECT 'episode', episode_key, error FROM episodes WHERE error IS NOT NULL
            UNION ALL SELECT 'fragment', fragment_key, errors FROM fragments
              WHERE status='failed' ORDER BY 1, 2 LIMIT 20
            """
        ).fetchall()
        result["recent_errors"] = [list(row) for row in errors]
        return result
    finally:
        connection.close()


def run_pipeline(config: ConversionConfig, *, lineage_id: str) -> dict[str, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    config.validate()
    fingerprint = _assert_identity(config)
    _recover_state(config)
    episodes = discover_episodes(config)
    LOGGER.info(
        "Discovered %d ABC episodes: %s",
        len(episodes),
        ", ".join(f"{split}={sum(ep.split == split for ep in episodes)}" for split in config.splits),
    )
    jobs = build_plan(config, episodes, fingerprint)
    results = run_conversion(config, jobs)
    manifest = finalize_dataset(config, results, fingerprint, lineage_id)
    destination = publish_to_bos(config, manifest) if config.publish else config.local_version_dir
    LOGGER.info("ABC conversion complete and verified: %s", destination)
    return manifest
