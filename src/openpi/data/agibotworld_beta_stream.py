"""Sequential BOS -> PFS -> RLDS pipeline for AgiBotWorld Beta.

The source observation archives are intentionally never indexed with random reads.
Each archive is consumed from beginning to end, only the three training videos are
spooled to PFS, and the spool is removed after its RLDS fragment is verified.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import concurrent.futures
import contextlib
import dataclasses
import datetime
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
import threading
import time
from typing import Any, BinaryIO

import tqdm_loggable.auto as tqdm

from openpi.data import agibotworld_beta as base

LOGGER = logging.getLogger(__name__)

_BLOCK_SIZE = 512
_COPY_BUFFER_SIZE = 16 * 2**20
_PROGRESS_INTERVAL = 256 * 2**20
_STATE_SCHEMA_VERSION = 1
_SQLITE_BUSY_TIMEOUT_MS = 10_000
_SQLITE_LOCK_RETRIES = 12
_SQLITE_RETRY_INITIAL_SECONDS = 0.05
_SQLITE_RETRY_MAX_SECONDS = 5.0
_ARCHIVE_RANGE = re.compile(r"^(?P<start>\d+)-(?P<end>\d+)\.tar$")
_WANTED_VIDEO_FILES = {filename: camera for camera, filename in base.CAMERA_FILES.items()}


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    input_root: Path = Path("/mnt/bos/dataset/agibotworld-beta")
    pfs_work_root: Path = Path("/mnt/pfs/rhos-vla/chenyuan/agibotworld-beta-rlds-work")
    output_root: Path = Path("/mnt/bos/dataset/RLDS")
    dataset_name: str = "agibotworld_beta"
    version: str = "1.0.0"
    image_height: int = 224
    image_width: int = 224
    jpeg_quality: int = 90
    episodes_per_shard: int = 8
    stream_readers: int = 2
    convert_workers: int = 0
    spool_limit_gib: int = 1024
    decoder: base.Decoder = base.Decoder.AUTO
    decoder_threads: int = 1
    gpu_workers_per_device: int = 4
    alignment_tolerance: int = 2
    task_ids: tuple[int, ...] = ()
    episode_ids: tuple[int, ...] = ()
    observation_archives: tuple[Path, ...] = ()
    max_episodes: int | None = None
    allow_incomplete: bool = False
    publish: bool = True

    def validate(self) -> None:
        if not self.input_root.is_dir():
            raise FileNotFoundError(f"AgiBotWorld Beta input root not found: {self.input_root}")
        if not re.fullmatch(r"[a-z0-9_]+", self.dataset_name):
            raise ValueError("dataset_name must contain only lowercase letters, digits, and underscores")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("version must have the form MAJOR.MINOR.PATCH")
        if self.stream_readers <= 0 or self.convert_workers < 0:
            raise ValueError("stream_readers must be positive and convert_workers must be non-negative")
        if self.spool_limit_gib <= 0 or self.episodes_per_shard <= 0:
            raise ValueError("spool_limit_gib and episodes_per_shard must be positive")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive or omitted")
        if self.decoder_threads <= 0:
            raise ValueError("decoder_threads must be positive")
        self.pfs_work_root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.pfs_work_root / "state" / "pipeline.sqlite3"

    @property
    def local_input_root(self) -> Path:
        return self.pfs_work_root / "source"

    @property
    def fragment_dir(self) -> Path:
        return self.pfs_work_root / "fragments"

    @property
    def local_output_root(self) -> Path:
        return self.pfs_work_root / "output"

    @property
    def local_version_dir(self) -> Path:
        return self.local_output_root / self.dataset_name / self.version

    @property
    def bos_version_dir(self) -> Path:
        return self.output_root / self.dataset_name / self.version

    def converter_config(self) -> base.ConverterConfig:
        return base.ConverterConfig(
            input_root=self.local_input_root,
            output_root=self.local_output_root,
            dataset_name=self.dataset_name,
            version=self.version,
            image_height=self.image_height,
            image_width=self.image_width,
            jpeg_quality=self.jpeg_quality,
            episodes_per_shard=self.episodes_per_shard,
            workers=self.convert_workers,
            decoder=self.decoder,
            decoder_threads=self.decoder_threads,
            gpu_workers_per_device=self.gpu_workers_per_device,
            alignment_tolerance=self.alignment_tolerance,
            skip_bad_episodes=True,
        )


@dataclasses.dataclass(frozen=True)
class ObservationArchive:
    path: Path
    task_id: int
    ordinal: int


@dataclasses.dataclass(frozen=True)
class ScanResult:
    archive: str
    task_id: int
    bytes_read: int
    selected_members: int
    ready_episodes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FragmentJob:
    key: str
    path: Path
    episodes: tuple[base.EpisodeRef, ...]
    config: base.ConverterConfig
    decoder: base.Decoder
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class FragmentResult:
    key: str
    path: str | None
    episode_keys: tuple[str, ...]
    frames: int
    bytes_written: int
    sha256: str | None
    elapsed_seconds: float
    errors: tuple[tuple[str, str], ...]


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _is_sqlite_lock_error(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if code is not None and code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(error).lower()
    return "database is locked" in message or "database table is locked" in message


class _RetryingConnection(sqlite3.Connection):
    """SQLite connection that survives transient writer contention.

    WAL permits concurrent readers but still has one writer. A read snapshot that
    is upgraded after another connection commits raises SQLITE_BUSY_SNAPSHOT
    immediately and ignores busy_timeout; rolling that snapshot back before the
    retry is required.
    """

    def _retry_locked(self, operation: Callable[[], Any], operation_name: str) -> Any:
        delay = _SQLITE_RETRY_INITIAL_SECONDS
        for attempt in range(1, _SQLITE_LOCK_RETRIES + 1):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_lock_error(exc) or attempt == _SQLITE_LOCK_RETRIES:
                    raise
                if getattr(exc, "sqlite_errorname", "") == "SQLITE_BUSY_SNAPSHOT":
                    sqlite3.Connection.rollback(self)
                LOGGER.warning(
                    "SQLite %s contention; retrying attempt %d/%d in %.2fs",
                    operation_name,
                    attempt + 1,
                    _SQLITE_LOCK_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, _SQLITE_RETRY_MAX_SECONDS)
        raise AssertionError("unreachable")

    def execute(self, sql: str, parameters: Sequence[Any] = (), /) -> sqlite3.Cursor:
        execute = super().execute
        return self._retry_locked(lambda: execute(sql, parameters), "execute")

    def executemany(self, sql: str, seq_of_parameters: Iterable[Sequence[Any]], /) -> sqlite3.Cursor:
        executemany = super().executemany
        parameters = tuple(seq_of_parameters)
        return self._retry_locked(lambda: executemany(sql, parameters), "executemany")

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        executescript = super().executescript
        return self._retry_locked(lambda: executescript(sql_script), "executescript")

    def commit(self) -> None:
        commit = super().commit
        self._retry_locked(commit, "commit")


def _connect_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000,
        factory=_RetryingConnection,
    )
    connection.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS archives (
            source_path TEXT PRIMARY KEY, task_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, status TEXT NOT NULL,
            offset INTEGER NOT NULL DEFAULT 0, global_pax TEXT NOT NULL DEFAULT '{}',
            selected_members INTEGER NOT NULL DEFAULT 0, error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS copies (
            source_path TEXT PRIMARY KEY, destination TEXT NOT NULL, size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL, status TEXT NOT NULL, bytes_copied INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT, error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS spool_members (
            task_id INTEGER NOT NULL, episode_id INTEGER NOT NULL, camera TEXT NOT NULL,
            path TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
            source_archive TEXT NOT NULL, member_name TEXT NOT NULL,
            PRIMARY KEY(task_id, episode_id, camera)
        );
        CREATE TABLE IF NOT EXISTS episodes (
            task_id INTEGER NOT NULL, episode_id INTEGER NOT NULL, status TEXT NOT NULL,
            source_archive TEXT NOT NULL, error TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY(task_id, episode_id)
        );
        CREATE TABLE IF NOT EXISTS fragments (
            fragment_key TEXT PRIMARY KEY, source_archive TEXT NOT NULL, fragment_index INTEGER NOT NULL,
            path TEXT NOT NULL, status TEXT NOT NULL, episode_keys TEXT NOT NULL,
            frames INTEGER NOT NULL DEFAULT 0, bytes_written INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT, errors TEXT NOT NULL DEFAULT '[]', fingerprint TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uploads (
            source_path TEXT PRIMARY KEY, destination TEXT NOT NULL, size INTEGER NOT NULL,
            sha256 TEXT NOT NULL, status TEXT NOT NULL, bytes_copied INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS archives_status_idx ON archives(status, ordinal);
        CREATE INDEX IF NOT EXISTS episodes_source_idx ON episodes(source_archive, status);
        CREATE INDEX IF NOT EXISTS fragments_source_idx ON fragments(source_archive, fragment_index);
        """
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if row is not None and int(row[0]) != _STATE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported streaming state schema {row[0]} at {path}; expected {_STATE_SCHEMA_VERSION}")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(_STATE_SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def _recover_interrupted_state(config: StreamConfig) -> None:
    connection = _connect_state(config.state_path)
    try:
        with connection:
            connection.execute("UPDATE archives SET status='pending' WHERE status='scanning'")
            connection.execute("UPDATE episodes SET status='ready' WHERE status='converting'")
            connection.execute("UPDATE fragments SET status='pending' WHERE status='writing'")
        completed_spool = [
            (int(row[0]), int(row[1]), Path(row[2]))
            for row in connection.execute(
                """
                SELECT s.task_id, s.episode_id, s.path
                FROM spool_members s JOIN episodes e
                  ON e.task_id=s.task_id AND e.episode_id=s.episode_id
                WHERE e.status IN ('converted', 'failed')
                """
            )
        ]
        spool_root = (config.pfs_work_root / "spool").resolve()
        for _, _, path in completed_spool:
            resolved = path.resolve()
            if not resolved.is_relative_to(spool_root):
                raise RuntimeError(f"Refusing to clean a spool path outside this workdir: {resolved}")
            resolved.unlink(missing_ok=True)
        with connection:
            connection.executemany(
                "DELETE FROM spool_members WHERE task_id=? AND episode_id=?",
                {(task_id, episode_id) for task_id, episode_id, _ in completed_spool},
            )
        if config.fragment_dir.is_dir():
            for partial in config.fragment_dir.glob(".fragment-*.tfrecord.*.partial"):
                partial.unlink()
    finally:
        connection.close()


def _assert_config_identity(config: StreamConfig) -> None:
    payload = {
        "input_root": str(config.input_root.resolve()),
        "dataset_name": config.dataset_name,
        "version": config.version,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "jpeg_quality": config.jpeg_quality,
        "alignment_tolerance": config.alignment_tolerance,
        "task_ids": sorted(config.task_ids),
        "episode_ids": sorted(config.episode_ids),
        "observation_archives": sorted(str(path.resolve()) for path in config.observation_archives),
        "max_episodes": config.max_episodes,
    }
    encoded = json.dumps(payload, sort_keys=True)
    connection = _connect_state(config.state_path)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='pipeline_config'").fetchone()
        if row is not None and row[0] != encoded:
            raise RuntimeError(
                f"Streaming workdir {config.pfs_work_root} belongs to a different dataset selection or schema; "
                "resume with the original arguments or use a fresh --pfs-work-root"
            )
        with connection:
            connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('pipeline_config', ?)", (encoded,))
    finally:
        connection.close()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=_COPY_BUFFER_SIZE) as stream:
        while chunk := stream.read(_COPY_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _copy_exact(
    source: BinaryIO,
    size: int,
    destination: BinaryIO | None,
    digest: Any | None,
    progress: Callable[[int], None] | None,
) -> None:
    remaining = size
    since_progress = 0
    while remaining:
        chunk = source.read(min(_COPY_BUFFER_SIZE, remaining))
        if not chunk:
            raise EOFError(f"Unexpected EOF with {remaining} payload bytes remaining")
        if destination is not None:
            destination.write(chunk)
        if digest is not None:
            digest.update(chunk)
        remaining -= len(chunk)
        since_progress += len(chunk)
        if progress is not None and since_progress >= _PROGRESS_INTERVAL:
            progress(since_progress)
            since_progress = 0
    if progress is not None and since_progress:
        progress(since_progress)


def _parse_tar_number(field: bytes) -> int:
    if field and field[0] & 0x80:
        return int.from_bytes(bytes([field[0] & 0x7F]) + field[1:], "big", signed=False)
    value = field.rstrip(b"\0 ").lstrip(b" \0")
    return int(value or b"0", 8)


def _parse_tar_header(block: bytes, offset: int) -> tuple[str, int, bytes]:
    if len(block) != _BLOCK_SIZE:
        raise EOFError(f"Short tar header at byte {offset}")
    stored_checksum = _parse_tar_number(block[148:156])
    checksum = sum(block[:148]) + 8 * ord(" ") + sum(block[156:])
    if checksum != stored_checksum:
        raise ValueError(f"Invalid tar checksum at byte {offset}: {checksum} != {stored_checksum}")
    name = block[:100].split(b"\0", 1)[0]
    prefix = block[345:500].split(b"\0", 1)[0]
    raw_name = prefix + (b"/" if prefix and name else b"") + name
    return raw_name.decode("utf-8", errors="surrogateescape"), _parse_tar_number(block[124:136]), block[156:157]


def _parse_pax(payload: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    position = 0
    while position < len(payload):
        space = payload.find(b" ", position)
        if space < 0:
            raise ValueError("Malformed PAX record without a length separator")
        length = int(payload[position:space])
        record = payload[space + 1 : position + length]
        if len(record) != position + length - space - 1 or not record.endswith(b"\n"):
            raise ValueError("Malformed PAX record length")
        key, separator, value = record[:-1].partition(b"=")
        if separator:
            result[key.decode(errors="surrogateescape")] = value.decode(errors="surrogateescape")
        position += length
    return result


def _member_selection(name: str) -> tuple[int, str] | None:
    parts = PurePosixPath(name.removeprefix("./")).parts
    if len(parts) != 3 or parts[1] != "videos":
        return None
    camera = _WANTED_VIDEO_FILES.get(parts[2])
    if camera is None:
        return None
    try:
        return int(parts[0]), camera
    except ValueError:
        return None


def _episode_can_use_archive(
    connection: sqlite3.Connection,
    task_id: int,
    episode_id: int,
    source_archive: str,
) -> bool:
    row = connection.execute(
        "SELECT status, source_archive FROM episodes WHERE task_id=? AND episode_id=?",
        (task_id, episode_id),
    ).fetchone()
    if row is None or row[1] == source_archive:
        return row is None or row[0] != "converted"
    status, previous_source = row
    if status != "failed":
        return False
    stale_paths = [
        Path(item[0])
        for item in connection.execute(
            "SELECT path FROM spool_members WHERE task_id=? AND episode_id=?",
            (task_id, episode_id),
        )
    ]
    for path in stale_paths:
        path.unlink(missing_ok=True)
    with connection:
        connection.execute("DELETE FROM spool_members WHERE task_id=? AND episode_id=?", (task_id, episode_id))
        connection.execute(
            "DELETE FROM episodes WHERE task_id=? AND episode_id=? AND source_archive=?",
            (task_id, episode_id, previous_source),
        )
    return True


def _record_spool_member(
    connection: sqlite3.Connection,
    *,
    task_id: int,
    episode_id: int,
    camera: str,
    path: Path,
    size: int,
    sha256: str,
    source_archive: str,
    member_name: str,
) -> bool:
    with connection:
        connection.execute(
            """
            INSERT INTO spool_members(task_id, episode_id, camera, path, size, sha256, source_archive, member_name)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, episode_id, camera) DO UPDATE SET
              path=excluded.path, size=excluded.size, sha256=excluded.sha256,
              source_archive=excluded.source_archive, member_name=excluded.member_name
            """,
            (task_id, episode_id, camera, str(path), size, sha256, source_archive, member_name),
        )
        count = connection.execute(
            """
            SELECT COUNT(*) FROM spool_members
            WHERE task_id=? AND episode_id=? AND source_archive=?
            """,
            (task_id, episode_id, source_archive),
        ).fetchone()[0]
        if count == len(base.CAMERA_FILES):
            connection.execute(
                """
                INSERT INTO episodes(task_id, episode_id, status, source_archive, error, updated_at)
                VALUES(?, ?, 'ready', ?, NULL, ?)
                ON CONFLICT(task_id, episode_id) DO UPDATE SET
                  status='ready', source_archive=excluded.source_archive, error=NULL, updated_at=excluded.updated_at
                """,
                (task_id, episode_id, source_archive, _utc_now()),
            )
            return True
    return False


def scan_observation_archive(
    config: StreamConfig,
    archive: ObservationArchive,
    progress: Callable[[int], None] | None = None,
) -> ScanResult:
    """Sequentially consume one tar and spool only the three selected videos."""
    source = archive.path.resolve()
    source_text = str(source)
    stat = source.stat()
    selected_episode_ids = set(config.episode_ids)
    connection = _connect_state(config.state_path)
    try:
        row = connection.execute(
            "SELECT size, mtime_ns, status, offset, global_pax, selected_members FROM archives WHERE source_path=?",
            (source_text,),
        ).fetchone()
        if row is not None and (row[0], row[1]) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"Source archive changed after streaming began: {source}")
        if row is not None and row[2] in {"complete", "selection_complete"}:
            ready = tuple(
                item[0]
                for item in connection.execute(
                    "SELECT episode_id FROM episodes WHERE source_archive=? AND status='ready' ORDER BY episode_id",
                    (source_text,),
                )
            )
            return ScanResult(source_text, archive.task_id, int(row[3]), int(row[5]), ready)
        offset = int(row[3]) if row is not None else 0
        global_pax = json.loads(row[4]) if row is not None else {}
        selected_members = int(row[5]) if row is not None else 0
        with connection:
            connection.execute(
                """
                INSERT INTO archives(source_path, task_id, ordinal, size, mtime_ns, status, offset,
                                     global_pax, selected_members, error, updated_at)
                VALUES(?, ?, ?, ?, ?, 'scanning', ?, ?, ?, NULL, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                  status='scanning', error=NULL, updated_at=excluded.updated_at
                """,
                (
                    source_text,
                    archive.task_id,
                    archive.ordinal,
                    stat.st_size,
                    stat.st_mtime_ns,
                    offset,
                    json.dumps(global_pax),
                    selected_members,
                    _utc_now(),
                ),
            )
        ready_episodes: set[int] = set()
        selection_complete = False
        local_pax: dict[str, str] = {}
        long_name: str | None = None
        with source.open("rb", buffering=0) as raw:
            raw.seek(offset)
            stream = io.BufferedReader(raw, buffer_size=_COPY_BUFFER_SIZE)
            position = offset
            while True:
                header_offset = position
                header = _read_exact(stream, _BLOCK_SIZE)
                position += _BLOCK_SIZE
                if progress is not None:
                    progress(_BLOCK_SIZE)
                if header == b"\0" * _BLOCK_SIZE:
                    second = _read_exact(stream, _BLOCK_SIZE)
                    position += _BLOCK_SIZE
                    if progress is not None:
                        progress(_BLOCK_SIZE)
                    if second != b"\0" * _BLOCK_SIZE:
                        raise ValueError(f"A single zero block was followed by data at byte {position}")
                    trailing = stat.st_size - position
                    if trailing < 0:
                        raise ValueError(f"Tar stream exceeded the source size at byte {position}")
                    if trailing:
                        _copy_exact(stream, trailing, None, None, progress)
                        position += trailing
                    break
                name, size, typeflag = _parse_tar_header(header, header_offset)
                effective_name = local_pax.get("path", global_pax.get("path", long_name or name))
                if "size" in local_pax:
                    size = int(local_pax["size"])
                elif "size" in global_pax:
                    size = int(global_pax["size"])
                special_payload: bytes | None = None
                selection = _member_selection(effective_name) if typeflag in {b"0", b"\0"} else None
                selected = False
                if selection is not None:
                    episode_id, camera = selection
                    selected = (
                        not selected_episode_ids or episode_id in selected_episode_ids
                    ) and _episode_can_use_archive(connection, archive.task_id, episode_id, source_text)
                if selected:
                    episode_id, camera = selection  # type: ignore[misc]
                    spool_dir = config.pfs_work_root / "spool" / str(archive.task_id) / str(episode_id)
                    spool_dir.mkdir(parents=True, exist_ok=True)
                    destination = spool_dir / f"{camera}.mp4"
                    partial = destination.with_suffix(".mp4.partial")
                    digest = hashlib.sha256()
                    try:
                        with partial.open("wb", buffering=_COPY_BUFFER_SIZE) as output:
                            _copy_exact(stream, size, output, digest, progress)
                            output.flush()
                            os.fdatasync(output.fileno())
                        current = source.stat()
                        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                            raise RuntimeError(f"Source archive changed while it was being read: {source}")
                        os.replace(partial, destination)
                    except BaseException:
                        partial.unlink(missing_ok=True)
                        raise
                    selected_members += 1
                    if _record_spool_member(
                        connection,
                        task_id=archive.task_id,
                        episode_id=episode_id,
                        camera=camera,
                        path=destination,
                        size=size,
                        sha256=digest.hexdigest(),
                        source_archive=source_text,
                        member_name=effective_name,
                    ):
                        ready_episodes.add(episode_id)
                        selection_complete = bool(selected_episode_ids) and selected_episode_ids <= ready_episodes
                elif typeflag in {b"x", b"g", b"L", b"K"}:
                    if size > 16 * 2**20:
                        raise ValueError(f"Unreasonably large tar extension record ({size} bytes) in {source}")
                    special_payload = _read_exact(stream, size)
                    if progress is not None:
                        progress(size)
                else:
                    _copy_exact(stream, size, None, None, progress)
                position += size
                padding = (-size) % _BLOCK_SIZE
                if padding:
                    _copy_exact(stream, padding, None, None, progress)
                    position += padding
                if typeflag == b"g":
                    global_pax.update(_parse_pax(special_payload or b""))
                elif typeflag == b"x":
                    local_pax = _parse_pax(special_payload or b"")
                    continue
                elif typeflag == b"L":
                    long_name = (special_payload or b"").rstrip(b"\0\n").decode(errors="surrogateescape")
                    continue
                elif typeflag == b"K":
                    continue
                local_pax = {}
                long_name = None
                with connection:
                    connection.execute(
                        """
                        UPDATE archives SET offset=?, global_pax=?, selected_members=?, updated_at=?
                        WHERE source_path=?
                        """,
                        (position, json.dumps(global_pax), selected_members, _utc_now(), source_text),
                    )
                if selection_complete:
                    break
        if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
            raise RuntimeError(f"Source archive changed while it was being read: {source}")
        incomplete_rows = connection.execute(
            """
            SELECT episode_id, COUNT(*) FROM spool_members
            WHERE source_archive=? GROUP BY episode_id HAVING COUNT(*) != ?
            """,
            (source_text, len(base.CAMERA_FILES)),
        ).fetchall()
        if not selection_complete:
            for episode_id, camera_count in incomplete_rows:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO episodes(task_id, episode_id, status, source_archive, error, updated_at)
                        VALUES(?, ?, 'failed', ?, ?, ?)
                        ON CONFLICT(task_id, episode_id) DO UPDATE SET
                          status='failed', source_archive=excluded.source_archive,
                          error=excluded.error, updated_at=excluded.updated_at
                        """,
                        (
                            archive.task_id,
                            episode_id,
                            source_text,
                            f"Incomplete camera set: found {camera_count}/{len(base.CAMERA_FILES)}",
                            _utc_now(),
                        ),
                    )
                _cleanup_episode_spool(connection, archive.task_id, episode_id)
        with connection:
            connection.execute(
                """
                UPDATE archives SET status=?, offset=?, selected_members=?, error=NULL, updated_at=?
                WHERE source_path=?
                """,
                (
                    "selection_complete" if selection_complete else "complete",
                    position,
                    selected_members,
                    _utc_now(),
                    source_text,
                ),
            )
        return ScanResult(
            source_text,
            archive.task_id,
            position,
            selected_members,
            tuple(sorted(ready_episodes)),
        )
    except BaseException as exc:
        with connection:
            connection.execute(
                "UPDATE archives SET status='failed', error=?, updated_at=? WHERE source_path=?",
                (f"{type(exc).__name__}: {exc}", _utc_now(), source_text),
            )
        raise
    finally:
        connection.close()


def resumable_copy(
    source: Path,
    destination: Path,
    state_path: Path,
    progress: Callable[[int], None] | None = None,
) -> tuple[int, str]:
    """Copy a file sequentially through a durable partial file and return size/hash."""
    source = source.resolve()
    stat = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    connection = _connect_state(state_path)
    try:
        row = connection.execute(
            "SELECT destination, size, mtime_ns, status, sha256 FROM copies WHERE source_path=?",
            (str(source),),
        ).fetchone()
        if row is not None and (row[0], row[1], row[2]) != (
            str(destination),
            stat.st_size,
            stat.st_mtime_ns,
        ):
            raise RuntimeError(f"Copy source or destination changed since the previous run: {source}")
        if (
            row is not None
            and row[3] == "complete"
            and destination.is_file()
            and destination.stat().st_size == stat.st_size
        ):
            return stat.st_size, str(row[4])
        if destination.is_file() and destination.stat().st_size == stat.st_size:
            digest_text = _sha256_file(destination)
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO copies(source_path, destination, size, mtime_ns, status,
                                                  bytes_copied, sha256, error, updated_at)
                    VALUES(?, ?, ?, ?, 'complete', ?, ?, NULL, ?)
                    """,
                    (
                        str(source),
                        str(destination),
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_size,
                        digest_text,
                        _utc_now(),
                    ),
                )
            return stat.st_size, digest_text
        copied = partial.stat().st_size if partial.is_file() else 0
        if copied > stat.st_size:
            raise RuntimeError(f"Partial file is larger than its source: {partial}")
        digest = hashlib.sha256()
        if copied:
            with partial.open("rb", buffering=_COPY_BUFFER_SIZE) as existing:
                while chunk := existing.read(_COPY_BUFFER_SIZE):
                    digest.update(chunk)
        with connection:
            connection.execute(
                """
                INSERT INTO copies(source_path, destination, size, mtime_ns, status, bytes_copied,
                                   sha256, error, updated_at)
                VALUES(?, ?, ?, ?, 'copying', ?, NULL, NULL, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                  status='copying', bytes_copied=excluded.bytes_copied, error=NULL, updated_at=excluded.updated_at
                """,
                (str(source), str(destination), stat.st_size, stat.st_mtime_ns, copied, _utc_now()),
            )
        with source.open("rb", buffering=0) as raw_source, partial.open("ab", buffering=_COPY_BUFFER_SIZE) as output:
            raw_source.seek(copied)
            stream = io.BufferedReader(raw_source, buffer_size=_COPY_BUFFER_SIZE)
            remaining = stat.st_size - copied
            while remaining:
                chunk = stream.read(min(_COPY_BUFFER_SIZE, remaining))
                if not chunk:
                    raise EOFError(f"Short source read while copying {source}")
                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
                if progress is not None:
                    progress(len(chunk))
                if copied % (1024 * 2**20) < len(chunk):
                    output.flush()
                    os.fdatasync(output.fileno())
                    with connection:
                        connection.execute(
                            "UPDATE copies SET bytes_copied=?, updated_at=? WHERE source_path=?",
                            (copied, _utc_now(), str(source)),
                        )
            output.flush()
            os.fdatasync(output.fileno())
        current = source.stat()
        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"Source changed while copying: {source}")
        os.replace(partial, destination)
        digest_text = digest.hexdigest()
        with connection:
            connection.execute(
                """
                UPDATE copies SET status='complete', bytes_copied=?, sha256=?, error=NULL, updated_at=?
                WHERE source_path=?
                """,
                (copied, digest_text, _utc_now(), str(source)),
            )
        return copied, digest_text
    except BaseException as exc:
        with connection:
            connection.execute(
                "UPDATE copies SET status='failed', error=?, updated_at=? WHERE source_path=?",
                (f"{type(exc).__name__}: {exc}", _utc_now(), str(source)),
            )
        raise
    finally:
        connection.close()


def discover_observation_archives(config: StreamConfig) -> tuple[ObservationArchive, ...]:
    if config.observation_archives:
        paths = [path.resolve() for path in config.observation_archives]
    else:
        task_dirs = (
            [config.input_root / "observations" / str(task_id) for task_id in config.task_ids]
            if config.task_ids
            else sorted((config.input_root / "observations").glob("*"), key=lambda path: int(path.name))
        )
        paths = [path.resolve() for directory in task_dirs for path in directory.glob("*.tar")]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Observation archives not found: {missing[:5]}")

    def key(path: Path) -> tuple[int, int, int, str]:
        task_id = int(path.parent.name)
        match = _ARCHIVE_RANGE.fullmatch(path.name)
        return task_id, int(match["start"]) if match else math.inf, int(match["end"]) if match else math.inf, path.name

    paths.sort(key=key)
    archives = tuple(ObservationArchive(path, int(path.parent.name), ordinal) for ordinal, path in enumerate(paths))
    if not archives:
        raise FileNotFoundError("No observation tar archives matched the requested selection")
    return archives


def _matching_proprio_archives(config: StreamConfig) -> tuple[Path, ...]:
    episode_ids = sorted(set(config.episode_ids))
    paths: list[Path] = []
    for path in sorted((config.input_root / "proprio_stats").glob("*.tar")):
        match = _ARCHIVE_RANGE.fullmatch(path.name)
        if match is None:
            LOGGER.warning("Ignoring proprio archive with an unrecognized range: %s", path)
            continue
        if episode_ids:
            start, end = int(match["start"]), int(match["end"])
            if not any(start <= episode_id <= end for episode_id in episode_ids):
                continue
        paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError("No proprio archive covers the requested selection")
    return tuple(paths)


def prepare_local_metadata(
    config: StreamConfig,
    archives: Sequence[ObservationArchive],
) -> Path:
    """Copy task annotations and required proprio tars to PFS, then index locally."""
    task_ids = sorted({archive.task_id for archive in archives})
    task_sources = [config.input_root / "task_info" / f"task_{task_id}.json" for task_id in task_ids]
    proprio_sources = list(_matching_proprio_archives(config))
    all_sources = task_sources + proprio_sources
    total = sum(path.stat().st_size for path in all_sources)
    completed = 0
    connection = _connect_state(config.state_path)
    try:
        for source in all_sources:
            row = connection.execute(
                "SELECT status, bytes_copied, size FROM copies WHERE source_path=?", (str(source.resolve()),)
            ).fetchone()
            if row is not None:
                completed += int(row[2] if row[0] == "complete" else row[1])
    finally:
        connection.close()
    with tqdm.tqdm(
        total=total,
        initial=min(completed, total),
        desc="BOS -> PFS metadata/proprio",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for source in task_sources:
            if not source.is_file():
                raise FileNotFoundError(f"Task annotation file not found: {source}")
            destination = config.local_input_root / "task_info" / source.name
            before = destination.stat().st_size if destination.is_file() else 0
            _, _ = resumable_copy(source, destination, config.state_path, progress.update)
            if before and before < source.stat().st_size:
                # The progress initial value already included durable partial bytes, not a complete destination.
                pass
        local_proprio: list[Path] = []
        for source in proprio_sources:
            destination = config.local_input_root / "proprio_stats" / source.name
            _, _ = resumable_copy(source, destination, config.state_path, progress.update)
            local_proprio.append(destination)
    index_path = config.pfs_work_root / "state" / "proprio_index.sqlite3"
    connection = base._connect_index(index_path)  # noqa: SLF001
    try:
        requests = [(str(path.resolve()), "proprio", None) for path in local_proprio]
        pending = [request for request in requests if not base._archive_is_current(connection, Path(request[0]))]  # noqa: SLF001
        for request in tqdm.tqdm(pending, desc="Indexing PFS proprio", unit="tar", dynamic_ncols=True):
            base._store_archive_scan(connection, base._scan_archive(request))  # noqa: SLF001
    finally:
        connection.close()
    return index_path


def _episode_refs_for_archive(
    config: StreamConfig,
    archive: ObservationArchive,
    proprio_index: Path,
) -> tuple[base.EpisodeRef, ...]:
    source_text = str(archive.path.resolve())
    state = _connect_state(config.state_path)
    proprio = base._connect_index(proprio_index)  # noqa: SLF001
    annotations = base._load_annotations(config.local_input_root, {archive.task_id})  # noqa: SLF001
    episodes: list[base.EpisodeRef] = []
    try:
        rows = state.execute(
            """
            SELECT episode_id FROM episodes
            WHERE source_archive=? AND status IN ('ready', 'converting') ORDER BY episode_id
            """,
            (source_text,),
        ).fetchall()
        for (episode_id,) in rows:
            camera_rows = state.execute(
                """
                SELECT camera, path, size, member_name FROM spool_members
                WHERE task_id=? AND episode_id=? AND source_archive=? ORDER BY camera
                """,
                (archive.task_id, episode_id, source_text),
            ).fetchall()
            annotation = annotations.get((archive.task_id, episode_id))
            proprio_row = proprio.execute(
                """
                SELECT archive, name, offset, size FROM proprio_members
                WHERE task_id=? AND episode_id=? ORDER BY archive LIMIT 1
                """,
                (archive.task_id, episode_id),
            ).fetchone()
            if len(camera_rows) != len(base.CAMERA_FILES) or annotation is None or proprio_row is None:
                missing = []
                if len(camera_rows) != len(base.CAMERA_FILES):
                    missing.append("camera")
                if annotation is None:
                    missing.append("annotation")
                if proprio_row is None:
                    missing.append("proprio")
                with state:
                    state.execute(
                        "UPDATE episodes SET status='failed', error=?, updated_at=? WHERE task_id=? AND episode_id=?",
                        (f"Missing {'/'.join(missing)}", _utc_now(), archive.task_id, episode_id),
                    )
                _cleanup_episode_spool(state, archive.task_id, episode_id)
                continue
            cameras = {
                camera: base.MemberRef(path, member_name, 0, size) for camera, path, size, member_name in camera_rows
            }
            local_proprio = base.MemberRef(*proprio_row)
            original_proprio = str(config.input_root / "proprio_stats" / Path(local_proprio.archive).name)
            episodes.append(
                base.EpisodeRef(
                    task_id=archive.task_id,
                    episode_id=episode_id,
                    cameras=cameras,
                    proprio=local_proprio,
                    annotation=annotation,
                    source_observation_archive=source_text,
                    source_proprio_archive=original_proprio,
                )
            )
    finally:
        state.close()
        proprio.close()
    return tuple(episodes)


def _fragment_key(archive: ObservationArchive, index: int) -> str:
    digest = hashlib.sha256(str(archive.path.resolve()).encode()).hexdigest()[:12]
    return f"{archive.ordinal:05d}-{archive.task_id}-{digest}-{index:04d}"


def _fragment_jobs(
    config: StreamConfig,
    archive: ObservationArchive,
    episodes: Sequence[base.EpisodeRef],
    decoder: base.Decoder,
) -> tuple[FragmentJob, ...]:
    converter_config = config.converter_config()
    jobs: list[FragmentJob] = []
    for index in range(math.ceil(len(episodes) / config.episodes_per_shard)):
        selected = tuple(episodes[index * config.episodes_per_shard : (index + 1) * config.episodes_per_shard])
        key = _fragment_key(archive, index)
        fingerprint = base.plan_fingerprint(converter_config, selected)
        jobs.append(
            FragmentJob(
                key=key,
                path=config.fragment_dir / f"fragment-{key}.tfrecord",
                episodes=selected,
                config=converter_config,
                decoder=decoder,
                fingerprint=fingerprint,
            )
        )
    return tuple(jobs)


def _write_fragment(job: FragmentJob) -> FragmentResult:
    import tensorflow as tf

    with contextlib.suppress(RuntimeError):
        tf.config.set_visible_devices([], "GPU")
    start = time.monotonic()
    job.path.parent.mkdir(parents=True, exist_ok=True)
    partial = job.path.with_name(f".{job.path.name}.{os.getpid()}.partial")
    features = base.make_tfds_features(job.config)
    episode_keys: list[str] = []
    errors: list[tuple[str, str]] = []
    frames = 0
    try:
        with tf.io.TFRecordWriter(str(partial)) as writer:
            for episode in job.episodes:
                try:
                    payload, count = base._serialize_episode(episode, job.config, job.decoder, features)  # noqa: SLF001
                except Exception as exc:  # Keep valid episodes in the fragment and quarantine only the bad one.
                    errors.append((episode.key, f"{type(exc).__name__}: {exc}"))
                    continue
                writer.write(payload)
                episode_keys.append(episode.key)
                frames += count
        if not episode_keys:
            partial.unlink(missing_ok=True)
            return FragmentResult(job.key, None, (), 0, 0, None, time.monotonic() - start, tuple(errors))
        os.replace(partial, job.path)
        size = job.path.stat().st_size
        digest = _sha256_file(job.path)
        _atomic_write_json(
            job.path.with_suffix(".json"),
            {
                "fragment_key": job.key,
                "fingerprint": job.fingerprint,
                "episode_keys": episode_keys,
                "frames": frames,
                "bytes_written": size,
                "sha256": digest,
                "errors": errors,
            },
        )
        return FragmentResult(
            job.key,
            str(job.path),
            tuple(episode_keys),
            frames,
            size,
            digest,
            time.monotonic() - start,
            tuple(errors),
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _existing_fragment(job: FragmentJob) -> FragmentResult | None:
    sidecar = job.path.with_suffix(".json")
    if not job.path.is_file() or not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text())
    if payload.get("fingerprint") != job.fingerprint:
        return None
    if payload.get("episode_keys") != [episode.key for episode in job.episodes]:
        # A prior run with per-episode errors must be retried, so duplicate archives can provide a fallback.
        return None
    if payload.get("bytes_written") != job.path.stat().st_size:
        return None
    digest = _sha256_file(job.path)
    if payload.get("sha256") != digest:
        return None
    return FragmentResult(
        job.key,
        str(job.path),
        tuple(payload["episode_keys"]),
        int(payload["frames"]),
        int(payload["bytes_written"]),
        digest,
        0.0,
        tuple((str(key), str(error)) for key, error in payload.get("errors", [])),
    )


def _register_fragment_pending(
    connection: sqlite3.Connection,
    job: FragmentJob,
    archive: ObservationArchive,
    fragment_index: int,
) -> None:
    with connection:
        connection.execute(
            """
            INSERT INTO fragments(fragment_key, source_archive, fragment_index, path, status,
                                  episode_keys, fingerprint, updated_at)
            VALUES(?, ?, ?, ?, 'pending', ?, ?, ?)
            ON CONFLICT(fragment_key) DO UPDATE SET
              path=excluded.path, episode_keys=excluded.episode_keys, fingerprint=excluded.fingerprint,
              status=CASE WHEN fragments.status='complete' THEN fragments.status ELSE 'pending' END,
              updated_at=excluded.updated_at
            """,
            (
                job.key,
                str(archive.path.resolve()),
                fragment_index,
                str(job.path),
                json.dumps([episode.key for episode in job.episodes]),
                job.fingerprint,
                _utc_now(),
            ),
        )
        connection.executemany(
            "UPDATE episodes SET status='converting', updated_at=? WHERE task_id=? AND episode_id=?",
            ((_utc_now(), episode.task_id, episode.episode_id) for episode in job.episodes),
        )


def _cleanup_episode_spool(connection: sqlite3.Connection, task_id: int, episode_id: int) -> None:
    paths = [
        Path(row[0])
        for row in connection.execute(
            "SELECT path FROM spool_members WHERE task_id=? AND episode_id=?", (task_id, episode_id)
        )
    ]
    for path in paths:
        path.unlink(missing_ok=True)
    with connection:
        connection.execute("DELETE FROM spool_members WHERE task_id=? AND episode_id=?", (task_id, episode_id))
    for directory in {path.parent for path in paths}:
        with contextlib.suppress(OSError):
            directory.rmdir()


def _accept_fragment_result(
    connection: sqlite3.Connection,
    result: FragmentResult,
) -> None:
    error_keys = {key for key, _ in result.errors}
    with connection:
        connection.execute(
            """
            UPDATE fragments SET path=?, status=?, episode_keys=?, frames=?, bytes_written=?,
                                 sha256=?, errors=?, updated_at=? WHERE fragment_key=?
            """,
            (
                result.path or "",
                "complete" if result.path else "failed",
                json.dumps(result.episode_keys),
                result.frames,
                result.bytes_written,
                result.sha256,
                json.dumps(result.errors),
                _utc_now(),
                result.key,
            ),
        )
        for key in result.episode_keys:
            task_id, episode_id = map(int, key.split("/"))
            connection.execute(
                """
                UPDATE episodes SET status='converted', error=NULL, updated_at=?
                WHERE task_id=? AND episode_id=?
                """,
                (_utc_now(), task_id, episode_id),
            )
        for key, error in result.errors:
            task_id, episode_id = map(int, key.split("/"))
            connection.execute(
                """
                UPDATE episodes SET status='failed', error=?, updated_at=?
                WHERE task_id=? AND episode_id=?
                """,
                (error, _utc_now(), task_id, episode_id),
            )
    for key in (*result.episode_keys, *sorted(error_keys)):
        task_id, episode_id = map(int, key.split("/"))
        _cleanup_episode_spool(connection, task_id, episode_id)


def _spool_bytes(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COALESCE(SUM(size), 0) FROM spool_members").fetchone()[0])


def _resolve_stream_decoder(config: StreamConfig, episodes: Sequence[base.EpisodeRef]) -> base.Decoder:
    path = config.pfs_work_root / "state" / "decoder.json"
    if path.is_file():
        previous = base.Decoder(json.loads(path.read_text())["decoder"])
        if config.decoder not in {base.Decoder.AUTO, previous}:
            raise RuntimeError(
                f"Streaming workdir already uses decoder {previous.value}; requested {config.decoder.value}"
            )
        return previous
    if not episodes:
        return base.Decoder.CPU if config.decoder == base.Decoder.AUTO else config.decoder
    selected = base.select_decoder(config.converter_config(), episodes[0], available_jobs=len(episodes))
    _atomic_write_json(path, {"decoder": selected.value, "selected_at": _utc_now()})
    return selected


def _convert_worker_count(config: StreamConfig, decoder: base.Decoder) -> tuple[int, int]:
    gpu_count = base.nvidia_gpu_count() if decoder == base.Decoder.NVIDIA else 0
    if config.convert_workers:
        return config.convert_workers, gpu_count
    if decoder == base.Decoder.NVIDIA:
        return max(1, gpu_count * config.gpu_workers_per_device), gpu_count
    return base.physical_core_count(), 0


def _drain_archive_conversion(
    config: StreamConfig,
    archive: ObservationArchive,
    proprio_index: Path,
    executor: concurrent.futures.ProcessPoolExecutor,
    decoder: base.Decoder,
) -> tuple[int, int, int]:
    episodes = _episode_refs_for_archive(config, archive, proprio_index)
    if not episodes:
        return 0, 0, 0
    jobs = _fragment_jobs(config, archive, episodes, decoder)
    state = _connect_state(config.state_path)
    future_jobs: dict[concurrent.futures.Future[FragmentResult], FragmentJob] = {}
    results: list[FragmentResult] = []
    try:
        for index, job in enumerate(jobs):
            existing = _existing_fragment(job)
            _register_fragment_pending(state, job, archive, index)
            if existing is not None:
                results.append(existing)
            else:
                future_jobs[executor.submit(_write_fragment, job)] = job
        with tqdm.tqdm(
            total=len(future_jobs),
            desc=f"Converting task {archive.task_id} {archive.path.name}",
            unit="fragment",
            leave=False,
            dynamic_ncols=True,
        ) as progress:
            for future in concurrent.futures.as_completed(future_jobs):
                result = future.result()
                results.append(result)
                progress.set_postfix(episodes=len(result.episode_keys), frames=result.frames, refresh=False)
                progress.update()
        for result in results:
            _accept_fragment_result(state, result)
        return (
            sum(len(result.episode_keys) for result in results),
            sum(result.frames for result in results),
            sum(result.bytes_written for result in results),
        )
    finally:
        state.close()


def stream_and_convert(
    config: StreamConfig,
    archives: Sequence[ObservationArchive],
    proprio_index: Path,
) -> dict[str, int]:
    """Stream archives with per-task ordering and convert each completed archive on PFS."""
    # Decoder auto-selection needs one real spooled episode. The first archive is therefore streamed synchronously.
    first = archives[0]
    byte_total = sum(archive.path.stat().st_size for archive in archives)
    state = _connect_state(config.state_path)
    try:
        initial_bytes = sum(
            min(int(row[1]), int(row[0]))
            for row in state.execute("SELECT size, CASE WHEN status='complete' THEN size ELSE offset END FROM archives")
        )
    finally:
        state.close()
    totals = {"episodes": 0, "frames": 0, "bytes": 0, "archives": 0}
    with tqdm.tqdm(
        total=byte_total,
        initial=min(initial_bytes, byte_total),
        desc="Sequential BOS observation stream",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as byte_progress:
        scan_observation_archive(config, first, byte_progress.update)
        first_episodes = _episode_refs_for_archive(config, first, proprio_index)
        decoder = _resolve_stream_decoder(config, first_episodes)
        workers, gpu_count = _convert_worker_count(config, decoder)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=base._worker_initialize,  # noqa: SLF001
            initargs=(decoder, gpu_count),
        ) as convert_executor:
            pending = list(archives[1:])
            # One archive per task stays active. Conversion finishes before the next archive for that task,
            # which makes duplicate fallback deterministic.
            with (
                concurrent.futures.ThreadPoolExecutor(max_workers=config.stream_readers) as scan_executor,
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(2, config.stream_readers * 2)
                ) as conversion_coordinators,
            ):
                scan_futures: dict[concurrent.futures.Future[ScanResult], ObservationArchive] = {}
                conversion_futures: dict[concurrent.futures.Future[tuple[int, int, int]], ObservationArchive] = {
                    conversion_coordinators.submit(
                        _drain_archive_conversion,
                        config,
                        first,
                        proprio_index,
                        convert_executor,
                        decoder,
                    ): first
                }
                active_tasks: set[int] = {first.task_id}
                progress_lock = threading.Lock()

                def report(delta: int) -> None:
                    with progress_lock:
                        byte_progress.update(delta)

                while pending or scan_futures or conversion_futures:
                    state = _connect_state(config.state_path)
                    try:
                        spool_full = _spool_bytes(state) >= config.spool_limit_gib * 2**30
                    finally:
                        state.close()
                    if not spool_full:
                        for candidate in list(pending):
                            if len(scan_futures) >= config.stream_readers:
                                break
                            if candidate.task_id in active_tasks:
                                continue
                            pending.remove(candidate)
                            active_tasks.add(candidate.task_id)
                            scan_futures[scan_executor.submit(scan_observation_archive, config, candidate, report)] = (
                                candidate
                            )
                    if not scan_futures and pending and spool_full and not conversion_futures:
                        raise RuntimeError(
                            "PFS spool limit reached without an active conversion; inspect failed episodes with --status"
                        )
                    all_futures = set(scan_futures) | set(conversion_futures)
                    if not all_futures:
                        continue
                    done, _ = concurrent.futures.wait(
                        all_futures, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done & set(scan_futures):
                        archive = scan_futures.pop(future)
                        try:
                            future.result()
                            conversion_future = conversion_coordinators.submit(
                                _drain_archive_conversion,
                                config,
                                archive,
                                proprio_index,
                                convert_executor,
                                decoder,
                            )
                            conversion_futures[conversion_future] = archive
                        except BaseException:
                            active_tasks.remove(archive.task_id)
                            raise
                    for future in done & set(conversion_futures):
                        archive = conversion_futures.pop(future)
                        try:
                            counts = future.result()
                            totals["episodes"] += counts[0]
                            totals["frames"] += counts[1]
                            totals["bytes"] += counts[2]
                            totals["archives"] += 1
                        finally:
                            active_tasks.remove(archive.task_id)
                    if config.max_episodes is not None and totals["episodes"] >= config.max_episodes:
                        pending.clear()
    return totals


def _pipeline_fingerprint(config: StreamConfig, rows: Sequence[sqlite3.Row | tuple[Any, ...]]) -> str:
    payload = {
        "schema": base.DATASET_SCHEMA_VERSION,
        "dataset_name": config.dataset_name,
        "version": config.version,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "jpeg_quality": config.jpeg_quality,
        "alignment_tolerance": config.alignment_tolerance,
        "fragments": [list(row) for row in rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def finalize_local_dataset(config: StreamConfig) -> dict[str, Any]:
    """Link verified PFS fragments into deterministic TFDS shard names and validate metadata."""
    state = _connect_state(config.state_path)
    try:
        failed_episodes = state.execute("SELECT COUNT(*) FROM episodes WHERE status='failed'").fetchone()[0]
        failed_archives = state.execute("SELECT COUNT(*) FROM archives WHERE status='failed'").fetchone()[0]
        if (failed_episodes or failed_archives) and not config.allow_incomplete:
            raise RuntimeError(
                f"Refusing to finalize with failed episodes={failed_episodes}, failed archives={failed_archives}; "
                "inspect --status or explicitly pass --allow-incomplete"
            )
        rows = state.execute(
            """
            SELECT f.fragment_key, f.path, f.episode_keys, f.frames, f.bytes_written, f.sha256, a.ordinal,
                   f.fragment_index
            FROM fragments f JOIN archives a ON a.source_path=f.source_archive
            WHERE f.status='complete' ORDER BY a.ordinal, f.fragment_index, f.fragment_key
            """
        ).fetchall()
    finally:
        state.close()
    if not rows:
        raise RuntimeError("No verified PFS fragments are available for finalization")
    fingerprint = _pipeline_fingerprint(config, rows)
    config.local_version_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    results: list[base.ShardResult] = []
    for index, row in enumerate(rows):
        source = Path(row[1])
        if source.stat().st_size != row[4] or _sha256_file(source) != row[5]:
            raise RuntimeError(f"Fragment verification failed before finalization: {source}")
        name = f"{config.dataset_name}-train.tfrecord-{index:05d}-of-{len(rows):05d}"
        expected_names.add(name)
        destination = config.local_version_dir / name
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size or _sha256_file(destination) != row[5]:
                raise RuntimeError(f"Conflicting finalized shard already exists: {destination}")
        else:
            os.link(source, destination)
        results.append(
            base.ShardResult(
                shard_index=index,
                path=str(destination),
                episode_keys=tuple(json.loads(row[2])),
                frames=int(row[3]),
                bytes_written=int(row[4]),
                elapsed_seconds=0.0,
                errors=(),
            )
        )
    stale = [
        path
        for path in config.local_version_dir.glob(f"{config.dataset_name}-train.tfrecord-*")
        if path.name not in expected_names
    ]
    if stale:
        raise RuntimeError(f"Stale finalized shards require a fresh version/workdir: {stale[:3]}")
    manifest = base.finalize_dataset(config.converter_config(), results, fingerprint)
    manifest["pipeline"] = "sequential_bos_pfs_stream_v1"
    manifest["pfs_fragments"] = len(rows)
    manifest["failed_episodes"] = failed_episodes
    manifest["failed_archives"] = failed_archives
    base._atomic_write_json(config.local_version_dir / "conversion_manifest.json", manifest)  # noqa: SLF001
    validate_tfds_directory(config.local_version_dir, expected_episodes=int(manifest["episodes"]))
    return manifest


def validate_tfds_directory(path: Path, *, expected_episodes: int | None = None) -> dict[str, int]:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(path)
    episodes = int(builder.info.splits["train"].num_examples)
    if expected_episodes is not None and episodes != expected_episodes:
        raise RuntimeError(f"TFDS reports {episodes} episodes at {path}, expected {expected_episodes}")
    shards = sorted(path.glob("*.tfrecord-*"))
    if not shards:
        raise RuntimeError(f"No TFRecord shards found at {path}")
    for sample in dict.fromkeys((shards[0], shards[len(shards) // 2], shards[-1])):
        count = sum(1 for _ in tf.data.TFRecordDataset(str(sample)).take(1))
        if count != 1:
            raise RuntimeError(f"TFRecord shard has no readable examples: {sample}")
    return {"episodes": episodes, "shards": len(shards)}


def _publish_file_order(version_dir: Path) -> list[Path]:
    files = [path for path in version_dir.iterdir() if path.is_file()]
    return sorted(
        files, key=lambda path: (not path.name.startswith(f"{path.parent.parent.name}-train.tfrecord-"), path.name)
    )


def publish_to_bos(config: StreamConfig, manifest: dict[str, Any]) -> Path:
    """Upload to a hidden BOS directory, validate readback, then atomically publish."""
    final = config.bos_version_dir
    hidden = final.parent / f".{config.version}.uploading"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        existing = validate_tfds_directory(final, expected_episodes=int(manifest["episodes"]))
        LOGGER.info("BOS output already exists and validates: %s", existing)
        return final
    hidden.mkdir(parents=True, exist_ok=True)
    files = _publish_file_order(config.local_version_dir)
    # TFRecords go first; TFDS metadata and the manifest are copied only after every shard is durable.
    metadata_names = {"dataset_info.json", "features.json", "conversion_manifest.json", "LICENSE"}
    ordered = [path for path in files if path.name not in metadata_names]
    ordered += [path for path in files if path.name in metadata_names]
    total = sum(path.stat().st_size for path in ordered)
    with tqdm.tqdm(total=total, desc="PFS -> BOS publish", unit="B", unit_scale=True, dynamic_ncols=True) as progress:
        for source in ordered:
            destination = hidden / source.name
            _, digest = resumable_copy(source, destination, config.state_path, progress.update)
            state = _connect_state(config.state_path)
            try:
                with state:
                    state.execute(
                        """
                        INSERT OR REPLACE INTO uploads(source_path, destination, size, sha256, status,
                                                       bytes_copied, updated_at)
                        VALUES(?, ?, ?, ?, 'complete', ?, ?)
                        """,
                        (
                            str(source),
                            str(destination),
                            source.stat().st_size,
                            digest,
                            source.stat().st_size,
                            _utc_now(),
                        ),
                    )
            finally:
                state.close()
    validate_tfds_directory(hidden, expected_episodes=int(manifest["episodes"]))
    ready = hidden / "READY"
    ready.write_text(json.dumps({"validated_at": _utc_now(), "manifest": manifest}, sort_keys=True) + "\n")
    os.replace(hidden, final)
    validate_tfds_directory(final, expected_episodes=int(manifest["episodes"]))
    return final


def status(config: StreamConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        return {"state": "not_started", "state_path": str(config.state_path)}
    connection = sqlite3.connect(f"file:{config.state_path}?mode=ro", uri=True, timeout=60)
    try:
        result: dict[str, Any] = {
            "state_path": str(config.state_path),
            "spool_bytes": _spool_bytes(connection),
            "archives": dict(connection.execute("SELECT status, COUNT(*) FROM archives GROUP BY status")),
            "episodes": dict(connection.execute("SELECT status, COUNT(*) FROM episodes GROUP BY status")),
            "fragments": dict(connection.execute("SELECT status, COUNT(*) FROM fragments GROUP BY status")),
            "copies": dict(connection.execute("SELECT status, COUNT(*) FROM copies GROUP BY status")),
            "uploads": dict(connection.execute("SELECT status, COUNT(*) FROM uploads GROUP BY status")),
        }
        archive_bytes = connection.execute(
            "SELECT COALESCE(SUM(size), 0), COALESCE(SUM(CASE WHEN status IN ('complete', 'selection_complete') THEN size ELSE offset END), 0) FROM archives"
        ).fetchone()
        result["archive_bytes_total"] = int(archive_bytes[0])
        result["archive_bytes_processed"] = int(archive_bytes[1])
        errors = connection.execute(
            """
            SELECT 'archive', source_path, error FROM archives WHERE error IS NOT NULL
            UNION ALL SELECT 'episode', task_id || '/' || episode_id, error FROM episodes WHERE error IS NOT NULL
            ORDER BY 1, 2 LIMIT 20
            """
        ).fetchall()
        result["recent_errors"] = [list(row) for row in errors]
        return result
    finally:
        connection.close()


def run_pipeline(config: StreamConfig, *, lineage_id: str | None = None) -> dict[str, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    config.validate()
    _assert_config_identity(config)
    _recover_interrupted_state(config)
    archives = discover_observation_archives(config)
    LOGGER.info(
        "Streaming plan: archives=%d readers=%d spool_limit=%d GiB workdir=%s",
        len(archives),
        config.stream_readers,
        config.spool_limit_gib,
        config.pfs_work_root,
    )
    proprio_index = prepare_local_metadata(config, archives)
    totals = stream_and_convert(config, archives, proprio_index)
    LOGGER.info("PFS conversion totals: %s", totals)
    manifest = finalize_local_dataset(config)
    if lineage_id is not None:
        manifest["lineage_id"] = lineage_id
        base._atomic_write_json(config.local_version_dir / "conversion_manifest.json", manifest)  # noqa: SLF001
    destination = publish_to_bos(config, manifest) if config.publish else config.local_version_dir
    LOGGER.info("Streaming conversion complete and verified: %s", destination)
    return manifest
