"""SQLite records, scenario storage, and retained-file maintenance."""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    REAL,
    URL,
    Column,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
    cast,
    create_engine,
    func,
    inspect,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from reef.core.errors import ArchiveConflict, ArchiveCorrupt, ReefError
from reef.storage.commit_log import CommitLog, CommitLogScenarioStore, rewrite_commit_log_scenario
from reef.storage.records import RecordRetention
from reef.storage.scenario import ScenarioStorage, StorageArchive
from reef.storage.sql_records import RecordTables, SQLRecordRetention, SQLRecordStore

_METADATA = MetaData()
_AGENT_RECORD = Table(
    "agent_record",
    _METADATA,
    Column("sequence", Integer, primary_key=True, nullable=True),
    Column("agent_record_id", Text, nullable=False, unique=True),
    Column("scenario", Text, nullable=False),
    Column("request_type", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("created_at", REAL, nullable=False),
    Column("references_json", Text, nullable=False),
    Column("artifact_json", Text),
    Column("compacted_at", REAL),
    Column("body_bytes", Integer, nullable=False, server_default="0"),
    # A purged sequence must never be reused by a later append.
    sqlite_autoincrement=True,
)
_CONSUMED_RECORD = Table(
    "consumed_agent_record",
    _METADATA,
    Column("agent_record_id", Text, primary_key=True, nullable=True),
    Column("content_sha256", Text, nullable=False),
)
_COMPACTION_RECEIPTS = Table(
    "compaction_receipts",
    _METADATA,
    Column("scenario", Text, primary_key=True),
    Column("receipt_id", Text, primary_key=True),
    Column("compacted_ids_json", Text, primary_key=True),
    Column("metadata_json", Text, nullable=False),
    Column("recorded_at", REAL, nullable=False),
)
Index("agent_record_scenario_sequence", _AGENT_RECORD.c.scenario, _AGENT_RECORD.c.sequence)
Index(
    "agent_record_scenario_type_sequence",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.request_type,
    _AGENT_RECORD.c.sequence,
)
Index(
    "agent_record_active_sequence",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.sequence,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_(None),
)
Index(
    "agent_record_active_type_sequence",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.request_type,
    _AGENT_RECORD.c.sequence,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_(None),
)
Index(
    "agent_record_compacted_at",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.compacted_at,
    _AGENT_RECORD.c.sequence,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_not(None),
)
Index(
    "agent_record_retention",
    _AGENT_RECORD.c.compacted_at,
    _AGENT_RECORD.c.sequence,
    _AGENT_RECORD.c.body_bytes,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_not(None),
)

_TABLES = RecordTables(_AGENT_RECORD, _CONSUMED_RECORD, _COMPACTION_RECEIPTS)


class SQLiteRecordStore(SQLRecordStore):
    """Store scenario records in append order.

    The SQLite schema uses an ``agent_record`` table with an
    ``agent_record_id`` column, mirroring the wire id
    (``x-reef-agent-record-id``).

    Passing a filesystem path makes the store durable. The default in-memory
    database keeps standalone/test construction lightweight; production callers
    should always pass a path.

    Training reads hide compacted rows. Explicit audit reads retain access to
    their bodies until :meth:`purge_compacted` physically removes them.
    """

    _SQLITE_ID_CHUNK_SIZE = 900
    _WAL_SWITCH_TIMEOUT = 30.0
    _WAL_RETRY_INTERVAL = 0.01

    def __init__(self, database: str | Path | None = None) -> None:
        self._database = ":memory:" if database is None else str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        super().__init__(_TABLES, id_chunk_size=self._SQLITE_ID_CHUNK_SIZE)
        self._engine = create_engine(
            URL.create("sqlite+pysqlite", database=self._database),
            connect_args={"timeout": 30, "check_same_thread": False},
            poolclass=NullPool,
            hide_parameters=True,
        )
        # Hold one connection for the store's lifetime, including for :memory:.
        # The lock serializes all use across HTTP and training threads.
        self._connection = self._engine.connect()
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    @property
    def database(self) -> str:
        return self._database

    def _enable_wal(self) -> None:
        """Switch the database to WAL, waiting out a concurrent opener.

        SQLite takes an exclusive lock to change the journal mode and answers
        SQLITE_BUSY immediately instead of running the busy handler, so the
        connection timeout does not cover this statement. Openers that race to
        upgrade a database still in rollback mode therefore have to retry: the
        first one converts it and the rest read back ``wal`` on a later attempt.
        A database already in WAL answers on the first try, so the steady state
        costs nothing.
        """
        deadline = time.monotonic() + self._WAL_SWITCH_TIMEOUT
        observed = None
        while True:
            # A contended switch shows up either way: as SQLITE_BUSY, or as the
            # unchanged mode read back, which SQLite returns instead of raising.
            try:
                row = self._connection.exec_driver_sql("PRAGMA journal_mode = WAL").fetchone()
            except OperationalError as exc:
                if "locked" not in str(exc.orig) and "busy" not in str(exc.orig):
                    raise
            else:
                observed = None if row is None else str(row[0]).lower()
                if observed == "wal":
                    return
            if time.monotonic() >= deadline:
                raise ReefError(
                    f"could not switch {self._database} to WAL within {self._WAL_SWITCH_TIMEOUT:g}s; "
                    f"another connection held the database (journal mode is {observed or 'unknown'})"
                )
            time.sleep(self._WAL_RETRY_INTERVAL)

    def _initialize(self) -> None:
        with self._lock:
            if self._database != ":memory:":
                self._enable_wal()
                self._connection.exec_driver_sql("PRAGMA synchronous = FULL")
                self._connection.commit()
            with self._connection.begin():
                # Keep Alembic's column upgrades and the Core backfill in one
                # SQLite transaction, serialized across concurrent openers.
                self._connection.exec_driver_sql("BEGIN IMMEDIATE")
                if inspect(self._connection).has_table(_AGENT_RECORD.name):
                    columns = {column["name"] for column in inspect(self._connection).get_columns(_AGENT_RECORD.name)}
                    if not {"compacted_at", "body_bytes"} <= columns:
                        operations = Operations(MigrationContext.configure(self._connection))
                        if "compacted_at" not in columns:
                            operations.add_column(_AGENT_RECORD.name, Column("compacted_at", REAL))
                        if "body_bytes" not in columns:
                            operations.add_column(
                                _AGENT_RECORD.name, Column("body_bytes", Integer, nullable=False, server_default="0")
                            )
                            self._connection.execute(
                                _AGENT_RECORD.update().values(
                                    body_bytes=func.length(cast(_AGENT_RECORD.c.payload_json, LargeBinary))
                                    + func.length(cast(_AGENT_RECORD.c.references_json, LargeBinary))
                                    + func.coalesce(func.length(cast(_AGENT_RECORD.c.artifact_json, LargeBinary)), 0)
                                )
                            )
                _METADATA.create_all(self._connection)
                # create_all skips indexes on tables that already existed.
                for index in _AGENT_RECORD.indexes:
                    index.create(self._connection, checkfirst=True)

    @contextmanager
    def _transaction(self, scenario: str, *, write: bool) -> Iterator[Connection]:
        with self._lock, self._connection.begin():
            # The driver's legacy mode does not begin a transaction for SELECT.
            # Writes also lock before reading retry state across connections.
            self._connection.exec_driver_sql("BEGIN IMMEDIATE" if write else "BEGIN")
            yield self._connection

    def _insert_if_absent(
        self,
        connection: Connection,
        table: Table,
        values: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> bool:
        return connection.execute(table.insert().prefix_with("OR IGNORE"), values).rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._connection.close()
            self._engine.dispose()

    def __enter__(self) -> SQLiteRecordStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class SQLiteRecordRetention:
    """Apply a deployment's body-retention limits to its SQLite record files.

    The scenario storage service serializes cleanup with file archival.
    """

    def __init__(self, retention: RecordRetention) -> None:
        self._retention = retention
        self._queries = SQLRecordRetention(_TABLES)

    def prune(self, directory: Path) -> int:
        """Purge expired bodies, then the oldest bodies across this directory to meet the budget.

        The caller must serialize this sweep with scenario file moves. Deletes
        commit in batches of 256 and never touch active records or retry metadata.
        Concurrent compaction may exceed the budget until the next sweep.
        """
        cutoff = time.time() - self._retention.days * 86400
        paths = sorted((*directory.glob("*.sqlite3"), *(directory / "archived").rglob("*.sqlite3")))
        purged = 0
        total = 0
        retained_paths: list[str] = []
        for database_path in paths:
            with self._connect(str(database_path)) as connection:
                with connection.begin():
                    inspector = inspect(connection)
                    columns = (
                        {column["name"] for column in inspector.get_columns(_AGENT_RECORD.name)}
                        if inspector.has_table(_AGENT_RECORD.name)
                        else set()
                    )
                if not {"compacted_at", "body_bytes"} <= columns:
                    # Old stores have no retained compacted bodies; migration belongs to SQLiteRecordStore.
                    continue
                retained_paths.append(str(database_path))
                while True:
                    with connection.begin():
                        count = self._queries.purge_expired(connection, before=cutoff)
                    purged += count
                    if count < 256:
                        break

                with connection.begin():
                    total += self._queries.retained_bytes(connection)
        if total <= self._retention.max_bytes:
            return purged
        pending: dict[str, list[int]] = {path: [] for path in retained_paths}
        rows = heapq.merge(*(self._rows(path) for path in retained_paths))
        for _, path, sequence, size in rows:
            pending[path].append(sequence)
            total -= size
            if len(pending[path]) == 256:
                purged += self._delete(path, pending[path])
                pending[path].clear()
            if total <= self._retention.max_bytes:
                break
        for path, sequences in pending.items():
            if sequences:
                purged += self._delete(path, sequences)
        return purged

    @staticmethod
    @contextmanager
    def _connect(path: str) -> Iterator[Connection]:
        # URI mode=rw prevents maintenance from recreating a moved database.
        engine = create_engine(
            URL.create("sqlite+pysqlite", database=Path(path).resolve().as_uri(), query={"mode": "rw", "uri": "true"}),
            connect_args={"timeout": 30},
            poolclass=NullPool,
            hide_parameters=True,
        )
        try:
            with engine.connect() as connection:
                yield connection
        finally:
            engine.dispose()

    def _rows(self, path: str) -> Iterator[tuple[float, str, int, int]]:
        after_time: float = -math.inf
        after_sequence = 0
        while True:
            with self._connect(path) as connection, connection.begin():
                rows = self._queries.page(connection, after_time=after_time, after_sequence=after_sequence)
            if not rows:
                return
            for compacted_at, sequence, size in rows:
                yield compacted_at, path, sequence, size
            after_time, after_sequence = rows[-1][:2]

    def _delete(self, path: str, sequences: list[int]) -> int:
        with self._connect(path) as connection, connection.begin():
            return self._queries.delete(connection, sequences)


class SQLiteScenarioStorage(ScenarioStorage):
    """Open existing hashed SQLite and JSONL paths and manage their lifecycle."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._lock = RLock()
        self._closed = False
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def durable(self) -> bool:
        return self._directory is not None

    def open(self, scenario: str) -> CommitLogScenarioStore:
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            database = None if self._directory is None else self._directory / f"{key}.sqlite3"
            commit_log = None if self._directory is None else CommitLog(self._directory / f"{key}.commits.jsonl")
            records = SQLiteRecordStore(database)
            try:
                return CommitLogScenarioStore(scenario, records, commit_log)
            except BaseException:
                records.close()
                raise

    def archive(self, scenario: str) -> StorageArchive | None:
        with self._lock:
            self._ensure_open()
            paths = self.state_paths(scenario)
            if self._directory is None:
                return None
            destination = (
                self._directory
                / "archived"
                / f"{self._scenario_key(scenario)}-{self._archive_stamp()}-{uuid.uuid4().hex}"
            )
            moved: list[str] = []
            for path in paths:
                if path.exists():
                    destination.mkdir(parents=True, exist_ok=True)
                    target = destination / path.name
                    shutil.move(str(path), str(target))
                    moved.append(str(target))
            if not moved:
                return None
            return StorageArchive(locations=tuple(moved), token={"archived_dir": str(destination)})

    def restore(
        self,
        archive: StorageArchive,
        *,
        source: str,
        target: str,
        settlement_receipt: str,
    ) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            if self._directory is None:
                raise ArchiveCorrupt("in-memory storage has no archived generation to restore")
            archived_dir = Path(archive.token.get("archived_dir", ""))
            if not archived_dir.is_dir():
                raise ArchiveCorrupt(f"archived storage directory is missing: {archived_dir}")
            source_key = self._scenario_key(source)
            target_key = self._scenario_key(target)
            active_paths = tuple(
                self._directory / f"{target_key}{suffix}"
                for suffix in (".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".commits.jsonl")
            )
            if any(path.exists() for path in active_paths):
                raise ArchiveConflict(
                    f"scenario {target!r} already has an active record store; refusing to overwrite it"
                )
            staged_database, staged_commit_log = self._stage_archived_generation(
                archived_dir, source_key, target_key, source=source, target=target
            )
            try:
                self._settle_restored_generation(
                    staged_database,
                    target,
                    source=source,
                    rename=source != target,
                    settlement_receipt=settlement_receipt,
                )
                database_destination = self._directory / f"{target_key}.sqlite3"
                os.replace(staged_database, database_destination)
                log_destination: Path | None = None
                if staged_commit_log is not None:
                    log_destination = self._directory / f"{target_key}.commits.jsonl"
                    os.replace(staged_commit_log, log_destination)
            finally:
                shutil.rmtree(staged_database.parent, ignore_errors=True)
            restored = [str(database_destination)]
            if log_destination is not None:
                restored.append(str(log_destination))
            return tuple(restored)

    def _stage_archived_generation(
        self,
        archived_dir: Path,
        source_key: str,
        target_key: str,
        *,
        source: str,
        target: str,
    ) -> tuple[Path, Path | None]:
        """Copy an archived generation into a staging directory, rekeyed for the target name."""
        if self._directory is None:
            raise ArchiveCorrupt("in-memory storage has no directory to stage a restore in")
        directory = self._directory
        archived_database = archived_dir / f"{source_key}.sqlite3"
        archived_commit_log = archived_dir / f"{source_key}.commits.jsonl"
        if not archived_database.exists() and not archived_commit_log.exists():
            raise ArchiveCorrupt(f"archived storage directory holds no generation for {source!r}: {archived_dir}")
        staging = directory / ".restore-staging" / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=False)
        staged_database = staging / f"{target_key}.sqlite3"
        staged_commit_log: Path | None = None
        try:
            if archived_database.exists():
                shutil.copy2(archived_database, staged_database)
            if archived_commit_log.exists():
                staged_commit_log = staging / f"{target_key}.commits.jsonl"
                if source == target:
                    shutil.copy2(archived_commit_log, staged_commit_log)
                else:
                    rewrite_commit_log_scenario(archived_commit_log, staged_commit_log, source=source, target=target)
            return staged_database, staged_commit_log
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _settle_restored_generation(
        self,
        staged_database: Path,
        target: str,
        *,
        source: str,
        rename: bool,
        settlement_receipt: str,
    ) -> None:
        """Rewrite scenario identity and retire every restored record from training replay.

        Restored rows stay readable through audit pages and keep their append
        deduplication, but none of them can form a training batch: the scenario
        is recovered with its committed watermark, so the records it archived
        never re-enter the training queue. The archived database itself is not
        touched; this runs on the staging copy.
        """
        if not staged_database.exists():
            return
        records = SQLiteRecordStore(staged_database)
        try:
            if rename:
                with records._transaction(target, write=True) as connection:
                    connection.execute(
                        _AGENT_RECORD.update().where(_AGENT_RECORD.c.scenario == source).values(scenario=target)
                    )
                    connection.execute(
                        _COMPACTION_RECEIPTS.update()
                        .where(_COMPACTION_RECEIPTS.c.scenario == source)
                        .values(scenario=target)
                    )
            retired: set[str] = set()
            after_sequence = 0
            while True:
                # Audit pages include already-compacted rows; the cursor is the sequence.
                rows = records.audit_page(target, after_sequence=after_sequence, limit=512)
                if not rows:
                    break
                for stored in rows:
                    if stored.compacted_at is None:
                        retired.add(stored.item.agent_record_id)
                after_sequence = rows[-1].sequence
                if len(rows) < 512:
                    break
            if retired:
                records.compact(
                    target,
                    frozenset(retired),
                    receipt_id=settlement_receipt,
                    receipt_metadata={"kind": "archive-restore", "source_scenario": source, "restored_as": target},
                )
            # Outside a write transaction: wal_checkpoint cannot run inside one. Flush the WAL into
            # the database file so placing just the .sqlite3 copy loses nothing.
            with records._lock:
                records._connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
                records._connection.commit()
        finally:
            records.close()

    def prune(self, *, days: float, max_bytes: int) -> int:
        with self._lock:
            self._ensure_open()
            retention = RecordRetention(days, max_bytes)
            return 0 if self._directory is None else SQLiteRecordRetention(retention).prune(self._directory)

    def rollback_restore(
        self,
        archive: StorageArchive,
        *,
        target: str,
        restored_locations: tuple[str, ...],
    ) -> None:
        with self._lock:
            for location in restored_locations:
                path = Path(location)
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{path}{suffix}").unlink(missing_ok=True)

    def state_paths(self, scenario: str) -> tuple[Path, ...]:
        """Existing local state paths; the stable process lock is never moved."""
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            if self._directory is None:
                return ()
            return tuple(
                self._directory / name
                for name in (f"{key}.sqlite3", f"{key}.sqlite3-wal", f"{key}.sqlite3-shm", f"{key}.commits.jsonl")
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")

    @staticmethod
    def _scenario_key(scenario: str) -> str:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        return hashlib.sha256(scenario.encode("utf-8")).hexdigest()

    @staticmethod
    def _archive_stamp() -> str:
        return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
