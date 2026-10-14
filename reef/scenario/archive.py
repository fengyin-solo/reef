"""Auditable manifests for deleted scenarios and their cross-process restore fencing.

Deleting a scenario moves its registration, record store, commit log, model
settings, and recipe directories under various ``archived`` locations. The
:class:`ArchiveCatalog` ties those locations together in one JSON manifest
that records who deleted what, when, from which release, and from which
process. The manifest is the only thing restore reads; its summary is all a
list endpoint returns.

Manifests live under ``<record-dir>/archived/manifests`` and POSIX file locks
under ``<record-dir>/.locks`` serialize delete, list, and restore across
threads and processes. With no record directory the catalog keeps manifests
in memory for the process lifetime, which matches a deployment whose storage
itself is not durable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from reef.artifact.repository import RegistrationArchive
from reef.core.errors import ArchiveConflict, ArchiveCorrupt, UnknownArchive
from reef.storage.scenario import StorageArchive

ARCHIVE_FORMAT = "reef-scenario-archive/1"
_ARCHIVE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def utc_stamp(epoch: float | None = None) -> str:
    moment = time.time() if epoch is None else epoch
    return datetime.fromtimestamp(moment, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ArchiveSourceProcess:
    """The process that deleted (or restored) the scenario."""

    pid: int
    host: str
    process_id: str

    def to_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "host": self.host, "process_id": self.process_id}

    @classmethod
    def from_dict(cls, value: object) -> ArchiveSourceProcess:
        if not isinstance(value, dict):
            raise ValueError("source process must be an object")
        pid = value.get("pid")
        host = value.get("host")
        process_id = value.get("process_id")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
            raise ValueError("source process pid must be a non-negative integer")
        if not isinstance(host, str) or not host:
            raise ValueError("source process host must be a non-empty string")
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("source process id must be a non-empty string")
        return cls(pid=pid, host=host, process_id=process_id)


@dataclass(frozen=True)
class ArchivedRelease:
    """The release the scenario served when it was deleted."""

    release_id: str
    content_id: str

    def to_dict(self) -> dict[str, object]:
        return {"release_id": self.release_id, "content_id": self.content_id}

    @classmethod
    def from_dict(cls, value: object) -> ArchivedRelease:
        if not isinstance(value, dict):
            raise ValueError("release must be an object")
        release_id = value.get("release_id")
        content_id = value.get("content_id")
        if not isinstance(release_id, str) or not release_id:
            raise ValueError("release id must be a non-empty string")
        if not isinstance(content_id, str) or not content_id:
            raise ValueError("content id must be a non-empty string")
        return cls(release_id=release_id, content_id=content_id)


@dataclass(frozen=True)
class ArchivedRecipeDir:
    """One recipe-owned directory that moved beside itself."""

    name: str
    parent: str
    archived: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "parent": self.parent, "archived": self.archived}

    @classmethod
    def from_dict(cls, value: object) -> ArchivedRecipeDir:
        if not isinstance(value, dict):
            raise ValueError("recipe directory entry must be an object")
        name = value.get("name")
        parent = value.get("parent")
        archived = value.get("archived")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(parent, str)
            or not parent
            or not isinstance(archived, str)
            or not archived
        ):
            raise ValueError("recipe directory entry requires non-empty name, parent, and archived paths")
        return cls(name=name, parent=parent, archived=archived)


@dataclass(frozen=True)
class ArchiveRestoration:
    """Who consumed an archive by restoring it."""

    restored_as: str
    restored_at: str
    restored_at_epoch: float
    source_process: ArchiveSourceProcess

    def to_dict(self) -> dict[str, object]:
        return {
            "restored_as": self.restored_as,
            "restored_at": self.restored_at,
            "restored_at_epoch": self.restored_at_epoch,
            "source_process": self.source_process.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ArchiveRestoration:
        if not isinstance(value, dict):
            raise ValueError("restoration must be an object")
        restored_as = value.get("restored_as")
        restored_at = value.get("restored_at")
        restored_at_epoch = value.get("restored_at_epoch")
        if not isinstance(restored_as, str) or not restored_as:
            raise ValueError("restoration requires restored_as")
        if not isinstance(restored_at, str) or not restored_at:
            raise ValueError("restoration requires restored_at")
        if not isinstance(restored_at_epoch, (int, float)) or isinstance(restored_at_epoch, bool):
            raise ValueError("restoration epoch must be a number")
        return cls(
            restored_as=restored_as,
            restored_at=restored_at,
            restored_at_epoch=float(restored_at_epoch),
            source_process=ArchiveSourceProcess.from_dict(value.get("source_process")),
        )


@dataclass
class ScenarioArchive:
    """The auditable summary and restore coordinates of one deleted scenario."""

    archive_id: str
    scenario: str
    deleted_at: str
    deleted_at_epoch: float
    source_process: ArchiveSourceProcess
    release: ArchivedRelease
    storage: StorageArchive | None
    registration: RegistrationArchive | None
    model_config: tuple[str, ...]
    recipe_dirs: tuple[ArchivedRecipeDir, ...]
    status: str = "archived"
    restoration: ArchiveRestoration | None = None

    @property
    def locations(self) -> tuple[str, ...]:
        entries: list[str] = list(self.model_config)
        if self.storage is not None:
            entries.extend(self.storage.locations)
        if self.registration is not None:
            entries.extend(self.registration.locations)
        entries.extend(item.archived for item in self.recipe_dirs)
        return tuple(entries)

    @property
    def consumed(self) -> bool:
        return self.status == "consumed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ARCHIVE_FORMAT,
            "archive_id": self.archive_id,
            "scenario": self.scenario,
            "deleted_at": self.deleted_at,
            "deleted_at_epoch": self.deleted_at_epoch,
            "source_process": self.source_process.to_dict(),
            "release": self.release.to_dict(),
            "storage": None if self.storage is None else self.storage.to_dict(),
            "registration": None if self.registration is None else self.registration.to_dict(),
            "model_config": list(self.model_config),
            "recipe_dirs": [item.to_dict() for item in self.recipe_dirs],
            "locations": list(self.locations),
            "status": self.status,
            "restoration": None if self.restoration is None else self.restoration.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ScenarioArchive:
        if not isinstance(value, dict):
            raise ValueError("archive manifest must be an object")
        if value.get("format") != ARCHIVE_FORMAT:
            raise ValueError(f"unsupported archive manifest format: {value.get('format')!r}")
        archive_id = value.get("archive_id")
        scenario = value.get("scenario")
        deleted_at = value.get("deleted_at")
        deleted_at_epoch = value.get("deleted_at_epoch")
        if not isinstance(archive_id, str) or not _ARCHIVE_ID_PATTERN.fullmatch(archive_id):
            raise ValueError("archive id must be 32 hex characters")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("archive manifest requires a scenario name")
        if not isinstance(deleted_at, str) or not deleted_at:
            raise ValueError("archive manifest requires a deletion time")
        if not isinstance(deleted_at_epoch, (int, float)) or isinstance(deleted_at_epoch, bool):
            raise ValueError("archive manifest deletion epoch must be a number")
        raw_model_config = value.get("model_config", ())
        raw_recipe_dirs = value.get("recipe_dirs", ())
        if not isinstance(raw_model_config, list) or not all(isinstance(item, str) for item in raw_model_config):
            raise ValueError("model_config paths must be a list of strings")
        if not isinstance(raw_recipe_dirs, list):
            raise ValueError("recipe_dirs must be a list")
        raw_storage = value.get("storage")
        raw_registration = value.get("registration")
        status = value.get("status", "archived")
        if status not in ("archived", "consumed"):
            raise ValueError(f"unknown archive status {status!r}")
        raw_restoration = value.get("restoration")
        if status == "consumed" and raw_restoration is None:
            raise ValueError("a consumed archive must record its restoration")
        return cls(
            archive_id=archive_id,
            scenario=scenario,
            deleted_at=deleted_at,
            deleted_at_epoch=float(deleted_at_epoch),
            source_process=ArchiveSourceProcess.from_dict(value.get("source_process")),
            release=ArchivedRelease.from_dict(value.get("release")),
            storage=None if raw_storage is None else StorageArchive.from_dict(raw_storage),
            registration=None if raw_registration is None else RegistrationArchive.from_dict(raw_registration),
            model_config=tuple(raw_model_config),
            recipe_dirs=tuple(ArchivedRecipeDir.from_dict(item) for item in raw_recipe_dirs),
            status=status,
            restoration=None if raw_restoration is None else ArchiveRestoration.from_dict(raw_restoration),
        )

    def summary(self) -> dict[str, Any]:
        """The fields a list endpoint exposes: identity, timing, origin, release, locations, state."""
        return {
            "archive_id": self.archive_id,
            "scenario": self.scenario,
            "deleted_at": self.deleted_at,
            "source_process": self.source_process.to_dict(),
            "release": self.release.to_dict(),
            "locations": list(self.locations),
            "status": self.status,
            "restoration": None if self.restoration is None else self.restoration.to_dict(),
        }


class ArchiveCatalog:
    """Owns archive manifest files and the locks that fence their lifecycle."""

    def __init__(self, directory: Path | None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._manifests_dir = None if self._directory is None else self._directory / "archived" / "manifests"
        self._locks_dir = None if self._directory is None else self._directory / ".locks"
        # Re-entrant: list/read/consume run inside the catalog-wide lock and take it themselves.
        self._memory_lock = RLock()
        self._memory: dict[str, ScenarioArchive] = {}
        if self._manifests_dir is not None:
            self._manifests_dir.mkdir(parents=True, exist_ok=True)
        if self._locks_dir is not None:
            self._locks_dir.mkdir(parents=True, exist_ok=True)

    @property
    def durable(self) -> bool:
        return self._directory is not None

    @staticmethod
    def new_archive_id() -> str:
        return uuid.uuid4().hex

    @contextmanager
    def catalog_lock(self):
        """Serialize manifest state changes across processes (restore/consume)."""
        if self._locks_dir is None:
            with self._memory_lock:
                yield
            return
        with self._file_lock(self._locks_dir / "archive-catalog.lock"):
            yield

    @contextmanager
    def scenario_locks(self, *names: str):
        """Serialize delete and restore of the same scenario names across processes."""
        if self._locks_dir is None:
            yield
            return
        locks_dir = self._locks_dir
        paths = tuple(
            locks_dir / f"scenario-{hashlib.sha256(name.encode('utf-8')).hexdigest()}.lock"
            for name in sorted(set(names))
        )
        with ExitStack() as stack:
            import fcntl

            handles = [stack.enter_context(open(path, "a+b")) for path in paths]
            try:
                for handle in handles:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                for handle in reversed(handles):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def write(self, archive: ScenarioArchive) -> None:
        if self._manifests_dir is None:
            with self._memory_lock:
                self._memory[archive.archive_id] = archive
            return
        self._write_file(archive)

    def read(self, archive_id: str) -> ScenarioArchive:
        if not _ARCHIVE_ID_PATTERN.fullmatch(archive_id):
            raise UnknownArchive(f"unknown archive {archive_id!r}")
        if self._manifests_dir is None:
            with self._memory_lock:
                archive = self._memory.get(archive_id)
            if archive is None:
                raise UnknownArchive(f"unknown archive {archive_id!r}")
            return archive
        path = self._manifest_path(archive_id)
        if not path.is_file():
            raise UnknownArchive(f"unknown archive {archive_id!r}")
        try:
            return ScenarioArchive.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ArchiveCorrupt(f"archive manifest {archive_id} is unreadable or invalid: {exc}") from exc

    def list(self) -> tuple[ScenarioArchive, ...]:
        if self._manifests_dir is None:
            with self._memory_lock:
                return tuple(sorted(self._memory.values(), key=lambda item: item.deleted_at_epoch))
        archives: list[ScenarioArchive] = []
        for path in sorted(self._manifests_dir.glob("*.json")):
            try:
                archives.append(ScenarioArchive.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, json.JSONDecodeError):
                # One corrupt manifest must not hide the other archives; restore surfaces the error.
                continue
        return tuple(sorted(archives, key=lambda item: item.deleted_at_epoch))

    def consume(
        self,
        archive_id: str,
        *,
        restored_as: str,
        source_process: ArchiveSourceProcess,
    ) -> ScenarioArchive:
        archive = self.read(archive_id)
        if archive.consumed:
            prior_restored_as = None if archive.restoration is None else archive.restoration.restored_as
            raise ArchiveConflict(f"archive {archive_id} was already restored as {prior_restored_as!r}")
        epoch = time.time()
        archive.status = "consumed"
        archive.restoration = ArchiveRestoration(
            restored_as=restored_as,
            restored_at=utc_stamp(epoch),
            restored_at_epoch=epoch,
            source_process=source_process,
        )
        if self._manifests_dir is None:
            with self._memory_lock:
                self._memory[archive_id] = archive
            return archive
        self._write_file(archive)
        return archive

    def _manifest_path(self, archive_id: str) -> Path:
        if self._manifests_dir is None:
            raise ArchiveCorrupt("archive catalog has no durable manifest directory")
        return self._manifests_dir / f"{archive_id}.json"

    def _write_file(self, archive: ScenarioArchive) -> None:
        path = self._manifest_path(archive.archive_id)
        payload = json.dumps(archive.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".manifest-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


__all__ = [
    "ARCHIVE_FORMAT",
    "ArchiveCatalog",
    "ArchiveRestoration",
    "ArchiveSourceProcess",
    "ArchivedRecipeDir",
    "ArchivedRelease",
    "ScenarioArchive",
    "utc_stamp",
]
