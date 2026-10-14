"""Storage boundary for a scenario's records, commits, and recovery.

Artifact publication remains the responsibility of the committer. A
store settles record consumption and the committed scenario state together;
implementations may use a database transaction or a recoverable journal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.errors import ReefError
from reef.storage.commits import CommitRecord
from reef.storage.records import RecordStore


class ScenarioStoreConflict(ReefError):
    """A commit's expected step or content conflicts with the stored history."""


class ScenarioStore(ABC):
    """One scenario's storage session, including its record connection.

    The caller closes the session when the scenario is unloaded. Closing is
    idempotent; subsequent reads and writes fail. Records from other scenarios
    must never be committed or compacted through this session.
    """

    @property
    @abstractmethod
    def records(self) -> RecordStore:
        """The records connection owned by this session."""

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether committed state survives a process restart."""

    @abstractmethod
    def history(self) -> tuple[CommitRecord, ...]:
        """Read committed steps in increasing order, including rollbacks."""

    @abstractmethod
    def training_run_position(self) -> tuple[int, int]:
        """Return the last rollback step and subsequent training count."""

    @abstractmethod
    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        """Commit exactly ``expected_step + 1`` and settle its record compaction.

        Conflicting writers are serialized across sessions. An identical retry
        (ignoring only ``recorded_at``) returns the original record, even after
        later steps, and repairs interrupted compaction. Different content or
        a stale expected step raises ``ScenarioStoreConflict``.

        An exception can occur after durability but before compaction. Callers
        inspect history or retry the same commit before repeating training.
        """

    @abstractmethod
    def recover(self, *, checkpoint: CommitRecord | None) -> CommitRecord | None:
        """Reconcile a checkpoint commit with history and replay compaction.

        ``None`` denotes initial registration before any committed step.
        A checkpoint ahead of history is adopted durably before compaction.
        Validate scenario identity and continuity after the checkpoint. Return
        the committed head, or ``None`` at creation, without publishing artifacts.
        """

    @abstractmethod
    def close(self) -> None:
        """Release this session's records and other resources exactly once."""


@dataclass(frozen=True)
class StorageArchive:
    """One archived storage generation, serialized inside a scenario archive manifest.

    ``locations`` are the moved files or URLs an operator can audit. ``token``
    carries the coordinates :meth:`ScenarioStorage.restore` needs; its keys are
    implementation-private and its values are plain strings so the manifest is
    JSON without custom encoders.
    """

    locations: tuple[str, ...]
    token: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"locations": list(self.locations), "token": dict(self.token)}

    @classmethod
    def from_dict(cls, value: object) -> StorageArchive:
        if not isinstance(value, dict):
            raise ValueError("storage archive must be an object")
        raw_locations = value.get("locations")
        raw_token = value.get("token")
        if not isinstance(raw_locations, list) or not all(isinstance(item, str) for item in raw_locations):
            raise ValueError("storage archive locations must be a list of strings")
        if not isinstance(raw_token, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in raw_token.items()
        ):
            raise ValueError("storage archive token must be an object of strings")
        return cls(locations=tuple(raw_locations), token=dict(raw_token))


class ScenarioStorage(ABC):
    """Open sessions and maintain their storage without exposing file paths.

    The owner serializes archive and retention with lifecycle changes and
    closes a scenario's sessions before archiving it. Closing storage releases
    shared resources only; sessions remain owned by their scenarios.
    """

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether newly opened sessions retain commits across restarts."""

    @abstractmethod
    def open(self, scenario: str) -> ScenarioStore:
        """Open an independently owned session for this scenario."""

    @abstractmethod
    def archive(self, scenario: str) -> StorageArchive | None:
        """Archive a closed scenario's state; ``None`` when nothing is durable.

        The archived generation is never modified or deleted, so a failed
        restore leaves it usable again.
        """

    @abstractmethod
    def restore(
        self,
        archive: StorageArchive,
        *,
        source: str,
        target: str,
        settlement_receipt: str,
    ) -> tuple[str, ...]:
        """Copy an archived generation back into active storage.

        Reactivate it under ``target`` (``source`` restores the original name),
        rewriting scenario identity inside the data when the names differ.
        Records the archived commit log never consumed are retired from
        training (compacted, still readable for audit) under
        ``settlement_receipt`` so restoring a scenario never feeds its old
        records back into the training queue.

        Raise :class:`reef.core.errors.ArchiveConflict` when ``target`` already
        has another active generation; the caller's archive stays untouched.
        Return the active locations the generation now occupies.
        """

    @abstractmethod
    def rollback_restore(
        self,
        archive: StorageArchive,
        *,
        target: str,
        restored_locations: tuple[str, ...],
    ) -> None:
        """Remove the active generation a failed :meth:`restore` placed, leaving the archive intact."""

    @abstractmethod
    def prune(self, *, days: float, max_bytes: int) -> int:
        """Purge retained compacted bodies across active and archived storage."""

    @abstractmethod
    def close(self) -> None:
        """Release shared storage resources; repeated closes are harmless."""


__all__ = ["ScenarioStorage", "ScenarioStore", "ScenarioStoreConflict", "StorageArchive"]
