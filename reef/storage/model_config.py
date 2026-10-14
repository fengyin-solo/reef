"""Read, replace, and archive private per-scenario model JSON files.

Registry operations serialize access and validate model values before writing.
These functions preserve the existing filenames and owner-only permissions;
they hold no cache or runtime state and do not select a record backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from reef.core.errors import ArchiveConflict, ArchiveCorrupt


def _model_path(directory: Path | None, scenario: str) -> Path | None:
    if directory is None:
        return None
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    return directory / f"{key}-model.json"


def read_model_config(directory: Path | None, scenario: str) -> object:
    """Read the stored value, or null if no override has been saved."""
    path = _model_path(directory, scenario)
    return None if path is None or not path.exists() else json.loads(path.read_text())


def write_model_config(directory: Path | None, scenario: str, value: object) -> None:
    """Atomically replace a validated value; directory=None selects memory-only use."""
    path = _model_path(directory, scenario)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".model-")
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def archive_model_config(directory: Path | None, scenario: str) -> tuple[str, ...]:
    """Move a removed scenario's settings out of the active namespace."""
    path = _model_path(directory, scenario)
    if path is None or not path.exists():
        return ()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    destination = path.parent / "archived" / f"{key}-{stamp}-{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / path.name
    shutil.move(str(path), str(target))
    return (str(target),)


def restore_model_config(directory: Path | None, archived_path: str, *, target: str) -> str:
    """Copy an archived model settings file back under the restored scenario name.

    Settings are name-independent, so restoring under a different name only
    changes the file's hash prefix. The archived file is never removed here;
    an occupied target is refused.
    """
    if directory is None:
        raise ArchiveCorrupt("model settings were archived without a record directory")
    source_file = Path(archived_path)
    if not source_file.is_file():
        raise ArchiveCorrupt(f"archived model settings file is missing: {archived_path}")
    key = hashlib.sha256(target.encode("utf-8")).hexdigest()
    destination = Path(directory) / f"{key}-model.json"
    if destination.exists():
        raise ArchiveConflict(f"model settings already exist for scenario {target!r}; refusing to overwrite them")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination)
    return str(destination)
