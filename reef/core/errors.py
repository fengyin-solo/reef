"""Common error hierarchy for Reef.

All Reef-raised exceptions derive from :class:`ReefError` so callers can
catch any domain error with one exception type.
"""


class ReefError(Exception):
    """Base class for all Reef-raised exceptions."""


class UnknownScenario(ReefError):
    """The scenario does not exist and the caller may not create it implicitly."""


class UnknownArchive(ReefError):
    """No scenario archive exists for the archive id the caller named."""


class ArchiveConflict(ReefError):
    """An archive cannot be restored: it was consumed, its target name is occupied, or its storage generation clashes."""


class ArchiveCorrupt(ReefError):
    """An archive manifest cannot be read or used for a restore."""


class DeployConfigError(ReefError):
    """A ``reef serve`` deployment config cannot be loaded or is invalid."""
