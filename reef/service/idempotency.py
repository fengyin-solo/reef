"""Completed-response idempotency cache for inference requests.

A client that sends ``Idempotency-Key`` (or ``x-reef-agent-record-id``, which
also pins the returned receipt) pays the provider at most once per key:
concurrent duplicates wait for the first request, safe retries replay the
stored response without invoking the handler or appending another record, and
a key reused with a different request body or a different artifact release is
rejected with a conflict. Only fully successful responses enter the cache;
stream responses are cached after their terminal event was written. The cache
is TTL- and capacity-bound, is cleared when its scenario is deleted, and holds
responses for replay only — it never creates ``AgentRecord`` values, so a
cache hit is not trained on a second time.

All methods run in the service's asyncio event loop; no internal locking is
needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.errors import ReefError

#: Claim outcomes for one idempotency key, checked without awaiting.
CLAIM_CACHED = "cached"
CLAIM_WAITING = "waiting"
CLAIM_LEADER = "leader"


class IdempotencyConflict(ReefError):
    """An idempotency key was reused with a different request or release boundary."""


class IdempotencyInProgress(ReefError):
    """A duplicate request is still waiting for the first request to finish."""


class InferenceLeaderExecutionFailed(ReefError):
    """Internal control signal: the in-flight leader failed; the waiter may try itself."""


@dataclass(frozen=True)
class IdempotencySettings:
    """Limits for the inference response cache; ``enabled=False`` restores legacy behavior."""

    enabled: bool = True
    ttl_seconds: float = 3600.0
    capacity: int = 10_000
    wait_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("inference idempotency ttl_seconds must be positive")
        if self.capacity <= 0:
            raise ValueError("inference idempotency capacity must be positive")
        if self.wait_timeout_seconds <= 0:
            raise ValueError("inference idempotency wait_timeout_seconds must be positive")


def canonical_request_fingerprint(path: str, payload: Mapping[str, Any]) -> str:
    """Hash one request path and body independent of key order or whitespace."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(path.encode("utf-8") + b"\n" + body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedBufferedInference:
    agent_record_id: str
    release_id: str
    request_fingerprint: str
    response_payload: Mapping[str, Any]
    expires_at: float


@dataclass(frozen=True)
class CachedStreamedInference:
    agent_record_id: str
    release_id: str
    request_fingerprint: str
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    expires_at: float


CachedInference = CachedBufferedInference | CachedStreamedInference


@dataclass(frozen=True)
class InferenceFlight:
    """The leader's registration; the token proves completion calls come from it."""

    token: int
    future: asyncio.Future[CachedInference]


@dataclass(frozen=True)
class InferenceCacheClaim:
    kind: str
    entry: CachedInference | None = None
    flight: InferenceFlight | None = None


class InferenceResponseCache:
    """TTL/LRU store of completed responses and registry of in-flight leaders."""

    def __init__(self, settings: IdempotencySettings) -> None:
        self._settings = settings
        self._entries: OrderedDict[tuple[str, str], CachedInference] = OrderedDict()
        self._flights: dict[tuple[str, str], InferenceFlight] = {}
        self._next_token = 0

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def wait_timeout_seconds(self) -> float:
        return self._settings.wait_timeout_seconds

    def claim(self, scenario: str, key: str, request_fingerprint: str) -> InferenceCacheClaim:
        """Return a completed entry, an existing in-flight leader, or leadership itself."""
        cache_key = (scenario, key)
        entry = self._entries.get(cache_key)
        if entry is not None:
            if entry.expires_at <= time.monotonic():
                self._entries.pop(cache_key, None)
            else:
                self._entries.move_to_end(cache_key)
                self._require_same_request(entry, request_fingerprint)
                return InferenceCacheClaim(CLAIM_CACHED, entry=entry)
        flight = self._flights.get(cache_key)
        if flight is not None:
            return InferenceCacheClaim(CLAIM_WAITING, flight=flight)
        self._next_token += 1
        flight = InferenceFlight(token=self._next_token, future=asyncio.get_running_loop().create_future())
        self._flights[cache_key] = flight
        return InferenceCacheClaim(CLAIM_LEADER, flight=flight)

    def complete_buffered(
        self,
        scenario: str,
        key: str,
        flight: InferenceFlight,
        *,
        agent_record_id: str,
        release_id: str,
        request_fingerprint: str,
        response_payload: Mapping[str, Any],
    ) -> CachedBufferedInference:
        entry = CachedBufferedInference(
            agent_record_id=agent_record_id,
            release_id=release_id,
            request_fingerprint=request_fingerprint,
            response_payload=dict(response_payload),
            expires_at=time.monotonic() + self._settings.ttl_seconds,
        )
        self._store(scenario, key, flight, entry)
        return entry

    def complete_streamed(
        self,
        scenario: str,
        key: str,
        flight: InferenceFlight,
        *,
        agent_record_id: str,
        release_id: str,
        request_fingerprint: str,
        status: int,
        headers: Mapping[str, str],
        body: bytes,
    ) -> CachedStreamedInference:
        entry = CachedStreamedInference(
            agent_record_id=agent_record_id,
            release_id=release_id,
            request_fingerprint=request_fingerprint,
            status=status,
            headers=tuple(headers.items()),
            body=body,
            expires_at=time.monotonic() + self._settings.ttl_seconds,
        )
        self._store(scenario, key, flight, entry)
        return entry

    def fail(self, scenario: str, key: str, flight: InferenceFlight, error: Exception) -> None:
        """Release waiters after a leader failure; nothing is cached so retries re-execute."""
        cache_key = (scenario, key)
        current = self._flights.get(cache_key)
        if current is None or current.token != flight.token:
            return
        self._flights.pop(cache_key, None)
        if not flight.future.done():
            flight.future.set_exception(error)
            # Mark the exception retrieved so an un-awaited failure logs no warning.
            flight.future.exception()

    def clear_scenario(self, scenario: str) -> None:
        """Drop a deleted scenario's entries and fail any requests still waiting on them."""
        for cache_key in [cache_key for cache_key in self._entries if cache_key[0] == scenario]:
            self._entries.pop(cache_key, None)
        flight_keys = [cache_key for cache_key in self._flights if cache_key[0] == scenario]
        for cache_key in flight_keys:
            flight = self._flights.pop(cache_key)
            if not flight.future.done():
                error = IdempotencyConflict(f"scenario {scenario!r} was deleted while the request was in progress")
                flight.future.set_exception(error)
                flight.future.exception()

    def purge_expired(self) -> None:
        now = time.monotonic()
        expired = [cache_key for cache_key, entry in self._entries.items() if entry.expires_at <= now]
        for cache_key in expired:
            self._entries.pop(cache_key, None)

    def _store(
        self,
        scenario: str,
        key: str,
        flight: InferenceFlight,
        entry: CachedInference,
    ) -> None:
        cache_key = (scenario, key)
        current = self._flights.get(cache_key)
        if current is None or current.token != flight.token:
            # The scenario was cleared while the leader ran; do not re-register it.
            return
        self._flights.pop(cache_key, None)
        self._entries[cache_key] = entry
        self._entries.move_to_end(cache_key)
        while len(self._entries) > self._settings.capacity:
            self._entries.popitem(last=False)
        flight.future.set_result(entry)

    @staticmethod
    def _require_same_request(entry: CachedInference, request_fingerprint: str) -> None:
        if entry.request_fingerprint != request_fingerprint:
            raise IdempotencyConflict("idempotency key was already used with a different request payload or path")


__all__ = [
    "CLAIM_CACHED",
    "CLAIM_LEADER",
    "CLAIM_WAITING",
    "CachedBufferedInference",
    "CachedInference",
    "CachedStreamedInference",
    "IdempotencyConflict",
    "IdempotencyInProgress",
    "IdempotencySettings",
    "InferenceCacheClaim",
    "InferenceFlight",
    "InferenceLeaderExecutionFailed",
    "InferenceResponseCache",
    "canonical_request_fingerprint",
]
