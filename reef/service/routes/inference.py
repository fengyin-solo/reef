from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from aiohttp import web

from reef.runtime.interfaces import MULTIMODAL_ROUTES, InferenceHandler
from reef.service.idempotency import (
    CLAIM_CACHED,
    CLAIM_WAITING,
    CachedBufferedInference,
    CachedStreamedInference,
    IdempotencyConflict,
    IdempotencyInProgress,
    InferenceCacheClaim,
    InferenceFlight,
    InferenceLeaderExecutionFailed,
)
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.streaming import (
    SSEFrameDecoder,
    chat_completion_chunk_identity,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)
from reef.service.wire import RELEASE_ID_HEADER, InferenceIdempotency

logger = logging.getLogger(__name__)


async def _release_header(request_service: RequestService, headers: Mapping[str, str]) -> dict[str, str]:
    """``x-reef-release-id`` on a file serving scenario's inference responses, so a resident harness learns of a new head on its next call."""
    head = await asyncio.to_thread(request_service.harness_head, headers)
    return {} if head is None else {RELEASE_ID_HEADER: head}


class _SSERelay:
    """Relay upstream SSE frames to the client, holding back the terminal event.

    The terminal event is withheld so Reef can append its receipt frames before
    it, and the chat-completion identity those frames need is picked up from the
    first frame that carries one. Every byte passed to the client is also
    appended to ``sent`` so a completed stream can be replayed from cache.
    """

    def __init__(self, path: str, response: web.StreamResponse, sent: bytearray) -> None:
        self._path = path
        self._response = response
        self._sent = sent
        self._decoder = SSEFrameDecoder()
        self.chat_identity: dict[str, Any] = {}
        self.terminal: bytes | None = None

    async def feed(self, chunk: bytes) -> bool:
        """Write the frames completed by ``chunk``; return whether the terminal arrived."""
        return await self._write(self._decoder.feed(chunk))

    async def finish(self) -> bool:
        """Flush what the decoder still holds; return whether the terminal arrived.

        A trailing partial frame is passed through as-is: the stream ended
        without a terminal event, so the client gets whatever upstream sent.
        """
        frames, remainder = self._decoder.finalize()
        if await self._write(frames):
            return True
        if remainder:
            await self._response.write(remainder)
            self._sent.extend(remainder)
        return False

    async def _write(self, frames: Iterable[bytes]) -> bool:
        for frame in frames:
            if is_terminal_sse_event(self._path, frame):
                self.terminal = frame
                return True
            if not self.chat_identity:
                self.chat_identity = chat_completion_chunk_identity(frame)
            await self._response.write(frame)
            self._sent.extend(frame)
        return False


async def _buffered_replay(
    request_service: RequestService,
    headers: Mapping[str, str],
    entry: CachedBufferedInference,
) -> web.Response:
    response_headers = {"x-reef-agent-record-id": entry.agent_record_id}
    response_headers.update(await _release_header(request_service, headers))
    return web.json_response(dict(entry.response_payload), headers=response_headers)


def _streamed_replay(entry: CachedStreamedInference) -> web.Response:
    return web.Response(body=entry.body, status=entry.status, headers=dict(entry.headers))


async def _await_idempotent_flight(
    request_service: RequestService,
    claim: InferenceCacheClaim,
) -> CachedBufferedInference | CachedStreamedInference:
    try:
        return await asyncio.wait_for(
            request_service.await_inference_flight(claim),
            timeout=request_service.inference_cache.wait_timeout_seconds,
        )
    except TimeoutError as exc:
        raise IdempotencyInProgress(
            "a matching inference request is still being processed; retry later"
        ) from exc


async def _idempotent_buffered(
    request: web.Request,
    payload: dict[str, Any],
    *,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
    idempotency_key: InferenceIdempotency,
) -> web.Response:
    while True:
        claim = request_service.claim_inference(idempotency_key, request.path, payload)
        if claim.kind == CLAIM_CACHED and claim.entry is not None:
            if not isinstance(claim.entry, CachedBufferedInference):
                raise IdempotencyConflict("idempotency key was already used for a streaming request")
            await request_service.assert_cached_boundary(claim.entry, request.headers)
            return await _buffered_replay(request_service, request.headers, claim.entry)
        if claim.kind == CLAIM_WAITING:
            try:
                entry = await _await_idempotent_flight(request_service, claim)
            except InferenceLeaderExecutionFailed:
                continue
            if not request_service.same_inference_request(entry, request.path, payload):
                raise IdempotencyConflict(
                    "idempotency key was already used with a different request payload or path"
                )
            if not isinstance(entry, CachedBufferedInference):
                raise IdempotencyConflict("idempotency key was already used for a streaming request")
            await request_service.assert_cached_boundary(entry, request.headers)
            return await _buffered_replay(request_service, request.headers, entry)

        flight = claim.flight
        if flight is None:
            raise RuntimeError("idempotency leader claim carries no flight")
        try:
            response_payload, item = await request_service.infer_with_data(
                request.headers,
                payload,
                request.path,
                inference_handler,
                agent_record_id=idempotency_key.agent_record_id,
            )
        except BaseException as exc:
            request_service.fail_inference(
                idempotency_key,
                flight,
                exc if isinstance(exc, Exception) else RuntimeError("inference request was cancelled"),
            )
            raise
        request_service.complete_buffered_inference(
            idempotency_key,
            flight,
            item,
            request.path,
            payload,
            response_payload,
        )
        headers = {"x-reef-agent-record-id": item.agent_record_id}
        headers.update(await _release_header(request_service, request.headers))
        return web.json_response(response_payload, headers=headers)


async def _relay_inference_stream(
    request: web.Request,
    payload: dict[str, Any],
    *,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
    idempotency_key: InferenceIdempotency | None,
) -> web.StreamResponse:
    """Stream one upstream inference to the client and record what it sent.

    With an idempotency key, a completed response is cached and replays never
    invoke the handler or append another record; only the leader executes.
    """
    if idempotency_key is not None:
        while True:
            claim = request_service.claim_inference(idempotency_key, request.path, payload)
            if claim.kind == CLAIM_CACHED and claim.entry is not None:
                if not isinstance(claim.entry, CachedStreamedInference):
                    raise IdempotencyConflict("idempotency key was already used for a non-streaming request")
                await request_service.assert_cached_boundary(claim.entry, request.headers)
                return _streamed_replay(claim.entry)
            if claim.kind == CLAIM_WAITING:
                try:
                    entry = await _await_idempotent_flight(request_service, claim)
                except InferenceLeaderExecutionFailed:
                    continue
                if not request_service.same_inference_request(entry, request.path, payload):
                    raise IdempotencyConflict(
                        "idempotency key was already used with a different request payload or path"
                    )
                if not isinstance(entry, CachedStreamedInference):
                    raise IdempotencyConflict("idempotency key was already used for a non-streaming request")
                await request_service.assert_cached_boundary(entry, request.headers)
                return _streamed_replay(entry)
            return await _stream_leader(
                request,
                payload,
                request_service=request_service,
                inference_handler=inference_handler,
                idempotency_key=idempotency_key,
                flight=claim.flight,
            )
    return await _stream_leader(
        request,
        payload,
        request_service=request_service,
        inference_handler=inference_handler,
        idempotency_key=None,
        flight=None,
    )


async def _stream_leader(
    request: web.Request,
    payload: dict[str, Any],
    *,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
    idempotency_key: InferenceIdempotency | None,
    flight: InferenceFlight | None,
) -> web.StreamResponse:
    """Execute one (possibly first-and-only) streaming inference and relay it."""
    cache_completed = False

    def fail_waiters(reason: Exception) -> None:
        nonlocal cache_completed
        if idempotency_key is not None and flight is not None and not cache_completed:
            request_service.fail_inference(idempotency_key, flight, reason)

    try:
        upstream, pending = await request_service.start_stream(
            request.headers,
            payload,
            request.path,
            inference_handler,
            agent_record_id=None if idempotency_key is None else idempotency_key.agent_record_id,
        )
    except BaseException as exc:
        fail_waiters(exc if isinstance(exc, Exception) else RuntimeError("inference request was cancelled"))
        raise
    response_headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in ("x-reef-agent-record-id", RELEASE_ID_HEADER)
    }
    response_headers.update(await _release_header(request_service, request.headers))
    content_type = next(
        (value for name, value in response_headers.items() if name.lower() == "content-type"),
        "",
    )
    is_sse = "text/event-stream" in content_type.lower()
    if not is_sse:
        response_headers["x-reef-agent-record-id"] = pending.item.agent_record_id
    response = web.StreamResponse(status=upstream.status, headers=response_headers)
    body = bytearray()
    client_body = bytearray()
    complete = False
    error = None
    record_attempted = False
    try:
        await response.prepare(request)
        if is_sse:
            relay = _SSERelay(request.path, response, client_body)
            async for chunk in upstream.chunks:
                body.extend(chunk)
                if await relay.feed(chunk):
                    break
            if relay.terminal is None:
                await relay.finish()
            if relay.terminal is None:
                error = "upstream SSE ended without a terminal event"
            else:
                record_attempted = True
                item = request_service.record_stream(
                    pending,
                    stream_record(upstream, bytes(body), complete=True),
                )
                for frame in receipt_sse_events(
                    request.path,
                    relay.chat_identity,
                    relay.terminal,
                    item.agent_record_id,
                ):
                    await response.write(frame)
                    client_body.extend(frame)
                if idempotency_key is not None and flight is not None:
                    request_service.complete_streamed_inference(
                        idempotency_key,
                        flight,
                        item,
                        request.path,
                        payload,
                        status=upstream.status,
                        headers=response_headers,
                        body=bytes(client_body),
                    )
                    cache_completed = True
                complete = True
        else:
            async for chunk in upstream.chunks:
                body.extend(chunk)
                await response.write(chunk)
                client_body.extend(chunk)
            complete = True
    except ConnectionResetError:
        error = "client disconnected"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        fail_waiters(exc)
        raise
    finally:
        if error is not None:
            logger.warning(
                "stream for %s (record %s) ended early: %s",
                request.path,
                pending.item.agent_record_id,
                error,
            )
        try:
            await upstream.close()
        finally:
            stored = None
            if not record_attempted:
                try:
                    stored = request_service.record_stream(
                        pending,
                        stream_record(upstream, bytes(body), complete=complete, error=error),
                    )
                except BaseException as exc:
                    if idempotency_key is not None and flight is not None:
                        request_service.fail_inference(
                            idempotency_key,
                            flight,
                            exc if isinstance(exc, Exception) else RuntimeError("inference stream was cancelled"),
                        )
                    raise
            if (
                idempotency_key is not None
                and flight is not None
                and not cache_completed
                and stored is not None
                and complete
                and error is None
            ):
                request_service.complete_streamed_inference(
                    idempotency_key,
                    flight,
                    stored,
                    request.path,
                    payload,
                    status=upstream.status,
                    headers=response_headers,
                    body=bytes(client_body),
                )
                cache_completed = True
            if not cache_completed and idempotency_key is not None and flight is not None:
                request_service.fail_inference(
                    idempotency_key,
                    flight,
                    RuntimeError(error or "inference stream did not complete"),
                )
    return response


def register_inference_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
) -> None:
    async def inference(request: web.Request) -> web.StreamResponse:
        payload = await read_object(request)
        idempotency_key = request_service.inference_idempotency(request.headers)
        if payload.get("stream") is True:
            return await _relay_inference_stream(
                request,
                payload,
                request_service=request_service,
                inference_handler=inference_handler,
                idempotency_key=idempotency_key,
            )
        if idempotency_key is None:
            response_payload, item = await request_service.infer_with_data(
                request.headers,
                payload,
                request.path,
                inference_handler,
            )
            headers = {"x-reef-agent-record-id": item.agent_record_id}
            headers.update(await _release_header(request_service, request.headers))
            return web.json_response(response_payload, headers=headers)
        return await _idempotent_buffered(
            request,
            payload,
            request_service=request_service,
            inference_handler=inference_handler,
            idempotency_key=idempotency_key,
        )

    async def multimodal(request: web.Request) -> web.StreamResponse:
        """Relay a multimodal call through the scenario's recipe; the answer streams back unchanged, unrecorded."""
        payload = await read_object(request)
        if payload.get("stream") is True:
            raise web.HTTPBadRequest(text=f"{request.path} does not stream; send the request without stream")
        upstream = await request_service.relay_multimodal(request.headers, payload, request.path)
        response = web.StreamResponse(status=upstream.status, headers=upstream.headers)
        try:
            await response.prepare(request)
            async for chunk in upstream.chunks:
                await response.write(chunk)
        finally:
            await upstream.close()
        return response

    app.router.add_post("/v1/chat/completions", inference)
    app.router.add_post("/v1/responses", inference)
    app.router.add_post("/v1/messages", inference)
    app.router.add_post("/v1/messages/count_tokens", inference)
    for path in MULTIMODAL_ROUTES:
        app.router.add_post(path, multimodal)


__all__ = ["register_inference_routes"]
