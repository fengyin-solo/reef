from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

from aiohttp import web

from reef.runtime.interfaces import MULTIMODAL_ROUTES, InferenceHandler
from reef.service.request_service import (
    STREAM_END_CLIENT_DISCONNECTED,
    STREAM_END_IDLE_TIMEOUT,
    STREAM_END_TOTAL_TIMEOUT,
    STREAM_END_UPSTREAM_ERROR,
    RequestService,
    StreamEndReason,
)
from reef.service.routes.payload import read_object
from reef.service.streaming import (
    SSEFrameDecoder,
    chat_completion_chunk_identity,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)
from reef.service.wire import RELEASE_ID_HEADER

logger = logging.getLogger(__name__)

#: How often a relayed stream checks whether its client went away while
#: blocked on the next upstream chunk.
_DISCONNECT_POLL_INTERVAL_S = 0.1


async def _release_header(request_service: RequestService, headers: Mapping[str, str]) -> dict[str, str]:
    """``x-reef-release-id`` on a file serving scenario's inference responses, so a resident harness learns of a new head on its next call."""
    head = await asyncio.to_thread(request_service.harness_head, headers)
    return {} if head is None else {RELEASE_ID_HEADER: head}


class _SSERelay:
    """Relay upstream SSE frames to the client, holding back the terminal event.

    The terminal event is withheld so Reef can append its receipt frames before
    it, and the chat-completion identity those frames need is picked up from the
    first frame that carries one.
    """

    def __init__(self, path: str, response: web.StreamResponse) -> None:
        self._path = path
        self._response = response
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
        return False

    async def _write(self, frames: Iterable[bytes]) -> bool:
        for frame in frames:
            if is_terminal_sse_event(self._path, frame):
                self.terminal = frame
                return True
            if not self.chat_identity:
                self.chat_identity = chat_completion_chunk_identity(frame)
            await self._response.write(frame)
        return False


class _StreamIdleTimeout(Exception):
    """No upstream chunk arrived within the configured idle bound."""

    def __init__(self, idle_timeout_s: float) -> None:
        super().__init__(f"stream idle timeout: no upstream chunk for {idle_timeout_s:g}s")
        self.idle_timeout_s = idle_timeout_s


class _StreamTotalTimeout(Exception):
    """The stream exceeded its configured overall duration bound."""

    def __init__(self, total_timeout_s: float) -> None:
        super().__init__(f"stream timeout exceeded after {total_timeout_s:g}s")
        self.total_timeout_s = total_timeout_s


class _ClientDisconnected(Exception):
    """The downstream client closed the connection while the stream was open."""


async def _client_disconnect_watcher(request: web.Request) -> None:
    """Return once the request's client connection is gone.

    The server does not cancel a handler when a client disappears, so a relay
    blocked on an upstream chunk would never notice without this check. Aiohttp
    drops the protocol's transport reference on ``connection_lost``, so a
    ``None`` transport while the handler is running means the client left.
    """
    while True:
        transport = request.transport
        if transport is None or transport.is_closing():
            return
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)


class _BoundedChunks:
    """Iterate upstream chunks under an overall deadline and an idle deadline.

    Each upstream read runs as its own task so either deadline, or a client
    disconnect, can cancel it at the current transcription point. Cancelling
    an async generator read unwinds its ``finally`` blocks, which is what makes
    the provider stop generating instead of running the request to completion.
    """

    def __init__(
        self,
        chunks: AsyncIterator[bytes],
        request: web.Request,
        *,
        total_timeout_s: float,
        idle_timeout_s: float,
    ) -> None:
        self._iterator = chunks.__aiter__()
        self._request = request
        self._total_timeout_s = total_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._deadline_monotonic = 0.0
        self._watcher: asyncio.Task[None] | None = None
        self._reading: asyncio.Task[bytes] | None = None

    def __aiter__(self) -> _BoundedChunks:
        return self

    async def __anext__(self) -> bytes:
        loop = asyncio.get_running_loop()
        if self._watcher is None:
            self._deadline_monotonic = loop.time() + self._total_timeout_s
            self._watcher = asyncio.ensure_future(_client_disconnect_watcher(self._request))
        watcher = self._watcher
        remaining_s = self._deadline_monotonic - loop.time()
        if remaining_s <= 0:
            raise _StreamTotalTimeout(self._total_timeout_s)
        self._reading = asyncio.ensure_future(self._iterator.__anext__())
        reading = self._reading
        done, _ = await asyncio.wait(
            {reading, watcher},
            timeout=min(self._idle_timeout_s, remaining_s),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if watcher in done:
            await self._cancel_reading()
            raise _ClientDisconnected
        if reading in done:
            # Natural EOF (StopAsyncIteration) and upstream failures propagate to
            # the relay, which records them as an upstream error.
            return reading.result()
        await self._cancel_reading()
        if self._deadline_monotonic - loop.time() <= 0:
            raise _StreamTotalTimeout(self._total_timeout_s)
        raise _StreamIdleTimeout(self._idle_timeout_s)

    async def aclose(self) -> None:
        """Stop watching and cancel any read still blocked on the upstream."""
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None
        await self._cancel_reading()

    async def _cancel_reading(self) -> None:
        reading = self._reading
        self._reading = None
        if reading is None:
            return
        if not reading.done():
            reading.cancel()
        # Discard whatever the read produced, including an exception: the
        # timeout or disconnect that interrupted it is the operative outcome.
        # Awaiting a cancelled generator read also lets its finally close the
        # provider connection.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reading


async def _relay_inference_stream(
    request: web.Request,
    payload: dict[str, Any],
    *,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
) -> web.StreamResponse:
    """Stream one upstream inference to the client and record what it sent."""
    limits = request_service.stream_time_limits
    upstream, pending = await request_service.start_stream(
        request.headers,
        payload,
        request.path,
        inference_handler,
    )
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
    complete = False
    error = None
    end_reason: StreamEndReason | None = None
    record_attempted = False
    chunks = _BoundedChunks(
        upstream.chunks,
        request,
        total_timeout_s=limits.total_timeout_s,
        idle_timeout_s=limits.idle_timeout_s,
    )
    try:
        await response.prepare(request)
        if is_sse:
            relay = _SSERelay(request.path, response)
            async for chunk in chunks:
                body.extend(chunk)
                if await relay.feed(chunk):
                    break
            if relay.terminal is None:
                await relay.finish()
            if relay.terminal is None:
                error = "upstream SSE ended without a terminal event"
                end_reason = STREAM_END_UPSTREAM_ERROR
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
                complete = True
        else:
            async for chunk in chunks:
                body.extend(chunk)
                await response.write(chunk)
            complete = True
    except _StreamIdleTimeout as exc:
        error = str(exc)
        end_reason = STREAM_END_IDLE_TIMEOUT
    except _StreamTotalTimeout as exc:
        error = str(exc)
        end_reason = STREAM_END_TOTAL_TIMEOUT
    except _ClientDisconnected:
        error = "client disconnected"
        end_reason = STREAM_END_CLIENT_DISCONNECTED
    except ConnectionResetError:
        error = "client disconnected"
        end_reason = STREAM_END_CLIENT_DISCONNECTED
    except asyncio.CancelledError:
        # Handler cancellation (the test server enables it; a deployment may)
        # reaches the relay mid-await. A gone or closing client transport
        # means the client disconnected, so clean up and record that. A live
        # transport makes this a shutdown cancellation, which must propagate.
        transport = request.transport
        if transport is not None and not transport.is_closing():
            raise
        error = "client disconnected"
        end_reason = STREAM_END_CLIENT_DISCONNECTED
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        end_reason = STREAM_END_UPSTREAM_ERROR
        raise
    finally:
        if error is not None:
            logger.warning(
                "stream for %s (record %s) ended early: %s",
                request.path,
                pending.item.agent_record_id,
                error,
            )
        await chunks.aclose()
        try:
            await upstream.close()
        finally:
            if not record_attempted:
                request_service.record_stream(
                    pending,
                    stream_record(upstream, bytes(body), complete=complete, error=error),
                    end_reason=end_reason,
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
        if payload.get("stream") is True:
            return await _relay_inference_stream(
                request,
                payload,
                request_service=request_service,
                inference_handler=inference_handler,
            )
        response_payload, item = await request_service.infer_with_data(
            request.headers,
            payload,
            request.path,
            inference_handler,
        )
        headers = {"x-reef-agent-record-id": item.agent_record_id}
        headers.update(await _release_header(request_service, request.headers))
        return web.json_response(response_payload, headers=headers)

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
