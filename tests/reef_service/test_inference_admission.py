"""Bounded inference admission: the concurrency cap, waiting queue, overload rejection,
slot lifetime across streaming, disconnect reclamation, and the training pause/resume gate."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.interfaces import (
    InferenceAdmissionController,
    InferenceHandler,
    InferenceOverloaded,
    InferenceRuntime,
    InferenceStream,
)
from reef.service.app import create_app
from reef.service.request_service import RequestService
from reef.service.streaming import stream_record
from reef.storage.sqlite import SQLiteScenarioStorage

SCENARIO_HEADERS = {"x-reef-scenario": "math"}
INFERENCE_PATH = "/v1/chat/completions"


# -- Controller ---------------------------------------------------------------


@pytest.mark.unit
def test_unlimited_default_admits_every_concurrent_request() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController()
        handles = [await controller.acquire() for _ in range(5)]
        assert controller.status == {
            "open": True,
            "active": 5,
            "waiting": 0,
            "rejected_total": 0,
            "max_active": None,
            "max_waiting": None,
        }
        for handle in handles:
            handle.release()
        assert controller.status["active"] == 0

    asyncio.run(run())


@pytest.mark.unit
def test_requests_at_the_cap_wait_and_then_run_in_arrival_order() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController(max_active=1, max_waiting=2)
        first = await controller.acquire()
        assert controller.status["active"] == 1

        second = asyncio.create_task(controller.acquire())
        third = asyncio.create_task(controller.acquire())
        await _wait_for(lambda: controller.status["waiting"] == 2)
        assert controller.status["active"] == 1

        with pytest.raises(InferenceOverloaded) as overloaded:
            await controller.acquire()
        assert overloaded.value.retry_after_s == 1.0
        assert controller.status["rejected_total"] == 1
        assert not second.done() and not third.done()

        first.release()
        second_handle = await asyncio.wait_for(second, 1)
        assert controller.status["active"] == 1 and controller.status["waiting"] == 1
        second_handle.release()
        third_handle = await asyncio.wait_for(third, 1)
        third_handle.release()
        assert controller.status == {
            "open": True,
            "active": 0,
            "waiting": 0,
            "rejected_total": 1,
            "max_active": 1,
            "max_waiting": 2,
        }

    asyncio.run(run())


@pytest.mark.unit
def test_zero_length_queue_rejects_immediately_with_the_retry_interval() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController(max_active=1, max_waiting=0, retry_after_s=2.5)
        active = await controller.acquire()
        with pytest.raises(InferenceOverloaded) as overloaded:
            await controller.acquire()
        assert overloaded.value.retry_after_s == 2.5
        assert "2.5" in str(overloaded.value)
        assert controller.status["waiting"] == 0
        assert controller.status["rejected_total"] == 1
        active.release()

    asyncio.run(run())


@pytest.mark.unit
def test_cancelling_a_waiting_request_frees_its_queue_place() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController(max_active=1, max_waiting=1)
        active = await controller.acquire()
        waiting = asyncio.create_task(controller.acquire())
        await _wait_for(lambda: controller.status["waiting"] == 1)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert controller.status["waiting"] == 0 and controller.status["active"] == 1

        replacement = asyncio.create_task(controller.acquire())
        await _wait_for(lambda: controller.status["waiting"] == 1)
        active.release()
        replacement_handle = await asyncio.wait_for(replacement, 1)
        replacement_handle.release()
        assert controller.status["active"] == 0 and controller.status["rejected_total"] == 0

    asyncio.run(run())


@pytest.mark.unit
def test_pause_parks_requests_and_resume_grants_them_fifo() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController(max_active=1, max_waiting=1)
        controller.close()

        first = asyncio.create_task(controller.acquire())
        await _wait_for(lambda: controller.status["waiting"] == 1)
        assert controller.status["active"] == 0
        with pytest.raises(InferenceOverloaded):
            await controller.acquire()
        assert not first.done()

        controller.open()
        first_handle = await asyncio.wait_for(first, 1)
        assert controller.status["active"] == 1 and controller.status["waiting"] == 0
        first_handle.release()

    asyncio.run(run())


@pytest.mark.unit
def test_admission_configuration_is_validated() -> None:
    with pytest.raises(ValueError):
        InferenceAdmissionController(max_active=0)
    with pytest.raises(ValueError):
        InferenceAdmissionController(max_waiting=-1)
    with pytest.raises(ValueError):
        InferenceAdmissionController(retry_after_s=0)


async def _wait_for(predicate, *, timeout: float = 1.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before the timeout")


# -- RequestService -----------------------------------------------------------


class _BlockingHandler(InferenceHandler):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, object]) -> dict[str, object]:
        del artifact, path, payload
        self.started.set()
        await self.release.wait()
        return {"choices": [{"message": {"content": "ok"}}]}


class _SlowStreamHandler(InferenceHandler):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("streaming requests use inference_stream")

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, object]) -> InferenceStream:
        del artifact, path, payload

        async def chunks():
            self.started.set()
            await self.release.wait()
            yield b"data: ok\n\n"

        return InferenceStream(
            status=200,
            headers={"Content-Type": "text/event-stream"},
            chunks=chunks(),
        )


class _BoundedRuntime(InferenceRuntime):
    """In-process runtime with the base bounded controller and a fixed handler."""

    def __init__(self, handler: InferenceHandler, **kwargs) -> None:
        super().__init__(base_url="http://engine", **kwargs)
        self._handler = handler

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler


def _bounded_service(
    tmp_path: Path,
    *,
    max_active: int | None,
    max_waiting: int | None,
    retry_after_s: float = 1.0,
    handler: InferenceHandler | None = None,
) -> tuple[Dispatcher, _BoundedRuntime, InferenceHandler, RequestService]:
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(initial),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "scenarios"),
    )
    handler = handler or _BlockingHandler()
    runtime = _BoundedRuntime(
        handler,
        max_active=max_active,
        max_waiting=max_waiting,
        retry_after_s=retry_after_s,
    )
    scenario = dispatcher.get_or_create_scenario("math")
    scenario.model_config.runtime = runtime
    return dispatcher, runtime, handler, RequestService(dispatcher)


@pytest.mark.unit
def test_buffered_requests_queue_at_capacity_and_a_full_queue_overloads(tmp_path: Path) -> None:
    dispatcher, runtime, handler, service = _bounded_service(tmp_path, max_active=1, max_waiting=1)

    async def run() -> None:
        first = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        await asyncio.wait_for(handler.started.wait(), 1)
        assert runtime.inference_admission_status["active"] == 1

        second = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        await _wait_for(lambda: runtime.inference_admission_status["waiting"] == 1)

        with pytest.raises(InferenceOverloaded):
            await service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler)
        assert runtime.inference_admission_status["rejected_total"] == 1

        handler.release.set()
        await asyncio.gather(asyncio.wait_for(first, 1), asyncio.wait_for(second, 1))
        assert runtime.inference_admission_status["active"] == 0
        assert runtime.inference_admission_status["waiting"] == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_disconnecting_a_waiting_then_executing_request_returns_its_slot(tmp_path: Path) -> None:
    dispatcher, runtime, handler, service = _bounded_service(tmp_path, max_active=1, max_waiting=1)

    async def run() -> None:
        active = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        await asyncio.wait_for(handler.started.wait(), 1)
        queued = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        await _wait_for(lambda: runtime.inference_admission_status["waiting"] == 1)

        # A client that gives up while queued leaves the queue, letting the next request wait.
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await _wait_for(lambda: runtime.inference_admission_status["waiting"] == 0)

        replacement = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        await _wait_for(lambda: runtime.inference_admission_status["waiting"] == 1)

        # The executing client disconnecting frees its slot; the waiter fills it and completes.
        handler.release.set()
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        await asyncio.wait_for(replacement, 1)
        assert runtime.inference_admission_status["active"] == 0
        assert runtime.inference_admission_status["waiting"] == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_long_stream_holds_its_slot_until_the_response_ends(tmp_path: Path) -> None:
    stream_handler = _SlowStreamHandler()
    dispatcher, runtime, _, service = _bounded_service(tmp_path, max_active=1, max_waiting=0, handler=stream_handler)

    async def run() -> None:
        stream, pending = await service.start_stream(SCENARIO_HEADERS, {}, INFERENCE_PATH, stream_handler)
        chunks = stream.chunks
        first_chunk = asyncio.create_task(chunks.__anext__())
        await asyncio.wait_for(stream_handler.started.wait(), 1)
        assert runtime.inference_admission_status["active"] == 1

        # The slot stays occupied for the whole open stream, so new work is turned away.
        with pytest.raises(InferenceOverloaded):
            await service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, stream_handler)
        assert runtime.inference_admission_status["rejected_total"] == 1

        stream_handler.release.set()
        body = await asyncio.wait_for(first_chunk, 1)
        with pytest.raises(StopAsyncIteration):
            await chunks.__anext__()
        service.record_stream(pending, stream_record(stream, body, complete=True))
        assert runtime.inference_admission_status["active"] == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_stream_disconnect_reclaims_the_slot_for_a_waiter(tmp_path: Path) -> None:
    stream_handler = _SlowStreamHandler()
    dispatcher, runtime, _, service = _bounded_service(tmp_path, max_active=1, max_waiting=1, handler=stream_handler)

    async def run() -> None:
        stream, pending = await service.start_stream(SCENARIO_HEADERS, {}, INFERENCE_PATH, stream_handler)
        chunks = stream.chunks
        reading = asyncio.create_task(chunks.__anext__())
        await asyncio.wait_for(stream_handler.started.wait(), 1)
        assert runtime.inference_admission_status["active"] == 1

        # The client goes away mid-stream; the route closes upstream and records the early end.
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        await stream.close()
        service.record_stream(
            pending,
            stream_record(stream, b"", complete=False, error="client disconnected"),
        )
        assert runtime.inference_admission_status["active"] == 0

        stream_handler.release.set()
        next_stream, next_pending = await service.start_stream(SCENARIO_HEADERS, {}, INFERENCE_PATH, stream_handler)
        assert runtime.inference_admission_status["active"] == 1
        next_chunks = next_stream.chunks
        body = await asyncio.wait_for(next_chunks.__anext__(), 1)
        with pytest.raises(StopAsyncIteration):
            await next_chunks.__anext__()
        service.record_stream(next_pending, stream_record(next_stream, body, complete=True))
        assert runtime.inference_admission_status["active"] == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_paused_admission_queues_and_resume_continues_in_order(tmp_path: Path) -> None:
    dispatcher, runtime, handler, service = _bounded_service(tmp_path, max_active=1, max_waiting=2)
    runtime.pause_admission()

    async def run() -> None:
        first = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        second = asyncio.create_task(service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler))
        await _wait_for(lambda: runtime.inference_admission_status["waiting"] == 2)
        assert runtime.inference_admission_status["active"] == 0
        assert not handler.started.is_set()

        with pytest.raises(InferenceOverloaded):
            await service.infer(SCENARIO_HEADERS, {}, INFERENCE_PATH, handler)

        runtime.resume_admission()
        await asyncio.wait_for(handler.started.wait(), 1)
        assert runtime.inference_admission_status["active"] == 1
        handler.release.set()
        await asyncio.gather(asyncio.wait_for(first, 1), asyncio.wait_for(second, 1))
        assert runtime.inference_admission_status["active"] == 0
        assert runtime.inference_admission_status["waiting"] == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


# -- HTTP ---------------------------------------------------------------------


@pytest.mark.unit
def test_http_overload_is_a_server_503_with_a_retry_after_header(tmp_path: Path) -> None:
    dispatcher, runtime, handler, _ = _bounded_service(tmp_path, max_active=1, max_waiting=1, retry_after_s=2.0)

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            active = asyncio.create_task(client.post(INFERENCE_PATH, headers=SCENARIO_HEADERS, json={}))
            await _wait_for(lambda: runtime.inference_admission_status["active"] == 1)

            queued = asyncio.create_task(client.post(INFERENCE_PATH, headers=SCENARIO_HEADERS, json={}))
            await _wait_for(lambda: runtime.inference_admission_status["waiting"] == 1)

            overloaded = await client.post(INFERENCE_PATH, headers=SCENARIO_HEADERS, json={})
            assert overloaded.status == 503
            assert overloaded.headers["Retry-After"] == "2"
            assert "overloaded" in await overloaded.text()

            handler.release.set()
            for response in await asyncio.gather(asyncio.wait_for(active, 1), asyncio.wait_for(queued, 1)):
                assert response.status == 200
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
