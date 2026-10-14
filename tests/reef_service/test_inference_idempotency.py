from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.dispatcher import build_default_dispatcher
from reef.inference.http import HttpInferenceHandler
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import IdempotencySettings, create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import TrainStepResult

SCENARIO = "idempotent-chat"
KEY_HEADER = {"Idempotency-Key": "key-1"}


class CountingInferenceHandler(InferenceHandler):
    def __init__(self, response=None, gate: asyncio.Event | None = None) -> None:
        self.calls = 0
        self.paths: list[str] = []
        self._response = response
        self._gate = gate
        self.fail_once = False

    async def inference(self, artifact, path, payload):
        del artifact, payload
        self.calls += 1
        self.paths.append(path)
        if self._gate is not None:
            await self._gate.wait()
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("provider failed")
        return self._response if self._response is not None else {"choices": [{"index": 0}]}


def app_with(handler, *, settings: IdempotencySettings | None = None, tmp_path=None):
    dispatcher = build_default_dispatcher(
        local_artifact_dir=tmp_path / "local" if tmp_path is not None else None,
        scenario_storage=SQLiteScenarioStorage(),
    )
    app = create_app(dispatcher, inference_handler=handler, inference_idempotency=settings)
    return app, dispatcher


def records(dispatcher, scenario: str = SCENARIO):
    return dispatcher.get_or_create_scenario(scenario).records.replay(scenario)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/chat/completions", {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/responses", {"model": "reef", "input": "hi"}),
        ("/v1/messages", {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}),
    ],
)
def test_replay_returns_cached_response_without_calling_provider(path, payload) -> None:
    async def run() -> None:
        handler = CountingInferenceHandler(response={"answer": "same"})
        app, dispatcher = app_with(handler)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
            first = await client.post(path, headers=headers, json=payload)
            second = await client.post(path, headers=headers, json=payload)
            assert first.status == second.status == 200
            assert await first.json() == await second.json() == {"answer": "same"}
            receipt = first.headers["x-reef-agent-record-id"]
            assert receipt and receipt == second.headers["x-reef-agent-record-id"]
            assert handler.calls == 1
            assert handler.paths == [path]
            # Cache replays append no record: the exchange must not be trained on twice.
            stored = records(dispatcher)
            assert len(stored) == 1
            assert stored[0].agent_record_id == receipt
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_pinned_receipt_header_is_honored_and_replayed() -> None:
    async def run() -> None:
        handler = CountingInferenceHandler()
        app, dispatcher = app_with(handler)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        headers = {"x-reef-scenario": SCENARIO, "x-reef-agent-record-id": "client-receipt-123"}
        try:
            first = await client.post("/v1/chat/completions", headers=headers, json=payload)
            second = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert first.headers["x-reef-agent-record-id"] == "client-receipt-123"
            assert second.headers["x-reef-agent-record-id"] == "client-receipt-123"
            assert handler.calls == 1
            assert records(dispatcher)[0].agent_record_id == "client-receipt-123"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_conflicting_idempotency_headers_are_rejected() -> None:
    async def run() -> None:
        app, _ = app_with(CountingInferenceHandler())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "x-reef-scenario": SCENARIO,
                    "Idempotency-Key": "key-1",
                    "x-reef-agent-record-id": "other-receipt",
                },
                json={"messages": []},
            )
            assert response.status == 400
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_same_key_with_different_payload_conflicts() -> None:
    async def run() -> None:
        handler = CountingInferenceHandler()
        app, _ = app_with(handler)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
            first = await client.post(
                "/v1/chat/completions", headers=headers, json={"messages": [{"content": "one"}]}
            )
            conflict = await client.post(
                "/v1/chat/completions", headers=headers, json={"messages": [{"content": "two"}]}
            )
            assert first.status == 200
            assert conflict.status == 409
            assert handler.calls == 1
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_same_key_across_release_boundary_conflicts(tmp_path) -> None:
    async def run() -> None:
        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "model.txt").write_text("base")
        repository_backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
        dispatcher = build_default_dispatcher(
            backend_factory=repository_backend, scenario_storage=SQLiteScenarioStorage()
        )
        handler = CountingInferenceHandler()
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=handler)))
        await client.start_server()
        try:
            headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
            first = await client.post("/v1/chat/completions", headers=headers, json={"messages": []})
            assert first.status == 200
            old_release = records(dispatcher)[0].artifact_ref.release_id

            candidate = tmp_path / "candidate"
            candidate.mkdir()
            (candidate / "model.txt").write_text("trained")
            dispatcher._commit_result(SCENARIO, TrainStepResult(state=None, artifact=Artifact.local(candidate)))
            assert dispatcher.get_or_create_scenario(SCENARIO).current_artifact_ref().release_id != old_release

            replay = await client.post("/v1/chat/completions", headers=headers, json={"messages": []})
            assert replay.status == 409
            assert handler.calls == 1
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_concurrent_duplicates_execute_once_and_share_receipt() -> None:
    async def run() -> None:
        gate = asyncio.Event()
        handler = CountingInferenceHandler(gate=gate)
        app, dispatcher = app_with(handler)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": [{"content": "hi"}]}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            first_request = asyncio.ensure_future(
                client.post("/v1/chat/completions", headers=headers, json=payload)
            )
            await asyncio.sleep(0.05)
            second_request = asyncio.ensure_future(
                client.post("/v1/chat/completions", headers=headers, json=payload)
            )
            await asyncio.sleep(0.05)
            assert handler.calls == 1
            gate.set()
            first, second = await asyncio.gather(first_request, second_request)
            assert first.status == second.status == 200
            assert await first.json() == await second.json()
            assert first.headers["x-reef-agent-record-id"] == second.headers["x-reef-agent-record-id"]
            assert handler.calls == 1
            assert len(records(dispatcher)) == 1
        finally:
            gate.set()
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_duplicate_while_leader_runs_gets_identifiable_conflict_after_timeout() -> None:
    async def run() -> None:
        settings = IdempotencySettings(wait_timeout_seconds=0.05)
        gate = asyncio.Event()
        handler = CountingInferenceHandler(gate=gate)
        app, _ = app_with(handler, settings=settings)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": []}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            leader = asyncio.ensure_future(
                client.post("/v1/chat/completions", headers=headers, json=payload)
            )
            await asyncio.sleep(0.05)
            waiter = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert waiter.status == 409
            gate.set()
            assert (await leader).status == 200
            assert handler.calls == 1
        finally:
            gate.set()
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_failed_leader_is_not_cached_and_retry_executes_again() -> None:
    async def run() -> None:
        handler = CountingInferenceHandler()
        handler.fail_once = True
        app, dispatcher = app_with(handler)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": []}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            failed = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert failed.status == 500
            retried = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert retried.status == 200
            assert handler.calls == 2
            assert len(records(dispatcher)) == 1
            assert records(dispatcher)[0].agent_record_id == retried.headers["x-reef-agent-record-id"]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_expired_entry_is_executed_again() -> None:
    async def run() -> None:
        settings = IdempotencySettings(ttl_seconds=0.05)
        handler = CountingInferenceHandler()
        app, _ = app_with(handler, settings=settings)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": []}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            assert (await client.post("/v1/chat/completions", headers=headers, json=payload)).status == 200
            await asyncio.sleep(0.08)
            assert (await client.post("/v1/chat/completions", headers=headers, json=payload)).status == 200
            assert handler.calls == 2
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_capacity_evicts_oldest_entries() -> None:
    async def run() -> None:
        settings = IdempotencySettings(capacity=1)
        handler = CountingInferenceHandler()
        app, _ = app_with(handler, settings=settings)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": []}
        try:
            assert (
                await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": SCENARIO, "Idempotency-Key": "key-1"},
                    json=payload,
                )
            ).status == 200
            assert (
                await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": SCENARIO, "Idempotency-Key": "key-2"},
                    json=payload,
                )
            ).status == 200
            # key-1 was evicted by key-2, so its replay executes again.
            assert (
                await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": SCENARIO, "Idempotency-Key": "key-1"},
                    json=payload,
                )
            ).status == 200
            assert handler.calls == 3
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_deleting_scenario_clears_its_cache() -> None:
    async def run() -> None:
        handler = CountingInferenceHandler()
        app, _ = app_with(handler)
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": []}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            assert (await client.post("/v1/chat/completions", headers=headers, json=payload)).status == 200
            assert (await client.delete(f"/reef/scenarios/{SCENARIO}")).status == 200
            # The recreated scenario of the same name starts without cached responses.
            assert (await client.post("/v1/chat/completions", headers=headers, json=payload)).status == 200
            assert handler.calls == 2
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_disabled_idempotency_keeps_legacy_behavior() -> None:
    async def run() -> None:
        handler = CountingInferenceHandler()
        app, dispatcher = app_with(handler, settings=IdempotencySettings(enabled=False))
        client = TestClient(TestServer(app))
        await client.start_server()
        payload = {"messages": []}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            first = await client.post("/v1/chat/completions", headers=headers, json=payload)
            second = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert first.status == second.status == 200
            assert first.headers["x-reef-agent-record-id"] != second.headers["x-reef-agent-record-id"]
            assert handler.calls == 2
            assert len(records(dispatcher)) == 2
        finally:
            await client.close()

    asyncio.run(run())


def _sse_upstream(chunks_for_request, counter):
    async def upstream(request):
        counter["hits"] += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in chunks_for_request:
            await response.write(chunk)
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream)
    return upstream_app


@pytest.mark.unit
def test_streaming_replay_returns_same_frames_without_calling_provider(tmp_path) -> None:
    async def run() -> None:
        content = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        done = b"data: [DONE]\n\n"
        counter = {"hits": 0}
        upstream_server = TestServer(_sse_upstream([content, done], counter))
        await upstream_server.start_server()
        dispatcher = build_default_dispatcher(
            local_artifact_dir=tmp_path / "local", scenario_storage=SQLiteScenarioStorage()
        )
        backend = HttpInferenceHandler(str(upstream_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=backend)))
        await client.start_server()
        payload = {"messages": [{"content": "hi"}], "stream": True}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            first = await client.post("/v1/chat/completions", headers=headers, json=payload)
            first_body = await first.read()
            second = await client.post("/v1/chat/completions", headers=headers, json=payload)
            second_body = await second.read()
            assert first.status == second.status == 200
            assert first_body == second_body
            assert counter["hits"] == 1
            [stored] = records(dispatcher)
            assert stored.payload["response"]["complete"] is True
            # Both streams carry the same receipt in their terminal metadata frame.
            receipts = [
                next(
                    frame["reef"]["agent_record_id"]
                    for line in body.decode().splitlines()
                    if line.startswith("data:") and '"reef"' in line
                    for frame in [json.loads(line.removeprefix("data: "))]
                )
                for body in (first_body, second_body)
            ]
            assert receipts[0] == receipts[1] == stored.agent_record_id
        finally:
            await client.close()
            await upstream_server.close()

    asyncio.run(run())


@pytest.mark.unit
def test_streaming_without_terminal_is_not_cached_and_retry_runs_provider(tmp_path) -> None:
    async def run() -> None:
        partial = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        counter = {"hits": 0}
        upstream_server = TestServer(_sse_upstream([partial], counter))
        await upstream_server.start_server()
        dispatcher = build_default_dispatcher(
            local_artifact_dir=tmp_path / "local", scenario_storage=SQLiteScenarioStorage()
        )
        backend = HttpInferenceHandler(str(upstream_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=backend)))
        await client.start_server()
        payload = {"messages": [{"content": "hi"}], "stream": True}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            first = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert first.status == 200
            assert await first.read() == partial
            retried = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert await retried.read() == partial
            assert counter["hits"] == 2
            stored = records(dispatcher)
            assert [record.payload["response"]["complete"] for record in stored] == [False, False]
        finally:
            await client.close()
            await upstream_server.close()

    asyncio.run(run())


@pytest.mark.unit
def test_concurrent_streaming_duplicates_share_one_upstream_call(tmp_path) -> None:
    async def run() -> None:
        content = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        done = b"data: [DONE]\n\n"
        release = asyncio.Event()
        counter = {"hits": 0}

        async def upstream(request):
            counter["hits"] += 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(content)
            await release.wait()
            await response.write(done)
            return response

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", upstream)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        dispatcher = build_default_dispatcher(
            local_artifact_dir=tmp_path / "local", scenario_storage=SQLiteScenarioStorage()
        )
        backend = HttpInferenceHandler(str(upstream_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=backend)))
        await client.start_server()
        payload = {"messages": [{"content": "hi"}], "stream": True}
        headers = {"x-reef-scenario": SCENARIO, **KEY_HEADER}
        try:
            first = asyncio.ensure_future(client.post("/v1/chat/completions", headers=headers, json=payload))
            await asyncio.sleep(0.1)
            second = asyncio.ensure_future(client.post("/v1/chat/completions", headers=headers, json=payload))
            await asyncio.sleep(0.1)
            assert counter["hits"] == 1
            release.set()
            first_response, second_response = await asyncio.gather(first, second)
            assert await first_response.read() == await second_response.read()
            assert counter["hits"] == 1
        finally:
            release.set()
            await client.close()
            await upstream_server.close()

    asyncio.run(run())
