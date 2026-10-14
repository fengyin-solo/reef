"""Scenario archive manifests and restore: summaries, conflicts, corruption, races, restart, cleanup."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.test_scenario_store import commit, record

from reef.artifact import InMemoryRepositoryBackend
from reef.core.errors import ArchiveConflict, ArchiveCorrupt, UnknownArchive
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.service.app import create_app
from reef.storage.commit_log import CommitLog
from reef.storage.sqlite import SQLiteScenarioStorage


def _durable_dispatcher(tmp_path: Path) -> tuple[Dispatcher, Path]:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    record_dir = tmp_path / "records"
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=record_dir,
        scenario_storage=SQLiteScenarioStorage(record_dir),
    )
    return dispatcher, record_dir


def _inference_record(record_id: str, scenario: str) -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"text": record_id},
    )


def _manifest_files(record_dir: Path) -> list[Path]:
    return sorted((record_dir / "archived" / "manifests").glob("*.json"))


def _delete_and_get_archive_id(dispatcher: Dispatcher, scenario: str = "delivery") -> str:
    result = dispatcher.delete_scenario(scenario)
    return result["archive_id"]


def test_delete_writes_an_archive_manifest_with_the_audit_summary(tmp_path: Path) -> None:
    dispatcher, record_dir = _durable_dispatcher(tmp_path)
    created = dispatcher.get_or_create_scenario("delivery")
    assert created is not None
    release = created.repository.require_current_artifact()
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    dispatcher.accept_record(_inference_record("record-2", "delivery"))

    result = dispatcher.delete_scenario("delivery")

    manifests = _manifest_files(record_dir)
    assert len(manifests) == 1
    document = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert document["archive_id"] == result["archive_id"]
    assert document["scenario"] == "delivery"
    assert document["deleted_at"]
    assert document["release"] == {"release_id": release.release_id, "content_id": release.content_id}
    assert document["source_process"]["pid"] > 0
    assert document["source_process"]["host"]
    assert document["source_process"]["process_id"]
    assert document["status"] == "archived"
    assert document["restoration"] is None
    assert document["locations"] == result["archived"]
    assert any(path.endswith(".sqlite3") for path in document["locations"])

    summaries = dispatcher.list_archived_scenarios()
    assert len(summaries) == 1
    summary = summaries[0]
    assert set(summary) == {
        "archive_id",
        "scenario",
        "deleted_at",
        "source_process",
        "release",
        "locations",
        "status",
        "restoration",
    }
    assert summary["scenario"] == "delivery"
    assert summary["release"]["release_id"] == release.release_id
    dispatcher.close()


def test_restore_brings_the_scenario_and_its_records_back(tmp_path: Path) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    dispatcher.accept_record(_inference_record("record-2", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)
    assert not dispatcher.has_scenario("delivery")

    result = dispatcher.restore_scenario(archive_id)

    assert result["restored_as"] == "delivery"
    assert dispatcher.has_scenario("delivery")
    assert {row["scenario"] for row in dispatcher.list_scenarios()} == {"delivery"}
    restored = dispatcher.get_or_create_scenario("delivery", allow_implicit_creation=False)
    assert restored is not None
    records = dispatcher.read_records("delivery", after_sequence=0, limit=50)["records"]
    assert [row["agent_record_id"] for row in records] == ["record-1", "record-2"]


def test_restored_records_remain_auditable_but_never_reenter_the_training_queue(tmp_path: Path) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    dispatcher.accept_record(_inference_record("record-2", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)

    dispatcher.restore_scenario(archive_id)
    restored = dispatcher.get_or_create_scenario("delivery", allow_implicit_creation=False)
    assert restored is not None

    # Training replay (replay_page/count) sees no unread records; audit pages keep every row.
    assert restored.store.records.count("delivery") == 0
    audit = restored.store.records.audit_page("delivery", after_sequence=0, limit=50)
    assert [stored.item.agent_record_id for stored in audit] == ["record-1", "record-2"]
    assert all(stored.compacted_at is not None for stored in audit)
    assert restored.trainer.operational_metrics()["records/unread_count"] == 0
    assert restored.prepare_training_step() is None
    # A retry of an old accepted record still deduplicates instead of training twice.
    duplicate = dispatcher.accept_record(_inference_record("record-1", "delivery"))
    assert duplicate.agent_record_id == "record-1"
    dispatcher.close()


def test_restore_under_a_new_name_rekeys_the_records(tmp_path: Path) -> None:
    dispatcher, record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)

    result = dispatcher.restore_scenario(archive_id, new_name="refund")

    assert result["restored_as"] == "refund"
    assert not dispatcher.has_scenario("delivery")
    assert dispatcher.has_scenario("refund")
    records = dispatcher.read_records("refund", after_sequence=0, limit=50)["records"]
    assert [row["agent_record_id"] for row in records] == ["record-1"]
    # The old name has no rows anywhere: the stored scenario identity was rewritten.
    import hashlib

    old_key = hashlib.sha256(b"delivery").hexdigest()
    assert not list(record_dir.glob(f"{old_key}*"))
    dispatcher.close()


def test_restore_refuses_an_occupied_target_name(tmp_path: Path) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)
    # The name was recreated after deletion: a new generation exists and must win.
    recreated = dispatcher.get_or_create_scenario("delivery")
    assert recreated is not None
    recreated.repository.require_current_artifact()

    with pytest.raises(ArchiveConflict, match="already exists"):
        dispatcher.restore_scenario(archive_id)
    with pytest.raises(ArchiveConflict, match="already exists"):
        dispatcher.restore_scenario(archive_id, new_name="delivery")

    # The new generation is untouched and the archive remains unconsumed for a rename attempt.
    assert dispatcher.has_scenario("delivery")
    summary = dispatcher.list_archived_scenarios()[0]
    assert summary["status"] == "archived"
    # The user can still recover under another name.
    renamed = dispatcher.restore_scenario(archive_id, new_name="delivery-old")
    assert renamed["restored_as"] == "delivery-old"
    dispatcher.close()


def test_restore_validates_the_new_name(tmp_path: Path) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    archive_id = _delete_and_get_archive_id(dispatcher)
    for bad_name in ("", "   ", "a/b", ".", ".."):
        with pytest.raises(ValueError):
            dispatcher.restore_scenario(archive_id, new_name=bad_name)


def test_restoring_the_same_archive_twice_conflicts(tmp_path: Path) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    archive_id = _delete_and_get_archive_id(dispatcher)
    dispatcher.restore_scenario(archive_id)

    with pytest.raises(ArchiveConflict, match="already restored"):
        dispatcher.restore_scenario(archive_id)
    summary = dispatcher.list_archived_scenarios()[0]
    assert summary["status"] == "consumed"
    assert summary["restoration"]["restored_as"] == "delivery"
    dispatcher.close()


def test_unknown_archive_id_is_a_404_domain_error(tmp_path: Path) -> None:
    dispatcher, _ = _durable_dispatcher(tmp_path)
    with pytest.raises(UnknownArchive):
        dispatcher.restore_scenario("a" * 32)
    with pytest.raises(UnknownArchive):
        dispatcher.restore_scenario("not-a-uuid")
    dispatcher.close()


def test_corrupt_manifest_surfaces_a_corrupt_error_and_is_hidden_from_the_list(tmp_path: Path) -> None:
    dispatcher, record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    archive_id = _delete_and_get_archive_id(dispatcher)
    manifest = _manifest_files(record_dir)[0]
    manifest.write_text("{not json", encoding="utf-8")

    assert dispatcher.list_archived_scenarios() == ()
    with pytest.raises(ArchiveCorrupt, match="unreadable or invalid"):
        dispatcher.restore_scenario(archive_id)
    # The archived data was not touched by the failed restore.
    archived_state = [path for path in (record_dir / "archived").iterdir() if path.name != "manifests"]
    assert len(archived_state) == 1 and any(p.name.endswith(".sqlite3") for p in archived_state[0].iterdir())
    dispatcher.close()


def test_concurrent_restores_of_one_archive_have_exactly_one_winner(tmp_path: Path) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)
    barrier = Barrier(2)
    outcomes: list[object] = []
    outcomes_lock = threading.Lock()

    def restore() -> None:
        barrier.wait()
        try:
            outcome: object = dispatcher.restore_scenario(archive_id)
        except Exception as exc:  # recording the cross-thread outcome
            outcome = exc
        with outcomes_lock:
            outcomes.append(outcome)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(restore), executor.submit(restore)]
        for future in futures:
            future.result()

    assert len(outcomes) == 2
    winners = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    losers = [outcome for outcome in outcomes if isinstance(outcome, ArchiveConflict)]
    assert len(winners) == 1 and len(losers) == 1
    assert winners[0]["restored_as"] == "delivery"
    assert dispatcher.has_scenario("delivery")
    assert dispatcher.list_archived_scenarios()[0]["status"] == "consumed"
    dispatcher.close()


def test_concurrent_restores_across_separate_processes_locks_have_one_winner(tmp_path: Path) -> None:
    """Two dispatchers (each with its own locks but sharing the durable directory) serialize restore."""
    first, record_dir = _durable_dispatcher(tmp_path)
    first.get_or_create_scenario("delivery")
    first.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(first)
    first.close()

    second, _ = _durable_dispatcher(tmp_path)
    third, _ = _durable_dispatcher(tmp_path)
    barrier = Barrier(2)
    outcomes: list[object] = []
    outcomes_lock = threading.Lock()

    def restore(dispatcher: Dispatcher) -> None:
        barrier.wait()
        try:
            outcome: object = dispatcher.restore_scenario(archive_id)
        except Exception as exc:  # recording the cross-thread outcome
            outcome = exc
        with outcomes_lock:
            outcomes.append(outcome)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(restore, second), executor.submit(restore, third)]
        for future in futures:
            future.result()

    winners = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    assert len(winners) == 1
    assert isinstance(outcomes[0], ArchiveConflict) or isinstance(outcomes[1], ArchiveConflict)
    # The loser lost on the storage generation or on the consumed manifest, and never overwrote anything.
    assert (second.has_scenario("delivery"), third.has_scenario("delivery")).count(True) == 1
    second.close()
    third.close()


def test_archive_listing_and_restore_work_after_a_restart(tmp_path: Path) -> None:
    first, record_dir = _durable_dispatcher(tmp_path)
    first.get_or_create_scenario("delivery")
    first.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(first)
    first.close()

    second, _ = _durable_dispatcher(tmp_path)
    assert record_dir.exists()
    summaries = second.list_archived_scenarios()
    assert [row["archive_id"] for row in summaries] == [archive_id]
    assert summaries[0]["scenario"] == "delivery"
    assert not second.has_scenario("delivery")

    result = second.restore_scenario(archive_id)
    assert result["restored_as"] == "delivery"
    records = second.read_records("delivery", after_sequence=0, limit=50)["records"]
    assert [row["agent_record_id"] for row in records] == ["record-1"]
    second.close()


def test_consumed_archive_cleans_its_archived_data_but_keeps_the_manifest(tmp_path: Path) -> None:
    dispatcher, record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)
    archived_state_dirs = [path for path in (record_dir / "archived").iterdir() if path.name not in ("manifests",)]
    assert len(archived_state_dirs) == 1

    dispatcher.restore_scenario(archive_id)

    # The data copies are gone; the consumed manifest stays as the audit record.
    assert not archived_state_dirs[0].exists()
    manifest = _manifest_files(record_dir)[0]
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["status"] == "consumed"
    assert document["restoration"]["restored_as"] == "delivery"
    dispatcher.close()


def test_failed_restore_leaves_the_archive_restorable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher, _record_dir = _durable_dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("delivery")
    dispatcher.accept_record(_inference_record("record-1", "delivery"))
    archive_id = _delete_and_get_archive_id(dispatcher)

    def fail_resolve(*args, **kwargs):
        raise RuntimeError("injected resolution failure")

    monkeypatch.setattr(dispatcher._registry, "get_or_create", fail_resolve)
    with pytest.raises(RuntimeError, match="injected resolution failure"):
        dispatcher.restore_scenario(archive_id)

    # No half-restored active state, and the same archive restores successfully on retry.
    assert not dispatcher.has_scenario("delivery")
    monkeypatch.undo()
    result = dispatcher.restore_scenario(archive_id)
    assert result["restored_as"] == "delivery"
    records = dispatcher.read_records("delivery", after_sequence=0, limit=50)["records"]
    assert [row["agent_record_id"] for row in records] == ["record-1"]
    dispatcher.close()


def test_storage_restore_rejects_a_newer_generation_without_touching_the_archive(tmp_path: Path) -> None:
    storage = SQLiteScenarioStorage(tmp_path)
    first = storage.open("math")
    first.records.append(record("record-1"))
    first.commit_step(expected_step=0, commit=commit())
    first.close()
    archived = storage.archive("math")
    assert archived is not None
    archived_copy_files = tuple(Path(path) for path in archived.locations)

    # Same name reopened: a new active generation occupies the hash.
    new_generation = storage.open("math")
    new_generation.records.append(record("record-later"))
    new_generation.close()

    with pytest.raises(ArchiveConflict, match="active record store"):
        storage.restore(archived, source="math", target="math", settlement_receipt="reef-archive-restore:test")

    # The archive directory is intact and the new generation keeps its row.
    assert all(path.exists() for path in archived_copy_files)
    check = storage.open("math")
    assert check.records.get_for_audit("math", "record-later") is not None
    assert check.records.get_for_audit("math", "record-1") is None
    check.close()
    storage.close()


def test_storage_restore_rekeys_and_retires_for_a_renamed_target(tmp_path: Path) -> None:
    storage = SQLiteScenarioStorage(tmp_path)
    first = storage.open("alpha")
    first.records.append(record("record-1", scenario="alpha"))
    first.commit_step(expected_step=0, commit=commit(1, scenario="alpha"))
    first.close()
    archived = storage.archive("alpha")
    assert archived is not None
    archived_database = next(Path(path) for path in archived.locations if path.endswith(".sqlite3"))

    storage.restore(archived, source="alpha", target="beta", settlement_receipt="reef-archive-restore:rename")

    restored = storage.open("beta")
    history = restored.history()
    assert [entry.step for entry in history] == [1]
    assert all(entry.scenario == "beta" for entry in history)
    assert restored.records.get_for_audit("beta", "record-1") is not None
    restored.close()
    # The archived database is the untouched copy: it still belongs to alpha.
    import sqlite3

    with sqlite3.connect(archived_database) as connection:
        rows = connection.execute("SELECT DISTINCT scenario FROM agent_record").fetchall()
    assert rows == [("alpha",)]
    storage.close()


def test_storage_restore_rewrites_the_commit_log_scenario(tmp_path: Path) -> None:
    storage = SQLiteScenarioStorage(tmp_path)
    session = storage.open("alpha")
    session.records.append(record("record-1", scenario="alpha"))
    session.commit_step(expected_step=0, commit=commit(1, scenario="alpha"))
    session.close()
    archived = storage.archive("alpha")
    assert archived is not None

    storage.restore(archived, source="alpha", target="beta", settlement_receipt="reef-archive-restore:rename-log")
    import hashlib

    target_key = hashlib.sha256(b"beta").hexdigest()
    records = CommitLog(tmp_path / f"{target_key}.commits.jsonl").records()
    assert [entry.scenario for entry in records] == ["beta"]
    storage.close()


@pytest.mark.unit
def test_archive_list_and_restore_over_http() -> None:
    async def run() -> None:
        from reef.dispatcher import build_default_dispatcher

        client = TestClient(TestServer(create_app(build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()))))
        await client.start_server()
        try:
            await client.post("/reef/scenarios", json={"name": "delivery"})
            deleted = await client.delete("/reef/scenarios/delivery")
            archive_id = (await deleted.json())["archive_id"]

            listing = await client.get("/reef/scenarios/archived")
            assert listing.status == 200
            body = await listing.json()
            assert [row["archive_id"] for row in body["archives"]] == [archive_id]
            assert "storage" not in body["archives"][0]

            missing = await client.post(f"/reef/scenarios/archives/{'f' * 32}/restore", json={})
            assert missing.status == 404

            restored = await client.post(f"/reef/scenarios/archives/{archive_id}/restore", data=None)
            assert restored.status == 200
            restored_body = await restored.json()
            assert restored_body["restored_as"] == "delivery"

            second = await client.post(f"/reef/scenarios/archives/{archive_id}/restore", json={})
            assert second.status == 409

            bad_name = await client.post(f"/reef/scenarios/archives/{archive_id}/restore", json={"name": "a/b"})
            # Malformed requests are rejected at the boundary before any archive state check.
            assert bad_name.status == 400
            assert await (await client.get("/reef/scenarios")).json() == {
                "scenarios": [
                    {
                        "scenario": "delivery",
                        "loaded": True,
                        "release_id": restored_body["release_id"],
                        "content_id": restored_body["content_id"],
                    }
                ]
            }
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_restore_http_validates_name_before_touching_archives(tmp_path: Path) -> None:
    async def run() -> None:
        dispatcher, _ = _durable_dispatcher(tmp_path)
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            await client.post("/reef/scenarios", json={"name": "delivery"})
            archive_id = (await (await client.delete("/reef/scenarios/delivery")).json())["archive_id"]
            bad = await client.post(f"/reef/scenarios/archives/{archive_id}/restore", json={"name": "a/b"})
            assert bad.status == 400
            # Still restorable after the rejected request.
            ok = await client.post(f"/reef/scenarios/archives/{archive_id}/restore", json={})
            assert ok.status == 200
        finally:
            await client.close()

    asyncio.run(run())
