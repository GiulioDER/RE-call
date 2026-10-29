"""Harness failures must not leak users or turn failed teardown into success.

Red proof, baseline 2d2f28c5: every parametrized node below failed its intended
assertion on the Linux test host before fixing run_scenario and run. Exception cases observed
missing deletion calls; failed-delete cases observed an unreported HTTP503 or
the original transport error instead of an explicit cleanup failure. The same
unchanged nodes passed against the fixes. Both HTTP boundaries are offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import aml_locomo_route_compare as comparison
from scripts import aml_same_user_concurrency as concurrency


@pytest.mark.parametrize("phase", ["initial_add", "replay"])
def test_scenario_settles_adds_and_cleans_users_after_failure(phase: str) -> None:
    """BUG-001: pre-fix run_scenario skips both deletes after the injected ReadError."""
    async def exercise() -> None:
        added: set[str] = set()
        deleted: list[str] = []
        calls = 0
        pending = 0
        pending_at_delete: list[int] = []

        async def transport(request: httpx.Request) -> httpx.Response:
            nonlocal calls, pending
            body = json.loads(request.content)
            user = body["user_id"]
            if request.url.path == "/v1/delete":
                pending_at_delete.append(pending)
                deleted.append(user)
                added.discard(user)
                return httpx.Response(200, json={})
            calls += 1
            call = calls
            if phase == "initial_add" and call == 1:
                raise httpx.ReadError("initial add failed", request=request)
            if phase == "replay" and call == 3:
                raise httpx.ReadError("receipt replay failed", request=request)
            pending += 1
            try:
                await asyncio.sleep(0.01)
                added.add(user)
                return httpx.Response(200, json={"raw_count": 1})
            finally:
                pending -= 1

        async with httpx.AsyncClient(
            base_url="https://harness.test", transport=httpx.MockTransport(transport)
        ) as client:
            with pytest.raises(httpx.ReadError):
                await concurrency.run_scenario(client, "test", 2, [[], []], "run", 1, 0)
            assert sorted(deleted) == ["c9-same-user-run-test-u0", "c9-same-user-run-test-u1"]
            assert pending_at_delete == [0, 0]
            assert added == set()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["status", "transport"])
def test_scenario_reports_cleanup_failure_and_attempts_every_user(failure: str) -> None:
    """BUG-001: pre-fix ignores HTTP503 or stops at the first delete transport error."""
    async def exercise() -> None:
        deleted: list[str] = []

        def transport(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/add":
                return httpx.Response(200, json={"raw_count": 1})
            deleted.append(json.loads(request.content)["user_id"])
            if len(deleted) == 1:
                if failure == "transport":
                    raise httpx.ReadError("delete failed", request=request)
                return httpx.Response(503)
            return httpx.Response(200, json={})

        async with httpx.AsyncClient(
            base_url="https://harness.test", transport=httpx.MockTransport(transport)
        ) as client:
            with pytest.raises(Exception) as caught:
                await concurrency.run_scenario(client, "test", 2, [[], []], "run", 1, 0)
            assert len(deleted) == 2
            assert isinstance(caught.value, RuntimeError)
            assert "cleanup failed" in str(caught.value)

    asyncio.run(exercise())


def _comparison_client(monkeypatch, tmp_path, *, interrupt: bool, failure: str | None):
    import recall_aml.__main__ as entrypoint
    import starlette.testclient

    users = ["comparison-test-u0", "comparison-test-u1"]
    adds = [{"user_id": user} for user in users]
    question = {
        "question_id": "q1", "category": 1, "user_id": users[0],
        "query": "a turn", "gold_turns": ["a turn"], "gold_sessions": ["s1"],
    }
    monkeypatch.setattr(comparison, "build_corpus", lambda *args: (adds, [question]))
    monkeypatch.setattr(entrypoint, "build_app", lambda: object())
    monkeypatch.setenv("RECALL_AML_API_KEY", "test-only")
    # run replaces this module function; restore the original when this fixture ends.
    import recall_aml.service as service
    monkeypatch.setattr(service, "route_query", service.route_query)

    class Client:
        open = False

        def __init__(self, app):
            self.deleted: list[str] = []

        def __enter__(self):
            self.open = True
            return self

        def __exit__(self, *args):
            self.open = False

        def get(self, *args, **kwargs):
            return SimpleNamespace(json=lambda: {"variant": "test"})

        def post(self, path, *, json, headers):
            assert self.open, "cleanup must happen before TestClient closes"
            if path == "/v1/search" and interrupt:
                raise KeyboardInterrupt("measurement interrupted")
            if path == "/v1/delete":
                self.deleted.append(json["user_id"])
                if len(self.deleted) == 1 and failure:
                    if failure == "transport":
                        raise httpx.ReadError("delete failed")
                    return SimpleNamespace(status_code=503)
            return SimpleNamespace(status_code=200, json=lambda: {"data": []})

    client = Client(object())
    monkeypatch.setattr(starlette.testclient, "TestClient", lambda app: client)
    path = tmp_path / "data.json"
    path.write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        data=path, run_id="test", limit_conversations=None,
        limit_questions=None, expected_variant="test",
    )
    return client, args, users


def test_comparison_cleans_users_on_interruption(monkeypatch, tmp_path) -> None:
    """BUG-002: pre-fix run skips both deletes after the Search KeyboardInterrupt."""
    client, args, users = _comparison_client(
        monkeypatch, tmp_path, interrupt=True, failure=None
    )
    with pytest.raises(KeyboardInterrupt):
        comparison.run(args)
    assert client.deleted == users
    assert not client.open


@pytest.mark.parametrize("failure", ["status", "transport"])
def test_comparison_reports_cleanup_failure_and_attempts_every_user(
    monkeypatch, tmp_path, failure: str
) -> None:
    """BUG-002: pre-fix ignores HTTP503 or stops at the first delete transport error."""
    client, args, users = _comparison_client(
        monkeypatch, tmp_path, interrupt=False, failure=failure
    )
    with pytest.raises(Exception) as caught:
        comparison.run(args)
    assert client.deleted == users
    assert isinstance(caught.value, RuntimeError)
    assert "cleanup failed" in str(caught.value)
