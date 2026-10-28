"""Regressions for BUG-001 and DAT-001 from the audit of PRs 800 through 810.

Red proof, 2026-10-28: all five nodes failed their intended assertions on
master a0a09650a9ff30ae31460af5f1b6a202466d1d5b in isolated remote execution. The three
preflight parameters failed at the deadline assertion, the parse queue at its
deadline assertion, and the ledger test at the suppression assertion.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx
import pytest

from recall_aml.forget import drop_hits, ledger_entries
from recall_aml.identity import tenant_for
from recall_aml.models import AddRequest
from tests.test_aml_add_response_budget import (
    BUDGET, GatedCompiler, ReceiptRepository, _add, _app, _client_for,
    _pending, _service, _until,
)
from tests.test_aml_forget import _hit, _messages, _service as _forget_service


class HeldBody(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __aiter__(self):
        self.entered.set()
        await self.release.wait()
        yield json.dumps({
            "request_id": "held", "user_id": "u", "session_id": "s",
            "messages": [{"role": "user", "content": "we chose postgres"}],
        }).encode()


@pytest.mark.parametrize("stage", ["body", "identity", "receipt"])
def test_add_preflight_respects_the_response_budget(stage: str, monkeypatch) -> None:
    """BUG-001: each preflight stall must return pending and release admission.

    Baseline app.create_app.run_within_budget has no preflight deadline.
    Each exact parameter node must fail 'preflight outlived the response budget',
    then pass unchanged after bounding preflight.
    """
    import recall_aml.app as app_module

    repository = ReceiptRepository()
    compiler = GatedCompiler()
    compiler.gate.set()
    service = _service(repository, compiler)
    entered = threading.Event()
    release = threading.Event()
    receipt_release = asyncio.Event()
    real_identity = app_module.add_identity
    real_stored = service.stored_add

    def held_identity(model: AddRequest):
        entered.set()
        release.wait(timeout=5)
        return real_identity(model)

    async def held_receipt(*args: Any):
        entered.set()
        await receipt_release.wait()
        return await real_stored(*args)

    if stage == "identity":
        monkeypatch.setattr(app_module, "add_identity", held_identity)
    elif stage == "receipt":
        monkeypatch.setattr(service, "stored_add", held_receipt)

    async def scenario() -> None:
        body = HeldBody()
        async with _client_for(_app(service, slots=1)) as client:
            if stage == "body":
                task = asyncio.create_task(client.post(
                    "/v1/add", headers={"x-api-key": "k"}, content=body,
                ))
                await body.entered.wait()
            else:
                task = asyncio.create_task(_add(client))
                await _until(entered.is_set)
            try:
                done, _ = await asyncio.wait({task}, timeout=BUDGET + 0.7)
                assert task in done, "preflight outlived the response budget"
                assert _pending(task.result())
            finally:
                release.set()
                receipt_release.set()
                body.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # Timeouts must give both gates back so a later valid Add can run.
            assert (await _add(client, "recovery")).status_code == 200

    asyncio.run(scenario())


def test_add_waiting_for_a_parse_slot_respects_its_budget() -> None:
    """BUG-001: a slow reader cannot hold another Add past its response budget.

    Baseline must fail 'parse queue outlived the response budget', rather than a
    test timeout. The first body is released and all request tasks are collected.
    """
    compiler = GatedCompiler()
    compiler.gate.set()
    service = _service(ReceiptRepository(), compiler)

    async def scenario() -> None:
        body = HeldBody()
        async with _client_for(_app(service, slots=1)) as client:
            first = asyncio.create_task(client.post(
                "/v1/add", headers={"x-api-key": "k"}, content=body,
            ))
            await body.entered.wait()
            second = asyncio.create_task(_add(client, "queued"))
            try:
                done, _ = await asyncio.wait({second}, timeout=BUDGET + 0.7)
                assert second in done, "parse queue outlived the response budget"
                # Once the first request times out the queued request can finish.
                response = second.result()
                assert response.status_code == 200 or _pending(response)
            finally:
                body.release.set()
                for task in (first, second):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(first, second, return_exceptions=True)
            assert (await _add(client, "recovery")).status_code == 200

    asyncio.run(scenario())


def test_distinct_adds_keep_both_forget_contexts_and_retries_are_idempotent(monkeypatch) -> None:
    """DAT-001: a later Add must not replace an earlier suppression association.

    Baseline service._record_forget_requests/forget.ledger_id omits Add identity.
    This unchanged node must fail the final suppression assertion on baseline,
    then pass after including the Add's stable request identity.
    """
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, repository = _forget_service()
    tenant = tenant_for("forget-collision-user")
    old_exchange = "Your orange tabby deserves a comfortable leather carrier."
    first = AddRequest(
        request_id="forget-first", user_id="forget-collision-user", session_id="same-session",
        messages=_messages(
            ("user", "Help me choose a pet carrier."),
            ("assistant", old_exchange),
            ("user", "Please forget my orange tabby cat Garfield."),
        ),
    )
    second = AddRequest(
        request_id="forget-second", user_id="forget-collision-user", session_id="same-session",
        messages=_messages(
            ("user", "Help me choose a desk lamp."),
            ("assistant", "A warm lamp is comfortable for evening reading."),
            ("user", "Please forget my orange tabby cat Garfield."),
        ),
    )
    hit = _hit("old-exchange", old_exchange, "same-session")

    async def scenario() -> None:
        await service.add(first)
        before = repository.forget_requests(tenant)
        assert drop_hits([hit], ledger_entries(before))[0] == []
        first_ids = {row.id for row in before}
        await service.add(second)
        await service.add(first)  # Stored-receipt retry must not add another row.
        after = repository.forget_requests(tenant)
        assert drop_hits([hit], ledger_entries(after))[0] == [], (
            "a later Add restored evidence suppressed by an earlier forget request"
        )
        assert len(after) == 2
        assert first_ids <= {row.id for row in after}

    asyncio.run(scenario())
