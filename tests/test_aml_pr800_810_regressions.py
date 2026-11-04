"""Regressions for BUG-001 and DAT-001 from the audit of PRs 800 through 810.

Red proof, 2026-10-28: all five nodes failed their intended assertions on
master a0a09650a9ff30ae31460af5f1b6a202466d1d5b in isolated remote execution. The three
preflight parameters failed at the deadline assertion, the parse queue at its
deadline assertion, and the ledger test at the suppression assertion.

Recovery flake, 2026-11-04 (CI runs 41839327205 and 41869536951, ``assert 503 == 200``):
the recovery Add was sent once and is itself bound by the 0.3 s response budget, so on a
loaded runner it was answered pending (its own compile, or its wait for the single Add slot
that the parse test's still-running ``queued`` flight held). The four budget nodes now send it
through ``_recovered``, which resends while answered pending. Proof on a remote test host against 311510bf9,
all four nodes each time: with an induced load (a pytest plugin adding 0.5 s to
``GatedCompiler.compile_anchored_v3`` and ``ReceiptRepository.record_receipt``) the old test
failed 4 of 4 at the recovery ``status_code == 200`` and the new one passed 4 of 4; with the
mutation ``async with parse_slots:`` replaced by a never released ``parse_slots.acquire()``
in ``recall_aml.app.create_app.run_within_budget``, old and new both failed 4 of 4 at that
same assertion; unmutated and unloaded, the new test passed 4 of 4 in five runs.
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

#: How many times a recovery Add is resent while answered pending. Each attempt lasts at most
#: one response budget, so this bounds a test whose admission really leaked to about 5 s.
RECOVERY_ATTEMPTS = 16


async def _recovered(client: httpx.AsyncClient, request_id: str = "recovery") -> httpx.Response:
    """Send one Add the way AML does: resend it while the service answers it pending.

    The recovery Add is itself bound by the 0.3 s response budget, so a single attempt measured
    the runner's speed, not admission. Two CI flakes on 2026-11-04 (``assert 503 == 200``) were
    exactly that: on a loaded runner the recovery's own work, or its wait for the one Add slot
    that a still-running earlier flight held, outlived 0.3 s and was answered pending. The
    service's pending answer is the explicit "not yet" signal: a resend joins the running flight
    or reads its receipt, so it never enters ``service.add`` again. Any other answer is returned
    at once. Admission that really leaked keeps every attempt pending, so the caller's
    ``status_code == 200`` still fails.
    """
    response = await _add(client, request_id)
    for _ in range(RECOVERY_ATTEMPTS - 1):
        if not _pending(response):
            return response
        response = await _add(client, request_id)
    return response


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
        # The ASGI client does not run lifespan setup; prepare a spare execution lane.
        warm = threading.Barrier(2)
        await asyncio.gather(
            asyncio.to_thread(warm.wait, 2), asyncio.to_thread(warm.wait, 2),
        )
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
                # A fresh Add uses clean hooks while the original work remains blocked.
                monkeypatch.setattr(app_module, "add_identity", real_identity)
                monkeypatch.setattr(service, "stored_add", real_stored)
                assert (await _recovered(client)).status_code == 200
                assert service.add_calls == 1
                assert (tenant_for("u"), "recovery") in repository.stored
            finally:
                release.set()
                receipt_release.set()
                body.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

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
            assert (await _recovered(client)).status_code == 200

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
