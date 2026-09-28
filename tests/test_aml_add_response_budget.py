"""A slow Add is answered before the tunnel cuts it, and its resend joins the running work.

AML reaches C9 through a Cloudflare tunnel that answers 524 to an origin silent for about 100
seconds; AML then waits 60 seconds (a 524 carries no ``Retry-After``) and resends the same
request, which used to queue for an Add slot of its own behind every waiting compile. With
``HostedSettings.add_response_budget_seconds`` set, ``recall_aml.add_flight`` answers 503 with
``Retry-After: 1`` inside the budget, keeps the Add running, lets a resend join it, and answers a
resend of a finished Add from its receipt without a slot.

The service here is the production ``HostedService`` configured as C9, over an in-memory
repository whose receipts behave as the real ones do (same id and fingerprint: the stored
answer; same id, other payload: ``IdempotencyConflict``). The compile waits on an event, so a
test decides exactly when an Add is slow and when it finishes; it then fails with a timeout,
which C9 stores raw-only, as it would in production.

Red proof, 2026-09-28, each by mutating the named production line with this file unchanged,
watching the named assertion fail, then restoring it and watching the test pass:

* ``test_a_slow_add_is_answered_pending_and_its_resend_gets_the_result``: replacing
  ``if settings.add_response_budget_seconds > 0:`` in the ``add`` route of ``create_app``
  (``recall_aml/app.py``) with ``if False:`` failed ``assert first.status_code == 503`` with
  ``200``: the Add was answered only when its compile gave up, 3 seconds later.
* ``test_a_resend_while_the_add_runs_joins_it_instead_of_queueing_again``: making
  ``AddFlights.join_or_start`` (``recall_aml/add_flight.py``) skip its lookup and always start a
  new flight failed ``assert service.add_calls == 1`` with ``3``: both resends queued a flight
  of their own.
* ``test_a_resend_of_a_finished_add_is_answered_from_its_receipt_without_a_slot``: deleting the
  ``service.stored_add`` early return from ``run_within_budget`` failed
  ``assert resend.status_code == 200`` with ``503``: the resend queued for the one Add slot,
  held by another user's slow Add, until the budget ran out.
* ``test_a_failed_flight_is_forgotten_so_the_resend_does_the_work_afresh``: deleting
  ``del self._running[identity]`` from ``AddFlights._finished`` failed
  ``assert resend.status_code == 200`` with ``503`` (the resend joined the dead flight and was
  answered its ``CompilerCreditExhausted``).
* ``test_a_budget_of_zero_keeps_the_add_waiting_until_it_finishes``: changing that same route
  condition to ``>= 0`` failed ``assert response.status_code == 200`` with ``503``.
* ``test_the_budget_setting_is_validated_and_reported``: deleting the
  ``add_response_budget_seconds`` check in ``HostedSettings.__post_init__``
  (``recall_aml/config.py``) failed ``pytest.raises(ValueError)`` with DID NOT RAISE.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx
import pytest

from recall.errors import IdempotencyConflict
from recall_aml.app import create_app
from recall_aml.config import DEFAULT_ADD_RESPONSE_BUDGET_SECONDS, HostedSettings
from recall_aml.models import AddRequest, AddResponse
from recall_aml.service import HostedService
from recall_aml.variants import variant
from tests.test_aml_c9_coding_hardening import C9, HEADERS, RecordingRepository

BUDGET = 0.3
#: How long a gated compile waits for its test before giving up on its own.
GATE_TIMEOUT = 3.0


class ReceiptRepository(RecordingRepository):
    """Keeps receipts and answers them back, with the production conflict rule."""

    def __init__(self) -> None:
        super().__init__()
        self.stored: dict[tuple[str, str], tuple[str, str]] = {}

    def get_receipt(self, tenant: str, request_id: str, fingerprint: str) -> str | None:  # type: ignore[override]
        found = self.stored.get((tenant, request_id))
        if found is None:
            return None
        if found[0] != fingerprint:
            raise IdempotencyConflict()
        return found[1]

    def record_receipt(self, tenant: str, request_id: str, fingerprint: str, result: str) -> None:
        super().record_receipt(tenant, request_id, fingerprint, result)
        self.stored[(tenant, request_id)] = (fingerprint, result)


class OutOfCredit(Exception):
    status_code = 402


class GatedCompiler:
    """A compile that runs until its test opens the gate, then fails with ``error``."""

    def __init__(self, error: Exception | None = None) -> None:
        self.gate = threading.Event()
        self.calls = 0
        self.error = error or TimeoutError("provider timeout")

    def compile_anchored_v3(self, *args: Any) -> list[Any]:
        self.calls += 1
        self.gate.wait(timeout=GATE_TIMEOUT)
        raise self.error


class CountingService(HostedService):
    """Counts how many times an Add enters the service, which a joined resend must not do."""

    add_calls = 0

    async def add(self, request: AddRequest, *, fingerprint: str | None = None) -> AddResponse:
        self.add_calls += 1
        return await super().add(request, fingerprint=fingerprint)


def _service(repository: ReceiptRepository, compiler: Any) -> CountingService:
    behavior = variant(C9)
    return CountingService(
        repository,  # type: ignore[arg-type]
        compiler,
        object(),  # type: ignore[arg-type]
        behavior=behavior,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={behavior.context_embedding_profile: object()},  # type: ignore[dict-item]
    )


def _client(service: HostedService, *, budget: float = BUDGET, slots: int = 8) -> httpx.AsyncClient:
    settings = HostedSettings(
        database_url="postgresql://unused",
        api_key="k",
        git_commit="c",
        add_concurrency=slots,
        add_response_budget_seconds=budget,
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings, service)),
        base_url="http://c9",
        timeout=30,
    )


async def _add(client: httpx.AsyncClient, request_id: str = "r1", user: str = "u") -> httpx.Response:
    return await client.post(
        "/v1/add",
        headers=HEADERS,
        json={
            "request_id": request_id,
            "user_id": user,
            "session_id": "s",
            "messages": [{"role": "user", "content": f"we chose postgres for {request_id}"}],
        },
    )


async def _until(condition: Any, seconds: float = GATE_TIMEOUT + 2) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


def _pending(response: httpx.Response) -> bool:
    return (
        response.status_code == 503
        and response.headers.get("retry-after") == "1"
        and response.headers.get("x-recall-add-pending") == "1"
        and response.json() == {"error": "add_in_progress"}
    )


def test_a_slow_add_is_answered_pending_and_its_resend_gets_the_result() -> None:
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)

    async def scenario() -> tuple[httpx.Response, float, httpx.Response]:
        async with _client(service) as client:
            try:
                started = time.monotonic()
                first = await _add(client)
                elapsed = time.monotonic() - started
                compiler.gate.set()
                await _until(lambda: repository.receipts == ["r1"])
                return first, elapsed, await _add(client)
            finally:
                compiler.gate.set()

    first, elapsed, resend = asyncio.run(scenario())

    assert first.status_code == 503, first.text
    assert _pending(first)
    assert elapsed < BUDGET + 1.0
    assert resend.status_code == 200, resend.text
    assert resend.json()["compiler_fallback"] is True
    assert compiler.calls == 1
    assert service.add_calls == 1
    assert repository.receipts == ["r1"]


def test_a_resend_while_the_add_runs_joins_it_instead_of_queueing_again() -> None:
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)

    async def scenario() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        async with _client(service) as client:
            try:
                first = await _add(client)
                # Sent while the compile is still held, so it can only join or queue again.
                resend_while_running = await _add(client)
                compiler.gate.set()
                return first, resend_while_running, await _add(client)
            finally:
                compiler.gate.set()

    first, resend_while_running, last = asyncio.run(scenario())

    assert _pending(first)
    assert _pending(resend_while_running)
    assert last.status_code == 200, last.text
    assert service.add_calls == 1
    assert compiler.calls == 1
    assert repository.receipts == ["r1"]


def test_a_resend_of_a_finished_add_is_answered_from_its_receipt_without_a_slot() -> None:
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)

    async def scenario() -> tuple[httpx.Response, httpx.Response, float]:
        async with _client(service, slots=1) as client:
            try:
                compiler.gate.set()
                finished = await _add(client, "r1", "u")
                compiler.gate.clear()
                # Another user's slow Add now holds the only Add slot.
                other = await _add(client, "r2", "v")
                assert _pending(other)
                started = time.monotonic()
                resend = await _add(client, "r1", "u")
                return finished, resend, time.monotonic() - started
            finally:
                compiler.gate.set()

    finished, resend, elapsed = asyncio.run(scenario())

    assert finished.status_code == 200, finished.text
    assert resend.status_code == 200, resend.text
    assert resend.json() == finished.json()
    assert elapsed < BUDGET
    assert service.add_calls == 2  # r1 once and r2 once; the resend of r1 never entered


def test_a_failed_flight_is_forgotten_so_the_resend_does_the_work_afresh(caplog) -> None:
    repository = ReceiptRepository()
    compiler = GatedCompiler(OutOfCredit("insufficient credits"))
    service = _service(repository, compiler)

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        async with _client(service) as client:
            try:
                first = await _add(client)
                compiler.gate.set()
                # Logged by the flight's done callback, after it has forgotten the flight.
                await _until(lambda: _failed_flights(caplog) != [])
                compiler.error = TimeoutError("provider timeout")
                return first, await _add(client)
            finally:
                compiler.gate.set()

    with caplog.at_level("INFO", logger="recall_aml"):
        first, resend = asyncio.run(scenario())

    assert _pending(first)
    assert resend.status_code == 200, resend.text
    assert service.add_calls == 2
    assert compiler.calls == 2
    assert repository.receipts == ["r1"]
    assert _failed_flights(caplog) == ["CompilerCreditExhausted"]


def _failed_flights(caplog: Any) -> list[str]:
    return [
        getattr(record, "error_class")
        for record in caplog.records
        if record.message == "hosted_add_flight_failed"
    ]


def test_a_budget_of_zero_keeps_the_add_waiting_until_it_finishes() -> None:
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)

    async def scenario() -> httpx.Response:
        async with _client(service, budget=0) as client:
            loop = asyncio.get_running_loop()
            loop.call_later(BUDGET * 3, compiler.gate.set)
            try:
                return await _add(client)
            finally:
                compiler.gate.set()

    response = asyncio.run(scenario())

    assert response.status_code == 200, response.text
    assert "x-recall-add-pending" not in response.headers


def test_the_budget_setting_is_validated_and_reported(monkeypatch) -> None:
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="add_response_budget_seconds"):
            HostedSettings("postgresql://unused", "k", "c", add_response_budget_seconds=bad)

    monkeypatch.setenv("RECALL_AML_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("RECALL_AML_API_KEY", "k")
    monkeypatch.setenv("RECALL_AML_GIT_COMMIT", "c")
    monkeypatch.setenv("RECALL_AML_EMBED_LOCK_PATH", "lock")
    monkeypatch.setenv("RECALL_AML_AUTHORIZED_USER_ID", "*")
    monkeypatch.delenv("RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS", raising=False)
    assert HostedSettings.from_env().add_response_budget_seconds == DEFAULT_ADD_RESPONSE_BUDGET_SECONDS
    assert DEFAULT_ADD_RESPONSE_BUDGET_SECONDS < 100  # Cloudflare's origin timeout
    monkeypatch.setenv("RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS", "0")
    assert HostedSettings.from_env().add_response_budget_seconds == 0

    async def version() -> dict[str, Any]:
        async with _client(_service(ReceiptRepository(), GatedCompiler()), budget=42.5) as client:
            return (await client.get("/version", headers=HEADERS)).json()

    assert asyncio.run(version())["add_response_budget_seconds"] == 42.5
