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

Audit fixes, 2026-09-30. Red proof ran on the testbench host against the four production files at
the PR head before the fixes (``add_flight.py``, ``app.py``, ``config.py``, ``service.py``), this
file unchanged, each failing in its own assertion; then green on the fixed tree (13 passed):

* ``test_shutdown_lets_a_pending_add_finish_before_the_pool_closes``: ``[[]] == [['r1']]``, the
  pool closed before the pending Add wrote its receipt (``AddFlights.drain`` in the lifespan).
* ``test_shutdown_reports_the_adds_it_could_not_wait_for``: ``[] == [1]``, no log of the abandoned
  flight.
* ``test_adds_beyond_the_flight_cap_wait_without_reading_their_body``: ``6 == 2``, every queued Add
  was decoded and held (the ``flight_slots`` gate before ``_parse``).
* ``test_a_joined_resend_does_not_keep_its_own_copy_of_the_add``: ``[4] == [1]``, each joined
  resend kept its own decoded request (``del model`` on the joined path).
* ``test_the_budget_from_the_environment_is_off_by_default_and_bounded``: ``80.0 == 0``, unset
  turned the path on (``_add_response_budget_from_env``).
* ``test_a_finished_flight_is_never_joined``: ``running()`` returned the finished task (it now
  treats a done task as absent). A first version awaited the flight to finish and passed on the
  unfixed code, because ``asyncio.wait`` resumes after the cleanup callback; it recreates the
  window directly instead.
* ``test_every_error_and_fast_path_gives_its_flight_slot_back`` (added at the architect gate): the
  early-exit release in ``run_within_budget`` changed to ``if owned and False:`` failed the codes
  assertion with 503s where 422s were expected: two leaked slots stopped every later request.
* ``test_another_payload_under_the_same_id_never_gets_this_adds_answer`` is a control that passes
  on both; mutating ``AddIdentity.fingerprint`` to ``field(compare=False)`` (flights keyed on
  tenant and request id only) failed ``200 == 409``: the other payload was handed this Add's answer.
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


# ---------------------------------------------------------------- audit fixes (2026-09-30)


def _app(service: HostedService, *, budget: float = BUDGET, slots: int = 8, shutdown: Any = None) -> Any:
    settings = HostedSettings(
        database_url="postgresql://unused",
        api_key="k",
        git_commit="c",
        add_concurrency=slots,
        add_response_budget_seconds=budget,
    )
    return create_app(settings, service, shutdown=shutdown)


def _client_for(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://c9", timeout=30)


async def _add_payload(
    client: httpx.AsyncClient, request_id: str, user: str, text: str
) -> httpx.Response:
    return await client.post(
        "/v1/add",
        headers=HEADERS,
        json={
            "request_id": request_id,
            "user_id": user,
            "session_id": "s",
            "messages": [{"role": "user", "content": text}],
        },
    )


def test_shutdown_lets_a_pending_add_finish_before_the_pool_closes() -> None:
    """FIX-A: an Add answered pending is no request task, so uvicorn stopped waiting for it and the
    lifespan closed the pool under it; the Add was cancelled mid-flight with no receipt."""
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)
    at_shutdown: list[list[str]] = []
    app = _app(service, shutdown=lambda: at_shutdown.append(list(repository.receipts)))

    async def scenario() -> httpx.Response:
        async with app.router.lifespan_context(app):
            async with _client_for(app) as client:
                first = await _add(client)
            asyncio.get_running_loop().call_later(BUDGET, compiler.gate.set)
        return first

    try:
        first = asyncio.run(scenario())
    finally:
        compiler.gate.set()

    assert _pending(first)
    assert at_shutdown == [["r1"]]


def test_shutdown_reports_the_adds_it_could_not_wait_for(monkeypatch, caplog) -> None:
    """FIX-A: when the drain runs out, the abandoned Adds are counted in the log, not dropped
    silently (a cancelled flight used to leave no trace)."""
    import recall_aml.app as app_module

    monkeypatch.setattr(app_module, "SHUTDOWN_DRAIN_SECONDS", 0.2, raising=False)
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)
    app = _app(service, shutdown=lambda: None)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            async with _client_for(app) as client:
                assert _pending(await _add(client))

    with caplog.at_level("INFO", logger="recall_aml"):
        try:
            asyncio.run(scenario())
        finally:
            compiler.gate.set()

    abandoned = [r for r in caplog.records if r.message == "hosted_add_flights_abandoned"]
    assert [getattr(record, "flights", None) for record in abandoned] == [1]


def test_adds_beyond_the_flight_cap_wait_without_reading_their_body(monkeypatch) -> None:
    """FIX-B: every queued Add held its decoded body (up to 44 MiB) while it waited for a slot,
    where the old path had not read it yet; 40 pending Adds held 40 bodies. At most twice the Add
    concurrency are now decoded and held; the rest wait unread and are answered pending."""
    import recall_aml.app as app_module

    parsed: list[str] = []
    real_parse = app_module._parse

    async def counting_parse(request: Any, model: Any) -> Any:
        result = await real_parse(request, model)
        parsed.append(getattr(result, "request_id", "?"))
        return result

    monkeypatch.setattr(app_module, "_parse", counting_parse)
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)

    async def scenario() -> list[httpx.Response]:
        async with _client_for(_app(service, slots=1)) as client:
            try:
                return list(
                    await asyncio.gather(*(_add(client, f"r{index}", f"u{index}") for index in range(6)))
                )
            finally:
                compiler.gate.set()

    responses = asyncio.run(scenario())

    assert all(_pending(response) for response in responses)
    assert len(parsed) == 2


def test_a_joined_resend_does_not_keep_its_own_copy_of_the_add(monkeypatch) -> None:
    """FIX-B: a resend that joins a running flight kept its own decoded copy of the Add for as
    long as it waited; only the flight's copy is ever used."""
    import gc
    import weakref

    import recall_aml.app as app_module

    alive: list[weakref.ref[Any]] = []
    real_parse = app_module._parse

    async def tracking_parse(request: Any, model: Any) -> Any:
        result = await real_parse(request, model)
        alive.append(weakref.ref(result))
        return result

    monkeypatch.setattr(app_module, "_parse", tracking_parse)
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)
    held: list[int] = []

    async def scenario() -> None:
        async with _client_for(_app(service, budget=1.0, slots=1)) as client:
            try:
                first = asyncio.create_task(_add(client))
                await _until(lambda: compiler.calls == 1)
                resends = [asyncio.create_task(_add(client)) for _ in range(3)]
                await _until(lambda: len(alive) == 4)
                await asyncio.sleep(0.1)
                gc.collect()
                held.append(sum(ref() is not None for ref in alive))
                await asyncio.gather(first, *resends)
            finally:
                compiler.gate.set()

    asyncio.run(scenario())

    assert held == [1]


def test_every_error_and_fast_path_gives_its_flight_slot_back(monkeypatch) -> None:
    """FIX-B: the flight slot is taken before the body is read, so every way out of the handler
    must return it. After 422s, 403s, an answer from the receipt and a 409 from the receipt read,
    three new Adds must still find both slots (add_concurrency 1, so 2) free: 2 parsed, 1 queued."""
    import recall_aml.app as app_module

    parsed: list[str] = []
    real_parse = app_module._parse

    async def counting_parse(request: Any, model: Any) -> Any:
        result = await real_parse(request, model)
        parsed.append(getattr(result, "request_id", "?"))
        return result

    monkeypatch.setattr(app_module, "_parse", counting_parse)
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)
    settings = HostedSettings(
        database_url="postgresql://unused",
        api_key="k",
        git_commit="c",
        add_concurrency=1,
        add_response_budget_seconds=BUDGET,
        authorized_user_id="u",
    )
    codes: list[int] = []

    async def scenario() -> list[int]:
        async with _client_for(create_app(settings, service)) as client:
            try:
                for _ in range(5):
                    response = await client.post(
                        "/v1/add",
                        headers={**HEADERS, "content-type": "application/json"},
                        content=b"{not json",
                    )
                    codes.append(response.status_code)
                for _ in range(5):
                    codes.append((await _add(client, "rf", "intruder")).status_code)
                compiler.gate.set()
                codes.append((await _add_payload(client, "rok", "u", "first")).status_code)
                codes.append((await _add_payload(client, "rok", "u", "first")).status_code)
                codes.append((await _add_payload(client, "rok", "u", "other")).status_code)
                compiler.gate.clear()
                parsed.clear()
                last = await asyncio.gather(*(_add(client, f"q{index}", "u") for index in range(3)))
                return [response.status_code for response in last]
            finally:
                compiler.gate.set()

    last = asyncio.run(scenario())

    assert codes == [422] * 5 + [403] * 5 + [200, 200, 409]
    assert last == [503, 503, 503]
    assert len(parsed) == 2


def test_the_budget_from_the_environment_is_off_by_default_and_bounded(monkeypatch) -> None:
    """FIX-D: unset turned the new path on at 80 s, the only recall_aml switch on by default, so
    an upgrade changed Add answers silently; and 100 s or more (the tunnel's cut) or a fraction of
    a second (AML's 32 attempts gone in seconds) were accepted."""
    for name, value in (
        ("RECALL_AML_DATABASE_URL", "postgresql://unused"),
        ("RECALL_AML_API_KEY", "k"),
        ("RECALL_AML_GIT_COMMIT", "c"),
        ("RECALL_AML_EMBED_LOCK_PATH", "lock"),
        ("RECALL_AML_AUTHORIZED_USER_ID", "*"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS", raising=False)
    assert HostedSettings.from_env().add_response_budget_seconds == 0

    for good, expected in (("0", 0.0), (" ", 0.0), ("10", 10.0), ("80", 80.0), ("95", 95.0)):
        monkeypatch.setenv("RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS", good)
        assert HostedSettings.from_env().add_response_budget_seconds == expected

    for bad in ("0.5", "9", "96", "100", "80000", "80s"):
        monkeypatch.setenv("RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS", bad)
        with pytest.raises(ValueError, match="RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS"):
            HostedSettings.from_env()


def test_a_finished_flight_is_never_joined(monkeypatch) -> None:
    """FIX-G: a flight is forgotten by a done callback one loop turn after it finishes; a resend
    looked up in that turn joined the dead flight and got its error instead of a fresh start."""
    from recall_aml.add_flight import AddFlights, AddIdentity

    identity = AddIdentity("aml_t", "digest", "fingerprint")

    async def scenario() -> tuple[bool, str]:
        flights: AddFlights[str] = AddFlights()

        async def failing() -> str:
            raise RuntimeError("transient provider failure")

        async def fresh() -> str:
            return "fresh"

        first, _ = flights.join_or_start(identity, failing)
        await asyncio.wait({first})
        # The state of the window between the flight finishing and its done callback running:
        # awaiting it here lets the callback run first, so the window is recreated directly.
        flights._running[identity] = first
        assert flights.running(identity) is None
        task, joined = flights.join_or_start(identity, fresh)
        return joined, "joined the dead flight" if joined else await task

    assert asyncio.run(scenario()) == (False, "fresh")


def test_another_payload_under_the_same_id_never_gets_this_adds_answer() -> None:
    """FIX-J control: the same request id with another payload is its own flight and ends 409; it
    must never join the running Add and be handed that Add's result."""
    repository = ReceiptRepository()
    compiler = GatedCompiler()
    service = _service(repository, compiler)

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        async with _client_for(_app(service, budget=2.0)) as client:
            try:
                first = asyncio.create_task(_add_payload(client, "r1", "u", "we chose postgres"))
                await _until(lambda: compiler.calls == 1)
                other = asyncio.create_task(_add_payload(client, "r1", "u", "we chose mysql"))
                await asyncio.sleep(0.1)
                compiler.gate.set()
                return await first, await other
            finally:
                compiler.gate.set()

    first, other = asyncio.run(scenario())

    assert first.status_code == 200, first.text
    assert other.status_code == 409, other.text
    assert repository.receipts == ["r1"]
