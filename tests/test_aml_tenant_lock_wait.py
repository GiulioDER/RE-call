"""The tenant queue: a mutation of one user waits for the user's earlier ones, in order, holding nothing.

Served: every Add, delete and sparse backfill takes the cross-process tenant lock with a blocking
``pg_advisory_lock``, which the pool's 25 s statement timeout cancels. In Coding Full 1's first
half hour (2026-09-28, 09:01 to 09:31 UTC) 69 Add requests answered 503 ``QueryCanceled`` that
way, and the owner's journal check (2026-09-30: 51 completions, 42 distinct, 16 over 100 s) puts
roughly 55 to 65 of them on distinct same-user Adds queued behind a compile, which #806 does not
absorb. Each held an Add slot idle for 25 s, and the lock then sat free until AML's next retry.

``RECALL_AML_TENANT_LOCK_WAIT_SECONDS`` gives each user a first-in-first-out queue in the process
(C9 runs one worker), entered before the Add slot: a queued mutation holds no slot, thread or
connection, starts the moment the one ahead finishes, and only the head of the queue takes the
tenant lock. Unset keeps the served path. This replaces the first version's polling of
``pg_try_advisory_lock``, which the audit of #811 found held the Add slot, lost arrival order, let
a blocking delete overtake an Add and leaked the lock on cancellation.

Red proofs, run 2026-10-01 on the testbench host, each failing at the assertion named, then green.
The polling version of this PR (``03fdc9db``) has no queue, so the queue tests are proved on a
mutation of the new code, and only the parsing and startup tests against that version:

* ``test_an_unset_budget_keeps_the_served_path``: the ``if wait <= 0`` early exit removed from
  ``HostedService.tenant_gate``; fails at ``entered == ["both"]`` with ``["refused"]`` (the gate
  times out at once instead of being a no-op).
* ``test_queued_mutations_start_in_arrival_order_after_the_holder``: the gate entered without
  waiting when it is held (``if gate.lock.locked(): yield; return``); fails at the event order.
* ``test_a_queued_add_holds_no_add_slot``: ``admitted`` takes ``add_slots`` before the gate;
  fails at ``other in done`` (the other user's Add waits for the slot the queued Add holds).
* ``test_a_full_queue_answers_tenant_busy_and_stores_nothing``: the ``max_waiters`` check removed;
  fails at ``third in done`` (the third Add queues instead of being answered).
* ``test_a_spent_budget_raises_tenant_lock_busy``: ``timeout=wait`` replaced by ``timeout=None``;
  fails at ``attempt in done``.
* ``test_a_delete_waits_behind_a_queued_add``: ``delete_user`` without ``tenant_gate``; fails at
  the event order (the delete runs while the holder still holds the queue).
* ``test_a_mutation_queued_at_shutdown_does_not_start``: the ``_stopping`` check after the gate is
  acquired removed; fails at ``isinstance(..., TenantLockBusy)``.
* ``test_the_budget_is_parsed_strictly``: baseline ``03fdc9db``, which accepted ``1801`` and
  ``١٢`` and named no variable; fails at the ``1801`` row (DID NOT RAISE).
* ``test_the_app_refuses_a_queue_the_response_budget_cannot_carry``: baseline ``03fdc9db``; fails
  with DID NOT RAISE at the first row.
* ``test_the_budget_is_reported_by_version``: the ``/version`` key removed; fails at the equality.

Added at the architect gate (each line below survived the first test set under the mutation
named; now red at the assertion named):

* ``test_the_app_shutdown_stops_queued_mutations``: the lifespan's ``begin_shutdown()`` call
  removed; fails at ``service._stopping is True``.
* ``test_a_mutation_arriving_after_shutdown_is_refused_at_once``: the check before queueing
  removed; fails at ``attempt in done`` (it queues behind the holder).
* ``test_a_sparse_backfill_waits_behind_a_queued_add``: ``prepare_sparse_user`` outside the
  queue; fails at the event order.
* ``test_a_timed_out_or_cancelled_waiter_leaves_no_stuck_queue``: ``users`` left incremented on
  the timeout path; fails at ``users == 1``.
* ``test_the_queue_events_are_logged``: the ``hosted_tenant_queue_full`` log removed; fails at
  its ``in messages``.
* ``test_a_delete_waits_at_most_the_response_budget``: the app passing no budget to
  ``delete_user``; fails at ``seen == [{"budget": 10}]``.
* ``test_the_app_refuses_a_queue_with_one_add_slot``: the ``add_concurrency < 2`` check removed;
  fails with DID NOT RAISE.
* ``test_the_queue_cap_holds_during_a_hand_off`` (differential review): the cap's first version,
  ``gate.lock.locked() and waiting >= max_waiters`` with ``waiting`` from ``locked()``; fails at
  ``refused == ["C"]`` with ``[]`` (both late arrivals admitted).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from starlette.testclient import TestClient

from recall_aml.app import create_app
from recall_aml.config import HostedSettings
from recall_aml.identity import tenant_for
from recall_aml.models import AddResponse
from recall_aml.service import TenantLockBusy
from tests.test_aml_multimodal_scope import TIMESTAMP_MS, _add_text_memory
from tests.test_aml_specialist_fusion import _service

ENV = "RECALL_AML_TENANT_LOCK_WAIT_SECONDS"
TENANT = tenant_for("queue-user")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _settings(**overrides) -> HostedSettings:
    return HostedSettings("postgresql://unused", "secret", "abc123", **overrides)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def test_an_unset_budget_keeps_the_served_path() -> None:
    """Invariant: unset, the gate is a no-op (no queue is created, nothing waits) and an Add
    takes the served path."""
    service, repository, _, _, _ = _service("C7_routed_specialists")
    entered: list[str] = []

    async def scenario() -> None:
        try:
            async with service.tenant_gate(TENANT):
                async with service.tenant_gate(TENANT):
                    entered.append("both")
        except TenantLockBusy:
            entered.append("refused")

    asyncio.run(scenario())
    assert entered == ["both"]
    assert service._tenant_gates == {}
    assert service.tenant_lock_wait_seconds == 0
    _add_text_memory(service, "queue-user")
    assert repository.receipts


def test_queued_mutations_start_in_arrival_order_after_the_holder(monkeypatch) -> None:
    """Invariant: with the queue on, a mutation enters only when the one ahead leaves, in
    arrival order."""
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")
    events: list[str] = []

    async def scenario() -> None:
        release = asyncio.Event()

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                events.append("holder in")
                await release.wait()
                events.append("holder out")

        async def queued(name: str) -> None:
            async with service.tenant_gate(TENANT):
                events.append(name)

        first = asyncio.create_task(holder())
        await _settle()
        waiters = [asyncio.create_task(queued(name)) for name in ("A", "B", "C")]
        await _settle()
        release.set()
        await asyncio.gather(first, *waiters)

    asyncio.run(scenario())
    assert events == ["holder in", "holder out", "A", "B", "C"]
    assert service._tenant_gates == {}, "an empty queue is dropped"


def _queue_app(monkeypatch, *, add_concurrency: int):
    """A C7 app with the queue on, whose ``service.add`` blocks until the test releases it."""
    monkeypatch.setenv(ENV, "200")
    service, _, _, _, _ = _service("C7_routed_specialists")
    app = create_app(_settings(add_concurrency=add_concurrency, add_response_budget_seconds=10), service)
    started: list[str] = []
    gates: dict[str, asyncio.Event] = {}

    async def fake_add(model, *, fingerprint=None):
        started.append(model.request_id)
        await gates.setdefault(model.request_id, asyncio.Event()).wait()
        return AddResponse(
            request_id=model.request_id,
            user_id=model.user_id,
            session_id=model.session_id,
            raw_count=1,
            compiled_count=0,
            compiler_fallback=False,
        )

    service.add = fake_add
    return service, app, started, gates


def _add_body(request_id: str, user_id: str) -> dict:
    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": "s",
        "messages": [{"role": "user", "content": "The receipt was 14 euros.", "timestamp": TIMESTAMP_MS}],
    }


async def _until(condition, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("precondition not reached")
        await asyncio.sleep(0.01)


def test_a_queued_add_holds_no_add_slot(monkeypatch) -> None:
    """Invariant: a same-user Add queued behind a running one holds no Add slot, so another
    user's Add runs at once (the audit's finding 1: the polling version held the slot)."""
    service, app, started, gates = _queue_app(monkeypatch, add_concurrency=2)
    tenant = tenant_for("busy-user")

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://aml") as client:

            def post(request_id: str, user_id: str):
                return asyncio.create_task(
                    client.post("/v1/add", headers={"X-Api-Key": "secret"}, json=_add_body(request_id, user_id))
                )

            first = post("x1", "busy-user")
            await _until(lambda: started == ["x1"])
            second = post("x2", "busy-user")
            await _until(lambda: service._tenant_gates.get(tenant) is not None and service._tenant_gates[tenant].users == 2)
            gates["y1"] = asyncio.Event()
            gates["y1"].set()
            other = post("y1", "other-user")
            done, _ = await asyncio.wait({other}, timeout=3)
            assert other in done, "another user's Add must not wait for the queued one"
            assert other.result().status_code == 200
            assert started == ["x1", "y1"], "the queued same-user Add has not started"
            gates["x1"].set()
            await _until(lambda: "x2" in started)
            gates["x2"].set()
            assert (await first).status_code == 200
            assert (await second).status_code == 200

    asyncio.run(scenario())


def test_a_full_queue_answers_tenant_busy_and_stores_nothing(monkeypatch) -> None:
    """Invariant: with ``add_concurrency`` Adds of one user already running or queued, the next
    is answered 503 ``tenant_busy`` with ``Retry-After: 60`` at once, without starting, so one
    user holds at most half the flight slots."""
    service, app, started, gates = _queue_app(monkeypatch, add_concurrency=2)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://aml") as client:

            def post(request_id: str):
                return asyncio.create_task(
                    client.post("/v1/add", headers={"X-Api-Key": "secret"}, json=_add_body(request_id, "busy-user"))
                )

            first = post("x1")
            await _until(lambda: started == ["x1"])
            second = post("x2")
            await _until(lambda: service._tenant_gates[tenant_for("busy-user")].users == 2)
            third = post("x3")
            done, _ = await asyncio.wait({third}, timeout=5)
            assert third in done, "a full queue must answer at once"
            refused = third.result()
            assert refused.status_code == 503
            assert refused.json() == {"error": "tenant_busy"}
            assert refused.headers.get("Retry-After") == "60"
            assert "x3" not in started
            for name in ("x1", "x2"):
                gates.setdefault(name, asyncio.Event()).set()
            await asyncio.gather(first, second)

    asyncio.run(scenario())


def test_a_spent_budget_raises_tenant_lock_busy(monkeypatch) -> None:
    """Invariant: a mutation still queued when its budget runs out raises ``TenantLockBusy``
    and never enters."""
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")

    async def scenario() -> None:
        release = asyncio.Event()
        entered: list[str] = []

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()

        async def queued() -> None:
            async with service.tenant_gate(TENANT, budget=0.05):
                entered.append("queued")

        first = asyncio.create_task(holder())
        await _settle()
        attempt = asyncio.create_task(queued())
        done, _ = await asyncio.wait({attempt}, timeout=2)
        assert attempt in done, "the budget must end the wait"
        assert isinstance(attempt.exception(), TenantLockBusy)
        assert entered == []
        release.set()
        await first

    asyncio.run(scenario())


def test_a_delete_waits_behind_a_queued_add(monkeypatch) -> None:
    """Invariant: a delete joins the user's queue, so it never overtakes an earlier Add (the
    audit's finding 4: a blocking delete beat every polling Add)."""
    monkeypatch.setenv(ENV, "1800")
    service, repository, _, _, _ = _service("C7_routed_specialists")
    events: list[str] = []
    repository.delete_tenant = lambda tenant: events.append("delete") or 0

    async def scenario() -> None:
        release = asyncio.Event()

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()
                events.append("holder out")

        async def queued_add() -> None:
            async with service.tenant_gate(TENANT):
                events.append("add")

        first = asyncio.create_task(holder())
        await _settle()
        add = asyncio.create_task(queued_add())
        await _settle()
        delete = asyncio.create_task(service.delete_user("queue-user"))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(first, add, delete)

    asyncio.run(scenario())
    assert events == ["holder out", "add", "delete"]


def test_a_mutation_queued_at_shutdown_does_not_start(monkeypatch) -> None:
    """Invariant: once shutdown has begun, a queued mutation reaching its turn raises
    ``TenantLockBusy`` instead of starting a compile the drain would cut off."""
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")

    async def scenario() -> BaseException | None:
        release = asyncio.Event()
        entered: list[str] = []

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()

        async def queued() -> None:
            async with service.tenant_gate(TENANT):
                entered.append("queued")

        first = asyncio.create_task(holder())
        await _settle()
        waiter = asyncio.create_task(queued())
        await _settle()
        service.begin_shutdown()
        release.set()
        await first
        await asyncio.wait({waiter})
        assert entered == []
        return waiter.exception()

    assert isinstance(asyncio.run(scenario()), TenantLockBusy)


@pytest.mark.parametrize("value", ["24", "1801", "ten", "١٢", "-5", "1.5"])
def test_the_budget_is_parsed_strictly(monkeypatch, value: str) -> None:
    """Invariant: a set value must be ASCII whole seconds within 25 to 1800, and the error names
    the variable."""
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        _service("C7_routed_specialists")


@pytest.mark.parametrize(("value", "seconds"), [("", 0), ("0", 0), ("25", 25), (" 1800 ", 1800)])
def test_the_accepted_budgets(monkeypatch, value: str, seconds: int) -> None:
    monkeypatch.setenv(ENV, value)
    service, _, _, _, _ = _service("C7_routed_specialists")
    assert service.tenant_lock_wait_seconds == seconds


@pytest.mark.parametrize(
    ("wait", "budget", "message"),
    [
        ("200", 0, "needs RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS"),
        ("1800", 10, "at most 264"),
    ],
)
def test_the_app_refuses_a_queue_the_response_budget_cannot_carry(
    monkeypatch, wait: str, budget: float, message: str
) -> None:
    """Invariant: the queue starts only with #806's response budget on, and never waits longer
    than 24 of AML's 32 attempts at that budget."""
    monkeypatch.setenv(ENV, wait)
    service, _, _, _, _ = _service("C7_routed_specialists")
    with pytest.raises(ValueError, match=message):
        create_app(_settings(add_response_budget_seconds=budget), service)


def test_a_coherent_queue_starts(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")
    create_app(_settings(add_response_budget_seconds=80), service)


def test_the_budget_is_reported_by_version(monkeypatch) -> None:
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")
    client = TestClient(create_app(_settings(add_response_budget_seconds=80), service))

    assert client.get("/version").json().get("tenant_lock_wait_seconds") == 1800



# Architect gate on the redesign (2026-10-01): lines no test could fail on, and the bound on a
# delete's wait. Red proofs as in the module docstring.


def test_a_timed_out_or_cancelled_waiter_leaves_no_stuck_queue(monkeypatch) -> None:
    """Invariant: a waiter that times out or is cancelled leaves the queue's bookkeeping clean:
    a later mutation enters at once and an empty queue is dropped."""
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")

    async def scenario() -> None:
        release = asyncio.Event()
        entered: list[str] = []

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()

        async def queued(name: str, budget: float | None = None) -> None:
            async with service.tenant_gate(TENANT, budget=budget):
                entered.append(name)

        first = asyncio.create_task(holder())
        await _settle()
        timed_out = asyncio.create_task(queued("timed out", budget=0.05))
        cancelled = asyncio.create_task(queued("cancelled"))
        await asyncio.sleep(0.1)
        cancelled.cancel()
        done, _ = await asyncio.wait({timed_out, cancelled}, timeout=2)
        assert done == {timed_out, cancelled}, "both waiters must have left the queue"
        assert isinstance(timed_out.exception(), TenantLockBusy)
        assert cancelled.cancelled()
        assert service._tenant_gates[TENANT].users == 1, "only the holder is left"
        release.set()
        await first
        assert service._tenant_gates == {}
        await asyncio.wait_for(queued("later"), timeout=1)
        assert entered == ["later"]
        assert service._tenant_gates == {}

    asyncio.run(scenario())


def test_a_mutation_arriving_after_shutdown_is_refused_at_once(monkeypatch) -> None:
    """Invariant: once shutdown has begun, a new mutation is refused before it queues, not after
    waiting behind the holder."""
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")

    async def scenario() -> None:
        release = asyncio.Event()

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()

        async def late() -> None:
            async with service.tenant_gate(TENANT, budget=5):
                pass

        first = asyncio.create_task(holder())
        await _settle()
        service.begin_shutdown()
        attempt = asyncio.create_task(late())
        done, _ = await asyncio.wait({attempt}, timeout=0.5)
        assert attempt in done, "a mutation arriving after shutdown must not queue"
        assert isinstance(attempt.exception(), TenantLockBusy)
        release.set()
        await first

    asyncio.run(scenario())


def test_the_app_shutdown_stops_queued_mutations(monkeypatch) -> None:
    """Invariant: the app's lifespan begins the service's shutdown before it drains flights."""
    monkeypatch.setenv(ENV, "200")
    service, _, _, _, _ = _service("C7_routed_specialists")
    app = create_app(_settings(add_concurrency=2, add_response_budget_seconds=10), service)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            assert service._stopping is False
        assert service._stopping is True

    asyncio.run(scenario())


def test_a_sparse_backfill_waits_behind_a_queued_add(monkeypatch) -> None:
    """Invariant: a sparse backfill joins the user's queue like a delete."""
    from dataclasses import replace

    monkeypatch.setenv(ENV, "1800")
    service, repository, _, _, _ = _service("C7_routed_specialists")
    service._behavior = replace(service._behavior, learned_sparse=True)
    events: list[str] = []
    repository.backfill_sparse = lambda tenant: events.append("backfill") or {}

    async def scenario() -> None:
        release = asyncio.Event()

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()
                events.append("holder out")

        first = asyncio.create_task(holder())
        await _settle()
        backfill = asyncio.create_task(service.prepare_sparse_user("queue-user"))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(first, backfill)

    asyncio.run(scenario())
    assert events == ["holder out", "backfill"]


def test_a_delete_waits_at_most_the_response_budget(monkeypatch) -> None:
    """Invariant: the app bounds a delete's (and a backfill's) queue wait by the response
    budget, since each is one plain request under the tunnel's cut and the stop timeout."""
    monkeypatch.setenv(ENV, "200")
    service, _, _, _, _ = _service("C7_routed_specialists")
    seen: list[dict] = []

    async def capture_delete(user_id, **kwargs):
        seen.append(kwargs)
        return 0

    service.delete_user = capture_delete
    client = TestClient(create_app(_settings(add_concurrency=2, add_response_budget_seconds=10), service))

    response = client.post("/v1/delete", headers={"X-Api-Key": "secret"}, json={"user_id": "queue-user"})

    assert response.status_code == 200
    assert seen == [{"budget": 10}]


def test_the_queue_events_are_logged(monkeypatch, caplog) -> None:
    """Invariant: a full queue and a spent budget are logged, the evidence a live run reads."""
    import logging

    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")

    async def scenario() -> None:
        release = asyncio.Event()

        async def holder() -> None:
            async with service.tenant_gate(TENANT):
                await release.wait()

        async def queued(**kwargs) -> None:
            async with service.tenant_gate(TENANT, **kwargs):
                pass

        first = asyncio.create_task(holder())
        await _settle()
        waiting = asyncio.create_task(queued(budget=5))
        await _settle()
        refused = asyncio.create_task(queued(max_waiters=1))
        expired = asyncio.create_task(queued(budget=0.05))
        done, _ = await asyncio.wait({refused, expired}, timeout=2)
        assert done == {refused, expired}, "a full queue and a spent budget must both answer"
        assert isinstance(refused.exception(), TenantLockBusy)
        assert isinstance(expired.exception(), TenantLockBusy)
        release.set()
        await asyncio.gather(first, waiting)

    with caplog.at_level(logging.INFO, logger="recall_aml"):
        asyncio.run(scenario())
    messages = [record.getMessage() for record in caplog.records]
    assert "hosted_tenant_queue_full" in messages
    assert "hosted_tenant_queue_busy" in messages
    assert "hosted_tenant_queue_waited" in messages


def test_the_app_refuses_a_queue_with_one_add_slot(monkeypatch) -> None:
    """Invariant: with one Add slot no same-user Add could queue (every one would be answered
    tenant_busy, worse than the served wait), so the app refuses the combination."""
    monkeypatch.setenv(ENV, "200")
    service, _, _, _, _ = _service("C7_routed_specialists")
    with pytest.raises(ValueError, match="RECALL_AML_ADD_CONCURRENCY of at least 2"):
        create_app(_settings(add_concurrency=1, add_response_budget_seconds=10), service)


def test_the_queue_cap_holds_during_a_hand_off(monkeypatch) -> None:
    """Invariant: the cap counts the woken waiter whose turn has not run yet, so arrivals in that
    instant (the lock reads free) cannot overrun it (differential review, note 1)."""
    monkeypatch.setenv(ENV, "1800")
    service, _, _, _, _ = _service("C7_routed_specialists")

    async def scenario() -> tuple[list[str], list[str]]:
        entered: list[str] = []
        refused: list[str] = []

        async def entrant(name: str) -> None:
            try:
                async with service.tenant_gate(TENANT, max_waiters=1):
                    entered.append(name)
            except TenantLockBusy:
                refused.append(name)

        holder = service.tenant_gate(TENANT)
        await holder.__aenter__()
        first = asyncio.create_task(entrant("A"))
        await _settle()
        late = [asyncio.create_task(entrant(name)) for name in ("B", "C")]
        # Release without yielding: B and C run before A's wake-up, with the lock reading free.
        await holder.__aexit__(None, None, None)
        await asyncio.gather(first, *late)
        return entered, refused

    entered, refused = asyncio.run(scenario())
    assert refused == ["C"]
    assert entered == ["A", "B"]
