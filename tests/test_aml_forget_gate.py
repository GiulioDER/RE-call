"""R2-1 forget suppression behind the ledger gate (``RECALL_AML_FORGET_GATE``), on top of PR 805.

As built, suppression never acts on the code route, and 246 of 253 PersonaMem-v2 forget Searches
route to code as sent, so it almost never acts. The ledger gate lets the tenant's own forget ledger
decide instead: a user who never asked to forget has no ledger entries (the detector finds no
request in any of 196 Coding sessions), so nothing changes for them. Design: recall-lab
``research/designs/2026-09-28-route-architecture.md``, reviewed 2026-09-28. Unset keeps ``route``.

Red proofs, each run 2026-09-28 against the deliberate mutation named, failing at the assertion
named, then green after restoring the line:

* ``test_the_ledger_gate_acts_on_a_code_route_search``: the gate condition in
  ``HostedService.search`` reverted to ``specialist_route != "code"``; fails at
  ``forget_requests_applied == 1`` with 0 (``drop`` parameter; the run stops at the first).
* ``test_the_route_gate_is_the_default_and_exempts_the_code_route``: the unset default changed to
  ``ledger``; fails at ``service.forget_gate == "route"``.
* ``test_the_gate_is_reported_by_version``: ``code_route`` always ``exempt``; fails at the
  ``ledger-gated`` equality.
* ``test_an_unknown_gate_stops_service_startup``: ``self.forget_gate`` removed from the startup
  read; fails with DID NOT RAISE.
* ``test_a_tenant_that_never_asked_to_forget_is_untouched_under_the_ledger_gate``:
  ``_forget_entries`` reading another user's ledger (``tenant_for("code-user")``); fails at
  ``forget_requests_applied == 0`` with 1. A first version of this test had no other user with a
  ledger in the service, and that mutation survived it; the test now holds one. Dropping with an
  empty ledger (``if forget.mode == "drop"`` without ``forget_entries``) also survives, correctly:
  it is the same behaviour, since nothing matches an empty ledger.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.test_aml_forget import CODE_QUERY, DETAIL, _ingest, _rounds, _search, _service, _text

GATE = "RECALL_AML_FORGET_GATE"


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)


@pytest.mark.parametrize("mode", ["drop", "stub", "annotate"])
def test_the_ledger_gate_acts_on_a_code_route_search(monkeypatch, mode) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", mode)
    monkeypatch.setenv(GATE, "ledger")
    service, _ = _service()
    _ingest(service, "code-user")

    response = _search(service, "code-user", query=CODE_QUERY)

    assert response.specialist_route == "code"
    assert response.forget_requests_applied == 1
    if mode in {"drop", "stub"}:
        assert not any(DETAIL in _text(item) for item in response.data)


def test_the_route_gate_is_the_default_and_exempts_the_code_route(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, _ = _service()
    _ingest(service, "code-user")

    response = _search(service, "code-user", query=CODE_QUERY)

    assert service.forget_gate == "route"
    assert response.forget_requests_applied == 0
    assert any(DETAIL in _text(item) for item in response.data)


def test_a_tenant_that_never_asked_to_forget_is_untouched_under_the_ledger_gate(monkeypatch) -> None:
    """Every Coding tenant's case: no ledger entry, so the code-route Search is served as before.

    Another user of the same service HAS asked to forget the same detail, so a gate that read the
    wrong tenant's ledger would show here. Under the route gate that could never matter on the
    code route; under the ledger gate it would.
    """
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    baseline_service, _ = _service()
    _ingest(baseline_service, "code-user")
    for request in _rounds("coding-like")[:-1]:  # the same memories, no forget request
        asyncio.run(baseline_service.add(request))
    baseline = _search(baseline_service, "coding-like", query=CODE_QUERY)

    monkeypatch.setenv(GATE, "ledger")
    service, repository = _service()
    _ingest(service, "code-user")
    for request in _rounds("coding-like")[:-1]:
        asyncio.run(service.add(request))
    response = _search(service, "coding-like", query=CODE_QUERY)

    assert response.forget_requests_applied == 0
    assert [item.model_dump_json() for item in response.data] == [
        item.model_dump_json() for item in baseline.data
    ]


def test_the_gate_is_reported_by_version(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "annotate")
    service, _ = _service()
    assert service.forget_suppression_profile["code_route"] == "exempt"

    monkeypatch.setenv(GATE, "ledger")
    assert service.forget_suppression_profile["code_route"] == "ledger-gated"


def test_an_unknown_gate_stops_service_startup(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "tenant")
    with pytest.raises(ValueError, match="unknown forget gate"):
        _service()
