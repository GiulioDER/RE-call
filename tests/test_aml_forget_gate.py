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

Added by the audit of #812 (2026-10-01), red-proved on the testbench host:

* ``test_the_ledger_gate_acts_on_every_route``: the annotation discarded on the code route
  (``items = items if specialist_route == "code" else annotated``) fails the code-route annotate
  row at the note assertion; the gate made to mean "code route only"
  (``(self.forget_gate == "ledger") == (specialist_route == "code")``) fails the context rows at
  ``forget_requests_applied == 1``.
* ``test_the_ledger_gate_never_cuts_or_drops_code``: ``keep_code`` always ``None``; fails at the
  ``return [...]`` assertion (stub cuts it, drop removes it). The fallback ``elif`` disabled
  (``and False``) fails both modes at the ``ANNOTATION_PREFIX`` assertion; ``keep_code`` made to
  keep every item (``lambda _text: True``) fails both at the prose assertion (the gift sentence
  still states the detail).
* ``test_the_ledger_gate_leaves_stubbed_prose_as_the_route_gate_serves_it`` (architect gate and
  differential review on the audit): code protection on every route (the
  ``specialist_route == "code"`` term removed from ``protect_code``) fails at the D.C. stub
  assertion; the gate made to mean "code route only" fails there too.
* ``test_the_code_route_note_counts_only_code``: ``only=keep_code`` removed from the fallback
  ``annotate_items`` call, which is the audit fix as first written; fails at
  ``not any(... startswith(ANNOTATION_PREFIX))``.
* ``test_data_gates_allow_forget_under_its_ledger_gate``: baseline a2fb64bd; fails at the match
  (the forget refusal fires instead).
* ``test_the_gate_is_a_variant_setting_the_environment_overrides``: a mutation dropping
  ``self._behavior.forget_gate`` from the property (env or ``route``) fails at ``== "ledger"``.
* ``test_the_ledger_gate_with_forget_off_is_reported``: baseline a2fb64bd; fails at the
  warning assertion.
* ``test_the_gate_is_reported_by_version``: the ``"off"`` branch removed; fails at the last
  equality. ``test_an_unknown_gate_stops_service_startup`` now matches the message that names
  the variable and its values; baseline a2fb64bd fails at the match.
"""

from __future__ import annotations

import asyncio

import pytest

from dataclasses import replace
import logging

from recall_aml.forget import ANNOTATION_PREFIX, STUB_SENTENCE
from recall_aml.models import AddRequest
from recall_aml.variants import variant
from tests.test_aml_forget import (
    CODE_QUERY,
    CONTEXT_QUERY,
    DETAIL,
    _ingest,
    _messages,
    _rounds,
    _search,
    _service,
    _text,
)

GATE = "RECALL_AML_FORGET_GATE"


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)


@pytest.mark.parametrize("mode", ["drop", "stub", "annotate"])
@pytest.mark.parametrize(("query", "route"), [(CODE_QUERY, "code"), (CONTEXT_QUERY, "context")])
def test_the_ledger_gate_acts_on_every_route(monkeypatch, mode, query, route) -> None:
    """Invariant: under the ledger gate each mode acts on the code route and on the others, and
    does what the mode says (audit of #812: the counter alone let a no-op pass)."""
    monkeypatch.setenv("RECALL_AML_FORGET", mode)
    monkeypatch.setenv(GATE, "ledger")
    service, _ = _service()
    _ingest(service, "code-user")

    response = _search(service, "code-user", query=query)

    assert response.specialist_route == route
    assert response.forget_requests_applied == 1
    if mode == "drop":
        assert not any(DETAIL in _text(item) for item in response.data)
    elif mode == "stub":
        assert not any(DETAIL in _text(item) for item in response.data)
        assert any(STUB_SENTENCE in _text(item) for item in response.data)
    else:
        assert _text(response.data[0]).startswith(ANNOTATION_PREFIX)
        assert any(DETAIL in _text(item) for item in response.data)


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

    monkeypatch.setenv("RECALL_AML_FORGET", "off")
    assert service.forget_suppression_profile["code_route"] == "off"


def test_an_unknown_gate_stops_service_startup(monkeypatch) -> None:
    monkeypatch.setenv(GATE, "tenant")
    with pytest.raises(ValueError, match="RECALL_AML_FORGET_GATE must be one of route, ledger"):
        _service()


# Audit of #812 (CCA STANDARD, 2026-10-01). Red proofs ran on the testbench host against the
# rebased pre-fix commit (a2fb64bd) unless a mutation is named; see the module docstring.

CODE_MEMORY = (
    "```python\n# pens.py\ndef inventory():\n"
    "    return ['I collect vintage fountain pens from Italian makers']\n```"
)


def _ingest_with_code(service, user: str) -> None:
    _ingest(service, user)
    asyncio.run(
        service.add(
            AddRequest(
                request_id=f"{user}-code",
                user_id=user,
                session_id=f"{user}:code",
                messages=_messages(("user", CODE_MEMORY), ("assistant", "Inventory function added.")),
            )
        )
    )


@pytest.mark.parametrize("mode", ["stub", "drop"])
def test_the_ledger_gate_never_cuts_or_drops_code(monkeypatch, mode) -> None:
    """Invariant: under the ledger gate, stub and drop leave a code-shaped item as written (the
    sentence split cut code mid-token; drop removed code a task needs) and it is annotated
    instead, while prose stating the target is still stubbed or dropped."""
    monkeypatch.setenv("RECALL_AML_FORGET", mode)
    monkeypatch.setenv(GATE, "ledger")
    service, _ = _service()
    _ingest_with_code(service, "code-user")

    response = _search(service, "code-user", query=CODE_QUERY, top_k=20)

    code_line = "return ['I collect vintage fountain pens from Italian makers']"
    assert any(code_line in _text(item) for item in response.data), "the code memory is served whole"
    assert _text(response.data[0]).startswith(ANNOTATION_PREFIX)
    assert not any("any gift ideas" in _text(item) and DETAIL in _text(item) for item in response.data)


ACROSS_SENTENCES = "Vintage. Fountain. Pens. Italian. Makers."
# Prose the code detector reads as code ("D.C." matches its file-extension pattern).
LOOKS_LIKE_CODE = "We moved to Washington, D.C. last spring. I collect vintage fountain pens from Italian makers."


def _ingest_across_sentences(service, user: str) -> None:
    _ingest(service, user)
    for name, text in (("across", ACROSS_SENTENCES), ("dc", LOOKS_LIKE_CODE)):
        asyncio.run(
            service.add(
                AddRequest(
                    request_id=f"{user}-{name}",
                    user_id=user,
                    session_id=f"{user}:{name}",
                    messages=_messages(("user", text), ("assistant", "Noted, a fine hobby.")),
                )
            )
        )


def test_the_ledger_gate_leaves_stubbed_prose_as_the_route_gate_serves_it(monkeypatch) -> None:
    """Invariant: off the code route, ledger + stub serves exactly what route + stub serves; the
    code fallback note never fires on prose stub chose to leave (architect gate on the #812 audit:
    an item stating the target only across sentences was annotated), and prose the detector
    mistakes for code is still stubbed (differential review: "D.C." exempted a window)."""
    monkeypatch.setenv("RECALL_AML_FORGET", "stub")
    route_service, _ = _service()
    _ingest_across_sentences(route_service, "code-user")
    expected = _search(route_service, "code-user", query=CONTEXT_QUERY, top_k=20)

    monkeypatch.setenv(GATE, "ledger")
    service, _ = _service()
    _ingest_across_sentences(service, "code-user")
    response = _search(service, "code-user", query=CONTEXT_QUERY, top_k=20)

    assert any(ACROSS_SENTENCES in _text(item) for item in response.data), "precondition: served unchanged"
    dc_items = [item for item in response.data if "Washington, D.C." in _text(item)]
    assert dc_items, "precondition: the D.C. memory is served"
    assert all(STUB_SENTENCE in _text(item) and DETAIL not in _text(item) for item in dc_items)
    assert not any(_text(item).startswith(ANNOTATION_PREFIX) for item in response.data)
    assert [item.model_dump_json() for item in response.data] == [
        item.model_dump_json() for item in expected.data
    ]


def test_the_code_route_note_counts_only_code(monkeypatch) -> None:
    """Invariant: on the code route under ledger + stub, the fallback note names only what code
    left as written states; prose stub chose to serve (a target stated only across sentences)
    does not raise it."""
    monkeypatch.setenv("RECALL_AML_FORGET", "stub")
    monkeypatch.setenv(GATE, "ledger")
    service, _ = _service()
    _ingest(service, "code-user")
    asyncio.run(
        service.add(
            AddRequest(
                request_id="code-user-across",
                user_id="code-user",
                session_id="code-user:across",
                messages=_messages(("user", ACROSS_SENTENCES), ("assistant", "Noted, a fine hobby.")),
            )
        )
    )

    response = _search(service, "code-user", query=CODE_QUERY, top_k=20)

    assert response.specialist_route == "code"
    assert any(ACROSS_SENTENCES in _text(item) for item in response.data), "precondition: served unchanged"
    assert not any(_text(item).startswith(ANNOTATION_PREFIX) for item in response.data)


def test_data_gates_allow_forget_under_its_ledger_gate(monkeypatch) -> None:
    """Invariant: the #810 guard refuses forget only under its route gate; under the ledger gate
    startup passes the forget check (this test double then stops at the image-vector check)."""
    monkeypatch.setenv("RECALL_AML_ROUTE_GATES", "data")
    monkeypatch.setenv("RECALL_AML_T1_GATE", "content")
    monkeypatch.setenv("RECALL_AML_FORGET", "annotate")
    monkeypatch.setenv(GATE, "ledger")
    with pytest.raises(ValueError, match="has_multimodal_vectors"):
        _service()


def test_the_gate_is_a_variant_setting_the_environment_overrides(monkeypatch) -> None:
    """Invariant: the gate is a ``HostedVariant`` field (C9 keeps ``route``), overridden by the
    environment, which is read case-insensitively."""
    assert variant("C9_routed_specialists_grounded_graph_atomic").forget_gate == "route"
    service, _ = _service()
    service._behavior = replace(service._behavior, forget_gate="ledger")
    assert service.forget_gate == "ledger"
    monkeypatch.setenv(GATE, " ROUTE ")
    assert service.forget_gate == "route"


def test_the_ledger_gate_with_forget_off_is_reported(monkeypatch, caplog) -> None:
    """Invariant: the ledger gate set while forget is off has no effect, and startup says so."""
    monkeypatch.setenv(GATE, "ledger")
    with caplog.at_level(logging.WARNING, logger="recall_aml"):
        _service()
    assert any("RECALL_AML_FORGET_GATE=ledger set while" in record.getMessage() for record in caplog.records)
