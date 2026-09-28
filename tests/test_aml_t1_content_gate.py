"""W4: T-1's content gate and its v2 (relative, anchored) week render.

Round two, 2026-09-28. 90.6% of the first Textual Full's Searches took the code route, where the
route gate switches T-1 off, so T-1 barely reached the traffic it was built for. The content gate
resolves on every route and protects code by what the text looks like instead; the v2 render keeps
week expressions relative, which AML's LoCoMo and LongMemEval judge requires. Both are off by
default: the served C9 renders exactly as before.

Each test names the mutation of the production code it was watched to fail on (the red proof).
"""

from __future__ import annotations

from datetime import date

import pytest

from recall_aml.temporal_render import looks_like_code, resolve_relative_times, resolve_text
from recall_aml.variants import variant
from tests.test_aml_relative_dates_and_adjacency import (
    CODE_QUERY,
    _add,
    _item,
    _search,
)
from tests.test_aml_specialist_fusion import _service

ANCHOR = date(2023, 5, 8)  # a Monday


@pytest.mark.parametrize(
    ("phrase", "resolution"),
    [
        ("last week", "[= the week before 2023-05-08]"),
        ("this week", "[= the week of 2023-05-08]"),
        ("next week", "[= the week after 2023-05-08]"),
        ("last weekend", "[= the weekend before 2023-05-08]"),
        ("next weekend", "[= the weekend after 2023-05-08]"),
        ("two weeks ago", "[= 2 weeks before 2023-05-08]"),
        ("a week ago", "[= 1 week before 2023-05-08]"),
        # Not week-based: exactly as v1.
        ("yesterday", "[= 2023-05-07]"),
        ("3 days ago", "[≈ 2023-05-05]"),
        ("last month", "[= 2023-04]"),
        ("last Friday", "[= 2023-05-05]"),
    ],
)
def test_v2_keeps_week_expressions_relative_and_everything_else_as_v1(phrase: str, resolution: str) -> None:
    """Invariant: under v2 a week expression is rendered relative to the item's own day, and no
    other expression changes.

    Red proof: `_resolve_v2_week` returning None unconditionally sends every week expression back
    to the v1 form (``[week of 2023-05-01]``) and fails the equality for the week rows.
    """
    assert resolve_text(f"We met {phrase}.", ANCHOR, render="v2") == f"We met {phrase} {resolution}."


def test_code_punctuation_protects_a_phrase_but_prose_punctuation_does_not() -> None:
    """Invariant: under the content gate a phrase glued to code (``date.today()``,
    ``today.strftime``, ``$yesterday``, ``yesterday=True``) is left as written, while a phrase
    ending a sentence or clause is still resolved.

    Red proof: counting ``.`` after a phrase as code whatever follows it (dropping the
    identifier condition in `_glued_to_code`) leaves "We met yesterday." unresolved and fails the
    first assertion.
    """
    assert resolve_text("We met yesterday.", ANCHOR, skip_code=True) == "We met yesterday [= 2023-05-07]."
    assert resolve_text("Plans for next week: go", ANCHOR, skip_code=True).startswith(
        "Plans for next week [week of 2023-05-15]:"
    )
    for code in ("x = date.today()", 'today.strftime("%d")', "$yesterday ok", "since yesterday=True"):
        assert resolve_text(code, ANCHOR, skip_code=True) == code, code


def test_the_content_gate_leaves_an_item_that_looks_like_code_untouched() -> None:
    """Invariant: an item with two or more code signals is kept byte for byte by the content gate,
    and the same item is resolved by the route gate's plain render.

    Red proof: removing the `looks_like_code` skip in `resolve_relative_times` resolves the
    ``yesterday`` in the comment and fails the first assertion.
    """
    code = "def load(): return cache == None  # rebuilt yesterday"
    assert looks_like_code(code)
    assert not looks_like_code("We met yesterday; it was fun (really).")
    items = [_item("c", code, 2)]
    assert resolve_relative_times(items, skip_code=True)[0].content == code
    assert resolve_relative_times(items)[0].content != code


def test_the_content_gate_resolves_prose_on_the_code_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: with ``RECALL_AML_T1_GATE=content`` a prose item is resolved even when the query
    took the code route, which the route gate never does.

    Red proof: dropping ``content_gate or`` from the gate condition in `HostedService.search` puts
    the route gate back and fails the ``endswith`` assertion.
    """
    monkeypatch.setenv("RECALL_AML_RESOLVE_RELATIVE_TIMES", "1")
    monkeypatch.setenv("RECALL_AML_T1_GATE", "content")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add(service, "w4-code", "w4-c", "I adopted a puppy yesterday.", 2)
    response = _search(service, "w4-code", CODE_QUERY)
    assert response.data, "precondition: the item is returned"
    assert any(str(item.content).endswith("yesterday [= 2023-05-02].") for item in response.data)
    assert "+t1-content-gate-v1" in service.search_content_profile


def test_the_served_c9_keeps_the_route_gate_and_the_v1_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: W4 changes nothing served until the owner switches it on: C9 keeps the route
    gate and v1, and its search-content profile carries no W4 suffix.

    Red proof: setting ``relative_times_gate="content"`` on the C9 variant fails the first
    assertion.
    """
    c9 = variant("C9_routed_specialists_grounded_graph_atomic")
    assert (c9.relative_times_gate, c9.relative_times_render) == ("route", "v1")
    monkeypatch.delenv("RECALL_AML_T1_GATE", raising=False)
    monkeypatch.delenv("RECALL_AML_T1_RENDER", raising=False)
    monkeypatch.setenv("RECALL_AML_RESOLVE_RELATIVE_TIMES", "1")
    service, _, _, _, _ = _service("C7_routed_specialists")
    assert service.search_content_profile.endswith("+relative-times-resolved-v1")


@pytest.mark.parametrize(("name", "value"), [("RECALL_AML_T1_GATE", "everywhere"), ("RECALL_AML_T1_RENDER", "v3")])
def test_a_bad_w4_value_stops_service_startup(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    """Invariant: an unknown gate or render is refused when the service starts, not on a Search.

    Red proof: removing ``self.relative_times_gate`` from the startup read in
    `HostedService.__init__` lets the service start and fails the gate row.
    """
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _service("C7_routed_specialists")
