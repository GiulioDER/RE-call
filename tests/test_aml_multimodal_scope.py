"""MM-1 (which Searches return images) and MM-3 (dated image items) behaviour proofs.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md.

Served C8 and C9 return a multimodal memory's images only when ``route_query`` picks the
``multimodal`` route, which needs one of ten English visual words in the question; the Stage 0
census found 4.4% of public multimodal questions do. These tests pin the two default-off
alternatives and the guard the pre-registration requires: a tenant with no image memories gets
byte-identical Search output under every setting.

The routed-specialist fixture is borrowed from ``tests/test_aml_specialist_fusion.py``. C7 is
used because it shares C9's routing, preservation and visual leg while needing no Add-time
compiler; the scope and date settings are read by ``HostedService`` for every variant.

Red proofs, each run on 2026-09-25 against a deliberate mutation of the production line named,
failing at the assertion named, then green after restoring it:

* ``test_preserve_scope_returns_the_image_of_a_memory_text_retrieval_found``: the preserve
  condition in ``HostedService.search`` reverted to ``visual_route`` only; fails at
  ``isinstance(first.content, list)`` (content is the text window).
* ``test_dual_scope_runs_the_visual_leg_on_a_question_without_a_visual_word``: the visual-leg
  condition reverted to ``visual_route`` only; fails at ``response.visual_leg``.
* ``test_a_tenant_without_images_gets_identical_search_output_under_every_setting``: the
  ``if visual_route or visual_hits`` guard replaced by ``if True``; the ``dual`` parameters fail
  at the first dump equality (RRF rescoring changes every item's score). Only ``dual`` runs the
  visual leg, so only its parameters can see this mutation. A first version of this test put the
  ``visual_leg`` check first and went red there instead, which is not proof of the output guard.
* ``test_dated_multimodal_content_puts_one_date_part_first_on_image_items``: the
  ``dated_multimodal_items`` call removed from ``HostedService.search``; fails at the first-part
  equality.
* ``test_dated_multimodal_items_dates_list_content_once_and_leaves_text_items_alone``:
  ``[header, *item.content]`` replaced by ``item.content`` in ``dated_multimodal_items``; fails at
  the part-count equality.
* ``test_an_unknown_scope_stops_service_startup``: the startup read of both overrides removed from
  ``HostedService.__init__``; fails because no ``ValueError`` is raised.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from recall_aml.models import AddRequest, SearchItem, SearchRequest, TextContentPart
from recall_aml.window_format import dated_multimodal_items
from tests.test_aml_multimodal import _png_data_url
from tests.test_aml_specialist_fusion import _service

TIMESTAMP_MS = 1726133200000  # 2024-09-12 09:26:40 UTC
DATE_PART = "[2024-09-12 09:26 UTC]"
QUERY = "parking receipt total"  # no visual, code or conversational signal: the code route


def _add_image_memory(service, user_id: str = "scope-user") -> None:
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": f"{user_id}-image",
                    "user_id": user_id,
                    "session_id": "scope-session",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "the parking receipt from the mall"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _png_data_url(b"receipt")},
                                },
                            ],
                            "timestamp": TIMESTAMP_MS,
                        }
                    ],
                }
            )
        )
    )


def _add_text_memory(service, user_id: str) -> None:
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": f"{user_id}-text",
                    "user_id": user_id,
                    "session_id": "text-session",
                    "messages": [
                        {
                            "role": "user",
                            "content": "The parking receipt total was 14 euros.",
                            "timestamp": TIMESTAMP_MS,
                        }
                    ],
                }
            )
        )
    )


def _search(service, user_id: str = "scope-user", query: str = QUERY):
    return asyncio.run(
        service.search(SearchRequest.model_validate({"query": query, "user_id": user_id}))
    )


def _has_image(item: SearchItem) -> bool:
    return isinstance(item.content, list) and any(
        getattr(part, "type", "") == "image_url" for part in item.content
    )


def test_route_scope_keeps_images_off_a_question_without_a_visual_word(monkeypatch) -> None:
    """Served behaviour (arm B): the image memory is found by its text, its image never returned."""
    monkeypatch.delenv("RECALL_AML_MULTIMODAL_SCOPE", raising=False)
    service, _, _, _, multimodal = _service("C7_routed_specialists")
    _add_image_memory(service)

    response = _search(service)

    assert response.specialist_route == "code"
    assert response.data
    assert not any(_has_image(item) for item in response.data)
    assert multimodal.query_inputs == []
    assert response.visual_leg is False


def test_preserve_scope_returns_the_image_of_a_memory_text_retrieval_found(monkeypatch) -> None:
    """Arm P: same text ranking, the found memory rendered with its original image."""
    monkeypatch.setenv("RECALL_AML_MULTIMODAL_SCOPE", "preserve")
    service, _, _, _, multimodal = _service("C7_routed_specialists")
    _add_image_memory(service)

    response = _search(service)

    first = response.data[0]
    assert response.specialist_route == "code"
    assert isinstance(first.content, list)
    assert _has_image(first)
    assert multimodal.query_inputs == [], "preserve must not run the visual leg"
    assert response.visual_leg is False


def test_dual_scope_runs_the_visual_leg_on_a_question_without_a_visual_word(monkeypatch) -> None:
    """Arm D: the visual leg runs and is fused although the router chose the code route."""
    monkeypatch.setenv("RECALL_AML_MULTIMODAL_SCOPE", "dual")
    service, _, _, _, multimodal = _service("C7_routed_specialists")
    _add_image_memory(service)

    response = _search(service)

    assert response.specialist_route == "code"
    assert response.visual_leg
    assert multimodal.query_inputs == [QUERY]
    assert _has_image(response.data[0])


@pytest.mark.parametrize("scope", ["route", "preserve", "dual"])
@pytest.mark.parametrize("dated", ["0", "1"])
def test_a_tenant_without_images_gets_identical_search_output_under_every_setting(
    monkeypatch, scope: str, dated: str
) -> None:
    """Apparatus check 1: every Textual and Coding user must see exactly today's Search output."""
    monkeypatch.delenv("RECALL_AML_MULTIMODAL_SCOPE", raising=False)
    monkeypatch.delenv("RECALL_AML_DATED_MULTIMODAL", raising=False)
    baseline_service, _, _, _, _ = _service("C7_routed_specialists")
    _add_text_memory(baseline_service, "text-user")
    baseline = _search(baseline_service, "text-user")

    monkeypatch.setenv("RECALL_AML_MULTIMODAL_SCOPE", scope)
    monkeypatch.setenv("RECALL_AML_DATED_MULTIMODAL", dated)
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_text_memory(service, "text-user")
    response = _search(service, "text-user")

    assert [item.model_dump_json() for item in response.data] == [
        item.model_dump_json() for item in baseline.data
    ]
    assert response.model_dump_json() == baseline.model_dump_json()
    assert response.visual_leg is False


def test_dated_multimodal_content_puts_one_date_part_first_on_image_items(monkeypatch) -> None:
    """MM-3 in the service: an image item reaches the reader with its own date as text."""
    monkeypatch.setenv("RECALL_AML_MULTIMODAL_SCOPE", "preserve")
    monkeypatch.setenv("RECALL_AML_DATED_MULTIMODAL", "1")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add_image_memory(service)

    first = _search(service).data[0]

    assert isinstance(first.content, list)
    assert first.content[0] == TextContentPart(type="text", text=DATE_PART)
    assert _has_image(first)


def test_dated_multimodal_items_dates_list_content_once_and_leaves_text_items_alone() -> None:
    """MM-3 as a pure transform: one leading part on list content, nothing else changes."""
    created = datetime(2024, 9, 12, 9, 26, 40, tzinfo=timezone.utc)
    parts = [
        {"type": "text", "text": "the parking receipt"},
        {"type": "image_url", "image_url": {"url": _png_data_url(b"receipt")}},
    ]
    common = {"source": "s", "session_id": "d1", "kind": "multimodal"}
    image_item = SearchItem.model_validate(
        {"id": "a", "content": parts, "created_at": created, "score": 0.5, **common}
    )
    text_item = SearchItem.model_validate(
        {"id": "b", "content": "a text item", "created_at": created, "score": 0.6, **common}
    )
    undated = SearchItem.model_validate({"id": "c", "content": parts, "score": 0.4, **common})

    dated = dated_multimodal_items([image_item, text_item, undated])

    assert [item.id for item in dated] == ["a", "b", "c"]
    assert len(dated[0].content) == len(image_item.content) + 1
    assert dated[0].content[0] == TextContentPart(type="text", text=DATE_PART)
    assert dated[0].content[1:] == image_item.content
    assert dated[0].score == image_item.score
    assert dated[1] == text_item
    assert dated[2] == undated


def test_an_unknown_scope_stops_service_startup(monkeypatch) -> None:
    """A mistyped experiment setting must fail loudly, not route every Search to the fallback."""
    monkeypatch.setenv("RECALL_AML_MULTIMODAL_SCOPE", "always")
    with pytest.raises(ValueError, match="unknown multimodal scope"):
        _service("C7_routed_specialists")
