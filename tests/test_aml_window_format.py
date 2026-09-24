"""The two date formats of docs/preregistrations/2026-09-25-aml-c9-window-format.md.

Red proof, 2026-09-25, each by mutating the named production line and watching the named test fail
in its assertion, then restoring it:

* ``test_the_detector_tells_a_coding_trajectory_from_a_conversation``: raising
  ``CODING_MESSAGE_SHARE`` in ``recall_aml/window_format.py`` to ``1.01`` made every Add a
  conversation, and the trajectory assertion failed.
* ``test_dated_items_prefix_the_created_at_in_utc_minutes``: dropping ``.astimezone(timezone.utc)``
  in ``dated_items`` rendered ``13:56`` for a +02:00 time and failed the equality.
* ``test_per_track_windows_date_a_conversation_and_not_a_trajectory``: deleting the
  ``if per_track_windows:`` branch in ``recall_aml/service.py`` ``build_chunks`` left the
  conversation content-only, and the ``timestamp:`` assertion failed.
* ``test_a_dated_search_ranks_exactly_as_an_undated_one``: deleting the ``dated_search_content``
  branch before ``SearchResponse`` in ``HostedService.search`` returned undated content and failed
  the prefix assertion.
* ``test_served_c9_keeps_both_formats_off``: setting ``per_track_windows=True`` on the C9 variant
  in ``recall_aml/variants.py`` failed the first assertion.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from recall_aml.models import AddRequest, Message, SearchItem, SearchRequest
from recall_aml.variants import DEFAULT_VARIANT, variant
from recall_aml.window_format import dated_items, looks_like_coding
from tests.test_aml_c9_coding_hardening import C9, FailingCompiler, RecordingRepository, add
from tests.test_aml_hosted import make_service

WHEN = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)

CONVERSATION = [
    Message(role="user", content="Caroline: I went to an LGBTQ support group yesterday.", timestamp=WHEN),
    Message(role="user", content="Melanie: That sounds lovely! How did it feel?", timestamp=WHEN),
    Message(role="user", content="Caroline: Honestly, freeing. I felt accepted; I'll go again.", timestamp=WHEN),
    Message(role="user", content="Melanie: I painted a sunrise last week [shared image: a lake].", timestamp=WHEN),
]
TRAJECTORY = [
    Message(role="user", content="The July reconciliation is short. Work out why.", timestamp=WHEN),
    Message(role="assistant", content='{"file_path": "incidents/reconciliation_gap.md"}', timestamp=WHEN),
    Message(role="assistant", content="def consolidate(rows):\n    return {r.key: r for r in rows}", timestamp=WHEN),
    Message(role="assistant", content="The dedup key drops distinct orders from delta.", timestamp=WHEN),
]


def test_the_detector_tells_a_coding_trajectory_from_a_conversation() -> None:
    assert looks_like_coding(TRAJECTORY) is True
    assert looks_like_coding(CONVERSATION) is False
    assert looks_like_coding([]) is False


def test_dated_items_prefix_the_created_at_in_utc_minutes() -> None:
    plus_two = timezone(timedelta(hours=2))
    items = [
        SearchItem(id="a", content="Caroline went to the group.", created_at=WHEN.astimezone(plus_two),
                   source="s", session_id="s", kind="raw", score=1.0),
        SearchItem(id="b", content="naive", created_at=datetime(2023, 5, 8, 13, 56),
                   source="s", session_id="s", kind="raw", score=0.9),
        SearchItem(id="c", content="undated", source="s", session_id="s", kind="raw", score=0.8),
    ]
    dated = dated_items(items)
    assert [item.content for item in dated] == [
        "[2023-05-08 13:56 UTC] Caroline went to the group.",
        "[2023-05-08 13:56 UTC] naive",
        "undated",
    ]
    assert [item.id for item in dated] == ["a", "b", "c"]


def test_per_track_windows_date_a_conversation_and_not_a_trajectory() -> None:
    per_track = dataclasses.replace(variant(C9), per_track_windows=True)

    def raw_texts(messages: list[Message]) -> list[str]:
        repository = RecordingRepository()
        response = add(
            c9_client_with(repository, per_track),
            [m.model_dump(mode="json") for m in messages],
        )
        assert response.status_code == 200, response.text
        return [
            chunk.text
            for chunk in repository.persisted["raw"]
            if chunk.metadata.get("record_type") == "raw"
        ]

    conversation = raw_texts(CONVERSATION)
    trajectory = raw_texts(TRAJECTORY)
    assert conversation and all("timestamp: 2023-05-08" in text for text in conversation)
    assert trajectory and not any("timestamp:" in text for text in trajectory)


def c9_client_with(repository: RecordingRepository, behavior):
    from starlette.testclient import TestClient

    from recall_aml.app import create_app
    from recall_aml.config import HostedSettings
    from recall_aml.service import HostedService

    service = HostedService(
        repository,
        FailingCompiler(),
        object(),
        behavior=behavior,
        multimodal_embedder=object(),
        specialist_retrievers={behavior.context_embedding_profile: object()},
    )
    settings = HostedSettings(database_url="postgresql://unused", api_key="k", git_commit="c")
    return TestClient(create_app(settings, service))


@pytest.mark.anyio
async def test_a_dated_search_ranks_exactly_as_an_undated_one() -> None:
    results = {}
    for dated in (False, True):
        behavior = dataclasses.replace(variant(DEFAULT_VARIANT), dated_search_content=dated)
        service, _, _ = make_service(behavior=behavior)
        await service.add(
            AddRequest(
                request_id="r1",
                user_id="user-a",
                session_id="session-a",
                messages=[Message(role="user", content="run pytest for WidgetError", timestamp=WHEN)],
            )
        )
        results[dated] = await service.search(
            SearchRequest(query="WidgetError", user_id="user-a", top_k=5)
        )
    plain, dated_result = results[False].data, results[True].data
    assert plain
    assert [item.id for item in dated_result] == [item.id for item in plain]
    for before, after in zip(plain, dated_result, strict=True):
        if before.created_at is None:
            assert after.content == before.content
        else:
            assert after.content == f"[2023-05-08 13:56 UTC] {before.content}"
    assert any(item.created_at is not None for item in dated_result)


def test_served_c9_keeps_both_formats_off() -> None:
    served = variant(C9)
    assert served.per_track_windows is False
    assert served.dated_search_content is False
    assert served.content_only_windows is True

