"""Session reconstruction and view planning for the hosted atomic rescue builder.

Every test builds its windows with the service's own ``build_chunks`` and the C8 window settings,
so a change in how the service cuts or labels windows fails here rather than in a live artifact.

Red proofs (2026-09-22, each mutation applied to ``scripts/build_aml_atomic_rescue_artifact.py``
and reverted):

* ``test_sessions_rebuild_exactly_from_c8_windows`` failed when ``rebuild_sessions`` extended
  ``words`` with every token of each window (dropped the overlap slice).
* ``test_a_missing_segment_is_refused`` failed when the segment completeness check was removed.
* ``test_windows_that_disagree_on_their_overlap_are_refused`` failed when the overlap comparison
  was removed.
* ``test_every_view_maps_to_the_raw_window_that_contains_it`` failed when ``plan_views`` mapped
  each view to ``chunk_ids[0]``.
* ``test_builder_binds_to_the_corpus_identity_the_service_serves`` failed when
  ``served_corpus_fingerprint`` returned the raw store digest for a graph-sidecar variant (the
  first builder's behaviour, which the live VPS3 dry run exposed).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from recall.types import Chunk
from recall_aml.models import AddRequest, Message
from recall_aml.service import build_chunks
from recall_aml.variants import variant
from scripts.build_aml_atomic_rescue_artifact import BuildRefusal, plan_views, rebuild_sessions


C8 = variant("C8_routed_specialists_grounded_graph")


def _raw_windows(session: str, messages: list[str]) -> list[Chunk]:
    request = AddRequest(
        request_id=f"req-{session}",
        user_id="user",
        session_id=session,
        messages=[Message(role="user", content=content) for content in messages],
    )
    chunks = build_chunks(
        request,
        [],
        word_window_size=C8.word_window_size,
        word_window_stride=C8.word_window_stride,
        content_only_windows=C8.content_only_windows,
        stable_window_identity=C8.stable_window_order,
    )
    return [chunk for chunk in chunks if chunk.metadata.get("record_type") == "raw"]


def _messages(count: int) -> list[str]:
    return [
        " ".join(f"m{index}w{position}" for position in range(97)) + f" finished step {index}."
        for index in range(count)
    ]


def test_sessions_rebuild_exactly_from_c8_windows() -> None:
    messages = _messages(5)
    windows = _raw_windows("s-1", messages)
    assert len(windows) >= 4
    (rebuilt,) = rebuild_sessions(reversed(windows))
    assert rebuilt.text == " ".join(" ".join(messages).split())
    assert (rebuilt.window_size, rebuilt.window_stride) == (160, 120)
    assert rebuilt.chunk_ids == tuple(chunk.id for chunk in windows)


def test_a_missing_segment_is_refused() -> None:
    windows = _raw_windows("s-2", _messages(5))
    with pytest.raises(BuildRefusal, match="missing window segments"):
        rebuild_sessions(windows[:1] + windows[2:])


def test_windows_that_disagree_on_their_overlap_are_refused() -> None:
    windows = _raw_windows("s-3", _messages(5))
    tampered = windows[1]
    words = tampered.text.split()
    words[0] = "tampered"
    windows[1] = replace(tampered, text=" ".join(words))
    with pytest.raises(BuildRefusal, match="overlap"):
        rebuild_sessions(windows)


def test_two_texts_for_one_segment_are_refused() -> None:
    windows = _raw_windows("s-4", _messages(3))
    clash = replace(windows[0], id="other", text=windows[0].text + " extra")
    with pytest.raises(BuildRefusal, match="two different windows"):
        rebuild_sessions([*windows, clash])


def test_every_view_maps_to_the_raw_window_that_contains_it() -> None:
    windows = _raw_windows("s-5", _messages(6)) + _raw_windows("s-6", _messages(2))
    text_of = {chunk.id: chunk.text for chunk in windows}
    sessions = rebuild_sessions(windows)
    for strategy in ("sentence", "micro"):
        texts, rows = plan_views(sessions, strategy)
        assert len(texts) == len(rows) > 0
        assert {row["chunk_id"] for row in rows} == set(text_of)
        for text, row in zip(texts, rows, strict=True):
            assert text in text_of[str(row["chunk_id"])]


def test_builder_binds_to_the_corpus_identity_the_service_serves() -> None:
    import asyncio

    from recall_aml.retrieval import HostedRetriever
    from recall_aml.service import HostedService
    from scripts.build_aml_atomic_rescue_artifact import served_corpus_fingerprint

    # Complete statuses, shaped like describe_corpus output: the service's merge reads every
    # count with _status_int and silently keeps the raw identity if any is missing.
    counts = {
        "chunk_count": 3,
        "compiled_chunk_count": 0,
        "source_session_count": 1,
        "authored_relation_count": 0,
        "eligible_relation_count": 0,
        "store_relation_count": 0,
    }
    base = {"generation_id": "g", "corpus_sha256": "a" * 64, "raw_corpus_sha256": "b" * 64, **counts}
    graph = {
        "generation_id": "g",
        "corpus_sha256": "c" * 64,
        "compiled_corpus_sha256": "d" * 64,
        "compiled_kind_counts": {},
        "compiler_profile_counts": {},
        **counts,
    }

    class _Repository:
        def corpus_status(self, tenant: str) -> dict[str, object]:
            return dict(base)

        def graph_corpus_status(self, tenant: str) -> dict[str, object]:
            return dict(graph)

    class _Embedder:
        dim = 4
        name = "fake"

    service = HostedService(
        _Repository(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        HostedRetriever(_Embedder(), object()),  # type: ignore[arg-type]
        behavior=C8,
        multimodal_embedder=object(),  # type: ignore[arg-type]  # never called by _corpus_status
        specialist_retrievers={
            C8.context_embedding_profile: HostedRetriever(_Embedder(), object())  # type: ignore[arg-type]
        },
    )
    status = asyncio.run(service._corpus_status("aml_tenant"))
    assert status["graph_status"] == "ready"
    served = status["corpus_sha256"]
    assert served_corpus_fingerprint(base, graph) == served
    assert served != base["corpus_sha256"]
    assert served_corpus_fingerprint(base, None) == base["corpus_sha256"]
