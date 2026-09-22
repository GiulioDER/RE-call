"""Offline invariants of the C8 atomizer reference harness.

Red proofs (2026-09-22, each mutation applied to ``scripts/c8_atomizer_reference.py`` and reverted):

* ``test_every_sampled_span_has_a_containing_gold_window`` failed when ``gold_segments`` used
  ``end < window_end`` instead of ``end <= window_end``.
* ``test_fusion_matches_the_c8_rrf_order`` failed when ``fuse`` sorted by ``+fused[item]``.
* ``test_rescue_lands_at_dense_rank_six_and_is_scored_against_the_off_control`` failed when
  ``summarise`` counted a gain for a query the control had already answered (dropped the
  ``base > cutoff`` clause).
* ``test_probe_requests_pin_the_snapshot_and_strict_schema`` failed when
  ``PROVIDER_ROUTING`` allowed fallbacks.
* ``test_probe_generation_stops_at_the_budget`` failed when the budget check was removed.
* ``test_a_question_copying_the_span_is_rejected`` failed when ``MAX_COPIED_RUN`` was raised to 50.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from recall.atomic_rescue import clear_atomic_rescue_artifact_cache, load_atomic_rescue_artifact, write_atomic_rescue_artifact  # noqa: E402
from recall.types import ScoredChunk  # noqa: E402
from scripts import c8_atomizer_reference as ref  # noqa: E402


def _rendered(sessions: int = 6, words: int = 400) -> dict[str, str]:
    return {
        f"sessions/t{index}/s.jsonl": " ".join(
            f"s{index}w{position}{'.' if position % 11 == 10 else ''}" for position in range(words)
        )
        for index in range(sessions)
    }


def test_every_sampled_span_has_a_containing_gold_window() -> None:
    rendered = _rendered()
    spans = ref.sample_spans(rendered)
    assert len(spans) == 2 * len(rendered)
    assert spans == ref.sample_spans(rendered)
    for span in spans:
        assert span.gold, span
        words = rendered[span.session].split()
        assert span.text == " ".join(words[span.start : span.end])
        windows = {
            window.segment: window
            for window in ref.build_windows({span.session: rendered[span.session]})
        }
        for segment in span.gold:
            assert span.text in windows[segment].chunk.text
    # A span ending exactly on a window's last word is contained by that window.
    assert 0 in ref.gold_segments(400, 150, 160)


def test_windows_match_the_hosted_word_windows() -> None:
    rendered = _rendered(sessions=1, words=400)
    windows = ref.build_windows(rendered)
    assert [window.segment for window in windows] == [0, 1, 2]
    assert all(window.chunk.metadata["source_session_id"] for window in windows)
    assert len({window.chunk.id for window in windows}) == 3


def _hits(windows: list[ref.Window], order: list[int]) -> list[ScoredChunk]:
    return [ScoredChunk(windows[index].chunk, 1.0 / (rank + 1)) for rank, index in enumerate(order)]


def test_fusion_matches_the_c8_rrf_order() -> None:
    windows = ref.build_windows(_rendered(sessions=2, words=300))
    dense = _hits(windows, [0, 1, 2])
    lexical = _hits(windows, [2, 3])
    fused = ref.fuse(dense, lexical)
    # Window 2 is in both lists and must lead; 0 (dense rank 1) beats 3 (lexical rank 2).
    assert fused[0] == windows[2].chunk.id
    assert fused.index(windows[0].chunk.id) < fused.index(windows[3].chunk.id)


def _unit(vector: list[float]) -> Any:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_rescue_lands_at_dense_rank_six_and_is_scored_against_the_off_control(tmp_path: Path) -> None:
    clear_atomic_rescue_artifact_cache()
    rendered = _rendered(sessions=4, words=300)
    windows = ref.build_windows(rendered)
    count = len(windows)
    dimension = count + 1
    # Window i points along axis i; the query points mostly along axis 0 with a small tail so the
    # dense order is 0, 1, 2, ... and the gold window (the last) sits far below the top five.
    matrix = np.stack([_unit([1.0 if column == row else 0.0 for column in range(dimension)]) for row in range(count)])
    query = _unit([1.0 - 0.01 * column if column < count - 1 else 0.001 for column in range(dimension)])
    gold = windows[-1]
    view_vec = np.zeros(dimension, dtype=np.float32)
    view_vec[count - 1] = 1.0
    view_vec[0] = 0.9
    view_vec = view_vec / np.linalg.norm(view_vec)
    manifest = write_atomic_rescue_artifact(
        tmp_path / "artifact",
        matrix=np.stack([view_vec]),
        views=[{"chunk_id": gold.chunk.id, "source": gold.session, "parent_ordinal": gold.segment, "view_ordinal": 0}],
        generation_id="g",
        calibration_id="c",
        pipeline_fingerprint="p",
        corpus_fingerprint="0" * 64,
        embedding_profile=ref.EMBEDDING_PROFILE,
        embedding_fingerprint="f",
        ordinary_chunk_count=count,
        source_commit="test",
    )
    artifact = load_atomic_rescue_artifact(manifest)
    q = ref.Query("q1", "probe", "unused", frozenset({gold.chunk.id}), frozenset({gold.session}))
    already = ref.Query("q2", "probe", "unused", frozenset({windows[0].chunk.id}), frozenset({windows[0].session}))
    off = [ref.replay(item, query, matrix, windows, [], None) for item in (q, already)]
    on = [ref.replay(item, query, matrix, windows, [], artifact) for item in (q, already)]
    assert on[0].rescued == gold.chunk.id and on[0].ranked[5] == gold.chunk.id
    assert on[0].ranked[:5] == off[0].ranked[:5]
    assert off[0].ranked.index(gold.chunk.id) > 5
    report = ref.summarise([q, already], {"off": off, "sentence": on}, {w.chunk.id: w.session for w in windows})
    arm = report["sentence"]
    assert arm["exact_gain@6"] == 1
    assert arm["exact_loss@6"] == 0
    assert arm["exact@1"] == report["off"]["exact@1"] == 1
    assert arm["attempted"] == arm["active"] == arm["candidate_available"] == 2
    assert arm["fallback"] == 0
    # The second query was already answered at rank 1; its rescue is a different window.
    assert arm["rescued_is_exact_gold"] == 1


def _fake_response(answerable: bool, question: str) -> dict[str, Any]:
    return {
        "model": ref.MODEL,
        "provider": "OpenAI",
        "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{"message": {"content": json.dumps({"answerable": answerable, "question": question})}}],
    }


def test_probe_requests_pin_the_snapshot_and_strict_schema(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return _fake_response(True, "which directory did the cleanup script skip on purpose?")

    summary = ref.generate_probes(_rendered(sessions=2), tmp_path / "p.jsonl", call=call)
    assert summary["kept"] == 4
    payload = seen[0]
    assert payload["model"] == "openai/gpt-4o-mini-2024-07-18"
    assert payload["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert payload["temperature"] == 0
    assert payload["response_format"]["json_schema"]["strict"] is True
    rows = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert {row["split"] for row in rows} <= {"dev", "confirm"}
    assert all(row["gold_segments"] for row in rows)


def test_probe_generation_stops_at_the_budget(tmp_path: Path) -> None:
    calls = 0

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _fake_response(True, "what did the retry loop finally change in the parser?")

    with pytest.raises(RuntimeError, match="budget exhausted"):
        ref.generate_probes(_rendered(sessions=6), tmp_path / "p.jsonl", call=call, budget_usd=0.0025)
    assert calls == 3


def test_a_question_copying_the_span_is_rejected() -> None:
    span = "the migration ledger records the filename next to the version number always"
    assert ref.probe_rejection(True, "why does the migration ledger records the filename next?", span) == "copied_span"
    assert ref.probe_rejection(True, "what does the schema ledger store beside each version?", span) is None
    assert ref.probe_rejection(False, "anything at all here?", span) == "not_answerable"
    assert ref.probe_rejection(True, "why?", span) == "question_length"
