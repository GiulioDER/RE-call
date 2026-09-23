"""Offline invariants of round 3 of the C8 atomizer study.

Red proofs (2026-09-23, each mutation applied to ``scripts/c8_atomizer_round3.py`` and reverted):

* ``test_fresh_spans_come_only_from_dev_sessions`` failed when the ``split_for(session) != "dev"``
  filter was removed.
* ``test_fresh_spans_never_overlap_a_used_span`` failed when ``taken`` started empty instead of
  from the used spans.
* ``test_the_writer_is_not_the_atom_model_and_is_strict`` failed when ``WRITER_MODEL`` was set to
  the gpt-4o-mini snapshot.
* ``test_a_refused_view_gate_leaves_the_c8_ranking_identical_to_off`` failed through its
  admitted-case assertion when ``view_gated_replay`` fused the unmodified dense list.

Added for M1 (2026-09-23, same method):

* ``test_m1_spans_come_from_every_session_and_stay_disjoint`` failed with
  ``assert {'dev'} == {'confirm', 'dev'}`` when ``fresh_spans`` ignored ``dev_only`` and kept the
  dev-half filter.
* ``test_fused_placement_keeps_the_off_top_five_where_dense_placement_breaks_it`` failed on the
  top-five equality when ``fused_replay`` returned ``ref.replay`` with the artifact, i.e. the dense
  placement before fusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from recall.atomic_rescue import (  # noqa: E402
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    write_atomic_rescue_artifact,
)
from scripts import c8_atomizer_reference as ref  # noqa: E402
from scripts import c8_atomizer_round3 as round3  # noqa: E402


def _rendered(count: int = 24, words: int = 300) -> dict[str, str]:
    return {
        f"sessions/t{index}/s.jsonl": " ".join(f"s{index}w{position}" for position in range(words))
        for index in range(count)
    }


def test_fresh_spans_come_only_from_dev_sessions() -> None:
    rendered = _rendered()
    assert {ref.split_for(session) for session in rendered} == {"dev", "confirm"}
    spans = round3.fresh_spans(rendered, {})
    assert spans
    assert {ref.split_for(span.session) for span in spans} == {"dev"}
    assert spans == round3.fresh_spans(rendered, {})


def test_fresh_spans_never_overlap_a_used_span() -> None:
    rendered = _rendered()
    dev = sorted(session for session in rendered if ref.split_for(session) == "dev")
    # Leave exactly one 60-word gap free in each dev session: every span must land inside it.
    used = {session: [(0, 120), (180, 300)] for session in dev}
    spans = round3.fresh_spans(rendered, used)
    assert spans
    for span in spans:
        assert 120 <= span.start and span.end <= 180


def test_the_writer_is_not_the_atom_model_and_is_strict(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return {
            "model": payload["model"],
            "provider": "X",
            "usage": {"cost": 0.0001},
            "choices": [{"message": {"content": json.dumps({"answerable": True, "question": "which port did the cleanup service bind to?"})}}],
        }

    summary = round3.generate_fresh_probes(_rendered(6), {}, tmp_path / "p.jsonl", call=call)
    assert summary["kept"] > 0
    assert seen[0]["model"] != ref.MODEL
    assert not seen[0]["model"].startswith("openai/")
    assert seen[0].get("response_format", {}).get("json_schema", {}).get("strict") is True
    rows = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert {row["split"] for row in rows} == {"dev3"}


def _unit(values: list[float]) -> Any:
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_a_refused_view_gate_leaves_the_c8_ranking_identical_to_off(tmp_path: Path) -> None:
    clear_atomic_rescue_artifact_cache()
    windows = ref.build_windows(_rendered(8, 100))
    count = len(windows)
    dim = count + 1
    matrix = np.stack([_unit([1.0 if c == r else 0.0 for c in range(dim)]) for r in range(count)])
    query = _unit([1.0 - 0.05 * c if c < count else 0.0 for c in range(dim)])
    protected, target = windows[4], windows[-1]

    def artifact(name: str, protected_score: float, target_score: float) -> Any:
        def view(score: float) -> Any:
            vector = np.zeros(dim, dtype=np.float32)
            vector[0] = score
            vector[count] = (1 - score * score) ** 0.5
            return vector

        manifest = write_atomic_rescue_artifact(
            tmp_path / name,
            matrix=np.stack([view(protected_score), view(target_score)]),
            views=[
                {"chunk_id": protected.chunk.id, "source": protected.session, "parent_ordinal": 0, "view_ordinal": 0},
                {"chunk_id": target.chunk.id, "source": target.session, "parent_ordinal": 0, "view_ordinal": 0},
            ],
            generation_id="g",
            calibration_id="c",
            pipeline_fingerprint="p",
            corpus_fingerprint="0" * 64,
            embedding_profile=ref.EMBEDDING_PROFILE,
            embedding_fingerprint="f",
            ordinary_chunk_count=count,
            source_commit="test",
        )
        return load_atomic_rescue_artifact(manifest)

    off = ref.replay(ref.Query("q", "probe", "", frozenset(), frozenset()), query, matrix, windows, [], None)
    refused = round3.view_gated_replay(query, matrix, windows, [], artifact("refuse", 0.8, 0.5))
    assert refused.rescued is None and refused.fallback is False
    assert refused.ranked == off.ranked
    admitted = round3.view_gated_replay(query, matrix, windows, [], artifact("admit", 0.5, 0.8))
    assert admitted.rescued == target.chunk.id
    assert admitted.ranked.index(target.chunk.id) == 5


def test_m1_spans_come_from_every_session_and_stay_disjoint() -> None:
    rendered = _rendered()
    spans = round3.fresh_spans(rendered, {}, seed=round3.M1_SEED, dev_only=False)
    assert {ref.split_for(span.session) for span in spans} == {"dev", "confirm"}
    used = {session: [(0, 120), (180, 300)] for session in rendered}
    for span in round3.fresh_spans(rendered, used, seed=round3.M1_SEED, dev_only=False):
        assert 120 <= span.start and span.end <= 180


def test_fused_placement_keeps_the_off_top_five_where_dense_placement_breaks_it(
    tmp_path: Path,
) -> None:
    clear_atomic_rescue_artifact_cache()
    windows = ref.build_windows(_rendered(110, 100))
    count = len(windows)
    assert count > ref.CANDIDATE_K
    matrix = np.stack([_unit([1.0 if c == r else 0.0 for c in range(count)]) for r in range(count)])
    query = _unit([1.0 - 0.005 * c for c in range(count)])
    target = windows[-1]
    manifest = write_atomic_rescue_artifact(
        tmp_path / "fused",
        matrix=np.stack([query.astype(np.float32)]),
        views=[{"chunk_id": target.chunk.id, "source": target.session, "parent_ordinal": 0, "view_ordinal": 0}],
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
    # Deep dense windows lead the lexical list; the target sits twentieth, so without a rescue it
    # stays out of the fused top five, and a dense-rank-six vote is enough to lift it into it.
    lexical = [ref.ScoredChunk(windows[index].chunk, 1.0) for index in range(80, 99)]
    lexical.append(ref.ScoredChunk(target.chunk, 1.0))
    q = ref.Query("q", "probe", "", frozenset(), frozenset())
    off = ref.replay(q, query, matrix, windows, lexical, None)
    dense = ref.replay(q, query, matrix, windows, lexical, artifact)
    fused = round3.fused_replay(query, matrix, windows, lexical, artifact)
    assert target.chunk.id not in off.ranked[:5]
    # The fixture discriminates: the dense placement changes the top five.
    assert target.chunk.id in dense.ranked[:5]
    assert fused.ranked[:5] == off.ranked[:5]
    assert fused.ranked[5] == target.chunk.id
    assert fused.rescued == target.chunk.id and fused.fallback is False
    assert len(fused.ranked) == len(set(fused.ranked))
