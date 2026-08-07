from pathlib import Path

import numpy as np
import pytest

from benchmarks.mtrag.late_interaction import (
    LATE_ARMS,
    LateArm,
    _resolve_arm,
    arm_record,
    assert_complete,
    holm_family,
    load_pairs_inverted,
    score_stream,
    validate_sample,
)
from recall.rerank import LateInteractionReranker


def _by_name(name: str) -> LateArm:
    return next(a for a in LATE_ARMS if a.name == name)


def test_arms_are_frozen_before_any_score():
    """Arms are declared in code, as SPARSE_ARMS is, so they cannot be edited after seeing a
    number without it showing up in the diff."""
    assert isinstance(LATE_ARMS, tuple)
    assert [a.name for a in LATE_ARMS] == ["li_colbertv2", "li_answerai", "li_jina"]


def test_permissive_arms_are_deployable():
    assert _by_name("li_colbertv2").deployable is True
    assert _by_name("li_answerai").deployable is True


def test_jina_is_not_deployable():
    arm = _by_name("li_jina")
    assert arm.licence == "cc-by-nc-4.0"
    assert arm.deployable is False


def test_holm_family_accepts_deployable_arms():
    assert holm_family([_by_name("li_colbertv2"), _by_name("li_answerai")]) == (
        "li_colbertv2",
        "li_answerai",
    )


def test_holm_family_refuses_a_non_deployable_arm():
    """THE containment gate. The verdict that gates the follow-on project is computed from a
    family li_jina cannot mechanically enter. A refusal, not a docstring."""
    with pytest.raises(ValueError, match="li_jina"):
        holm_family([_by_name("li_colbertv2"), _by_name("li_jina")])


def test_holm_family_refuses_even_a_lone_non_deployable_arm():
    with pytest.raises(ValueError, match="li_jina"):
        holm_family([_by_name("li_jina")])


def test_arm_record_carries_the_taint():
    """Numbers get lifted out of these archives into later documents. A lifted number must arrive
    with its licence attached rather than as a bare float."""
    assert arm_record(_by_name("li_jina")) == {
        "arm": "li_jina",
        "checkpoint": "jinaai/jina-colbert-v2",
        "licence": "cc-by-nc-4.0",
        "deployable": False,
    }


def test_every_arm_checkpoint_is_registered():
    from recall.rerank import LATE_INTERACTION_MODELS

    for arm in LATE_ARMS:
        assert arm.checkpoint in LATE_INTERACTION_MODELS


def test_arm_with_an_unregistered_checkpoint_raises_on_licence():
    """LATE_ARMS is frozen, so this branch is unreachable today. It is tested because the failure
    it prevents is a future arm added without a matching registry entry, which would otherwise
    reach `deployable` and be answered from a licence that does not exist."""
    with pytest.raises(ValueError, match="unregistered checkpoint"):
        LateArm("li_future", "some/unrecorded").licence


def test_holm_family_returns_every_name_when_handed_a_single_use_iterator():
    """Reading the argument twice would exhaust the iterator on the `blocked` scan, and the return
    would then be an empty tuple: silent omission of arms the caller believed were included.

    Every arm here is DEPLOYABLE, and that is the whole point. With a blocked arm in the iterator
    the function raises during the first read and never reaches the second, so that version of
    this test passes against the unmaterialised implementation and proves nothing. Verified: the
    unfixed code raises on a mixed iterator and returns () on this one.
    """
    assert holm_family(iter([_by_name("li_colbertv2"), _by_name("li_answerai")])) == (
        "li_colbertv2",
        "li_answerai",
    )


class _FakeEncoder:
    def __init__(self, table):
        self._table = table
        self.passage_batches: list[list[str]] = []

    def query_embed(self, texts):
        return [np.array(self._table[t]) for t in texts]

    def passage_embed(self, texts):
        texts = list(texts)
        self.passage_batches.append(texts)
        return [np.array(self._table[t]) for t in texts]


def test_score_stream_emits_only_requested_pairs():
    table = {"qa": [[1.0, 0.0]], "qb": [[0.0, 1.0]], "d1": [[1.0, 0.0]], "d2": [[0.0, 1.0]]}
    rows = list(
        score_stream(
            _FakeEncoder(table),
            queries={"qa": "qa", "qb": "qb"},
            docs=[("d1", "d1"), ("d2", "d2")],
            pairs={"d1": {"qa"}, "d2": {"qa", "qb"}},
        )
    )
    assert {(r["qid"], r["doc_id"]) for r in rows} == {("qa", "d1"), ("qa", "d2"), ("qb", "d2")}


def test_score_stream_computes_maxsim():
    table = {"qa": [[1.0, 0.0]], "d1": [[1.0, 0.0]]}
    rows = list(
        score_stream(
            _FakeEncoder(table),
            queries={"qa": "qa"},
            docs=[("d1", "d1")],
            pairs={"d1": {"qa"}},
        )
    )
    assert rows == [{"qid": "qa", "doc_id": "d1", "score": pytest.approx(1.0)}]


def test_score_stream_encodes_each_document_exactly_once():
    """The point of streaming. A document referenced by many queries is encoded once, not once
    per pair, which is what makes this cheaper than 241,270 cross-encoder forward passes."""
    table = {"qa": [[1.0, 0.0]], "qb": [[0.0, 1.0]], "d1": [[1.0, 0.0]]}
    encoder = _FakeEncoder(table)
    list(
        score_stream(
            encoder,
            queries={"qa": "qa", "qb": "qb"},
            docs=[("d1", "d1")],
            pairs={"d1": {"qa", "qb"}},
        )
    )
    assert [t for batch in encoder.passage_batches for t in batch] == ["d1"]


def test_score_stream_batches_documents():
    table = {"qa": [[1.0, 0.0]], **{f"d{i}": [[1.0, 0.0]] for i in range(5)}}
    encoder = _FakeEncoder(table)
    list(
        score_stream(
            encoder,
            queries={"qa": "qa"},
            docs=[(f"d{i}", f"d{i}") for i in range(5)],
            pairs={f"d{i}": {"qa"} for i in range(5)},
            batch_size=2,
        )
    )
    assert [len(b) for b in encoder.passage_batches] == [2, 2, 1]


def test_score_stream_skips_documents_with_no_pairs():
    table = {"qa": [[1.0, 0.0]], "d1": [[1.0, 0.0]], "unused": [[1.0, 0.0]]}
    encoder = _FakeEncoder(table)
    rows = list(
        score_stream(
            encoder,
            queries={"qa": "qa"},
            docs=[("d1", "d1"), ("unused", "unused")],
            pairs={"d1": {"qa"}},
        )
    )
    assert [r["doc_id"] for r in rows] == ["d1"]
    assert [t for batch in encoder.passage_batches for t in batch] == ["d1"]


def test_score_stream_refuses_an_unknown_query_id():
    """A pair naming a query the caller did not supply is a dump/scorer mismatch, and scoring it
    as anything at all would fabricate a number. G3 depends on this raising."""
    table = {"qa": [[1.0, 0.0]], "d1": [[1.0, 0.0]]}
    with pytest.raises(KeyError, match="ghost"):
        list(
            score_stream(
                _FakeEncoder(table),
                queries={"qa": "qa"},
                docs=[("d1", "d1")],
                pairs={"d1": {"ghost"}},
            )
        )


def test_score_stream_scores_an_unscoreable_document_last_instead_of_aborting():
    """MUST match `LateInteractionReranker.rerank`. If this path raised, the validate gate would
    compare a live ranking that places the document last against an offloaded run that has no
    score for it at all, and `rerank_order` refuses a candidate with no score."""
    table = {"qa": [[1.0, 0.0]], "empty": [], "d1": [[1.0, 0.0]]}
    rows = list(
        score_stream(
            _FakeEncoder(table),
            queries={"qa": "qa"},
            docs=[("empty", "empty"), ("d1", "d1")],
            pairs={"empty": {"qa"}, "d1": {"qa"}},
        )
    )
    by_doc = {r["doc_id"]: r["score"] for r in rows}
    assert by_doc["empty"] == float("-inf")
    assert by_doc["d1"] == pytest.approx(1.0)


def test_score_stream_refuses_a_query_with_no_tokens():
    """Deliberately NOT salvaged, matching `rerank`. With no query tokens nothing can be ranked."""
    table = {"qa": [], "d1": [[1.0, 0.0]]}
    with pytest.raises(ValueError, match="has no tokens"):
        list(
            score_stream(
                _FakeEncoder(table),
                queries={"qa": "qa"},
                docs=[("d1", "d1")],
                pairs={"d1": {"qa"}},
            )
        )


def test_load_pairs_inverted(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"qid": "q1", "doc_id": "d1"}\n'
        '{"qid": "q2", "doc_id": "d1"}\n'
        '{"qid": "q1", "doc_id": "d2"}\n',
        encoding="utf-8",
    )
    assert load_pairs_inverted(path) == {"d1": {"q1", "q2"}, "d2": {"q1"}}


def test_load_pairs_inverted_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"qid": "q1", "doc_id": "d1"}\n\n', encoding="utf-8")
    assert load_pairs_inverted(path) == {"d1": {"q1"}}


def test_assert_complete_passes_when_every_pair_is_scored():
    # No assert: `assert_complete` returns None and signals success by NOT raising, so the call
    # completing IS the assertion. `assert f(...) is None` would read as a test that checks
    # nothing, which is worse than no assert at all.
    assert_complete({"d1": {"q1"}}, {"d1": {"q1"}})


def test_assert_complete_raises_on_a_missing_score():
    """G3. A missing score does NOT raise on its own, it sinks the document to the bottom of the
    ranking. That is the `ef_search` failure shape: `_query_learned_sparse` returned 6 of 100 and
    no test caught it, a timing anomaly did. So counts are asserted, never assumed."""
    with pytest.raises(ValueError, match="1 pair"):
        assert_complete({"d1": {"q1", "q2"}}, {"d1": {"q1"}})


def test_assert_complete_raises_on_a_wholly_unscored_document():
    with pytest.raises(ValueError, match="1 pair"):
        assert_complete({"d1": {"q1"}, "d2": {"q1"}}, {"d1": {"q1"}})


def _live_reranker(table):
    return LateInteractionReranker(_FakeEncoder(table), model_name="colbert-ir/colbertv2.0")


# Multi-token documents are load-bearing in the fixture below. With one token per document, `mean`
# and `max` coincide, the G5 mutation becomes invisible, and the test that proves the gate can fail
# would itself fail. `multi` carries two tokens precisely so the mutation flips the ORDER, which is
# what `validate_sample` compares. Verified numerically before this plan was written.
_TABLE = {
    "q": [[1.0, 0.0]],
    "multi": [[1.0, 0.0], [0.0, 1.0]],  # maxsim 1.0, but MEAN 0.5
    "mid": [[0.6, 0.8]],                # maxsim 0.6, and mean 0.6
    "far": [[0.0, 1.0]],                # maxsim 0.0
}
_ROWS = [{"task_id": "t1", "query": "q", "candidates": ["far", "mid", "multi"]}]
_DOCS = {"far": "far", "mid": "mid", "multi": "multi"}


def test_validate_matches_when_offloaded_scores_agree():
    scores = {"t1": {"far": 0.0, "mid": 0.6, "multi": 1.0}}
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, scores)
    assert report["verdict"] == "MATCH"
    assert report["max_score_delta"] < 1e-9


def test_validate_mismatches_when_the_offloaded_order_is_wrong():
    """G2's whole purpose. An offloaded ordering that merely looks reasonable produces a
    publishable nDCG that RE-call itself would never compute."""
    scores = {"t1": {"far": 9.0, "mid": 0.6, "multi": 1.0}}
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, scores)
    assert report["verdict"] == "MISMATCH"
    assert report["failures"]


def test_the_mutation_fixture_actually_mutates():
    """Guards the guard below. If mean and max ever coincide on this fixture, the G5 test passes
    vacuously and proves nothing, which is the exact failure mode G5 exists to prevent."""
    query = np.array(_TABLE["q"])
    for doc in ("far", "mid", "multi"):
        tokens = np.array(_TABLE[doc])
        assert float((query @ tokens.T).max(axis=1).sum()) == pytest.approx(
            {"far": 0.0, "mid": 0.6, "multi": 1.0}[doc]
        )
    assert float((query @ np.array(_TABLE["multi"]).T).mean(axis=1).sum()) == pytest.approx(0.5)


def test_validate_detects_the_mean_for_max_mutation():
    """G5, as an automated test rather than a manual ritual. Scores computed with `mean` instead
    of `max` must make the gate go RED. If this passes, the gate is vacuous."""
    query = np.array(_TABLE["q"])
    mutated = {
        "t1": {
            doc: float((query @ np.array(_TABLE[doc]).T).mean(axis=1).sum())
            for doc in ("far", "mid", "multi")
        }
    }
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, mutated)
    assert report["verdict"] == "MISMATCH"


def test_validate_reports_the_worst_score_delta():
    scores = {"t1": {"far": 0.0, "mid": 0.6, "multi": 1.0004}}
    report = validate_sample(_live_reranker(_TABLE), _ROWS, _DOCS, scores)
    assert report["max_score_delta"] == pytest.approx(0.0004, abs=1e-9)


def test_resolve_arm_refuses_a_noncommercial_arm_without_the_optin():
    """The containment gate must hold on EVERY entry point. `validate` used to grant itself the
    waiver by passing `accept_noncommercial_license=not arm.deployable`."""
    with pytest.raises(SystemExit, match="accept-noncommercial"):
        _resolve_arm("li_jina", False)


def test_resolve_arm_allows_a_noncommercial_arm_with_the_optin():
    assert _resolve_arm("li_jina", True).name == "li_jina"


def test_resolve_arm_allows_a_deployable_arm_without_the_optin():
    assert _resolve_arm("li_colbertv2", False).name == "li_colbertv2"


def test_resolve_arm_refuses_an_unknown_arm():
    with pytest.raises(SystemExit, match="unknown arm"):
        _resolve_arm("li_nonexistent", False)
