import numpy as np
import pytest

from benchmarks.mtrag.late_interaction import (
    LATE_ARMS,
    LateArm,
    arm_record,
    holm_family,
    score_stream,
)


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
