from __future__ import annotations

import json
from pathlib import Path

from recall.types import Chunk, ScoredChunk
from recall_mcp import service
from scripts.run_live_retrieval_leg_audit import _summarize
from scripts.run_live_tty_graph_precision import _command


def _hit(chunk_id: str, source: str, ordinal: int, score: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        Chunk(
            chunk_id,
            f"stored/{source}",
            chunk_id,
            {"file": source, "ord": ordinal},
        ),
        score,
    )


def test_retrieval_leg_audit_requires_generation_pin(monkeypatch) -> None:
    """Candidate identity is exposed only under both private benchmark gates.

    Red proof node ``retrieval-leg-gate-01`` changes the production conjunction to a disjunction.
    The flag-only assertion then returns true and fails at its intended assertion.
    """
    monkeypatch.delenv("RECALL_BENCHMARK_PIN", raising=False)
    monkeypatch.delenv("RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT", raising=False)
    assert service._retrieval_leg_benchmark_audit_enabled() is False

    monkeypatch.setenv("RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT", "1")
    assert service._retrieval_leg_benchmark_audit_enabled() is False

    monkeypatch.setenv("RECALL_BENCHMARK_PIN", "1")
    assert service._retrieval_leg_benchmark_audit_enabled() is True


def test_retrieval_leg_payload_records_both_ranked_legs() -> None:
    """The audit retains dense and lexical source identity at the registered depth.

    Red proof node ``retrieval-leg-payload-01`` replaces the sparse store result with an empty
    list. The sparse identity assertion then fails while dense retrieval still succeeds.
    """

    class Store:
        dense = [_hit("d1", "recall/dense.md", 2), _hit("shared", "recall/shared.md", 3)]
        sparse = [_hit("s1", "recall/sparse.md", 4), _hit("shared", "recall/shared.md", 3)]

        def query_dense(self, vector, *, k, source=None):
            assert vector == [0.2, 0.8]
            assert k == service.BENCHMARK_RETRIEVAL_LEG_DEPTH
            assert source is None
            return self.dense

        def query_sparse(self, query, *, k, vec, source=None):
            assert query == "question"
            assert vec == [0.2, 0.8]
            assert k == service.BENCHMARK_RETRIEVAL_LEG_DEPTH
            assert source is None
            return self.sparse

    payload = service._retrieval_leg_benchmark_audit_payload(
        Store(), "question", [0.2, 0.8], None
    )

    assert payload["depth"] == 100
    assert [(row["source"], row["rank"]) for row in payload["dense"]] == [
        ("recall/dense.md", 1),
        ("recall/shared.md", 2),
    ]
    assert [(row["source"], row["rank"]) for row in payload["sparse"]] == [
        ("recall/sparse.md", 1),
        ("recall/shared.md", 2),
    ]


def _item(chunk_id: str, source: str) -> dict[str, object]:
    return {"chunk_id": chunk_id, "source": source, "ordinal": 0, "cosine": 0.8}


def _row(
    query_index: int,
    gold: str | None,
    dense: list[dict[str, object]],
    sparse: list[dict[str, object]],
    *,
    outcome: str = "completed",
) -> dict[str, object]:
    return {
        "query_index": query_index,
        "query": {
            "id": f"q{query_index}",
            "query": f"question {query_index}",
            "answerable": gold is not None,
            "relevant_files": [gold] if gold else [],
        },
        "outcome": outcome,
        "trusted_evidence": [],
        "audit": {"depth": 100, "dense": dense, "sparse": sparse},
    }


def test_summary_classifies_each_candidate_failure_boundary() -> None:
    """Every answerable miss lands in exactly one retrieval boundary.

    Red proof node ``retrieval-leg-summary-01`` changes the fusion-loss branch to inspect only the
    dense leg. The lexical-only fixture then moves to the wrong class and this test fails.
    """
    filler_dense = [_item(f"d{i}", f"recall/d{i}.md") for i in range(20)]
    filler_sparse = [_item(f"s{i}", f"recall/s{i}.md") for i in range(20)]
    rows = [
        _row(0, "recall/reachable.md", [_item("g0", "recall/reachable.md")], []),
        _row(
            1,
            "recall/selection.md",
            [*_item_list("a", 10), _item("g1", "recall/selection.md")],
            [],
        ),
        _row(
            2,
            "recall/fusion.md",
            filler_dense,
            [*filler_sparse[:19], _item("g2", "recall/fusion.md")],
        ),
        _row(
            3,
            "recall/deep.md",
            [*_item_list("x", 20), _item("g3", "recall/deep.md")],
            [],
        ),
        _row(4, "recall/absent.md", filler_dense, filler_sparse),
        _row(5, None, [], [], outcome="abstained"),
    ]

    summary = _summarize(rows)

    assert summary["classification"] == {
        "reachable_fused_top10": [0],
        "selection_loss_fused_11_to_20": [1],
        "fusion_loss_from_leg_top20": [2],
        "deep_candidate_only_21_to_100": [3],
        "candidate_generation_miss_at_100": [4],
    }
    assert summary["source_gold_hit_rates"]["union"]["20"]["hits"] == 3
    assert summary["served_unanswerable_abstentions"] == 1


def test_unanswerable_abstention_uses_evidence_decision_not_generator_outcome() -> None:
    """A disabled answer provider cannot make a false retrieval answer count as abstention.

    Red proof node ``retrieval-leg-abstention-01`` is the previous scorer, which counts top level
    ``outcome == 'abstained'``. Both fixtures then count as abstentions and this assertion fails.
    """
    empty = _row(0, None, [], [], outcome="abstained")
    false_answer = _row(1, None, [], [], outcome="abstained")
    false_answer["trusted_evidence"] = [_item("near", "recall/near.md")]

    summary = _summarize([empty, false_answer])

    assert summary["served_unanswerable_abstentions"] == 1
    assert summary["served_unanswerable_answers"] == 1


def _item_list(prefix: str, count: int) -> list[dict[str, object]]:
    return [_item(f"{prefix}{index}", f"recall/{prefix}{index}.md") for index in range(count)]


def test_tty_command_enables_leg_audit_only_when_requested(monkeypatch) -> None:
    """The live runner selects committed code and explicitly enables the private trace.

    Red proof node ``retrieval-leg-command-01`` omits the audit environment assignment. The
    audited command assertion then fails while the ordinary command remains valid.
    """
    code_root = "/home/sentiment/recall-repos/source-gold-audit"
    monkeypatch.setenv("RECALL_BENCHMARK_REMOTE_CODE_ROOT", code_root)
    ordinary = _command(
        "memory", "voyage:voyage-4", "/srv/memory", "fast", "combined", "none", 1, 32, 0.10
    )[-1]
    audited = _command(
        "memory",
        "voyage:voyage-4",
        "/srv/memory",
        "fast",
        "combined",
        "none",
        1,
        32,
        0.10,
        "generation-one",
        benchmark_retrieval_leg_audit=True,
    )[-1]

    assert "RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT" not in ordinary
    assert "RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT=1" in audited
    assert f"cd {code_root}" in audited
    assert f"PYTHONPATH={code_root}" in audited


def test_source_gold_query_set_preserves_legacy_labels_and_live_successor() -> None:
    """The revised set keeps all old labels while replacing an obsolete source target.

    Red proof node ``source-gold-successor-01`` points memory-003 back at the superseded twelve
    minute memo. The successor assertion then fails without changing query count or answerability.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "preregistrations"
        / "2026-09-13-memory-queries-source-gold.json"
    )
    queries = json.loads(path.read_text(encoding="utf-8"))
    answerable = [query for query in queries if query["answerable"]]

    assert len(queries) == 50
    assert len(answerable) == 22
    assert all(query["relevant_ids"] for query in answerable)
    assert all(query["relevant_files"] for query in answerable)
    assert queries[2]["relevant_files"] == ["recall/full-suite-takes-31-minutes-not-12.md"]
