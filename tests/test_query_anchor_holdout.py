from __future__ import annotations

from scripts.run_live_query_anchor_holdout import aggregate, select_candidate
from scripts.validate_query_anchor_inventory import validate_inventory


def _item(source: str, text: str, chunk_id: str) -> dict[str, object]:
    return {"source": source, "text": text, "chunk_id": chunk_id}


def test_select_candidate_appends_supported_first_proposal_to_empty_base() -> None:
    proposal = _item("gold.md", "alpha beta gamma evidence", "candidate")
    pool = [proposal, _item("other.md", "unrelated material", "other")]

    selected = select_candidate("alpha beta gamma?", [], [proposal], pool)

    assert selected == [proposal]


def test_select_candidate_preserves_nonempty_base() -> None:
    base = _item("base.md", "existing evidence", "base")
    proposal = _item("gold.md", "alpha beta gamma evidence", "candidate")
    pool = [base, proposal]

    selected = select_candidate("alpha beta gamma?", [base], [base, proposal], pool)

    assert selected == [base]


def test_precision_gate_fails_below_frozen_floor() -> None:
    row = {
        "expected_answerability": "answerable",
        "base_count": 0,
        "candidate_count": 1,
        "added_count": 1,
        "base_prefix_preserved": True,
        "base_exact_span_covered": False,
        "candidate_exact_span_covered": True,
        "exact_span_gain": True,
        "exact_span_loss": False,
        "base_source_hit": False,
        "candidate_source_hit": True,
        "source_hit_gain": True,
        "source_hit_loss": False,
        "added_exact_span_items": 1,
        "added_non_gold_items": 0,
        "control_activation": False,
    }
    nonexact = {**row, "candidate_exact_span_covered": False, "exact_span_gain": False}
    nonexact["added_exact_span_items"] = 0
    nonexact["added_non_gold_items"] = 1

    result = aggregate([row, nonexact, nonexact, nonexact, nonexact])

    assert result["decision"] == "FAIL_EXACT_SPAN_PRECISION"


def test_inventory_comparison_never_needs_source_values_in_receipt() -> None:
    pool = {
        "queries": [
            {
                "expected_answerability": "answerable",
                "gold_sources": ["source.md"],
                "source_sha256": "a" * 64,
            },
            {
                "expected_answerability": "unanswerable",
                "gold_sources": [],
                "source_sha256": None,
            },
        ]
    }

    result = validate_inventory(pool, [("source.md", "a" * 64)])

    assert result == {
        "frozen_sources": 1,
        "matched_sources": 1,
        "missing_sources": 0,
        "digest_mismatches": 0,
        "decision": "HOLDOUT_LINEAGE_VALID",
    }


def test_inventory_comparison_normalizes_generation_root_namespace() -> None:
    pool = {
        "queries": [
            {
                "expected_answerability": "answerable",
                "gold_sources": ["recall/source.md"],
                "source_sha256": "a" * 64,
            }
        ]
    }

    result = validate_inventory(
        pool,
        [("/home/sentiment/recall-repos/memory/recall/source.md", "a" * 64)],
    )

    assert result["decision"] == "HOLDOUT_LINEAGE_VALID"
    assert result["matched_sources"] == 1
