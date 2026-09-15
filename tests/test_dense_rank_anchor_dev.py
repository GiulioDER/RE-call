from __future__ import annotations

from scripts.run_dense_rank_anchor_dev import choose_dense_candidates


def _item(source: str, text: str) -> dict[str, object]:
    return {"chunk_id": source, "source": source, "text": text}


def test_compatible_rule_skips_incompatible_dense_rank_one() -> None:
    incompatible = _item("first.md", "alpha unrelated material")
    compatible = _item("second.md", "alpha beta gamma evidence")
    pool = [incompatible, compatible]

    selected = choose_dense_candidates(
        "What alpha beta gamma evidence was recorded?", pool, pool
    )

    assert selected["dense_first_anchor_safe"] == incompatible
    assert selected["dense_first_anchor_compatible"] == compatible


def test_both_dense_rules_reject_a_corpus_absent_anchor() -> None:
    candidate = _item("candidate.md", "alpha beta gamma evidence")

    selected = choose_dense_candidates(
        "What alpha beta velnora evidence was recorded?", [candidate], [candidate]
    )

    assert selected == {
        "dense_first_anchor_safe": None,
        "dense_first_anchor_compatible": None,
    }
