from __future__ import annotations

from scripts.inventory_atomic_fact_release_confirmation import (
    build_inventory,
    development_exclusions,
)


def test_global_uniqueness_includes_excluded_development_sources(tmp_path) -> None:
    """An excluded source cannot make the same fact eligible in an untouched source.

    Red proof: the plausible mutation skipped excluded sources before populating the global
    content map. This test failed at the intended empty-inventory assertion because the duplicate
    fact in `recall/new.md` was incorrectly admitted.
    """

    root = tmp_path / "recall"
    root.mkdir()
    duplicate = (
        "The release candidate requires a certified paired quality result before serving changes."
    )
    (root / "seen.md").write_text(f"# Seen\n\n{duplicate}\n", encoding="utf-8")
    (root / "new.md").write_text(f"# New\n\n{duplicate}\n", encoding="utf-8")

    rows, _ = build_inventory([("recall", root)], {"recall/seen.md"})

    assert rows == []


def test_development_exclusions_do_not_absorb_unrelated_negative_trace_sources() -> None:
    """Only atomic-development populations contribute to the new exclusion union.

    Red proof: adding an unrelated `negative.md` source to any frozen development population made
    this test fail at the exact exclusion-set assertion. The production builder receives no trace
    payload in this function, preserving the registered boundary.
    """

    extractive = {"queries": [{"gold_sources": ["recall/extractive.md"]}]}
    pilot = {"queries": [{"gold_sources": ["recall/pilot.md"]}]}
    source_gold = [
        {"answerable": True, "relevant_files": ["recall/live.md"]},
        {"answerable": False, "relevant_files": ["recall/control.md"]},
    ]
    old_candidates = [{"source": "recall/census.md"}]

    excluded, counts = development_exclusions(
        extractive_pool=extractive,
        pilot_pool=pilot,
        source_gold=source_gold,
        old_candidates=old_candidates,
    )

    assert excluded == {
        "recall/extractive.md",
        "recall/pilot.md",
        "recall/live.md",
        "recall/census.md",
    }
    assert counts["union"] == 4
