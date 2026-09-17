from __future__ import annotations

from scripts.run_atomic_fact_release_confirmation import (
    one_sided_paired_probability,
    summarize,
)


def _candidate(source: str, ordinal: int) -> dict[str, object]:
    return {"source": source, "ordinal": ordinal, "score": 1.0}


def test_release_summary_requires_prospective_net_gain_and_loss_control() -> None:
    """The release decision counts paired rescues and losses rather than raw arm totals.

    Red proof: the plausible mutation classified a dense-only row as a gain. This test failed at
    the intended exact loss and net assertions, reporting seven gains and zero losses instead of
    six gains and one loss.
    """

    rows = []
    for index in range(96):
        gold_source = f"recall/gold-{index}.md"
        distractors = [_candidate(f"recall/d-{index}-{rank}.md", 0) for rank in range(6)]
        dense = list(distractors)
        rescue = list(distractors[:5])
        if index < 6:
            rescue.append(_candidate(gold_source, 2))
        elif index == 6:
            dense[5] = _candidate(gold_source, 2)
            rescue.append(_candidate(f"recall/other-{index}.md", 0))
        else:
            rescue.append(_candidate(f"recall/other-{index}.md", 0))
        rows.append(
            {
                "gold_source": gold_source,
                "gold_ordinal": 2,
                "dense": dense,
                "dense5_atomic1": rescue,
            }
        )

    result = summarize(rows)

    assert result["comparison"]["exact"]["gains"] == 6
    assert result["comparison"]["exact"]["losses"] == 1
    assert result["comparison"]["exact"]["net"] == 5
    assert not result["quality_checks"]["exact_net_gain_gte_6"]
    assert not result["quality_checks"]["exact_p_lte_0_05"]


def test_one_sided_probability_matches_frozen_exact_binomial() -> None:
    """Use the preregistered one-sided exact discordant-pair probability."""

    assert one_sided_paired_probability(6, 0) == 0.015625
    assert one_sided_paired_probability(6, 1) == 0.0625
    assert one_sided_paired_probability(0, 0) == 1.0
