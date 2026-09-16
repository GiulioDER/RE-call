from __future__ import annotations

from scripts.run_atomic_fact_blind_rescue import (
    Candidate,
    _paired,
    dense_preserving_rescue,
)


def _candidate(source: str, ordinal: int) -> Candidate:
    return Candidate(source, ordinal, f"{source}:{ordinal}", 1.0 / (ordinal + 1))


def test_rescue_preserves_dense_five_and_skips_atomic_duplicates() -> None:
    """Append the first atomic parent absent from dense top five.

    Red proof: before the first green run, the deliberate mutation removed the duplicate filter
    and duplicate invariant in `dense_preserving_rescue`. This node failed at the intended sixth
    identity assertion with `dense.md` instead of `atomic/new.md`.
    """

    dense = [_candidate("dense.md", ordinal) for ordinal in range(6)]
    atomic = [dense[1], dense[4], _candidate("atomic/new.md", 0)]
    rescued = dense_preserving_rescue(dense, atomic)
    assert [item.identity for item in rescued[:5]] == [item.identity for item in dense[:5]]
    assert rescued[5].identity == ("atomic/new.md", 0)


def test_paired_counts_gain_loss_and_ties() -> None:
    """Count paired rescue movement against dense rank six without changing direction.

    Red proof: before the first green run, the deliberate mutation reversed dense and rescue in
    `_paired`. This node failed at the intended assertion with one gain and two losses instead of
    two gains and one loss.
    """

    def row(dense_hit: bool, rescue_hit: bool, suffix: str) -> dict[str, object]:
        gold = {"source": f"gold-{suffix}.md", "ordinal": 0, "score": 1.0}
        miss = {"source": f"miss-{suffix}.md", "ordinal": 1, "score": 0.5}
        dense = [miss.copy() for _ in range(6)]
        rescue = [miss.copy() for _ in range(6)]
        if dense_hit:
            dense[5] = gold.copy()
        if rescue_hit:
            rescue[5] = gold.copy()
        return {
            "gold_source": gold["source"],
            "gold_ordinal": 0,
            "dense": dense,
            "dense5_atomic1": rescue,
        }

    rows = [
        row(False, True, "gain-a"),
        row(False, True, "gain-b"),
        row(True, False, "loss"),
        row(True, True, "tie"),
    ]
    assert _paired(rows, "exact") == {"gains": 2, "losses": 1, "ties": 1, "net": 1}
