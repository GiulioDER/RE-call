"""The threshold probe's baseline must stay the floor that actually ships.

The probe's docstring calls 0.50 "the shipped absolute floor" and every candidate rule is scored
against it. If the library default moves and the probe's grid does not, the probe goes on
reporting a stale number under a label saying it measures what ships.

Pinned by a TEST rather than by importing the constant. `ABSOLUTE_FLOORS` is a candidate grid and
results are keyed `f"{rule}@{param}"`, so if the shipped default ever became 0.40 an imported
constant would silently collapse two grid points onto one key and drop the baseline row with no
error. A test can report that they drifted apart; a shared reference can only hide it.
"""
from __future__ import annotations

from benchmarks.beam.threshold_probe import ABSOLUTE_FLOORS
from recall.guards import DEFAULT_GAP_THRESHOLD


def test_the_baseline_floor_is_the_one_the_library_ships() -> None:
    assert ABSOLUTE_FLOORS[0] == DEFAULT_GAP_THRESHOLD, (
        f"threshold_probe's baseline is {ABSOLUTE_FLOORS[0]} but recall.guards ships "
        f"{DEFAULT_GAP_THRESHOLD}. Update ABSOLUTE_FLOORS[0] — and re-run the probe, because "
        f"every candidate rule in results/FINDINGS.md §9m was scored against the old baseline."
    )


def test_the_candidate_grid_has_no_duplicate_points() -> None:
    # Results are keyed by parameter value, so a duplicate silently overwrites a row.
    assert len(set(ABSOLUTE_FLOORS)) == len(ABSOLUTE_FLOORS), (
        f"duplicate floors in {ABSOLUTE_FLOORS}: two grid points would share a result key and "
        f"one would vanish from the report without erroring."
    )
