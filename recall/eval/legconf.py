"""Per-query, per-leg decisiveness — the quantity the Phase 0 diagnostic measures.

`leg_confidence` moved to `recall.fusion` (Phase 1 graduated it to the serving path — see that
module's docstring for the corrected provenance). It is re-exported here so `legdiag` and its
tests are untouched.

`more_decisive` stays here: it is diagnostic-only, and it is the thing that was falsified, not
`leg_confidence` itself.

FALSIFIED 2026-07-28 -> results/legdiag/FINDINGS_phase0.md. The trigger this module supports did
not clear its Q1 gate: firing-group hit@5 (0.7081) was HIGHER than non-firing (0.6164), not lower
-- leg disagreement selects for retrieval successes, not failures. `more_decisive` does not
consume this. Read that finding before treating `more_decisive` as a live path to production.

Being under `recall/eval/` does not keep it out of the installed package, either: `pyproject.toml`
packages `recall` as a whole (`[tool.hatch.build.targets.wheel]` -> `packages = ["recall",
"recall_mcp"]`), so `recall/eval/` ships in the wheel regardless of whether anything in `recall/`
calls it.
"""
from __future__ import annotations

from collections.abc import Sequence

from recall.fusion import leg_confidence  # re-exported: Phase 1 moved it to the serving path

__all__ = ["leg_confidence", "more_decisive"]


def more_decisive(sparse_scores: Sequence[float], dense_scores: Sequence[float]) -> bool:
    """True when the sparse leg was more decisive than the dense leg on this query.

    Both legs are scored over their top `min(len(sparse), len(dense))` candidates. That
    truncation is the point: `leg_confidence` is the z-score of a sample maximum, which grows
    with sample size on its own (E[conf] on iid noise runs 1.39/1.67/1.95 at n=5/10/20). The
    dense leg always returns exactly `candidate_k` candidates while the sparse leg returns only
    tsquery matches, so comparing them at their natural depths would measure how many chunks
    matched the query text rather than which leg was more decisive.

    Returns False when the common depth is below 2 — there is no spread to compare, and the
    caller treats "no information" as "do not fire".
    """
    m = min(len(sparse_scores), len(dense_scores))
    if m < 2:
        return False
    sparse_top = sorted(sparse_scores, reverse=True)[:m]
    dense_top = sorted(dense_scores, reverse=True)[:m]
    return leg_confidence(sparse_top) > leg_confidence(dense_top)
