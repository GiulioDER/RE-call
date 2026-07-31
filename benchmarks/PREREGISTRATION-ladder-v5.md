# Pre-registration — Answerability Ladder v5: the paid embedder

**Date:** 2026-07-31 · **Status:** written and committed BEFORE the arm runs
**Manifest:** `results/ladder/manifest_v2.jsonl`, digest
`5534c61356acaa7b62ac5a79dbec7383674fc052984d10c1d0cc89e26a532bd5` — the SAME frozen manifest as
v2, v3 and v4, verified at run start via `--expected-digest`
**Analysis:** `benchmarks/ladder/analyze_v3.py`, unchanged and already committed, for the fourth
consecutive arm. Reusing it verbatim is the point: an analysis edited for this arm would be a free
parameter, and this is the arm most likely to tempt one.

## 1. What this arm changes

`--embedder voyage:voyage-4-large`, RE-call's published best embedder
([[reference-recall-best-configuration-2026-07-28]]) and the last untested component of its best
configuration. Everything else is the shipped default — no reranker, `k=5`, `candidate_k=20` — so
this is a single-variable comparison against the **v2 default arm**, exactly as v3 (`gte-base`) was.

This is **paid work**, which a standing decision closed on 2026-07-29
([[feedback-no-paid-api-work-on-recall-2026-07-29]]). That decision is overridden here explicitly by
the user, on a measured basis rather than a vibe: the run embeds **5 882 unique documents,
~234 000 tokens**, once, because the embedding cache is content-addressed. That is cents, not the
credit-blocked lanes the rule was written about.

**Measured before choosing the manifest**: the standard subset would have cost the *same* API tokens
(same 5 882 unique documents — 40 questions still span every LOCOMO conversation) while being unable
to answer the discrimination question at all (subset near-rung noise ±0.04 against a between-system
spread of 0.016). The subset saves wall clock, not money, so the full manifest is the instrument.

## 2. Prior work

`docs_search(source_type="memory", query="voyage-4-large embedder abstention floor 0.50 starves
percentage cosine distribution")`, run 2026-07-31 before this file was committed, plus
`results/FINDINGS.md` §2. `gap_warning: false`, top-3 cosine 0.635 / 0.613 / 0.607. Three hits are
load-bearing and all point the same way:

- [[project-recall-beam-bestconfig-blocked-2026-07-28]] — the 23.3 % figure, stated exactly: the
  shipped 0.50 floor starves **14 of 60** on `voyage-4-large`. ⚠️ **Measured on BEAM, n=60**, not on
  LOCOMO and not on this manifest. That is why P2's bound is loose: the direction is well
  evidenced, the magnitude is from a different corpus at 5 % of this arm's n. The same memo records
  the calibrated replacement at 0.300, and notes `from_samples` proposed 0.547 which would have
  starved 42.6 % — a reminder that this constant is fragile in both directions.
- [[project-recall-threshold-embedder-fragile-2026-07-28]] — the general result: 0.50 is an
  ABSOLUTE cosine, and cosine level is a property of how a model was trained, not a quantity
  comparable across models. Inert on `bge-small`, starves 7 % on `text-embedding-3-small`.
- `FINDINGS.md` §2 — three embedders, three unrelated cosine regimes; `voyage-3` unanswerable
  cosines sit at 0.09–0.32 against answerable 0.53–0.70. The 0.50 constant lands inside voyage's
  gap by luck and below bge's entire distribution.

So the floor result is essentially known and this arm re-measures it on the frozen manifest at
n=1200. What is **not** known is whether a better embedder reads the *answerability axis* better,
which is P1.

## 3. Predictions

**P1 (KILL).** Discrimination at the near rung does not materially improve:
`|AUC_v5(answerable vs r=0.00) − 0.5674| ≤ 0.05`.
*A wider band than v4's ±0.03 because this changes the embedder, not a retrieval parameter, and v3
showed a family change moves the curve by ~0.010.* **If AUC rises above 0.62, that is a real
discovery**: it would mean the near-rung limit is an embedder property rather than the retrieval-
recall ceiling that `project-recall-nearmiss-signal-exhaustion` measured at 0.885.

**P2.** The shipped floor stops being inert and becomes over-eager:
`below_floor_rate ≥ 0.10`, against 0.00083 (bge-small), 0.0 (gte-base) and 0.00167 (tuned).
*Directional prediction from the 23.3 % figure, with a deliberately loose bound because that number
was measured on a different corpus slice.* This is the arm's most confident claim.

**P3.** The graded axis survives the embedder change: per-question monotone (non-increasing)
≥ 0.70, as in v2 (0.865), v3 (0.865) and v4 (0.775).

**P4.** The cosine scale does not overlap `bge-small`'s observed [0.4945, 0.8238].
*v3 already showed this for `gte-base` [0.7620, 0.9332]; if v5's range overlaps bge's, one of the
two runs is misconfigured and the arm should be discarded rather than interpreted.*

## 4. What a PASS of P1 would mean

That **four** configurations — two local embedder families, a tuned retrieval stack, and the best
paid embedder available — all read this axis within noise of each other. At that point the honest
conclusion is not "we have not found the right configuration" but "**this axis is insensitive to
configuration**", which is a finding about the axis and a warning to anyone using near-rung
abstention AUC to choose a memory system.

It would also close the last open lever. `project-recall-nearmiss-signal-exhaustion` named retrieval
depth as the next step; v4 swept it 12× and moved AUC 0.006. If the embedder moves it no further,
the ceiling is where that memo said it was: **46 of 200 pairs retrieve identical evidence in both
arms**, and no scoring change can separate them.

## 5. The deflating reading, registered in advance

If P1 and P2 both pass, this arm says: the best embedder money can buy here does not read
answerability better, and it *breaks* the shipped abstention default in the opposite direction from
the free ones. Both halves are unflattering to the system under test, both are published, and
neither is surprising given §2. Registering that now means a clean null cannot later be narrated as
an insight.
