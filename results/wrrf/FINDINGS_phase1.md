# Phase 1 — weighted fusion loses at the depth that ships, and the pool-100 baseline does not reproduce

**Date:** 2026-07-28 · **Verdict: do not ship weighted fusion.** Preregistered gate `B < A − 0.02` fired.
**Preregistration:** [`2026-07-28-weighted-fusion-phase1-design.md`](../../docs/superpowers/specs/2026-07-28-weighted-fusion-phase1-design.md), committed `583359f` before implementation.
**Artifacts:** `arm_{A,B,C,D}_*.json` in this directory — per-arm reports behind every figure.

## Reproduce

```bash
for spec in "A rrf 20" "B wrrf 20" "C rrf 100" "D wrrf 100"; do set -- $spec
  python -m recall.eval.locomo --data locomo10.json --dsn "$RECALL_DSN" \
    --k 5 --k-curve 1,5,10,20 --candidate-k $3 --fusion $2 \
    --table wrrf_r2_$1 --out results/wrrf/arm_$1_$2_pool$3.json
done
```

`locomo10.json` is gitignored and fetched on demand — see `recall/eval/locomo.py`'s module docstring.

## The four arms

| arm | fusion | pool/leg | hit@5 | 95% Wilson | n |
|---|---|---|---|---|---|
| **A** | rrf (shipped) | 20 | **0.6706** | [0.6467, 0.6936] | 1,536 |
| **B** | **wrrf** | 20 | **0.6445** | [0.6203, 0.6681] | 1,536 |
| **C** | rrf (shipped) | 100 | 0.6615 | [0.6374, 0.6847] | 1,536 |
| **D** | **wrrf** | 100 | 0.6491 | [0.6249, 0.6726] | 1,536 |

## Apparatus: pool 20 PASSES perfectly, pool 100 FAILS

Arm A against §9a's published pool-20 depth curve — **identical at every depth**, not merely within
tolerance:

| k | published §9a | arm A today | Δ |
|---|---|---|---|
| 1 | 0.3978 | 0.3978 | **0.0000** |
| 5 | 0.6706 | 0.6706 | **0.0000** |
| 10 | 0.7780 | 0.7780 | **0.0000** |
| 20 | 0.8548 | 0.8548 | **0.0000** |

Arm C against §9a's published pool-100 curve — **systematically higher**:

| k | published §9a | arm C today | Δ |
|---|---|---|---|
| 1 | 0.3900 | 0.3939 | +0.0039 |
| 5 | **0.5957** | **0.6615** | **+0.0658** |
| 10 | 0.6901 | 0.7533 | +0.0632 |
| 20 | 0.7819 | 0.8210 | +0.0391 |

The preregistered rule was *"C ≠ 0.596 ± 0.01 → fail the run, do not interpret it."* **It fired.**

## Result 1 — weighted fusion LOSES at pool 20. This one is trustworthy.

**B − A = −0.0261.** The gate was `B < A − 0.02 → do not ship`, and it fires. Because arm A
reproduces the published curve *exactly* at four depths, this is a clean measurement of a real
loss, not a harness artifact. Pool 20 is the shipped default and what users run, so this alone
settles the ship decision.

The preregistered prediction was `B ≈ A, |Δ| < 0.02`, reasoned from the Phase 0 artifact: median
`conf_dense` 2.649 vs `conf_sparse` 2.829, so the legs are usually close and the weights should sit
near 0.5/0.5. **MISS** — the effect is real and negative.

The mechanism named in advance remains the prime suspect: Phase 0 measured hit@5 by |conf gap|
quartile at 0.688 / 0.711 / 0.688 / **0.583**, so large disagreement predicts failure, and weighting
leans hardest exactly there. See *Limitations* — the stratified confirmation could not be computed
as specified.

## Result 2 — the pool-100 comparison is void, and that is itself the finding

D − C = −0.0124, which would also be a loss. **It is not reported as one**, because C is not the
configuration §9a published and the preregistered gate says an incomparable arm is not interpreted.

The disagreement matters beyond this experiment. Both runs claim the **same** configuration —
LOCOMO, fastembed/`bge-small`, k=5, `candidate_k=100`, 10 conversations, no reranker, n=1,536 — and
both claim to postdate the [#84](https://github.com/GiulioDER/RE-call/pull/84) dense-scan widening.
They differ by up to **6.6 points**. One of them is wrong.

It is not simple nondeterminism: the pool-20 arm of the same session reproduces to **0.0000 at four
depths**, so the harness, embedder, index build and fusion are all reproducible here. Whatever
differs is specific to the `candidate_k=100` path. Two checks already run and *not* sufficient:

- `query_dense(k=100)` returns **100 rows** today, verified directly against arm C's own table — so
  the dense leg is not truncated now, and the `ef_search` cap that voided the *earlier* pool-100
  control is not the explanation for this one.
- The #84 fix (`7e0c1a6`, 2026-07-25) predates the artifact's commit (2026-07-26), so a
  before/after ordering does not explain it either.

**Why this is worth chasing rather than filing away:** §9a's headline claim that *"a 5× deeper pool
measurably dilutes a fused prefix"* — the 0.671 → 0.596 drop, −0.075 — rests entirely on the
disputed number. Measured today the same contrast is **C − A = −0.0091**, roughly **one eighth** of
the published effect. If today's arm C is right, that claim is substantially overstated, and the
motivating premise of this very phase was too. This is the second time a pool-depth control on this
benchmark has had to be questioned; the first ended in a retraction (§9a's own withdrawn k=50 row).

## Prediction scorecard

All preregistered, quoted from `583359f`.

| prediction | outcome | verdict |
|---|---|---|
| `B ≈ A`, \|Δ\| < 0.02 | **−0.0261** | ❌ MISS |
| `D − C ≥ +0.037` (recover half the dilution) | −0.0124, and **void** — apparatus failed | ❌ MISS |
| B and D both ≥ their RRF counterparts | both below | ❌ MISS |
| "The most likely failure is a null, and it ships as one" | not a null — a measured **loss** | ❌ MISS |

Four for four, in the losing direction. The design's stated expectation that this would most likely
do nothing was itself too optimistic.

## Limitations

- **The stratified Δ by |conf gap| quartile — specified in the preregistration — could not be
  computed.** The LOCOMO harness records per-question *hits*, not per-question leg confidences;
  only the Phase 0 `legdiag` artifact carries those. They could be joined on question text, but
  they come from different index builds, so the join would attach approximate confidences to exact
  outcomes. Rather than present that as the preregistered analysis, it is recorded as **not done**.
  Confirming the |conf gap| mechanism needs a run that records both, which is a new measurement.
- **Abstention is invariant across all four arms by construction, not by finding.** `trusted_search`
  — the category-5 path — takes no `fusion` selector, so every arm's abstention column is identical
  because the code cannot differ, not because weighting has no effect on it. The harness emits the
  number; it is not evidence and is not reported above.
- Pool is a control variable here, not a lever. Nothing in this run licenses a change to
  `DEFAULT_CANDIDATE_K`.

## What this closes, and what it opens

**Closed: weighted fusion by per-query leg decisiveness.** It costs 2.6 points at the shipped depth.
A different weighting scheme is a new hypothesis needing its own preregistration — and it inherits a
weaker motive than this one had, because the dilution it would target now measures −0.009 rather
than −0.075.

**Opened: §9a's pool-100 row needs re-measurement.** Not by this branch. It is a published number
that a clean-apparatus run contradicts by 6.6 points, on a benchmark where the pool-depth control
has already required one retraction.
