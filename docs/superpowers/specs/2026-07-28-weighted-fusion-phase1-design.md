# Weighted fusion (Phase 1) — preregistration

**Date:** 2026-07-28 · **Status:** preregistered, not yet implemented
**Predecessor:** [`2026-07-28-weighted-fusion-prf-phase0-design.md`](2026-07-28-weighted-fusion-prf-phase0-design.md)
— Phase 0 killed the PRF trigger and left this lever untouched.

## Why

`_rrf` scores `dense[r]` and `sparse[r]` identically. §9a measured the consequence directly:

| pool per leg | hit@5 |
|---|---|
| 20 | **0.671** |
| 100 | **0.596** |

**A 5× deeper pool makes retrieval worse.** The diagnosed mechanism is that equal leg weights
interleave five times as many low-rank candidates into every prefix. That is not a hypothesis to
establish — it is measured, and it is the single clearest defect in the retrieval path.

Phase 0 sized the addressable target on the shipped configuration: **349 of 1,536 questions
(22.7%)** have the gold chunk inside the fused candidate pool but ranked below k=5. That is what
ranking can fix, as distinct from the 162 (10.5%) that were never retrieved at all.

Supporting precedent from a corpus where the comparison had power: on **FinanceBench** (n=150,
72,151 chunks, `voyage-finance-2`, `hnsw.ef_search` corrected) growing the pool 40 → 100 moved
dense-only + reranker **0.393 → 0.507, p<0.001**. A deeper pool pays when fusion is not fighting
it. The claim here is that RRF's symmetry is what turns the same move negative on LOCOMO.

## The change

For leg `L` with per-query decisiveness `conf(L)` (`recall.eval.legconf.leg_confidence`, the
affine-invariant z-score of the leg's top candidate within its own candidates, already built and
property-tested in Phase 0):

```
w_L = clip(conf(L), 0) / Σ_L clip(conf(L), 0)          # weights sum to 1
score(d) = Σ_L  w_L · 1/(k + rank_L(d))                 # weighted RRF
```

Fallback: if every leg has `conf = 0` (no spread anywhere), weights are uniform — i.e. exactly
today's behaviour.

**Zero fitted constants.** No temperature, no per-corpus α, no threshold. This is deliberate and is
the same discipline that made Phase 0's trigger portable: §2 established that a fitted constant does
not transfer across embedders, and every constant introduced here would have to be re-fitted per
corpus by someone with a labelled set they do not have.

**Equal confidence reduces exactly to today.** `w_dense = w_sparse = 0.5` scales every fused score
by a constant, which cannot reorder anything. So current behaviour is a special case of the new one
— asserted as a backward-compatibility test, not assumed.

`leg_confidence` moves from `recall/eval/` into `recall/` when this ships, because it becomes part
of the serving path. That is the condition its own module docstring set for the move.

## Arms

One index build, four arms, scored from one retrieval per question per arm.

| arm | fusion | pool/leg | known baseline |
|---|---|---|---|
| A | RRF (shipped) | 20 | **0.671** §9a |
| B | **WRRF** | 20 | — |
| C | RRF (shipped) | 100 | **0.596** §9a |
| D | **WRRF** | 100 | — |

Two arms have published baselines to re-derive, which doubles as the apparatus check.

## Preregistered predictions

Written before any arm runs. Each carries its reasoning so it can be scored as *right for the right
reason*.

- **D > C by ≥ 0.037** — recovering at least half the 0.075 dilution. Reasoning: the dilution is
  caused by the flat leg's low-rank tail entering the prefix at full weight; down-weighting an
  undecisive leg suppresses exactly that tail, and at pool 100 the tail is five times longer, so
  this is where weighting has the most to remove.
- **B ≈ A, |Δ| < 0.02** — near-null at pool 20. Reasoning grounded in the Phase 0 artifact, not
  intuition: median `conf_dense` 2.649 vs `conf_sparse` 2.829, so the two legs are usually close in
  decisiveness and the derived weights sit near 0.5/0.5. There is little for weighting to do when
  both legs are full and comparably peaked.
- **B and D both ≥ their RRF counterparts** — weighting should not *hurt*.
- **The most likely failure is a null**, and it ships as one.

### The risk this design is most exposed to, named in advance

Phase 0's artifact shows hit@5 against |conf gap| quartile running **0.688 / 0.711 / 0.688 /
0.583** — *large disagreement in decisiveness predicts failure.* Weighting is by construction most
aggressive exactly on those queries. So there is a live mechanism by which WRRF could **lose**: it
hands the prefix to one leg hardest on the queries where neither leg is reliable.

If B or D comes back below its RRF counterpart, that is the first place to look, and the check is
already specified: report Δ(WRRF − RRF) **stratified by |conf gap| quartile**. A method that helps
in Q1–Q3 and hurts in Q4 is a different finding from one that simply does not work, and the two
must not be reported as the same thing.

## Decision rules, fixed in advance

| gate | rule | consequence |
|---|---|---|
| **D vs C** | D ≤ C | The dilution mechanism is not what §9a diagnosed, or weighting cannot address it. **Do not ship.** Publish the null against the published 0.596. |
| **B vs A** | B < A − 0.02 | Weighting hurts the shipped configuration. **Do not ship**, regardless of what D does — pool 20 is what users run. |
| stratified | helps Q1–Q3, hurts Q4 | Do not ship as an unconditional default; report the interaction and stop. Gating it on |conf gap| would be a **new hypothesis needing its own preregistration**, not a rescue. |
| apparatus | A ≠ 0.671 ± 0.01, or C ≠ 0.596 ± 0.01, or n ≠ 1,536 exact | The run is not comparable to §9a. **Fail the run**, do not interpret it. |

## Out of scope

- Any candidate-pool default change. Pool is a *control variable* here, not a lever — mixing them
  would confound weighting with depth, which is the mistake §9a already had to retract once.
- Reranking, embedder swaps, chunking, PRF, the private-46 corpus. One variable at a time.
- Score-based (non-rank) fusion. RRF's rank basis is what makes it immune to the cosine/`ts_rank`
  scale mismatch; replacing it is a different, larger experiment.

## What gets published

Four arms with Wilson CIs and paired McNemar between A↔B and C↔D; the Δ stratified by |conf gap|
quartile; the predictions scored including the wrong ones; the retained per-question artifact; and
the reproduce command. Both directions ship — a null against §9a's published 0.596 is a result.
