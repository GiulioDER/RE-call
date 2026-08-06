# Multi-query diversity + nested RRF on MTRAG-human dev: preregistration

Status: **PREREGISTERED**. Written 2026-08-06, before any score from this experiment was observed.
Handoff that commissioned it: `2026-08-06-handoff-query-diversity.md`.

Everything below the "Predicted outcome" heading is a commitment. Results go in a separate
`RESULTS` section appended after the run, and the arms, contrasts and decision rule in this file
are not edited afterwards.

## The question

Coverage is the measured MTRAG bottleneck (R@100 0.7377 raw / 0.7599 reranked against a ~0.95
saturation threshold). The reranker lever is closed: two cross-encoders 25x apart in size bury the
same SPLADE-only gold documents (90 vs 91 of 123 below rank 10), so those documents are not
rankable from a `(query, passage)` pair. The remaining lever is the query.

MTRAGEval's rank-1 system (AILS-NTUA, nDCG@5 0.5776) issues five LLM reformulations to a single
well-aligned retriever and fuses them with variance-aware nested RRF. This experiment asks whether
the **fusion of diverse queries** is what buys their gain, using three query files MTRAG-human
already ships, so the mechanism is tested with zero LLM calls and zero GPU.

## What rank 1 actually did

Read from the paper (arXiv 2603.10524 / ACL Anthology 2026.semeval-1.175), not from memory, because
two figures about this system were previously mis-imported.

Nested RRF, their Equation 2: `Score_final(d) = Σ_s w_s^(c) · 1/(k^(c) + rank_s(d))`.

- **Inner level**: the three *high-variance* strategies (HyDE, Chain-of-Thought, Anchor-Keyword)
  are pre-aggregated into one "Weak Consensus" ranking, equal weights, `k_internal = 40`.
- **Outer level**: Weak Consensus is fused with the two *stable* strategies (Minimal,
  Corpus-Specific) using corpus-dependent weights.
- "Variance-aware" names the **grouping**, not a computed statistic. The paper does not define a
  variance calculation; it separates high-variance from low-variance strategies so that no single
  volatile reformulation can dominate the fused ranking.

Their ablation (Table 4) is **cumulative**, and the decomposition matters more than the headline:

| configuration | R@5 |
|---|---|
| no rewriting | 0.483 |
| + Minimal | 0.527 |
| + Corpus-Specific | 0.558 |
| + CoT | 0.573 |
| + HyDE | 0.584 |
| + Anchor-Keyword | 0.591 |
| + nested RRF fusion | 0.607 |

So of the headline +25.7%, **adding reformulations under flat fusion is worth 0.483 → 0.591**, and
the **nested topology is worth 0.591 → 0.607** (+0.016 absolute, +2.7% relative). The dominant term
is diversity; nesting is a refinement. The arm set below is built to measure those two separately
rather than to reproduce one number.

## Mapping onto the three shipped files

`mtrag-human/retrieval_tasks/<domain>/<domain>_{lastturn,questions,rewrite}.jsonl`, aligned on
`_id`, 777 rows total (clapnq 208, cloud 188, fiqa 180, govt 201).

| variant | file | rank-1 analogue | individually measured |
|---|---|---|---|
| `last` | `_lastturn` | "no rewriting" | the baseline |
| `rewrite` | `_rewrite` | "Minimal", but GOLD | +0.0321 nDCG@5, inconclusive |
| `full` | `_questions` | no analogue; the volatile one | **−0.0972 nDCG@5** |

⚠️ The two prior figures were re-read from
`/var/lib/recall-benchmarks/2026-08-06-mtrag-retrieval-2x2/results/`, and they are **nDCG@5 at
candidate_k=20 on the pre-SPLADE lexical hybrid, with no R@100 recorded at all** (lastturn 0.28488,
rewrite_gold 0.31694, questions_concat 0.18770). They are therefore *not* comparable to this
experiment's numbers, and `mq_rewrite` / `mq_full` are re-measured here on the SPLADE configuration
so that every arm in the family shares one pipeline.

`full` being individually harmful is the point. Two of three variants are weak-or-harmful on their
own; if their fusion beats the best of them, fusion is the mechanism.

## Power: does the deciding cell have any?

Computed before freezing anything, from the shipped files.

| | queries | share |
|---|---|---|
| judged, total | 777 | 100% |
| **all three variants byte-identical** (turn 1 of every conversation) | **102** | 13.1% |
| at least two variants distinct | 675 | 86.9% |
| all three distinct | 568 | 73.1% |
| `last` == `rewrite` (byte-identical) | 209 | 26.9% |

The 102 turn-1 queries are a **structural zero**: identical query text produces identical leg
rankings, and RRF over identical rankings is order-preserving, so their per-query delta is exactly
0.0 by construction, not by measurement. They cannot contribute power. The deciding cell for
P1, M1, T1 and R1 is therefore **675 queries**.

⚠️ **B1 is the exception**, and it is easy to get wrong. It compares 33-deep legs against
100-deep ones, so its delta is driven by the truncation and is perfectly capable of being non-zero
on a query whose variants are identical. Its deciding cell is all **777**. Reporting B1 over the
675 would silently drop 102 queries that carry real signal for that contrast alone.
(`tests/test_multiquery.py` pins both the structural-zero property and this exception to it.)

(The prior 2x2 run reported 217 "unchanged" rather than 209 because it compared
`.lower().rstrip("?. ")`. The stricter byte comparison is the right one here: only byte-identical
variants are *guaranteed* to produce a zero delta.)

Minimum detectable effect, anchored on the nearest measured contrast on this exact metric and
sample (SPLADE vs lexical: +0.0331, CI [+0.0202, +0.0457], n=777 → SE 0.00651, SD 0.181):

- MDE at 80% power, two-sided α=0.05 ≈ 2.80 × SE ≈ **±0.018 R@100**.
- Diluting by the 102 structural zeros costs about 7% of effect size (mean scales by 675/777,
  SD by √(675/777)), so the effective MDE is ≈ **±0.019**.

The ship bar below is +0.020, which sits just above the MDE. That is deliberate: the study can
distinguish "clears the bar" from "does not", so the gate can both fire and fail. If the true SD of
this contrast is materially larger than the SPLADE contrast's, the CI will say so and the verdict
becomes INCONCLUSIVE rather than negative.

## Frozen arms

Frozen 2026-08-06 before any score was observed. All arms use `use_dense=True`,
`sparse_backend="splade"`, `candidate_k=100`, no reranker; the SPLADE profile is
`prithivida__Splade_PP_en_v1`. `mq_last` is configuration-identical to the archived `hybrid_splade`.

| arm | variants fused | topology | role |
|---|---|---|---|
| `mq_last` | last | single query | **control** |
| `mq_rewrite` | rewrite | single query | ceiling of single-query rewriting |
| `mq_full` | full | single query | the volatile variant, alone |
| `mq_nested3` | last, full, rewrite | nested: legs→variant, variants→final | **primary** |
| `mq_nested2` | last, rewrite | nested | robustness |
| `mq_flat6` | last, full, rewrite | flat: one RRF over all 6 leg rankings | topology probe |
| `mq_nested3_vw` | last, full, rewrite | nested, outer weights (1.0, **0.5**, 1.0) | variance-aware |
| `mq_nested3_budget33` | last, full, rewrite | nested, every leg truncated to 33 | budget control |

RRF damping constant is **k=60 at both levels, for every arm**, which is RE-call's own `_rrf`
default. Rank 1 used `k_internal=40` for their weak-consensus group; adopting it here would change
the control's fusion as well and break comparability with the archived baseline, so it is not
adopted and this is recorded as a deliberate deviation.

`mq_nested3_vw` fixes `w_full = 0.5` **a priori**, on the ground that `full` is the only variant
with a measured negative effect. It is not tuned, and no other weight setting will be run.

### Why `mq_nested3_budget33` and not a deeper control

`mq_nested3` unions up to 600 candidates (3 variants × 2 legs × 100) before truncation to 100,
against the control's 200. The obvious control is to give the single query 3× the depth. **That is
not possible on this system**: `recall.store.sparse_ef_search` caps at pgvector's
`hnsw.ef_search` ceiling of 1000, and 1000 was measured to yield exactly 100 rows on this shared
sidecar index. Asking the SPLADE leg for 300 would silently return ~100, the same class of failure
as the `ef_search` truncation bug that returned 6 of 100.

So the budget is matched from the other direction: truncate every leg ranking to 33, giving
3 × 2 × 33 = 198 ≈ the control's 200. It costs no extra retrieval, and it answers the deployment
question: is multi-query worth it at *fixed total retrieval cost*?

## Contrasts, decision rule, and the Holm family

Decision metric is **R@100**, paired over all 777 judged queries. Coverage is the bottleneck and
R@100 is what a coverage lever must move.

| id | contrast | what it decides |
|---|---|---|
| **P1** | `mq_nested3` − `mq_last` | does multi-query fusion move coverage at all |
| **M1** | `mq_nested3` − `mq_rewrite` | **fusion vs rewriting quality** |
| **T1** | `mq_nested3` − `mq_flat6` | does the nesting topology matter |
| **R1** | `mq_nested2` − `mq_nested3` | does including a harmful variant hurt |
| **B1** | `mq_nested3_budget33` − `mq_last` | diversity at fixed retrieval budget |

Holm family = **these five R@100 contrasts**, α=0.05. nDCG@5, R@5 and R@10 are recorded for every
arm as descriptive secondaries; they are **not** in the family and **cannot** trigger a ship.

Statistics per `reference-validation-standards`, reusing `analyse_contrasts.paired_stats`
unchanged: paired bootstrap over queries (n=10000, ≥2000 required), sign-flip permutation
(n=5000), Holm-Bonferroni step-down at 0.05.

Both the full-set mean (over 777, the number that would ship) and the deciding-cell mean (over the
queries where variants differ) are reported for every contrast. The full-set mean is the one the
decision rule reads.

### Decision rule

**SHIPS** iff all three hold:
1. P1 point estimate ≥ **+0.020 R@100**, and
2. P1's 95% paired bootstrap CI excludes zero, and
3. P1 is Holm-significant within the family.

**The mechanism is FUSION** (⇒ an LLM only has to supply variety, and renting a GPU is justified)
iff M1's CI excludes zero and M1 > 0.

**The mechanism is REWRITING QUALITY** ⇒ **the lever is DEAD** if M1 ≤ 0 or its CI spans zero.
Gold rewrite is an unattainable ceiling for an LLM rewriter, so a fusion that cannot beat it offers
no headroom an LLM could reach. In that case: no GPU rental, and the finding is recorded as a
closed hypothesis.

If P1 clears (2) and (3) but lands below +0.020, the verdict is **INCONCLUSIVE, real but
sub-threshold**, and the follow-up is stated in the results, not decided here.

A point estimate on its own decides nothing.

## Predicted outcome

Written before running. Recorded so the interpretation is not chosen after seeing the numbers.

1. **P1 positive and clears +0.020.** Confidence: moderate-high. Three rankings unioned under a
   truncation at 100 mechanically raises the chance a gold document is inside the cut, and rank 1
   measured a large diversity gain on the same benchmark family.
2. **M1 positive but small, roughly +0.010 to +0.020.** Confidence: moderate. This is the contrast
   the whole experiment exists for, and it is the one I am least able to predict.
3. **T1 ≈ 0 on R@100.** Confidence: high. Nesting reweights *order*; R@100 counts a *set*, and set
   membership is far less sensitive to fusion topology than ordering is. If T1 is large, my model
   of why nesting helps is wrong.
4. **R1 ≈ 0, or slightly negative** (i.e. dropping `full` does not help, and may hurt). Confidence:
   moderate. RRF's rank damping is supposed to absorb a weak ranking; if a variant measured at
   −0.0972 nDCG@5 can be fused in without cost, that is a direct demonstration of what
   variance-aware fusion buys.
5. **B1 positive but clearly smaller than P1.** Confidence: moderate. Some of P1 should be raw
   candidate mass rather than diversity.

The most informative failure would be **P1 large and M1 ≈ 0**: fusion works, but only because gold
rewrite carries it, which would close the lever despite a headline gain. The analysis reports M1
whatever P1 does.

## Validation gates, run before any contrast is computed

1. **Control reproduction.** `mq_last` must reproduce the archived `hybrid_splade` raw
   R@100 = 0.7377 and nDCG@5 = 0.3573 to four decimals. The pipeline is deterministic (fixed HNSW
   index, fixed ef_search, deterministic encoders), so anything else means this run's retrieval is
   not the archived run's and no contrast against it is legitimate. **Deviation stops the run.**
2. **Leg depth.** Every leg of every query must return the 100 rows it asked for. Shortfalls are
   counted and reported per leg per variant; this is the `ef_search` trap, which no test caught and
   only a timing anomaly exposed.
3. **Fusion reconstruction.** Rebuilding `mq_last` from dumped per-leg rankings must reproduce
   `HybridRetriever.search()`'s ordering **exactly** on a sample. The offload harness earned this
   gate the hard way: a fusion that merely looks reasonable produces publishable-looking numbers
   that RE-call would never compute.
4. **Query-side encoder identity.** The venv must still reproduce the stored SPLADE vectors at
   cosine 1.00000000 before any measurement.

## Scope and standing constraints

- **Dev split only.** MTRAG-UN is sealed; the harness defaults to `--split dev` and a test asserts
  it. Nothing here touches it.
- **No LLM calls, no GPU, no paid API.** This experiment is CPU retrieval over files already on
  disk. A GPU rental is a *possible consequence* of the result, not part of it.
- **Database `recall_splade` on VPS2 only.** `sentiment_agent` is the money-path DB; no DDL there.
- VPS2 is shared with another session (load average 23 on 12 cores at the time of writing). The
  dump is scheduled around that and nothing belonging to the other session is killed.

## Amendment 1, 2026-08-06, made BLIND

⏱️ Recorded while the dump was still retrieving its third variant. **No arm had been fused and no
score from this experiment existed**, so this is a change to the rule, not a change to a rule that
had been seen to fail. It only ever makes shipping harder.

Prompted by a review from a parallel session that had independently designed the same experiment.
Three gaps were raised; one was real.

**1. The rule was one-sided (ACCEPTED).** R@100 alone decided the ship. An arm that lifted
coverage while degrading the ranking would clear the bar. That is not hypothetical on this
project: the archived SPLADE run is exactly that shape ("the headline reverses under reranking,
and that is the finding"), and rank 1 reports the same trade. Added `VETO_METRICS =
("nDCG@5", "nDCG@10", "R@5", "R@10")` on the P1 arm pair. A veto trips when a metric's whole 95%
CI sits below zero, and produces the verdict `BLOCKED_BY_RANKING_REGRESSION`.

- Vetoes can **only block, never trigger** a ship, so this cannot become a fishing device.
- Vetoes are deliberately **not** in the Holm family and **not** multiplicity-corrected. Holm
  suppresses false positives among discovery claims; correcting a safety guard makes harm harder
  to detect and biases the procedure toward shipping. A guard has to be easy to trip.
- A merely negative point estimate does not block. An established regression does.

**2. "The deciding-cell predicate is arm-agnostic and dilutes R1" (REJECTED, no number changes).**
The claim was that on a query where `rewrite == last` but `full` differs, the query counts as
deciding while `mq_nested2` is a no-op. That conflates two things. R1 is
`mq_nested2` − `mq_nested3`, and `mq_nested3` **fuses `full`**, which is precisely what differs
there, so the delta is genuinely non-zero. R1's structural zero is "all three identical" (102
queries), giving the same cell of 675. The reviewing session's own implementation builds the cell
from the union of the two arms' variants and returns 675 for R1 as well, agreeing with this code
and not with its prose.

The underlying principle is still better than a hardcoded rule, so `deciding()` now derives the
cell from the union of the variants the two arms actually fuse. It reproduces 675 / 777 exactly
for the five declared contrasts and would give the right answer for a contrast added later over a
different variant set.

**3. No destination diagnostic (ACCEPTED, diagnostic only).** Added `coverage_destination`: of the
gold `mq_nested3` has that `mq_last` does not, where does it rank? The same question closed the
reranker lever. It carries its own caveat in its output, because SPLADE's extra gold also looked
healthy raw and was then buried below rank 10 by **both** cross-encoders, so a good number here is
not evidence the documents survive reranking.

**Also adopted:** a third reporting population, "all three variants distinct" (n=568), alongside
the full set (777) and the deciding cell (675). Contributed by the same session, which derived the
209 / 102 / 568 split independently and got the identical numbers. Its accompanying claim that
reporting the 777 "dilutes a real effect by 27% of no-ops" is **not** adopted: 27% is the
`last`==`rewrite` share, which is the no-op fraction for a rewrite-only contrast. For P1 the
no-ops are the 102 where all three coincide, so the dilution is 13.1%, as stated above.

## Deliverable

A decision on P1 with paired CIs and Holm correction, plus an archived run under
`/var/lib/recall-benchmarks/2026-08-06-mtrag-multi-query-diversity/` with a SHA256 manifest and a
`NOTE.md` carrying the caveats, matching the shape of the 2026-08-06 SPLADE archive.

---

# RESULTS (appended 2026-08-06, after the run)

Nothing above this line was edited after scores were observed. Full write-up and caveats:
`/var/lib/recall-benchmarks/2026-08-06-mtrag-multi-query-diversity/NOTE.md`.

**Verdict: SHIPS. Mechanism: FUSION. GPU rental justified.**

All four validation gates passed first. `mq_last` reproduced the archived `hybrid_splade` raw
figures exactly (R@100 0.7377, nDCG@5 0.3573), and rebuilding it from the dumped legs matched a
live `HybridRetriever.search()` ordering on 25/25 sampled queries.

| id | contrast | Δ R@100 | 95% CI | p | Holm |
|---|---|---|---|---|---|
| P1 | nested3 − last | **+0.1236** | [+0.1023, +0.1457] | 0.0002 | yes |
| M1 | nested3 − gold rewrite | **+0.0351** | [+0.0220, +0.0492] | 0.0002 | yes |
| T1 | nested3 − flat6 | −0.0039 | [−0.0101, +0.0018] | 0.197 | no |
| R1 | nested2 − nested3 | **−0.0185** | [−0.0308, −0.0063] | 0.0022 | yes |
| B1 | budget33 − last | **+0.1174** | [+0.0953, +0.1403] | 0.0002 | yes |

R@100 moves 0.7377 → 0.8613. No ranking veto tripped: nDCG@5 +0.0082 (CI spans zero), nDCG@10
+0.0222, R@5 +0.0186, R@10 +0.0543.

Four findings: fusion beats the **gold** rewrite, so the lever is diversity and not rewriting
quality (M1); fusing the individually worst variant **helps**, so reformulations need not be good
to be worth fusing (R1); the gain is diversity and not retrieval budget (B1, 95% of P1); and the
nested topology bought nothing over flat here (T1).

Predictions 1, 3 and 4 were correct, 2 was directionally right with the magnitude under-predicted,
and **5 was wrong**: I expected candidate mass to explain a visible share of the gain and it
explains almost none.

⛔ `mq_nested3` consumes a GOLD human rewrite and is **not deployable**; "SHIPS" is a decision
about the mechanism under the preregistered rule, not a recommendation to deploy that arm. The
post-hoc no-gold arm (`last`+`full` only) gains +0.0842 R@100 but trips three ranking vetoes.
