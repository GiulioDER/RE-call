# Does the multi-query coverage gain CONVERT under reranking? Preregistration

Status: **PREREGISTERED**. Written 2026-08-06, before any pair was scored.
Predecessor: `2026-08-06-multi-query-diversity-design.md` (the raw run, R@100 0.7377 to 0.8613).

Everything above the RESULTS heading is a commitment.

## The question, and why it is the first thing to do

The raw run established that fusing three reformulations moves R@100 by +0.1236,
CI [+0.1023, +0.1457]. **Every number in it is raw retrieval.** Two facts say that is not enough:

1. **The precedent.** SPLADE moved R@100 by +0.0303 and the ranking gain did not follow: reranked
   nDCG@5 came out 0.3769 against a lexical control at 0.3811, marginally BEHIND. The archived
   note's own headline is "the headline reverses under reranking, and that is the finding".
2. **The destination diagnostic.** Of the 279 gold documents `mq_nested3` adds over `mq_last`,
   **200 sit below rank 10** in the raw fused ranking. That is the same shape as the SPLADE-only
   gold that two cross-encoders, 25x apart in size, then buried.

So the coverage might be real and worthless. This experiment settles it, and it can invalidate the
practical value of the entire predecessor run, which is why it is step 1.

## 🔑 The design decision, and the metric it rules out

RE-call reranks the WHOLE fused pool and truncates afterwards. Doing that here would confound the
arm with the pool width, because the widths differ enormously (measured realised medians:
`mq_last` 167, `mq_nested3` 315). Width alone is known to move this result:
`closed-hypothesis-recall-rerank-pool-interaction-2026-08-05` found the SAME MiniLM got WORSE as
the pool widened, "a wider pool did not give the cross-encoder more to select from, it gave it
more rope". A naive whole-pool comparison would hand the primary arm nearly twice the rope and
report the consequence as an arm effect.

**Therefore: every arm is reranked over exactly the top 100 of its own fused ranking.** Equal
width for every arm; the only thing that varies is WHICH 100 documents the fusion selected. That
is the controlled comparison this question needs.

⚠️ **Consequence, stated so it cannot be walked back later: R@100 is INVARIANT under this design.**
Permuting a fixed 100-document set cannot change which documents are in it, so reranked R@100
equals raw R@100 by construction, for every arm. **R@100 therefore cannot be the decision metric
here and no version of it will be reported as evidence of conversion.** The decision metric is
**nDCG@5**.

⚠️ This is a controlled contrast, not an end-to-end system measurement. RE-call in production
would rerank the whole pool. A separate whole-pool run is the way to measure the shipped system,
and its result would not be comparable to this one.

## Amendment 1, 2026-08-06, made BLIND

⏱️ Recorded before any pair was scored. The CPU attempt was killed at 2,797 of 112,646 pairs and
its partial score file was **deleted**, so no score from this experiment exists or has been looked
at. Two changes, both of which widen what the run measures rather than changing how it is judged.

**1. A GPU is being rented, so `BAAI/bge-reranker-v2-m3` is added as a declared SECONDARY model.**
The primary and the decision rule stay on MiniLM, which is what RE-call ships. BGE is 568M against
MiniLM's 22M, and the archived SPLADE run found the two indistinguishable on precisely this
question (they buried 91 and 90 of the same 123 gold documents). Running both while the hardware
is up costs minutes and turns "MiniLM did not convert it" into "neither a small nor a large
cross-encoder converts it", which is a much stronger statement and is the one that closed the
reranker lever last time. If the two disagree, the shipped model decides and the disagreement is
itself the finding.

**2. The WHOLE-POOL analysis is added as a declared SECONDARY, from the same scoring run.**
Measured pool medians are `mq_last` 167, `mq_nested3` 315, `mq_nested2_nogold` 284, so the width
confound the equal-width design exists to avoid is real and large (1.9x). The equal-width contrast
remains the PRIMARY and the only thing the decision rule reads. The whole-pool numbers answer the
different, obvious question of what the shipped system would do end to end, and they are reported
**explicitly labelled as confounded with pool width**. They are not in the Holm family.

Scoring the union of both pair sets costs 241,270 pairs instead of 112,646, which is minutes on a
GPU and saves a second rental.

## Model: MiniLM primary, BGE secondary

`cross-encoder/ms-marco-MiniLM-L-6-v2`, revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`, which
is `recall.rerank.CrossEncoderReranker`'s pinned default and therefore **the reranker RE-call
actually ships**. `scripts/score_pairs.py` already takes it.

MiniLM is the model the decision rule reads, because the decision is about the system RE-call
ships. `BAAI/bge-reranker-v2-m3` (apache-2.0, pin `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`) is
scored alongside it as a declared secondary, per Amendment 1.

⚠️ **CPU is not viable at this size and that is measured, not assumed.** The first attempt ran
MiniLM on VPS2 and sustained **5.2 to 6.7 pairs/s**, projecting to ~4.7 hours for the equal-width
payload alone and far longer for the union. MTRAG passages reach the 512-token limit, so every
pair is a full-length forward pass. The run was killed and its partial output deleted.

## Frozen arms

Reranked over their own top 100, from the archived legs of the predecessor run.

| arm | what it is | why it is here |
|---|---|---|
| `mq_last` | control, configuration-identical to archived `hybrid_splade` | the baseline every contrast is against |
| `mq_nested3` | the primary fusion arm | the coverage gain under test |
| `mq_nested2_nogold` | `last`+`full`, no gold, no LLM | the only deployable arm; raw nDCG@5 was −0.0447 |

`mq_nested2_nogold` was post-hoc in the predecessor run. It is declared here **before any pair is
scored**, so for this experiment it is preregistered.

## Contrasts, decision metric and Holm family

Decision metric **nDCG@5**, paired over the 777 judged dev queries. nDCG@10, R@5 and R@10 are
reported alongside for every contrast as secondaries. Holm family = the three contrasts below on
nDCG@5, alpha 0.05. Statistics via `analyse_contrasts.paired_stats` and `holm`, imported unchanged:
paired bootstrap n=10000, sign-flip permutation n=5000.

| id | contrast | what it decides |
|---|---|---|
| **C1** | `mq_nested3` − `mq_last` | **does the coverage convert?** |
| **C2** | `mq_nested2_nogold` − `mq_last` | does the deployable arm's ranking cost survive reranking? |
| **C3** | `mq_nested3` − `mq_nested2_nogold` | what the gold rewrite is worth once reranked |

### Decision rule

- **CONVERTS** iff C1's mean is positive AND its CI excludes zero AND it is Holm-significant.
- **MATERIALLY CONVERTS** iff, in addition, C1 >= **+0.010 nDCG@5**.
- **DOES NOT CONVERT** if C1's CI spans zero. In that case the multi-query lever reproduces the
  SPLADE outcome, the R@100 headline does not translate into what a reader sees, and the honest
  reading is that coverage is no longer the binding constraint.
- **REVERSES** if C1's CI lies entirely below zero.

C2 decides separately whether the no-gold arm is deployable at all: it was raw-blocked by a
−0.0447 nDCG@5 regression, and reranking is the obvious candidate to repair it.

## Validation gates, before any contrast is computed

1. **Offloaded ordering matches the in-process reranker.** `rerank_offload.rerank_order` is reused
   rather than reimplemented, and a sample is checked against a live `CrossEncoderReranker`. The
   existing tolerance rationale applies: scores must agree as arithmetic and order must be exact
   where metrics are cut.
2. **Control reproduction.** `mq_last` reranked at width 100 is a configuration nobody has run
   before, so there is no archived figure to reproduce. Instead the RAW `mq_last` numbers must
   still match the archive exactly (R@100 0.7377, nDCG@5 0.3573) when recomputed from the same
   rankings file, confirming the input to this stage is the frozen run's output.
3. **Every pair scored.** `rerank_order` raises on a candidate with no score rather than treating
   it as zero. No arm may have a missing pair.

## Predicted outcome

Written before scoring.

1. **C1 is small, and I expect it may well be null: roughly 0.00 to +0.02 nDCG@5.** Confidence:
   moderate. The two facts in "The question" both point that way, and the raw nDCG@5 gain was only
   +0.0082 with a CI spanning zero. If C1 lands above +0.03 my model of where the added gold sits
   is wrong.
2. **C2 stays negative but shrinks** from its raw −0.0447. Confidence: moderate. A cross-encoder
   re-reads the passage against the query and should recover some of what the noisy concatenated
   query's ranking cost.
3. **C3 is positive.** Confidence: moderate-high. Gold was worth +0.0529 nDCG@5 raw and it is the
   variant that carries ranking quality.

The most consequential outcome is **C1 null**: it would mean the predecessor run's headline is a
coverage number that a reader never sees, and the next lever is ranking rather than more recall.
I am recording that now so the interpretation is not chosen after the fact.
