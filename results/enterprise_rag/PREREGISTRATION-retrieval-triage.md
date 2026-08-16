# Pre-registration: is the missing evidence already in the pool, and can we tell in advance?

**Date:** 2026-08-15   **Status:** predicted, not yet measured

## Registration

```yaml
registration_commit: c8828db65f0577aa7b999e5bb4fee46fe7515e61
registration_authored: 2026-08-15T21:52:54+00:00
label_source: results/enterprise_rag/judgements.gpt-5.4.medium.json
retrieval_fixture_digest: b6405b77a2d75472e03c651c2b51b9a62bde4a6d0da6f1c65597091e7492a774
```

## The question

Two, from one retrieval pass over all 500 questions. No generation, no judge.

1. **Budget or quality?** Of the gold documents that never reached the answer, what fraction were
   already in the 200-candidate pool and were cut at the `k=8` boundary?
2. **Can we tell in advance?** Is there a signal available AT QUERY TIME, before any answer is
   generated, that separates the questions whose gold was missed from the questions whose gold was
   found?

Question 2 is the one that decides whether any of this becomes a product. A system that knows
after the fact that it needed more evidence has learned nothing useful. A system that knows
BEFORE answering can spend depth on the ~quarter of queries that need it and stay cheap on the
rest.

## Why this, and why now

`ANALYSIS-where-reasoning-could-help.md` segments all 500 rows and finds **21.6 of 36.2 aggregate
points (60%) are rows where retrieval never delivered the gold evidence**. The corpus is 511,963
documents, about **619M tokens**, so it is **619x too large to put in a 1M context window**. No
competitor is stuffing this corpus.

What a large context actually buys is a **looser k**: retrieve 100 and let the model filter,
rather than retrieve 200 and commit to 8. That is a budget decision, not a retrieval skill, and it
is the specific advantage worth testing rather than assuming.

⚠️ **The strategic reason not to simply raise k**: `info_not_found` scores 20/20 and
`miscellaneous` 100% today. Abstention is the one axis this system leads on, and abstention gets
harder with more distractors, not easier. Matching a competitor by raising k trades the lead for
the deficit. **Selective depth is the only version that keeps both**, and it requires question 2 to
have a yes.

## What I predict

**Metric names, because a rate is named by its denominator.**

- `pool_recovery_rate` = missed gold documents present in the top-200 pool ÷ **missed gold
  documents**.
- `recall_at_k` = gold documents present in the top-k ÷ **gold documents**, for k in {8, 200}.
- `triage_auc` = ROC AUC of one query-time feature separating **questions** whose gold was fully
  retrieved from questions whose gold was not. Denominator is questions, n=500.

| # | quantity | point | interval |
|---|---|---|---|
| T1 | `pool_recovery_rate` | **0.45** | 0.20 to 0.70 |
| T2 | `recall_at_8`, all 500 | 0.72 | 0.65 to 0.80 |
| T3 | `recall_at_200`, all 500 | **0.88** | 0.78 to 0.96 |
| T4 | **`triage_auc`, best single feature** | **0.70** | 0.55 to 0.85 |
| T5 | `triage_auc` of the SHIPPED `gap_warning` flag alone | 0.55 | 0.50 to 0.68 |

**Ordering predictions:**

- **O1.** The best triage feature is a **retrieval-score** feature (top-1 rerank score, or the
  decay between rank 1 and rank 8), not a **query-text** feature (length, question words,
  conjunction count). If a text feature wins, the signal is about question difficulty rather than
  about this retrieval's confidence, and it would generalise differently.
- **O2.** `recall_at_200` exceeds `recall_at_8` by more than 0.10. If the pool holds barely more
  gold than the top-8 does, then no amount of "look deeper" helps and the long-context advantage
  is not a k advantage on this corpus.
- **O3.** Multi-gold questions have a lower `recall_at_8` than single-gold questions. Already
  implied by the segmentation, and included because it is free and it checks the retrieval
  fixture against the judgements I already hold.

## What would falsify this

- **T1 near 0**: the missed gold is not in the pool either. Raising k cannot help, the "budget not
  quality" thesis is wrong for this corpus, and the work belongs in embedding or query rewriting
  rather than in depth.
- 🔑 **T4 at or below 0.55**: **there is no query-time triage signal**, and "know in advance when
  reasoning is needed" fails on this evidence. That is the outcome that would kill the strategy,
  and it must be reported as loudly as a success.
- **O2 fails**: the pool is no richer than the top-8, which makes T1 and T4 moot.
- **T5 well above T4**: the shipped `gap_warning` already does the job and no new mechanism is
  needed. That would be a good outcome and an embarrassing one.

⚠️ **AUC on 500 questions with roughly 160 positives has a standard error near 0.025.** A measured
0.70 is not distinguishable from 0.65. Any feature selected as "best" out of several is selected
on the same data it is scored on, which inflates it: **T4 is reported as an upper bound and the
winning feature must be re-scored on a held-out split before it is believed.**

## How it will be measured

- **Retrieval**: one pass over all 500 questions against `ber_voy_lex_12k_full` on VPS2, under the
  frozen configuration recorded in the fixture provenance, capturing for each question the ranked
  top-200 candidates with scores, and the top-8 after rerank. Frozen and digested like the
  supersession fixture, so the index leaves the loop.
- **Labels**: `document_recall_pct` from `judgements.gpt-5.4.medium.json`, already committed. Not
  recomputed, so the label cannot drift to fit the features.
- **Features**, all computable before an answer exists: top-1 rerank score; mean top-8 score; score
  decay rank1 minus rank8; the shipped `gap_warning`; distinct documents in the top-8; query
  character length; count of coordinating conjunctions; question-word count.
- **AUC**: computed directly, no library fitting, no threshold tuning.

### Apparatus checks, with known answers, before the numbers are read

| # | check | known answer |
|---|---|---|
| R1 | retrieval of the same question twice | byte-identical ranked list. ⚠️ The supersession probe found temperature 0 is NOT deterministic for GENERATION; retrieval has no sampling, so this must hold, and if it does not the whole fixture is unusable |
| R2 | AUC of a random feature | ≈ 0.50 |
| R3 | AUC of the LABEL used as a feature | 1.00 |
| R4 | `recall_at_8` recomputed from the fixture vs `document_recall_pct` in the judgements | agree within rounding on at least 90% of rows; a large disagreement means the fixture's retrieval is not the retrieval that was judged |

R4 is the one that matters most: this run's retrieval deviates from the submitted run (SPLADE on
CPU, `rerank_document_chars` 3900), and R4 measures how much that deviation actually moved
retrieval rather than assuming it was small.

## What I already know

- 21.6 of 36.2 points are retrieval misses; 12.8 of that is single-gold questions where the one
  document never came back, a 94% failure rate on those rows.
- The multi-document penalty is 33 to 37 points and survives a judge swap.
- The supersession probe returned a null inside its own noise floor, and its instrument could not
  see what made its rows wrong. **This experiment is deliberately retrieval-only for that reason**:
  no generator, no judge, no sampling, so the measurement is repeatable in a way that one was not.

## Confounds I can name now

1. **Feature selection on the scoring data** inflates T4. Named above; a held-out split is required
   before the winning feature is believed.
2. **The label is one judge's.** `document_recall_pct` is mechanical rather than judged, which is
   why it is the label, but the SEGMENT boundaries came from `answer_correct`, which is not.
3. **This retrieval is not the judged retrieval.** R4 measures the gap rather than assuming it.
4. **Chunk-level versus document-level.** Gold is document ids; retrieval returns chunks. A
   document counts as retrieved if any of its chunks is present, which is the same rule the
   benchmark's own scorer uses.
5. **Class imbalance.** Roughly 160 of 500 questions miss gold. AUC handles it; a raw accuracy
   would not, which is why AUC is the registered metric.

---

## Deviation, recorded before the numbers are read: the reranker is disabled

**Appended 2026-08-16. No prediction above is edited.**

The registered plan captured the top-200 **after** reranking. That version measured at ~83
seconds per question, so 500 questions is about 11.5 hours and roughly $15 of Voyage reranking,
and the reranker is essentially the whole of both.

⚠️ **T1 and T3 do not need a reranker at all.** The question is whether the gold document is in the
CANDIDATE POOL, and the pool is produced by the three retrieval legs. The reranker only chooses
which 8 of the pool survive. So this run uses `--reranker none`, which:

- costs query embeddings only, and finishes in minutes rather than hours;
- measures the **true fused pool**, where the registered version measured a pool that had already
  been filtered by the reranker. That is a **more exact** answer to the registered question, not a
  weaker one.

**What is given up, and it is real.** `recall_at_8` under no reranker is not the shipped
configuration's top-8, so **T2 is not scored by this run** and T4 and T5 rest on fusion scores
rather than cross-encoder scores. Both remain open. The triage AUC needs either a local
cross-encoder or a separate reranked pass, and neither is done here.

⚠️ The 12-question pilot that preceded this is **not** superseded evidence, it is different
evidence: its pool was reranker-filtered, so its `pool_recovery_rate` of 3/3 and this run's are
measuring two different pools. They are reported separately and must not be pooled.

---

## Result (2026-08-16)

**Status: measured.** All 500 questions, `--reranker none`, fixture `b6405b77…`. No generation, no
judge, no sampling. **No prediction above is edited.**

### The headline is split, and one half is the registered killer

| # | registered | measured | verdict |
|---|---|---|---|
| T1 | `pool_recovery_rate` 0.45, [0.20, 0.70] | **0.635** (176/277) | **CORRECT** |
| T2 | `recall_at_8` 0.72, [0.65, 0.80] | 0.626 (464/741) | **not scored by this run**, no reranker |
| T3 | `recall_at_200` 0.88, [0.78, 0.96] | **0.864** (640/741) | **CORRECT** |
| T4 | `triage_auc` best feature 0.70, [0.55, 0.85] | **0.537** | 🔑 **FALSIFIED, at the killer threshold** |
| T5 | `gap_warning` AUC 0.55, [0.50, 0.68] | 0.5015 | inside, at the floor: it is chance |
| O1 | best feature is a retrieval-score one | `query_chars`, a TEXT feature | **falsified** |
| O2 | pool lift > +0.10 | **+0.2375** | **HELD** |

### 🔑 T1 holds: the missing evidence is mostly already retrieved

**176 of 277 missed gold documents (63.5%) were sitting in the candidate pool** and were discarded
at the `k=8` cut. `recall_at_200` is 0.864 against 0.626 at k=8, a lift of **+0.24**.

So on this corpus the retrieval "failures" are **mostly a budget decision, not a retrieval-quality
failure**. A competitor with a large context window wins those rows without retrieving any better
than this system does. That is the competitive thesis, and it survived its test.

### 🔑 T4 fails, and the pre-registration says to report this as loudly as a success

**The best single query-time feature scored AUC 0.537.** The registered falsifier reads: *"T4 at or
below 0.55: there is no query-time triage signal, and 'know in advance when reasoning is needed'
fails on this evidence."*

It is worse than the number alone suggests. The random control R2 scored **0.4718**, and at n=470
the standard error is about 0.023, so **anything between roughly 0.45 and 0.55 is indistinguishable
from random**. Every feature tested falls in that band:

- `query_chars` 0.537, `neg_top1_score` 0.532, `gap_warning` 0.5015, `question_words` 0.500,
  `conjunctions` 0.500, `distinct_docs_topk` 0.4985, `neg_mean_topk` 0.4533.

⚠️ **The shipped `gap_warning` scores 0.5015. It is pure chance on this task.** T5's interval
included that, so T5 is technically correct, but the useful statement is that the flag carries no
information about whether retrieval missed the gold.

**On this evidence, we cannot tell in advance which queries need more depth.** The half of the
strategy that makes selective depth a product rather than a cost does not have support yet.

### One post-hoc observation, flagged as post-hoc and NOT counted

`score_decay` scored **0.3644**, which is meaningfully BELOW chance and therefore anti-predictive:
reversed, it would score 0.636. The direction is intuitive, a flat score profile meaning nothing
stood out. ⛔ **This is not a result of this experiment.** Its sign would be chosen after seeing the
data, which is exactly the fitting the registration warned about. It is a hypothesis for a NEW
pre-registration with a held-out split, and it must not be reported as a 0.636 triage signal.

### O3, free and unregistered as a prediction, but the most actionable table here

Gold-missed rate by question type: `completeness` **90%**, `project_related` **77.5%**,
`semantic` 47.2%, `conflicting_info` 40%, `basic` 18.3%, `constrained` 16.7%, `miscellaneous` 15%,
`intra_document_reasoning` **5%**.

An 18x spread between the best and worst type. ⚠️ `question_type` is a benchmark label and is NOT
available at query time, so this is not a triage feature. It says the signal EXISTS in the
question; it does not say we can currently read it.

### What follows

1. **T1 supports raising depth**, and T4 says we cannot yet target it. Selective depth is
   unsupported today; *unconditional* depth is supported but pays the abstention cost the
   registration named.
2. **The next question is not another feature sweep.** Seven hand-picked features all landed inside
   the noise band; an eighth is unlikely to escape it. The `score_decay` direction and the
   question-type spread both suggest the signal is real but not linear in any single scalar.
3. **`recall_at_8` needs the reranker** before any claim about the shipped configuration's top-8.
