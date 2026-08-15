# Pre-registration: is the missing evidence already in the pool, and can we tell in advance?

**Date:** 2026-08-15   **Status:** predicted, not yet measured

## Registration

```yaml
registration_commit: PENDING
registration_authored: PENDING
label_source: results/enterprise_rag/judgements.gpt-5.4.medium.json
retrieval_fixture_digest: PENDING
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
