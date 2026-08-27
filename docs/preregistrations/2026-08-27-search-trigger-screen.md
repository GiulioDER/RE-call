# Pre-registration: can anything decide WHICH writes deserve a search?

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_trigger_screen.py`, committed with this record.

## The question

Draft-time search finds the governing memo for **14 of 14** recorded misses, and fires just as
confidently on writes with no hazard at all. The threshold frontier showed no confidence cut
separates the two; the fusion frontier showed no weighting of the two legs does either, across
twelve variants. Both closed with the same verdict: **the remaining question is a TRIGGER, not a
retriever.**

So: **is there any signal, other than the retrieval score, that decides which writes deserve a
search — keeping the 14 of 14 while staying silent on writes with nothing to find?**

## Two kinds of trigger, and the distinction is the cost

A **pre-search** trigger inspects the draft and decides whether to search at all. It saves the
search, its latency, and the tokens its results would consume. A **post-search** filter runs the
search and decides whether to surface the results. It saves only attention. Both are worth
measuring and they are not interchangeable, so this record labels every candidate and never
compares one against the other on cost.

| id | trigger | kind | cost per write |
|---|---|---|---|
| `T0` | always search | — | the baseline the previous records measured |
| `T1` | **score margin**: surface only if rank-1 exceeds rank-2 by more than δ, swept | post-search | a search, no model |
| `T2` | **hazard vocabulary**: search only if the draft contains a term the corpus's memos distinctively warn about, derived from the corpus by tf-idf and frozen before use | pre-search | a set lookup |
| `T3` | **operation class**: search only on a draft that writes a file, spawns a subprocess, or runs a VCS command | pre-search | a regex |
| `T4` | **LLM gate**: one `claude-haiku-4.5` call on the draft alone — "could this code fail in a way a note might warn about?" | pre-search | one small model call |

⚠️ **`T1` is adjacent to what the fusion record closed and is included deliberately.** That record
killed *absolute score thresholds*; a margin is a different statistic on the same retrieval, and it
has never been measured here. If it works it is a post-search filter only, and the record will say
so rather than describing it as a trigger.

## Design

- **One retrieval capture, persisted.** The fusion probe kept only summaries and this record pays
  for retrieval again as a result. This probe writes **per-draft** records (query, per-leg ranked
  lists, governing-memo rank, top-5 scores) so every future trigger question is free. That is the
  same "collect once, sweep offline" discipline, applied to the artifact rather than the run.
- **Retriever:** `lexical_only`, the variant that reaches 14 of 14, on `probe2_control`.
- **Positives:** the 14 registered miss sessions and their recorded draft payloads.
- **Negatives, two populations, reported separately and never pooled:**
  - `N_clean` — the 18 `ctl-stage-by-pathspec` draft queries, whose hazard is genuinely absent
    from the corpus. Small, and the honest denominator for "fires when there is nothing to find".
  - `N_wide` — every draft in the 14 positive sessions that does NOT retrieve the governing memo
    at top-5, roughly 200 of them. Searching these is waste even though their session contains a
    real hazard elsewhere. This is an **operational** definition of hazard-free, and it is
    **circular for `T1`** (which is computed from the same retrieval) and sound for `T2`–`T4`
    (which never see it). `T1` is scored on `N_clean` only, and the record will say so.

**Endpoints, per trigger:**

1. **Coverage** — of the 14 sessions, how many still surface the governing memo: the trigger must
   fire on at least one draft that retrieves it. This is the number that must not regress.
2. **False-trigger** — the share of `N_clean` (and separately `N_wide`) the trigger fires on.
3. **Suppression** — the share of all positive-session drafts the trigger silences, which is the
   cost saving a pre-search trigger buys.

## What I predict

Per `[[i-over-predict-effect-magnitudes]]`. These are information-side changes, where this lane has
under-called four times running, so I predict at the arithmetic without discounting; the exception
is `T3`, where the arithmetic itself is unfavourable.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **`T3` is useless: it fires on >= 0.85 of all drafts.** Nearly every recorded payload writes a file or runs a command; that is what a coding session IS | below 0.85 |
| 2 | **`T2` coverage 12 to 14 of 14**, because the hazardous draft contains the hazard's own identifiers by construction, which is why draft search worked at all | outside the band |
| 3 | **`T2` false-trigger on `N_clean` is 0.40 to 0.80.** The negatives are git operations against a git-heavy corpus, so its vocabulary will overlap | outside the band |
| 4 | **`T4` is the best discriminator: coverage >= 12 with `N_clean` false-trigger <= 0.35** — the only candidate that reasons about the code rather than matching strings | coverage below 12, or false-trigger above 0.35 |
| 5 | **`T1` fails: no δ gives coverage >= 12 with `N_clean` false-trigger <= 0.35.** A margin is a different statistic but the same signal, and that signal has now failed to separate twice | such a δ exists |
| 6 | **Cost: under 30 minutes and under 1.00 USD** (~470 haiku calls for `T4`, one retrieval pass, everything else free) | either bound exceeded |

## Decision rule, as a cross product

Per `[[state-the-partition-over-the-cross-product]]`. For the BEST trigger by coverage, bands are
coverage (`<=9`, `10-12`, `>=13`) crossed with `N_clean` false-trigger (`<=0.35`, `0.36-0.70`,
`>0.70`). Nine cells, every one assigned:

| coverage \ ft | `<=0.35` | `0.36-0.70` | `>0.70` |
|---|---|---|---|
| **`<=9`** | **DEMOTE**: the trigger works but costs a third of the recall the retriever earned | **KILL**: worse than the retriever on both axes | **KILL** |
| **`10-12`** | **BUILD the trigger**, name it, preregister the live A/B including its cost | **GATE**: preregister one iteration on the best candidate only | **KILL** |
| **`>=13`** | **BUILD**, and this is the outcome that reopens the whole direction | **GATE**, with the operating point recorded | **KILL the trigger lane**: nothing separates, and the direction ends with "retrieval solved, gating impossible" |

The three `>0.70` cells all kill, on the same standard the threshold and fusion records were held
to: a signal that fires on most hazard-free writes is not a trigger whatever its coverage.

**If the killing cell is reached, that is the end of this line of work**, and the honest summary is
that draft-time search retrieves perfectly and cannot be gated — which is a publishable negative
result about agent memory, not a failure to find a trick.

## What I already know

- Draft-time recall: 14/14 (`lexical_only`), 11/14 (production RRF), and 7/14 served.
- No confidence threshold and no fusion weighting separates positives from negatives: 0 viable
  points in each of two registered sweeps.
- Draft payloads: median 60 characters, ~8-10 per session, 3 of 501 over the server's 4,096-char
  limit.
- The trust layer never abstained on a negative: 0 of 18.

## Confounds I can name now

1. **`N_clean` is 18 queries from 4 sessions**, and every previous record in this lane has flagged
   it. It is the binding limitation on predictions 3, 4 and 5. `N_wide` exists to give power, at
   the price of an operational ground truth. If the two populations disagree, that disagreement is
   the finding and the record will report it rather than choosing the friendlier one.
2. **`T2`'s vocabulary is derived from the same corpus the search runs against**, so it is closer
   to a cheap retrieval than to an independent signal. Its result should be read as an upper bound
   on what string matching can do, not as a separate mechanism.
3. **`T4` is a model judging code it did not write**, with a fixed committed prompt at temperature
   0, and every verdict recorded. It is also the only candidate whose cost scales with every write.
4. **Ground truth for `N_wide` is defined by the retriever**, which is the circularity named above.
   `T1` is excluded from it by construction.
5. **Coverage counts a session as covered if ANY of its drafts fires.** With ~10 drafts per session
   that is generous, and it is the same asymmetry the fusion record used, kept for comparability
   and stated here so it cannot be read as a new choice.

<!-- frozen_above -->

## Result (2026-08-27)

**Status: measured. Both controls passed** (retrieval positive control, and the T0 scoring control
reproducing the always-search baseline at 14 / 1.000 / 0.000 exactly).
Artifact: `benchmarks/artifacts/agent_ab/trigger-screen.json`, which persists all 196 per-draft
records so no future trigger question pays for retrieval again.

Population: 178 positive drafts across the 14 sessions, of which **46 (26%) are hazard-bearing**
(they retrieve the governing memo at top-5); 132 `N_wide`; 18 `N_clean`.

| trigger | kind | coverage | ft_clean | ft_wide | suppression |
|---|---|---:|---:|---:|---:|
| `T0_always` | — | 14/14 | 1.000 | 1.000 | 0.000 |
| `T1_margin_0.0` | post | 11/14 | 0.556 | 0.568 | 0.438 |
| `T1_margin_0.01` | post | 11/14 | 0.500 | 0.477 | 0.506 |
| `T1_margin_0.02` | post | 10/14 | 0.389 | 0.326 | 0.652 |
| `T1_margin_0.05` | post | 5/14 | 0.056 | 0.099 | 0.888 |
| **`T2_vocab_df2`** | **pre** | **10/14** | **0.167** | 0.258 | **0.674** |
| `T2_vocab_df3` | pre | 10/14 | 0.278 | 0.326 | 0.601 |
| `T2_vocab_df5` | pre | 10/14 | 0.333 | 0.348 | 0.567 |
| `T2_vocab_df10` | pre | 10/14 | 0.389 | 0.553 | 0.405 |
| `T3_operation` | pre | 10/14 | **1.000** | **0.159** | 0.753 |
| `T4_llm_gate` | pre | 9/14 | 0.389 | 0.561 | 0.405 |

**Triggers clearing coverage >= 10 AND ft_clean <= 0.35: three, all `T2` (df2, df3, df5).**

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | `T3` fires on >= 0.85 of drafts | fires on **0.247** (suppression 0.753) | **falsified, badly** |
| 2 | `T2` coverage 12–14 | **10** at every df | **falsified, below** |
| 3 | `T2` ft_clean 0.40–0.80 | **0.167–0.389** | **falsified, below** (better than predicted) |
| 4 | `T4` is the best discriminator, coverage >= 12 and ft_clean <= 0.35 | coverage **9**, ft_clean **0.389** | **falsified on both** |
| 5 | `T1` fails: no δ with coverage >= 12 and ft_clean <= 0.35 | no δ reaches coverage 12 at all | confirmed |
| 6 | under 30 minutes, under 1.00 USD | ~25 minutes, ~196 haiku calls, ~0.05 USD | confirmed |

🔑 **The headline: a dumb string-membership test beat the LLM gate on every axis.** `T2_vocab_df2`
covers 10 of 14 while firing on 3 of 18 hazard-free writes and suppressing 67% of searches;
`T4_llm_gate` covers 9 while firing on 7 of 18 and suppressing 41%. The candidate I predicted would
win because it "reasons about the code rather than matching strings" was dominated by matching
strings, at a fraction of the cost and with no model call per write.

⚠️ **`T3` is the registered population-disagreement case, and it fired.** Its `ft_clean` is 1.000
and its `ft_wide` is 0.159 — the widest divergence in the table. The cause is the confound this
record named in advance: `N_clean` is `ctl-stage-by-pathspec`, whose every draft is a git command,
and `T3` fires on git commands by construction. **`T3`'s clean false-trigger rate measures the
negative set, not the trigger.** That is exactly why the two populations were registered separately
and never pooled, and it is the clearest argument in this lane for a bigger, more varied negative
set before any trigger is trusted.

## ⛔ The decision rule is ambiguous in its SELECTION, and this is a third distinct failure

The rule scores "the BEST trigger by coverage". Read literally, that is `T1_margin_0.01` at 11 of
14, whose `ft_clean` of 0.500 lands in the `0.36-0.70` band, giving the cell **GATE: preregister
one iteration on the best candidate only.**

But `T2_vocab_df2`, at coverage 10 with `ft_clean` 0.167, sits squarely in the **BUILD** cell. A
selection by coverage alone picks a dominated candidate.

**The registered outcome is GATE, and I am not substituting the friendlier reading after the
fact.** Recording the flaw rather than routing around it:

- The previous record's failure was an incomplete partition over OUTCOMES.
  `[[state-the-partition-over-the-cross-product]]` fixed that, and this record's nine cells are
  complete and were read without a gap.
- **This failure is different: the outcome space was partitioned, the CANDIDATE space was not.**
  A rule that scores one selected candidate silently assumes a dominant one exists. With a
  frontier of candidates, "best by X" can select something another candidate dominates on both
  axes.
- The fix is mechanical and belongs beside the partition rule: **when a probe compares more than
  one candidate, register the SELECTION as a Pareto rule over the endpoint pair, not a sort on one
  endpoint** — "the candidate on the Pareto frontier maximising coverage subject to ft_clean <=
  0.35, and if that set is empty, the one maximising coverage".

## What this licenses

**GATE**: one registered iteration on `T2`, which is the candidate the rule's own cell language
points at ("the best candidate"), read as the best by the rule's stated intent rather than by its
literal sort. That iteration must, before anything is built:

1. **Enlarge `N_clean`.** Eighteen drafts from one git-flavoured task have now distorted one
   trigger's headline number outright (`T3`) and are the binding limit on every other. This is the
   single highest-value thing left in this lane.
2. Fit the `df` threshold on **held-out families**, since df2/df3/df5 were all read from the same
   14 sessions that define coverage.
3. Price the coverage loss honestly: 10 of 14 against the retriever's 14 of 14 means the trigger
   discards 4 sessions the retrieval had already solved.

**Not licensed:** building `T2` into anything, or quoting 0.167 as a false-trigger rate outside
this record. It is 3 of 18.
