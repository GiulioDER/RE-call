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
