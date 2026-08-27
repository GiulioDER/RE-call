# Pre-registration: is the headline an artifact of one judge and three task families?

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_validate.py`, committed with this record.

## Why this record exists

Everything measured in this lane today now rests on two unexamined foundations:

1. **One judge.** A single `claude-haiku-4.5` call with a single prompt produced the corrected
   actionable recall (10 of 14), the 27 actionable labels, and every re-derived trigger population.
   Its agreement with anything has never been measured. The negative-set record names the asymmetry
   as a confound — the control bounded false negatives and said nothing about false positives — and
   nothing has tested it.
2. **Three task families.** Draft-time search, the vocabulary trigger, and the `df<=2` threshold
   were all measured on `ts-lf-rewrite`, `ts-worktree-import` and `ts-sample-covers-tail`, 14
   sessions. The threshold in particular was CHOSEN by looking at those sessions. Nothing has been
   tested out of sample.

Two questions, run together because they share a pipeline and because either one alone leaves the
other unanswered.

## Part A — does a second judge agree?

- **Population:** the same 46 memo-retrieving drafts, unchanged.
- **Judges:** the committed prompt verbatim, at temperature 0, under
  **`anthropic/claude-sonnet-5`** (stronger, same family: tests capability) and
  **`google/gemini-2.5-pro`** (different family: tests independence, since same-family errors
  correlate).
- **Endpoints:** raw agreement and Cohen's κ against the haiku labels, per judge and between the
  two new ones; and **the actionable recall each judge implies**, which is the number that actually
  propagates.

## Part B — does it hold on families never used?

Four families were never touched by any draft-search measurement: `ts-autouse-tmp-path` (6
sessions, 65 drafts), `ts-bounded-runner` (6, 78), `ts-false-zero-search` (6, 13),
`ts-separator-canary` (6, 43). **24 sessions, 199 drafts, four governing memos.**
`ts-raise-on-missing` is excluded again for the reason both prior records give: its governing memo
is the one reconstruction-approximate source in the corpus.

⚠️ **These are not a random holdout and the difference cuts against a clean read.** All four are
families where the GOAL query already worked in the original run, so their memos may be easier to
reach by any route. A held-out result that matches the fitted one is therefore weaker evidence than
it looks, and a held-out result that is WORSE is correspondingly stronger. The record will say
which way it fell rather than treating either as neutral.

- **Draft-search recall:** the same lexical retrieval, judged for actionability by the same haiku
  judge, giving actionable recall over 24 sessions to compare against **10/14 = 0.714**.
- **Trigger transfer:** the vocabulary trigger at the **`df<=2` threshold fitted on the other
  families**, not refitted here. Coverage over reachable held-out sessions, against the fitted
  **9/10 = 0.90**; false-trigger on the same 18 `N_clean` drafts, against **0.167**.

## What I predict

Rates, not absolute counts, because an absolute band is only meaningful against the denominator it
was written for — the defect that voided the trigger grid one record ago.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Sonnet agrees with haiku on 0.75 to 0.90** of the 46 labels | outside the band |
| 2 | **Gemini agrees on 0.65 to 0.85** — lower than sonnet, because a different family decorrelates the errors this is meant to expose | outside the band, or gemini agrees MORE than sonnet |
| 3 | **The implied actionable recall moves by at most 2 sessions under either judge** (that is, stays within 8 to 12 of 14). If it moves more, every number downstream of the label is provisional | it moves by 3 or more under either |
| 4 | **Held-out actionable recall is 0.60 to 0.85** (14 to 20 of 24), i.e. within striking distance of the fitted 0.714 | outside the band |
| 5 | **Held-out trigger coverage DROPS relative to the fitted 0.90, to 0.55 to 0.80.** The `df<=2` threshold was chosen on the fitted families and some shrinkage is the honest expectation | above 0.85 (no shrinkage at all, which would be suspicious) or below 0.55 |
| 6 | **Held-out false-trigger stays at 0.167**, because `N_clean` is unchanged and the vocabulary is corpus-derived rather than family-derived | it moves at all |
| 7 | **Cost: under 45 minutes and under 0.50 USD** | either bound exceeded |

## Decision rules, stated separately because the parts can disagree

**Part A**, on the implied actionable recall across the three judges:

| spread across judges | verdict |
|---|---|
| **<= 1 session** | the label is stable; downstream numbers stand as measured |
| **2 sessions** | the label is soft; every derived number carries a "+/- 1 session" note from here on |
| **>= 3 sessions** | ⛔ **the label is unstable.** Actionable recall, the 27 labels and the re-derived trigger populations all become provisional, and no product decision may rest on them until a labelling protocol with measured agreement replaces the single judge |

**Part B**, on held-out draft-search recall crossed with held-out trigger coverage:

| recall \ coverage | `>=0.80` | `0.55-0.79` | `<0.55` |
|---|---|---|---|
| **`>=0.60`** | **GENERALISES**: both hold; the fitted numbers are not an artifact of three families | **RECALL GENERALISES, TRIGGER DOES NOT**: report the trigger as fitted-only and refit on all families before any use | **TRIGGER IS OVERFITTED**; the vocabulary result is a property of three families, not of the method |
| **`0.40-0.59`** | **WEAK**: the trigger transfers but the retrieval it gates does not; investigate the families that fail before anything else | **WEAK on both**; one registered iteration, no build | **DOES NOT GENERALISE** |
| **`<0.40`** | impossible in practice; if it happens the instrument is wrong and the run is void | **DOES NOT GENERALISE** | **DOES NOT GENERALISE**: draft search is a property of the three families it was found on, and the lane's headline is retracted |

## What I already know

- Fitted actionable recall 10/14 (0.714); 27 of 46 retrieving drafts actionable.
- Fitted vocabulary trigger at `df<=2`: coverage 9/10 (0.90), `N_clean` false-trigger 0.167,
  suppression 0.674; it dominates the LLM gate on every axis.
- The four held-out families all HIT with goal queries in the original run, which is why they were
  never in the miss set.

## Confounds I can name now

1. **The holdout is not random**, as stated above, and the direction of that bias is named.
2. **A second judge is still a judge.** Agreement measures stability, not correctness; three models
   can share a blind spot, and the same-family pair is expected to.
3. **`N_clean` is still 18 git-flavoured drafts** and Part B does nothing to fix that. Prediction 6
   is a consistency check on the pipeline, not evidence about false triggers.
4. **`ts-false-zero-search` has only 13 drafts across 6 sessions**, so its per-family numbers are
   anecdote; the record will quote counts beside any family-level rate.
5. **Judging the held-out families uses the haiku judge**, whose stability Part A is measuring in
   the same run. If Part A fails, Part B's labels inherit the problem, and the record must say so
   rather than reporting Part B as though it were independent.

<!-- frozen_above -->

## Result (2026-08-27)

Positive control passed. Artifact: `benchmarks/artifacts/agent_ab/validation.json`.

### Part A — the label is UNSTABLE, and that is the headline

| judge | agreement vs haiku | Cohen's κ | implied actionable recall |
|---|---:|---:|---:|
| `claude-haiku-4.5` (the one everything used) | — | — | **10/14** |
| `claude-sonnet-5` (stronger, same family) | 0.630 | **0.327** | **6/14** |
| `gemini-2.5-pro` (different family) | 0.739 | 0.508 | **9/14** |
| sonnet vs gemini | — | 0.405 | — |

**Spread: 4 sessions.** The registered rule at `>= 3` reads:

> ⛔ **the label is unstable.** Actionable recall, the 27 labels and the re-derived trigger
> populations all become provisional, and no product decision may rest on them until a labelling
> protocol with measured agreement replaces the single judge.

Every κ is in the "fair to moderate" range for a label that is load-bearing, and **the strongest
judge is the strictest**: sonnet finds 6 of 14 where haiku found 10. So today's correction from
14/14 to 10/14 may not have gone far enough. **The honest statement of draft-time search's
actionable recall is now a range: 6 to 10 of 14, and a single-judge protocol cannot narrow it.**

### Part B — held-out families

| measure | fitted (3 families) | held out (4 families) |
|---|---:|---:|
| actionable recall | 10/14 = 0.714 | **15/24 = 0.625** |
| vocabulary trigger coverage at `df<=2` | 9/10 = 0.900 | **14/15 = 0.933** |

Per family, counts rather than rates (6 sessions each): `ts-autouse-tmp-path` 6/6 reached and 6/6
covered; `ts-bounded-runner` 4/6 and 4/6; `ts-separator-canary` 3/6 and 2/6;
`ts-false-zero-search` **2/6** and 2/6.

The registered grid cell is (recall `>=0.60`, coverage `>=0.80`) = **GENERALISES**.

⚠️ **But that cell may not be read as independent confirmation, exactly as this record required in
advance**: Part B's labels come from the haiku judge that Part A just showed unstable. Its recall
figure inherits the instability in full.

🔑 **One asymmetry rescues part of it, and it is a methodological point worth keeping.** Trigger
coverage is a RATIO whose numerator and denominator are both drawn from the same labels — of the
sessions judged reachable, how many does the trigger fire on. A systematic judge bias moves both
together and largely cancels. **Actionable recall has no such protection**: it is a count against a
fixed denominator of sessions, so a stricter judge lowers it directly. That is why coverage
transferred at 0.933 while recall fell to 0.625, and it means **the trigger result is materially
more robust to label instability than the recall result is.**

### Predictions

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | sonnet agrees 0.75–0.90 | **0.630** | **falsified, below** |
| 2 | gemini agrees 0.65–0.85 and LOWER than sonnet | 0.739, and **higher** than sonnet | **falsified on both clauses** |
| 3 | implied recall moves by at most 2 sessions | **4** | **falsified** |
| 4 | held-out recall 0.60–0.85 | **0.625** | confirmed |
| 5 | held-out trigger coverage 0.55–0.80 (some shrinkage) | **0.933**, no shrinkage | **falsified, above** |
| 6 | held-out false-trigger stays 0.167 | **not measured** — the probe computes no `N_clean` figure for the held-out families | **untested; my error in the script, not a result** |
| 7 | under 45 minutes, under 0.50 USD | ~15 minutes, ~230 calls | confirmed |

Two confirmed, four falsified, one untested. Prediction 2 is the instructive miss: I assumed a
different-family judge would decorrelate and therefore agree LESS, and the opposite happened —
the same-family stronger model was the outlier. **Capability, not family, was the axis that
mattered**, which is the reverse of the reasoning that chose the two judges.

### What this licenses, and what it retracts

- ⛔ **Retracted as a point estimate: "actionable recall is 10 of 14."** It is 6 to 10 of 14
  depending on the judge, and no number in that range is currently defensible over the others.
  Everywhere that figure appears it must carry the range.
- **The trigger comparison survives** — the vocabulary test dominating the LLM gate is a dominance
  relation between two candidates scored on identical labels, and it transferred to four unseen
  families at 0.933 coverage with the threshold fitted elsewhere. That is the most robust result
  in this lane.
- **Nothing is licensed for a build.** The prerequisite is now a labelling protocol with measured
  agreement — multiple judges with a resolution rule, or a small human-labelled gold set to
  calibrate against — and that is a registration of its own.
- **`ts-false-zero-search` at 2 of 6 is the family to look at first** if anyone continues: it has
  13 drafts across 6 sessions, the fewest of any family, and both the recall and the sample are
  small enough that the difference may be draft volume rather than difficulty.
