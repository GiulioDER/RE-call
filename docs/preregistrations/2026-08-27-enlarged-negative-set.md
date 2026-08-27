# Pre-registration: a real negative set, and what it does to the trigger result

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_repo_negatives.py`, committed with this record.

## The question

Every trigger number in this lane rests on **18 draft queries from four sessions of one task**,
all of them git commands. That set has already distorted one trigger's headline outright: the
operation-class trigger scored a false-trigger rate of 1.000 on it and 0.159 on the wide set,
because it fires on git commands by construction. The trigger record named the negative set as the
binding limit on the whole lane and the highest-value work remaining.

**Does the vocabulary trigger's 3-of-18 survive contact with a real negative population, and does
the ranking of the five triggers hold?**

## Where a HARD negative comes from

The ideal negative is code written **in this repository, in its style, touching its tooling**, that
nonetheless has no memo warning about it. Code from an unrelated project is an easy negative: it
fails to trigger for uninteresting reasons and would flatter every candidate.

So the population is **this repository's own git history**: added-line blocks from commits on
`origin/master`, which is real code of exactly the kind an agent writes here. Measured before
writing this record: 120 commits yield 1,407 blocks in the 40-to-4096 character band, median 180
characters.

## Design

- **Sample:** 200 added-line blocks from `origin/master`, Python files, 40 to 4,096 characters,
  drawn with a fixed seed over a recorded commit list so the sample is reproducible. Sampling
  spreads across commits so no single change dominates.
- **Ground truth by judge**, the same model and the same committed prompt as the precision run
  (`anthropic/claude-haiku-4.5`, temperature 0, actionable relevance rather than topical
  similarity). Each block is retrieved against `probe2_control` with `lexical_only` and its top-5
  judged. A block is:
  - **`N_repo`** (a true negative) if **0 of 5** hits are judged actionable;
  - **`P_repo`** (a true positive) if **>= 1** is.
- **`P_repo` is a bonus this record claims deliberately**: an independent positive population, not
  drawn from the 14 benchmark sessions of three task families, against which trigger coverage can
  be measured for the first time outside the benchmark.
- **Re-score all five triggers** (`T1` margin, `T2` vocabulary at each df, `T3` operation class,
  `T4` LLM gate, against `T0`) on `N_repo` and `P_repo`, reusing the committed implementations
  unchanged so the only thing that differs from the trigger record is the population.

## ⛔ Apparatus control: the judge IS the ground truth, so the judge is measured first

A labelling error here becomes a population error, silently. So before any trigger is scored:

**The judge must call the KNOWN governing memo actionable for the 14 benchmark miss sessions'
hazard-bearing drafts.** Those are cases where the correct answer is known independently — the
benchmark author declared the memo, and the draft demonstrably retrieves it. If the judge marks
fewer than **11 of 14** of them actionable, its labels are not trustworthy, **the run is VOID**, and
no trigger number is read from it.

This is the "control your own contribution" rule from
`[[a-null-is-the-cheapest-result-to-fabricate]]`: the retrieval control checks the input, and this
checks the instrument this record actually adds.

## What I predict

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Judge control passes: 12 to 14 of 14** known governing memos called actionable | below 11 (void) or the band is missed |
| 2 | **`P_repo` is 5% to 20% of the 200 blocks.** Most real code has no memo about it, which is the premise of the whole trigger question | outside the band |
| 3 | ⚠️ **`T2_vocab_df2`'s false-trigger rate RISES on `N_repo`, to 0.25–0.50** from 0.167. Eighteen git commands is a narrow slice; real repository code touches distinctive vocabulary far more often | outside the band |
| 4 | **`T3_operation`'s false-trigger rate FALLS sharply, to 0.30–0.60** from 1.000, because the git-command artifact disappears with the git-only negative set | outside the band |
| 5 | **The ranking holds: `T2` still dominates `T4`** on the Pareto pair (coverage, false-trigger) | `T4` dominates, or neither dominates |
| 6 | **Cost: under 40 minutes and under 0.60 USD** (200 retrievals, ~1,000 judge calls, ~70 more for the control) | either bound exceeded |

## Decision rule, with the SELECTION registered as a Pareto rule

Per `[[state-the-partition-over-the-cross-product]]`, whose third occurrence was a rule that sorted
candidates on one endpoint and so selected a dominated one. **The selection here is stated over the
endpoint pair before any number:**

> **Selected candidate** = the trigger on the Pareto frontier of (coverage on `P_repo`,
> false-trigger on `N_repo`) that maximises coverage subject to false-trigger <= 0.35. If that set
> is empty, the one that minimises false-trigger subject to coverage >= 0.60. If both are empty,
> the one maximising coverage, and the outcome is read in the `>0.70` column.

Bands for the selected candidate: coverage on `P_repo` (`<0.60`, `0.60-0.79`, `>=0.80`) crossed
with false-trigger on `N_repo` (`<=0.35`, `0.36-0.70`, `>0.70`). Nine cells:

| coverage \ ft | `<=0.35` | `0.36-0.70` | `>0.70` |
|---|---|---|---|
| **`<0.60`** | **DEMOTE**: gates well, misses too much of what it should catch | **KILL** | **KILL** |
| **`0.60-0.79`** | **BUILD the trigger**, preregister the live A/B with its cost | **GATE**: one registered iteration, on the selected candidate only | **KILL** |
| **`>=0.80`** | **BUILD**, and the trigger question is settled | **GATE**, operating point recorded | **KILL the trigger lane** |

## What I already know

- Trigger results on the 18-draft set: vocabulary df2 covers 10/14 firing on 3/18 and suppressing
  67%; the LLM gate covers 9/14 firing on 7/18; operation class fires on 18/18.
- Only 46 of 178 benchmark drafts are hazard-bearing.
- The judge, on the precision run, called 0.056 of negative-session slots actionable and 0.253 of
  the misses' slots, so it does discriminate; this record measures how well against known answers.

## Confounds I can name now

1. **The judge is the ground truth.** Prediction 1's control is the only thing standing between a
   judge error rate and a silently mislabelled population. It bounds false negatives (a governing
   memo called unactionable) and says nothing about false positives (an irrelevant memo called
   actionable), which would shrink `N_repo` and make triggers look better. Stated as the asymmetry
   it is; a follow-up could measure it by judging known-irrelevant pairs.
2. **`N_repo` blocks are added lines, not whole files**, so they lack surrounding context an agent
   would have. That makes them harder for the judge and for vocabulary matching alike.
3. **Commit history is not draft history.** Committed code is reviewed and fixed; a draft is
   pre-review. If anything this biases `P_repo` DOWN, since hazards were removed before commit,
   which is the safe direction for prediction 2 but means `P_repo` understates real draft risk.
4. **The corpus contains memos ABOUT this repository's commits**, so some blocks may retrieve a
   memo describing the very change they come from. That is a true positive for the trigger question
   and a strange one for the agent question; the record will count it and note it.
5. **200 blocks is a sample, not the population**, and the seed and commit list are recorded so the
   sample can be enlarged rather than re-drawn.

<!-- frozen_above -->

## Result (2026-08-27): VOID by the registered judge control, and the void is the finding

**The judge called only 6 of 14 known governing memos actionable, below the registered floor of
11. Per the rule above, the run stopped before sampling and no trigger number is read from it.**
The 200-block negative set was never built.

**The judge was right and the ground truth was wrong**, which is the opposite of what the control
was written to catch. Inspecting the eight disagreements:

| session | its memo-retrieving draft | chars | rank |
|---|---|---:|---:|
| `ts-lf-rewrite#r1` | `ls -la scripts/ \| head -20` | 26 | **1** |
| `ts-worktree-import#r1` | `ls -la benchmarks/` | 18 | 4 |
| `ts-worktree-import#r5` | `ls -la benchmarks/` | 18 | 4 |

⛔ **"The draft retrieves the governing memo" is not "the governing memo applies to this draft",
and that proxy has been the measure underneath this entire lane since the direction screen.** A
directory listing retrieves the CRLF memo at rank 1 because it shares the token `scripts`.

Structural extent, measured on the committed trigger artifact: **7 of 14 sessions have no
memo-retrieving draft longer than 200 characters**, and 15 of 46 retrieving drafts are shell
one-liners under 60 characters.

## Verification of the affected headline (`scripts/agent_ab_actionable_recall.py`)

All 46 memo-retrieving drafts judged, rather than the first per session, so the corrected figure is
measured rather than bounded:

| measure | value |
|---|---|
| retrieval-only recall, **as previously published** | 14/14 |
| **ACTIONABLE recall** (>= 1 draft where the memo truly applies) | **10/14** |
| memo-retrieving drafts where the memo actually applies | **27/46 (0.587)** |

The four sessions that fall away are `ts-lf-rewrite#r3` and `ts-worktree-import#r1/#r2/#r5`, each
of which had exactly one retrieving draft, and in three cases that draft is `ls -la benchmarks/`.

**What survives:** draft-time search still vastly outperforms goal-vocabulary search, 10 of 14
against 1 of 14, and the mechanism is unchanged for the ten sessions where a real operation
retrieves a memo that really applies. **What does not:** the 14/14 figure, and any statement built
on it.

⚠️ **Consequence for the trigger screen, stated rather than quietly absorbed:** its populations
were built on the same proxy — "hazard-bearing" meant "retrieves the memo", so 46 of 178 drafts
were labelled hazard-bearing when 27 are actionable. Its coverage numbers are therefore measured
against a population that is 41% mislabelled, and they need re-derivation against the judged labels
before any trigger claim is repeated. The false-trigger side is unaffected, because `N_clean` was
labelled by the benchmark rather than by retrieval.

## What this licenses

The enlarged negative set is still the right next step and is now **blocked on a prerequisite**:
the positive labels must come from actionability, not retrieval. The order is:

1. Re-derive the trigger screen's populations from `actionable-recall.json`'s judged labels.
2. Re-run this record's sampling with the judge control re-measured against the CORRECTED control
   set — that is, drafts where the memo is known to apply, not merely known to be retrieved. The
   floor of 11 of 14 was written against a control set that was itself 8/14 wrong, so the floor
   needs restating in a new record rather than reused.

**Neither is done here, and the void stands.** The run cost ~70 judge calls and about two minutes,
and it prevented a 200-block population from being built on a broken label.
