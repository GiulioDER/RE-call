# Pre-registration: the gold-set analysis plan, written before any label exists

**Date:** 2026-08-27   **Status:** predicted, instrument built, NOT yet labelled
**Instrument:** `scripts/agent_ab_build_gold_set.py` → `benchmarks/artifacts/agent_ab/gold/`

## Why this is registered now rather than after the labels arrive

The labels do not exist yet. That is exactly when the analysis must be fixed: once a human's
answers are in hand, every choice about which judge to compare, which metric to lead with, and how
to handle the uncertain items becomes a choice that can be made to favour a conclusion. This record
fixes all three before a single label is written.

## What the instrument is

Three judges on the same 46 items implied actionable recalls of 10, 9 and 6 of 14 (κ 0.33 to 0.51),
so no model label is defensible and the lane's recall figure is a range rather than a number. A
human settles it. **30 items** selected from the 46:

- **all 20 SPLIT items**, where the judges disagree. They carry the information about which model
  is right.
- **10 UNANIMOUS items**, 5 all-yes and 5 all-no, as controls for a SHARED blind spot. If a person
  disagrees with a unanimous verdict, all three models are wrong together — which no amount of
  inter-model agreement could ever reveal, and which is the failure this lane should fear most
  given how much of it has been model-labelled.

Blinding: the sheet carries no model verdict, no session id and no memo name; items are shuffled
with a fixed seed; the key is a separate file. The labelling question is the judges' question
**verbatim** — if it differs the comparison is void.

`?` is an allowed answer and is treated as data, not as a missing value: it means the item as shown
does not settle the question, which is itself a finding about the task.

## What I predict

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Gemini agrees with the human best**, on the split items, of the three judges | haiku or sonnet agrees more |
| 2 | **Haiku is the most permissive against the human**: its false-positive count (model yes, human no) exceeds its false-negative count by at least 3 on the split items | it does not, or the gap is under 3 |
| 3 | **Sonnet is the most conservative against the human**: its false NEGATIVES exceed its false positives by at least 3 on the split items | it does not |
| 4 | **On the 10 unanimous controls the human agrees with the models on at least 8.** A shared blind spot is the thing I most fear and least expect | 7 or fewer, which would be the most important result in this record |
| 5 | **The human answers `?` on 3 to 8 of the 30.** Fewer means the task is clearer than the models' disagreement suggests; more means the items lack context a labeller needs and the instrument is at fault | outside the band |
| 6 | **The human's implied actionable recall over the 14 sessions falls between sonnet's 6 and haiku's 10, inclusive.** If it falls outside that range, all three judges share a bias in the same direction | outside 6 to 10 |

## The analysis, fixed now

1. **Primary metric: agreement with the human on the 20 split items**, per judge, reported as a
   raw rate with the count beside it and Cohen's κ. The split items are the discriminating set;
   agreement over all 30 would be inflated by the unanimous controls, which is why the primary
   metric is stated as the split subset before the numbers exist.
2. **`?` items are excluded from agreement rates and reported separately with their count.** They
   are not counted as agreement with anybody.
3. **The unanimous controls are analysed on their own** as the shared-blind-spot check, never
   pooled into the primary metric.
4. **The winning judge is whichever has the highest κ on the split items**, and if two are within
   0.05 the result is reported as a tie and neither is adopted.
5. **The implied actionable recall under human labels replaces the range** only if the unanimous
   control passes at >= 8 of 10. Below that, the human labels are themselves suspect on the same
   axis and the record says so instead of crowning them.

## What this cannot settle

- **One labeller is not an inter-rater study.** A single human is a gold standard by convention,
  not by measurement, and this record does not pretend otherwise. If the answers disagree sharply
  with all three models, the honest next step is a second labeller, not a conclusion.
- **30 items is small.** Every rate will be quoted with its count, and a difference of one or two
  items is not a difference.
- **The labeller wrote the memos.** These are this project's own notes, so the labeller knows what
  each memo means better than any model does — which is the point, and also a source of hindsight
  bias about whether a hazard "really" applies. Named here because it cannot be removed.

<!-- frozen_above -->

## 🔁 Correction and supersession appended 2026-08-27, before any label was written

**Nothing above is edited.** Two things in this record are wrong, and the second retires it.

**1. The confound "the labeller wrote the memos" is FALSE.** The user states plainly: *"I didn't
write all the memos."* They were written by prior agent sessions. So the hindsight-bias confound I
named does not exist, and neither does the justification I built on it — that the labeller would
know each memo's meaning better than a model. **The opposite is closer to true**, which is why the
instrument met the honest answer it deserved: *"I can't label, I don't know the right answer."*

That inability is not a labeller failing. It is evidence that **"would this note's failure strike
this code" has no accessible ground truth by inspection**, which is a property of the question I
chose, not of the person asked.

**2. A judge-free, EXECUTABLE ground truth already existed in the archive and I did not look for
it.** Every session carries `metadata.check`, the result of running the agent's artifact against
the task's oracle, with trap-specific evidence. For the 14 registered miss sessions:

| outcome | sessions | evidence |
|---|---:|---|
| fell into exactly the trap the memo describes | **8** | 5 × `rewrote the file with 27 carriage returns against eol=lf`; 3 × `reached only 6 of 51 files over 120 seeds: head bias` |
| avoided the trap WITHOUT the memo | **5** | all `ts-worktree-import`: `reported this checkout's number with the decoy on the path` |
| failed before reaching the trap | 1 | `ts-lf-rewrite#r4`: `scripts/bump_version.py was not written` |

**So the memo was actually needed in 8 of 14 sessions, was demonstrably NOT needed in 5, and 1 is
indeterminate** — determined by execution, not opinion, and recorded months before any of today's
probes ran.

🔑 **This lands inside the judges' range (6 to 10) and settles it from outside.** It also supplies
what no judge could: **five objective NEGATIVES** — sessions where the agent avoided the hazard
unaided, so surfacing the memo would have been noise. Every negative population used today was
either 18 git commands or a retrieval-defined proxy.

**This record is SUPERSEDED.** The gold set is not built, the labels are not collected, and the
predictions above are left standing and unscored. The replacement is a re-derivation on checker
outcomes, which needs its own registration because its endpoint ("would draft search have surfaced
the memo in the sessions that actually needed it") is a different quantity from the one measured
today.

⚠️ **The lesson is the expensive part, and it is mine.** I reached for an LLM judge, then a second
and third judge, then a human labeller — three escalating attempts to manufacture a label — while
an executable oracle sat in the same archive I had been reading all day. **Look for a ground truth
that was RECORDED before reaching for one that must be produced.**
