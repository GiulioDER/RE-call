# Pre-registration — the Answerability Ladder, v2

**Date:** 2026-07-29 · Written **before** the v2 manifest is built and before any v2 arm is run.
The git history of this file is the evidence.

**This is a new pre-registration, not an edit to v1.** `PREREGISTRATION-ladder.md` and its addendum
stand unchanged, and the v1 result is published as a result — see `results/ladder/H1_VERDICT.md`.
Editing v1 to make it look like it had asked the right question is the exact failure a
pre-registration exists to prevent.

**Prior work searched before this design was fixed** — `docs_search(source_type="memory")` on
"distractor corpus far gap versus near miss abstention regime unrelated documents present" and on
"BEAM benchmark evaluation results answerability abstention". The load-bearing hit is
[[project-recall-abstention-bounded-domain-2026-07-24]], and **it changes what this file can
honestly claim to test**:

> Far gaps — works. Unanswerable queries genuinely off-topic: abstention accuracy **1.00** (PEPs),
> **0.89** (private corpus); the cosine distributions are disjoint. Near-miss — fails: false-abstain
> 0.481 while retrieval hit@5 was 0.970.

Both **endpoints of the v2 ladder are therefore already measured**, on two corpora, by two
harnesses. `r = 0.00` is the near-miss regime known to fail; `r = 1.00` is the far-gap regime known
to work. So P1 below is **not a discovery — it is a positive control**, and it should be read as
one: if it fails, the harness is broken, not the field's understanding. Six candidate abstention
signals were also already measured there and all failed, so v2 does not propose a seventh.

**The genuinely untested question, and the only reason to run v2, is the SHAPE between those two
endpoints.** Nothing in the prior work interpolates them. P2 is that question.

## 0. Why v2 exists: v1 measured the wrong thing, and we can say precisely what

v1's ladder excised documents from a question's own conversation, and **ingest scope was that one
conversation**. So its top rung, `d=max`, removed every document there was. Measured from the
manifest and corpus before any verdict was computed:

| v1 rung | median surviving documents in the ingested slice |
|---|---|
| original | 629 |
| d=0 | 628 |
| d=4 | 624 |
| d=16 | 612 |
| d=64 | 564 |
| **d=max** | **0** |

Two defects, both in the ladder rather than in any system under test:

1. **`d=max` was an empty corpus.** A system that abstains there has not recognised an
   unanswerable question; it has nothing to retrieve. v1's pre-registered contrast was `d=0` vs
   `d=max`, so its headline could report PASS for a system whose only abstention trigger is an
   empty index.
2. **The other rungs were near-duplicates.** Absolute widths of 0/4/16/64 against a median cluster
   of 629 turns remove at most ~10 % of the topic. The powers-of-four ladder was calibrated for
   clusters an order of magnitude smaller than LOCOMO's conversations.

The v1 arm confirmed the consequence: abstention was ~0 at every rung whose corpus was non-empty,
and 1.0 at the rung whose corpus was empty. **A step at the cliff, flat everywhere else.**

## 1. What changes, and why each change is forced by the diagnosis

**Change A — the ingested slice contains distractor conversations.** Each question is scored
against its own conversation **plus 2 other LOCOMO conversations**, chosen by a fixed seeded rule
and identical across every rung of that question. This is what makes the top rung meaningful: with
the question's own conversation fully excised, the index still holds ~1 300 turns of real
conversational text that simply does not contain the answer. That is the far-gap regime the design
described and v1 could not reach.

**Change B — widths are fractions of the question's own cluster, not absolute counts.** Rungs are
`r ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`, where `r` is the fraction of the question's conversation
excised **in addition to gold** (gold is always excised, at every rung). `r = 0.00` is gold only —
the topic intact, one fact absent, the BEAM regime. `r = 1.00` is the whole conversation gone with
distractors remaining — the far-gap regime. A fraction cannot be mis-scaled by corpus size, which
is the error v1 made.

**Unchanged from v1**, deliberately: the excision-ordering function (BM25 over the whole corpus,
ties by `doc_id` ascending), the frozen-manifest release model, mechanical labels, the paired
design, λ ∈ {1, 3, 10}, and shipped defaults for every system.

## 2. Fixed parameters

- **Rungs:** `r ∈ {0.00, 0.25, 0.50, 0.75, 1.00}` as defined above. Five rungs plus the answerable
  original, so six instances per question.
- **Distractors:** exactly 2 conversations per question, drawn from the 9 that are not its own,
  selected by `random.Random(0)` keyed on the question's `cluster_id` so every question in a
  conversation gets the same pair, and the choice is reproducible from the manifest alone.
- **Question sample:** 200, seed 0, drawn after sorting by `question_id` from the 1 533 usable.
  Fewer than v1's 300 because each state now indexes ~3 conversations instead of 1; with 200 pairs
  the standard error of a paired mean over deltas in {−1, 0, +1} is at most 1/√200 ≈ 0.071, so the
  0.15 effect below still sits at ~2.1 SE.
- **λ ∈ {1, 3, 10}.** Fixed here. Choosing λ after seeing results is forbidden by this file.
- **Tie-breaking:** equal BM25 scores rank by `doc_id` ascending.

## 3. Predictions, committed now

These are written knowing v1's outcome, which makes them *more* constrained, not less — v1 has
already ruled out the comfortable answer.

- **P1 — a POSITIVE CONTROL, not a finding.** Correct-abstain at `r = 1.00` exceeds `r = 0.00` by
  **more than 0.15**, with a bootstrap 95 % CI on the paired difference excluding zero. Prior work
  already measured both endpoints (far-gap 1.00/0.89, near-miss failing), so a PASS here confirms
  the harness reproduces a known result and **must not be reported as a discovery**. A FAIL means
  the harness is broken; it is still the kill condition, but its diagnostic meaning is inverted
  relative to v1's P1.
- **P2 — and this is the prediction I expect to lose.** The curve is **not** monotone-graded; it is
  a **step**. Specifically, correct-abstain at `r = 0.75` is closer to `r = 0.00` than to
  `r = 1.00` (difference from `r=0.00` below 0.15). v1's evidence — abstention ~0 with 90 % of the
  topic removed — predicts that RE-call's guard engages only when the topic is essentially gone.
  **If P2 holds, the honest headline is that abstention is a cliff, not a curve**, and the
  benchmark's contribution is locating the cliff rather than drawing a gradient.
- **P3.** False-abstain on the answerable originals is **below 0.10**. v1's own pre-registration
  predicted above 0.30, from a LongMemEval measurement of 0.481, and v1 measured essentially zero.
  This file predicts the v1 observation, not the prior literature, and records that v1's P3 was a
  clear miss.
- **P4.** Rebuilding rings with a random-within-cluster neighbour function preserves the sign and
  rough magnitude of P1. If it does not, BM25 ordering is a confound and no curve ships.

## 4. What would falsify the whole approach

P1 failing. Not "a disappointing result" — the retirement of both ladders. If a system does not
abstain more when the answer's entire topic has been replaced by unrelated conversation than when
only one fact is missing, then excision distance does not move abstention, and the axis is a
fiction.

There is a second, subtler failure this file names in advance so it cannot be explained away
later: **if correct-abstain at `r = 1.00` is high but false-abstain on the originals is also
high**, the system is not discriminating — it is simply abstaining more as the index shrinks, and
the ladder is measuring index size under a different name. Both numbers are reported together, and
the write-up must state their relationship rather than the first alone.

## 5. Known to cut against us

- v1 found RE-call's abstention essentially binary on this corpus. v2 may simply relocate the cliff
  rather than reveal a curve; P2 predicts exactly that.
- The arm runs with **no calibration for `bge-small`**, so abstention uses the untuned 0.50 cosine
  floor. That is the correct configuration under the suite's shipped-defaults rule, and it is also
  the exact constant already measured as not comparable across embedders. Every v2 number inherits
  that caveat.
- Distractor conversations are still LOCOMO conversations. They are unrelated to the question's
  topic but share genre, register and speaker style, so `r = 1.00` is a *far gap within one
  domain*, not a far gap in general.
- LOCOMO's evidence labels share an annotation pass with its answer key, 6.4 % of which is wrong.

## 6. What v2 does NOT measure

Whether an *answered* question was answered **correctly** — v2 still has no judge, so every
accuracy is an upper bound. Also unchanged: extraction quality, multi-hop reasoning, currency,
attribution, cost and latency.

---

## Addendum, 2026-07-29 — a pre-run diagnostic, and one recorded field

Appended before the v2 arm ran. **No prediction, threshold, rung, sample or λ above is changed.**
The headline remains shipped defaults. What follows is a diagnostic run on 3 questions and a
decision about what the harness *records*.

### The diagnostic

The v2 smoke run answered **every** question at every rung, including `r = 1.00` where the
question's whole conversation is gone and only distractors remain. Probing the underlying scores:

| rung | top-1 cosine (min / median / max) | shipped 0.50 floor fires |
|---|---|---|
| `r = 0.00` — topic intact, one turn removed | 0.6494 / **0.7319** / 0.7370 | 0 / 3 |
| `r = 1.00` — own conversation gone, distractors only | 0.5752 / **0.5997** / 0.6359 | 0 / 3 |

Two things follow, and they point in opposite directions:

1. **The signal is real.** Removing the entire topic moves top-1 cosine by ≈ 0.13 in the correct
   direction. The axis this benchmark was built to price *does* exist in the underlying scores.
2. **The shipped threshold cannot express it.** At `r = 1.00` the top-1 cosine is still ~0.60,
   comfortably above the 0.50 floor. The floor fires nowhere, at any rung. This is
   [[project-recall-threshold-embedder-fragile-2026-07-28]] exactly: 0.50 sits at the 0th
   percentile of five of six embedder distributions.

**So P1's outcome is predetermined, and this file says so before the run rather than after.** The
shipped-defaults arm will abstain essentially nowhere, and P1 — already demoted to a positive
control — will FAIL. Its pre-registered reading was "a FAIL means the harness is broken". That
reading is now **corrected in advance**: the harness is fine and the axis is fine; the *shipped
threshold* is below the entire score distribution on this embedder.

### The one change: record `top_cosine`

The harness now records each response's top-1 cosine alongside its abstain/answer decision.
Nothing about the pre-registered metric changes — abstention is still RE-call's own shipped
verdict, and the headline is still computed from it.

The reason is that abstention at any threshold *t* is exactly `top_cosine < t`. Recording the
score makes **the entire threshold sweep computable from one arm**, post hoc and free, instead of
requiring a separate multi-hour run per candidate threshold. A run whose outcome is already known
becomes a run that measures the whole family.

**Guard against the obvious abuse:** a threshold chosen after seeing which one maximises the
effect is not a result. The sweep is reported as a **curve over all thresholds**, never as a
best-threshold headline, and the shipped 0.50 is always marked on it. Any specific alternative
threshold would need its own pre-registration and its own held-out arm.
