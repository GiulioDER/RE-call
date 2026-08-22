# Pre-registration: does an expanded query set certify the voyage:voyage-4 memory corpus?

**Date:** 2026-08-22   **Status:** predicted, not yet measured

## The question

Does roughly doubling the labelled **answerable** class lift the separability estimate's 95% CI
lower bound to 0.9 or above, so `gen_2aa437979700456c8e0f8a3e48888272` can be promoted?

Yes/no, decided by one number: `separability_ci[0] >= 0.9`.

## What I predict

**No. It will not certify.** I expect the lower bound to rise from 0.864 to roughly **0.88–0.91**,
straddling the bar rather than clearing it, with the point estimate roughly unchanged at
**0.93–0.95**.

The arithmetic behind that, stated now so it can be checked rather than rationalised later. The
current interval is [0.864, 1.000] around a point estimate of 0.9383, so the margin below the
estimate is **0.074**. Interval width falls about as `1/sqrt(n)`. To bring the lower bound to 0.9
with the estimate stuck at 0.938 the margin must fall to 0.038, a factor of ~1.95, which needs
**n roughly 3.8x larger — about 84 answerable queries, not 45**.

So doubling cannot succeed on n alone. It certifies only if the **point estimate itself rises**,
to about 0.96+, which requires the new queries to separate better than the existing ones, not
merely to be more numerous.

I am predicting against my own effort here on purpose: I am about to author these queries, and
[[i-over-predict-effect-magnitudes]] records eleven of twelve predictions falsified, every one too
high by two to four times.

## What would falsify this

- Lower bound reaches 0.9 or more at ~45 answerable queries. (Prediction wrong; promote.)
- Lower bound *falls* below 0.864, or the point estimate drops below 0.90. (Prediction wrong in the
  other direction: my authored queries are worse than the existing set, which would mean the
  expansion actively harmed the measurement.)

## How it will be measured

```
MEMORY_EMBEDDER='voyage:voyage-4' MEMORY_TENANT=memory \
  ./.venv/bin/python bin/calibrate_memory.py bin/memory_queries.voyage.json
```

on VPS2, against the live `memory` tenant (9,477 chunks, voyage:voyage-4).

- **Metric:** `separability`, and specifically the **lower bound of its bootstrap 95% interval**.
  The rate's denominator is the labelled query set: currently 22 answerable + 28 unanswerable = 50.
- **Target n:** ~45 answerable, 28 unanswerable retained unchanged.
- Secondary, recorded but not decisive: leave-one-out false-confident and false-abstain (currently
  0.179 / 0.136 under voyage, against 0.036 / 0.045 under bge-large).

**Authoring rule, fixed before writing a single query,** because the hazard here is fitting the
sample to the threshold:

1. Answerable queries are drawn from memos I can name **without searching**, one query per memo,
   phrased as a question a session would actually ask, never by quoting the memo's title.
2. I will write the whole set **once** and measure **once**. If it fails, the result is the result.
   No re-rolling queries and re-measuring until it passes.
3. The 28 existing unanswerable queries are kept verbatim. Changing both classes at once would make
   any movement unattributable.

## What I already know

Searched memory before predicting.

- [[calibrated-thresholds-and-the-overlap]]: measured 2026-08-17, **4 of 4 corpora overlap**
  (`min(answerable) - max(unanswerable)` negative every time: memory −0.048, re-call-code −0.007 and
  −0.032, mem-bench-code −0.097). **So overlap does not by itself prevent certification** — bge-large
  memory certified at separability 0.989 *with* −0.048 of overlap. What refused this run was the
  interval, not the overlap. My earlier framing of −0.209 as the disqualifying number was wrong.
- [[voyage-memory-generation-is-built-not-promoted]]: the run being repeated, and why the generation
  is parked.
- bge-large baseline on this same corpus and query set: threshold 0.71, separability 0.9886, LOO
  0.036 / 0.045.
- voyage first attempt: threshold 0.479, separability 0.9383, CI [0.864, 1.000], LOO 0.179 / 0.136,
  overlap −0.209, n = 22/28.

## Confounds I can name now

- **I author the queries and I want them to pass.** The authoring rule above exists for this. The
  one-shot constraint is the part that actually binds.
- **Author-side knowledge inflates answerability.** I have read many of these memos today, so my
  phrasing may sit closer to their wording than a real session's would, which would raise
  separability for a reason that does not transfer.
- **The corpus moved under the baseline.** The bge-large numbers were measured on 1,080 files; this
  runs against 1,214. A difference could be corpus growth rather than encoder.
- **`calibrate_memory.py` measures the live tenant, not the new generation.** Its own docstring says
  a calibration is bound to its corpus. So this certifies the *corpus*, and binding it to
  `gen_2aa43797` is a further step that must be checked, not assumed.
- **Chunk-level `relevant_ids` are not validated by this metric.** The measurement uses top cosine
  per query, so a wrong `relevant_ids` would not fail loudly. It is recorded for other uses.

---

## Result (2026-08-22)

**Status:** measured. **The prediction held: it did not certify.**

```
NOT certified - separability 0.937 clears 0.9 but its 95% interval
[0.883, 0.992] does not: 45/28 samples cannot establish the bar.

answerable   n= 45  min=0.373  p25=0.533  med=0.600  max=0.718
unanswerable n= 28  min=0.254  p75=0.452  med=0.412  max=0.582
separation (min answerable - max unanswerable): -0.209
threshold: 0.4760   separability: 0.937   ci: (0.8827, 0.9919)
leave-one-out: false-confident 0.214   false-abstain 0.133
```

| | predicted | measured | |
|---|---|---|---|
| certifies? | **no** | **no** | right |
| CI lower bound | 0.88 to 0.91 | **0.8827** | inside the range |
| point estimate | 0.93 to 0.95 | **0.937** | inside the range |

**Gap: essentially none, which is itself the finding.** The interval narrowed exactly as the
`1/sqrt(n)` model in the prediction said it would. Predicted shrink factor `sqrt(45/22) = 1.43`;
measured `0.074 / 0.054 = 1.37`. The point estimate did not move (0.9383 to 0.937), so the
certification failure was never about sample size alone, and doubling the class bought 0.019 of
lower bound.

**Extrapolating on the same model, now with two points instead of one: reaching a lower bound of
0.9 needs the margin to fall from 0.054 to 0.037, a further factor of 1.46, so n x 2.13 - about 96
answerable queries.** The pre-registered estimate was ~84 from a single point; 96 is the better
number and both say the same thing, which is that this is not a labelling problem you finish in an
afternoon.

⛔ **The result that should stop the next attempt: leave-one-out false-confidence got WORSE, 0.179
to 0.214.** More queries, a worse error rate. My 23 authored queries do not separate better than
the original 22 - they separate slightly worse. So the point estimate will not climb toward the
0.96 it needs by adding more of the same, and the honest reading is that **voyage:voyage-4 does not
separate this corpus as cleanly as bge-large did** (0.937 against 0.9886, LOO 0.214/0.133 against
0.036/0.045), rather than that the query set is merely small.

**Confounds, revisited against the result.** The "author-side knowledge inflates answerability"
confound predicted my queries would score too easily. The opposite happened, which retires that
worry and raises a different one: phrasing a question the way a session actually asks it is
genuinely harder for the encoder than quoting a memo. That is the realistic case, so 0.214 is
probably the more honest error rate, not a pessimistic one.

**Decision: `gen_2aa437979700456c8e0f8a3e48888272` stays `ready` and unpromoted.** Nothing changed
about the corpus, the active generation, or what search returns today.
