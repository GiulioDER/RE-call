# Pre-registration: corroboration as a second admission axis for near-miss abstention

**Date:** 2026-08-19   **Status:** predicted, not yet measured

Origin: `docs/AGENT_MEMORY_FIELD_REVIEW.md`, item 1. The idea is not mine. It is Rule 4 of a
dev.to piece (`israelhen153`, "Agent memory v2, seven rules after the poisoning", 2026-06-23):
confidence is allowed to order a pile of unverified claims, and is not allowed to promote any of
them, because the hallucination that started that incident was itself high confidence. Promotion
has to come from independent corroboration, and sources are ordered by kind rather than by score.

That is an argument, from someone who has built none of it, that a scalar score cannot do this job.
FINDINGS §10b is the same conclusion reached here by measurement. This record tests whether the
remedy that argument implies survives contact with the numbers.

## The question

On the LongMemEval abstention split, does a corroboration count (how many independent sessions in
the retrieved pool support the top hit) separate answerable from unanswerable questions better than
the dense cosine already shipping, measured as Mann-Whitney AUC over the same 500 questions?

## What I predict

**I predict a null, and I am writing this down because the null is the likely outcome and would
otherwise be reported as a disappointment rather than as the result it is.**

1. **Primary.** `corroboration_count` AUC lands in **0.55 to 0.78**, with the point estimate more
   likely below `dense_top1`'s 0.753 than above it. I do not expect it to reach 0.826, the upper
   bound of the best signal already tested.
2. **Secondary.** A two-signal rule combining `dense_top1` with `corroboration_count` achieves
   held-out balanced error **no better than 0.28**. The shipped rule scores 0.305 and the in-sample
   ceiling for a threshold alone is 0.285, so I am predicting the combination buys at most a
   fraction of a gap that is itself in-sample.
3. **Tertiary, and the only one that would justify the work.** Restricted to the overlap band where
   the classes actually collide (top-1 cosine in [0.61, 0.78], which covers the answerable q05 to
   q75 range and almost the whole unanswerable range), `corroboration_count` AUC is **at or below
   0.65**. This is the conditional question. A signal that only separates where cosine already
   separates has added nothing.

Reasoning behind the null: corroboration is computed from chunk-to-chunk similarity inside a pool
that was itself selected by query-to-chunk similarity, so the two are correlated by construction.
The three relevance signals already tested cluster at 0.739 to 0.753 despite being structurally
different (bi-encoder, cross-encoder, RRF hybrid), which is evidence that this family has a ceiling
near 0.75 on this class, and corroboration is arguably a fourth member of that family rather than a
departure from it.

## What would falsify this

Any one of these:

- `corroboration_count` AUC point estimate **above 0.826**, which would place it outside the
  interval of every signal measured so far.
- Two-signal held-out balanced error **at or below 0.26**.
- Overlap-band AUC **above 0.70**, which would mean it carries information exactly where cosine
  does not, even if its unconditional AUC is unremarkable.

A fourth outcome is neither confirmation nor falsification and must be reported on its own terms:
**AUC significantly below 0.50**, meaning corroboration fires *more* on unanswerable questions. That
is plausible here (see confound 3) and it is informative. It must be reported as an inverted signal,
with the direction stated. It must **not** be silently reported as `1 - AUC` and called a success.

## How it will be measured

**Signal definition, fixed here so it cannot drift to fit the number.** For one query, retrieve the
candidate pool as the shipped pipeline does. Take the top-1 chunk. `corroboration_count` is the
number of **distinct LongMemEval sessions**, other than the top-1 chunk's own session, contributing
at least one chunk to the pool whose **chunk-to-chunk** cosine against the top-1 chunk is at or
above tau.

- The unit is the **session**, not the chunk, because chunk count is a function of chunking
  granularity and would let a finely split session corroborate itself.
- The session is LongMemEval's natural write-cohort unit and stands in for the write-event identity
  RE-call does not currently record (field review, item 6).
- **tau is fixed a priori at 0.75** and is not tuned on this 500-question set. Fitting tau here
  would be an in-sample fit, the exact defect FINDINGS §2b retracted a number for. If tau turns out
  to matter, that is a separate record with a disjoint split, not an edit to this one.
- No LLM call anywhere in the signal. Preserving "no memory-layer LLM calls" is a measured,
  published advantage and this experiment must not spend it.

**Dataset and n.** LongMemEval `longmemeval_s_cleaned`, the same 500 questions and the same
per-question haystacks as FINDINGS §10b: **n = 500, of which 470 answerable and 30 unanswerable.**
The 30 is the benchmark's own abstention-class size and cannot be enlarged.

**Metric, by name.** Mann-Whitney AUC of the two calibration classes, with the 95% interval from
`recall.calibration.separability_interval` (Hanley and McNeil 1982, over both classes). The rate in
prediction 2 is **balanced error**, the mean of false-abstain (denominator: the 470 answerable) and
false-confidence (denominator: the 30 unanswerable). Both denominators are stated because a rate is
named by its denominator.

**Harness.** Extends `recall/eval/longmemeval.py` and the per-question path in
`recall/eval/longmemeval_perq.py`, adding one signal alongside the six already scored in
RESULTS.md §8. Regeneration path for the existing arm:

```bash
python -m recall.eval.longmemeval --dataset longmemeval_s_cleaned.json --out ./s_out
python -m recall.eval.labelled --corpus ./s_out/corpus --questions ./s_out/questions.json
python -m recall.eval.longmemeval_perq --questions ./s_out/questions.json --master <indexed-table>
```

**Apparatus verification, before any new number is believed.** Exit code 0 is not a measurement.

1. **Reproduce `dense_top1` = 0.753 [0.680, 0.826] through the modified harness.** If the existing
   signal does not come back at its published value, nothing measured alongside it means anything,
   and that is the stop condition.
2. **Singleton assertion.** On a synthetic corpus where the answer-bearing chunk appears in exactly
   one session, `corroboration_count` must be 0. A counter that never returns 0 is measuring pool
   size.
3. **Far-gap non-regression.** On the PEPs set, where abstention already scores accuracy 1.00,
   adding the signal must not reduce it. A near-miss fix that breaks the far-gap case is a
   regression wearing a new name.

## What I already know

Searched before predicting, so this is not a re-measurement of something settled.

- **FINDINGS §10b**: false-abstain 0.481 on the comparable arm; top-1 cosine AUC 0.753 over
  n=500 (470 / 30); the best in-sample threshold scores balanced error 0.285 against the shipped
  rule's 0.305; driving false-abstain to 0.05 costs false-confidence around 0.78. Recalibration was
  ruled out by measurement.
- **RESULTS.md §8**: the six signals and their intervals. `dense_top1` 0.753 [0.680, 0.826],
  `rerank_top1` 0.742, `hybrid_top1` 0.739, `entail_max` 0.648, `margin_1_5` 0.579, `ratio_1_5`
  0.545. Nothing reaches the roughly 0.90 a usable gate needs.
- **The recorded conclusion is an exclusion, not a shrug**: the bar sits outside the best signal's
  interval. Any new signal is therefore competing against a closed question, and the burden here is
  high on purpose.
- **FINDINGS §5**: the QNLI judge's residual near-miss false-confidence of 0.50 transferred exactly
  to a corpus it had never seen. Answerability judges have been tried; this is not that.
- **Field review Part 2b, measured 2026-08-19**: zero of 152 memos in RE-call's own memory store,
  and zero of 59 documents in `docs/`, declare `valid_from`, `valid_until` or `supersedes`. This is
  why the experiment runs on LongMemEval and not on the dogfood corpus: only LongMemEval supplies a
  usable independence unit today.

## Confounds I can name now

1. **Correlation with the selecting signal.** The pool is chosen by query-to-chunk similarity, so
   chunk-to-chunk similarity inside it is not independent evidence. This is the reason for the
   predicted null and it is also the reason prediction 3 (the overlap band) is the one that matters.
2. **The interval, not the point, is the binding constraint.** With 30 unanswerable samples the
   AUC interval is roughly ±0.07. To exclude the current best I need a point above 0.826; to
   establish the roughly 0.90 bar I need a *lower bound* at 0.90, which needs a point near 0.94.
   A genuinely useful signal could easily land unestablishable, and that outcome must be reported
   as unestablished rather than as promising.
3. **LongMemEval's unanswerable class is near-miss by construction**, and each haystack is one
   user's own sessions. Topically related sessions therefore exist for unanswerable questions too,
   and may exist in greater number precisely because the topic is present while the answer is not.
   This is the mechanism that could invert the signal, and it is why the inverted case is written
   into the falsification section in advance.
4. **tau at 0.75 is an unvalidated constant.** It is fixed a priori to avoid an in-sample fit, which
   means the measurement is of tau=0.75 specifically and not of corroboration in general. A null
   here does not exclude a different tau; it excludes this one, and the record must say so.
5. **Chunking granularity** varies across sessions, so the session-level de-duplication is doing
   load-bearing work. If it is implemented wrongly the signal degenerates into pool size, which
   correlates with nothing useful. Apparatus check 2 exists for this.
6. **This tests a proxy for the article's claim, not the claim.** Rule 4 orders sources by *kind*
   (a tool result outranks a confident model, user confirmation outranks both). A count of similar
   sessions has no notion of source kind at all. A null here is evidence against *this cheap
   proxy*, and is not evidence against provenance-ordered corroboration, which RE-call cannot test
   until it records write-event identity and source kind.
