# Pre-registration: K-2 v2, same-subject adjacency by embedding similarity

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Successor to K-2 in
`docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md`, which as
registered **never fired** on LoCoMo: 0 of 720 questions reordered, because the best cross-day
subject-word Jaccard between 160-word windows reaches at most 0.250 (median 0.156) against a
threshold of 0.35 that I had fixed without looking at a window. That result stands and is not
revised here. This record replaces only the similarity signal and the way its threshold is chosen;
everything else about K-2 is carried over unchanged.

Baseline: served C9 at `3eb447c4` (content-only windows, `dated_search_content` on). Local
directional experiment over public data.

## The question

With the similarity signal taken from the retrieval embeddings and its threshold fixed on a
dataset that is never scored here, does placing same-subject, different-day items side by side,
newest first, raise BEAM contradiction_resolution plus knowledge_update against H, without lowering
overall accuracy or Coding MRR?

## What changes from K-2 v1, and what does not

**Changes.** Two items in the top 30 are linked when the cosine similarity of their document
embeddings is at least **τ** and they fall on different UTC days. Embeddings are Voyage
`voyage-code-4` document vectors, the space C9's primary index ranks in, so in production they
are read from the index for the top 30 ids and cost no API call at Search.

**Unchanged from v1.** Top 30 only; items past rank 30 untouched; different-day condition;
connected components capped at 4 members kept by link weight to the best-ranked member; each group
moved to its best member's rank, newest first; no text added, dropped or rewritten; default off
behind a flag.

## How τ is fixed, before anything is scored

τ comes from **LongMemEval**, which none of this record's measurements score:
`xiaowu0162/longmemeval-cleaned` at revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`.

1. **Questions:** all 78 `knowledge-update` questions in `longmemeval_s_cleaned.json`. Every one
   has at least two evidence sessions on different dates, each with a turn marked `has_answer`
   (checked 2026-09-25 on the oracle file: 78 of 78).
2. **Windows:** each session is rendered as C9 renders an Add (message contents joined, no role,
   no timestamp) and cut into C9's windows (160 words, stride 120) by C9's own window function.
3. **Positive pairs:** for each question, the window holding the evidence turn of its earliest
   evidence session paired with the window holding the evidence turn of its latest one. 78 pairs.
4. **Negative pairs:** for each question, 20 pairs of the evidence window against a random window
   from a non-evidence session on another date, plus 20 pairs of two random windows from two
   different non-evidence sessions on different dates. 3,120 pairs, drawn once with
   `random.Random(20260925)`.
5. **Embed** every window in those pairs with `voyage-code-4` in document mode.
6. **τ is the cosine at which the negative link rate is 0.25%** (the 99.75th percentile of the
   negative cosines). The per-pair rate is set that low because the top 30 holds about 400
   cross-day pairs, so 0.25% means about one spurious link per question; 2% would link almost
   everything, which is the failure v1's falsifier names.
7. **Stop rule at calibration:** if fewer than 20% of the 78 positive pairs reach τ, the embedding
   cannot separate same-subject from unrelated windows at that false-link rate, and K-2 v2 stops
   there: nothing is answered and nothing is spent on OpenRouter.

τ, the positive recall at τ and the negative distribution are committed before any LoCoMo or BEAM
mechanism metric is computed.

## What I predict

Written before any LongMemEval window is embedded. My effect predictions have run two to four times
too high (memory `i-over-predict-effect-magnitudes`), and I just set v1's threshold without data,
so the calibration bands are wide on purpose.

**Calibration (LongMemEval, no model call beyond Voyage):**

| Metric | Predicted |
|---|---|
| τ (voyage-code-4 cosine) | 0.70 to 0.90 |
| Positive pairs reaching τ | 0.20 to 0.50 |

**Mechanism (no model call):**

| Metric | Predicted |
|---|---|
| LoCoMo questions (the committed 720) whose top 30 K-2 v2 reorders | 0.40 to 0.85 |
| BEAM questions whose top 30 K-2 v2 reorders | 0.40 to 0.85 |

These sit high because of the calibration rule itself: at a 0.25% false-link rate, the roughly
400 cross-day pairs in a top 30 give about one spurious link per question, so at least one link is
likely (about 63%) even with no real same-subject pair. Windows from one LoCoMo conversation are
also more alike than LongMemEval's random sessions, which pushes the share up, towards the 0.90
falsifier below. A first draft of this table said 0.20 to 0.60, which contradicted the rule; it
was corrected before this record was committed (memory
`preregistered-predictions-must-agree-with-each-other`).

**Answers**, arms paired on identical retrieval, read and judged as in the T-1/K-2 record:

| Contrast | Set | Predicted |
|---|---|---|
| K2v2 − H | BEAM contradiction_resolution + knowledge_update (80) | **+0.02**, band −0.02 to +0.06 |
| K2v2 − H | BEAM contradiction_resolution alone (40) | +0.01, band −0.02 to +0.05 |
| K2v2 − H | BEAM 10-type mean | −0.005, band −0.02 to +0.01 |
| K2v2 − H | LoCoMo, the 720 | −0.3 points, band −1.5 to +0.5 |
| K2v2 − K0 | Coding K3 MRR | −0.01, band −0.04 to 0.00 |

The answer bands are v1's, unchanged. Fixing the mechanism does not lift the cap that matters most:
34 of 40 BEAM contradiction answers are a bare "Yes." or "No." under AML's "be direct and concise"
prompt.

## What would falsify this

- **Calibration:** positive recall at τ below 0.20 (the stop rule above).
- **Mechanism:** a reorder share above 0.90 on LoCoMo or BEAM, meaning τ links almost everything;
  or below 0.05, meaning it is inert again. Either stops the answer stage.
- **Answers:** K2v2 − H on BEAM contradiction plus knowledge update at or below 0.

## Decision rule

Unchanged from K-2 v1: recommend if K2v2 − H on BEAM contradiction plus knowledge update is at
least +0.03, LoCoMo 720 at least −0.5, BEAM 10-type mean at least −0.01, and Coding MRR at least
−0.01; any change to C9 needs an explicit user decision and never during an AML job.

## How it will be measured

1. **Calibrate** on VPS3 (steps 1 to 7 above); commit τ and the calibration report.
2. **Build** v2: `same_subject_adjacent` takes an optional vector per item and τ; the service reads
   the top 30 vectors from the primary index. Unit tests seen red against deliberate mutations
   before green, as for v1.
3. **Mechanism** on the LoCoMo draw (`results/aml-t1k2/locomo-draw.json`), embedding each item's
   stored window text with `voyage-code-4` in document mode (the stored retrieval keeps text, not
   vectors).
4. **Answers, LoCoMo.** If v2 is ready before the T-1 LoCoMo run starts, the K2v2 arm joins it,
   interleaved with H, H2 and T1. If not, K2v2 is answered in its own run with a fresh H3 arm
   interleaved beside it, and compared only with H3, never with an H answered in another session
   (memory `llm-reader-runs-drift-between-sessions`).
5. **BEAM** and the **Coding** guard as in the T-1/K-2 record: BEAM after the VPS2 hands-off rule
   is lifted (or a VPS3 re-collect), Coding after a re-collect that keeps item content.

**Spend.** Voyage only for calibration and item embeddings (a few thousand windows). OpenRouter only
for answers: the LoCoMo K2v2 arm is about USD 4 to 6 inside the T-1/K-2 cap, BEAM about USD 2.
Same pinned DeepSeek V4.1 Flash reader, the USD 40 balance floor and two workers.

## Apparatus checks, fixed now

1. The calibration report states the number of positive and negative pairs actually built, and
   refuses to fix τ if either count is below 90% of the plan (70 positives, 2,808 negatives).
2. Render-only on every stored row, as for v1: same multiset of items, identical contents, ranks
   31 to 100 unchanged.
3. The offline vectors are checked against the index where possible: for any item whose window
   text is in the VPS3 embedding cache written by the collect, the cached and recomputed vectors
   must agree to cosine at least 0.99.
4. Unit tests with red proofs for the vector path: a pair above τ on different days links; the
   same pair on one day does not; a pair below τ does not.

## What I already know

- K-2 v1: never fires on LoCoMo; its parameters were fixed without data (the v1 record's result
  section).
- Voyage query embeddings are not deterministic (memory `voyage-query-embeddings-are-not-deterministic`);
  document vectors recomputed here may differ slightly from the stored ones, which apparatus check 3
  bounds.
- Chronological order for every question cost BEAM temporal −0.13 and information extraction
  −0.17, and helped event ordering +0.10: reordering has a price when it is not gated.

## Confounds I can name now

- **Calibration transfer.** LongMemEval sessions are user and assistant chat about personal facts;
  LoCoMo is two-person chat; BEAM is long user and assistant conversations. A τ fixed on one may
  link more or less on the others; the mechanism shares are measured before anything is answered
  for this reason.
- **Negatives are not verified unrelated.** A random window from another session can be about the
  same thing, which pushes τ up and makes K-2 v2 conservative, never looser.
- **LongMemEval is one of AML's Textual sources.** It is used only to fix τ here, never scored,
  and τ is a single number.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

### Amendment 1, 2026-09-25, after apparatus check 1 refused and before any cosine was computed

The first calibration run (`243ff2af`) stopped at apparatus check 1, as designed, before embedding
anything: it built 67 positive pairs against a planned 78 (floor 70) and 2,680 negatives against
3,120. No vector and no cosine exists from that run.

The shortfall is the plan's error, not the harness's. Of the 78 knowledge-update questions, 11
cannot form a different-day positive pair at all: 6 are LongMemEval's abstention variants (`_abs`),
whose evidence sessions carry no marked answer turn by design; 2 have a marked turn in only one of
their two evidence sessions (`22d2cb42`, `eace081b`); 3 have both evidence sessions on the same day
(`618f13b2`, `a2f3aa27`, `0977f2af`). The oracle check that set "78 of 78" counted evidence sessions,
not marked turns on different days.

So the planned universe is now the **eligible** questions: at least two evidence sessions with a
marked answer turn, on different days (`eligible` in `scripts/aml_k2v2_calibrate.py`). That is 67,
and all 67 were built, with 40 negatives each (2,680). Apparatus check 1 now compares against the
eligible count, with the same 90% floor. The negative-link rate that fixes τ (0.25%), the stop rule
(positive recall below 0.20), every prediction and the decision rule are unchanged.

## Result, calibration (2026-09-25)

**Status:** τ fixed; LoCoMo and BEAM mechanism and answers not yet run.

`python scripts/aml_k2v2_calibrate.py` at `c7e04528` on VPS3, Voyage `voyage-code-4-v1` document
vectors, 3,800 windows. Report: `results/aml-k2v2/calibration.json` (and the refused first run,
`calibration-run1-refused.json`).

| Metric | Measured | Predicted | Gap |
|---|---|---|---|
| Positive pairs (eligible questions) | 67 of 67 | at least 90% | passes check 1 |
| Negative pairs | 2,680 of 2,680 | at least 90% | passes check 1 |
| **τ** (99.75th percentile of negative cosines) | **0.641** | 0.70 to 0.90 | **below the band** |
| Negative link rate at τ | 0.26% | 0.25% by construction | |
| **Positive pairs reaching τ** | **0.433** (29 of 67) | 0.20 to 0.50 | inside |
| Stop rule (positive recall below 0.20) | not met | | K-2 v2 continues |
| Median cosine, positives / negatives | 0.63 / 0.357 | | |

**What the gap means.** voyage-code-4 places unrelated chat windows lower than I expected (median
0.357, so the tail that fixes τ sits at 0.64, not 0.70 or more), and the same-subject windows at a
median of 0.63, straddling τ. The embedding separates the two groups well at the median, but at a
false-link rate low enough to leave most questions untouched, it catches 43% of same-subject pairs.
This τ is now fixed and is not revisited on LoCoMo or BEAM.

Next, per the record: build v2's vector path with red-proved tests, then the LoCoMo mechanism share
on the committed 720 before any answer.
