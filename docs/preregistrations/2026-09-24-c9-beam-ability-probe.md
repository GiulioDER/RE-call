# Pre-registration: where does served C9 lose BEAM's summarization, chain and temporal questions?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

The official AML Textual smoke scored 0 on B2 (causal chain and intermediate steps) and F1
(summarization and long-history synthesis), and 20 on C1 (dates), on one to three questions each.
AML forbids analysing its evaluation data, so this uses the public BEAM 100K split instead: 20
conversations, 90 dated batches, 1.8M words, 400 questions (40 per ability type). Input file
`beam100k.jsonl`, sha256 `62a95048...cd`, built by `benchmarks/beam/aml_c9_prep.py` from
`100K-00000-of-00001.parquet` (sha256 `c0519be2...ace`).

- **Service:** the official C9 on VPS2 (127.0.0.1:18015, `385c6074`), with one throwaway user per
  conversation and 6 workers. The user chose this, and the AML Full run waits for the result.
- **Path:** each batch is one Add; each question is one Search at `top_k` 100.
- **Answering and judging:** AML's public BEAM pipeline (answer prompt, batch rubric judge,
  event-ordering alignment F1 times Kendall tau), with Qwen3-14B for both answer and judge.
- **Assumed, not known:** the context is `content` in returned order, blank-line separated, with
  no `created_at`; and the platform's real judge model is unknown.

Three measures:

1. **`returned`:** the rubric score per type, answering in C9's own order.
2. **`chronological`:** the same 100 items re-ordered by batch, then by position. Run only for the
   five types below.
3. **`coverage`:** the judge scores whether each rubric point is present in the retrieved context
   at all. Run for the same five types.

The five types are summarization (F1), multi_session_reasoning and event_ordering (B2),
temporal_reasoning (C1), and information_extraction as a control.

## Predictions (mean rubric score, 0 to 1)

`[[i-over-predict-effect-magnitudes]]` records predictions too high by 2 to 4 times, and today's
concurrency run was the first to miss in the other direction. So these ranges are wide on purpose.

| type | returned | coverage | chronological minus returned |
| --- | --- | --- | --- |
| summarization | 0.10 to 0.40 (point 0.25) | 0.20 to 0.60 | +0.00 to +0.10 |
| multi_session_reasoning | 0.15 to 0.50 (point 0.30) | 0.30 to 0.70 | -0.03 to +0.05 |
| event_ordering (rubric) | 0.15 to 0.45 (point 0.30) | 0.30 to 0.70 | +0.02 to +0.15 |
| temporal_reasoning | 0.05 to 0.35 (point 0.15) | 0.20 to 0.60 | +0.00 to +0.10 |
| information_extraction | 0.40 to 0.80 (point 0.60) | 0.60 to 0.95 | -0.05 to +0.03 |

**Service health:** all 90 Adds stored, 0 graph and 0 atomic fallbacks, all 400 Searches return 200.

## How the result will be read

- **The reader misses:** if coverage minus returned is 0.20 or more for a type, retrieval found the
  evidence and the reader lost it. For that type, context order and format are the lever.
- **Retrieval misses:** if coverage is itself below 0.40, retrieval never returned the evidence.
  There the lever is coverage (diversification across sessions, neighbours, session summaries), not
  order.
- **Order matters:** if chronological beats returned by 0.05 or more on summarization or
  event_ordering, time order is worth testing as a Search-side change.

This cannot show what the platform's real reader and judge would score, and it runs one split. The
dates issue belongs to another session; this run keeps C9 exactly as served, so it includes that
handicap.

## Result

(appended below after the run; nothing above is edited)

### Run 1, 2026-09-24 20:18 to 21:31 UTC, official C9 `385c6074` on VPS2, 6 workers

This used `benchmarks/beam/aml_c9_probe.py` at `b32cfaaa`. The report is
`docs/results/2026-09-24-c9-beam-ability-probe-report.json`, and the raw outputs are archived on
VPS2 at `/root/c9-beam-probe-out-20260924.tgz`. All 20 users were deleted afterwards (322,052
rows). Model spend was $2.42, over 3,603 calls. Ingest spend (Voyage and the compiler) is not
metered here.

**Service health:**
- 90 of 90 Adds stored, none retried, the slowest in 187 s.
- **13 compiler fallbacks, which I did not predict.** 11 were compiles where every citation was a
  bare anchor index. That is fixed by #757, merged and deployed after this run.
- 400 of 400 Searches, all with 100 items, 0 graph fallbacks and 0 atomic fallbacks.

| type | returned | coverage | chronological minus returned |
| --- | --- | --- | --- |
| summarization | **0.304** (predicted 0.10 to 0.40, held) | **0.950** (predicted 0.20 to 0.60, falsified high) | **-0.027** (predicted +0.00 to +0.10, falsified low) |
| multi_session_reasoning | **0.550** (predicted 0.15 to 0.50, falsified high) | **0.923** (predicted 0.30 to 0.70, falsified high) | **-0.070** (predicted -0.03 to +0.05, falsified low) |
| event_ordering (rubric) | **0.225** (predicted 0.15 to 0.45, held) | **0.911** (predicted 0.30 to 0.70, falsified high) | **+0.101** (predicted +0.02 to +0.15, held) |
| temporal_reasoning | **0.290** (predicted 0.05 to 0.35, held) | **0.529**, 6 unscored (predicted 0.20 to 0.60, held) | **-0.132** (predicted +0.00 to +0.10, falsified low) |
| information_extraction | **0.831** (predicted 0.40 to 0.80, falsified high) | **0.838** (predicted 0.60 to 0.95, held) | **-0.167** (predicted -0.05 to +0.03, falsified low) |

The other types, answered in returned order only: abstention 0.500, contradiction_resolution
**0.063**, instruction_following 0.494, knowledge_update 0.525, preference_following 0.775.

Event ordering's alignment F1 times Kendall tau was 0.008 returned and 0.011 chronological.
That is near zero in both arms, because the answers are not the newline-separated event lists the
alignment splits on.

**Reading, by the rules written above:**

1. **The reader loses what retrieval found.** Coverage minus returned is 0.65 for summarization,
   0.37 for multi-session reasoning and 0.69 for event ordering, far past the 0.20 threshold. So
   for F1 and B2 the retrieved context held the evidence, and Qwen3-14B did not use it. Retrieval
   coverage is not the lever there. Caveat: coverage is an LLM judgement over a context of about
   20K tokens, and it has not been validated. A lenient judge would inflate it, so treat the gap as
   an upper bound on the reader's loss, not an exact figure.
2. **Temporal is the one retrieval-limited type.** Its coverage is 0.53, and 6 of 40 questions
   could not be scored. That fits the dates finding measured by the other session.
3. **Chronological order is not a global fix.** It helps event ordering by +0.10 and hurts
   everything else, including information extraction by -0.17. That pattern says the top of the
   context dominates what this reader uses: relevance order puts the best evidence first, and time
   order buries it. The next cheap, offline test is fewer items, re-answering from the same
   retrieval with the top 10, 20 and 40 instead of 100. The contract allows returning fewer than
   `top_k`. It needs its own pre-registration.
4. **Contradiction resolution at 0.06 is the lowest type**, and it was not in scope here. It
   belongs with D1 and D2 in AML's taxonomy.

What I got wrong: I under-predicted coverage on every in-scope type, by 0.3 to 0.4, and I expected
time order to help or be neutral. The misses run in both directions, so this record does not fit
the single over-prediction pattern.
