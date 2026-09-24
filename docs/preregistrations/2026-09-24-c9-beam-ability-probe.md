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
