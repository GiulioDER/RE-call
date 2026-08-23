# Pre-registration: LoCoMo n=49 dress run on the Bench'd champion configuration

**Date:** 2026-08-23   **Status:** predicted, not yet measured

## The question

What does the LongMemEval-tuned champion configuration (A6: voyage-4 dense + lexical,
rerank-2.5, DeepSeek v4 pro synthesis with extended thinking off, top_k=10, session
granularity, threshold 0.0 suppressed) score on the LoCoMo n=49 stratified sample, the exact
cell their leaderboard rows (LlamaIndex 54.8, LLM baseline 50.4) were measured on?

## What I predict

- Nuance overall **58 to 68**: above their leader by a smaller margin than on LongMemEval,
  because the config was tuned on LongMemEval and LoCoMo's multi-hop and open-domain questions
  are harder for a digest step, but LoCoMo carries session dates the adapter surfaces, so
  temporal should not crater.
- Ingest happens at most 10 times (one per conversation) despite 49 items: the tenant cache
  turns the interleaved sampler's re-ingests into cache hits. Wall time under 90 minutes.
- Mean recall tokens stay under 100.

## What would falsify this

A score at or below their LLM baseline's 50.4 (the memory layer would be adding nothing on
this benchmark), or more than 10 conversation ingests (the tenant cache failed), or a
crater specifically on temporal despite dates being available (the session-date headers are
not reaching the digest).

## How it will be measured

```bash
benchd run -a re-call -b locomo-v1 -n 49 --judge --key ./keys/private.key
```

Same harness, seed-42 stratified sample, A6 env knobs, local session database, adapter at
commit HEAD (tenant-cache rewrite). Metrics: nuance overall and per dimension from the signed
manifest, ingest count from tenant count, cost from the OpenRouter meter and the token counter.

## What I already know

LongMemEval slice results in `2026-08-23-benchd-tuning-arms.md` (champion 75.0 on n=60). Their
LoCoMo cell is n=49, so ±1 question is ±2 points. LoCoMo category 5 is excluded by their
loader; every scored question is answerable.

## Confounds I can name now

- Cross-benchmark transfer: the config was selected on LongMemEval; a LoCoMo-specific
  weakness (long multi-hop chains) would look like general weakness.
- The n=49 sample is small; per-dimension counts are tiny (single digits for some types).
- LoCoMo conversations are ~10x larger corpora than LongMemEval oracle items, so top_k=10
  covers a far smaller fraction; the hit-rate regime differs from the tuning slice.

## Result (2026-08-23)
**Status:** measured

Measured: **89.8** (44/49), against their leaderboard cell's LlamaIndex 54.8 and LLM baseline
50.4. Per dimension: temporal 10/10, reasoning 27/30, recall 7/9. Exactly 10 conversation
ingests across 49 items (tenant cache verified), 0 synthesis fallbacks, mean recall 33.1
tokens, 36.9 tokens per correct (their leader: 37.7), mean latency 4.2s. Estimated BMI 92.8
against their 68.3.
Predicted: 58 to 68.
**Gap: under-predicted by 22 points.** The prediction assumed cross-benchmark transfer loss;
instead LoCoMo plays to the champion config's strengths, because its session dates are
available to the adapter (unlike LongMemEval, where the loader drops them) and the digest step
does explicit date arithmetic with them: temporal went 10/10 where the tuning slice's temporal
was the weakest dimension. The five failures are three "likely yes/no" inference questions the
digest declines to speculate on and two partial-detail recalls.
