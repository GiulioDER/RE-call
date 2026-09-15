# Preregistration: post #629 graph answer quality replay

Date: 2026-09-12

Status: locked before the first answer generation request

## Objective

Measure whether the evidence retrieved by the deployed graph one hop arm produces better
generated answers than graph off when the answer model, prompt, token ceiling, temperature, and
pairing are fixed. Retrieval is not rerun in this phase.

## Frozen inputs

1. Retrieval artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-performance-post-629.json`.
2. Retrieval artifact SHA256: `bb6e62e575ee15433d6a3856500b0b72924ab02c2393c6ae93aab76ec5169869`.
3. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
4. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
5. Serving source commit: `3b628422b0f367b4f10201dccd2005df9c53ed8d`.
6. Arms: `off` and `one_hop`.
7. Pairing unit: query identity, using recorded pass 1 from each arm.
8. The prompt receives only the query and trusted evidence bundle from the retrieval artifact.

## Answer provider

OpenRouter chat completions with model `deepseek/deepseek-v4-flash`, temperature zero, reasoning
effort `none`, JSON response format, and a 512 token maximum. The API key is read from the
environment and is not written to the artifact. Model identity, provider latency, token counts,
prompt digest, and failures are recorded.

## Outcomes

Report separately for each arm and as paired one hop minus off deltas:

1. Valid answer envelope rate.
2. Nonempty answer rate.
3. Correct abstention rate on labelled unanswerable queries.
4. Citation validity rate.
5. Gold citation precision and recall.
6. Any gold citation rate and complete gold citation coverage.
7. Provider failure rate, latency, prompt tokens, completion tokens, and total tokens.
8. Query level gains, losses, and unchanged cases.

The model is not a factual judge. This run must not claim factual correctness. A fixed judge or
human review is a separate follow up using the preserved answers and the same judge for both arms.

## Decision rule

The replay is complete only if provider failures are at most two percent in each arm. The quality
result is nonregressing only if all of the following hold:

1. Valid answer envelope rate does not drop by more than two percentage points.
2. Citation validity does not drop by more than two percentage points.
3. Gold citation recall does not drop by more than two percentage points.
4. Correct abstention does not drop by more than two percentage points.

Any positive citation or answer result is evidence quality, not factual answer correctness. No
rollout recommendation is made when the latency gate is not satisfied.

## Reproduction

```powershell
python scripts/run_live_graph_answer_quality.py `
  C:\Users\gde00\AppData\Local\Temp\recall-live-graph-performance-post-629.json `
  C:\Users\gde00\AppData\Local\Temp\recall-live-post-629-graph-answer-quality.json `
  --source-commit 3b628422b0f367b4f10201dccd2005df9c53ed8d `
  --model deepseek/deepseek-v4-flash --workers 8 --timeout 180 --retries 3
```

The raw answer artifact is immutable after the first answer request. Any correction is a new
artifact.
