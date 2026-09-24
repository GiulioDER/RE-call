# Preregistration: post merge graph answer quality replay

Date: 2026-09-12

Status: locked before the first answer generation request

## Objective

Measure whether the evidence retrieved by the merged graph one hop arm produces better generated
answers than graph off when the answer model, prompt, token ceiling, temperature, and pairing are
fixed. Retrieval is not rerun in this phase.

## Frozen inputs

1. Retrieval artifact: `docs/results/2026-09-12-live-graph-performance-attribution-post-merge.json`.
2. Retrieval artifact SHA256: `7fca00460d1ce21eeff319fa5cce7bd83bb0aab497e114c6d6c6090b15a73ecb`.
3. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
4. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
5. Arms: `off` and `one_hop`.
6. Pairing unit: query identity, using recorded pass 1 from each arm.
7. The prompt receives only the query and trusted evidence bundle from the retrieval artifact.

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
  docs/results/2026-09-12-live-graph-performance-attribution-post-merge.json `
  docs/results/2026-09-12-post-merge-graph-answer-quality.json `
  --source-commit 1205e347473461266055a1532eabbbca7c0b4b95 `
  --model deepseek/deepseek-v4-flash --workers 8 --timeout 180 --retries 3
```

The raw answer artifact is immutable after the first answer request. Any correction is a new
artifact.
