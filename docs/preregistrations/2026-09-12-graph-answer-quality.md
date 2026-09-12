# Preregistration: graph answer quality replay

Date: 2026-09-12

Status: locked before the first answer generation request

## Objective

Measure whether the evidence retrieved by the graph one hop arm produces better generated
answers than graph off when the answer model, prompt, token budget, and temperature are fixed.
This is a paired answer replay over an already measured retrieval artifact. It does not re-run
retrieval and it does not use query labels in the prompt.

## Frozen inputs

1. Retrieval artifact: `docs/results/2026-09-12-live-graph-performance-attribution-full.json`.
2. Query set: `docs/preregistrations/2026-08-17-memory-queries.json`.
3. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
4. Tenant: `memory`.
5. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
6. Arms: `off` and `one_hop`.
7. Pairing unit: query identity, using the first recorded pass from each arm.
8. The answer provider receives only the production trusted evidence bundle and query.

## Answer provider

OpenRouter chat completions with model `deepseek/deepseek-v4-flash`, temperature zero,
reasoning effort `none`, JSON response format, and a 512 token maximum. The API key is read from
the environment and is never written to an artifact. The returned model identity, provider
latency, token counts, prompt digest, and failures are recorded.

## Quality outcomes

Report separately for each arm and as paired one hop minus off deltas:

1. Valid answer envelope rate.
2. Nonempty answer rate.
3. Correct abstention rate on the labelled unanswerable queries.
4. Citation validity rate.
5. Gold citation precision and recall, using the frozen `source:ordinal` labels.
6. Any gold citation rate and complete gold citation coverage where applicable.
7. Provider failure rate, latency, prompt tokens, completion tokens, and total tokens.
8. Query level gains, losses, and unchanged cases.

The answer provider is not a factual judge. This run must not claim factual correctness. A factual
judge or human review is a separate follow up using the stored answers and the same judge for both
arms.

## Decision rule

The graph answer quality result is useful only if:

1. The treatment has no more than a two percentage point drop in valid answer envelopes.
2. Citation validity does not regress by more than two percentage points.
3. Gold citation recall does not regress by more than two percentage points.
4. Correct abstention does not regress by more than two percentage points.

Any positive citation or answer result is reported as evidence quality, not factual answer
correctness. A result with provider failures above two percent is incomplete and cannot support a
quality conclusion.

## Reproduction

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-live-graph-performance-attribution-full.json docs/results/2026-09-12-live-graph-answer-quality.json --source-commit c9bd5c41e7a321defb57ef3be6fba3f43b2510d3 --workers 8
```

The raw answer artifact is immutable after the first request. Any correction is a new artifact.
