# Preregistration: graph answer context reliability diagnostic

Date: 2026-09-12

Status: locked before the diagnostic requests

## Question

Does the larger graph evidence bundle reduce DeepSeek V4 Flash answer reliability, and can a
bounded context preserve citation quality while reducing malformed or empty output?

## Frozen design

Use the one hop rows from the first recorded pass of
`docs/results/2026-09-12-live-graph-performance-attribution-full.json`. Include all 22 labelled
answerable queries. Empty trusted bundles make no provider request and remain in the denominator
for context coverage reporting. For each query, replay the same evidence prompt at three caps:

1. `full`: all trusted evidence items selected by the graph arm.
2. `cap5`: the first five trusted evidence items.
3. `cap3`: the first three trusted evidence items.

The cap changes only the evidence bundle. Query text, retrieval artifact, model, endpoint,
temperature, response format, reasoning setting, and output limit remain fixed. The provider is
OpenRouter `deepseek/deepseek-v4-flash`, temperature zero, no reasoning, JSON response format,
and a 512 token maximum. Raw provider output is retained for failed rows.

## Outcomes

Report per cap:

1. Provider failure rate and failure type.
2. Valid answer envelope rate.
3. Nonempty answer rate.
4. Any gold citation rate.
5. Gold citation precision and recall.
6. Provider p50 and p95 latency.
7. Prompt character count and evidence item count.

The diagnostic is not a factual correctness test. A cap is a candidate only if provider failure
rate falls to at most two percent and gold citation recall does not fall by more than two
percentage points versus `full`. This diagnostic does not replace the primary answer replay.

## Reproduction

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/diagnose_graph_answer_context.py docs/results/2026-09-12-live-graph-performance-attribution-full.json docs/results/2026-09-12-graph-answer-context-diagnostic.json --source-commit c9bd5c41e7a321defb57ef3be6fba3f43b2510d3 --workers 6
```

This diagnostic artifact is immutable after the first request. Corrections use a new artifact.
