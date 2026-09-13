# Preregistration: generation graph metadata cache answer quality

Date: 2026-09-12

Status: locked before the first answer generation request

## Objective

Measure whether the generated answers remain equal or improve after the generation scoped graph
metadata cache change. This is a paired replay over the new retrieval artifact. It does not rerun
retrieval and it does not provide query labels to the answer model.

## Frozen inputs

1. Retrieval artifact: `docs/results/2026-09-12-generation-graph-metadata-cache-performance.json`.
2. Retrieval artifact SHA256: `0333D36B3B4A93E2AE4DC87C7CAAE1A7FEFACB278E1C09067CE8A68B761FA310`.
3. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
4. Tenant: `memory`.
5. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
6. Arms: `off` and `one_hop`.
7. Pairing unit: query identity, using recorded pass one from each arm.
8. The answer provider receives only the production trusted evidence bundle and query.

## Answer provider

OpenRouter chat completions with model `deepseek/deepseek-v4-flash`, temperature zero, reasoning
effort `none`, JSON response format, and a 512 token maximum. The API key is read from the
environment and is never written to an artifact.

## Metrics and decision rule

Report valid answer envelope rate, nonempty answer rate, correct abstention rate, citation
validity, gold citation precision and recall, any gold citation rate, complete citation coverage,
provider failures, provider latency, token counts, and paired query level gains and losses.

The replay is usable only when provider failures are at most two percent. The treatment must not
drop valid envelopes, citation validity, gold citation recall, or correct abstention by more than
two percentage points. Any positive result is evidence about retrieval supported answers, not a
factual correctness claim.

## Reproduction

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-generation-graph-metadata-cache-performance.json docs/results/2026-09-12-generation-graph-metadata-cache-answer-quality.json --source-commit fe3a3bad7390197f35e91c6ed944fbb6a99c3574 --workers 8
```

The raw answer artifact is immutable after the first request. Any correction is a new artifact.
