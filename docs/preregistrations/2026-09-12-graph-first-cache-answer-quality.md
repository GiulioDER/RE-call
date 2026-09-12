# Preregistration: graph first cache answer quality replay

Date: 2026-09-12

Status: locked before the first answer generation request

## Objective

Measure whether the merged graph first semantic cache reuse changes generated answer quality or
only removes repeated graph projection work. The replay uses the paired first recorded pass from
the exact post merge retrieval artifact and does not rerun retrieval.

## Frozen inputs

1. Retrieval artifact: `docs/results/2026-09-12-graph-first-cache-performance-rerun.json`.
2. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
3. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
4. Arms: `off` and `one_hop`.
5. Pairing unit: query identity, using the first recorded pass from each arm.
6. The answer provider receives only the production trusted evidence bundle and query.

## Answer provider

OpenRouter chat completions with model `deepseek/deepseek-v4-flash`, temperature zero, reasoning
effort `none`, JSON response format, and a 512 token maximum. The API key is read from the
environment and is never written to an artifact. The model identity, provider latency, token
counts, prompt digest, and failures are recorded.

## Prediction

Because PR 631 changes only semantic graph projection loading and preserves the graph projection
contents, I predict byte identical trusted evidence for all paired queries. I predict no material
change in valid answer envelopes, nonempty answers, correct abstention, or citation metrics. The
provider failure rate is expected to remain at or below two percent.

## Decision rule

The replay is complete only if provider failures are at or below two percent. The cache change
passes the answer quality guard if one hop is no worse than graph off by two percentage points for
valid envelopes, citation validity, gold citation recall, and correct abstention. Any positive
result is evidence quality only and does not establish factual correctness.

## Reproduction

```powershell
$env:OPENROUTER_API_KEY='set outside the artifact'
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-graph-first-cache-performance-rerun.json docs/results/2026-09-12-graph-first-cache-answer-quality.json --source-commit a4043f3beb6eedca4a641cc01fb325c7af550abe --model deepseek/deepseek-v4-flash --workers 8
```

The raw answer artifact is immutable after the first request. Any correction is a new artifact.
