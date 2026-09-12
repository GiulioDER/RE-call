# Preregistration: full graph answer replay with five evidence items

Date: 2026-09-12

Status: locked before the cap five provider requests

## Objective

Measure whether limiting the answer prompt to the first five retrieval ordered trusted evidence
items preserves the graph citation benefit while reducing provider prompt size and latency. This
is a paired replay over the frozen retrieval artifact and does not rerun retrieval.

## Frozen design

1. Retrieval artifact: `docs/results/2026-09-12-live-graph-performance-attribution-full.json`.
2. Query set and pairing: the 50 off and 50 one hop rows already fixed by the quality replay.
3. Tenant and generation: unchanged from the source artifact.
4. Model: `deepseek/deepseek-v4-flash` through OpenRouter.
5. Temperature: zero.
6. Reasoning: explicitly `{"effort":"none"}`.
7. Completion limit: 512 tokens.
8. Treatment: `--max-evidence-items 5`, preserving retrieval order and all bundle metadata.
9. Control: the previously completed unbounded replay with the same provider settings.

## Decision rule

The cap is accepted for a follow up deployment only if:

1. Provider failure rate is no more than two percent in each arm.
2. Valid answer envelopes, citation validity, gold citation recall, and correct abstention do not
   drop by more than two percentage points in one hop relative to the unbounded replay.
3. One hop retains at least the unbounded replay's any gold citation rate.
4. One hop provider prompt token p95 falls by at least 25 percent.

The result measures evidence and citation behavior, not factual correctness. Factual judging remains
a separate paired experiment with one fixed judge.

## Reproduction

```powershell
python scripts/run_live_graph_answer_quality.py `
  docs/results/2026-09-12-live-graph-performance-attribution-full.json `
  docs/results/2026-09-12-live-graph-answer-quality-cap5.json `
  --source-commit ee40079710017a988886a38fa33900eb1b568874 `
  --model deepseek/deepseek-v4-flash --workers 8 --timeout 180 --retries 3 `
  --max-evidence-items 5
```
