# Preregistration: full graph answer replay with eight evidence items

Date: 2026-09-12

Status: locked before the cap eight provider requests

## Objective

Measure whether an eight item retrieval ordered evidence cap preserves the graph citation benefit
while reducing the answer prompt relative to the unbounded replay. This is a paired replay over the
frozen retrieval artifact and does not rerun retrieval.

## Frozen design

The design is identical to `2026-09-12-graph-answer-cap5.md`, except the treatment is
`--max-evidence-items 8`. The control is the completed unbounded replay using the same model,
provider, temperature, reasoning setting, completion limit, and source artifact.

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
  docs/results/2026-09-12-live-graph-answer-quality-cap8.json `
  --source-commit ee40079710017a988886a38fa33900eb1b568874 `
  --model deepseek/deepseek-v4-flash --workers 8 --timeout 180 --retries 3 `
  --max-evidence-items 8
```
