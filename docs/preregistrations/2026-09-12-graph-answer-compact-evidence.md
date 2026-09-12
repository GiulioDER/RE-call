# Preregistration: compact graph answer evidence payload

Date: 2026-09-12

Status: locked before the compact prompt requests

## Objective

Measure whether the answer prompt can retain every selected trusted passage while omitting
nonessential ranking, confidence, authority, and timestamp metadata. The compact payload keeps the
query, each `chunk_id`, and each passage `text`, so the model retains the evidence and citation
targets without receiving retrieval bookkeeping.

## Frozen design

1. Retrieval artifact: `docs/results/2026-09-12-live-graph-performance-attribution-full.json`.
2. Query pairing: the 50 off and 50 one hop rows fixed by the quality replay.
3. Model: `deepseek/deepseek-v4-flash` through OpenRouter.
4. Temperature: zero, reasoning effort `none`, and 512 maximum completion tokens.
5. Treatment: `--compact-evidence`, preserving the full selected item sequence.
6. Control: the completed unbounded replay with the same provider settings.

## Decision rule

The compact representation is accepted for a follow up deployment only if:

1. Provider failure rate is no more than two percent in each arm.
2. Valid answer envelopes, citation validity, gold citation recall, and correct abstention do not
   drop by more than two percentage points in one hop relative to the control.
3. One hop retains at least the control's any gold citation rate.
4. One hop provider prompt token p95 falls by at least 25 percent.

The result measures evidence and citation behavior, not factual correctness. Factual judging remains
a separate paired experiment with one fixed judge.

## Reproduction

```powershell
python scripts/run_live_graph_answer_quality.py `
  docs/results/2026-09-12-live-graph-performance-attribution-full.json `
  docs/results/2026-09-12-live-graph-answer-quality-compact-evidence.json `
  --source-commit ee40079710017a988886a38fa33900eb1b568874 `
  --model deepseek/deepseek-v4-flash --workers 8 --timeout 180 --retries 3 `
  --compact-evidence
```
