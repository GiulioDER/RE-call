# Preregistration: graph answer context diagnostic correction

Date: 2026-09-12

Status: locked before the correction requests

The first context diagnostic was measured with an omitted OpenRouter reasoning field. Its captured
failures show that DeepSeek used the full completion budget for hidden reasoning. The provider
contract and replay path now send `reasoning: {"effort":"none"}` explicitly. This correction
repeats the same frozen context sweep with that payload fix and does not replace the original
artifact.

## Fixed design

Use the same 22 answerable one hop rows, three context caps, model, temperature, JSON response
format, and 512 token maximum as
`docs/preregistrations/2026-09-12-graph-answer-context-diagnostic.md`.

The only changed factor is the explicit OpenRouter request field
`reasoning: {"effort":"none"}`. The caps remain `full`, `cap5`, and `cap3`. Raw provider output
is retained for every failed row. Report provider reliability, envelope validity, citation
precision and recall, prompt size, evidence item count, and provider latency.

This correction measures the provider configuration repair. It is not a production deployment
claim until the code change is merged and VPS2 is synchronized to that revision.

## Reproduction

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/diagnose_graph_answer_context.py docs/results/2026-09-12-live-graph-performance-attribution-full.json docs/results/2026-09-12-graph-answer-context-diagnostic-correction.json --source-commit <patched-checkout> --workers 6
```
