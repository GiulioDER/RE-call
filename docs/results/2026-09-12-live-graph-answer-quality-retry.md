# Graph answer quality correction pass

Measured 2026-09-12 after the primary answer replay. This is a separate correction artifact and
does not replace the preregistered 100 row result. It replays the six query pairs that had a
provider failure in the primary run, using the same model and settings.

| metric | graph off | graph one hop |
|---|---:|---:|
| correction rows | 6 | 6 |
| provider failures | 1 | 3 |
| provider failure rate | 16.7% | 50.0% |
| valid answer envelope | 83.3% | 50.0% |
| provider latency p50 | 2,551 ms | 4,256 ms |
| provider latency p95 | 2,758 ms | 4,256 ms |

The failures remained malformed JSON responses. One previously failed graph row recovered, but two
previously failed graph rows failed again and one additional graph row failed in the correction.
This confirms provider instability under the larger graph evidence prompts rather than a single
transient request failure.

The primary result remains incomplete because its graph provider failure rate was 8%, above the
preregistered 2% ceiling. The correction strengthens the decision to avoid promotion until the
answer prompt and output reliability are fixed or the graph context is bounded more tightly.

Raw correction artifact:
`docs/results/2026-09-12-live-graph-answer-quality-retry.json`.

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-live-graph-performance-attribution-full.json docs/results/2026-09-12-live-graph-answer-quality-retry.json --source-commit c9bd5c41e7a321defb57ef3be6fba3f43b2510d3 --workers 6 --query-index 2 --query-index 4 --query-index 6 --query-index 17 --query-index 18 --query-index 20
```
