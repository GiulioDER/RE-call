# Live graph answer quality replay

Measured 2026-09-12 from the first recorded pass of the frozen 50 query graph performance
artifact. The replay used OpenRouter `deepseek/deepseek-v4-flash`, temperature zero, no reasoning,
and a 512 token ceiling. The preregistration is
`docs/preregistrations/2026-09-12-graph-answer-quality.md`.

## Inputs

| field | value |
|---|---|
| source commit | `c9bd5c41e7a321defb57ef3be6fba3f43b2510d3` |
| generation | `gen_6559b6dbacfc4bc2847b65005f1ba1d4` |
| model | `deepseek/deepseek-v4-flash` |
| paired rows | 50 off, 50 one hop |
| answer prompt digest | `067a63719933cdd9cfa2bbda9f68921fa14aa4dbe3259ffd754bfcfa14a27deb` |

## Results

| metric | graph off | graph one hop | delta |
|---|---:|---:|---:|
| valid answer envelope | 96% | 92% | -4 pp |
| nonempty answer | 34% | 28% | -6 pp |
| correct abstention, unanswerable | 96.4% | 100% | +3.6 pp |
| any gold citation | 20% | 22% | +2 pp |
| complete gold citation coverage | 2% | 8% | +6 pp |
| mean gold citation precision | 0.480 | 0.511 | +0.031 |
| mean gold citation recall | 0.182 | 0.280 | +0.098 |
| provider failure rate | 4% | 8% | +4 pp |
| provider latency p50 | 4,646 ms | 5,036 ms | +390 ms |
| provider latency p95 | 7,547 ms | 14,748 ms | +7,201 ms |

On the 22 labelled answerable queries, mean gold citation recall increased from 0.182 to 0.280,
an absolute gain of 0.098. The gain came from six query pairs. One query lost citation recall,
and four provider failures prevented a clean end to end quality gate.

The one hop arm had four provider failures versus two off. Failures were malformed JSON, empty
content, or an answer without required text. The preregistered two percent failure ceiling is
therefore exceeded, so this is an incomplete quality result rather than a promotion result.

This run does not measure factual correctness. The answer text and citations are preserved for a
separate fixed judge or human review. Structural answer validity proves format and citation
identity only.

## Reproduction

```powershell
$env:RECALL_REASONING_ANSWER_ENABLED='1'
$env:RECALL_REASONING_ANSWER_PROVIDER='openrouter'
$env:RECALL_REASONING_ANSWER_MODEL='deepseek/deepseek-v4-flash'
$env:RECALL_REASONING_ANSWER_MAX_TOKENS='512'
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-live-graph-performance-attribution-full.json docs/results/2026-09-12-live-graph-answer-quality.json --source-commit c9bd5c41e7a321defb57ef3be6fba3f43b2510d3 --workers 8
```
