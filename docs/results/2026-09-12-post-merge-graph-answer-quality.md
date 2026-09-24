# Post merge graph answer quality replay

Measurement date: 2026-09-12

I replayed the first recorded pass from the post merge graph performance artifact with
OpenRouter `deepseek/deepseek-v4-flash`. Retrieval was not rerun, so both arms used paired evidence
from the same completed performance run.

## Run identity

1. Source commit: `1205e347473461266055a1532eabbbca7c0b4b95`.
2. Input artifact SHA256: `7fca00460d1ce21eeff319fa5cce7bd83bb0aab497e114c6d6c6090b15a73ecb`.
3. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
4. Model: `deepseek/deepseek-v4-flash`.
5. Temperature: 0.
6. Reasoning effort: `none`.
7. Maximum completion tokens: 512.
8. Paired rows: 50 graph off and 50 graph one hop.
9. Provider failures: 0 in both arms.
10. Answer prompt digest: `067a63719933cdd9cfa2bbda9f68921fa14aa4dbe3259ffd754bfcfa14a27deb`.
11. Raw artifact: `docs/results/2026-09-12-post-merge-graph-answer-quality.json`.

The run was preregistered in `docs/preregistrations/2026-09-12-post-merge-graph-answer-quality.md`
before the first answer request.

## Results

| Metric | Graph off | Graph one hop | Paired delta |
| --- | ---: | ---: | ---: |
| Valid answer envelope | 100% | 100% | 0 pp |
| Nonempty answer | 38% | 38% | 0 pp |
| Correct abstention on unanswerable | 96.4% | 96.4% | 0 pp |
| Any gold citation | 20% | 22% | +2 pp |
| Complete gold citation coverage | 4% | 6% | +2 pp |
| Mean gold citation precision | 0.447 | 0.509 | +0.061 |
| Mean gold citation recall | 0.211 | 0.248 | +0.036 |
| Provider latency p50, ms | 2859 | 3176 | +317 |
| Provider latency p95, ms | 16023 | 7278 | -8745 |
| Prompt tokens | 25105 | 36769 | +11664 |
| Completion tokens | 3095 | 3513 | +418 |

The paired query level citation recall delta was +0.016 across all 50 queries. One query gained
gold citation recall and one query lost recall. The other 48 were unchanged. Correct abstention,
valid envelopes, and nonempty answer rate were unchanged.

The one hop prompt used 46.5 percent more prompt tokens than graph off. That is an answer cost
increase even though the graph evidence metrics improved slightly.

## Decision

The answer quality replay is complete because provider failure rate was zero in both arms. It
passes the preregistered nonregression checks for valid envelopes, citation validity, gold citation
recall, and correct abstention. The citation improvement is evidence quality only. This artifact
does not measure factual correctness because DeepSeek was not used as a factual judge.

The overall deployment recommendation remains negative because the separate live performance run
failed the one hop availability guard. The graph should not be enabled by default until the extra
retrieval cost is removed and the unchanged performance comparison is rerun.

## Reproduction

```powershell
python scripts/run_live_graph_answer_quality.py `
  docs/results/2026-09-12-live-graph-performance-attribution-post-merge.json `
  docs/results/2026-09-12-post-merge-graph-answer-quality.json `
  --source-commit 1205e347473461266055a1532eabbbca7c0b4b95 `
  --model deepseek/deepseek-v4-flash --workers 8 --timeout 180 --retries 3
```

## Raw artefacts (appended 2026-09-24)

Neither raw JSON artefact named above is committed. Both embed text retrieved from a private
memory corpus, and this repository is public. They are held outside the tree, and these are their
hashes as found on 2026-09-24:

| Artefact | SHA256 on 2026-09-24 |
| --- | --- |
| `2026-09-12-post-merge-graph-answer-quality.json` | `8da90c9ea51a8847850359a254e9176242271f6796ae03d3329403458d8fe58a` |
| `2026-09-12-live-graph-performance-attribution-post-merge.json` | `5785c3a133e378b430ed78419c2c6ca41b9cfd726a81f828c7b7682996afe7f3` |

The second hash does **not** match the input SHA256 recorded above and in the preregistration
(`7fca0046…`). The file changed at some point after the replay froze it, and nothing records when
or why. Treat the copy held today as unverified against this run.
