# recall — retrieval evaluation

Reproduce the local (key-free) rows with `make eval` — needs Docker + the local embedder only. The Voyage cloud row appears when `VOYAGE_API_KEY` is set.

| embedder | fusion | P@5 | R@5 | MRR | nDCG@10 | FCR no-guard | FCR guard |
|---|---|---|---|---|---|---|---|
| hashing-64 | dense | 0.186 | 0.929 | 0.626 | 0.715 | 1.00 | 0.20 |
| hashing-64 | hybrid | 0.186 | 0.929 | 0.737 | 0.799 | 1.00 | 0.20 |
| hashing-64 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.20 |
| BAAI/bge-small-en-v1.5 | dense | 0.200 | 1.000 | 0.964 | 0.974 | 1.00 | 1.00 |
| BAAI/bge-small-en-v1.5 | hybrid | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 |
| BAAI/bge-small-en-v1.5 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 |
| voyage:voyage-3 | dense | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |
| voyage:voyage-3 | hybrid | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |
| voyage:voyage-3 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |

## Trust layer — superseded/expired memories vs plain search

STR = superseded-trust rate: how often a stale memory was presented as the answer on the validity-sensitive queries (lower is better). The final two columns verify the trust layer does not change ordinary answerable retrieval.

| embedder | STR baseline | STR recency | STR trust | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|
| hashing-64 | 1.00 | 0.83 | 0.00 | 0.25 | 0.00 | 0.737 | 0.737 |
| BAAI/bge-small-en-v1.5 | 0.83 | 1.00 | 0.00 | 0.75 | 1.00 | 1.000 | 1.000 |
| voyage:voyage-3 | 1.00 | 1.00 | 0.00 | 0.75 | 1.00 | 1.000 | 1.000 |

## Entailment abstention — near-miss queries (arms A/B/C)

Near-miss = a high-similarity memory that does NOT answer the query — the class a cosine threshold passes by construction. Arms: `threshold` = calibrated cosine threshold (status quo), `threshold+entail` = threshold plus the QNLI judge, `entail-only` = judge alone (ablation). The judge is identical across embedders — no per-embedder recalibration. The judge-ms column averages only over the queries the judge actually ran on (threshold-abstained queries never reach it), so in the stacked arm it can exceed the all-queries total mean.

| embedder | arm | near-miss FCR | gap FCR | false-abstain | MRR ans | judge ms (judged calls) | total ms/query |
|---|---|---|---|---|---|---|---|
| hashing-64 | threshold | 1.00 | 1.00 | 0.00 | 0.696 | 0 | 4 |
| hashing-64 | threshold+entail | 0.60 | 0.20 | 0.21 | 0.714 | 856 | 863 |
| hashing-64 | entail-only | 0.60 | 0.40 | 0.07 | 0.881 | 889 | 894 |
| BAAI/bge-small-en-v1.5 | threshold | 0.80 | 0.00 | 0.00 | 1.000 | 0 | 19 |
| BAAI/bge-small-en-v1.5 | threshold+entail | 0.50 | 0.00 | 0.07 | 0.929 | 149 | 139 |
| BAAI/bge-small-en-v1.5 | entail-only | 0.80 | 0.40 | 0.07 | 0.929 | 827 | 854 |
| voyage:voyage-3 | threshold | 0.40 | 0.00 | 0.00 | 1.000 | 0 | 306 |
| voyage:voyage-3 | threshold+entail | 0.40 | 0.00 | 0.07 | 0.929 | 125 | 390 |
| voyage:voyage-3 | entail-only | 0.80 | 0.40 | 0.07 | 0.929 | 1018 | 1329 |
