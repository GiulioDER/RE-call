# recall — retrieval evaluation

Reproduce the local (key-free) rows with `make eval` — needs Docker + the local embedder only. The Voyage cloud row appears when `VOYAGE_API_KEY` is set.

| embedder | fusion | P@5 | R@5 | MRR | nDCG@10 | FCR no-guard | FCR guard |
|---|---|---|---|---|---|---|---|
| hashing-64 | dense | 0.186 | 0.929 | 0.626 | 0.715 | 1.00† | 0.20 |
| hashing-64 | hybrid | 0.186 | 0.929 | 0.737 | 0.799 | 1.00† | 0.20 |
| hashing-64 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00† | 0.20 |
| BAAI/bge-small-en-v1.5 | dense | 0.200 | 1.000 | 0.964 | 0.974 | 1.00† | 1.00 |
| BAAI/bge-small-en-v1.5 | hybrid | 0.200 | 1.000 | 1.000 | 1.000 | 1.00† | 1.00 |
| BAAI/bge-small-en-v1.5 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00† | 1.00 |

_† FCR no-guard is ANALYTIC, not measured: with no gap guard the system never abstains, so every unanswerable query is answered confidently and the rate is 1.00 by definition. It is the reference point for FCR guard, not an observation._

_P@5 is mechanically capped at 0.20: each query has exactly one relevant doc, so the best possible precision@5 is 1/5. Read it as "answer found in the top 5" (binary), not as classical precision — R@5 / MRR / nDCG@10 are the informative ranking metrics._

Cost/latency (mean wall time per call):

| embedder | fusion | embed ms/query | rerank ms/query |
|---|---|---|---|
| hashing-64 | dense | 0.1 | 0.0 |
| hashing-64 | hybrid | 0.1 | 0.0 |
| hashing-64 | hybrid+rerank | 0.1 | 691.7 |
| BAAI/bge-small-en-v1.5 | dense | 16.4 | 0.0 |
| BAAI/bge-small-en-v1.5 | hybrid | 16.9 | 0.0 |
| BAAI/bge-small-en-v1.5 | hybrid+rerank | 21.0 | 735.6 |

## Trust layer — superseded/expired memories vs plain search

STR = superseded-trust rate: how often a stale memory was presented as the answer on the validity-sensitive queries (lower is better). The final two columns verify the trust layer does not change ordinary answerable retrieval.

| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|---|
| hashing-64 | 1.00 | 1.00 | 0.00 | 0.67 | 0.50 | 1.00 | 0.737 | 0.648 |
| BAAI/bge-small-en-v1.5 | 0.83 | 1.00 | 0.00 | 0.83 | 0.75 | 0.50 | 1.000 | 1.000 |

**Read STR trust together with trust coverage.** STR counts queries where a stale memory was served with verdict `ok`, so a system that returns nothing scores a perfect 0.00. The claim is 0.00 STR *at high coverage*; 0.00 STR at low coverage is a system that abstained its way to a good number.

95% Wilson score intervals for the headline rates (n in parentheses):

| embedder | STR trust | trust coverage | successor acc | abstain acc |
|---|---|---|---|---|
| hashing-64 | [0.00, 0.39] (n=6) | [0.30, 0.90] (n=6) | [0.15, 0.85] (n=4) | [0.34, 1.00] (n=2) |
| BAAI/bge-small-en-v1.5 | [0.00, 0.39] (n=6) | [0.44, 0.97] (n=6) | [0.30, 0.95] (n=4) | [0.09, 0.91] (n=2) |

## Entailment abstention — near-miss queries (arms A/B/C)

Near-miss = a high-similarity memory that does NOT answer the query — the class a cosine threshold passes by construction. Arms: `threshold` = calibrated cosine threshold (status quo), `threshold+entail` = threshold plus the QNLI judge, `entail-only` = judge alone (ablation). The judge is identical across embedders — no per-embedder recalibration. The judge-ms column averages only over the queries the judge actually ran on (threshold-abstained queries never reach it), so in the stacked arm it can exceed the all-queries total mean.

| embedder | arm | near-miss FCR | gap FCR | false-abstain | MRR ans | judge ms (judged calls) | total ms/query |
|---|---|---|---|---|---|---|---|
| hashing-64 | threshold | 0.70 | 0.60 | 0.29 | 0.429 | 0 | 6 |
| hashing-64 | threshold+entail | 0.30 | 0.20 | 0.64 | 0.357 | 476 | 336 |
| hashing-64 | entail-only | 0.60 | 0.40 | 0.07 | 0.881 | 1142 | 1150 |
| BAAI/bge-small-en-v1.5 | threshold | 1.00 | 0.00 | 0.07 | 0.929 | 0 | 25 |
| BAAI/bge-small-en-v1.5 | threshold+entail | 0.50 | 0.00 | 0.14 | 0.857 | 218 | 203 |
| BAAI/bge-small-en-v1.5 | entail-only | 0.80 | 0.40 | 0.07 | 0.929 | 1040 | 1071 |
