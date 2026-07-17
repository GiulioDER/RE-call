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

## Trust layer — superseded/expired memories vs plain search

STR = superseded-trust rate: how often a stale memory was presented as the answer on the validity-sensitive queries (lower is better). The final two columns verify the trust layer does not change ordinary answerable retrieval.

| embedder | STR baseline | STR trust | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|
| hashing-64 | 1.00 | 0.00 | 0.25 | 0.00 | 0.737 | 0.737 |
| BAAI/bge-small-en-v1.5 | 0.83 | 0.00 | 0.75 | 1.00 | 1.000 | 1.000 |
