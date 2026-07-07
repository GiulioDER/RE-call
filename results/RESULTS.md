# recall — retrieval evaluation

Reproduce the local (key-free) rows with `make eval` — needs Docker + the local embedder only. Cloud rows appear when `VOYAGE_API_KEY`/`OPENAI_API_KEY` are set.

| embedder | fusion | P@5 | R@5 | MRR | nDCG@10 | FCR no-guard | FCR guard |
|---|---|---|---|---|---|---|---|
| hashing-64 | dense | 0.200 | 1.000 | 0.680 | 0.757 | 1.00 | 0.00 |
| hashing-64 | hybrid | 0.200 | 1.000 | 0.785 | 0.836 | 1.00 | 0.00 |
| hashing-64 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |
| BAAI/bge-small-en-v1.5 | dense | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.80 |
| BAAI/bge-small-en-v1.5 | hybrid | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.80 |
| BAAI/bge-small-en-v1.5 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.80 |
| voyage:voyage-3 | dense | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |
| voyage:voyage-3 | hybrid | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |
| voyage:voyage-3 | hybrid+rerank | 0.200 | 1.000 | 1.000 | 1.000 | 1.00 | 0.00 |
