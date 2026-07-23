# recall — evaluation at scale

Generated corpus (`recall.eval.synthetic`), not the 14-document demo corpus. Reproduce with `python -m recall.eval.scale --embedder fastembed --filler 0 --seed 1234`.

- corpus: **600 chunks** across 600 files
- queries: **550** (200 answerable, 100 unanswerable, 150 successor, 100 abstain)
- embedder: `BAAI/bge-small-en-v1.5` · index time: 31.5s

## Retrieval under index pressure

| measurement | value |
|---|---|
| recall@5, unfiltered | 1.0000 [0.9812, 1.0000] (n=200) |
| recall@5, `source`-filtered | 1.0000 [0.9812, 1.0000] (n=200) |
| search latency p50 / p95 / p99 (ms) | 46.9 / 65.4 / 79.2 |

The filtered arm restricts the query to the one source that holds the answer, so recall of 1.00 is the only correct result. **It is also the only result this arm can produce, so do not read it as evidence about HNSW.** It scores the hybrid retriever, whose sparse leg is an exact `tsv @@ websearch_to_tsquery` scan — filter-aware and independent of the vector index — and every generated answerable document is a single chunk, so `source = ...` selects exactly one row. Degrading the ANN path arbitrarily leaves this number at 1.0000. HNSW recall under a `source` filter is measured directly against `query_dense` in `tests/test_hnsw_filtered_recall.py`, where it does collapse without tuning.

## Trust layer

| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 1.00 | 1.00 | 0.00 | 1.00 | 0.14 | 0.00 | 1.000 | 1.000 |

**Read STR trust together with trust coverage.** STR counts queries where a stale memory was served with verdict `ok`, so a system that returns nothing scores a perfect 0.00. The claim is 0.00 STR *at high coverage*; 0.00 STR at low coverage is a system that abstained its way to a good number.

95% Wilson score intervals for the headline rates (n in parentheses):

| embedder | STR trust | trust coverage | successor acc | abstain acc |
|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | [0.00, 0.02] (n=250) | [0.98, 1.00] (n=250) | [0.09, 0.20] (n=150) | [0.00, 0.04] (n=100) |
