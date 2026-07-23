# recall — evaluation at scale

Generated corpus (`recall.eval.synthetic`), not the 14-document demo corpus. Reproduce with `python -m recall.eval.scale --embedder hashing --filler 50000 --seed 1234`.

- corpus: **50600 chunks** across 850 files
- queries: **550** (200 answerable, 100 unanswerable, 150 successor, 100 abstain)
- embedder: `hashing-64` · index time: 221.5s

## Retrieval under index pressure

| measurement | value |
|---|---|
| recall@5, unfiltered | 1.0000 [0.9812, 1.0000] (n=200) |
| recall@5, `source`-filtered | 1.0000 [0.9812, 1.0000] (n=200) |
| search latency p50 / p95 / p99 (ms) | 67.2 / 196.6 / 353.9 |

The filtered arm restricts the query to the one source that holds the answer, so recall of 1.00 is the only correct result. **It is also the only result this arm can produce, so do not read it as evidence about HNSW.** It scores the hybrid retriever, whose sparse leg is an exact `tsv @@ websearch_to_tsquery` scan — filter-aware and independent of the vector index — and every generated answerable document is a single chunk, so `source = ...` selects exactly one row. Degrading the ANN path arbitrarily leaves this number at 1.0000. HNSW recall under a `source` filter is measured directly against `query_dense` in `tests/test_hnsw_filtered_recall.py`, where it does collapse without tuning.

## Trust layer

| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|---|
| hashing-64 | 0.92 | 0.93 | 0.00 | 0.01 | 0.00 | 0.99 | 0.500 | 0.993 |

**Read STR trust together with trust coverage.** STR counts queries where a stale memory was served with verdict `ok`, so a system that returns nothing scores a perfect 0.00. The claim is 0.00 STR *at high coverage*; 0.00 STR at low coverage is a system that abstained its way to a good number.

95% Wilson score intervals for the headline rates (n in parentheses):

| embedder | STR trust | trust coverage | successor acc | abstain acc |
|---|---|---|---|---|
| hashing-64 | [0.00, 0.02] (n=250) | [0.00, 0.03] (n=250) | [0.00, 0.02] (n=150) | [0.95, 1.00] (n=100) |
