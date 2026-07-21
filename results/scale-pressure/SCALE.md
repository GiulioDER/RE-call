# recall — evaluation at scale

Generated corpus (`recall.eval.synthetic`), not the 14-document demo corpus. Reproduce with `python -m recall.eval.scale --embedder hashing --filler 50000 --seed 1234`.

- corpus: **50600 chunks** across 850 files
- queries: **550** (200 answerable, 100 unanswerable, 150 successor, 100 abstain)
- embedder: `hashing-64` · index time: 126.0s

## Retrieval under index pressure

| measurement | value |
|---|---|
| recall@5, unfiltered | 1.0000 [0.9812, 1.0000] (n=200) |
| recall@5, `source`-filtered | 1.0000 [0.9812, 1.0000] (n=200) |
| search latency p50 / p95 / p99 (ms) | 10.7 / 13.5 / 16.5 |

The filtered arm restricts the query to the one source that holds the answer, so recall of 1.00 is the only correct result. A shortfall is HNSW post-filtering: the graph walk cannot see the `WHERE` clause, so it can fail to surface a row the table certainly contains.

## Trust layer

| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|---|
| hashing-64 | 0.35 | 0.58 | 0.00 | 0.47 | 0.00 | 0.53 | 0.585 | 0.873 |

**Read STR trust together with trust coverage.** STR counts queries where a stale memory was served with verdict `ok`, so a system that returns nothing scores a perfect 0.00. The claim is 0.00 STR *at high coverage*; 0.00 STR at low coverage is a system that abstained its way to a good number.

95% Wilson score intervals for the headline rates (n in parentheses):

| embedder | STR trust | trust coverage | successor acc | abstain acc |
|---|---|---|---|---|
| hashing-64 | [0.00, 0.02] (n=250) | [0.41, 0.53] (n=250) | [0.00, 0.02] (n=150) | [0.43, 0.62] (n=100) |
