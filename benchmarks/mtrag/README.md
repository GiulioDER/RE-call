# MTRAG-UN / MTRAGEval Task A

This adapter evaluates RE-call on the fixed, four-domain public MTRAG-UN release used by
MTRAGEval. It indexes the official passage-level ClapNQ, Cloud, FiQA, and Govt corpora and scores
the released qrels with macro-averaged nDCG and Recall at 1, 3, 5, and 10.

The frozen arms are declared in `run.py`. `recall_default_last` is the primary result; the
reranked and recent-history arms are secondary or competitive configurations, not replacements
chosen after looking at the test scores.

```bash
python -m benchmarks.mtrag.run \
  --mtrag-root /path/to/mt-rag-benchmark \
  --output-dir /path/to/results \
  --dsn-env-file /path/to/env-with-DATABASE_URL \
  --phase all
```

Indexing is resumable at a passage boundary: an interrupted run skips the number of unique rows
already present in each domain table. The adapter never drops tables.

The chunk tables carry `FORCE ROW LEVEL SECURITY`, so a plain `SELECT count(*)` returns 0 unless
`recall.tenant_id` is set. That is the isolation model, not an empty index.

The first completed run is archived on VPS2 at
`/var/lib/recall-benchmarks/2026-08-04-mtrag-symmetric-baseline/`, with a SHA256 manifest, the
byte-exact adapter that produced it under `runner/`, and a `NOTE.md` recording the arms,
invocation, model identities and the reason its latency figures are diagnostic only. See
`docs/ENTERPRISE_PROGRAM_STATUS.md` for the validation verdicts.

