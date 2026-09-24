# Live graph performance attribution after PR 627

Measurement date: 2026-09-12

I ran the locked 50 query comparison after merge commit `1205e347473461266055a1532eabbbca7c0b4b95`.
The live serving snapshot was synchronized on VPS2, and a fresh MCP process completed the run.

## Run identity

1. Query count: 50.
2. Recorded passes: 5 per arm, 250 rows per arm.
3. Warmup passes: 1 per arm, unscored.
4. Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.
5. Tenant: `memory`.
6. Embedder: `voyage:voyage-4`.
7. Generation: `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
8. Graph policy fingerprint: `0151785350a8c678b3e254e4f9b356f71602b4a022dc3cc3f0a848b36e45cf59`.
9. Artifact: `docs/results/2026-09-12-live-graph-performance-attribution-post-merge.json`.

Remeasure command:

```powershell
python scripts/run_live_graph_performance_attribution.py `
  --generation-id gen_6559b6dbacfc4bc2847b65005f1ba1d4 `
  --output docs/results/2026-09-12-live-graph-performance-attribution-post-merge.json `
  --timeout 240 --passes 5 --warmup-passes 1 --max-steps 12 `
  --max-graph-nodes 32 --max-evidence-tokens 2048 `
  --variant combined --control none
```

## Latency result

| Metric | Graph off p50 | Graph off p95 | One hop p50 | One hop p95 |
| --- | ---: | ---: | ---: | ---: |
| Server total, ms | 461.609 | 598.949 | 1059.968 | 1283.636 |
| Client observed, ms | 1059.752 | 1366.796 | 1633.424 | 2120.204 |
| Baseline retrieval, ms | 460.964 | 598.223 | 1059.865 | 1282.492 |
| Database statements | 7 | 7 | 17 | 17 |
| Database transferred application bytes | 145617 | 151130 | 444604 | 456252 |

The one hop server p95 is 2.143 times graph off. The secondary availability guard therefore fails:
1283.636 ms is above 1197.898 ms, which is twice the graph off p95.

## Attribution

One hop p95 spans were:

1. Readiness check: 6.686 ms.
2. Projection load: 0.019 ms.
3. Adjacency construction: 0 ms.
4. Query embedding: 232.690 ms.
5. Candidate fetch: 4.012 ms.
6. Cosine rescoring: 3.341 ms.
7. Trust reevaluation: 0 ms.
8. Planner execution: 0 ms.

The recorded one hop rows had 250 projection cache hits and 250 adjacency cache hits, with no
recorded misses because the miss occurred during the unscored warmup. The graph discovered 3360
candidates in total, fetched 1541 candidate chunks, and rescored 3360 candidates. The combined
policy refused graph admission for 85 rows and admitted graph candidates for 165 rows. No new
trusted evidence was added on this query set.

The main cost is inside the one hop retrieval path. The graph path performs a separate query
embedding before the baseline retrieval has exposed its query vector, then performs additional
database work. This is the highest ROI follow up: share the already computed retrieval vector with
graph first expansion and measure the same frozen run again.

## Quality result

This runner has the answer provider disabled, so it measures retrieval evidence quality rather than
generated answer correctness. On the final recorded pass, hit at five was 0.22 for both arms and
MRR was 0.21 for both arms. The paired deltas were zero for all 50 queries. The deterministic paired
bootstrap 95 percent interval was `[0.000, 0.000]` for both hit at five and MRR, and the paired
permutation p value was 1.0 for both. This shows no retrieval quality regression or gain on this
frozen set.

## Single flight probe

The separate same key eight request probe completed on 2026-09-12 in 9449.824 ms wall time. It
reported one projection single flight owner and seven waiters. The cold projection load was about
1.8 to 2.0 seconds per response while the shared build was in progress. This confirms coordination
works, but it does not reduce the cold build cost itself.

## Gate status

1. Primary cold versus warm graph stage gate: `PENDING`. The existing runner does not persist the
   unscored warmup spans, so it cannot provide the required cold p95 denominator.
2. Secondary one hop availability guard: `FAIL`.
3. Retrieval quality noninferiority: `PASS` on this frozen set.
4. Instrumentation completeness: `PASS` for all required fields in 500 scored rows.
5. Generated answer quality: not measured by this artifact.

The result does not support enabling one hop by default. The next test should first land or isolate
query vector reuse, then rerun the unchanged preregistered comparison and the DeepSeek answer replay.

## Raw artefact (appended 2026-09-24)

The JSON artefact named above is not committed: it embeds text retrieved from a private memory
corpus, and this repository is public. It is held outside the tree. Its SHA256 on 2026-09-24 was
`5785c3a133e378b430ed78419c2c6ca41b9cfd726a81f828c7b7682996afe7f3`, which does not match the
`7fca0046…` recorded as the frozen input in
`docs/preregistrations/2026-09-12-post-merge-graph-answer-quality.md`.
