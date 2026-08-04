| chunks | cand_k | rerank | total | embed | dense | sparse | meta | fusion | rerank | **store** | resid | **store share** | sparse fire |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20050 | 20 | no | 37.1 | 20.0 | 10.5 | 4.4 | 1.7 | 0.3 | 0.0 | **16.6** | 0.3 | **44.7%** | 100% |
| 20050 | 20 | yes | 1605.2 | 25.2 | 22.6 | 7.2 | 3.2 | 0.3 | 1546.5 | **32.9** | 0.3 | **2.1%** | 100% |
| 20050 | 250 | no | 266.0 | 21.4 | 236.9 | 3.9 | 1.6 | 1.6 | 0.0 | **242.4** | 0.6 | **91.1%** | 100% |
| 20050 | 250 | yes | 18204.8 | 32.9 | 468.1 | 7.2 | 5.4 | 2.2 | 17687.9 | **480.8** | 1.1 | **2.6%** | 100% |

All figures are ms/query, means over `repeats x n_queries`, measured warm (each configuration gets a discarded full-pipeline warm-up pass). `store` = dense + sparse + meta, where meta is `newest_indexed_at()`, the per-search round trip that sits outside every `stage_ms` bracket. `store share` is the ceiling on any store swap: a replacement that cost nothing would remove exactly this fraction.

⚠️ **Scope.** Corpus is SYNTHETIC (`recall.eval.synthetic`), 20050 chunks, embedder `bge-small-symmetric-v1`, seed 1234, 50 queries x 2 repeats. Two limits follow. (1) The sparse leg here is not representative: commit `9a5165b` measured sparse median **496 ms** on a real 72k-chunk corpus, where this run measures single-digit ms. The cost is corpus-vocabulary dependent and neither figure generalises. (2) At `candidate_k=250` the dense leg runs at `hnsw.ef_search = min(k x multiplier, 1000)` against 80 at k=20 — so the k=250 row prices an OVER-FETCH SETTING inside the store, not Postgres against another backend. A different engine re-pays that walk rather than removing it.
