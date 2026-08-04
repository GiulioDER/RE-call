| chunks | cand_k | rerank | total | embed | dense | sparse | meta | fusion | rerank | **store** | resid | **store share** | sparse fire |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20050 | 20 | no | 49.9 | 27.1 | 14.5 | 5.6 | 2.2 | 0.3 | 0.0 | **22.3** | 0.3 | **44.6%** | 100% |
| 20050 | 20 | yes | 1753.0 | 27.8 | 27.4 | 6.7 | 3.7 | 0.3 | 1686.7 | **37.8** | 0.3 | **2.2%** | 100% |
| 20050 | 250 | no | 304.5 | 24.0 | 271.6 | 4.4 | 1.9 | 1.9 | 0.0 | **277.9** | 0.7 | **91.3%** | 100% |
| 20050 | 250 | yes | 15075.1 | 28.4 | 428.3 | 5.7 | 10.3 | 2.0 | 14599.6 | **444.2** | 0.9 | **2.9%** | 100% |

All figures are ms/query, means over `repeats x n_queries`, measured warm (each configuration gets a discarded full-pipeline warm-up pass). `store` = dense + sparse + meta, where meta is `newest_indexed_at()`, the per-search round trip that sits outside every `stage_ms` bracket. `store share` is the ceiling on any store swap: a replacement that cost nothing would remove exactly this fraction.
