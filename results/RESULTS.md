# recall — retrieval evaluation

Reproduce the local (key-free) rows with `make eval` — needs Docker + the local embedder only. The Voyage cloud row appears when `VOYAGE_API_KEY` is set.

> **Regenerated under the held-out protocol.** Two things to know when comparing against an
> earlier copy of this file:
>
> - **`false-abstain` rose from 0.00 to 0.07 on the threshold arms.** A correction, not a
>   regression. The gap threshold is now refitted per query with that query's own sample held
>   out, and `best_threshold` places the boundary exactly on the *lowest* answerable cosine — so
>   holding that sample out lifts the boundary above it and the query abstains. The old 0.00 was
>   the optimiser being scored on its own objective. The drop in `MRR ans` (1.000 → 0.929 for
>   bge-small) is that same single abstained query.
> - **The `voyage:voyage-3` rows are absent, not zero:** this run was made without a
>   `VOYAGE_API_KEY`, so the cloud arm never executed. The previously published Voyage numbers
>   came from an older harness and are not comparable to the rows below; re-run with a key to
>   restore them.

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
| hashing-64 | hybrid+rerank | 0.1 | 780.8 |
| BAAI/bge-small-en-v1.5 | dense | 19.9 | 0.0 |
| BAAI/bge-small-en-v1.5 | hybrid | 20.4 | 0.0 |
| BAAI/bge-small-en-v1.5 | hybrid+rerank | 20.7 | 773.6 |

## Trust layer — superseded/expired memories vs plain search

STR = superseded-trust rate: how often a stale memory was presented as the answer on the validity-sensitive queries (lower is better). The final two columns verify the trust layer does not change ordinary answerable retrieval.

| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | abstain acc | MRR ans (base) | MRR ans (trust) |
|---|---|---|---|---|---|---|---|---|
| hashing-64 | 1.00 | 1.00 | 0.00 | 1.00 | 0.25 | 0.00 | 0.737 | 0.737 |
| BAAI/bge-small-en-v1.5 | 0.83 | 1.00 | 0.00 | 0.67 | 0.75 | 1.00 | 1.000 | 1.000 |

**Read STR trust together with trust coverage.** STR counts queries where a stale memory was served with verdict `ok`, so a system that returns nothing scores a perfect 0.00. The claim is 0.00 STR *at high coverage*; 0.00 STR at low coverage is a system that abstained its way to a good number.

95% Wilson score intervals for the headline rates (n in parentheses):

| embedder | STR trust | trust coverage | successor acc | abstain acc |
|---|---|---|---|---|
| hashing-64 | [0.00, 0.39] (n=6) | [0.61, 1.00] (n=6) | [0.05, 0.70] (n=4) | [0.00, 0.66] (n=2) |
| BAAI/bge-small-en-v1.5 | [0.00, 0.39] (n=6) | [0.30, 0.90] (n=6) | [0.30, 0.95] (n=4) | [0.34, 1.00] (n=2) |

## Entailment abstention — near-miss queries (arms A/B/C)

Near-miss = a high-similarity memory that does NOT answer the query — the class a cosine threshold passes by construction. Arms: `threshold` = calibrated cosine threshold (status quo), `threshold+entail` = threshold plus the QNLI judge, `entail-only` = judge alone (ablation). The judge is identical across embedders — no per-embedder recalibration. The judge-ms column averages only over the queries the judge actually ran on (threshold-abstained queries never reach it), so in the stacked arm it can exceed the all-queries total mean.

| embedder | arm | near-miss FCR | gap FCR | false-abstain | MRR ans | judge ms (judged calls) | total ms/query |
|---|---|---|---|---|---|---|---|
| hashing-64 | threshold | 1.00 | 1.00 | 0.07 | 0.696 | 0 | 4 |
| hashing-64 | threshold+entail | 0.60 | 0.20 | 0.21 | 0.714 | 716 | 697 |
| hashing-64 | entail-only | 0.60 | 0.40 | 0.07 | 0.881 | 1152 | 1158 |
| BAAI/bge-small-en-v1.5 | threshold | 0.80 | 0.00 | 0.07 | 0.929 | 0 | 31 |
| BAAI/bge-small-en-v1.5 | threshold+entail | 0.50 | 0.00 | 0.14 | 0.857 | 189 | 167 |
| BAAI/bge-small-en-v1.5 | entail-only | 0.80 | 0.40 | 0.07 | 0.929 | 1090 | 1118 |
