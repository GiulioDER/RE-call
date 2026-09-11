# Structural edge budget sensitivity

Measured 2026-09-11 on the full LoCoMo dataset, categories 1 through 4, with 1,536
questions. The run used the production `Indexer`, `PgVectorStore`, and `HybridRetriever`
path in a dedicated Docker PostgreSQL database. Every arm shared the same indexed
conversation and the same top-20 hybrid retrieval result. The baseline is the top 10
direct results. Treatments kept a fixed context cap of 10 and added retrieval-ranked
structural neighbors.

Run artifact: `2026-09-11-structural-edge-performance-locomo-most-direct.json`.

## Results

| arm | any evidence hit | complete evidence | MRR | precision | mean added items |
|---|---:|---:|---:|---:|---:|
| baseline, top 10 direct | 77.86% | 64.45% | 0.514354 | 9.15% | 0.00 |
| retrieval-ranked, 8 direct + 2 structural | **78.97%** | **66.02%** | **0.515562** | **9.35%** | 2.00 |
| retrieval-ranked, 9 direct + 1 structural | 78.19% | 65.23% | 0.514680 | 9.24% | 1.00 |
| category selective, 8 direct + 2 structural in categories 3 and 4 | 78.71% | 65.95% | 0.515309 | 9.29% | 1.21 |

Against the paired baseline, 9 plus 1 produced 27 complete-evidence rescues and 15
regressions, for a net gain of 12 questions. Its complete-hit rate improved by 0.78
percentage points. The 8 plus 2 arm produced 53 rescues and 29 regressions, for a net
gain of 24 questions and a 1.56 point improvement.

The direct comparison between the two structural budgets favors 8 plus 2: it wins on
complete-hit outcomes for 28 questions, while 9 plus 1 wins for 16. It also leads by
0.78 points on any-hit and complete-hit coverage. Retrieval latency was shared by all
arms, with a mean of 57.1 ms and p95 of 131.4 ms.

## Category result

Complete evidence coverage for 9 plus 1 versus its paired baseline was:

| category | baseline | 9 plus 1 | delta |
|---|---:|---:|---:|
| 1 | 22.34% | 21.63% | -0.71 pp |
| 2, temporal | 75.70% | 75.39% | -0.31 pp |
| 3 | 32.61% | 34.78% | +2.17 pp |
| 4 | 77.76% | 79.31% | +1.55 pp |

The one-edge arm concentrates its benefit in categories 3 and 4, but loses a small
amount in categories 1 and 2. The category-selective arm avoids those category 1 and 2
changes and is nearly tied with global 8 plus 2 while adding fewer edges on average.

The 1,536 added edges in 9 plus 1 were 509 conversation-order edges and 1,027 speaker
edges. Fifty-five added contexts matched annotated evidence, or 3.58% of additions.

## Decision

The 9 plus 1 configuration is better than the direct-only baseline in this run, but it
is not the best tested budget. Keep retrieval-ranked 8 plus 2 as the leading global
candidate, with category-selective 8 plus 2 as the lower-addition alternative. Do not
change the production default yet. Answer-stage replay with the configured provider and
citation validation remains a separate evaluation.
