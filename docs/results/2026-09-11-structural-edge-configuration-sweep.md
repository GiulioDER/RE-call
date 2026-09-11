# Structural edge configuration sweep

Measured 2026-09-11 on the full LoCoMo dataset, categories 1 through 4, with 1,536 questions.
The run used the production `Indexer`, `PgVectorStore`, and `HybridRetriever` path in a dedicated
Docker PostgreSQL database. Every configuration shared the same indexed conversation and the same
top-20 hybrid retrieval result. The baseline is the top 10 direct results. Each treatment kept a
fixed context cap of 10.

Run artifact: `2026-09-11-structural-edge-performance-locomo-sweep-full-isolated3.json`.

## Results

| arm | any evidence hit | complete evidence | MRR | precision | paired rescues | paired regressions |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 77.60% | 64.39% | 0.5095 | 9.15% | | |
| current, 5 seeds + 5 structural | 71.22% | 57.03% | 0.5004 | 8.07% | 40 | 138 |
| 8 direct + 2 structural | 75.78% | 61.91% | 0.5076 | 8.81% | 11 | 39 |
| retrieval-ranked neighbors | 77.73% | 64.32% | 0.5109 | 9.08% | 76 | 74 |
| retrieval-ranked, 8 direct + 2 structural | **79.43%** | **66.67%** | **0.5115** | **9.40%** | **45** | **17** |
| conversation order only | 71.29% | 57.42% | 0.5006 | 8.14% | 38 | 135 |

The winning arm improved over the paired baseline by 1.82 percentage points on any-hit,
2.28 points on complete evidence coverage, 0.20 points on MRR, and 0.25 points on precision.
It added exactly two structural items per question on average. Mean retrieval latency was 45.1 ms
and p95 was 64.1 ms for every arm because retrieval was shared.

## Category result

Complete evidence coverage for the winning arm versus baseline was:

| category | baseline | winning arm | delta |
|---|---:|---:|---:|
| 1 | 23.76% | 23.76% | 0.00 pp |
| 2, temporal | 76.95% | 76.95% | 0.00 pp |
| 3 | 28.26% | 29.35% | +1.09 pp |
| 4 | 77.17% | 81.21% | +4.04 pp |

The available LoCoMo structural relations were conversation order, speaker, and session date.
The selected winning additions were 1,355 conversation-order edges and 1,717 speaker edges.
Reply-continuity and explicit-entity fields were not present in this raw dataset.

## Decision

The current 5 plus 5 configuration is rejected. The retrieval-ranked 8 direct plus 2 structural
configuration passes the quality uplift gate and is the candidate for the next answer-stage replay.
This result measures retrieval only; DeepSeek answer quality and citation validation remain a
separate stage.
