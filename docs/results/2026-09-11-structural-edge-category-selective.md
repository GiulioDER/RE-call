# Category selective structural edge configuration

Measured 2026-09-11 on all 1,536 eligible LoCoMo questions. Categories 1 and 2 used the baseline
top 10 direct results. Categories 3 and 4 used 8 direct results plus 2 retrieval-ranked structural
neighbors. The run shared one indexed PostgreSQL corpus and one top-20 hybrid retrieval result per
question across all arms.

| arm | any evidence hit | complete evidence | MRR | precision | rescues | regressions |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 77.86% | 64.26% | 0.5128 | 9.13% | | |
| global 8 direct + 2 retrieval-ranked | 79.10% | 66.21% | 0.5142 | 9.37% | 42 | 23 |
| category selective | 78.91% | 65.95% | 0.5140 | 9.31% | 30 | 14 |

Category complete coverage for baseline, global, and selective was:

| category | baseline | global | selective |
|---|---:|---:|---:|
| 1 | 21.28% | 21.63% | 21.28% |
| 2, temporal | 76.32% | 77.26% | 76.32% |
| 3 | 31.52% | 32.61% | 32.61% |
| 4 | 77.65% | 80.62% | 80.62% |

The selective policy reduced regressions, but it also discarded useful gains in categories 1 and 2.
Its complete-coverage uplift was +1.69 percentage points, below the +2 point gate. The global
8 direct plus 2 retrieval-ranked configuration remains the stronger candidate, although its
complete-coverage uplift in this rerun was +1.95 points and therefore just below the preregistered
gate. No production default was changed. DeepSeek answer replay is still pending.

Full raw artifact: `2026-09-11-structural-edge-performance-locomo-category-selective.json`.
