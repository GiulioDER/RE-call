# Controlled graph tail replacement result

Measured 2026-09-11 on the full LoCoMo dataset, categories 1 through 4, with 1,536 questions.
The run used the production `Indexer`, `PgVectorStore`, and `HybridRetriever` path in an isolated
Docker PostgreSQL database. Each conversation was indexed once and each question shared one
top-20 hybrid retrieval result across all arms.

Preregistration: [2026-09-11-controlled-tail-replacement](../preregistrations/2026-09-11-controlled-tail-replacement.md).
Raw artifact: [2026-09-11-controlled-tail-replacement-locomo-full.json](2026-09-11-controlled-tail-replacement-locomo-full.json).

The controlled arms used the fixed development calibration mapping `threshold=0.50`, `scale=0.05`
for `fastembed`, with a calibrated margin of `0.05`. This mapping is reproducible but is not a
certified production calibration.

## Results

| arm | any evidence hit | complete evidence | MRR | precision | mean added items |
|---|---:|---:|---:|---:|---:|
| baseline, top 10 direct | 77.47% | 64.45% | 0.511354 | 9.14% | 0.00 |
| controlled, 9 direct + 1 graph | 77.80% | 64.78% | 0.511679 | 9.20% | 0.19 |
| baseline, top 5 direct | 67.12% | 53.32% | 0.497396 | 15.01% | 0.00 |
| controlled, 4 direct + 1 graph | 67.84% | 54.04% | 0.498828 | 15.18% | 0.10 |
| established, 8 direct + 2 graph | **78.78%** | **66.15%** | **0.512786** | **9.34%** | 2.00 |

Bootstrap intervals are paired question-level 95% intervals, with 10,000 resamples and seed
`20260911`.

| comparison | complete-evidence delta | 95% interval | rescues | regressions | net |
|---|---:|---:|---:|---:|---:|
| 9 direct + 1 graph vs top 10 | +0.33 pp | −0.07 to +0.78 pp | 8 | 3 | +5 |
| 4 direct + 1 graph vs top 5 | +0.72 pp | +0.20 to +1.30 pp | 15 | 4 | +11 |
| 8 direct + 2 graph vs top 10 | +1.69 pp | +0.65 to +2.80 pp | 49 | 23 | +26 |

The protected direct prefix was unchanged in all 1,536 controlled ten-item contexts and all
1,536 controlled five-item contexts. The 9 plus 1 policy made 293 replacements; the 4 plus 1
policy made 156 replacements.

## Category complete-evidence coverage

| category | top 10 | controlled 9 + 1 | top 5 | controlled 4 + 1 |
|---|---:|---:|---:|---:|
| 1 | 22.70% | 22.70% | 9.93% | 10.28% |
| 2, temporal | 76.01% | 75.70% | 66.98% | 66.98% |
| 3 | 31.52% | 32.61% | 26.09% | 25.00% |
| 4 | 77.65% | 78.24% | 65.64% | 66.94% |

## Decision

The 9 direct plus 1 graph arm does not clear the preregistered primary interval gate because its
confidence interval includes zero. The public 4 direct plus 1 graph arm clears that retrieval
gate with a small positive delta and no protected-prefix violations. The established 8 direct
plus 2 graph arm remains the strongest retrieval result in this run.

This does not authorize a production change. The controlled arms use a development calibration,
and answer correctness, citation support, and unsupported-claim behavior still require the
separate answer-stage replay.
