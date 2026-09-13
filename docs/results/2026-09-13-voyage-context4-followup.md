# Voyage Context 4 gold retrieval comparison

Measured 2026-09-13 on the frozen LoCoMo ten conversation benchmark.

## Result

| Arm | Embedder | Questions | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | Voyage 4 | 1,536 | 45.51% | 64.58% | 73.37% | 83.01% | 88.41% |
| Treatment | Voyage Context 4 | 1,536 | 48.31% | 70.05% | 79.62% | 88.67% | 92.38% |
| Paired delta | Context 4 minus Voyage 4 | 1,536 | +2.80 points | +5.47 points | +6.25 points | +5.66 points | +3.97 points |

The preregistered primary gate passed. The paired hit@5 delta was +6.25 percentage points, with a bootstrap 95% interval of [+3.65, +8.92] points. Context 4 rescued 151 questions and regressed 55, for a net gain of 96 questions.

## Category detail at hit@5

| Category | Voyage 4 | Context 4 | Delta |
| --- | ---: | ---: | ---: |
| cat1 | 70.21% | 73.76% | +3.55 points |
| cat2 temporal | 79.13% | 82.87% | +3.74 points |
| cat3 | 51.09% | 50.00% | -1.09 points |
| cat4 | 74.67% | 83.59% | +8.92 points |

The strongest effect is cat4, followed by cat2 temporal and cat1. Cat3 did not improve, so Context 4 is not a universal fix for the hardest category.

## Miss depth

The number of questions whose first gold evidence was beyond rank 20 fell from 178 to 117, a reduction of 61 questions. The depth curve also improved at every measured cutoff, which indicates a ranking improvement rather than only a cutoff specific effect.

## Protocol and implementation

Both arms used the production hybrid retrieval path with `candidate_k=20`, no reranker, separate fresh PostgreSQL tables, and separate tenants per conversation. The dataset SHA256 was `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

Voyage Context 4 is not a pure flat embedding model swap. The control embeds each per turn markdown document independently. The treatment sends ordered pre chunked turns in grouped Context 4 document inputs and embeds queries with Context 4 query mode. Voyage documents that each inner list is one document and that document chunks are returned in input order. The provider limits pre chunked requests without auto chunking to 32K tokens, so oversized conversations were split into ordered groups under a conservative 60,000 character request budget. No turns were dropped or locally concatenated.

The benchmark implementation is in `scripts/run_locomo_embedder_comparison.py`, source commit `2eadd573724d7369fb0a66b55aac9de37ae2e419`. The preregistration is `docs/preregistrations/2026-09-13-voyage-context4-gold-retrieval.md`.

Raw artifact SHA256: `62494e579d41c1a7328fc21fcd7447c0b1432f7f6303d1582dee9e0b50bfdd53`.
