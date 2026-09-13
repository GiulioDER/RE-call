# Voyage 4 versus local embedder follow up

Measured on 2026-09-13 using the preregistered protocol
`2026-09-12-embedder-local-followup`. Each arm used the frozen LOCOMO dataset, 10 conversations,
1,536 paired answerable questions, fresh isolated tables, `k=5`, `candidate_k=20`, and depth
curve `1,3,5,10,20`. Category 5 adversarial questions were excluded from gold retrieval scoring.

Preregistration commit: `3e693eba4a33c5d02eaa602cc49507635aab948d`.

## Primary result

The registered gate required a treatment hit@5 improvement of at least 2 percentage points over
Voyage 4, with a paired bootstrap interval excluding zero. No local treatment passed. All three
were below Voyage 4 and their hit@5 intervals excluded zero on the negative side.

| Treatment | Voyage 4 hit@5 | Treatment hit@5 | Delta | Bootstrap 95% interval | Rescues | Regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jina Embeddings v3 | 73.37% | 69.40% | -3.97 points | [-6.38, -1.69] | 52 | 113 |
| mxbai embed large v1 | 73.31% | 69.53% | -3.78 points | [-6.25, -1.50] | 54 | 112 |
| multilingual E5 large | 73.37% | 70.64% | -2.73 points | [-4.88, -0.65] | 47 | 89 |

The control values differ by at most 0.06 points between runs because the hosted Voyage calls are
repeated independently. The paired comparison remains valid within each run.

## Depth curve

| Treatment | Delta hit@1 | Delta hit@3 | Delta hit@5 | Delta hit@10 | Delta hit@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jina Embeddings v3 | -3.58 | -3.45 | -3.97 | -3.19 | -1.50 |
| mxbai embed large v1 | -3.38 | -2.08 | -3.78 | -2.99 | -0.59 |
| multilingual E5 large | +0.39 | -1.04 | -2.73 | -2.54 | -1.56 |

The local models do not recover the gap by widening the first stage to rank 20. Jina v3 and E5
remain below Voyage 4 at hit@20. mxbai is close at hit@20, but still loses substantially at the
actual final budget of five.

## Category hit@5

| Category | Voyage 4 control | Jina v3 | mxbai large | multilingual E5 |
| --- | ---: | ---: | ---: | ---: |
| cat1 | 70.21% | 68.79% | 65.60% | 65.60% |
| cat2-temporal | 79.13% | 74.14% | 75.08% | 74.77% |
| cat3 | 51.09% | 43.48% | 50.00% | 46.74% |
| cat4 | 74.67% | 70.63% | 70.87% | 73.37% |

Voyage 4 leads every category in every paired run except that mxbai is close on cat3. The largest
local regressions are in cat1 and cat4, which are important because they carry most of the question
mass.

## Miss attribution

| Treatment | Gold in ranks 1 through 5 | Gold first in ranks 6 through 10 | Gold first in ranks 11 through 20 | Absent by 20 |
| --- | ---: | ---: | ---: | ---: |
| Voyage 4 with Jina control | 1,127 | 148 | 83 | 178 |
| Jina v3 | 1,066 | 160 | 109 | 201 |
| Voyage 4 with mxbai control | 1,127 | 148 | 83 | 178 |
| mxbai large | 1,068 | 162 | 120 | 186 |
| Voyage 4 with E5 control | 1,126 | 149 | 84 | 177 |
| multilingual E5 | 1,085 | 151 | 98 | 202 |

The local models lose both shallow ranking and candidate recall. This is not a reranking problem
that can be repaired after the first stage.

## Operational notes

All three models ran through the installed FastEmbed 0.8.0 path on VPS2. Jina v3 used substantially
more CPU time than mxbai and E5 because its local ONNX artifact is larger. The E5 run emitted the
FastEmbed warning that its pooling behavior has changed to mean pooling in the installed version;
the result records the installed behavior and should not be compared with an older E5 benchmark
without pinning the pooling implementation.

## Decision

Keep Voyage 4 as the first stage. Do not replace it with any of these local models for quality.
E5 is the most reasonable local fallback when avoiding cloud egress matters, and mxbai is a useful
compact control, but neither is a quality improvement. Jina v3 is not attractive on this host due
to both lower retrieval quality and much slower CPU execution.

This run did not test Voyage Context 4, Jina v5, Qwen3, Cohere, or non Voyage rerankers because the
current VPS2 resolver and credentials do not expose those models. The next high value experiment is
to keep Voyage 4 embeddings, widen the candidate pool, and compare rerankers with gold in pool
reported separately. That requires adding or provisioning the relevant reranker adapters and a new
preregistration.

## Raw artifacts

* [Jina v3 JSON](2026-09-12-voyage4-vs-jina-v3.json), SHA256 `A03E889C3322FEC3B9F1DCCB7F63288D07C57D0B3281F93B3800FB981BDFF225`.
* [mxbai JSON](2026-09-13-voyage4-vs-mxbai.json), SHA256 `C38C7A47EDEE53E832066E21D2D3D1AB28D09988910CF2448F8F0926557CE562`.
* [multilingual E5 JSON](2026-09-13-voyage4-vs-e5-large.json), SHA256 `7C389F425913D33EBF37DA9C19D3AA645BE48F3AB8E62A62638DD27F23222435`.

