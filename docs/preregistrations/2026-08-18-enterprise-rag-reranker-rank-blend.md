# EnterpriseRAG reranker rank blend

**Date:** 2026-08-18  
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**RE-call base:** `55397af5`  
**Question:** Can reciprocal rank blending preserve the hybrid recall that the pure Voyage
reranker lost on held out project questions while retaining its exact document set improvement?

## Arms

The baseline is Voyage embedding plus lexical hybrid retrieval with `candidate_k=200`, `k=8`, no
reranker, no reasoning arm, and extractive output. The candidate uses the same hybrid pool and
Voyage `rerank-2.5`, then combines the original hybrid rank and Voyage rank with reciprocal rank
fusion using the fixed constant 60.

The preregistered rank weights are `0.25`, `0.50`, and `0.75`. Pure Voyage reranking at weight
`1.00` is the already measured reference. The Voyage relevance score is never written into the
trust score. Only ranks are blended, and the original `ScoredChunk` objects remain unchanged.

## Slices and measurements

Run the three new blend weights on the frozen five-question `project_related` dev slice and the
23-question held out project confirmation slice. Use the same VPS2 index, query embedding cache,
question hash, `candidate_k`, `k`, 4,000 character reranker limit, and extractive answer mode.
Use one capture for the initial screen and three repeated captures for any candidate that reaches
confirmation. Runtime must not read gold fields.

Report document recall, exact coverage, invalid extra documents, paired gains and losses, capture
stability, reranker calls, latency, and provider cost status.

## Prediction and promotion gate

The `0.50` blend is predicted to improve or preserve confirmation document recall relative to the
no-reranker baseline, improve exact coverage relative to no reranking, and avoid the pure
reranker's `+0.22` invalid-extra increase. A candidate is rejected if confirmation mean recall is
lower than baseline, gains do not exceed losses, or invalid extras rise materially without a recall
or exact-coverage gain.

No answer-quality run is authorized unless a blend passes this retrieval confirmation. Passing the
retrieval gate is not a leaderboard claim. Any accepted retrieval configuration requires a fixed
cheap-mini answer comparison and then a homogeneous 500-question official evaluation before a top
five claim.
