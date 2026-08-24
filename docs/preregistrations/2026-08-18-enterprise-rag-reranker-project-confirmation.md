# EnterpriseRAG reranker project confirmation

**Date:** 2026-08-18  
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**RE-call base:** `55397af5`  
**Slice:** project-related questions in `top5_slices/confirmation.ids`  

## Question

Does the Voyage reranker result measured on the five-question project dev screen reproduce on the
held-out project-related confirmation questions without increasing invalid document selection?

## Arms

* `hybrid_baseline`: Voyage embeddings, lexical hybrid retrieval, `candidate_k=200`, `k=8`, no
  reranker, no reasoning arm.
* `voyage_reranker`: the same retrieval configuration with `voyage:rerank-2.5` and a 4,000
  character reranker document limit.

The answer mode is extractive for this first confirmation. The comparison is retrieval-only and
does not use a model-generated query or reasoning expansion.

## Prediction

The Voyage reranker is predicted to preserve or improve mean document recall and exact document
coverage on the held-out project questions, with no material increase in invalid extra documents.
The effect is considered unconfirmed if the reranker loses on more questions than it gains, or if
extra documents rise materially while recall is unchanged.

## Gates

Report paired document recall, exact coverage, invalid extras, per-question gains and losses,
reranker latency, provider calls, and index configuration. A positive retrieval confirmation
authorizes a separate answer-quality comparison. It does not authorize a full leaderboard claim;
that requires a homogeneous 500-question official evaluation.
