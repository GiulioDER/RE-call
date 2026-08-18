# EnterpriseRAG reranker answer comparison

**Date:** 2026-08-18  
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**RE-call base:** `55397af5`  
**Slice:** five project-related questions from `top5_slices/dev.ids`  

## Question

Does Voyage reranking improve official answer score when the answer model, prompt, context limit,
question ids, and index are held fixed?

## Arms

* `hybrid_baseline`: Voyage embeddings, lexical hybrid retrieval, `candidate_k=200`, `k=8`, no
  reranker, no reasoning arm.
* `voyage_reranker`: the same configuration with `voyage:rerank-2.5` and a 4,000 character
  reranker document limit.

Both arms use `openai/gpt-5-mini` through the configured OpenRouter provider, `answer-mode
openrouter`, the baseline answer policy, and the official evaluator with `--no-correction` and
`--skip-citation-stripping`. No reasoning expansion is used.

## Prediction

The reranker arm is predicted to improve project-related document recall and exact coverage, based
on the already measured retrieval-only result, and to improve or preserve the combined answer
score without increasing invalid extra documents. The prediction is considered unsupported if the
answer score does not improve, or if correctness or citation document alignment regresses.

## Gates

Report correctness, completeness, combined score, document recall, exact coverage, invalid extra
documents, answer latency, model calls, and provider metadata. Do not promote from this five
question screen. A positive screen only authorizes the same paired test on the held-out slice and
then a homogeneous 500-question official evaluation.
