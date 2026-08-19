# ATM-Bench VPS2 Voyage reranker and reasoning run

Date: 2026-08-19

## Objective

Measure ATM-Bench retrieval on VPS2 with Voyage `rerank-2.5` and the RE-call
reasoning feature enabled. This run is the VPS2 control before the SPLADE GPU
arm. It must not modify the existing VPS2 RE-call checkout or its live
services.

## Frozen configurations

Run the same official ATM corpus and question files under two retrieval
profiles:

1. `fastembed:sentence-transformers/all-MiniLM-L6-v2` plus lexical hybrid and
   Voyage `rerank-2.5`.
2. `voyage:voyage-4-large` plus lexical hybrid and Voyage `rerank-2.5`.

Both use `candidate_k=200`, retrieval depths 1, 5, 10, 25, 50, and 100, and
the previously frozen development and holdout split. The reranker reorders the
full candidate pool before truncation.

Reasoning is enabled with `RECALL_REASONING=1`. The reasoning provider and model
identity must be recorded in the manifest. Reasoning failures must remain
visible and must not silently change the retrieval result.

## Metrics

Report official item recall, question hit recall, complete evidence recall, and
the short-answer list metrics separately. Also report reranker latency, Voyage
request counts, token or usage metadata when available, reasoning outcomes, and
provider failures. Do not combine reasoning output length with retrieval score.

## Selection rule

Choose the embedder and retrieval profile using development results only. Freeze
that choice before reading holdout results. The holdout and 31-question hard
file are confirmation only. No public submission or external leaderboard post
is authorized by this preregistration.

## Cost boundary

Voyage embedding and reranker calls are paid and are authorized only for this
VPS2 run. No Vast.ai instance is included here. SPLADE remains a separate GPU
arm and must use the same frozen query split and selection rule.

