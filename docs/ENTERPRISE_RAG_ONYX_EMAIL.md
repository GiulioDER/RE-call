# EnterpriseRAG-Bench Onyx Email Draft

This is the submission email context for Onyx. Fill in the final default
evaluator score after the in progress VPS2 run finishes.

## Recipient

`joachim@onyx.app`

## Subject

RE-call submission for EnterpriseRAG-Bench leaderboard

## Draft

Hi Joachim,

I would like to submit RE-call for the EnterpriseRAG-Bench leaderboard.

Public verification material:

| Item | Link |
| --- | --- |
| Pull request | `https://github.com/GiulioDER/RE-call/pull/282` |
| Submission guide | `https://github.com/GiulioDER/RE-call/blob/codex/enterprise-rag-bench/docs/ENTERPRISE_RAG_SUBMISSION.md` |
| Answer file | `https://github.com/GiulioDER/RE-call/blob/codex/enterprise-rag-bench/benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.answers.jsonl` |
| Answer manifest | `https://github.com/GiulioDER/RE-call/blob/codex/enterprise-rag-bench/benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.answers.manifest.json` |
| Evaluation cache builder | `https://github.com/GiulioDER/RE-call/blob/codex/enterprise-rag-bench/scripts/enterprise_rag_build_eval_cache.py` |
| OpenRouter scoring helper | `https://github.com/GiulioDER/RE-call/blob/codex/enterprise-rag-bench/scripts/enterprise_rag_score_openrouter.sh` |

Answer artifact provenance:

| Field | Value |
| --- | --- |
| System | RE-call |
| Answer generator | `openai/gpt-4o` through OpenRouter |
| Embeddings | `voyage:voyage-4-large` |
| Sparse retrieval | Postgres lexical plus SPLADE |
| SPLADE model | `prithivida/Splade_PP_en_v1` |
| Reranker | `voyage:rerank-2.5` |
| Candidate depth | `candidate_k=200`, final `k=8` |
| Answer file SHA256 | `05d01db6ee9350aaf9093b7bcac63fbbcdbfc4e7af3f2608b67cd8c8065c35ac` |
| Questions SHA256 | `f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905` |
| Documents ZIP SHA256 | `9d1174928696ad08bc15f3f104739519de633c1605a4ec2034e0e3c0087bc5cd` |
| RE-call revision | `858e1af6870a93aadec859a6d71b6ec807fcaf72` |

Evaluation disclosure:

The full default evaluator run uses the official EnterpriseRAG metrics based
evaluation flow, including citation stripping and document correction. For cost
control, I ran the judge as `openai/gpt-5.4` through OpenRouter with reasoning
disabled and single worker execution. I am declaring that judge setting so you
can decide whether to accept this score directly or rerun the public answer file
with your preferred judge settings.

Submitted score:

| Metric | Value |
| --- | ---: |
| Average correctness | `TODO_FINAL_CORRECTNESS` |
| Average completeness | `TODO_FINAL_COMPLETENESS` |
| Combined correctness and completeness | `TODO_FINAL_COMBINED` |
| Average document recall | `TODO_FINAL_RECALL` |
| Average invalid extra documents | `TODO_FINAL_INVALID_EXTRA_DOCS` |
| Corrected questions | `TODO_FINAL_CORRECTED_QUESTIONS` |

The answer file is public and should be directly evaluable with your official
benchmark release plus the reproduction commands in the guide above.

Best,
Giulio

## Finalization Checklist

Before sending:

1. Replace every `TODO_FINAL_*` value with the finished default evaluator score.
2. Push the branch so all links resolve.
3. If the PR is merged first, replace branch links with `main` links.
4. Attach or link the final sanitized summary JSON if it is committed.
