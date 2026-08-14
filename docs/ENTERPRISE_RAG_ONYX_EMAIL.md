# EnterpriseRAG-Bench Onyx Email Draft

This is the submission email context for Onyx. It discloses the local mixed
judge run and asks Onyx to rerun the public answer file if they prefer a
homogeneous judge configuration.

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
| Local score summary | `https://github.com/GiulioDER/RE-call/blob/codex/enterprise-rag-bench/benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.judge_gpt54_mixed_default.summary.json` |
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

The local score below uses the official EnterpriseRAG metrics based evaluation
flow, including citation stripping and document correction. I need to disclose
one caveat: because of budget constraints, the local score is a mixed judge run.
Rows `qst_0001` through `qst_0214` were scored with `openai/gpt-5.4` medium
reasoning through OpenRouter. Rows `qst_0215` through `qst_0500` were scored
with `openai/gpt-5.4` through OpenRouter with reasoning disabled. The answer
file itself is public, so please rerun it under your preferred homogeneous judge
configuration if that is required for leaderboard inclusion.

Submitted score:

| Metric | Value |
| --- | ---: |
| Average correctness | `65.60` |
| Average completeness | `53.48` |
| Combined correctness and completeness | `48.03` |
| Average document recall | `77.48` |
| Average invalid extra documents | `6.94` |
| Corrected questions | `16` |
| Completed questions | `500 / 500` |
| Local result SHA256 | `7692d4936a54d57c15c4d2fe30f93acdc0418193b802430e560ba1b018b9dd31` |

The answer file is public and should be directly evaluable with your official
benchmark release plus the reproduction commands in the guide above.

Best,
Giulio

## Finalization Checklist

Before sending:

1. Push the branch so all links resolve.
2. If the PR is merged first, replace branch links with `main` links.
3. Optionally attach the final sanitized summary JSON.
