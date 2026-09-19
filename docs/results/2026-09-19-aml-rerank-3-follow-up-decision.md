# AML Voyage rerank-3 follow-up decision

## Why a follow-up is required

The completed clean comparison validly selected `B0_raw` over Voyage `rerank-2.5`. After that
result was measured, Voyage's current model table was checked and found to list preview
`rerank-3` as its highest-accuracy reranker and recommendation for most applications. The existing
result must therefore be read as a model-specific rejection, not a rejection of all reranking.

Primary sources checked on 2026-09-19:

* [Voyage reranker models](https://docs.voyageai.com/docs/reranker)
* [Voyage pricing](https://docs.voyageai.com/docs/pricing)

The pricing page lists `rerank-3` at USD 0.05 per million processed tokens and a 200 million token
allowance. The experiment records a cost estimate and never assumes that an account allowance
applies.

## Clean comparison

The follow-up compares:

1. `B0_raw`, reranker OFF.
2. `B2_raw_rerank3`, identical raw retrieval followed by Voyage `rerank-3`.

Both arms use the existing five immutable `r2` tenants. Corpus bytes, embeddings, namespaces,
generation, table, queries, candidate width, RRF constant, result budget, and metrics remain fixed.
No Delete, Add, embedding, or indexing operation is authorized. Exact corpus hashes must pass
before Search begins.

The replay is staged. Present runs first with 34 queries, three captures, and both arms. A failed
present gate selects B0 and stops. Only a passing candidate may continue through absent,
superseded, contradictory, and adjacent. The retrieval, task-screen, and confirmation gates remain
the same as preregistration 089, with exact `rerank-3` identity replacing `rerank-2.5`.

## Boundaries

`rerank-3` is in preview. A missing exact identity, API rejection, fallback, invalid permutation,
or unexplained corpus mismatch invalidates the candidate. Provider availability is not evidence of
AML eligibility.

No official AML run is authorized. Voyage embedding eligibility still requires the organizer's
written confirmation. If the follow-up selects B0, reranking is OFF for the local candidate. If it
selects B2 but reranking is disallowed, the official candidate remains B0. If Voyage embeddings are
disallowed, both official runs remain blocked.

## Pre-execution implementation checkpoint

The product now registers `B2_raw_rerank3` separately from the unchanged `B1_raw_rerank` arm.
Reranker model identity is owned by the registered behavior and is emitted consistently through
the version response, Search diagnostics, and provider client construction. The AML Search JSON
body remains unchanged.

A plausible mutation that substituted `rerank-2.5` for the B2 model turned the exact identity test
red. After restoration, the focused suite completed with 61 passed and one skipped using exactly
three pytest workers. Ruff and `git diff --check` passed.

## Pre-execution repair 1

The first Stage 1 wrapper invocation stopped before service setup, schema work, provider readiness,
corpus access, or Search because the setup script's dedicated-path allowlist did not yet include
the new `/home/sentiment/recall-repos/aml-rerank3-*` worktree prefix. The repair adds only that
bounded prefix. It does not broaden variants, corpus access, identities, metrics, or gates.

The empty first-attempt result directory remains preserved at
`/home/sentiment/agent-memory-bench-rerank3-b6ca8ed/results/aml-rerank3-follow-up-v1/b9e2e899-b6ca8ed-present`.
The valid retry must use a new `r2` output path.

## Pre-execution repair 2

Before the retry, the shared `recall-aml-experiment.service` was found serving an independent
multiview experiment at commit `016a11eb`. No owning replay process remained, but the service may
still be expected by separate work. It was not stopped, restarted, or repointed.

The rerank-3 wrapper now uses the dedicated `recall-aml-rerank3.service`, runtime environment, and
port 18005. The setup interface accepts this instance only for B0 and B2. This is execution
isolation, not a change to corpus, retrieval, model, metrics, gates, or stopping rules.

## Pre-execution repair 3

The first dedicated-service invocation stopped before Search because its initial corpus-status
request returned HTTP 401. The dedicated service read its credential from `rerank3.env`, while the
wrapper incorrectly retained the shared `clean-reranker.env` path. Both credentials were present,
validly shaped, and distinct. No Voyage reranking request ran.

The empty attempt remains preserved at
`/home/sentiment/agent-memory-bench-rerank3-b6ca8ed/results/aml-rerank3-follow-up-v1/b9e2e899-73e0e8d-present-r2`.
The wrapper repair aligns its client credential with the dedicated service and requires a fresh
`present-r3` output path. It changes authentication wiring only, without changing the experiment.

## Measured result

The valid present replay selected `B0_raw` and stopped the experiment. Exact Voyage `rerank-3`
attempted and completed all 102 B2 requests with zero fallback, zero invalid permutations, and a
full output permutation of every pretruncation input candidate. B0 made zero reranker attempts.

Mean reciprocal rank fell from 0.3226447419731786 for B0 to 0.2592229101832591 for B2. Complete
coverage at 100 rose from 0.9411764705882353 to 0.9705882352941176, while source-session recall
remained 1.0. Search p95 rose from 435.5875360779464 ms to 1044.657205697149 ms, a ratio of
2.3982715738455207. The estimated B2 reranker cost was USD 0.313107975, not a provider invoice.

The strict present MRR improvement gate failed. The selector set `stage2_authorized` to false.
No replay on the other four conditions, executable task cell, DeepSeek solver call, GPT 4o mini
solver call, or official AML run was authorized.

The result rejects both tested rerankers for this frozen application. `rerank-2.5` and `rerank-3`
each reduced present MRR when applied to the full fused candidate set. `B0_raw` remains the local
AML coding candidate, with reranking OFF. Organizer confirmation for Voyage embedding eligibility
remains mandatory before any official run.

Served RE-call commit:
`d8833569cc9a5c628b32333baa6f3cf8b4944f9f`.

AMB harness commit:
`dd979d48bbe9027aec7f59f7dbcc45974668d486`.

Selection SHA-256:
`56e1ee688101cfecdf7d535f98b9357cdde38bb5569d5e9e865695a4f0cf9449`.

The machine-readable selector is
[`2026-09-19-aml-rerank-3-follow-up-selection.json`](2026-09-19-aml-rerank-3-follow-up-selection.json).
The raw records and service logs remain immutable on VPS2 under
`/home/sentiment/agent-memory-bench-rerank3-b6ca8ed/results/aml-rerank3-follow-up-v1/d8833569-dd979d4-present-r3`.
