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
