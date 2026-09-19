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
