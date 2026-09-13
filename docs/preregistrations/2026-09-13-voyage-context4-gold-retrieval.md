# Preregistration: Voyage Context 4 gold retrieval comparison

Date locked: 2026-09-13

## Objective

Test whether Voyage Context 4 improves gold evidence retrieval over the already measured Voyage 4 control on the frozen LoCoMo benchmark. The objective is retrieval quality and gold evidence coverage, not answer generation quality.

## Dataset and protocol

* Dataset: `locomo10.json`, the frozen ten conversation set used by the Voyage 3 versus Voyage 4 benchmark.
* Dataset SHA256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
* Paired answerable questions: 1,536, after excluding adversarial and rows without evidence.
* Control: `voyage:voyage-4`.
* Treatment: `voyage-context:voyage-context-4`.
* Retrieval: production hybrid retrieval path, `candidate_k=20`, scored at depths 1, 3, 5, 10, and 20, with headline metric hit@5.
* Reranking: none in both arms. This isolates the embedding and contextualized indexing change.
* Storage: fresh, separate PostgreSQL tables and a separate tenant per conversation.
* Execution: VPS2, one embedding or indexing process at a time.

## Context 4 indexing rule

The treatment is deliberately a model plus input mode test rather than a pure flat embedding model swap. The control embeds each per turn markdown document independently through the existing Indexer. The treatment sends all pre chunked turns from one conversation as one Voyage Context 4 document input, preserving turn order, and sends each query through the Context 4 query input mode. No turn is dropped or concatenated locally.

The treatment must preserve the existing chunk text, dialog identifiers, metadata, and gold alignment. Voyage requires pre chunked requests without auto chunking to remain within its 32K token request context, so an oversized conversation is split into ordered groups using a conservative 60,000 character request budget. No turn is dropped or silently truncated. This is a provider bound, not a new chunking or retrieval rule.

## Predictions and gate

Primary hypothesis: Context 4 improves pooled hit@5 by at least 2.0 percentage points versus Voyage 4, with a paired bootstrap 95 percent interval excluding zero.

Secondary predictions:

* The largest lift will be in multi hop and temporal categories, where adjacent turns and conversation context can disambiguate short utterances.
* The treatment will reduce the count of misses whose first gold evidence is beyond rank 20.
* The treatment may trade away some single turn precision because the contextual representation can make nearby turns more similar.

The result is a positive retrieval lead only if the paired hit@5 delta is positive and its interval excludes zero. A result below the 2.0 point target is an informative partial result, not a success against the preregistered target.

## Implementation record

* Source runner: `scripts/run_locomo_embedder_comparison.py`.
* Context 4 implementation commit: to be filled and committed before the corrected measurement.
* Measurement artifact: `docs/results/2026-09-13-voyage-context4-followup.json`.
* Report: `docs/results/2026-09-13-voyage-context4-followup.md`.
