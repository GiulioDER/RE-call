# EnterpriseRAG project k12 reader diagnostic

**Date:** 2026-08-18  
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**Purpose:** diagnostic answer-side comparison after the raw `k=12` retrieval arm failed its
invalid-extra guardrail.

## Question

Does the additional evidence returned by raw deterministic `k=12` improve official answer
correctness or completeness enough to justify researching a lower-noise document selection policy?
This is not a promotion test. It does not override the failed retrieval gate.

## Arms

Both arms use the same `openai/gpt-5-mini` answer model through the configured OpenRouter provider,
the same baseline answer prompt and `answer-policy baseline`, `max_context_chars=12000`, no
reasoning arm, no reranker, Voyage embeddings, lexical hybrid retrieval, `candidate_k=200`, and
the same 23-question project confirmation slice.

The baseline returns `k=8` documents. The candidate returns raw `k=12` documents. The official
evaluator runs with `--no-correction --skip-citation-stripping --parallelism 1` and the same cheap
mini judge configuration for both files. Runtime must not read gold fields.

## Measurements and interpretation

Report correctness, completeness, combined score, document recall, exact coverage, invalid extras,
answer latency, provider calls, parse failures, and per-question paired changes. The primary
diagnostic is the paired combined score difference. A gain is evidence for an answer-side context
research lane only; it is not evidence that raw `k=12` is safe to deploy. A loss or no gain closes
the raw-depth answer hypothesis.

Do not run a full 500-question evaluation from this diagnostic. Any follow-up must preregister a
new document selection or context packing policy and pass a held out retrieval and answer test.
