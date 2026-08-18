# EnterpriseRAG candidate pool 400 reranker limit amendment

**Date:** 2026-08-18  
**Parent preregistration:** `2026-08-18-enterprise-non-reasoning-reranker.md`

## Reason for amendment

The preregistered `candidate_k=400` Voyage reranker arm with the fixed 4,000 character document
truncation failed before producing an answer file. Voyage rejected the request because the batch
contained 666,493 tokens, above its 600,000 token limit. This failure is retained as evidence and
is not replaced.

## Additional arm

I will run the same completeness confirmation retrieval configuration with `candidate_k=400`,
`k=8`, lexical sparse retrieval, Voyage `rerank-2.5`, and a 2,000 character reranker document
truncation. The no reranker `candidate_k=400` result and the repeated `candidate_k=200` baseline
remain the paired references. The answer model remains disabled.

## Gate

This is a diagnostic fallback, not a promotion candidate unless it improves paired recall or exact
coverage without a material invalid extra or latency increase, remains capture stable, and has no
document id mismatch. Any positive result requires a separate repeated capture before answer
generation. Retrieval cost remains recorded as unavailable if the provider does not expose it.
