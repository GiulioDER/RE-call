# EnterpriseRAG project adaptive retrieval depth

**Date:** 2026-08-18  
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**RE-call base:** `55397af5`  
**Question:** Does increasing the deterministic project retrieval depth from `k=8` to `k=12`
recover expected project documents on held out questions without an unacceptable increase in
invalid documents?

## Arms

The baseline is the existing Voyage embedding plus lexical hybrid retrieval with `candidate_k=200`,
`k=8`, no reranker, no reasoning arm, and extractive output. The candidate keeps every setting
fixed and changes only `k` to 12. The candidate uses the same 23 question file at
`results/enterprise_rag/project_slices/confirmation.ids` and the same VPS2 index.

## Prediction and gate

The candidate is predicted to improve document recall and exact coverage because project questions
require a mean of 4.22 expected documents and the current lane has substantial missing-document
headroom. It passes retrieval confirmation only if mean document recall improves, exact coverage
does not fall, and the mean invalid-extra increase is at most 2.0 documents per question. Capture
stability, latency, and provider calls must also be reported.

No answer-quality or leaderboard claim follows from passing this retrieval gate. If it passes, run
the same cheap-mini answer reader and official evaluator on baseline and candidate. If it fails,
reject deterministic depth as a project promotion candidate and do not test larger `k` values on
the same confirmation rows without a new hypothesis.
