# EnterpriseRAG top-five category-aware reader experiment

**Date:** 2026-08-18
**Status:** preregistered, execution pending
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`
**RE-call base:** `55397af5`

## Question

Can an answer-side evidence policy improve EnterpriseRAG-Bench score in categories where the
current RE-call run retrieves evidence but loses correctness or completeness during synthesis?

This experiment is separate from retrieval expansion. It does not inspect `expected_doc_ids`,
`answer_facts`, or gold answers at runtime. The policy receives only the question type, retrieved
chunks, and the document ids that will be submitted to the official evaluator.

## Arms

* `baseline`: current RE-call evidence order and prompt.
* `category_aware`: keep the submitted document ids and answer context aligned, prefer one
  representative chunk per selected document for multi-document categories, and add a bounded
  reader instruction for project-related, completeness, conflicting, constrained,
  intra-document, and high-level questions.

The existing retrieval configuration, answer model, question order, index, `k`, candidate pool,
and context limit remain fixed between arms.

## Slices

The weak-category confirmation set is generated from the official question release with:

```powershell
py -3 scripts/enterprise_rag_make_slices.py `
  --questions .benchdata/enterprise-rag-v1.0.0/questions.jsonl `
  --question-types project_related,completeness,conflicting_info,constrained,semantic,basic `
  --seed 366 `
  --out-dir results/enterprise_rag/top5_slices
```

The `dev.ids` file is for prompt and context-policy iteration. The `confirmation.ids` file is
held out until the policy is frozen. The full 500-question run is the final confirmation and must
use a separate evaluator result file.

## Metrics and gates

The primary metric is the official evaluator's per-question correctness multiplied by
completeness, summarized overall and by question type. The comparison script reports paired
question deltas and a deterministic bootstrap interval.

Secondary metrics are document recall, exact document coverage, invalid extra documents,
correctness, completeness, answer context document alignment, latency, tokens, and provider
calls. The candidate is rejected if it improves score by adding unsupported documents, increases
invalid extra documents materially, breaks the info-not-found category, or loses the deterministic
fallback path.

No result from the development slice will be presented as a leaderboard result. A promotion
requires an improvement on the held-out confirmation slice and then a homogeneous official
500-question evaluation under the same judge configuration as the baseline.
