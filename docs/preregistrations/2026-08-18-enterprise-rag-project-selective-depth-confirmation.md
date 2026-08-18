# EnterpriseRAG selective-depth confirmation amendment

**Date:** 2026-08-18  
**Parent preregistration:** `2026-08-18-enterprise-rag-project-selective-depth.md`  
**Development feature artifact:** `project_dev.retrieval_features.json`

## Development screening disclosure

The parent preregistration fixed the feature and threshold grid but did not fix a priority order
between its two feature candidates. The development screen is therefore exploratory for feature
selection and is disclosed here before confirmation. Among candidates satisfying the parent gate,
the highest development mean recall was obtained by `max_dense_score < 0.75`:

* baseline development recall: 53.77%
* selective development recall: 57.21%
* exact coverage: 23.53% in both arms
* mean invalid-extra delta: +1.24 documents per question
* expanded questions: 6 of 17

The eighth-hit feature at threshold 0.70 was also within the parent guardrail but had lower
development recall, 55.74%. The selected confirmation arm is frozen as `max_dense_score < 0.75`,
with baseline `k=8`, expanded `k=12`, `candidate_k=200`, no reranker, lexical hybrid retrieval,
and extractive output.

## Confirmation gate

Run the selected arm on the 23-question `project_slices/confirmation.ids` file with three repeated
captures. Compare it to the existing no-adaptation `k=8` artifact. The arm must improve mean
document recall, not reduce exact coverage, keep mean invalid-extra increase at or below 2.0,
remain capture-stable, and report latency and calls. A failed gate rejects selective depth. A
passing gate authorizes only a fixed cheap-mini answer comparison and never a top-five claim by
itself.
