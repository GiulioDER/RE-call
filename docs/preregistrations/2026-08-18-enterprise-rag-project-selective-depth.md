# EnterpriseRAG project selective retrieval depth

**Date:** 2026-08-18  
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**RE-call base:** `55397af5`  
**Question:** Can a non-gold confidence gate recover part of the `k=12` project recall gain
without submitting twelve documents for every project question?

## Arms

The baseline is the existing Voyage embedding plus lexical hybrid retrieval with `candidate_k=200`,
`k=8`, no reranker, no reasoning arm, and extractive output. The candidate uses the same retrieval
pool and obtains the top 12 documents, but submits `k=12` only when the selected non-gold confidence
feature is below its fixed threshold. Otherwise it submits the baseline top 8 documents.

The feature candidates are the maximum dense cosine score among the fused candidates and the dense
score of the eighth hybrid-ranked hit. The fixed threshold grid is `0.55`, `0.60`, `0.65`, `0.70`,
and `0.75`. The development selection rule is deterministic: choose the smallest threshold that
improves mean document recall by at least 1 percentage point, does not reduce exact coverage, and
increases invalid extras by no more than 2 documents per question. If no threshold passes, reject
the selective-depth hypothesis. The selected threshold is then frozen for confirmation.

No expected document ids, answer facts, or competitor outputs are available at runtime. The
feature artifact records only question id, question type, retrieval scores, selected document ids,
stage timings, and configuration.

## Slices and gates

Use the frozen `project_slices/dev.ids` 17-question development split for threshold selection and
`project_slices/confirmation.ids` 23-question confirmation split for the held out test. Keep Voyage
embedding, lexical backend, `candidate_k=200`, no reranker, extractive output, and the index fixed.
Use three repeated captures for the selected confirmation arm.

The confirmation gate requires improved mean recall, no exact-coverage decrease, mean invalid-extra
increase no more than 2.0, stable captures, and recorded latency and provider calls. Passing this
gate authorizes a fixed cheap-mini answer comparison only. It does not authorize a leaderboard or
top-five claim.
