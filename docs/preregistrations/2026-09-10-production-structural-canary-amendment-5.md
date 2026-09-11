# Amendment 5: remove obsolete graph cosine rejection

Recorded after the repaired paired query and before the next production mutation or measurement.
The query reached the graph and activated the authored `references` relation, but the candidate was
rejected by the serving checkout's obsolete seed-relative `cosine_admission` gate.

Apply only the `recall_mcp/service.py` change from calibrated graph reranking commit `ab68875` to
the production serving checkout. This preserves ordinary trust evaluation and changes graph
candidate ordering and admission to use the calibrated rerank path; it does not bypass calibration,
source authorization, temporal checks, relation direction checks, or the evidence fill policy.

After deployment, rerun the exact locked `off` and `one_hop` query with no changes to query, ranking
inputs, relation shape, or budgets. The canary remains valid only if both arms return the same active
generation, the baseline evidence is identical, and `one_hop` adds exactly `b.md` with an accepted
`references` relation. Remove all three canary sources and verify an empty inventory afterward.
