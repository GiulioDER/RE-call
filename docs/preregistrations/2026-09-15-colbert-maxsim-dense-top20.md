# ColBERT MaxSim selector over the original dense top 20

Status: predicted, not yet measured.

Registered 2026-09-15 before implementing the experiment runner, generating any ColBERT score,
or inspecting any row level outcome for this experiment.

## Question

The consumed empty base cohort already contains exact evidence in the original dense top 20 for
10 of 15 answerable rows, but only three have exact evidence at rank one. Lexical query
construction moved many candidates without increasing exact reach. Can the existing free local
`LateInteractionReranker`, using token level ColBERT MaxSim, improve exact evidence rank within
the fixed original dense candidate set?

This is a selector screen on already consumed development data. It cannot authorize serving or
spending a fresh holdout.

## Frozen population and lineage

Use only the 30 rows with an empty served base in the completed query anchor holdout, split into
15 answerable rows and 15 matched unanswerable controls. The query pool SHA256 is
`6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`. The completed holdout
SHA256 is `58b883af8d1527ef137913762f9a3a456198c8f818d8286bfc87a248c4d3fa5b`.

Pin generation `gen_2ccf2130f6c64d99a11a6bcb6f929dd8`, calibration
`cal_e50dac493112488ea5e7cf79d86c0099`, pipeline fingerprint
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus fingerprint
`f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312`.

The apparatus must reproduce the original dense gold source counts at ranks 1, 3, 5, 10, and 20
as 6, 6, 7, 8, and 10, and the exact span counts as 3, 3, 5, 8, and 10. It must also reproduce 14
answerable rows and zero controls passing the existing corpus supported three anchor safety gate.
Any mismatch in population, lineage, audit availability, or these aggregates is an apparatus
failure without a quality interpretation.

## Frozen selector

For each row, retrieve the complete original question once through the same pinned MCP server with
the retrieval leg and source admission benchmark audits enabled. Recover the original dense list
from that audit. Do not issue a transformed or auxiliary retrieval query.

Apply the existing corpus supported anchor safety gate to the complete original audited pool.
Rerank only a row for which exactly three safe anchors exist and none has source document frequency
zero. Leave every ineligible row in its original dense order. This gate must admit zero controls.

For each eligible row, take exactly the first 20 original dense chunks. Score the complete original
question against the exact stored chunk text with `colbert-ir/colbertv2.0` through
`LateInteractionReranker` and its distinct `query_embed` and `passage_embed` paths. Compute the
existing MaxSim score, which sums each query token's maximum dot product against document tokens.
Sort by descending MaxSim score. Resolve equal scores by preserving original dense order. Empty
document encodings sort last through `maxsim_or_last`; an empty query is an apparatus failure.

MaxSim may only reorder these 20 candidates. It must not widen retrieval, add a chunk, change chunk
text, use gold labels, use source names, use the answer span, generate text, tune a threshold, or
replace the stored dense cosine. Each reordered `ScoredChunk` keeps its original dense score for
downstream trust semantics.

All ColBERT embedding and scoring must run on VPS2 under the shared embedding lock and the existing
bounded process controls. Check both the lock and competing processes before starting. The private
score artifact must record the checkpoint and FastEmbed version. No model scoring may run on the
workstation.

## Measurements

Report original and MaxSim ordered gold source and exact span reachability at ranks 1, 3, 5, 10,
and 20 over all 15 answerable rows. Report eligible answerable and control counts. Exact and gold
source top 20 reachability must remain 10 because candidate membership is frozen.

Report how many eligible rows changed rank one, how many changed rank one candidates contain the
exact span, how many come from a gold source, exact precision among changed rank one candidates,
and gold source precision among changed rank one candidates. Report answerable rows whose minimum
exact rank improved, tied, or worsened. The primary outcome is exact rank, not source identity.

Private output may contain row identifiers, questions, chunk text, chunk IDs, raw MaxSim scores,
and gold booleans but must remain outside the repository. Public output contains aggregate counts
only.

## Prediction and decision

I predict token level MaxSim will raise exact rank one coverage from three rows to at least five
without reducing exact top 5 below five. I also predict at least two changed rank one candidates
will contain the exact span and that exact precision among all changed rank one candidates will be
at least 50 percent. The safety prediction is zero eligible controls.

Proceed to a fresh source disjoint selector validation only if every prediction above passes and
the frozen top 20 membership invariants hold. Otherwise stop generic reranking on this cohort. Do
not tune the model, pool width, score transformation, tie rule, eligibility gate, or thresholds
after measurement. On failure, move to a separately preregistered span grounded extractive reader
with an explicit null outcome, or revisit corpus and gold construction.
