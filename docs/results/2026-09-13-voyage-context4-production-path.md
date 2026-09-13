# Voyage Context 4 production path handoff

## Decision status

The Voyage Context 4 production path is implemented and has passed the isolated production path benchmark. Production promotion was not performed. No deployment, route change, fresh production calibration, or generation deletion was performed.

The benchmark supports a staged rollout recommendation, subject to a fresh calibration and shadow validation on a production manifest. The current evidence is sufficient to continue to controlled staging, not to promote directly.

## Preregistered evidence

The measurement protocol was committed before the benchmark at `83926389` in `docs/preregistrations/2026-09-13-voyage-context4-production-path.md`. The final benchmark runner revision was `f7b6a21b`.

The benchmark used the frozen 10 conversation LOCOMO dataset on VPS2, with SHA256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`, 1,536 paired answerable questions, candidate k 20, retrieval depths 1, 3, 5, 10, and 20, and hit at 5 as the primary metric. The benchmark database was the isolated throwaway database `recall_context4_bench_20260913b`. No serving database writes occurred.

The result artifact is `docs/results/2026-09-13-voyage-context4-production-path.json`, SHA256 `95b8e6fcb15e59368566b85fb688f488bb07a216cd0dd227f42e192e3993f70f`.

## Production path results

The four arms used the registered production profiles. Percentages below are over the same 1,536 questions.

Voyage 4 scored hit at 1 46.16%, hit at 3 65.30%, hit at 5 73.37%, hit at 10 83.01%, and hit at 20 88.28%.

Context 4 scored hit at 1 48.24%, hit at 3 69.86%, hit at 5 79.23%, hit at 10 88.41%, and hit at 20 91.99%.

Voyage 4 with the reranker scored hit at 1 68.16%, hit at 3 84.05%, hit at 5 87.24%, hit at 10 90.30%, and hit at 20 91.54%.

Context 4 with the reranker scored hit at 1 68.62%, hit at 3 85.94%, hit at 5 89.97%, hit at 10 93.16%, and hit at 20 94.40%.

The paired Context 4 delta without reranking was +5.86 percentage points at hit at 5, with bootstrap 95% interval +3.26 to +8.53 points. It produced 146 rescues and 56 regressions.

With the existing `voyage:rerank-2.5` stage, the paired delta was +2.73 percentage points, with bootstrap 95% interval +1.11 to +4.43 points. It produced 63 rescues and 21 regressions. The preregistered positive direction and interval above zero were met in both comparisons.

Without reranking, category hit at 5 deltas were cat1 +3.55 points, cat2 temporal +3.74 points, cat3 minus 2.17 points, and cat4 +8.32 points. With reranking, the corresponding deltas were +2.13, +2.49, minus 3.26, and +3.69 points. Cat3 remains the principal quality risk.

The artifact contains answer rows, paired bootstrap results, category summaries, profile identities, and generation IDs. It does not contain provider request counts, retry counts, or per arm wall clock timing. Those secondary operational measurements remain open and should be captured during staging shadow validation.

## Profile identity and generations

Voyage 4 used profile `voyage-4-v1`, dimension 1024, profile fingerprint `c683b7cbda24317be1c1c9540b8f4b577bf40d2ba8dd1409cd181c047b665e59`.

Context 4 used profile `voyage-context-4-v1`, dimension 1024, profile fingerprint `0c04428cbe3edf9113bd9891e15df0cb7afaccc6acbd54b8c40a26d4d12d693f`. Its identity includes the contextualized document grouping policy, query mode, output dimension, and request limits.

The ten Voyage 4 generation IDs were `gen_be379d9fce2a4a548eb9dc481ba02a84`, `gen_4026cb04ce51475289cbfdf931493b65`, `gen_10d8325bc73f465bb89ebc89f3581a55`, `gen_92b8587000fc4ec2b06a52a26d5302af`, `gen_41b944dac15d4a14bd108b3bc81dcf55`, `gen_f8e698c88b1b41b0a9a1f9c31ec335d0`, `gen_eaafbeb4dfdb4ce9bfcc15df1bd5371c`, `gen_d5fa867b6e2c47758f979f190da35b38`, `gen_6228097573ce47b4889bfa4af4e20844`, and `gen_1d04160ae35f486a810cbbc0b129feb0`.

The ten Context 4 generation IDs were `gen_b56349c5414b469d9fc42831a4147efd`, `gen_99a29ab1687a4587b1ea89f91ce45a78`, `gen_d80d2aced84546898a2a4c2b626834b6`, `gen_d3a9887b068c460e979820a85772656e`, `gen_6d307305d1f7424fb7ab66f48b6f04a9`, `gen_2f1244b700414c088ea32633cf6fcd15`, `gen_cc3e96ef8aaa4542a7a70f0e74cba88e`, `gen_e3fd0487ecf84bf99a9bf5e53fbc0eb9`, `gen_ca24de0858f645ea93bb0215b026b0ca`, and `gen_f3a72408b3ff4a659b54c114bde27225`.

The reranker arms reused their corresponding no reranker generations. No duplicate vector generations were created for reranking.

## Implementation safety

The implementation adds a grouped document embedding contract and uses the official Voyage contextualized chunk API shape of one nested list per document group. Prechunked request limits follow the provider contract: at most 1,000 inputs, 32,000 tokens per prechunked request, 16,000 chunks, and the conservative 60,000 character split guard. A chunk is never truncated. Splitting occurs only between chunks, and response group counts and vector counts are checked against the exact ordered input groups.

Conversation turns are one contextual document group in the benchmark. Ordinary independent files remain independent groups unless an explicit context group callback connects them. Graph and multi file ingestion preserve an explicit group across file boundaries. Incremental reuse includes the group fingerprint, so a changed group cannot silently reuse a vector produced with a different ordered group. Passage text caching is refused for grouped document embedders because the vector depends on the complete group.

Grouped writes are staged until the full group succeeds, then sliced back to the exact source chunk boundaries. Source locks and tombstone checks are retained. A provider or validation failure before grouped writes leaves the previous active generation available. Generation admission remains single writer through the existing generation manager and validation path.

The provider client uses an explicit timeout and bounded retries. Provider failures are surfaced as build failures rather than partial generation success. Hosted profile identity records the provider model, dimension, query and document modes, grouping policy, request limits, and profile fingerprint. The provider is hosted and therefore not byte attestable.

The official provider contract used for these decisions is documented at [Voyage contextualized chunk embeddings documentation](https://docs.voyageai.com/docs/contextualized-chunk-embeddings).

## Serving route and calibration verification

The serving route was rechecked after the isolated benchmark and was unchanged:

`memory` active generation `gen_6aaffd1f9712404c8fa5cee5a6af748a`, previous generation `gen_b4159fc3f1f04b93833b1e4cd8dd97a`.

The published calibration remains `cal_fb9135955aae4950b22529ae27019b96`, bound to the active Voyage 4 generation. No fresh calibration was created for Context 4 because no production generation was built or promoted. The previous generation and its published calibration remain available.

## Staged rollout plan

1. Build a Context 4 candidate from the production manifest pipeline with a new production generation ID. Keep the current active and previous generations unchanged.

2. Validate manifest digest, corpus fingerprint, profile fingerprint, grouped source counts, vector alignment, query mode, dimension, provider error classification, and generation readiness.

3. Run a fresh calibration against the new generation and bind the calibration to that exact generation and profile fingerprint. Reject the candidate if calibration quality or separability gates fail.

4. Run shadow retrieval against the current route and the candidate. Capture request counts, retry counts, provider failures, latency, token metadata, hit at 1, 3, 5, 10, and 20, and cat3 regressions.

5. Promote only after the shadow gate and operational cost and latency gates pass. Rollback is a route pointer swap to `gen_6aaffd1f9712404c8fa5cee5a6af748a` or the preserved previous generation `gen_b4159fc3f1f04b93833b1e4cd8dd97a`. Retain both old generations until the retention decision is explicit.

## Remaining risks

1. Voyage hosted model weights are provider controlled and are not byte pinned. Profile and generation fingerprints prevent local identity drift but cannot attest provider weight bytes.

2. Cat3 regressed in both production path comparisons. This needs targeted shadow analysis before promotion.

3. Contextualized groups use more memory and provider work than independent text passage calls. The staging run must record request width, token counts, retries, and wall time.

4. The benchmark used isolated test generations and did not exercise a production calibration or route cutover. It is evidence for the build and retrieval path, not evidence that promotion is safe by itself.

5. The provider request limits and SDK behavior can change. The registered limits and response alignment checks should fail closed when the contract changes.
