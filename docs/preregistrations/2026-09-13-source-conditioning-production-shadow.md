# Pre-registration: production shaped source conditioning shadow

**Date:** 2026-09-13  
**Status:** predicted, not yet measured  
**Training artifact:** `docs/results/2026-09-13-live-source-scoped-expansion.json`  
**Training artifact SHA256:** `531dfa31c4344df020ebda9be46e2b09365b0dd9c918aff6835f8c5b4e1bd666`  
**Validation population:** the committed 50 query memory gold set, comprising 22 answerable
queries, 25 essential facts, and 28 unanswerable controls  
**Query set SHA256:** `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f`  
**Fact labels SHA256:** `45c38731e138f6aed425635b78ee79b692e142f5ef040e0efffeb042ad47186b`  
**Validation generation:** `gen_808e6c2592494eebabf144eabb21f838`  
**Validation calibration:** `cal_bc65614ff5ba4471850d53384381d7d5`  
**Pipeline:** `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`  
**Corpus:** `079a63744f2a9ce44aae344a06f57c0f3d959499b674f66c95f3cfc16680e58a`  
**Embedder:** `voyage-context:voyage-context-4`, profile `voyage-context-4-v1`

## Question

Can the registered global source conditioned admission rule be expressed as a versioned,
production shaped shadow that never changes served evidence, and does its fixed fitted model retain
the gold and safety direction on the next Context 4 corpus refresh?

The preceding Context 4 audit disclosed these prior results on generation
`gen_18d5edd2e5e847c0af1ee37e40d27893`: served baseline 17 complete queries and 19 facts, global
alpha 0.08 at 18 complete and 20 facts, and one false answer in both arms. Alpha 0.15 reached 19
complete and 21 facts but was not the registered choice. Those results licensed this shadow work
and are not predictions. Full evidence is in
`docs/results/2026-09-13-live-source-scoped-expansion.md`.

The active generation and calibration above were observed before this preregistration with:

```bash
RECALL_ENV=production recall --tenant memory generation list
RECALL_ENV=production recall --tenant memory calibration list
```

The calibration is certified and carried forward from
`cal_23d8708ac550444fa4274ac617df0870` at corpus delta 0.0019255455712451862. Validation aborts on
any mismatch in the registered generation, calibration, pipeline, corpus, query digest, or label
digest.

## Fitted artifact contract

Fit one logistic source model on every source row in the frozen training artifact. The seven
features, standardization, per query weighting, class balance, unregularized intercept, L2
coefficient 1.0, 100 iteration limit, and `1e-8` convergence tolerance are unchanged from
`docs/preregistrations/2026-09-13-source-conditioned-chunk-admission.md`.

The fitted JSON artifact must contain and validate:

1. schema version and model identifier;
2. the ordered seven feature names;
3. means, scales, intercept, and coefficients as finite numbers;
4. alpha 0.08, raw cosine floor 0.30, reciprocal rank constant 60, candidate depth 20, and output
   budget five;
5. training generation, calibration, pipeline, corpus, query set digest, label digest, source row
   count, and training artifact digest;
6. compatible embedding profile `voyage-context-4-v1` and retrieval profile `fast`;
7. a SHA256 fingerprint over canonical JSON excluding the fingerprint field.

Loading refuses a changed fingerprint, wrong schema, reordered features, non finite number, wrong
alpha, changed retrieval parameters, or incompatible pipeline, embedding profile, retrieval
profile, or candidate depth. Corpus and generation are recorded training provenance but are not
serving compatibility keys because routine refreshes change them. The exact current serving corpus
and generation remain diagnostics on every shadow decision.

## Shadow execution contract

Add `RECALL_SOURCE_CONDITIONING_MODE=off|shadow`, default `off`. No `active` value exists in this
change. Shadow requires an explicit artifact path and a deterministic sample rate in `[0,1]`.
Sampling hashes the full query with SHA256 and compares the first 64 bits to the configured rate.
At rate one every query runs; at zero none runs.

For a sampled request, reuse the already computed query vector. Collect the same dense 20, lexical
20, and full fused trust pool used by the benchmark apparatus. Compute the seven source features,
apply the fixed artifact, and select at most five chunks using:

`adjusted_score = raw_cosine + 0.08 * (source_support - 0.5)`

Eligibility remains verdict `ok` or `low_confidence`, raw cosine at least 0.30, and adjusted score
at least the request's certified threshold. Validity, supersession, dependency, metadata, and
entailment rejections remain binding.

The ordinary retrieval result is returned byte for byte with respect to evidence identity,
ordering, verdict, and abstention. Shadow output is diagnostic only. It contains artifact and
lineage identities, selected count, baseline overlap count, would abstain, and SHA256 hashes of
selected chunk identifiers. It contains no candidate text, source path, raw chunk identifier, or
query.

Artifact load, compatibility, or shadow computation failure is isolated from serving. The ordinary
result is returned unchanged, the diagnostic reports a stable error code without exception text,
and a bounded metric increments. Source security contexts skip the shadow until the audit helpers
can carry the same authorization contract.

## Validation arms

Launch two new pinned MCP processes from one committed checkout:

1. `off`, with source conditioning disabled;
2. `shadow`, with the fitted artifact and sample rate one.

Run the same 50 queries through both. The off and shadow processes must return exactly the same
public evidence identities, ordering, verdicts, decisions, abstentions, and lineage. The private
benchmark audit is enabled only for validation and recomputes the candidate selection independently
from the emitted identifier hashes.

The candidate is scored with the frozen essential fact labels. The report records complete queries,
facts, source hits, context precision, false answers, exact shadow parity, shadow errors, internal
shadow span, total request time, and artifact identity.

## Predictions

1. Public serving parity will be exact for all 50 paired requests.
2. Shadow recomputation will match every emitted selected chunk hash and count.
3. The fixed source model will gain at least one complete query or one essential fact over the off
   baseline without losing on the other metric.
4. It will not increase false answers, will keep at most two of 28 false answers, and will keep
   context precision at least 0.55.
5. No shadow request will report an artifact or compatibility error.
6. The 100 public requests will complete in under 20 minutes. Shadow overhead is reported but has
   no promotion threshold because this implementation is intended for sampled observation, not
   synchronous active serving.

## Decision rule

`BUILD SAMPLED SHADOW` only when all six predictions hold. This licenses an opt in shadow deployment
at a low sample rate after review. It does not license active selection.

`REPAIR` when public parity, hash parity, artifact validation, or shadow error handling fails. Repair
requires another committed apparatus revision before remeasurement.

`GATE` when mechanics and safety pass but the candidate has no gold gain. Keep the implementation
off and expand independent gold before deciding whether to deploy shadow observation.

`CLOSE` when the candidate loses any complete query or fact, increases false answers, or falls below
0.55 context precision. Do not deploy it even as sampled shadow from this result.

The decision is evaluated before individual case analysis. This preregistration does not authorize
deployment, process restart, calibration, generation promotion, or route change.

## Reproduction command

After the fitter, artifact, shadow implementation, and runner are committed to a detached VPS2
checkout:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-shadow-<commit>'
$env:RECALL_SOURCE_COMMIT='<commit>'
python -u scripts/run_live_source_conditioning_shadow.py `
  --generation-id gen_808e6c2592494eebabf144eabb21f838 `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-13-live-source-conditioning-shadow.json
```

The result report must provide the final checkout, commit, artifact fingerprints, exact command,
and any pre-measure correction. Frozen numbers above are never edited after this commit.

<!-- frozen_above -->

## Result

### First attempt and apparatus correction

The first attempt returned `REPAIR` on 2026-09-13 because its parity helper compared the complete
evidence serialization, a stricter test than the registered identity, order, verdict, decision,
abstention, and lineage contract. It reported 47 of 50 full object matches, while all 50 independent
candidate hash comparisons matched and no shadow request errored. The preserved raw artifact is
`docs/results/2026-09-13-live-source-conditioning-shadow-attempt-1.json`, SHA256
`9617a3d993f15d925dcc45bb467fdcf49bc5da638c6e4d16f3270c41cb202a59`.

The first attempt used the exact reproduction command above with remote checkout
`/home/sentiment/recall-repos/source-conditioning-shadow-0368ccba` and source commit
`0368ccba18bb66093d1387e4bb51d5fa425ecce7`.

Before remeasurement, the apparatus now projects precisely the frozen parity fields and has a red
then green test proving that score only variation is ignored while evidence reordering still fails.
The first attempt remains a failed mechanical run and is not used for the final decision.
