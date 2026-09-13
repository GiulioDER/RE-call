# Pre-registration: source conditioning buried fact challenge

Date registered: 2026-09-13.

## Question

Can the fixed global source conditioning model complete decisive evidence from late passages of
long memos after ordinary serving has already identified the correct source?

The first independent holdout was directionally positive but saturated: the control covered
`17` of `18` facts, leaving only one target case. This second holdout changes the population, not
the model. It selects longer, disjoint sources and facts that occur late in those sources. No
retrieval output from either arm was inspected while constructing this set.

## Frozen substrate and treatment

The run uses memory generation `gen_18d5edd2e5e847c0af1ee37e40d27893`, calibration
`cal_23d8708ac550444fa4274ac617df0870`, pipeline fingerprint
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus fingerprint
`c054e26b28e62760c0fefa64815da8033433e94c2474f079ed8a2aec1b93fb1a`.

The treatment remains
`docs/results/2026-09-13-source-conditioning-model.json`, SHA256
`fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f`, fingerprint
`9c3e4a2d1a56d4c3bf4d77800dbdf9b02261549199af8ddbeb4ac11aafdc66c7`. Its fixed settings remain
alpha `0.08`, raw cosine floor `0.30`, candidate pool `20`, and item budget `5`.

Both arms use one request and one query vector. The control is public trusted evidence. The
treatment applies the existing artifact to the request's captured dense, lexical, and fused traces.
Graph expansion is off. No coefficient, threshold, floor, candidate depth, or budget may change.

## Frozen challenge set

The query file is
`docs/preregistrations/2026-09-13-memory-buried-fact-challenge-queries.json`, SHA256
`9eefe96c8bb07cb71b2bde71f359c222f6966e488c85097249e5536b6b074794`.

The fact file is
`docs/preregistrations/2026-09-13-memory-buried-fact-challenge-facts.json`, SHA256
`9e824f03caef370490f9be53b16451bc8fa29ee8d616aff7ff05351f6d0b86c5`.

The set contains `36` questions, split equally between answerable questions and unanswerable
controls. Each of the `18` answerable questions has one essential fact coverable by a single
canonical chunk. Every selected source has at least `7` canonical chunks, and each fact has a
supporting chunk in the source's final quarter. The `18` sources overlap neither the model fitting
gold sources nor the first independent holdout sources.

The population is intentionally a challenge set, not a prevalence estimate for all user questions.
Its purpose is to create enough correct source but missed passage cases to test the mechanism.
Labels are author constructed and mechanically validated, not independently human adjudicated.

## Primary population and metrics

The target population is selected from the control arm alone: answerable queries where public
evidence contains the gold source but omits the essential fact.

Primary metrics are target fact rescues, overall complete query change, and overall covered fact
change. Guardrails are complete and fact regressions by query, unanswerable answered count, labeled
context precision, target source losses, source hits, exact hash parity, immutable lineage, and
treatment liveness.

## Predictions

1. The challenge set will contain at least `4` target queries.
2. Fixed alpha `0.08` will rescue at least `1` target fact.
3. Treatment will lose no complete query and no fact that the control covered.
4. Treatment will not increase the number of answered unanswerable controls.
5. Treatment context precision will be no more than `0.05` below control.

## Apparatus validity

Return `REPAIR` unless file hashes, artifact hash and fingerprint, request count, query identifiers,
source disjointness, lineage identity, candidate hashes, public baseline hashes, shadow status, and
timing receipts all match the frozen contract. Control and treatment selections must differ on at
least `2` questions to prove treatment liveness.

Fact coverability must be validated before the live run with the production chunker. The run may
not refit the source model or use fact terms for selection.

## Decision rule

Return `ADVANCE` when the target population contains at least `4` queries, at least `1` target fact
is rescued, no complete query or fact regresses, unanswerable answers do not increase, and context
precision remains inside the `0.05` guardrail.

Return `CLOSE` when the target population contains at least `4` queries and the treatment rescues
none, causes any complete query or fact regression, increases unanswerable answers, or breaches the
precision guardrail. Return `INSUFFICIENT` when the apparatus is valid but the target population has
fewer than `4` queries. Return `REPAIR` for an apparatus failure.

`ADVANCE` would provide a second independent directional validation and license human label review
plus a current generation confirmation. It would not change production serving by itself.

## Reproduction planned

```powershell
python -m scripts.validate_memory_essential_facts `
  --labels docs/preregistrations/2026-09-13-memory-buried-fact-challenge-facts.json `
  --recall-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory `
  --sentiment-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory

$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-buried-<commit>'
$env:RECALL_SOURCE_COMMIT='<implementation-commit>'
python -u scripts/run_live_source_conditioning_holdout.py `
  --challenge buried `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-conditioning-buried-fact.json
```

[[frozen_above]]

Results may be appended below this marker. Nothing above it may be changed after commit.

## Registered result

Measured at `2026-09-13T20:37:42.593666+00:00` from source commit
`c1506ef89b262355c894403666f61d284677bf17`.

The registered decision is `INSUFFICIENT`. All `36` requests passed candidate hash parity, public
baseline hash parity, shadow status, timing receipt, and immutable lineage checks. The treatment
changed the selected identifiers on `18` queries, so it was live.

The control and treatment each completed `16` of `18` answerable queries and covered `16` of `18`
essential facts. Each found the gold source on the same `16` queries. The other two queries missed
the gold source in both arms. Consequently, no query satisfied the registered target definition of
control source hit plus control fact miss, versus a required minimum of `4`.

Control labeled context precision was `0.7656`. Treatment precision was `0.8228`. Each arm answered
one of `18` unanswerable controls. There were no complete or fact gains and no complete or fact
losses.

The raw artifact is
`docs/results/2026-09-13-live-source-conditioning-buried-fact.json`, SHA256
`842e9ceb19e3d5c1635e8499b183007571c5fd3078c7800a7491aeebc81c4e43`. The full interpretation and
exact reproduction command are in
`docs/results/2026-09-13-live-source-conditioning-buried-fact.md`.
