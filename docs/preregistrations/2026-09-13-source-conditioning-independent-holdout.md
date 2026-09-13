# Pre-registration: independent source conditioning holdout

Date registered: 2026-09-13.

## Question

Does the already frozen global source conditioning model improve essential fact retrieval on new
queries and new gold sources, especially when ordinary serving finds the right source but omits the
decisive passage?

This is an independent validation of the existing model. It is not a new fit, threshold sweep, or
source expansion experiment.

## Frozen substrate

The run uses memory generation `gen_18d5edd2e5e847c0af1ee37e40d27893`, calibration
`cal_23d8708ac550444fa4274ac617df0870`, pipeline fingerprint
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus fingerprint
`c054e26b28e62760c0fefa64815da8033433e94c2474f079ed8a2aec1b93fb1a`.

The treatment is the immutable artifact
`docs/results/2026-09-13-source-conditioning-model.json`, SHA256
`fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f`. Its fixed settings are
model `source-logistic-v1`, alpha `0.08`, raw cosine floor `0.30`, candidate pool `20`, and item
budget `5`. No coefficient, alpha, floor, source feature, candidate depth, or budget may change
after this registration.

Both arms come from one MCP request and therefore one query vector. The control is the public
trusted evidence. The treatment applies the frozen source model to the same captured dense,
lexical, and fused candidate traces. Graph expansion is off.

## New holdout

The frozen query set is
`docs/preregistrations/2026-09-13-memory-source-conditioning-holdout-queries.json`, SHA256
`cf91bb848a518991a49863c79ae3014a4aa9c4ba3f717877c2019266173b5940`.
It contains `36` questions, split equally between answerable questions and unanswerable controls.

The frozen fact labels are
`docs/preregistrations/2026-09-13-memory-source-conditioning-holdout-facts.json`, SHA256
`4d5c359a53f87ab5264920e3ba2aa9ef19ad4e1cd6a0b1bb5864d4c3af4b1fee`.
They contain one essential fact for each of the `18` answerable questions. Every fact is coverable
inside one canonical chunk under the production chunker. The `18` gold sources have no overlap
with the `22` answerable sources used to fit the source model.

I selected multi-chunk operational memos from the real RE-call memory corpus, then wrote a natural
question and one decisive fact from each source. I validated source existence and fact
coverability before retrieval. I did not inspect control or treatment retrieval for any holdout
question before freezing these files. The unanswerable controls ask for plausible internal facts
that are not asserted by the selected corpus. The labels are author constructed and mechanically
verifiable, but they are not an independent human adjudication. This limits any promotion claim.

## Primary population and metrics

The primary population is selected by the control arm alone. It contains answerable queries where
the public control includes at least one chunk from the gold source but does not cover the essential
fact. This is the correct source but missed passage failure class.

Primary metrics are target population fact rescues, target population fact regressions, overall
complete queries, and overall covered facts. Guardrails are unanswerable answered count, labeled
context precision, source hits, exact candidate hash parity, public baseline hash parity, lineage
identity, and treatment liveness.

## Predictions

1. The control selected target population will contain at least `4` of the `18` answerable queries.
2. Fixed alpha `0.08` will rescue at least `1` target fact and lose `0` target facts.
3. Across all answerable questions, treatment complete queries and covered facts will not be lower
   than control.
4. Treatment will not increase the number of answered unanswerable controls.
5. Treatment labeled context precision will be no more than `0.05` below control.

These are deliberately smaller predictions than the available ceiling. The previous same vector
result added one complete query on twenty two answerable questions, so a larger claim would not be
supported by the observed effect size.

## Apparatus validity

The run is void and returns `REPAIR` unless all of the following hold:

* Query and fact file hashes exactly match this record.
* Model artifact hash and fingerprint exactly match this record.
* The server returns the registered generation, calibration, pipeline, and corpus identity on every
  request.
* All `36` requests complete without a shadow error.
* Recomputed candidate hashes and public baseline hashes match all `36` requests.
* Control and treatment selected identifiers differ on at least `2` questions. This proves that the
  treatment is live. It is not a quality gate.
* Fact validation reports zero uncovered facts, query identifiers are unique, and the new gold
  source set has zero overlap with the fitting gold source set.

## Decision rule

Return `ADVANCE` only when the target population contains at least `4` questions, at least `1`
target fact is rescued, no target fact is lost, overall complete queries and facts do not decline,
unanswerable answers do not increase, and context precision stays within the registered `0.05`
guardrail.

Return `CLOSE` when the target population contains at least `4` questions and the treatment rescues
none, loses a target fact, reduces overall complete queries or facts, increases unanswerable
answers, or breaches the precision guardrail.

Return `INSUFFICIENT` when the apparatus is valid but the target population contains fewer than
`4` questions. Return `REPAIR` for any apparatus validity failure.

`ADVANCE` licenses consideration of active selection only after an independent human review of the
labels and a current production generation confirmation. It does not by itself change the serving
route.

## Reproduction planned

```powershell
python -m scripts.validate_memory_essential_facts `
  --labels docs/preregistrations/2026-09-13-memory-source-conditioning-holdout-facts.json `
  --recall-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory `
  --sentiment-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory

$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-holdout-<commit>'
$env:RECALL_SOURCE_COMMIT='<implementation-commit>'
python -u scripts/run_live_source_conditioning_holdout.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-conditioning-holdout.json
```

[[frozen_above]]

Results may be appended below this marker. Nothing above it may be changed after commit.

## Method correction before measurement

The phrase `target fact regressions` above is non-informative because the target population is
defined by a control fact miss. A treatment cannot lose a fact that the control did not cover. I
will therefore report target source losses as a diagnostic, but I will not score the registered
zero target fact loss statement as a prediction. The binding downside tests remain the overall
complete query and fact non-regression rules, the unanswerable control rule, and the context
precision guardrail. This correction was appended before any holdout retrieval was run.
