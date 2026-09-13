# Pre-registration: source conditioning alpha 0.15 development screen

Date registered: 2026-09-13.

## Status and question

This is a post hoc development screen, not independent validation. Both input holdouts have already
been measured with fixed alpha `0.08`, and their outcomes have been inspected. No result from this
screen can license active selection.

The question is whether the already observed global alpha `0.15` setting improves essential fact
coverage over alpha `0.08` across both holdouts without worsening their negative controls. A
positive result would license only fresh human reviewed validation of alpha `0.15`.

## Frozen substrate

The run uses memory generation `gen_18d5edd2e5e847c0af1ee37e40d27893`, calibration
`cal_23d8708ac550444fa4274ac617df0870`, pipeline fingerprint
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus fingerprint
`c054e26b28e62760c0fefa64815da8033433e94c2474f079ed8a2aec1b93fb1a`.

The source model is
`docs/results/2026-09-13-source-conditioning-model.json`, SHA256
`fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f`, fingerprint
`9c3e4a2d1a56d4c3bf4d77800dbdf9b02261549199af8ddbeb4ac11aafdc66c7`. All learned
coefficients, standardization values, raw cosine floor `0.30`, RRF constant `60`, candidate depth
`20`, and item budget `5` remain fixed. The comparator changes only alpha from `0.08` to `0.15`.

## Frozen inputs

The independent holdout uses:

* query SHA256 `cf91bb848a518991a49863c79ae3014a4aa9c4ba3f717877c2019266173b5940`;
* fact SHA256 `4d5c359a53f87ab5264920e3ba2aa9ef19ad4e1cd6a0b1bb5864d4c3af4b1fee`.

The buried fact holdout uses:

* query SHA256 `9eefe96c8bb07cb71b2bde71f359c222f6966e488c85097249e5536b6b074794`;
* fact SHA256 `9e824f03caef370490f9be53b16451bc8fa29ee8d616aff7ff05351f6d0b86c5`.

Together they contain `72` queries, `36` answerable questions, `36` unanswerable controls, and `36`
essential facts. Their source sets are mutually disjoint and also disjoint from the model fitting
gold sources.

## Same vector arms

Each query makes one live request. The request captures public evidence, dense and sparse candidate
traces, the complete trust pool, and the production alpha `0.08` shadow selection from one query
vector.

Three selections are scored:

1. public trusted evidence;
2. the immutable artifact at alpha `0.08`;
3. the same artifact values with alpha `0.15` substituted in memory for this screen only.

The alpha `0.08` arm must exactly match the production shadow hashes. The alpha `0.15` arm is never
returned publicly and is not serialized as a serving artifact.

## Predictions

1. Alpha `0.15` will cover at least one more essential fact than alpha `0.08` across the pooled
   answerable questions.
2. Alpha `0.15` will lose no complete query and no essential fact covered by alpha `0.08`.
3. Alpha `0.15` will not increase answered unanswerable controls in either holdout or in the pooled
   total.
4. Pooled labeled context precision at alpha `0.15` will be no more than `0.05` below alpha `0.08`.

The specific mechanistic prediction is that `buried-017` becomes complete because its reconstructed
source support `0.769714` moves its best chunk from adjusted score `0.397500` at alpha `0.08` to
approximately `0.416380` at alpha `0.15`, above threshold `0.4100`.

## Apparatus validity and decision rule

Return `REPAIR` unless all `72` requests match immutable lineage, all baseline and alpha `0.08`
candidate hashes match the same request shadow receipts, all shadow statuses and timing receipts
are present, and alpha `0.15` changes at least two selections relative to alpha `0.08`.

Return `PROMISING_FOR_FRESH_VALIDATION` if all four predictions pass. Return
`CLOSE_GLOBAL_ALPHA_INCREASE` if the apparatus is valid and any prediction fails.

Even `PROMISING_FOR_FRESH_VALIDATION` licenses only a new blinded or human reviewed comparison. It
does not license deployment, an active route change, or model artifact replacement.

## Reproduction planned

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-alpha015-<commit>'
$env:RECALL_SOURCE_COMMIT='<implementation-commit>'
python -u scripts/run_live_source_conditioning_alpha_screen.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-conditioning-alpha015-screen.json
```

[[frozen_above]]

Results may be appended below this marker. Nothing above it may be changed after commit.
