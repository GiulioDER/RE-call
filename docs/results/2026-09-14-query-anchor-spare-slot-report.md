# Query anchor spare-slot holdout failed without activating

Measured 2026-09-14. Verdict: `FAIL_NO_EXACT_SPAN_GAIN`. The frozen
`query_anchor_empty_base_v1` policy is not eligible for promotion or active serving.

## Result

The sealed holdout completed all 160 queries, comprising 80 answerable questions and 80 matched
absent-anchor controls. Base exact-span coverage was 49 of 80 and candidate exact-span coverage
was also 49 of 80. Base and candidate source hit counts were both 58 of 80.

The candidate produced zero exact-span gains, zero exact-span losses, zero source-hit gains, zero
source-hit losses, zero additions, and zero control activations. It preserved every base prefix and
never exceeded the one-item addition budget. Addition precision is undefined because nothing was
added.

The point prediction was two exact-span gains and five additions. The observed result was zero for
both. The policy therefore failed the preregistered requirement of at least one exact-span gain.
The safety conditions passed only because the rule never activated.

## Interpretation

Direct query-to-chunk anchor support separated development controls, but the frozen combination of
an empty base, three selected anchors, no zero-frequency anchor, and two-of-three chunk coverage did
not transfer to the untouched source cohort. This is evidence against deploying the combined rule,
not evidence that lexical query anchors have no useful signal. The next diagnostic must determine
whether empty-base rows lacked an original guarded proposal or whether proposals failed one of the
anchor predicates.

The completed rows contained 30 empty-base cases, evenly split between 15 answerable questions and
15 controls. A diagnostic may revisit only these consumed rows. It must report aggregate proposal
availability, anchor-count, zero-frequency-anchor, and chunk-coverage blocker counts. It may retain
numeric features and gold booleans only in a private artifact outside the repository. It must not
change this holdout verdict or fit another policy to the same sealed result.

## Frozen identity

The run used source commit `d6e7b6df`, policy commit `f72e8951`, generation
`gen_2ccf2130f6c64d99a11a6bcb6f929dd8`, calibration
`cal_e50dac493112488ea5e7cf79d86c0099`, pipeline
`57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`, and corpus
`f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312`.

The query pool SHA256 was
`6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`. The inventory receipt
SHA256 was `4c813a1b233d5acaeb3272684eaf76e8e01db6543ff0352e9625036628a80d9d`. The source
conditioning artifact SHA256 was
`fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f`.

The aggregate-safe result is `docs/results/2026-09-14-query-anchor-holdout.json`, SHA256
`58b883af8d1527ef137913762f9a3a456198c8f818d8286bfc87a248c4d3fa5b`. Its measured elapsed time
was 679647.167 milliseconds.

## Reproduction

From the repository root in PowerShell:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='d6e7b6df'
$env:RECALL_POLICY_COMMIT='f72e8951'
python -u scripts/run_live_query_anchor_holdout.py --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --artifact docs/results/2026-09-13-source-conditioning-model.json --inventory-receipt docs/results/2026-09-14-query-anchor-inventory.json --output docs/results/2026-09-14-query-anchor-holdout.json --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Focused verification on 2026-09-14:

```powershell
python -m pytest tests/test_query_anchor_admission.py tests/test_summarize_query_anchor_dev.py tests/test_query_anchor_holdout.py -q
python -m ruff check recall/query_anchor_admission.py scripts/summarize_query_anchor_dev.py scripts/validate_query_anchor_inventory.py scripts/run_live_query_anchor_holdout.py tests/test_query_anchor_admission.py tests/test_summarize_query_anchor_dev.py tests/test_query_anchor_holdout.py
git diff --check
```

The first command passed 11 tests. Ruff and the diff check passed.

## Empty-base diagnostic

Completed 2026-09-14 on only the 30 consumed empty-base rows registered above. The original guarded
source conditioner produced three proposals and no proposal on the other 27 rows. All three
proposals belonged to controls. It produced no proposal for any of the 15 answerable empty-base
rows. All three proposals selected three anchors, but every proposal had at least one anchor with
zero source document frequency. Two passed the chunk coverage condition, and none passed the full
frozen policy. None was a gold source or contained an exact span.

This localizes the failed transfer. The query-anchor rule correctly rejected the only unsafe
proposals it received. The existing guarded proposal generator supplied no candidate on the cases
where an answerable empty result could be improved. Reusing that proposal stream is therefore the
wrong dependency for the next experiment.

The private row-level artifact remains outside the repository at
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-14\holdout-empty-base-diagnostic.json`. The
aggregate-safe summary is
`docs/results/2026-09-14-query-anchor-empty-base-diagnostic.json`, SHA256
`6233a09175eb1e6658f07a2f521755157122332bed6c0006a26d51a63806403f`.

Reproduce the diagnostic from the repository root in PowerShell:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='00c6e73e'
$env:RECALL_POLICY_COMMIT='f72e8951'
python -u scripts/diagnose_query_anchor_holdout.py --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --holdout-result docs/results/2026-09-14-query-anchor-holdout.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output C:\Users\gde00\.codex\evals\query-anchor-2026-09-14\holdout-empty-base-diagnostic.json --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Focused diagnostic verification on 2026-09-14:

```powershell
python -m pytest tests/test_diagnose_query_anchor_holdout.py -q
python -m ruff check scripts/diagnose_query_anchor_holdout.py tests/test_diagnose_query_anchor_holdout.py
```

The first command passed one test and Ruff passed.
