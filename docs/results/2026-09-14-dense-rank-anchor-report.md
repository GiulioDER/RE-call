# Dense rank with anchor safety failed quality

Measured 2026-09-14. Verdict: `STOP_DENSE_RANK_RESCUE`.

The registered screen revisited only the consumed 30-row empty-base cohort. It did not use a new
holdout and cannot authorize serving.

Among 15 answerable rows, the exact span appeared within dense rank 1 for three rows, rank 5 for
five, rank 10 for eight, and rank 20 for 10. Gold source reachability was six at rank 1, seven at
rank 5, eight at rank 10, and 10 at rank 20.

`dense_first_anchor_safe` selected 14 answerable chunks and zero controls. It produced three exact
span gains, six gold-source selections, and 21.4% exact addition precision. The more restrictive
`dense_first_anchor_compatible` selected 13 answerable chunks and zero controls, but produced only
one exact-span gain, two gold-source selections, and 7.7% exact addition precision.

Neither rule met the preregistered 50.0% precision floor. The anchor compatibility condition harmed
ranking, while raw dense rank one remained too imprecise. This closes simple dense-rank rescue on
the consumed cohort. The next highest ROI experiment is a new query representation that changes
dense ordering, evaluated first on these consumed rows and then on a new source-disjoint paired
holdout only if it clears the same quality and control gates.

The private row-level artifact remains outside the repository at
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-14\dense-rank-anchor-development.json`. The
aggregate-safe result is `docs/results/2026-09-14-dense-rank-anchor-development.json`.
Its SHA256 is `54a74dee3f09e290912a678ab98bb50e477da33f8ef879a5b77c2bf87b443361`.

Reproduce from the repository root in PowerShell:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='39c0af0c'
$env:RECALL_POLICY_COMMIT='bc24504d'
python -u scripts/run_dense_rank_anchor_dev.py --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --holdout-result docs/results/2026-09-14-query-anchor-holdout.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output C:\Users\gde00\.codex\evals\query-anchor-2026-09-14\dense-rank-anchor-development.json --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Behavioral red proof on 2026-09-14 deliberately returned dense rank one for the compatible rule
even when it lacked two query anchors. Reproduce the intended failure at that mutation with:

```powershell
python -m pytest tests/test_dense_rank_anchor_dev.py::test_compatible_rule_skips_incompatible_dense_rank_one -q
```

Restoring the compatibility scan gave two passing tests and a clean Ruff check:

```powershell
python -m pytest tests/test_dense_rank_anchor_dev.py -q
python -m ruff check scripts/run_dense_rank_anchor_dev.py tests/test_dense_rank_anchor_dev.py
git diff --check
```
