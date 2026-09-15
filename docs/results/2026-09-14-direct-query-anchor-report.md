# Direct query anchor candidate ranking failed quality

Measured 2026-09-14. Verdict: `STOP_DIRECT_POOL_RANKING`.

The registered screen revisited only the 30 consumed empty-base rows from the completed query
anchor holdout. It did not use a new holdout and cannot authorize serving.

Gold source and exact-span evidence were both present in the audited candidate pool for 10 of 15
answerable rows. Dense retrieval contained the gold source and exact span for all 10. Sparse
retrieval contained seven gold sources and four exact spans.

The frozen direct ranker selected 14 answerable chunks and zero controls. Only two selected chunks
contained the exact span, three came from the gold source, and three selected sources contained the
exact span elsewhere in their audited chunks. Exact addition precision was 2 of 14, or 14.3%.
This missed the registered 50.0% precision floor despite meeting the minimum of two exact gains.

The result closes simple anchor-first ranking on this consumed cohort. It also shows that candidate
coverage is not the dominant failure for two thirds of the answerable empty-base cases: useful
evidence is already present in the dense candidate leg but is not selected precisely enough.
Another admission threshold over the inherited guarded proposal is low ROI. The next experiment
must improve ranking within the dense candidate set or improve the query representation used to
rank that set, while retaining the corpus-absent anchor rejection that kept controls at zero.

The private row-level artifact remains outside the repository at
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-14\direct-query-anchor-development.json`. The
aggregate-safe result is `docs/results/2026-09-14-direct-query-anchor-development.json`.
Its SHA256 is `4c3aea610e9fd0a4df29b80a771d6befa79f7c0990adacc8f3d6c410b06ff91e`.

Reproduce from the repository root in PowerShell:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='222979f9'
$env:RECALL_POLICY_COMMIT='e06f17f0'
python -u scripts/run_direct_query_anchor_dev.py --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --holdout-result docs/results/2026-09-14-query-anchor-holdout.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output C:\Users\gde00\.codex\evals\query-anchor-2026-09-14\direct-query-anchor-development.json --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Behavioral red proof on 2026-09-14 deliberately sorted anchor coverage ascending. Reproduce the
intended failure at that mutation with:

```powershell
python -m pytest tests/test_query_anchor_admission.py::test_direct_anchor_candidate_prefers_greater_chunk_coverage -q
```

Restoring descending coverage gave seven passing focused tests and a clean Ruff check:

```powershell
python -m pytest tests/test_query_anchor_admission.py tests/test_direct_query_anchor_dev.py -q
python -m ruff check recall/query_anchor_admission.py scripts/run_direct_query_anchor_dev.py tests/test_query_anchor_admission.py tests/test_direct_query_anchor_dev.py
git diff --check
```
