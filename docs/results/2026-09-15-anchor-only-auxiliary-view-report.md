# Anchor only auxiliary retrieval failed exact reach

Measured 2026-09-15. Verdict: `STOP_LEXICAL_ANCHOR_QUERY_CONSTRUCTION`.

The registered development experiment revisited only the 30 consumed empty-base rows from the
completed query anchor holdout. It did not select an item, change a served result, or use a fresh
holdout.

## Result

The apparatus reproduced every frozen baseline count. Among the 15 answerable rows, original dense
retrieval contained the exact span for 3 rows at rank 1, 5 at rank 5, 8 at rank 10, and 10 at rank
20. Gold-source reachability was 6, 6, 7, 8, and 10 at ranks 1, 3, 5, 10, and 20. The original
anchor safety gate admitted 14 answerable rows and zero controls.

The auxiliary query contained only the three rare, corpus-supported anchors in original question
order. At ranks 1, 3, 5, 10, and 20, it reached exact evidence for 2, 2, 2, 2, and 4 answerable rows.
It reached the gold source for 3, 3, 3, 4, and 7 rows.

The deduplicated union of original top 20 and auxiliary top 20 added 176 unique chunks. Gold-source
reachability rose from 10 rows to 11, but exact-span reachability remained 10. The auxiliary view
therefore made one gold source newly reachable and zero exact spans newly reachable. The registered
gate required at least two incremental exact rows and zero eligible controls. Safety passed, but
quality failed.

The prediction expected two of the five original exact misses to become reachable. The observed
result was zero. The prediction did not hold.

## Interpretation

The anchor only view substantially changed candidate membership without adding answer-bearing
evidence. This separates movement from quality: 176 novel candidates are not useful recall when
none contains a missing exact span. The result closes lexical anchor query construction on this
cohort. Do not tune anchor count, order, wording, cutoff, or the two-row gate.

The next highest ROI experiment is not another candidate generator. Original dense top 20 already
contains exact evidence for 10 of 15 answerable rows, so the immediate problem is choosing the
answer-bearing chunk precisely. RE-call already contains a free local ColBERT MaxSim late
interaction scorer. The next development screen should apply its default permissively licensed
model to the original dense top 20, compare exact and gold-source ranks with original dense order,
and remain behind the corpus-absent anchor safety gate. This differs from the failed generic cross
encoder because it scores token level query to passage interactions rather than one pooled pair
classification score.

The private row-level artifact remains outside the repository at
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\anchor-only-auxiliary-view-development.json`.
Its SHA256 is `75b9e861efaf6382839e3055df20d1c74b00856ec33bdc1954d57a33c87bbac5`.
The aggregate-safe result is
`docs/results/2026-09-15-anchor-only-auxiliary-view-development.json`.

## Reproduction

From the repository root in PowerShell:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='b4f9a494'
$env:RECALL_POLICY_COMMIT='201a32b2'
python -u scripts/run_anchor_only_aux_view_dev.py --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --holdout-result docs/results/2026-09-14-query-anchor-holdout.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\anchor-only-auxiliary-view-development.json --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Behavioral red proof on 2026-09-15 deliberately prefixed the original question to the auxiliary
anchors. The exact representation test failed at the intended assertion. Restoring the anchor only
query gave 19 passing focused tests, a clean Ruff check, and a clean targeted mypy check:

```powershell
python -m pytest tests/test_anchor_only_aux_view_dev.py tests/test_anchor_boosted_query_dev.py tests/test_mcp_architecture_seams.py -q
python -m ruff check scripts/run_anchor_only_aux_view_dev.py tests/test_anchor_only_aux_view_dev.py
python -m mypy --follow-imports=skip scripts/run_anchor_only_aux_view_dev.py
git diff --check
```
