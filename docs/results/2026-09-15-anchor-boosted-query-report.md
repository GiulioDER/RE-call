# Anchor boosted query representation failed quality

Measured 2026-09-15. Verdict: `STOP_SINGLE_VIEW_ANCHOR_BOOST`.

The registered development experiment revisited only the 30 consumed empty-base rows from the
completed query anchor holdout. It did not use a new holdout and cannot authorize serving.

## Result

The apparatus reproduced every frozen baseline count. Among the 15 answerable rows, original dense
retrieval contained the exact span for 3 rows at rank 1, 5 at rank 5, 8 at rank 10, and 10 at rank
20. The original query anchor safety gate admitted 14 answerable rows and zero controls.

The frozen transformation preserved the complete original question and appended one duplicate of
each of its three rare, corpus-supported anchors. It changed rank 1 for only one eligible row. That
new rank 1 result contained neither the exact span nor the gold source, which gave zero exact gains
and 0.0% exact addition precision. The registered gate required at least two exact gains, zero
controls, at least 50.0% exact addition precision, and no decrease in exact top 5 coverage.

Dense exact-span coverage was 3, 4, 4, 7, and 9 at transformed cutoffs 1, 3, 5, 10, and 20,
compared with 3, 3, 5, 8, and 10 for the original query. Gold-source coverage was 6, 7, 7, 8, and
9, compared with 6, 6, 7, 8, and 10. Exact rank improved on four answerable rows, tied on nine,
and worsened on two. The top 5 non-regression condition failed because exact coverage decreased
from five rows to four.

The prediction expected rank 1 to change for 6 to 10 of the 14 eligible answerable rows and exact
rank 1 coverage to rise from three rows to four through six. The observed result was one changed
rank 1 and no improvement in exact rank 1 coverage. The prediction did not hold.

## Interpretation

Repeating rare query terms inside the primary query is too weak to alter rank 1 reliably and can
displace relevant evidence deeper in the list. This closes single view anchor boosting on this
cohort. Repeating more terms, changing the repetition count, changing the anchor count, or tuning
thresholds on these rows would be unregistered reuse of the same development evidence.

The next experiment should keep the original dense query intact and score a separate anchor view.
That view should be evaluated first for incremental exact-span and gold-source reach in the union
of the two candidate lists. Only if it adds evidence should a separately preregistered selector be
allowed to spend one spare result slot. This design preserves the original ordering, tests whether
anchor vocabulary contributes complementary recall, and avoids forcing one embedding to represent
both the natural question and a reweighted lexical signal.

The private row-level artifact remains outside the repository at
`C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\anchor-boosted-query-development.json`. Its
SHA256 is `6cf3e349974785021721521469fbfacbae1612d02855979277fad7374fd5e3bb`.
The aggregate-safe result is
`docs/results/2026-09-15-anchor-boosted-query-development.json`.

## Reproduction

From the repository root in PowerShell:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/query-anchor-f72e8951'
$env:RECALL_SOURCE_COMMIT='8a43062c'
$env:RECALL_POLICY_COMMIT='38a066a3'
python -u scripts/run_anchor_boosted_query_dev.py --query-pool docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json --holdout-result docs/results/2026-09-14-query-anchor-holdout.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output C:\Users\gde00\.codex\evals\query-anchor-2026-09-15\anchor-boosted-query-development.json --generation-id gen_2ccf2130f6c64d99a11a6bcb6f929dd8 --calibration-id cal_e50dac493112488ea5e7cf79d86c0099 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint f737fd1ffcdb9275ceccc4092a5dd7cc873f374936d2d9d60e06618c3bec6312
```

Behavioral red proof on 2026-09-15 deliberately returned anchors in hash-ranked order instead of
their order in the original question. The focused test failed at the intended ordering assertion.
Restoring query order gave five passing focused tests and a clean Ruff check:

```powershell
python -m pytest tests/test_anchor_boosted_query_dev.py -q
python -m ruff check scripts/run_anchor_boosted_query_dev.py tests/test_anchor_boosted_query_dev.py
python -m pytest tests/test_mcp_architecture_seams.py -q
git diff --check
```
