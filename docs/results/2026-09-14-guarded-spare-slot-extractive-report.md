# Guarded spare slot exact span result

Date: 2026-09-14.

## Verdict

`extractive_strict_v1` failed the retrieval gate and is not eligible for promotion.

The rule was intentionally conservative. It inspected only the first guarded proposal and required
dual retrieval support, sparse source rank at most 2, nonnegative source margin, and cross-leg
fraction at least 0.5. It looked perfect on the private development subset, where it admitted 4
gold additions and no known non-gold additions. That separation did not generalize.

## Integrity

The live inventory matched all 250 frozen answerable sources by raw SHA256 before retrieval. There
were 0 missing sources, 0 digest mismatches, and no inventory truncation. The run stayed pinned to:

* generation `gen_5a945edfbc644e5db77906c06658dc49`;
* calibration `cal_e4c81ba2db404eafbeb59f297f3c1dd2`;
* pipeline `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`;
* corpus `d5570385d78065192b724d6f29ad18e3c996ff2ab82a679cac1913e558511a07`;
* pool SHA256 `66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`.

## Result

| Metric | Base | Candidate |
| --- | ---: | ---: |
| Exact span coverage | 149 of 250 | 149 of 250 |
| Gold source hit | 171 of 250 | 171 of 250 |

The candidate produced:

* 0 exact span gains and 0 exact span losses;
* 0 source hit gains and 0 source hit losses;
* 2 additions across 500 queries;
* 2 control activations across 250 controls;
* 0 additions containing the frozen answer span;
* 0.0 added item exact span precision;
* complete base prefix preservation;
* at most 1 addition per query.

The first decisive failure is the registered zero tolerance control gate. The absence of any gain
also means there is no quality upside to trade against that failure.

## Interpretation

Dense and sparse agreement, source margin, and cross-leg concentration describe how consistently a
source was retrieved. They do not establish that the source answers the actual query. A matched
absent identifier can preserve enough topical context to create strong source agreement even though
the requested entity is not present. More threshold tuning on the same four features is therefore
low value and would contaminate the consumed holdout.

## Next experiment by expected ROI

The next candidate should use the broader first guarded proposal as generation, then add a
deterministic query to evidence compatibility gate before admission. The highest value first signal
is anchor coverage: every opaque identifier in the query must occur in the proposed evidence, and a
minimum fraction of rare query terms must occur in the proposed source or chunk. This directly
targets the observed failure because both bad additions ignored an absent query anchor.

The present 500 query cohort may now be used only for development. A new untouched pool should be
built from the remaining eligible sources, excluding every source in both prior pools. The next
gate should retain exact span gain as the primary metric, keep zero control activation as mandatory,
and add realistic counterfactual controls so success cannot depend only on recognizing the synthetic
identifier prefix.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/guarded-spare-slot-89caaf43'
$env:RECALL_SOURCE_COMMIT='98b6ee9d'
$env:RECALL_POLICY_COMMIT='2d36eabd'
python -u scripts/run_live_guarded_spare_slot_extractive_holdout.py --query-pool docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json --artifact docs/results/2026-09-13-source-conditioning-model.json --inventory-receipt docs/results/2026-09-14-guarded-spare-slot-extractive-inventory.json --output docs/results/2026-09-14-guarded-spare-slot-extractive-holdout.json --generation-id gen_5a945edfbc644e5db77906c06658dc49 --calibration-id cal_e4c81ba2db404eafbeb59f297f3c1dd2 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint d5570385d78065192b724d6f29ad18e3c996ff2ab82a679cac1913e558511a07
```

The machine readable result is
`docs/results/2026-09-14-guarded-spare-slot-extractive-holdout.json`.
