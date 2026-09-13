# Buried fact source conditioning challenge result

Measured: `2026-09-13T20:37:42.593666+00:00`.

Registered decision: `INSUFFICIENT`.

Source commit: `c1506ef89b262355c894403666f61d284677bf17`.

Remote checkout: `/home/sentiment/recall-repos/source-conditioning-buried-c1506ef8`.

Raw artifact: `docs/results/2026-09-13-live-source-conditioning-buried-fact.json`.

Raw artifact SHA256: `842e9ceb19e3d5c1635e8499b183007571c5fd3078c7800a7491aeebc81c4e43`.

## Outcome

The frozen source conditioning model changed selections but did not change retrieval quality on
this challenge. Both arms completed `16` of `18` answerable queries, covered `16` of `18` essential
facts, found `16` of `18` gold sources, and answered one of `18` unanswerable controls.

| Arm | Complete | Facts | Source hits | Precision | False answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public same vector control | 16/18 | 16/18 | 16/18 | 0.7656 | 1/18 |
| Fixed source alpha 0.08 | 16/18 | 16/18 | 16/18 | 0.8228 | 1/18 |

All `36` requests passed candidate hash parity, public baseline hash parity, shadow status, timing
receipt, and immutable lineage checks. The treatment changed selected identifiers on `18` queries,
including `17` answerable queries, so the null quality result is not an activation failure.

## Why the target population was empty

The preregistered target required the public control to find the correct source but omit the
essential fact. That occurred on zero queries. Whenever the control found the gold source, it also
included the late essential fact. The two incomplete queries, `buried-015` and `buried-017`, missed
the gold source in both arms.

`buried-015` returned only
`agent-memory-bench/stress-paper-draft-is-unpublished.md` in the control and treatment. Its gold
source was absent. `buried-017` returned no admitted evidence in either arm. This treatment cannot
repair either failure because source conditioning can amplify evidence inside a source only after
base retrieval has discovered that source.

The registered minimum was four target queries. The valid run therefore returns `INSUFFICIENT`, not
`CLOSE`. The challenge did not test the intended mechanism despite selecting long sources and late
facts without inspecting retrieval output.

## Precision effect

The treatment selected `65` labeled source items among `79` answerable context items, versus `49`
among `64` for the control. The `0.0572` precision increase came from stronger same source
concentration, not additional fact coverage. Eight answerable queries gained one to three labeled
source items, while `buried-006` lost one labeled source item without losing its essential fact.

The unanswerable count stayed at one in each arm, but the treatment expanded that one false bundle
from two items across two incident sources to five items from one incident source. The registered
guardrail counted answered controls rather than false bundle volume, so this is a diagnostic caution
and not a registered regression.

## Interpretation

Across the refreshed Context 4 evaluations, correct source but missed decisive passage cases are
rare. The original 50 query result exposed one such case and source conditioning rescued it. The
first independent holdout exposed one and source conditioning rescued it. This deliberately harder
challenge exposed none. Continuing to manufacture longer source holdouts has low expected value.

The useful boundary is now clearer. Fixed source conditioning is a viable precision and completion
mechanism after source discovery, but it is not a source discovery mechanism. The next retrieval
experiment should diagnose the raw dense and lexical ranks for the two source misses, then choose
between trust admission repair and a source level first stage. It should not tune alpha, expand more
chunks, or reopen graph traversal.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-buried-c1506ef8'
$env:RECALL_SOURCE_COMMIT='c1506ef89b262355c894403666f61d284677bf17'
python -u scripts/run_live_source_conditioning_holdout.py `
  --challenge buried `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-conditioning-buried-fact.json
Get-FileHash docs/results/2026-09-13-live-source-conditioning-buried-fact.json -Algorithm SHA256
```

The compact result and diagnostic slices are reproduced with:

```powershell
$p=Get-Content -Raw docs/results/2026-09-13-live-source-conditioning-buried-fact.json |
  ConvertFrom-Json
$p.decision
$p.integrity | Format-List
$p.summary | ConvertTo-Json -Depth 8
$p.target_analysis | Format-List
$p.rows |
  Where-Object {
    $_.query.answerable -and
    (-not $_.scores.baseline.complete -or -not $_.scores.candidate.complete)
  } |
  Select-Object @{n='id';e={$_.query.id}}, scores, baseline_items, candidate_items |
  ConvertTo-Json -Depth 10
```
