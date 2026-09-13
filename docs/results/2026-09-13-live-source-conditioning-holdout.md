# Independent source conditioning holdout result

Measured: 2026-09-13T20:21:23.428980+00:00.

Registered decision: `INSUFFICIENT`.

Source commit: `103db4c298cba42f6bc3567b74be1a8d4d77f6f5`.

Remote checkout: `/home/sentiment/recall-repos/source-conditioning-holdout-103db4c2`.

Raw artifact: `docs/results/2026-09-13-live-source-conditioning-holdout.json`.

Raw artifact SHA256: `c0bc85b88880f4b8e57f445608a8f03b92f219ff1b665fc22f866cb2eea45267`.

## Outcome

The fixed source conditioning model was directionally positive on new queries and new sources, but
the holdout did not contain enough of the registered failure class to decide active selection.
Ordinary serving found the decisive fact for `17` of `18` answerable questions. Only
`holdout-007` had the correct source in the public bundle while omitting its essential fact. The
treatment repaired that case. The registered target population minimum was `4`, so the evaluator
correctly returned `INSUFFICIENT` rather than promoting a one case result.

| Arm | Complete | Facts | Source hits | Precision | False answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public same vector control | 17/18 | 17/18 | 18/18 | 0.6000 | 0/18 |
| Fixed source alpha 0.08 | 18/18 | 18/18 | 18/18 | 0.7500 | 0/18 |

All `36` requests passed candidate hash parity, public baseline hash parity, shadow status, timing
receipt, and immutable lineage checks. The treatment changed selected identifiers on `17` queries.
This is strong apparatus evidence that the treatment was live even though most labels were already
saturated by the control.

## The rescue

For `holdout-007`, the question asked whether one empty SSH process probe was enough to declare a
remote job dead. The public control returned one chunk from the correct source but not the passage
requiring consecutive negative observations from successful probes. Source conditioning selected
three chunks from the correct source and included that decisive passage.

The source support was `0.695453`. The rescued chunk had cosine `0.590259` and adjusted score
`0.605895`. The fixed model did not need a deeper query, a graph edge, or a new embedding call.

## Interpretation

The independent direction matches the earlier reused gold result: source evidence can complete a
document after ordinary retrieval identifies it. Safety and labeled precision also moved in the
right direction here. This result still cannot support active serving because the selected
population contained one case and the labels were author constructed rather than independently
human adjudicated.

The next quality run should keep the model unchanged and use another disjoint source set whose
labels deliberately target decisions in later chunks of long memos. Selection must depend on source
structure and memo content only, not on either retrieval arm. This addresses the observed benchmark
saturation without selecting cases from treatment outcomes.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-holdout-103db4c2'
$env:RECALL_SOURCE_COMMIT='103db4c298cba42f6bc3567b74be1a8d4d77f6f5'
python -u scripts/run_live_source_conditioning_holdout.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-conditioning-holdout.json
Get-FileHash docs/results/2026-09-13-live-source-conditioning-holdout.json -Algorithm SHA256
```

The compact summary is reproduced with:

```powershell
$p=Get-Content -Raw docs/results/2026-09-13-live-source-conditioning-holdout.json | ConvertFrom-Json
$p.decision
$p.integrity | Format-List
$p.summary | ConvertTo-Json -Depth 8
$p.target_analysis | Format-List
```
