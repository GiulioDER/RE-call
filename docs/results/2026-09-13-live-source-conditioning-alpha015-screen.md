# Source conditioning alpha 0.15 development screen result

Measured: `2026-09-13T21:03:05.259518+00:00`.

Registered decision: `CLOSE_GLOBAL_ALPHA_INCREASE`.

Source commit: `790fc95f437228f67fc31aef4836af741fb97e28`.

Remote checkout: `/home/sentiment/recall-repos/source-conditioning-alpha015-790fc95f`.

Raw artifact: `docs/results/2026-09-13-live-source-conditioning-alpha015-screen.json`.

Raw artifact SHA256: `0b441237d4cfd796efeb77fbe4b2b82288b4a697766a009dc183435212ec5946`.

## Outcome

Global alpha `0.15` is closed. Across the two inspected holdouts, it lost one complete query and one
essential fact versus fixed alpha `0.08`, gained none, and did not change the number of answered
unanswerable controls.

| Arm | Complete | Facts | Source hits | Precision | False answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public same vector control | 33/36 | 33/36 | 34/36 | 0.6791 | 1/36 |
| Fixed alpha 0.08 | 34/36 | 34/36 | 34/36 | 0.7844 | 1/36 |
| Screen alpha 0.15 | 33/36 | 33/36 | 33/36 | 0.7778 | 1/36 |

All `72` requests passed fixed candidate hash parity, public baseline hash parity, shadow status,
timing receipt, and immutable lineage checks. Alpha `0.15` changed `12` selections relative to
alpha `0.08`, so the negative result is not a treatment liveness failure.

## Per holdout result

On the first independent holdout, both alpha settings completed `18/18` queries and covered `18/18`
facts. Precision was `0.7500` at alpha `0.08` and `0.7640` at alpha `0.15`, with zero false answers
for both.

On the buried fact challenge, alpha `0.08` completed `16/18` queries and covered `16/18` facts.
Alpha `0.15` completed `15/18` and covered `15/18`. Precision moved from `0.8228` to `0.7927`, and
each arm answered one of `18` unanswerable controls.

## The regression

`buried-007` asks what must be established before porting a concurrency fix to a sibling module.
The public control and alpha `0.08` both covered the precondition fact from
`sentiment-agent/redacted-c3e2adcb8977`.

That gold source had support `0.370772`, below neutral support `0.5`. Raising alpha therefore
lowered its adjusted score. Alpha `0.15` filled four of five positions with a distractor source at
support `0.692217` and displaced the gold item. This is the failure mode expected from a global
linear boost: strong sources become more concentrated while weaker but relevant sources lose
representation.

## Failed mechanistic prediction

Alpha `0.15` did not recover `buried-017`; both alpha settings returned no admitted item. The
earlier `0.416380` projection came from a separate hosted embedding call in the post hoc retrieval
leg diagnostic. It did not reproduce in this same vector screen. Separate-call ranks can classify
the broad loss boundary, but they are not reliable enough to predict a near-threshold same-call
selection.

## Decision and next approach

Do not raise global alpha and do not continue alpha sweeps. Alpha `0.08` remains the only source
conditioning setting with repeated positive direction and no observed holdout regression, but the
independent target populations remain too small for active selection.

The remaining opportunity is not another global scalar. Raw fused retrieval found every buried
gold source by rank `5`, while trust and final selection exposed only `16/18`. A new treatment should
separate three decisions that the current scalar mixes together:

1. source credibility from dense, lexical, and cross-leg evidence;
2. per-chunk trust admission, including a separately guarded lexical-dominant lane;
3. source diversity under the five item budget so one high-support source cannot occupy every slot.

The next work should first replay these three stages offline over captured traces and report the
maximum recoverable facts plus negative-control exposure. Only a fixed rule with visible headroom
should receive a new live holdout. Graph traversal, deeper retrieval, global alpha tuning, and
general document expansion remain closed.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-alpha015-790fc95f'
$env:RECALL_SOURCE_COMMIT='790fc95f437228f67fc31aef4836af741fb97e28'
python -u scripts/run_live_source_conditioning_alpha_screen.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-conditioning-alpha015-screen.json
Get-FileHash docs/results/2026-09-13-live-source-conditioning-alpha015-screen.json -Algorithm SHA256
```

The compact result and regression row are reproduced with:

```powershell
$p=Get-Content -Raw docs/results/2026-09-13-live-source-conditioning-alpha015-screen.json |
  ConvertFrom-Json
$p.decision
$p.integrity | Format-List
$p.summary | ConvertTo-Json -Depth 8
$p.holdout_summaries | ConvertTo-Json -Depth 8
$p.comparison | Format-List
$p.rows |
  Where-Object { $_.query.id -eq 'buried-007' } |
  ConvertTo-Json -Depth 12
```
