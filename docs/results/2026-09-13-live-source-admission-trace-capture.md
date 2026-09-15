# Source admission trace capture

Measured: `2026-09-13T21:55:51.668870+00:00`.

Source commit: `962fc1f18870af7da57a5d426d1737ed0e827f98`.

Remote checkout: `/home/sentiment/recall-repos/source-admission-trace-962fc1f1`.

Raw artifact: `docs/results/2026-09-13-live-source-admission-trace-capture.json`.

Raw artifact SHA256: `facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9`.

## Purpose and integrity

This diagnostic repeat retained the complete same-vector trust pool and compact top `20` dense and
sparse leg identities for all `72` independent holdout requests. Candidate text appears once in the
trust pool and is omitted from the duplicated leg traces.

All `72` requests passed fixed alpha `0.08` candidate hash parity, public baseline hash parity,
shadow status, timing receipt, and immutable lineage checks. The artifact is suitable for
deterministic local policy replay. It is not new independent gold because both holdouts were already
inspected.

## Repeat result

Fixed alpha `0.08` again covered `34/36` facts. In this repeat, alpha `0.15` also covered `34/36`,
rescuing `buried-017` while losing `buried-007`. The prior registered screen covered `33/36` at
alpha `0.15` because `buried-017` did not cross admission in that separate query embedding call.
The repeated `buried-007` regression keeps global alpha `0.15` closed.

The change across calls is evidence that near-threshold source admission must be evaluated from the
same captured vector and pool. It does not invalidate the registered negative decision.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-admission-trace-962fc1f1'
$env:RECALL_SOURCE_COMMIT='962fc1f18870af7da57a5d426d1737ed0e827f98'
python -u scripts/run_live_source_conditioning_alpha_screen.py `
  --generation-id gen_18d5edd2e5e847c0af1ee37e40d27893 `
  --output docs/results/2026-09-13-live-source-admission-trace-capture.json
Get-FileHash docs/results/2026-09-13-live-source-admission-trace-capture.json -Algorithm SHA256
```
