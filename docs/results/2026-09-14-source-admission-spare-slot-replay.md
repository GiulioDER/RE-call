# Guarded spare-slot source rescue replay

Measured: 2026-09-14.

Development decision: `HEADROOM_FOR_FRESH_VALIDATION`.

Evaluator commit: `72fce9c6`.

Input trace: `docs/results/2026-09-13-live-source-admission-trace-capture.json`, SHA256
`facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9`.

Raw result: `docs/results/2026-09-14-source-admission-spare-slot-replay.json`, SHA256
`2ec1e795788d0613f7fa90664dc7298a9ded9d19b7d0cee99afa1f13daaa387d`.

## Status

This is post hoc development evidence. The failure cases and their source features were inspected
before the rule was fixed. The result establishes recoverable headroom and a concrete candidate for
fresh validation. It does not estimate production prevalence and cannot license active selection.

## Fixed replay rule

The treatment begins with the exact alpha `0.08` selection and never removes or reorders an accepted
item. It runs only when fewer than five items were selected. It considers only unrepresented sources
and independently trust-evaluated chunks with verdict `ok` or `low_confidence`.

A source enters the dual-leg lane when its best dense and sparse ranks are each at most `5` and its
maximum chunk cosine is at least `0.35`. A source enters the lexical-dominant lane when it is sparse
rank `1`, contributes at least two sparse top `10` chunks, and has maximum cosine at least `0.28`.
Eligible rescue chunks use lane floors `0.35` and `0.28` respectively. Sources are ordered by lane,
rank, fitted support, and stable source identity. Items are added round robin until the existing
five-item budget is full.

## Result

| Arm | Complete | Facts | Source hits | Precision | False answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed alpha 0.08 | 34/36 | 34/36 | 34/36 | 0.7844 | 1/36 |
| Spare-slot rescue | 36/36 | 36/36 | 36/36 | 0.7907 | 1/36 |

All `72` captured alpha `0.08` selections recomputed exactly, and all `72` candidate selections
preserved the alpha `0.08` prefix. The rule changed only two queries and added five items. It gained
`buried-015` and `buried-017`, with no complete or fact loss and no increase in answered
unanswerable controls.

The independent holdout was unchanged at `18/18` facts and zero false answers. The buried challenge
improved from `16/18` to `18/18` facts, while its precision rose from `0.8228` to `0.8333` and its
false answer count stayed at one of `18`.

## Rescue cases

For `buried-015`, alpha `0.08` had three items and no gold source. The lexical-dominant lane filled
the two spare slots with ordinals `10` and `12` from
`recall/benchd-official-rules-2026-08-23.md`, covering `open_followups`. The source was sparse rank
`1`, absent from dense top `20`, and had fitted support `0.368374`. This is the case a larger global
alpha cannot repair because it penalizes support below `0.5`.

For `buried-017`, alpha `0.08` abstained with zero items. The dual-leg lane added ordinals `10`, `9`,
and `8` from `recall/one-test-must-cross-the-seam.md`, covering `initial_state_test`. The source was
rank `1` in both legs and had fitted support `0.769714`.

## Why this differs from alpha 0.15

Global alpha `0.15` changes competition even when the five-item budget is already full. It repeatedly
lost `buried-007` by concentrating four slots on one high-support distractor source. Spare-slot
rescue cannot create that regression because it never replaces or reorders the base selection.

The remaining safety risk is added evidence on a query that should have remained unanswered. The
captured 36 negative controls did not expose that failure, but the gates were chosen after inspecting
this data. Fresh negatives and human review are therefore the binding next test.

## Next experiment

Implement this rule as a private same-vector shadow candidate using the already captured main-path
dense, sparse, and trust pool traces. Record aggregate activation counts, item counts, lane counts,
and identifier hashes. Public evidence must remain unchanged.

Validate on fresh human-reviewed queries sampled without using treatment outcomes. The primary
population is queries where alpha `0.08` returns fewer than five items and at least one rescue lane
fires. Require fact or answer support gains, zero evidence losses by construction, and no increase in
false answers. Do not use this inspected 72-query set as promotion evidence.

## Reproduction

```powershell
python -u scripts/replay_source_admission_spare_slots.py `
  --trace docs/results/2026-09-13-live-source-admission-trace-capture.json `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-14-source-admission-spare-slot-replay.json
Get-FileHash docs/results/2026-09-14-source-admission-spare-slot-replay.json -Algorithm SHA256
```

Focused verification on 2026-09-14:

```powershell
python -m pytest `
  tests/test_replay_source_admission_spare_slots.py `
  tests/test_live_source_conditioning_alpha_screen.py `
  tests/test_live_source_conditioning_holdout.py `
  tests/test_source_conditioning.py `
  tests/test_live_source_conditioned_admission.py `
  tests/test_reasoning_embedding_reuse.py `
  tests/test_doc_citations.py -q
python -m ruff check `
  scripts/replay_source_admission_spare_slots.py `
  tests/test_replay_source_admission_spare_slots.py
python -m mypy --explicit-package-bases --follow-imports=skip `
  scripts/replay_source_admission_spare_slots.py
git diff --check
```

Observed focused result: `60 passed`.
