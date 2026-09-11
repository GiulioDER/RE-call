# Graph cosine margin expansion

Status: preregistered before live measurement on 2026-09-10.

## Question

Does widening the graph cosine admission margin from `0.15` to `0.20` admit useful graph evidence
on the activated relation probes?

The selective gate is disabled in both arms with the diagnostic variant
`combined_no_selective`. All other graph filters remain enabled. This is a diagnostic comparison,
not authorization to change the production `combined` policy.

## Frozen inputs

* VPS2 serving commit: `bccf4d80d7a92701d10ffbfee9cf3ed3ab2eadeb`
* Tenant: `memory`
* Generation: `gen_b02a44a99917424ba3bd8011280e3712`
* Query set: `docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json`
* Query set SHA256: `f4812cb0681234f5f569aac6b77a1dd297d7c0386a6eb424c50d38edc75595ac`
* Five probes, one warmup pass, five recorded passes per invocation
* Answer provider: disabled
* Graph variant: `combined_no_selective`
* Relation control: `none`
* Hub threshold: `32`

## Arms

The runner records `off` and `one_hop` for each invocation. The primary comparison uses the
one hop rows. The repeated `off` rows are retained as a retrieval control.

* Control margin: `0.15`, policy fingerprint
  `f3017f33313b9dee242b7cd57c854affbfd9e9967255f97b8dd43419864e0e6c`
* Treatment margin: `0.20`, policy fingerprint
  `da3d2672cd0b6d446b794b10ebc167c0c8898ad4a15dc50329389b634d66c72f`

## Locked decision rule

The wider margin passes this diagnostic gate only if all conditions hold:

1. At least 2 of 5 probes record new trusted graph evidence at margin `0.20`.
2. Mean expected object hit@5 improves by at least `0.10` versus margin `0.15`.
3. Mean reciprocal rank does not decrease by more than `0.05`.
4. Trust state, answer outcome, and answer refusal remain unchanged.
5. Treatment one hop server p95 is no more than `1.25x` control one hop server p95.

The result remains diagnostic even if it passes. A production policy change requires a larger
preregistered quality and performance run.

## Exact measurement commands

Run the `0.15` control:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-cosine-015-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined_no_selective --control none --cosine-margin 0.15
```

Run the `0.20` treatment:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-cosine-020-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined_no_selective --control none --cosine-margin 0.20
```
