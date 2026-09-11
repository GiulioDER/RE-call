# Graph selective gate isolation

Status: preregistered before live measurement on 2026-09-10.

## Question

Does the production selective admission gate, rather than the other combined graph precision
filters, prevent useful one hop graph expansion on the activated relation probes?

This is a diagnostic policy comparison. The treatment is not a production rollout. It keeps
directional traversal, corroboration, hub suppression, and cosine admission identical to the
production `combined` policy, and disables only selective admission.

## Frozen inputs

* VPS2 serving commit: `b18fe95bd184c75a0eccef6b8030174ed44a0d37`
* Tenant: `memory`
* Generation: `gen_b02a44a99917424ba3bd8011280e3712`
* Query set: `docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json`
* Query set SHA256: `f4812cb0681234f5f569aac6b77a1dd297d7c0386a6eb424c50d38edc75595ac`
* Five probes, one warmup pass, five recorded passes per invocation
* Answer provider: disabled, so this run measures retrieval and graph evidence only
* Relation control: `none`
* Hub threshold: `32`
* Cosine margin: `0.10`

## Arms

The runner records `off` and `one_hop` for each invocation. The comparison uses the `one_hop`
rows from the two invocations and requires the `off` rows to remain a repeated retrieval control.

* Control: `RECALL_GRAPH_PRECISION_VARIANT=combined`, policy fingerprint
  `bd95d38fc1603f7f591096172bfd69a576ae99340afd1bda0d593a6b852e1854`
* Treatment: `RECALL_GRAPH_PRECISION_VARIANT=combined_no_selective`, policy fingerprint
  `de51f7a8b5f6ad757a02479252311e6b1604dc81ea2f9558bc06613a14ba3196`

## Locked decision rule

The treatment passes this diagnostic gate only if all conditions hold:

1. At least 2 of 5 probes produce one hop candidates.
2. Mean expected object hit@5 improves by at least `0.10` versus the control.
3. Mean reciprocal rank does not decrease by more than `0.05`.
4. Trust state, refusal reason, and answer outcome remain unchanged.
5. Treatment one hop server p95 is no more than `1.50x` control one hop server p95.

The result remains diagnostic even if it passes. A production policy change requires a larger
preregistered quality and performance run.

## Exact measurement commands

Run control first:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-selective-control-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined --control none
```

Run treatment second:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-selective-treatment-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined_no_selective --control none
```
