# Graph cosine margin expansion result

Measured on 2026-09-10 against VPS2 serving commit
`bccf4d80d7a92701d10ffbfee9cf3ed3ab2eadeb`.

## Outcome

Widening the graph cosine admission margin from `0.15` to `0.20` admitted substantially more
trusted graph evidence, but did not improve ranking quality on the frozen probes. Expected object
hit@5 stayed at `0.600` and MRR stayed at `0.290` in both one hop arms.

The preregistered quality gate therefore failed. Production `combined` remains unchanged.

## Frozen identity and artifacts

* Query set SHA256: `f4812cb0681234f5f569aac6b77a1dd297d7c0386a6eb424c50d38edc75595ac`
* Generation: `gen_b02a44a99917424ba3bd8011280e3712`
* Variant: `combined_no_selective`
* Control margin: `0.15`, policy fingerprint
  `f3017f33313b9dee242b7cd57c854affbfd9e9967255f97b8dd43419864e0e6c`
* Treatment margin: `0.20`, policy fingerprint
  `da3d2672cd0b6d446b794b10ebc167c0c8898ad4a15dc50329389b634d66c72f`
* Protocol: one warmup pass and five recorded passes per arm, 50 rows per artifact
* Control measured at: `2026-09-10T18:56:47.207508+00:00`
* Treatment measured at: `2026-09-10T18:58:51.854486+00:00`
* Control artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-cosine-015-20260910.json`
* Control artifact SHA256: `ad9d28f63016933480268c6e7e93fb03829f98a5114ae32c1bf943780e8eaf21`
* Treatment artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-cosine-020-20260910.json`
* Treatment artifact SHA256: `4eacf8760bbf9340b0572a3416b4b2ea02804e07d5b7ada04200e254ec1c2c2a`

## Quality and graph admission

| Metric | Margin 0.15 | Margin 0.20 | Delta |
| --- | ---: | ---: | ---: |
| Expected object hit@5 | 0.600 | 0.600 | 0.000 |
| Mean reciprocal rank | 0.290 | 0.290 | 0.000 |
| Probes with graph candidates | 5 of 5 | 5 of 5 | 0 |
| New trusted graph evidence | 20 | 80 | +60 |
| Cosine admission rejections | 603 | 509 | -94 |
| Trust admission rejections | 31 | 65 | +34 |
| Trust state | 25 trusted | 25 trusted | unchanged |
| Answer outcome | 25 abstained | 25 abstained | unchanged |
| Answer refusal | 25 `no_answer_provider` | 25 `no_answer_provider` | unchanged |

Per probe, new trusted evidence at margin `0.15` was `2`, `2`, `0`, `0`, and `0`. At margin
`0.20`, it was `8`, `2`, `1`, `5`, and `0`. The extra accepted evidence did not place the expected
object into the top five for activation_04 or activation_05.

## Performance attribution

| Metric | Margin 0.15 p95 | Margin 0.20 p95 | Delta |
| --- | ---: | ---: | ---: |
| Server total, ms | 727.125 | 725.607 | -0.2% |
| Client observed total, ms | 1,404.353 | 1,498.868 | +6.7% |
| Graph readiness check, ms | 6.794 | 6.524 | -0.270 |
| Candidate fetch, ms | 6.107 | 6.635 | +0.528 |
| Cosine rescoring, ms | 3.490 | 3.496 | +0.006 |
| Database statements, all 25 rows | 359 | 379 | +20 |
| Database transferred bytes, all 25 rows | 10,265,080 | 11,154,130 | +889,050 |
| Projection cache hits | 25 | 25 | unchanged |
| Adjacency cache hits | 25 | 25 | unchanged |

## Decision rule

The preregistered conditions produced four passes and one failure:

1. At least 2 of 5 probes record new trusted evidence at `0.20`: passed, 4 of 5.
2. Mean expected object hit@5 improves by at least 0.10: failed, delta 0.000.
3. MRR does not decrease by more than 0.05: passed, delta 0.000.
4. Trust, answer refusal, and outcome parity: passed.
5. Treatment server p95 no more than 1.25x control: passed, ratio 0.998x.

## Conclusion and next test

The cosine margin is not the only missing quality ingredient. A wider margin lets more graph
evidence through, but the added evidence is not ranked into the top five on this query set and
increases database traffic. I will not widen the production margin based on this result.

The next useful test is not another blind threshold increase. It should inspect whether the
candidate query scores and relation evidence are calibrated against the seed scores, then use a
larger frozen set to test a bounded candidate rerank or relation corroboration rule.

## Reproduction commands

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-cosine-015-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined_no_selective --control none --cosine-margin 0.15
```

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-cosine-020-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined_no_selective --control none --cosine-margin 0.20
```
