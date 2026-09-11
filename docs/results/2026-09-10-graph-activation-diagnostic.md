# Graph activation diagnostic

Measured: 2026-09-10 UTC

## Outcome

The directional graph diagnostic found potential quality value when traversal is actually
activated. Object document hit@5 improved from `0.400` with graph off to `0.600` with directional
one hop. The improvement came entirely from one of five relation probes, so this is a mechanism
signal, not a production quality estimate.

The diagnostic gate passes its activation condition and its point estimate:

1. All 5 of 5 probes produced admitted graph candidates.
2. Mean object document hit@5 improved by `+0.200`.
3. Trust state, outcome, and refusal reason were identical across arms: trusted, abstained, and
   `no_answer_provider`.

The diagnostic does not authorize enabling this policy in production. It removed four production
precision filters and produced substantial extra work.

## Frozen identity

| Field | Value |
|---|---|
| Query set | `docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json` |
| Query set SHA256 | `f4812cb0681234f5f569aac6b77a1dd297d7c0386a6eb424c50d38edc75595ac` |
| Queries | 5 content based relation probes |
| Generation | `gen_b02a44a99917424ba3bd8011280e3712` |
| Collection | 1 warmup plus 5 recorded passes per arm, 50 recorded rows |
| Variant | `directional` |
| Artifact | `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-activation-diagnostic-20260910.json` |
| Artifact SHA256 | `871293e144ee60c95fd2d32c4abbf4f2cf803ac4bff301daddff513623892b87` |
| Collection commit | `b8eb7759a7bb8065adffedc9a774cc6354f327eb` |

## Per probe result

| Probe | Off hit@5 | One hop hit@5 | Hit delta | Candidate rows | Accepted | New trusted evidence |
|---|---:|---:|---:|---:|---:|---:|
| activation_01 | 1.00 | 1.00 | 0.00 | 135 | 135 | 80 |
| activation_02 | 0.00 | 1.00 | +1.00 | 145 | 145 | 10 |
| activation_03 | 1.00 | 1.00 | 0.00 | 135 | 135 | 25 |
| activation_04 | 0.00 | 0.00 | 0.00 | 139 | 139 | 34 |
| activation_05 | 0.00 | 0.00 | 0.00 | 125 | 125 | 0 |

The only quality gain was `activation_02`, the relation from
`recall/a-workaround-outlives-the-problem-silently.md` to
`recall/recall-serves-trusted-from-vps2-over-ssh-stdio.md`. The expected object evidence appeared
at rank 5 after graph expansion. `activation_04` changed its trusted evidence list but still did
not return an expected object chunk, which shows that graph activity alone is not quality uplift.

The paired hit@5 deltas were `[0, 1, 0, 0, 0]`. The paired MRR deltas were `[0, 0.2, 0, 0, 0]`.
The deterministic paired bootstrap 95 percent intervals were `[0.000, 0.600]` for hit@5 and
`[0.000, 0.120]` for MRR. The paired sign permutation p value was `1.0` for each metric because
the set contains only five queries and one nonzero pair.

## Graph behavior

Across 25 one hop rows, directional traversal inspected 105 relations, discovered 679 candidates,
accepted all 679, and added 149 new trusted evidence items. No one hop request was refused by the
diagnostic variant. Graph off discovered no candidates by design.

The production combined run from the preceding report had zero candidates on all 30 relation cases.
The contrast isolates the current bottleneck: production selective and admission policies prevent
the traversal from reaching cases where the underlying graph can add evidence. The diagnostic also
shows that removing those filters admits many candidates, but most do not improve the expected
object hit.

## Latency cost

| Arm | Server p50 | Server p95 | Client p50 | Client p95 |
|---|---:|---:|---:|---:|
| Off | 832.916 ms | 992.212 ms | 1173.028 ms | 1622.059 ms |
| Directional one hop | 891.443 ms | 1237.082 ms | 1360.336 ms | 1944.741 ms |

The directional diagnostic increased server p95 by `24.7%` and client p95 by `19.9%`. This cost
is expected from the deliberately relaxed admission policy and is not a production recommendation.

## Conclusion and next test

Graph expansion can improve quality when it reaches a relevant relation: this run recovered one
object document that graph off missed. The effect is sparse, and four of five probes did not gain.
The production combined policy currently fails to reach the targeted relation cases, while the
unfiltered diagnostic pays a measurable latency cost.

The next test should keep the production combined policy and tune only one admission condition at
a time, starting with the selective gate. It should use a larger content based set and require
both graph activation coverage and no more than the preregistered quality and latency regressions.

## Reproduction

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-activation-diagnostic-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant directional --control none
```
