# Graph selective gate isolation result

Measured on 2026-09-10 against VPS2 after deployment of commit
`b18fe95bd184c75a0eccef6b8030174ed44a0d37`.

## Outcome

The selective gate was a real activation blocker on these five probes, but removing it did not
produce a quality benefit. The production `combined` control refused graph expansion on all 25
recorded one hop requests. The `combined_no_selective` treatment traversed all five probes and
fetched 654 candidates, but cosine admission rejected all 654. Expected object hit@5 stayed
`0.400` in both arms and mean reciprocal rank stayed `0.250` in both arms.

The preregistered diagnostic gate therefore failed on its quality condition. This does not support
removing selective admission from production.

## Frozen identity and artifacts

* Query set SHA256: `f4812cb0681234f5f569aac6b77a1dd297d7c0386a6eb424c50d38edc75595ac`
* Generation: `gen_b02a44a99917424ba3bd8011280e3712`
* Control: `combined`, policy fingerprint
  `bd95d38fc1603f7f591096172bfd69a576ae99340afd1bda0d593a6b852e1854`
* Treatment: `combined_no_selective`, policy fingerprint
  `de51f7a8b5f6ad757a02479252311e6b1604dc81ea2f9558bc06613a14ba3196`
* Protocol: one warmup pass and five recorded passes per arm, 50 rows per artifact
* Control measured at: `2026-09-10T18:33:00.992664+00:00`
* Treatment measured at: `2026-09-10T18:35:05.302572+00:00`
* Control artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-selective-control-20260910.json`
* Control artifact SHA256: `a3cc8d3502cd91eb448a54b4696165cda78d76fad2917271910270d950d992ff`
* Treatment artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-selective-treatment-20260910.json`
* Treatment artifact SHA256: `6b7562a580e33932a7e612428696699fe86f3c124869c717cd337247e3367990`

## Quality

| Metric | Control combined | Treatment without selective | Delta |
| --- | ---: | ---: | ---: |
| Expected object hit@5 | 0.400 | 0.400 | 0.000 |
| Mean reciprocal rank | 0.250 | 0.250 | 0.000 |
| Probes with graph candidates | 0 of 5 | 5 of 5 | +5 |
| New trusted evidence | 0 | 0 | 0 |
| Trust state | 25 trusted | 25 trusted | unchanged |
| Answer outcome | 25 abstained | 25 abstained | unchanged |
| Answer refusal | 25 `no_answer_provider` | 25 `no_answer_provider` | unchanged |

The control recorded `selective_gate` graph expansion refusal on all 25 one hop rows. The
treatment recorded no selective refusal. It inspected 80 relation activations, fetched and scored
654 candidates, and recorded 654 `cosine_admission` rejections. The treatment also recorded 4,695
`relation_evidence_not_trusted` rejections and 25 `hub_entity` rejections. The remaining filters,
not the selective gate, prevented admitted evidence from reaching the top five.

Per probe, treatment candidate counts were `27`, `29`, `22`, `28`, and `25` for activation_01
through activation_05. The expected object was already present for activation_01 and activation_03
in the control, and remained absent for activation_02, activation_04, and activation_05.

## Performance attribution

The server p95 comparison is below. The treatment stayed within the preregistered `1.50x`
latency guard, but added graph work without quality benefit.

| Metric | Control one hop p95 | Treatment one hop p95 | Delta |
| --- | ---: | ---: | ---: |
| Server total, ms | 708.530 | 735.768 | +3.8% |
| Client observed total, ms | 1,384.661 | 1,610.561 | +16.3% |
| Graph readiness check, ms | 7.200 | 11.470 | +4.270 |
| Candidate fetch, ms | 0.000 | 10.294 | +10.294 |
| Cosine rescoring, ms | 0.000 | 6.760 | +6.760 |
| Database statements, all 25 rows | 204 | 329 | +125 |
| Database transferred bytes, all 25 rows | 7,987,771 | 8,931,505 | +943,734 |
| Projection cache hits | 0 | 25 | +25 |
| Adjacency cache hits | 0 | 25 | +25 |

The control’s zero cache counts reflect the early selective return before projection. The treatment
had 25 projection and 25 adjacency cache hits, with no misses or single flight waiters.

## Decision rule

The preregistered conditions produced four passes and one failure:

1. At least 2 of 5 probes activated: passed, 5 of 5.
2. Mean expected object hit@5 improves by at least 0.10: failed, delta 0.000.
3. MRR does not decrease by more than 0.05: passed, delta 0.000.
4. Trust, answer refusal, and outcome parity: passed.
5. Treatment server p95 no more than 1.50x control: passed, ratio 1.038x.

## Conclusion and next test

Selective admission is not the only limiting factor. Once it was removed, the cosine admission
filter rejected every fetched candidate on this set. I will keep production `combined` unchanged.
The next isolation should preregister cosine margin behavior on the same activated relation probes,
with selective admission disabled only for the diagnostic arm and a bounded candidate acceptance
guard. A larger quality set is required before any production policy change.

## Reproduction commands

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-selective-control-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined --control none
```

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-activation-diagnostic-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-selective-treatment-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined_no_selective --control none
```
