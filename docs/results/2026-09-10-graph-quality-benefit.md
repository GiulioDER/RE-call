# Graph quality benefit run

Measured: 2026-09-10 UTC

## Outcome

The run did not demonstrate a retrieval quality benefit for production `one_hop`. The result is
not evidence that graph expansion can never help, because the 30 graph relation cases did not
activate the graph path. The production selective gate was therefore the dominant observed
condition.

The preregistered quality gate is `PENDING` for mechanism coverage, not passed and not failed:

1. Graph relation cases: 0 of 30 produced a graph candidate in any recorded pass.
2. Ordinary controls: 1 of 20 queries activated graph candidates.
3. No query changed its trusted evidence set between graph off and one hop.
4. No one hop request added new trusted evidence, including the activated control.

## Frozen identity

| Field | Value |
|---|---|
| Query set | `docs/preregistrations/2026-09-10-graph-quality-queries.json` |
| Query set SHA256 | `59dc6a2f258e7bb331124fd30239d34f2c0776eb30a08c41d34d9e1b1aa8f80a` |
| Queries | 50, 30 graph relation and 20 ordinary control |
| Generation | `gen_b02a44a99917424ba3bd8011280e3712` |
| Corpus fingerprint | `958147ad6f3a7269940077954449772e2d1cf98191c2594d507e5d66d8f8f3b2` |
| Pipeline fingerprint | `77c918cd93f9200b36e505ae874d49a1949f04a8a85ce5b8b72a8c135b472db7` |
| Collection | 1 warmup plus 5 recorded passes per arm, 500 recorded rows |
| Artifact | `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-quality-20260910.json` |
| Artifact SHA256 | `c86007255345b8405316da8edeb5ccb585808f081e2e7e8dff6cdc08d908bb39` |
| Collection commit | `1cfc2b574b5522ea2e5d6b5d10f32e389213031e` |

## Quality results

Hit@5 means that one expected evidence id appears in the five trusted evidence items. Graph
relation ids are matched by chunk id. Ordinary controls are matched by `source:ordinal`, with
the `recall/` source prefix normalized to the frozen control labels.

| Population | n | Off hit@5 | One hop hit@5 | Delta | Off MRR | One hop MRR | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Graph relation | 30 | 0.300 | 0.300 | 0.000 | 0.189 | 0.189 | 0.000 |
| Ordinary control | 20 | 0.750 | 0.750 | 0.000 | 0.231 | 0.231 | 0.000 |

The paired bootstrap 95 percent intervals for hit@5 and MRR deltas were `[0.000, 0.000]` in
both populations. The paired permutation p value was `1.0` for both metrics in both populations.
There were no gains and no losses at query level. Every query's trusted evidence list was
unchanged across the two arms in all five paired passes.

The answer provider was disabled, so answer correctness, citation correctness, and unsupported
claim rate were not estimable. Every response was trusted. The outcome was `abstained` for every
response because of the disabled answer provider or a corpus gap.

## Where the graph path activated

The one hop arm produced 40 discovered candidates and 10 accepted candidates across the full
run. All 40 candidates came from `control_17`, the ordinary query `should I propose paid API runs
for recall`, repeated over five passes. It discovered 8 and accepted 2 on each pass. Its
`graph_relation_new_trusted_evidence` count remained zero, so activation did not affect the
returned trusted evidence.

The 30 graph relation cases produced zero discovered and zero accepted candidates. Their one hop
refusals were 145 `selective_gate` refusals and 5 `graph_gate_not_met` outcomes caused by no
eligible relation after initial retrieval. Across all 250 one hop rows, the refusal counts were
220 `selective_gate` and 10 `no_trusted_seed`; 5 rows completed graph processing with candidates.

This localizes the next quality investigation: construct graph cases from content that reliably
retrieves the relation evidence chunk as a trusted seed, then measure whether the outgoing object
chunks change hit@5 or MRR. The current filename probes identify graph relations in the database,
but they do not exercise the serving seed contract.

## Secondary latency observation

These values are descriptive only and do not replace the separate preregistered performance
experiment.

| Population | Arm | Server p50 | Server p95 | Client p50 | Client p95 |
|---|---|---:|---:|---:|---:|
| Graph relation | Off | 482.224 ms | 622.258 ms | 937.488 ms | 1308.865 ms |
| Graph relation | One hop | 515.072 ms | 618.795 ms | 1012.203 ms | 1384.546 ms |
| Ordinary control | Off | 603.488 ms | 739.487 ms | 1017.604 ms | 1435.928 ms |
| Ordinary control | One hop | 625.263 ms | 818.665 ms | 1058.247 ms | 1515.538 ms |
| All | Off | 514.779 ms | 725.210 ms | 987.919 ms | 1416.549 ms |
| All | One hop | 545.305 ms | 748.669 ms | 1052.138 ms | 1475.769 ms |

## Reproduction

The preregistration and frozen set are:

* `docs/preregistrations/2026-09-10-graph-quality-benefit.md`
* `docs/preregistrations/2026-09-10-graph-quality-queries.json`

The collection command was:

```powershell
python scripts/run_live_graph_performance_attribution.py --query-set docs/preregistrations/2026-09-10-graph-quality-queries.json --output C:\Users\gde00\AppData\Local\Temp\recall-live-graph-quality-20260910.json --generation-id gen_b02a44a99917424ba3bd8011280e3712 --passes 5 --warmup-passes 1 --variant combined --control none
```

The offline scoring used the raw artifact payloads, matched expected ids as described above, and
used deterministic bootstrap and sign permutation seeds `20260910` and `20260911`.
