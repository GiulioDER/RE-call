# Live graph performance attribution, 2026-09-10

This report records the first corrected live run after the preregistration. The preregistration was
committed before measurement in
`docs/preregistrations/2026-09-10-live-graph-performance-attribution.md` and was not edited after
the first live request.

## Run identity

* VPS2 serving commit: `266438ff196fbd33a9b1ca25b5a4c2d0b152272a`
* Schema: current `0024`, required `0024`
* MCP handshake: 22 tools
* Generation: `gen_b02a44a99917424ba3bd8011280e3712`
* Query count: 50, five recorded passes per arm, 250 recorded requests per arm
* Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`
* Artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-graph-performance-20260910-corrected.json`

Remeasure command:

```powershell
.venv/Scripts/python.exe scripts/run_live_graph_performance_attribution.py `
  --generation-id gen_b02a44a99917424ba3bd8011280e3712 `
  --output $env:TEMP/recall-live-graph-performance-20260910-corrected.json `
  --passes 5 --warmup-passes 1 --variant combined --control none --timeout 240
```

## Result

| Metric | Graph off | One hop |
| --- | ---: | ---: |
| Client p50 / p95, ms | 1131 / 1635 | 2965 / 3476 |
| Server total p50 / p95, ms | 621 / 1020 | 2447 / 2764 |
| Baseline retrieval p50 / p95, ms | 621 / 1019 | 555 / 733 |
| Database statements p50 / p95 | 7 / 7 | 12 / 16 |
| Application payload bytes p50 / p95 | 144,026 / 149,405 | 9,912,172 / 9,920,275 |

The one hop server p95 was 2.71 times graph off. Baseline retrieval was slightly faster in one hop,
so the added latency is in the graph path, not retrieval.

## Attribution

For one hop, across all 250 recorded requests:

* Graph stage p50 / p95: 1,903 / 2,125 ms
* Readiness check p50 / p95: 1,902 / 2,125 ms
* Projection load p95: 0.024 ms
* Adjacency construction p95: 0 ms
* Candidate fetch p95: 0 ms, mean 0.153 ms
* Cosine rescoring p95: 0 ms, mean 0.064 ms
* Trust reevaluation p95: 0 ms, mean 0.005 ms
* Candidates were discovered and admitted in 5 of 250 requests, eight candidates each, 40 total
* Projection and adjacency cache hits were observed in 30 requests; misses occurred in the
  unrecorded warmup pass

The current bottleneck is the persisted graph readiness check. Projection, adjacency, rescoring,
and trust reevaluation are not the source of the measured tail.

## Quality and gates

All 250 paired requests had the same outcome, refusal reason, and evidence IDs. The run therefore
shows no quality regression, but it also shows no quality gain on this frozen set. All one hop
requests reported `graph_readiness=ready`.

The preregistered primary cold versus warm graph-stage gate is `PENDING`: the warmup pass was
intentionally unscored, so the corrected artifact contains no cold graph-stage sample. The warm
recorded one hop graph-stage p95 was 2,125 ms. The secondary availability comparison does not
support rollout: one hop server total p95 was 2,764 ms versus 1,020 ms for graph off.

The separate live eight-request same-key probe produced one projection single-flight owner and
seven waiters, with one projection miss per contender and no duplicate projection build. Probe wall
time was 24,449 ms; it is a concurrency diagnostic, not part of the sequential p95.

## Verification commands

Deployment parity was rechecked with:

```powershell
Get-Content -Raw -LiteralPath scripts/session_serving_remote.sh |
  ssh.exe -o BatchMode=yes -o ConnectTimeout=15 vps2 'bash -s -- verify'
```

Focused regression verification was 65 passed tests, and ruff passed for the changed files. The
single-flight test was also run against a deliberate coordination mutation and failed with two
loads instead of one, then passed after restoration.

