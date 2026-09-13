# Pre-registration: cache immutable generation graph metadata reads

**Date:** 2026-09-12  **Status:** preregistered before measurement

## Question

Does caching the immutable generation supersession closure and positive graph readiness marker
reduce the warm one-hop graph serving cost without changing graph admission or answer quality?

## Change under test

The treatment is commit `021aef78`, or the exact merge commit containing it. It changes
`GenerationStore.supersession_all` to cache the closure by immutable generation id and changes
`GenerationStore.graph_readiness` to cache only ready markers by generation id. Negative readiness
results remain uncached so a graph may become ready during an administrative build. The cache
returns defensive copies for the public supersession result.

## Frozen inputs

The comparison uses the existing 50 query set at
`docs/preregistrations/2026-08-17-memory-queries.json`, digest
`63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`, tenant `memory`, and the
same pinned generation used by the previous run. The control is the previously measured master
artifact `docs/results/2026-09-12-graph-first-cache-performance-rerun.json`. The treatment must
use the same command shape, one warmup pass, five recorded passes, graph off and graph one hop,
and the same server profile and graph policy arguments.

## Predictions and gates

The primary cost prediction is that recorded one-hop `db_result_bytes` p95 will fall by at least
40 percent from the prior `455355` byte p95, because the repeated full supersession result is no
longer transferred after the first request. Recorded one-hop `db_statement_count` p95 is
expected to fall from `17` to at most `15`. The no-candidate one-hop rows must also lose the
repeated graph metadata statements and remain eligible for the same selective gate decisions.

The quality guard is no loss in valid answer envelope rate, correct abstention rate, any gold
citation rate, or mean gold citation recall versus the prior DeepSeek V4 Flash replay. The exact
quality replay remains separate from retrieval performance and must use the already committed
quality preregistration and model settings.

Any prediction failure, schema or parity mismatch, changed policy fingerprint, changed query
set, generation mismatch, or quality regression rejects the optimization for live serving even if
latency improves.

## Measurement commands

After the treatment is merged and VPS2 is synchronized, record the serving commit, imported
`recall_mcp/service.py` SHA256, schema version, MCP tool count, and policy fingerprint with:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' scripts/session-serving.sh sync
& 'C:\Program Files\Git\bin\bash.exe' scripts/session-serving.sh verify
```

Then run the frozen performance comparison, writing a new immutable output file:

```powershell
$env:RECALL_SSH_EXECUTABLE='C:\Program Files\Git\usr\bin\ssh.exe'
python scripts/run_live_graph_performance_attribution.py --output docs/results/2026-09-12-generation-graph-metadata-cache-performance.json --generation-id <pinned-generation-id> --passes 5 --warmup-passes 1 --variant combined --control none
```

The report must compare client and server latency, readiness, projection and adjacency cache
counters, candidate counts and payload bytes, database statements, parameters and result bytes,
and the selective refusal and positive expansion smoke rows. Only after that comparison passes
its parity checks may the separate DeepSeek quality command be run.

## Analysis rule

Use p95 for the preregistered cost gates and report p50 beside it. Report the first warmup row as
a single cold sample, not as a cold p95. Do not claim the previous 50 percent warm graph latency
gate passed unless the treatment server p95 is at least 50 percent below the graph off server p95
on the same run. State explicitly when the cache reduces database work but graph one hop remains
slower overall.

