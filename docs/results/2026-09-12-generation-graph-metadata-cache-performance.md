# Generation graph metadata cache performance result

Measured on 2026-09-12 against VPS2 after merge commit
`fe3a3bad7390197f35e91c6ed944fbb6a99c3574`.

## Parity

1. Serving checkout: `fe3a3bad7390197f35e91c6ed944fbb6a99c3574`.
2. Imported `recall_mcp/service.py` SHA256:
   `4cda494ee0f251e351dba1cb3c779b643c447aca569b6a0be2baa24873f7de93`.
3. Schema: current `0025`, required `0025`.
4. MCP handshake: 22 tools.
5. Memory tenant: certified on generation `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.
6. One hop policy fingerprint:
   `0151785350a8c678b3e254e4f9b356f71602b4a022dc3cc3f0a848b36e45cf59`.

## Protocol

The frozen 50 query set was used with one warmup pass and five recorded passes per arm. The query
set digest is `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`.

Artifact: `2026-09-12-generation-graph-metadata-cache-performance.json`.

Remeasure command:

```powershell
$env:RECALL_SSH_EXECUTABLE='C:\Program Files\Git\usr\bin\ssh.exe'
python scripts/run_live_graph_performance_attribution.py --output docs/results/2026-09-12-generation-graph-metadata-cache-performance.json --generation-id gen_6559b6dbacfc4bc2847b65005f1ba1d4 --passes 5 --warmup-passes 1 --variant combined --control none
```

## Warm recorded results

| Metric | No graph p50 | No graph p95 | One hop p50 | One hop p95 |
| --- | ---: | ---: | ---: | ---: |
| Total server ms | 424.957 | 546.855 | 794.022 | 931.360 |
| Baseline retrieval ms | 424.887 | 546.167 | 793.904 | 930.949 |
| Graph readiness check ms | 0.000 | 0.000 | 0.007 | 0.017 |
| Projection load ms | 0.000 | 0.000 | 0.015 | 0.029 |
| Candidate fetch ms | 0.000 | 0.000 | 2.563 | 4.559 |
| Cosine rescoring ms | 0.000 | 0.000 | 1.377 | 2.958 |
| Database statements | 6 | 6 | 14 | 14 |
| Database result bytes | 54,767 | 58,156 | 80,880 | 90,711 |
| Database transferred bytes | 55,037 | 58,436 | 84,279 | 95,030 |

Compared with the prior pre cache one hop artifact, one hop p95 improved from 1,352.445 ms to
931.360 ms, a 31.1 percent reduction. Database result bytes p95 improved from 452,602 bytes to
90,711 bytes, an 80.0 percent reduction. Readiness p95 improved from 6.431 ms to 0.017 ms, a 99.7
percent reduction. Database statement p95 improved from 15 to 14.

The primary graph latency gate required one hop p95 to be at least 50 percent below no graph p95
on this run. It failed. One hop p95 was 70.3 percent above no graph p95. The optimization reduced
database work substantially, but graph serving remains slower overall because candidate retrieval
and rescoring still add cost.

## Cache and graph behavior

One hop recorded rows had 250 projection cache hits, zero misses, and zero single flight waiters.
The graph was ready on all 250 one hop rows. There were 165 rows with candidates and 85 selective
gate refusals. Candidate count summed to 3,360, while the bounded text fetch count summed to 1,540.

The selective refusal smoke row was query index 0, `why does the test suite have no default
database DSN`. It reported `graph_gate_not_met`, zero discovered candidates, and the expected
policy fingerprint.

The positive expansion smoke row was query index 1, `can two pytest sessions run at the same time
here`. It discovered and scored 24 candidates, accepted 24 `references` relation candidates, and
reported the expected policy fingerprint.

## Quality follow up

The answer quality replay was preregistered before any answer request in
`2026-09-12-generation-graph-metadata-cache-answer-quality.md`. It is blocked because
`OPENROUTER_API_KEY` is absent from both checked runtime environments and the local environment
files. No quality conclusion or factual answer claim is made from this performance run.
