# Graph first cache verification

Measured on 2026-09-12 after PR 631 merged and VPS2 synchronized.

## Deployment parity smoke

1. Serving checkout: `/home/sentiment/recall-repos/serving`, branch `serving-live`, commit
   `a4043f3beb6eedca4a641cc01fb325c7af550abe`.
2. Imported `recall_mcp/service.py` SHA256: `4cda494ee0f251e351dba1cb3c779b643c447aca569b6a0be2baa24873f7de93`.
3. Schema compatibility: current `0025`, required `0025`.
4. MCP handshake: 22 tools.
5. One hop policy fingerprint: `0151785350a8c678b3e254e4f9b356f71602b4a022dc3cc3f0a848b36e45cf59`.
6. Selective gate refusal: query index 0, `graph_gate_not_met`, zero admitted candidates.
7. Positive graph expansion: query index 1 discovered 24 candidates and admitted 24 `references`
   candidates. The query vector was reused. No new trusted evidence was added for this query.

The serving verification passed with `scripts/session-serving.sh verify` and the corpus remained
pinned to generation `gen_6559b6dbacfc4bc2847b65005f1ba1d4`.

## Frozen performance run

The locked 50 query set was run with one warmup pass and five recorded passes per arm. The raw
artifact is `2026-09-12-graph-first-cache-performance-rerun.json`. It was measured with:

```powershell
$env:RECALL_SSH_EXECUTABLE='C:\Program Files\Git\usr\bin\ssh.exe'
python scripts/run_live_graph_performance_attribution.py --output docs/results/2026-09-12-graph-first-cache-performance-rerun.json --generation-id gen_6559b6dbacfc4bc2847b65005f1ba1d4 --passes 5 --warmup-passes 1 --variant combined --control none
```

| Metric | Graph off | Graph one hop |
| --- | ---: | ---: |
| Client latency p50 / p95, ms | 774.476 / 1036.371 | 1360.118 / 2846.039 |
| Server total p50 / p95, ms | 533.502 / 696.476 | 943.580 / 1175.146 |
| Baseline retrieval p50 / p95, ms | 533.215 / 695.789 | 943.054 / 1174.596 |
| Database bytes p50 / p95 | 145617 / 149016 | 445180 / 455355 |
| Database statements p50 / p95 | 7 / 7 | 17 / 17 |

The one hop server p95 was 1.687 times graph off. The secondary availability guard passes because
1175.146 ms is below 2 times 696.476 ms. This is not an overall latency improvement.

The cache behavior is positive and isolated. Recorded one hop requests had 250 projection cache
hits and zero misses. The first warmup request recorded one projection miss, a 1971.385 ms
projection load, a 96.337 ms adjacency construction, and 10139143 application bytes. The recorded
one hop projection p95 was 0.033 ms. The single cold sample is not enough to certify the locked
cold versus warm graph stage p95 gate, so that primary gate remains pending.

The recorded one hop query vector was reused on 165 of 250 requests. Candidate payload p95 was
8222 bytes, candidate fetch p95 was 5.092 ms, and cosine rescoring p95 was 3.501 ms. The one hop
database byte p95 remains 3.056 times graph off, so graph expansion still adds database work.

The separate eight request same key probe recorded one projection single flight owner and seven
waiters. Its client batch time was 10177.006 ms. The raw probe is
`2026-09-12-graph-first-cache-single-flight.json`.

## DeepSeek answer quality replay

The answer replay used the exact performance artifact, `deepseek/deepseek-v4-flash`, temperature
zero, reasoning effort `none`, and a 512 token maximum. The preregistration is
`2026-09-12-graph-first-cache-answer-quality.md`, and the raw artifact is
`2026-09-12-graph-first-cache-answer-quality.json`.

| Metric | Graph off | Graph one hop | Difference |
| --- | ---: | ---: | ---: |
| Valid answer envelope rate | 1.000 | 1.000 | 0.000 |
| Nonempty answer rate | 0.340 | 0.380 | plus 0.040 |
| Correct abstention rate | 0.964 | 0.964 | 0.000 |
| Any gold citation rate | 0.220 | 0.220 | 0.000 |
| Complete gold citation coverage | 0.020 | 0.080 | plus 0.060 |
| Mean gold citation precision | 0.520 | 0.482 | minus 0.037 |
| Mean gold citation recall | 0.205 | 0.271 | plus 0.066 |
| Provider failure rate | 0.000 | 0.000 | 0.000 |

All 50 paired queries had unchanged valid answer status. Any gold citation was unchanged on all
50 pairs. The observed citation recall gain is evidence quality only, not factual correctness, and
is attributable to the one hop evidence differences rather than to cache reuse itself.

## Conclusion

PR 631 successfully removes repeated full projection loads on the warm path and preserves single
flight behavior. It does not yet make graph one hop faster than graph off overall. The next high
value optimization is reducing the remaining graph database work, especially the repeated
application payload and relation candidate inspection. The locked cold versus warm graph stage
gate remains pending because only one cold projection sample was captured.

Remeasure the raw performance result with the command above. Rerun the quality result with:

```powershell
python scripts/run_live_graph_answer_quality.py docs/results/2026-09-12-graph-first-cache-performance-rerun.json docs/results/2026-09-12-graph-first-cache-answer-quality.json --source-commit a4043f3beb6eedca4a641cc01fb325c7af550abe --model deepseek/deepseek-v4-flash --workers 8
```
