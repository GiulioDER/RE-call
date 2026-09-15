# Live source conditioning trace reuse result

**Measured:** 2026-09-13T19:34:19.491426+00:00  
**Registered verdict:** `SHIP LOW RATE SHADOW`  
**Generation:** `gen_ff9076737d834e21b0205c3a360bea99`  
**Calibration:** `cal_473e73efffe243d097742d652b596eea`  
**Pipeline:** `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`  
**Corpus:** `c1161c68125b45e6ff4fd8b673bc1a78e8b65ef4e5f229026b546298a74aa965`  
**Source commit:** `b2ca5d8639de4252db0a20bf1f128be463d13b50`  
**Remote checkout:** `/home/sentiment/recall-repos/source-conditioning-reuse-b2ca5d86`  
**Raw artifact:** `docs/results/2026-09-13-live-source-conditioning-trace-reuse.json`  
**Raw artifact SHA256:** `ee635d3b4cc9b0bcb9a109258e40c0d805bb5876f5758bcd3aa2a9f8327d6d47`

## Outcome

The main retrieval path can now supply source conditioning without repeating dense retrieval,
lexical retrieval, or the widened trusted search. It captures the already fetched legs and full
fused pool before the public `k=5` truncation, then performs only an in memory trust evaluation
and fitted source selection. The public trusted result remains unchanged.

All 50 reused selections exactly matched the former duplicate query implementation. Every shadow
and paired audit reported `status=ok`, and every timing receipt was present.

| Computation | Median | p95 |
|---|---:|---:|
| reused candidate trace | 1.720 ms | 4.123 ms |
| duplicate query path | 449.652 ms | 770.471 ms |
| reuse divided by duplicate | 0.003824 | 0.005351 |

The reused median was 99.6176 percent lower than the duplicate median. Reused p95 was 99.4649
percent lower than duplicate p95. The absolute reused p95 remained below the registered 15 ms
ceiling.

## Safety and mechanics

The callback captures candidates before truncation but returns a separately truncated result.
Deterministic tests prove identical public identifiers and order with capture enabled and disabled.
A query counting test proves one dense and one lexical query with capture enabled, the same counts
as ordinary serving. The extra trust evaluation suppresses operational search and verdict counters,
so one sampled diagnostic does not appear as a second user search.

The duplicate comparison is available only when both `RECALL_BENCHMARK_PIN=1` and
`RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT=1` are set. Ordinary shadow sampling never runs
that arm. Candidate content and raw identifiers remain private; the public shadow diagnostic emits
only aggregate counts and SHA256 identifier hashes. Source security requests remain excluded from
the shadow.

## Interpretation

This closes the main operational blocker identified by the positive same vector quality result.
It does not add new gold evidence by itself. It makes the already positive source conditioning
treatment cheap enough for a controlled 0.01 to 0.05 diagnostic sample.

The live run intentionally enabled the duplicate benchmark arm, so its 1355.724 ms median client
latency is not an estimate of ordinary shadow latency. The registered comparison uses request local
spans from the same process and exact query vector. A production sample without the audit gate is
the next source of end to end overhead evidence.

The next quality work should create fresh blinded gold for the observed failure class: the correct
source is present, but the public top chunks omit the decisive passage. Active selection still
requires independent quality evidence or online human judgments.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-reuse-b2ca5d86'
$env:RECALL_SOURCE_COMMIT='b2ca5d8639de4252db0a20bf1f128be463d13b50'
python -u scripts/run_live_source_conditioning_trace_reuse.py `
  --generation-id gen_ff9076737d834e21b0205c3a360bea99 `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-13-live-source-conditioning-trace-reuse.json
Get-FileHash docs/results/2026-09-13-live-source-conditioning-trace-reuse.json -Algorithm SHA256
```

The summary is reproduced from the raw artifact with:

```powershell
$p=Get-Content -Raw docs/results/2026-09-13-live-source-conditioning-trace-reuse.json | ConvertFrom-Json
$p.decision
$p.summary | Format-List
```
