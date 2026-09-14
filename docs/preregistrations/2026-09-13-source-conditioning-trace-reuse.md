# Pre-registration: source conditioning trace reuse

**Date:** 2026-09-13  
**Status:** predicted, not yet measured  
**Query set:** `docs/preregistrations/2026-09-13-memory-queries-source-gold.json`  
**Query set SHA256:** `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f`  
**Model artifact:** `docs/results/2026-09-13-source-conditioning-model.json`  
**Model artifact SHA256:** `fb304c68a6ded04e28bfd9f0e9f244e45c133609101f788b5741f1a06e81242f`  
**Validation generation:** `gen_ff9076737d834e21b0205c3a360bea99`  
**Validation calibration:** `cal_473e73efffe243d097742d652b596eea`  
**Validation corpus:** `c1161c68125b45e6ff4fd8b673bc1a78e8b65ef4e5f229026b546298a74aa965`

## Motivation

The registered same vector experiment returned `BUILD SAMPLED SHADOW`, but ordinary sampled
shadow execution repeats dense retrieval, lexical retrieval, and a widened trusted search after
the serving request already fetched the same candidate legs. The repeated work makes the positive
quality treatment unnecessarily expensive and prevents a responsible increase in sample rate.

The proposed implementation captures the dense and lexical legs plus the full fused pool before
the serving `k` truncation. It trust evaluates that captured pool in memory using the same resolved
calibration and supersession state. The public result remains the original truncated trusted
result. No candidate content or identifier is added to the public diagnostic.

## Fixed protocol

Run the committed 50 query set against the pinned active Context 4 generation in one MCP process.
Source conditioning runs at sample rate one. A private benchmark gate also recomputes the legacy
duplicate query arm from the exact served query vector.

For every request, record:

1. The identifier hashes selected from the reused trace.
2. The identifier hashes selected from the duplicate query arm.
3. The reused trace computation span and duplicate query computation span.
4. The source conditioning status, serving lineage, artifact fingerprint, and public baseline
   hashes.

The normal shadow path must use only the reused trace. The duplicate arm exists only when both the
generation pin and the new private benchmark audit flag are enabled.

## Predictions

1. Reused and duplicate selected identifier hashes match in exact order on all 50 requests.
2. All 50 source conditioning diagnostics report `status=ok`, with no missing timing receipt.
3. A unit level query counter proves that sampled reuse performs one dense and one lexical query,
   the same count as ordinary serving, while the benchmark duplicate arm is disabled.
4. The median reused computation span is at most 10 percent of the median duplicate computation
   span. Its p95 is at most 20 percent of duplicate p95 and at most 15 milliseconds absolute.
5. The public trusted result has the same identifiers and order with capture enabled and disabled
   in deterministic tests.

## Decision rule

`SHIP LOW RATE SHADOW` only when all five predictions pass. This licenses a 0.01 to 0.05 diagnostic
sample rate and no active selection.

`REPAIR` when any hash, lineage, artifact, query count, or public result parity check fails.

`KEEP LOW SAMPLE` when correctness passes but either latency ratio or the absolute p95 misses.

## Reproduction command

After the implementation and this preregistration are committed to the detached VPS2 checkout:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-conditioning-reuse-<commit>'
$env:RECALL_SOURCE_COMMIT='<commit>'
python -u scripts/run_live_source_conditioning_trace_reuse.py `
  --generation-id gen_ff9076737d834e21b0205c3a360bea99 `
  --artifact docs/results/2026-09-13-source-conditioning-model.json `
  --output docs/results/2026-09-13-live-source-conditioning-trace-reuse.json
```

<!-- frozen_above -->

## Result

`SHIP LOW RATE SHADOW` on 2026-09-13. Reused and duplicate candidate hashes matched on all 50
requests, all diagnostics and timings were present, and the public result parity tests passed.
The reused computation measured median 1.720 ms and p95 4.123 ms, against 449.652 ms and 770.471
ms for the duplicate query arm. Full results and reproduction are in
`docs/results/2026-09-13-live-source-conditioning-trace-reuse.md`.
