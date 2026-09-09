# Pre registration: graph projection cache latency

**Date:** 2026-09-09   **Status:** predicted, not yet measured

## The question

What are the cold and warm p50 and p95 latencies for a generation backed graph projection in the
same process?

## What I predict

With 100 samples over a deterministic 100 chunk in memory store, cold projection p50 will be at
least 5 times warm projection p50, and cold p95 will be at least 5 times warm projection p95.

## What would falsify this

Either cold ratio is below 5, or the measurement does not show a lower warm p50 and p95.

## How it will be measured

Run `./.venv/Scripts/python.exe work/measure_graph_cache.py --samples 100` from the repository
root. The script will time 100 cache misses after a reset and 100 cache hits for the same tenant,
generation, text mode, graph fingerprint, and policy fingerprint. It will report nearest rank p50
and p95 in milliseconds for each set.

## What I already know

The graph projection cache is intended for immutable generation backed stores. Existing tests cover
generation changes, graph fingerprint changes, text mode separation, and cache boundedness.

## Confounds I can name now

The synthetic store does not include PostgreSQL network latency, operating system scheduling varies
between samples, and the first Python import cost is outside the timed calls.
