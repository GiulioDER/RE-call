# Pre-registration: amortize write-time hook connection setup

**Date:** 2026-08-28  
**Status:** predicted, not yet measured  
**Scope:** latency engineering for the already shipped write-time hook

## Question

Can a persistent local relay, holding one PostgreSQL connection and accepting one JSON request per
payload, reduce the remote write-time hook cost without changing retrieved hits or fail-open safety?

This is not a task-success experiment and cannot establish that the hook's benefit is real. It tests
whether its measured latency is connection setup that can be amortized.

## Frozen comparison

The primary population is 30 qualifying draft payloads sampled from the existing write-time hook
probe. Each payload is sent three times in fixed order to both arms.

* **Cold arm:** the shipped process-per-call path. Each request imports the hook and opens a fresh
  PostgreSQL connection.
* **Relay arm:** one helper process imports `psycopg`, opens one connection, and serves the same
  requests over local newline-delimited JSON.

Both arms use the same DSN, tenant, active generation, SQL, top-k, and payload text. The relay may
reconnect only after a connection failure; reconnects are recorded and are not hidden in the median.

## Validity endpoints

1. Every successful arm pair returns the same ordered `(source, score)` results.
2. The relay never emits a deny decision and never raises into the caller.
3. A deliberately unreachable DSN produces a silent failure and a cooldown or relay backoff; after
   the first failure, the next five calls must complete in at most 0.50 seconds each.
4. No run is interpreted if the active generation, tenant, SQL, or payload sequence differs.

## Predictions

| prediction | expected result | falsified if |
|---|---|---|
| Remote cold median | 1.8 to 2.7 seconds | outside the band |
| Remote relay median after warm-up | at most 0.8 seconds | above 0.8 seconds |
| Result equality | 30 of 30 pairs | any mismatch |
| Relay setup amortization | first request slower than later requests | no warm-up drop |
| Unreachable follow-up calls | at most 0.50 seconds | any of five exceeds it |

## Decision rule

The relay is worth integrating only if all validity endpoints pass and the remote median falls by at
least 50 percent. Otherwise keep the shipped process-per-call path and leave
`write_time.enabled: false` as the safe default for remote or unavailable corpora.

No production integration is licensed by this microbenchmark alone. Any integration requires a
separate review of process ownership, shutdown, stale connections, credentials, and cross-project
tenant isolation.

## Re-measure

After the record is committed, run:

```powershell
python benchmarks/write_time_connection_reuse.py --dsn $env:RECALL_HOOK_DSN --tenant default
```

The command must print the artifact path, arm configuration, result equality, latency samples, and
the unreachable-corpus safety result. It must not print the DSN password.

<!-- frozen_above -->

## Amendment appended after the first attempted run

The first live invocation on 2026-08-28 was invalid. The relay protocol consumed the first request
as connection configuration and did not return a response for that request, so the parent received
one shifted response and then an early relay exit. Its artifact is retained and is not scored. The
relay child was corrected to process the configuration request as the first query before the valid
run below. The cold arm was measured against the locally configured corpus, not the preregistered
remote primary; it is therefore supplementary unless a remote DSN is supplied.

The second attempted run on 2026-08-28 was also invalid. Child processes started with the
benchmark directory as their import root and could not import `recall_hooks.write_time`; all 30
cold rows and the relay row were process failures before any corpus query. The child environment
now pins the repository root on `PYTHONPATH`. These artifacts remain retained and unscored.

## Results appended after measurement

Measured on 2026-08-28 against the configured `recall_repos` corpus through a temporary local SSH
forward to VPS2. The endpoint was remote even though the client DSN named `127.0.0.1`. All 30 cold
requests and all 30 relay requests returned `ok`.

| endpoint | observed | prediction | verdict |
|---|---:|---:|---|
| cold median | 3,706 ms | 1,800 to 2,700 ms | falsified, slower than predicted |
| relay median, all requests | 568 ms | at most 800 ms | confirmed |
| relay median after first request | 491 ms | not above 800 ms | confirmed |
| ordered hit equality | 30/30 | 30/30 | confirmed |
| relay warm-up drop | 3,005 ms first request, then 491 ms median | present | confirmed |
| unreachable follow-up calls | 5/5 at or below 500 ms | 5/5 | confirmed |

The relay reduced the all-request median by 84.7 percent and the steady-state median by 86.7
percent. The result is valid for latency engineering: the connection and process setup cost is
the dominant remote cost, and the relay preserved retrieval results and fail-open behavior.

This licenses a production integration review, not production integration itself. The next record
must cover relay ownership and shutdown, stale connections, credential handling, tenant isolation,
crash recovery, and whether a long-lived helper is acceptable for the supported client platforms.

Artifact: `results/write_time_connection_reuse_20260828T151400Z.json`. Re-measure with the command
under **Re-measure** after opening the documented VPS2 tunnel. The two invalid artifacts remain
retained and are not included in any rate.

## Correction appended 2026-08-28 by review. Nothing above is edited

Three defects in the run recorded above. Every number in the tables is left exactly as it was
written, and this section is appended rather than merged into them, because a record that gets
corrected in place stops being evidence of what was believed at the time.

**The caveat is here because the number is here.** The integration record beside this one had
already found the first defect and written it down there. That is the wrong place: a reader who
opens this file to find the figure sees a table ending in "confirmed" and no reason to look
further.

### 1. The run is not the frozen population, so it is not registered evidence

The artifact reports `payload_count: 10` and `repetitions: 3`, which is 30 requests per arm drawn
from **ten** distinct payloads. The frozen record above specifies 30 payloads sent three times,
which is ninety requests per arm from thirty payloads. So "30 of 30 pairs" counts request pairs,
not the registered payload population, and validity endpoint 4 refuses the interpretation. The
decision rule is therefore not satisfied by this run, whatever the latency shows, and a new
committed preregistration is required before any performance figure is published.

### 2. The reported medians are the upper-middle value of an even sample, not the median

Thirty samples have no middle element, and the two summary rows took the sixteenth rather than the
mean of the fifteenth and sixteenth. Recomputed from the same artifact, with the reported value
beside it:

| quantity | as reported above | conventional median |
|---|---:|---:|
| cold, 30 requests | 3,706 ms | 3,695.1 ms |
| relay, all 30 requests | 568 ms | 529.7 ms |
| relay, after the first request, 29 requests | 491 ms | 491.2 ms |

The third row is odd-sized, so it was already the median and is unchanged. Note the direction of
the error: the convention **understated** the relay, because it picked the upper of the two middle
relay samples while the two middle cold samples were 22 ms apart. The all-request reduction is 85.7
percent on conventional medians against the 84.7 percent recorded above, and the steady-state 86.7
percent is unchanged.

### 3. The artifact stays untracked, for a stronger reason than "parked"

Every row in it carries the `hits` the query returned, which is verbatim text from the private
memory corpus. It cannot be committed to a public repository, so the numbers here can never be made
reachable by tracking the file. Recompute them instead, from an artifact produced by the
**Re-measure** command above:

```bash
python -c "import json,statistics,sys; d=json.load(open(sys.argv[1])); print({k: round(statistics.median([r['elapsed_ms'] for r in d[k]]),1) for k in ('cold','relay')}, d['payload_count'], d['repetitions'])" results/<artifact>.json
```

### What survives all three

The direction and the cause, which is what the integration was argued from: **connection and
process setup, not the query, is the dominant cost against a remote corpus**, and a helper that
outlives one tool call removes it. That claim is supported by a first request of 3,005 ms against a
steady state under 500 ms in the same arm, which is a within-arm comparison and does not depend on
the payload count. The magnitude against the cold arm does depend on it, and is not claimed here.
