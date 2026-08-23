# Pre-registration: the official Bench'd runs (LongMemEval 500, LoCoMo full)

**Date:** 2026-08-23   **Status:** predicted, not yet measured

## The question

What does the champion configuration score on the two full official benchmarks, run once,
through `benchmarks/benchd/run_official.sh`, for submission to the benchd.ai leaderboard?

## What I predict

- **LongMemEval 500**: nuance overall **68 to 75**. The 75.0 tuning figure came from a
  60-question slice the config was tuned on; the full set is 88% unseen by the tuning, so some
  regression toward the mean is expected. Anything above the current leaderboard top (59.0)
  keeps the leaderboard claim; the prediction is the honest range, not the threshold.
- **LoCoMo full (~1,540, categories 1 to 4)**: nuance overall **82 to 90**. The n=49 dress
  measured 89.8 on a proportional stratified sample of the same population, so the full run
  should land near it with a tighter interval.
- Efficiency: tokens per correct answer **under 60** on both (dress runs: 46.9 and 36.9).
- Abstentions: **0** on both (threshold 0.0, suppression on).
- Spend: **under $2 (LongMemEval) and under $5 (LoCoMo)** on the OpenRouter meter, measured by
  the script's before/after snapshots; Voyage spend measured from its dashboard.
- Wall time at workers=4 (measured 2.4x, score stable 12/13/12 across workers 1/4/8 on the
  same slice): **under 1 hour (LongMemEval), under 3 hours (LoCoMo)**.

## What would falsify this

A LongMemEval score at or below 59.0 or a LoCoMo score at or below 54.8 falsifies
"leaderboard-leading". A nonzero abstention count, a signature that fails `benchd verify`, or
a run-record hash mismatch voids the run regardless of score: the run is then debugged and
rerun, and both runs are reported.

## How it will be measured

One invocation per benchmark, sequentially (never concurrently: they share the Voyage key):

```bash
bash benchmarks/benchd/run_official.sh <harness-dir> longmemeval-v1
bash benchmarks/benchd/run_official.sh <harness-dir> locomo-v1
```

Pins recorded per run in `run-record.json`: RE-call SHA, harness fork SHA (`5fa33db5`,
upstream `bd4824fe` plus the `--workers` patch and adapter registration), adapter SHA256
(`eeb7676a6200645a...`), dataset SHA256, signing key fingerprint `92dae5232b5c8af6`, full
`RECALL_BENCHD_*` config, meter snapshots. Database: this session's dedicated container,
schema wiped to empty immediately before the first run and not touched between the two
(different tables and per-run tenant namespaces isolate them). Metric:
`scores.nuance.overall` per benchmark from the signed manifest.

## What I already know

Tuning arms, abstention sweep, and LoCoMo dress: `2026-08-23-benchd-tuning-arms.md`,
`2026-08-23-benchd-abstention-threshold.md`, `2026-08-23-benchd-locomo-dress.md`. Current
leaderboard: LongMemEval 59.0 (LlamaIndex, LangChain), LoCoMo n=49 cell 54.8 (LlamaIndex).

## Confounds I can name now

- **Tuning leakage on LongMemEval** (12% of the test set steered config choices): named in the
  tuning prereg, unavoidable without a dev split, shared by every tuned system on the board.
  The LoCoMo run is the cleaner headline for exactly this reason.
- **Provider drift mid-run**: OpenRouter routing and voyage-4 embeddings are not
  bit-deterministic; at n=500 and n=1540 this is noise, not bias.
- **The n=49 leaderboard cell vs our full LoCoMo run**: their published 54.8 is a sample of
  the population we run in full; the comparison is population-vs-sample, stated as such. Our
  full-run manifest allows recomputing any subsample.
- **Duplicate conversation ingests under workers**: each worker adapter builds its own tenant
  for a conversation it encounters, so up to workers x conversations ingests on LoCoMo; costs
  cents and changes no result (tenants are namespaced per adapter instance).

## Amendment (2026-08-23, before the valid runs): first LongMemEval attempt VOID

The first official LongMemEval invocation (run_e00dcde48e15, 16:30Z) is void under the
registered void condition: 402 of 500 items returned adapter errors ("sorry, too many clients
already") because the tenant cache held one Postgres connection per conversation and the full
run has ~500 conversations against a ~100 connection server limit. The tuning slices (n<=60)
could not have surfaced this. Reported, not hidden: the manifest is kept, its score (14.4)
measures the crash, not the system.

Fix: an LRU cap on open stores (RECALL_BENCHD_STORE_CAP=12), rows persist across eviction.
Proven at 2x the limit before rerun: 200-item hashing run, 197 tenants, zero adapter errors.

New pins for the valid runs, replacing the ones named above (the prediction ranges and
falsification criteria are unchanged): RE-call `1cb2eb2` (LRU commit), harness fork
`ade275a2`, adapter SHA256 `f83a2e22...`.

Correction, same day: the amendment above names RE-call `1cb2eb2`; that SHA does not exist.
The LRU commit is `4500e7af`. Left wrong above per the no-edit rule; the run-record.json of
each valid run carries the authoritative SHA.

## Result: LongMemEval 500 (2026-08-23)
**Status:** measured (LoCoMo pending)

Measured: **67.2** (336/500), run_fe4489535939, signature VERIFIED, fingerprint
92dae5232b5c8af6, 0 adapter errors, 0 abstentions, 12/500 synthesis fallbacks. Per dimension:
recall 133/156, reasoning 92/133, temporal 111/211. Efficiency: 96.8 tokens per correct, mean
recall 65.1 tokens, mean latency 6.7s. Spend, metered: $1.07 OpenRouter. Wall: 21 minutes at
workers=4. Estimated BMI 76.7 against the leaderboard leader's 71.6.
Predicted: 68 to 75.
**Gap: 0.8 below the band's floor (4 questions).** The tuning-slice regression came in at the
high end of expectation: 75.0 on the tuned 60 became 67.2 on the full 500. Against the
leaderboard: 8.2 points above the current top (59.0). The falsification threshold (at or below
59.0) was not approached.

## Result: LoCoMo full (2026-08-23)
**Status:** measured; both official runs complete

Measured: **77.3** (1190/1540), run_49bfa2142bd1, signature VERIFIED, fingerprint
92dae5232b5c8af6, 0 adapter errors, 0 abstentions, 5/1540 synthesis fallbacks. Per dimension:
reasoning 801/937 (85%), recall 189/282 (67%), temporal 200/321 (62%). Efficiency: 52.4
tokens per correct, mean latency 3.9s. Spend, metered: $4.01 OpenRouter. Wall: 44 minutes at
workers=4. Estimated BMI 83.9 against LlamaIndex's 68.3 on their n=49 cell.
Predicted: 82 to 90.
**Gap: 4.7 below the band's floor.** The prediction leaned on the n=49 dress (89.8), and the
registered confound ("per-dimension counts are tiny") is the explanation: the dress's temporal
went 10/10 on a sample whose full-population rate is 62%. Against the leaderboard: 22.5 points
above their published cell (54.8), and the dress manifest for the exact n=49 cell is retained
alongside. The falsification threshold (at or below 54.8) was not approached.

Cumulative OpenRouter spend for the entire campaign, tuning through both officials including
the void run: $6.64 by meter (558.217 to 564.857).
