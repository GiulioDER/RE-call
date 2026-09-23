# Micro atomizer on the memory tenant: master parity before moving the serving checkout

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

The memory tenant serves the micro atomizer from `a42f035c`, the commit round 3 measured
(`2026-09-23-atomizer-production-rollout-r3-latency.md`). Master is now `986b70c9`, 14 commits
later, including #711 and #713, which restructured the trusted-search path and split the rescue
into `select_atomic_rescue` plus `place_atomic_selection_dense` and `place_atomic_selection_fused`.
None of those changes was meant to move a rank or a score. Does master serve the memory tenant
the same results as `a42f035c`, within the operator's 80 ms p95 budget?

## Frozen apparatus

- **Code:** `origin/master` at `986b70c9`, in a detached worktree of `serving-master` on VPS2. No
  migration and no dependency differs from `a42f035c` (checked with `git diff a42f035c 986b70c9 --
  recall/migrations pyproject.toml`, both empty).
- **Unchanged from round 3:** generation `gen_b6aefc110e0d42588f3a57c98aa29129` (r204, now retired
  but not collected; 11,735 chunks); artifact `~/atomizer-prod/registry`; questions file
  `49e00f9363e8508b…`, all 206 questions, dev then confirm; harness
  `scripts/memory_atomizer_production_check.py` as merged to master in #726 (its helpers are the
  verbatim copies in `scripts/atomizer_study_common.py`); arms `off`, `dense`, `fused`, `k=10`;
  scope `MemoryMax=4G`, `MemorySwapMax=0`, no CPU quota, nice 0. Output files carry an `-r4`
  suffix.
- **Baseline:** round 3's rows, `rows-{dev,confirm}-r3.jsonl`.
- **Parity** is computed per question for the `dense` arm (the placement in production): exact
  rank, top five and abstention, round 4 against round 3.

## Predictions

My recorded bias is to over-predict benefits and under-predict costs.

1. **Parity:** `dense` exact rank matches round 3 on at least 200 of 206, top five on at least
   195, abstention on at least 203. Round 3 matched round 2 on 206 of 206, so I expect the same or
   within one or two questions of it; any mismatch comes from query-time nondeterminism, not code.
2. **Latency:** `dense` atomic stage p95 between 30 and 90 ms, p99 below 160, pooled over 206.
   Host load decides most of this, and it is not controlled.
3. **Errors:** 0 in every arm.

## Decision rules

- **Parity passes** with exact rank matching on at least 200 of 206, top five on at least 195 and
  abstention on at least 203. Every mismatch is listed; a mismatch caused by the code change fails
  the round.
- **Latency passes** with p95 at most 80 ms (the operator's budget) and p99 at most 160 ms (twice
  p95, my derivation, as in round 3's appendix).
- **If both pass,** moving the serving checkout to master is eligible, and is reported to the
  operator before it is done. **If parity fails,** serving stays on `a42f035c`.

## Result

Not yet run.
