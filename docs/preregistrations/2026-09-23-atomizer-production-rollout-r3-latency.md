# Micro atomizer production rollout, round 3: latency only, after the lookup cut

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

Round 2 (`2026-09-23-atomizer-production-rollout-r2.md`) passed on quality with the `dense`
placement (dev and confirm) and failed on latency: atomic stage p95 103.8 ms, p99 165.3 ms over
206 queries. PR #714 removes or merges the two database lookups that followed the selection, and
PR #716 keeps the rescue out of scoped searches. Neither is meant to change a rank or a score. With
both merged, does the atomic stage fit the budget, and are the served results unchanged from round
2?

## Budget, as decided by the operator on 2026-09-23

- **p95 at most 60 ms.** The operator's decision after round 2, replacing the round 1 figure of 40.
- **p99 at most 120 ms.** Not set by the operator. It keeps the ratio of the original budget (p99
  twice p95, 40 and 80) and is my derivation, stated here so it cannot be mistaken for theirs.

Both are judged over all 206 questions together, `dense` arm, as in round 2.

## Frozen apparatus

- **Code:** `origin/master` at the merge commit of PR #716 (which follows #714), merged into
  `claude/atomizer-measure`; the VPS2 clone `~/recall-repos/atomizer-c8-work` fast-forwarded to it.
  The merge commit and its `recall/`, `recall_aml/` and `recall_mcp/` equality with that master
  commit are recorded in the appendix before the run.
- **Unchanged from round 2:** generation `gen_b6aefc110e0d42588f3a57c98aa29129`; artifact
  `~/atomizer-prod/registry`; questions file `49e00f9363e8508b…`; harness
  `scripts/memory_atomizer_production_check.py`, arms `off`, `dense`, `fused`, `k=10`; scope
  `MemoryMax=4G`, `MemorySwapMax=0`, no CPU quota, nice 0.
- **Both splits** are run (dev then confirm), into files with an `-r3` suffix. Their quality was
  decided in round 2 and is not re-decided here; they are reused only as a fixed query set.
- **Rank parity** is computed from the rows: for each question, the `dense` arm's exact rank and
  top five in round 3 against round 2.

## Predictions

My recorded bias is to over-predict benefits and under-predict costs.

1. **Latency, `dense`, 206 questions:** p50 between 20 and 35 ms, p95 between 30 and 55 ms, p99
   below 100 ms.
2. **Parity:** the `dense` exact rank matches round 2 on at least 200 of 206 questions, and the
   top five on at least 195. Mismatches, if any, come from query-time nondeterminism (the query
   embedding call or ties), not from the rescue.
3. **Errors:** 0 in every arm.

## Decision rules

- **Latency passes** with p95 at most 60 ms and p99 at most 120 ms.
- **Parity passes** with the exact rank matching on at least 200 of 206. Every mismatch is listed
  and explained in the result. A mismatch the rescue caused fails the round.
- **If both pass,** M3 (the live rollout of round 1's rules: serving sync, `.env`, live handshake,
  20 live searches carrying the stage, rollback rehearsal) is eligible. **If either fails,** no
  rollout, and the result goes to the operator.

## Result

Not yet run.
