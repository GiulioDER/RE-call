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

## Apparatus appendix, appended 2026-09-23 before any evaluation

Apparatus only; nothing had been evaluated when it was written.

- **Master under test:** `a42f035c` (PR #716 merge), which follows `ef58ebf1` (PR #714).
- **Measurement branch commit:** merge `1e936344` on `claude/atomizer-measure`. Its `recall/`,
  `recall_aml/`, `recall_mcp/`, `pyproject.toml`, `uv.lock` and `docs/PRODUCTION.md` are
  byte-identical to `a42f035c` (checked with `git diff --cached --quiet a42f035c -- <path>` before
  the merge commit); every other difference is a study harness or a preregistration.

## Result, appended 2026-09-23 after the run

The "Not yet run." line above is left as written. Reports `~/atomizer-prod/report-{dev,confirm}-r3.json`
and rows `~/atomizer-prod/rows-{dev,confirm}-r3.jsonl` (VPS2, private). 206 questions, generation
unchanged in both splits, 0 errors in every arm. VPS2 load average about 4 to 5 during the run
(about 2.6 to 3.0 in round 2), with the official AML run still going.

| arm | atomic p50 | atomic p95 | atomic p99 | max | total p95 |
|---|---:|---:|---:|---:|---:|
| `dense` | 25.5 | **74.4** | 118.1 | 139.2 | 785.9 |
| `fused` | 25.2 | 52.9 | 88.4 | 102.7 | 741.6 |
| `off` | n/a | n/a | n/a | n/a | 718.4 |

All times in ms, pooled over 206 questions. `dense` queries over 60 ms: 18 of 206 (11 dev, 7
confirm), spread through both runs rather than clustered; the first question of each split is a
cold start (end to end 3,302 and 2,823 ms against 974 and 1,027 for `off`).

**Parity against round 2:** `dense` exact rank, top five and abstention match on **206 of 206**.
`off` differs in its top five on 1 question, from query-time nondeterminism. Paired quality is
therefore unchanged: dev exact@6 +4/−0, confirm +4/−0.

**Scoring.**

1. Latency (p50 20 to 35, p95 30 to 55, p99 below 100): p50 **confirmed** (25.5); p95
   **falsified** (74.4); p99 **falsified** (118.1).
2. Parity (exact rank on at least 200, top five on at least 195): **confirmed**, 206 and 206.
3. Errors 0: **confirmed**.

**Decision under the frozen rules.** Parity passes. Latency **fails**: p95 74.4 against 60. (p99
118.1 is inside my derived 120.) So **no rollout**, and the result goes to the operator.

The lookup cut did what it was meant to do: the median fell from 40.4 to 25.5 ms with no rank
change. The tail it did not reach is the selection itself under host contention.

## Operator decision, appended 2026-09-23 after the result

After reading the result above, the operator raised the budget to **p95 at most 80 ms** and
authorized the memory-tenant rollout. This is a second, post hoc change of the budget, recorded
here as the operator's decision and not as a pass of the rule frozen above, which the round
**failed** at 60. Against 80, the measured `dense` p95 of 74.4 ms and p99 of 118.1 ms are inside
(p99 against my derived 160, twice p95, and also inside the 120 frozen above).

The same day the operator asked that nothing on VPS2 be touched while the official benchmark runs,
and chose to **defer the rollout until that run is over**. So M3 has not started, and nothing on
the VPS2 memory route has changed.
