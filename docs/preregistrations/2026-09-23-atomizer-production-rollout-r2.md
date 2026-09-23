# Micro atomizer production rollout, round 2: M2 dev rerun after the score and thread fix

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

Round 1 (`2026-09-23-atomizer-production-rollout.md`) failed M2 dev: no placement was eligible
and the atomic stage breached the latency budget. The post hoc diagnosis named two defects, both
fixed in PR #712: the rescued parent carried its view cosine into the trust layer, and the view
product ran twelve BLAS threads under a CPU quota. Does the fixed code now pass M2 dev under the
round 1 decision rules, and does the stage fit the budget under the live server's own limits?

## Frozen apparatus

- **Code:** branch `claude/atomizer-measure` at the commit that adds this file. Its `recall/`,
  `recall_aml/`, `recall_mcp/`, `pyproject.toml` and `uv.lock` are byte-identical to PR #712's head
  `5b1ca726` (checked with `git diff --cached --quiet 5b1ca726 -- <path>` before the merge commit
  `e2abec58`). The VPS2 clone `~/recall-repos/atomizer-c8-work` is fast-forwarded to it.
- **Unchanged from round 1:** generation `gen_b6aefc110e0d42588f3a57c98aa29129`; the artifact in
  `~/atomizer-prod/registry` (built by the production builder, not rebuilt); questions file SHA-256
  `49e00f9363e8508b…`, the same 103 dev questions; harness
  `scripts/memory_atomizer_production_check.py`, arms `off`, `dense`, `fused`, `k=10`.
- **Changed from round 1, deliberately:** the run has no `CPUQuota` and runs at nice 0, because the
  live memory MCP servers on VPS2 run at nice 0 in a user session scope with no CPU quota (checked
  2026-09-23 with `ps -o ni` on the running `recall_mcp.server` processes). `MemoryMax=4G` and
  `MemorySwapMax=0` stay. Output files carry an `-r2` suffix; round 1's files are not touched.
- **Abstention flips** are counted from the rows: queries where `off` abstained and the arm
  answered, and the reverse.

## The dev split is no longer blind

I read the round 1 dev rows to diagnose the defects, and the fix was designed from them. So a pass
on dev is weaker evidence than it was in round 1. The untouched confirm split carries the decision,
exactly as the round 1 rules already require.

## Predictions

My recorded bias is to over-predict benefits two to four times and under-predict costs.

1. **`off` is unchanged** from round 1 up to query-time nondeterminism: exact@6 83 ± 2, abstained
   30 ± 2.
2. **Abstention flips** (off abstains, arm answers): at most 2 per arm. None the other way.
3. **`fused`:** top five changed against `off` on at most 6 queries; exact@1 losses at most 1; net
   exact@6 between 0 and +4.
4. **`dense`:** net exact@6 between +2 and +6; exact@1 losses at most 3.
5. **Latency:** atomic stage p95 between 20 and 45 ms, p99 below 80 ms, in both arms.
6. **Errors:** 0 in every arm.

## Decision rules (round 1's, restated, unchanged)

- **Placement (dev).** `fused` is eligible with net exact@6 at least +2, at most 2 losses, top five
  unchanged on every query and no errors. `dense` is eligible with net exact@6 at least +2, at most
  2 losses, at most 1 exact@1 loss and no errors. Choose `fused` if eligible, else `dense` if
  eligible, else no rollout.
- **Confirm, once, chosen placement only against `off`,** on the 103 confirm questions, same
  settings as this dev run. Passes with net exact@6 at least +1, at most 2 losses, no errors (and
  top five unchanged, for `fused`).
- **Latency.** Atomic stage p95 at most 40 ms and p99 at most 80 ms over dev and confirm together.
  A breach is reported to the operator before any rollout.
- **Rollout (M3)** only after dev and confirm pass, with the live gates of round 1.

One note on the `fused` rule, written before measuring: with the parent scored as a chunk, the
served top five can still differ from `off` where the rescued parent clears the threshold on its
own chunk cosine and a top-five hit does not, because `evaluate` orders `ok` ahead of the rest.
The rule is kept as written anyway. It is strict, and it was frozen in round 1; relaxing it after
seeing a round would be exactly what preregistration exists to stop.

## Result

Not yet run.
