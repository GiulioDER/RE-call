# Pre-registration: does in-process Agent SDK memory beat an external MCP server?

**Date:** 2026-08-27   **Status:** measured, see the Result section appended below

## The question

Holding the tasks, the driver, the instruction and the corpus fixed, does delivering RE-call's
memory through **in-process Agent SDK tools** (`recall_agent`, no server, no per-session cold
start) change agent task success compared to delivering the same memory through the **external
stdio MCP server**?

Answerable by a number: the per-task task-success delta of the in-process run against the
per-task delta of `agent-ab-sdk-replication-001`, judged against the bands committed in
`benchmarks/agent_ab/sdk-inprocess-bands.json`.

## Why this run exists, and what it closes

Everything published so far about "MCP versus the Agent SDK" compares token overhead and
latency. I have not found a comparison of **work completed**, and I have not made one either:
`agent-ab-sdk-replication-001` swapped the DRIVER while deliberately holding the transport
constant, so it cannot speak to this. `recall_agent` shipped with 25 tests and zero outcome data.

This is the missing arm. It is also the arm that decides whether "MCP is obsolete" is a
capability claim or an ergonomics claim.

## What I predict

**Equivalence.** Every falsifier metric lands inside its band, and specifically:

- per-task success delta within **0.10** of the baseline's +0.325, same sign;
- per-pair success delta within **0.10** of +0.333;
- governing-memo reach within **0.08** of 0.605;
- search rate **at or slightly above** 0.905, and I will commit to a direction: I expect the
  in-process arm to search at least as often, because the ~11 s stdio cold start is the one
  mechanism by which a session could plausibly proceed before its memory is ready;
- median input-token overhead within **20,000** of +103,329, because both transports render
  results through the same `serving_json` and the tool descriptions are drift-tested identical;
- admitted pairs at least 52 of 56, and I expect FEWER gate discards than the stdio run, which
  lost 6 to server-availability failures that an in-process server cannot have;
- median wall-time overhead **lower** than the stdio arm's +12.4 s (recorded, not a falsifier).

**Mechanistic prediction, which is the one worth checking:** the transport will not move task
success, because three prior runs put the binding constraint in query FORMULATION, not in
delivery. Roughly four in ten successful searches still miss the governing memo, and nothing
about where the tool executes changes what the agent asks for.

Calibration note: my record says I over-predict effect magnitudes by two to four times. The
prediction above is therefore a null, which is the cheap prediction to make and the one I expect;
the falsifiable content is the DIRECTIONAL calls on search rate, discards and wall time.

## What would falsify this

Any `max_abs_diff` metric outside its committed band. In particular:

- **Superiority** (the claim under test): a per-task or per-pair delta more than 0.15 ABOVE the
  baseline would be evidence that in-process delivery improves agent performance, which is what
  the SDK-replaces-MCP argument asserts and what I predict will not happen.
- **Inferiority**: the same margin below.
- Search rate or reach differing by more than 0.15 either way.
- Token overhead differing by more than 55,000, which would mean the two surfaces are not
  rendering the same thing despite sharing a renderer.
- Falling below 48 admitted pairs VOIDS the run as wiring rather than falsifying equivalence.

Honesty about power, unchanged from the previous run and just as binding: ~50 pairs cannot
resolve a small difference. This is powered for a LARGE effect. A null here means "no large
effect was detectable at this n", never "the two are identical", and the result section must say
so in those words.

## How it will be measured

Identical to `agent-ab-sdk-replication-001` in every field except the on-arm transport: the same
10 committed tasks at their committed reps (56 pairs, 112 sessions), the same
`benchmarks/agent_ab/task-qualification.json`, the same instruction file
`instructions/hazard-query-v2.txt`, the same model `anthropic/claude-haiku-4.5` through
OpenRouter, the same additive arms (`claude_md` off, `claude_md_recall` on), the same
`pair_concurrency=1` and `timeout_s=1800`, and the same corpus.

**Corpus parity is the load-bearing control and it is verified, not assumed.** Both transports
must serve generation `gen_f01fc522293d40c99032cd088500e11d` with calibration
`cal_b40f2c6efa9443a3b5d727f71a92de77`, trusted and calibrated. Confirmed before writing this
record: an in-process `recall_search` against that DSN returned `trust_state=trusted`,
`calibrated=true` and exactly that generation and calibration id, matching what
`StdioRecallSpec.check()` reports. The run's `environment.json` records both.

```bash
python -u scripts/agent_ab_run_tasks.py --run-id agent-ab-inprocess-001 --driver sdk \
  --memory-transport in-process \
  --instruction-file benchmarks/agent_ab/instructions/hazard-query-v2.txt \
  --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default \
  > benchmarks/artifacts/agent_ab/inprocess-001.log 2>&1
python scripts/agent_ab_analyze_tasks.py --run-id agent-ab-inprocess-001
python scripts/agent_ab_compare_drivers.py \
  --run benchmarks/artifacts/agent_ab/agent-ab-inprocess-001/analysis.json \
  --baseline benchmarks/artifacts/agent_ab/agent-ab-sdk-replication-001/analysis.json \
  --bands benchmarks/agent_ab/sdk-inprocess-bands.json \
  --out results/agent_ab/agent_ab_inprocess_2026-08-27.json
```

A `--limit 2 --reps 1` smoke under `agent-ab-inprocess-001-smoke` runs first, is excluded from
the result, and exists to catch wiring.

⚠️ `benchmarks/artifacts/` is gitignored, so the baseline `analysis.json` is not carried by the
repository between worktrees. It was copied into this checkout's artifact directory before the
run, from the worktree that produced it. If a later reader cannot find it there, the committed
`results/agent_ab/agent_ab_sdk_replication_2026-08-26.json` carries the same baseline values that
the bands are written against.

**Apparatus verification, because predicting the outcome does not reveal a broken harness.**
Already done and recorded here rather than claimed later: one real session was driven with the
in-process server before this record was written. Its `system/init` listed
`mcp__recall__recall_search` and `mcp__recall__recall_evidence`, the server reported
`status: connected`, the agent called the tool once, and the trust layer's abstention reached the
answer intact. The admission gate therefore reads an in-process arm exactly as it reads a stdio
arm, which is the property the whole comparison rests on.

## Decision rule, stated now

- All falsifiers inside bands: transport does not move agent performance at this n. "MCP versus
  SDK" is then an ergonomics and safety-surface question, and I will say so in those terms and
  stop presenting it as a capability question.
- Per-task delta more than 0.15 ABOVE baseline: in-process delivery improves task success, the
  claim is supported, and the next question is which mechanism (search rate? latency? something
  in the tool loop?) rather than whether.
- Below baseline by that margin: the in-process surface costs performance, which would be a
  finding against the package I just shipped and must be published as such.
- Admitted pairs below 48: wiring, not evidence. Fix and re-run under a new id.

## What I already know

- `agent-ab-sdk-replication-001` (50 pairs admitted, 6 discarded): search 0.905, reach 0.605,
  per-task +0.325 with cluster CI [0.15, 0.50], per-pair +0.333, controls 1.000/1.000, median
  +103,329 input tokens, +12.4 s wall. Its own record is
  `docs/preregistrations/2026-08-26-sdk-driver-equivalence.md`.
- `agent-ab-skill-001` before it: search 1.000, reach 0.674, per-task +0.208.
- The stated next lever from both runs was **retrieval-side** (query expansion, indexing under
  operation vocabulary), explicitly NOT more plumbing. This run tests the plumbing anyway,
  because the claim is in the air and nobody has measured it, but the prior points at a null.
- `recall_agent` has no scope layer: the audit that shipped it found a secret-ingestion path
  through a tool parameter the MCP server withholds. That is a safety difference between the two
  transports, already fixed, and it is not what this run measures.

## Confounds I can name now

- **Cross-run comparison, again.** The baseline ran earlier the same day; OpenRouter may serve
  different weights or providers by the time this runs. The paired within-run control (the
  `claude_md` off arm) is what absorbs it, which is why the primary comparison is of DELTAS
  rather than of absolute success rates.
- **One shared memory object versus a fresh server per session.** The stdio arm spawns a new
  server for every session; the in-process arm serves every session from ONE
  `RecallAgentMemory`, whose `asyncio.Lock` serialises calls. At `pair_concurrency=1` only one
  on-arm session runs at a time, so no contention is expected, but this is an architectural
  difference beyond the transport and it is named here rather than discovered later.
- **Cold start removed is not nothing.** The stdio arm pays ~11 s per session before its first
  tool call. If that changes the agent's behaviour under a timeout, the effect would be real but
  would belong to latency rather than to in-process execution as such.
- **Embedder residency.** In-process, the fastembed model loads once in the runner and stays
  resident; stdio loads it per session. This changes host memory pressure on a 12 GB machine and
  could in principle affect throughput, though not correctness.
- The six baseline discards were caused by host reboots, not by the transport. If this run
  discards fewer, that is at least partly luck and must not be read as an in-process advantage
  without checking the discard reasons.

## Result (2026-08-27)

**Status:** measured. **Verdict: `not_equivalent`, and it broke on the metric I was most confident
about, for a reason I failed to name in advance.**

Committed summary: `results/agent_ab/agent_ab_inprocess_2026-08-27.json`. Run
`agent-ab-inprocess-001`: 112 records, 54 pairs admitted, 2 discarded, zero checker errors.

| metric | MCP (baseline) | in-process | band | verdict |
|---|---|---|---|---|
| admitted pairs | 50 | 54 | floor 48 | ok |
| search rate | 0.905 | 0.891 | ±0.15 | equivalent |
| governing-memo reach | 0.605 | 0.634 | ±0.15 | equivalent |
| per-task success delta | +0.325 | +0.196 | ±0.15 | equivalent |
| per-pair success delta | +0.333 | +0.196 | ±0.15 | equivalent |
| control tasks (on / off) | 1.000 / 1.000 | 1.000 / 1.000 | ±0.125 | equivalent |
| median input-token overhead | +106,946 → +103,329 | **+32,014** | ±55,000 | **OUTSIDE** |
| median wall-time overhead | +12.4 s | +1.1 s | recorded | recorded |

### The band that broke, and why it is not the finding it looks like

Median input-token overhead fell by 71,315, a 69% reduction, far outside the ±55,000 band. Read
naively that is "in-process delivery costs a third of the tokens MCP does", which is close to the
claim the industry makes about MCP's context cost.

It is not what I measured, and the reason is a confound I did not name in the confounds section:

**The two arms did not expose the same number of tools.** Counted from the sessions' own `init`
events: the MCP arm carried **18** memory tools in every on-arm session, the in-process arm
carried **2** (`recall_search` and `recall_evidence`; write tools are opt-in and were off). Each
tool definition is injected into the session's context, so 16 extra schemas were paid for on
every MCP session before the agent did anything.

So the token difference is overwhelmingly a **tool-surface** effect, not a **transport** effect.
The MCP server could expose two tools; `recall_agent` could expose eighteen. Nothing about where
the tool executes causes this.

That reframes the industry claim rather than confirming it: the context cost attributed to "MCP
versus in-process" is largely the cost of *how many tools are in context*, which is why tool
search and deferred loading close the gap without changing transport at all. Anyone quoting a
token multiple for MCP should check whether they are comparing transports or tool counts.

### What the run does answer

**In-process delivery did not improve agent task success.** Per-task +0.196 against the
baseline's +0.325; per-pair +0.196 (n=46, CI [0.065, 0.326], p=0.012) against +0.333. Both sit
inside the band, so the two transports are equivalent on the endpoint at this n, and the
numerically lower value is in the wrong direction for the claim under test. The superiority band
that would have supported "the SDK makes agents perform better" was not approached.

Mechanism transferred intact: 41 of 46 on-arm sessions searched, reach 0.634 among searchers,
off-arm memory calls zero. Both controls at 1.000/1.000 in both runs.

### Predicted against measured, including the misses

| prediction | measured | gap |
|---|---|---|
| per-task within 0.10 of +0.325, same sign | +0.196, diff 0.129 | **missed the point prediction**, inside the 0.15 band, same sign |
| per-pair within 0.10 of +0.333 | +0.196, diff 0.138 | **missed**, inside band |
| reach within 0.08 of 0.605 | 0.634, diff 0.029 | hit |
| search rate at or ABOVE 0.905 | 0.891 | **wrong in direction**, by 0.013 |
| tokens within 20,000 of +103,329 | +32,014, diff 71,315 | **badly wrong**, and the reason was a confound I did not name |
| admitted pairs at least 52, fewer discards than stdio | 54 admitted, 2 discarded vs 6 | hit, both parts |
| wall time lower than +12.4 s | +1.1 s | hit |

The token miss is the one worth keeping. I predicted a tight match because "both transports
render through the same `serving_json` and the descriptions are drift-tested identical", which
was true and irrelevant: I reasoned about the bytes of a RESULT and forgot the bytes of the TOOL
DEFINITIONS, which are paid once per session whether or not a tool is ever called. The
preregistration's confound list has four entries and none of them is this one.

### Honest limits

- ~50 pairs is powered for a large effect only. "Equivalent on task success" means no large
  effect was detectable at this n, never that the two are identical.
- The token comparison is confounded as described and must not be cited as a transport result.
  A clean version would expose the same tool set on both sides; that is a different run.
- Cross-run confound stands: the baseline ran earlier the same day and OpenRouter may serve
  differently. The paired within-run control absorbs it, which is why the comparison is of deltas.
- The two discards were `ts-raise-on-missing#r6` and `ts-autouse-tmp-path#r6`; fewer than the
  baseline's six, but the baseline's were caused by host reboots rather than by the transport, so
  this is not evidence of in-process reliability.

### Decision

Per the rule stated before the run: all task-success falsifiers landed inside their bands, so
**transport does not move agent performance at this n**, and "MCP versus the Agent SDK" is an
ergonomics and safety-surface question rather than a capability one. I will say it in those terms
and stop presenting it as a capability question.

The token band breach does not reverse that, because it is not a transport measurement. It
generates the next question instead: how much of the context cost attributed to MCP is simply
tool-definition bytes, and does trimming the served tool surface capture it without changing
transport at all. That is preregisterable and cheap, and it is a better question than the one
this run asked.
