# Pre-registration: is MCP's context cost just tool-definition bytes, and are ours wasted?

**Date:** 2026-08-27   **Status:** measured, see the Result section appended below

## The question

Two numbers, both answerable directly:

1. **What does one served tool cost, per turn, in input tokens?** Measured on a one-turn session
   that calls nothing, varying only the number of tools served.
2. **Does that per-turn cost, multiplied by a session's turns, reproduce the 71,315-token gap**
   observed between `agent-ab-sdk-replication-001` (18 tools served) and
   `agent-ab-inprocess-001` (2 tools served)?

If both hold, the "MCP costs several times the context" claim is a statement about tool COUNT,
not about transport, and it is a cost this project imposes on its own users by default.

## Why this run exists

`agent-ab-inprocess-001` broke its token band and I traced the cause to a confound: the MCP arm
served 18 memory tools per session and the in-process arm served 2. That explains the gap but
does not measure it, and an explanation nobody measured is a hypothesis.

The second half is not academic. Across **112 on-arm sessions in the two runs, the agents called
exactly one RE-call tool: `recall_search`, 139 times.** The other 17 were never invoked once. If
a tool definition costs real tokens on every turn, this project is charging every Claude Code
session for 17 tools it does not use, and that is our configuration decision, not MCP's.

## What I predict

**A per-tool, per-turn cost of roughly 250 to 400 input tokens**, and the relationship linear in
tool count over this range.

Concretely, on a ONE-turn session that calls nothing, relative to serving no memory tools:

- 2 tools: **+500 to +800** input tokens
- 4 tools: **+1,000 to +1,600**
- 18 tools: **+4,500 to +7,200**

**The reconciliation, which is the real prediction:** 16 extra tools at that per-turn rate, over
the ~15 median turns those sessions ran, predicts **60,000 to 96,000** extra input tokens per
session. The observed gap was **71,315**. If the measured slope lands such that
`slope × 16 × 15` brackets 71,315, the confound explanation is confirmed by an independent
measurement rather than asserted from a tool count.

**Internal control on transport.** The 2-tool and 4-tool arms use the SAME in-process transport,
so the slope between them is transport-free. If that within-transport slope agrees with the
slope from 2 tools to 18 (which crosses in-process to stdio), transport contributes nothing and
the effect is definitional bytes. If they disagree, the transport is doing something and I must
say so rather than blaming tool count.

Calibration note: I have over-predicted magnitudes two to four times repeatedly, so the ranges
above are deliberately wide and centred on a schema size I can sanity-check by reading the
docstrings, not on a hope.

## What would falsify this

- **A per-tool cost below ~100 or above ~800 tokens per turn.** Either would mean I have
  misidentified what is being injected.
- **A non-linear or flat relationship**: if 18 tools cost about the same as 2, tool definitions
  are not the mechanism and the 71,315 gap needs a different explanation entirely.
- **Reconciliation failure**: if `slope × 16 × turns` does not bracket 71,315, then tool count
  explains only part of the gap and I must find the rest before claiming the confound.
- **Transport disagreement**: if the within-transport slope (2 to 4, in-process) differs
  materially from the cross-transport slope, the confound story is incomplete.

## How it will be measured

One-turn sessions, identical prompt ("Reply with the single word READY and nothing else."),
`max_turns` unset but the prompt calls nothing, same model `anthropic/claude-haiku-4.5` through
OpenRouter, same SDK driver, `--bare`, no task sandbox. **5 repetitions per arm**, four arms:

| arm | tools served | transport |
|---|---|---|
| `none` | 0 | no memory server |
| `read` | 2 | in-process (`recall_agent`, write tools off) |
| `read+write` | 4 | in-process (`write_tools=True`) |
| `full` | 18 | external stdio MCP server |

Metric: **`input_tokens` as the harness already computes it** (fresh + cache-creation +
cache-read summed from the per-model aggregate, `claude_exec._usage_fields`), reported as the
median per arm, with the per-arm turn count recorded so a multi-turn session cannot be mistaken
for a one-turn one. n = 5 per arm, 20 sessions total.

Driver: `benchmarks/agent_ab/sdk_exec.py` directly, via a small script committed with this
record, so no task machinery, no sandbox and no checker is involved.

## What I already know

- `agent-ab-sdk-replication-001`: 18 tools, median 15 turns, median on-arm input 299,359;
  on-minus-off median +103,329.
- `agent-ab-inprocess-001`: 2 tools, median 14 turns, median on-arm input 270,899;
  on-minus-off median +32,014.
- Difference of differences: **71,315 tokens for 16 tools**, which over ~15 turns is ~297 tokens
  per tool per turn. That arithmetic is where the prediction above comes from, so this run is a
  genuine out-of-sample check of a number derived from other data, not a re-measurement.
- The 18 tools are listed in `recall_mcp/server.py`; only `recall_search` was ever called.

## Confounds I can name now

- **Transport is not held constant across all four arms.** The 18-tool arm is stdio and the
  others are in-process. This is why the 2-to-4 within-transport slope exists as an internal
  control, and why a disagreement between the two slopes falsifies rather than being explained
  away.
- **Prompt caching.** Input tokens here sum fresh, cache-read and cache-creation, so a cached
  tool block still counts. That is the right denominator for "what does the context cost", but
  it is NOT the same as what the provider bills, and the record must not conflate them.
- **A one-turn session may not inject definitions the same way a long one does.** The
  reconciliation step is what tests that: if per-turn cost were only paid once, `slope × 16 × 15`
  would badly overshoot 71,315.
- **Claude Code version and model may change tool serialisation.** Both are recorded.
- Five repetitions per arm is small. It is adequate only because the quantity is nearly
  deterministic (a fixed prompt and a fixed tool list), and the run reports the spread so a
  reader can see whether that assumption held.

## Result (2026-08-27)

**Status:** measured. **Question 1 answered; question 2 FALSIFIED, and it corrects a claim I
published earlier today.**

Artifact: `benchmarks/artifacts/agent_ab/tool-cost.json`. 20 sessions, 5 per arm, every one a
single turn. **Variance was exactly zero**: all five reps in every arm returned an identical
input-token count, which is the assumption n=5 rested on, now observed rather than hoped.

| arm | tools served | input tokens (1 turn) | vs none | per tool |
|---|---|---|---|---|
| none | 0 | 2,795 | — | — |
| read | 2 | 3,324 | +529 | 264.5 |
| read+write | 4 | 3,582 | +787 | 196.8 |
| full | 18 | 5,727 | +2,932 | 162.9 |

### Question 1: what a tool costs

**Roughly 130 to 155 input tokens per tool per turn, plus a fixed ~270 for serving any tools at
all.** The marginals are the honest numbers, not the averages above:

- 2 → 4 tools (both in-process, **transport held constant**): +258, so **129 per tool**.
- 4 → 18 tools (crosses in-process to stdio): +2,145, so **153 per tool**.

The internal control did its job. Those two slopes agree to within 24 tokens per tool, so
**transport contributes essentially nothing and the cost is definitional bytes**, exactly as the
confound story required. The residual difference is explainable by the tools themselves: the two
write tools have shorter descriptions than the reasoning and calibration tools.

Serving 18 tools instead of 2 costs **2,403 input tokens on every turn**.

### Question 2: the reconciliation, which failed

Predicted: `slope × 16 tools × ~15 turns` would bracket the 71,315-token gap, with a stated band
of 60,000 to 96,000.

Measured: 2,403 per turn × 15 turns = **36,045**. Against 14 turns, 33,642. Both fall well below
the predicted band, so **tool definitions explain only about half of the gap I attributed to
them.**

### Where the other half went, and the correction it forces

I went looking for the remainder and found it in my own arithmetic rather than in the sessions.

The 71,315 figure was a *difference of differences*: (MCP on − MCP off) − (in-process on −
in-process off). Its two control arms are supposed to be the same configuration, no memory tools
at all, so their difference should be near zero. It was not:

| | MCP run | in-process run | difference |
|---|---|---|---|
| ON arm median input | 299,359 | 270,899 | **28,460** |
| OFF arm median input | 180,022 | 222,053 | **−42,031** |

The control arms differ by 42,031 tokens in the *opposite* direction, which is pure run-to-run
variation between two runs executed hours apart. That inflated the difference-of-differences to
roughly twice the real effect.

**Comparing the ON arms directly, which is what the tool count actually changed: 28,460 tokens,
against 33,642 to 36,045 predicted from the per-turn measurement.** Those agree within about 20%,
which is the reconciliation the preregistration asked for, arrived at only after the first
attempt failed.

⚠️ **This corrects `docs/preregistrations/2026-08-27-in-process-vs-mcp-transport.md`.** Its result
section says the 71,315-token gap is explained by the tool-surface confound. The direction is
right and the magnitude is overstated roughly twofold: tool definitions account for about 36,000
of it, and the rest is control-arm noise. The number there is not edited, per the rule; a pointer
to this record is appended beneath it.

### Predicted against measured

| prediction | measured | gap |
|---|---|---|
| 250–400 tokens per tool per turn | 129–153 marginal | **missed, ~2× too high** |
| 2 tools: +500 to +800 | +529 | hit |
| 4 tools: +1,000 to +1,600 | +787 | **missed, below** |
| 18 tools: +4,500 to +7,200 | +2,932 | **missed, below** |
| linear in tool count | linear with a fixed intercept | close enough to stand |
| within-transport slope agrees with cross-transport | 129 vs 153 | hit, and it is the load-bearing one |
| reconciliation brackets 71,315 | 33,642–36,045 | **falsified** |

Five of seven predictions missed, every one in the same direction: too high. My calibration memo
says I over-predict magnitudes by two to four times, and this run came in at almost exactly two.
The one prediction that mattered structurally, that the two slopes would agree and therefore
implicate definitions rather than transport, held.

### What this means for the project, which is the part with a cost attached

Across the 112 on-arm sessions in the two completed runs, **the agents called exactly one RE-call
tool: `recall_search`.** The other 17 were never invoked. At 153 tokens per tool per turn over a
15-turn session, those 17 unused definitions cost roughly **39,000 input tokens per session**, on
every session, forever, for capability nobody in this workload used.

That is our configuration, not a property of MCP. The server exposes all 18 tools unconditionally;
scopes gate execution, not listing. A deployment that serves `recall_search` and `recall_evidence`
alone would pay 3,324 tokens of context instead of 5,727 per turn and lose nothing this workload
touched.

It also reframes the public claim honestly: the "MCP costs multiples of the context" figures are
measuring **how many tools are in context**, which is why tool search and deferred loading close
the gap without changing transport. On this corpus the transport itself was worth about 24 tokens
per tool, which is to say nothing.

### Limits

- One model, one CLI version, one prompt. Tool serialisation is a client implementation detail and
  these constants will move.
- Input tokens here sum fresh, cache-creation and cache-read, so a cached tool block still counts.
  That is the right denominator for context pressure and it is NOT what a provider bills.
- The 15-turn multiplier is a median from other runs, not measured here; a session's real cost
  scales with its own turn count.
- Zero variance means the quantity is deterministic for a fixed prompt and tool list. It does not
  mean the estimate generalises to prompts that fill the context.
