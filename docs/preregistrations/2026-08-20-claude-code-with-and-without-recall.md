# Claude Code with and without RE-call, on trap tasks

**Written 2026-08-20, before any paired session was run.** The wiring gate and the trap
qualification below were measured first and are reported as measurements, not predictions. Every
number under "Predictions" is a prediction and is not to be edited afterwards; the result is
appended beneath it.

Run identifier: `agent-ab-traps-001`.

## Question

Does an agent doing ordinary work in this repository make fewer known, expensive mistakes when it
has a retrieval memory layer than when it has the hand-written `CLAUDE.md`, and what does that cost
in tokens and wall time?

## What is already measured, and is not a prediction

| Fact | Value | How |
|---|---|---|
| `claude -p` on the subscription token | 401, OAuth revoked | ran it |
| Route used instead | OpenRouter Anthropic skin, `ANTHROPIC_BASE_URL=https://openrouter.ai/api` | HTTP 200, full session telemetry |
| Cold stdio RE-call startup | 13.3 s | `initialize` over stdio |
| Warm streamable-http RE-call | 458 ms handshake, 461 ms search | `scripts/agent_ab_gate.py` |
| CLI reported `total_cost_usd` | about 6x the list price of the tokens it billed for | 1,920 in / 45 out on Haiku read $0.0137 |
| Static memory bundle | 17,499 chars (`CLAUDE.md` + `MEMORY.md`) | `scripts/agent_ab_qualify.py` |
| Memory store | 636,191 chars across 185 files | same |
| Trap loci | 4 `memory_only`, 4 `both`, 2 `claude_md_only` | `benchmarks/agent_ab/trap-qualification.json` |

The corpus is **uncalibrated**: every search returns `trust_state: degraded`,
`failure_code: INDEX_NOT_READY`, under an uncertified 0.5 cosine floor. This result is therefore a
statement about retrieval with abstention untuned, and must say so wherever it is published.

## Design

Paired, one task at a time, both arms started together. Three arms, run as two preregistered
comparisons so the two-variant record schema is unchanged:

| Comparison | on arm | off arm | `metadata.off_arm_profile` |
|---|---|---|---|
| headline | `recall` | `claude_md` | `claude_md` |
| ceiling | `recall` | `bare` | `bare` |

Both arms of a pair get the same model, prompt, task input, working directory policy, tool
allowlist and resource limits from one call to `build_configs`. `--bare` is set in every arm, so
context arrives only through recorded flags. `docker` is denied in every arm: the trap tasks about
shared containers are scored from the denial, so the hazard is measured and never realised.

Task set: `benchmarks/agent_ab/tasks/traps.jsonl`, 10 tasks, 5 repetitions each.
Model: one model for the agent, a different family as judge, both recorded in the artifact.

**Admission.** A pair is discarded, not scored, unless the on arm's `system/init` lists a
`mcp__recall*` tool and the off arm's does not. This is not hypothetical: a session on 2026-08-20
ran with no RE-call tool and reported `success` with no error, which averages in as evidence that
RE-call does nothing. Discarded pairs are reported by task id. A tool that was **available and not
called** is admitted and counted: that is a behavioural result, not a wiring fault.

## Predictions

Written before measuring, and deliberately low. The record on this project is eleven of twelve
predictions falsified, every one too high by a factor of two to four, so each of these is set at
roughly a quarter to a half of the arithmetic ceiling, and a mechanism metric is predicted beside
each outcome.

**Primary endpoint: trap hit rate on the 4 `memory_only` traps, headline comparison.**

1. Off arm (`claude_md`) hit rate: **0.70**.
2. On arm (`recall`) hit rate: **0.45**.
3. Absolute reduction: **25 percentage points**. Falsified if the reduction is below 10 points, or
   if the 95% interval on the paired difference includes zero.

**Mechanism, predicted separately, because an outcome without one is a coincidence.**

4. `recall_search` is called in **at least 70%** of on-arm sessions on `memory_only` tasks.
5. The governing memo appears in the retrieved contexts in **at least 50%** of those sessions.
6. If 4 and 5 hold and prediction 3 fails, the failure is in the agent's *use* of retrieved
   evidence, not in retrieval. If 4 or 5 fails, it is discoverability, and the one-sentence prompt
   in `arms.RECALL_SYSTEM_PROMPT` is the thing to vary next.

**Controls, which are the reason the trap set contains traps RE-call should not win.**

7. `both` traps: absolute difference within **10 points** either way.
8. `claude_md_only` traps: the on arm is **worse by 10 to 30 points**. A result where the retrieval
   arm also wins these is evidence the arms are not isolated, not evidence of a stronger effect.

**Cost of the memory layer.**

9. On-arm input tokens **+20%** relative to `claude_md`, driven mostly by 16 MCP tool definitions
   entering the context whether or not a search happens.
10. On-arm wall time **+3 s** per session, median.
11. `recall_latency_ms` accounts for **under half** of that difference; the rest is the extra turn.

**Ceiling comparison.** Against `bare`, the `memory_only` reduction is **35 points**, and the
`both` traps also move, by **15 points**. If the `bare` comparison does not beat the `claude_md`
comparison on the `both` traps, the static bundle is not doing what it is assumed to do.

## Exclusions and stopping rules

A task is excluded from a metric when either arm is missing or failed, and excluded task ids are
recorded with the reason. Missing measurements stay null and are never replaced with zero. No task
removal, prompt change, model change, trap change or metric change after inspecting results; any
exploratory rerun takes a new run identifier and a new preregistration. The trap set and its loci
are frozen by `benchmarks/agent_ab/trap-qualification.json`, committed before this run, and the
suite refuses any disagreement between it and the task manifest.

## Analysis

Paired throughout. Wilcoxon signed-rank on tokens and wall time, McNemar on the binary trap and
success outcomes, bootstrap intervals on the paired differences. Quality is reported first, then
tokens and time, then retrieval overhead, then discarded pairs.

`total_cost_usd` from the CLI is **not** reported: it was measured wrong by about 6x through the
gateway. Spend is reconciled from the gateway's own accounting, and token counts are primary.

## Artifacts

Raw `stream-json` per session, gzipped, one file per task, arm and repetition; the `SessionRecord`
JSONL; the paired summary and the revision that produced it; the admission report including every
discarded pair; the trap scores; the judge model, prompts and raw verdicts; environment capture
including CLI version, model slug, corpus and generation identifiers, and the `system/init`
snapshot proving the tool was present.

**Publication gate.** The on arm retrieves from a corpus containing host inventory and pointers to
credential files, and retrieved chunks land verbatim in transcripts. No transcript is published
until it has been reviewed, and the preferred route is a rerun against a filtered corpus, which is
clean by construction rather than redacted after the fact.

## Result

*Not yet run. Appended below when it is, without editing anything above.*

---

## Superseded 2026-08-21, before any paired session was run

**Nothing above has been edited.** This record is superseded by
`2026-08-21-claude-code-with-and-without-recall-calibrated.md`, and it is kept because it is
evidence of what was predicted, and under which configuration, before the corpus changed.

The reason is stated in the section above: *"The corpus is **uncalibrated**... This result is
therefore a statement about retrieval with abstention untuned."* That is no longer the
configuration. The corpus was calibrated on 2026-08-21, and calibration is not a neutral
improvement to the same experiment. It changes the on arm's behaviour:

- the abstention threshold moved from an uncertified demonstration constant of **0.5** to a
  certified **0.731**;
- at 0.5 the corpus never abstained, so abstention was invisible; at 0.731 it abstains on **two of
  the four `memory_only` trap queries**, even though the governing memo is the top hit;
- serving a calibrated generation requires `RECALL_ENV=production`, which refuses the static
  bearer token the warm HTTP transport used, so the on arm moved from a 458 ms warm socket to a
  per-session stdio server of about 9 s.

Predictions 9 and 10 in particular were written against the warm transport and cannot be scored
against the stdio one. Scoring the old predictions against the new configuration would be reading
a record as though it had said something it did not, which is the failure this whole convention
exists to prevent. The successor repeats the endpoints, states the calibrated configuration, and
predicts again.
