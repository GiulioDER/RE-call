# Pre-registration: does unifying the instruction across all four channels help, or is one channel enough?

**Date:** 2026-08-25   **Status:** predicted, not yet measured
**Run identifier:** `agent-ab-channels-001`

## The question

RE-call ships its "when and how to search memory" instruction through **four** channels, and they
do not say the same thing. Does propagating the measured hazard-vocabulary recipe into the two
channels that currently teach the opposite raise the rate at which the governing memo reaches the
agent, on the same eight `memory_only` tasks, corpus and model as `agent-ab-skill-001`?

## Why this run exists

The four channels, as they stand today:

| Channel | Where | What it teaches |
|---|---|---|
| Skill | `plugin/skills/check-memory-before-acting/SKILL.md` | **decompose the task into operations, search for the operation plus its failure** |
| `CLAUDE.md` block | `recall/setup.py:1009` | "Call `recall_search` before proposing an idea, forming a hypothesis, or repeating past work" |
| `SessionStart` digest | `recall_hooks/__init__.py:188` | the same sentence, near-verbatim |
| Plugin README | `plugin/README.md` | prose about the skill, read by humans not agents |

Run `agent-ab-skill-001` (2026-08-23, 54 pairs) measured the skill's instruction and found the
governing memo reached the agent in **0.674** of sessions against a baseline **0.319**
(p = 0.0006). The honest half of that result, recorded in the skill itself: **the gain came from
searching at all, not from better queries.** The hit rate among sessions that searched moved only
0.600 to 0.674, which was not significant, and on the hardest task every session still asked in
goal vocabulary and missed.

That residual is what this run targets, and the hypothesis is specific: the two channels above
teach *goal* vocabulary ("proposing an idea", "forming a hypothesis", "repeating past work"), which
is precisely the framing skill-001 showed does not retrieve. They are not merely silent about the
recipe. They are a **standing instruction to do the failing thing**, injected before every first
turn, and the skill has to overcome them rather than merely be read.

⚠️ This is a hypothesis about a mechanism, not a measured fact. It is equally possible that one
channel is sufficient, the contradiction costs nothing, and unifying moves nothing — which is the
outcome the predictions below are written to be able to show.

## What changes

One thing: the wording of the two goal-vocabulary channels is replaced with the operation-plus-
failure recipe, distilled to fit each channel's budget. The `SessionStart` digest is the tighter
constraint, since it is charged to every session's context.

Everything else, tasks, fixtures, checkers, corpus generation, calibration, model, admission gate,
is held as in `agent-ab-skill-001`. The skill text itself is **not** changed by this run, so the
comparison isolates the two channels.

The replacement text is committed with this record as
`benchmarks/agent_ab/instructions/unified-channels-v1.txt`, and was checked for task
contamination the same way skill-001's was: no vocabulary from any of the eight governing memos or
task goals.

## What I predict

Discipline from [[i-over-predict-effect-magnitudes]]: eleven of twelve past predictions were
falsified high, by two to four times. So these are deliberately small, and the headline is
predicted to be **plausibly null**.

| # | Prediction | Ceiling arithmetic | Falsified if |
|---|---|---|---|
| 1 | **Primary: governing-memo rate among searchers 0.70 to 0.80** (skill-001 baseline 0.674) | ceiling ~0.90 at top-5 reachability; quarter-to-half of the +0.226 headroom is +0.06 to +0.11 | rate ≤ 0.674, or one-sided two-proportion test against skill-001's numerator with p ≥ 0.05 |
| 2 | **Search rate stays at or above 0.95** (skill-001 reached every session) | already at ceiling; this is a **guard**, not a gain | search rate falls below 0.90, which would mean the shorter digest suppressed the reflex skill-001 bought |
| 3 | **Reached rate 0.68 to 0.78** (skill-001 0.674) | the product of 1 and 2 | ≤ 0.674 |
| 4 | **The hardest task (`ts-lf-rewrite`) improves from its skill-001 figure** | goal vocabulary is the recorded cause of that specific miss, and this run removes two sources of it | unchanged, which localises the residual in the model rather than the instruction |
| 5 | **Cost: on-arm median input tokens move by less than ±5,000 against skill-001** | the digest gets shorter, the `CLAUDE.md` block gets longer, roughly cancelling | a move beyond ±5,000 in either direction, recorded as an unpredicted cost |
| 6 | **Control tasks unchanged** (both arms hold the file) | — | a significant control delta, which makes the run a harness artefact |

Prediction 1 is the headline and prediction 2 is its guard: a channel-unification that raised query
quality by suppressing searching would show as a win on 1 and a loss on 2, and reporting only 1
would be the fraud this pair exists to prevent.

**I expect this to be null.** The stated ranges are what would count as a real effect if one
exists; the honest prior, from skill-001's own decomposition, is that query vocabulary is
retrieval-side work and instruction wording has already given most of what it has to give.

## How it will be measured

```bash
docker start recall-agentab-corpus
python -u scripts/agent_ab_run_tasks.py --run-id agent-ab-channels-001 \
  --instruction-file benchmarks/agent_ab/instructions/unified-channels-v1.txt \
  --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default \
  > benchmarks/artifacts/agent_ab/channels-001.log 2>&1
python scripts/agent_ab_analyze_tasks.py --run-id agent-ab-channels-001
python scripts/agent_ab_compare_mechanism.py --run-id agent-ab-channels-001 \
  --baseline agent-ab-skill-001
```

Same design as skill-001: 8 `memory_only` tasks × 6 reps + 2 controls × 4 reps, off arm
`claude_md`, on arm `claude_md_recall`, `anthropic/claude-haiku-4.5` via OpenRouter. Metric names
carry over verbatim from skill-001's record so the two are comparable without re-derivation.

## What this run cannot settle

- **It cannot separate "the contradiction was costly" from "the new wording is better."** Both
  channels change at once, because shipping a product with three channels agreeing and one
  disagreeing is not a state worth measuring.
- **It is not powered for task success.** skill-001 was not either. Any task-success delta is
  recorded and is explicitly not a falsifier.
- **The changes land regardless of the result.** Removing a standing instruction to use the
  vocabulary that provably does not retrieve is correct on the evidence already in hand; this run
  asks whether it is *also* worth something measurable, and a null answers that honestly.
