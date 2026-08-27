# Pre-registration: deliberate draft-search, and the first endpoint that is not a proxy

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `benchmarks/agent_ab/instructions/draft-query-v3.txt` (the whole intervention) driven by
the existing `scripts/agent_ab_run_tasks.py --instruction-file`.

## The question

Executed ground truth settled two things today. Draft-time search surfaces the governing memo in
**11 of 11** sessions that demonstrably needed it. And it fires on **29 of 36** that did not, so the
per-write design is dead: the memo is needed in 23% of sessions, and nothing gates the other 77%.

One shape survives, and it removes the trigger problem by having no trigger: **the agent searches
when it decides to, and when it does, it queries with the code it is about to write rather than
with a description of its task.**

**Does instructing that change what the agent does — and does it change whether the agent falls
into the trap?**

## Why this cannot be a replay, and why that matters

Every measurement in this lane replayed recorded sessions. That was legitimate while the question
was "would retrieval have surfaced the memo". It is not available here: "the agent chose to search,
with a query it composed" is a live behaviour, and no replay can produce it.

So this is also the **first registration in this lane whose endpoint is task success rather than a
retrieval proxy** — and today's most expensive lesson was that a proxy which looks obviously right
can be wrong for a third of cases. The endpoint is the checker: did the artifact avoid the trap.

## The intervention

One file. The current `hazard-query-v2.txt` already says *"search for the OPERATIONS you are about
to perform, not for your goal"* — and the run that used it produced goal-vocabulary queries anyway,
missing `ts-lf-rewrite` 0 for 6. Describing the operations is not the same act as querying with the
artifact, which is the distinction the whole lane turns on.

`draft-query-v3.txt` says instead: before you save a file or run a command, **paste the text you are
about to write as the query**. No paraphrase, no description, the literal draft.

## Two stages, because stage 2 is expensive and stage 1 can kill it

**Stage 1 — mechanism smoke, 16 sessions** (8 families x 2 repeats), treatment arm only. Endpoint:
of sessions that searched, what share issued at least one query that is a **verbatim or near-verbatim
substring of a payload they later wrote** (normalised whitespace, >= 20 characters matched). That is
a mechanical test of whether the instruction changed the ACT, and it needs no judge.

⛔ **If stage 1 is below 0.40, stage 2 is not run.** An instruction that does not change what the
agent queries with cannot change anything downstream, and the earlier instruction's failure is
precedent: it was read, followed in spirit, and produced the wrong queries anyway.

**Stage 2 — paired A/B, 48 pairs** (8 families x 6 repeats), control = `hazard-query-v2.txt`,
treatment = `draft-query-v3.txt`, same tasks and seeds. Endpoint: checker pass, McNemar on
discordant pairs.

## ⚠️ The power analysis, stated before the run rather than discovered after

Base failure rate on this arm is **11 of 48 (23%)**. With 48 pairs, McNemar detects only a large
effect:

| failures rescued (of 11) | approximate p |
|---|---|
| 11 | ~0.001 |
| 8 | ~0.008 |
| 6 | ~0.03 |
| 4 | ~0.13, **not significant** |

**So this design can detect "rescues at least 6 of 11" and nothing smaller.** If the true effect is
3 or 4 rescues, the honest report is "no detectable effect at n=48", which is NOT "no effect", and
this record commits to writing it that way. Anyone wanting to detect a modest effect needs roughly
four times the sessions, and that decision belongs to whoever is paying for the run.

## What I predict

Per `[[i-over-predict-effect-magnitudes]]`: mechanism rates are the class I estimate well, benefits
get the bottom of the band, and the prior task-success A/B on this harness measured **+0.154 with a
CI crossing zero** at 0.674 reach — reach and effect are different things, which is the single most
relevant prior here.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Stage 1 mechanism: 0.55 to 0.85** of searching sessions issue a verbatim draft query. An explicit "paste the text" instruction is a decision change, where this lane has under-called four times | outside the band |
| 2 | **Stage 1 does NOT reach 1.00.** Some tasks are answered without a file write early enough for the instruction to bite | it reaches 1.00 |
| 3 | ⚠️ **Stage 2 rescues 2 to 6 of the 11 failures**, i.e. below or at the edge of what this design can detect. Retrieval reaches 11 of 11; the gap between reaching the agent and changing its behaviour is where the prior null lives | outside the band |
| 4 | **No pair regresses from pass to fail by more than 2.** The instruction adds a search, it does not remove capability | 3 or more regressions |
| 5 | **Cost: stage 1 under 40 minutes; stage 2 roughly the prior 112-session run's cost**, which is the expensive part and the reason for the gate | stage 1 over 40 minutes |

## Decision rule, cross product over stage-1 mechanism and stage-2 rescues

| mechanism \ rescues | `>= 6` | `3-5` | `<= 2` |
|---|---|---|---|
| **`>= 0.55`** | **BUILD**: the instruction changes the act and the outcome; ship it as the memory usage instruction and preregister a confirmation on held-out families | **PROMISING BUT UNDERPOWERED**: report as "no detectable effect at n=48" and register a larger run with the sample size the power table demands. Do NOT ship on a non-significant positive | **KILL**: the agent searches correctly, retrieval reaches it, and it still falls in. The failure is downstream of memory entirely, which is a finding about agents and not about retrieval |
| **`0.40-0.54`** | **BUILD WITH A CAVEAT**: the effect appears despite partial adoption; state the adoption rate beside the effect | **UNDERPOWERED**, as above | **KILL** |
| **`< 0.40`** | impossible in practice; if it happens the mechanism metric is wrong and the run is void | stage 2 not run | stage 2 not run |

The two `<= 2` cells kill it, and that outcome would be the most informative of the three: it would
locate the remaining loss **after** retrieval and **after** the agent reads the memo, which is a
different problem from anything this lane has measured.

## What I already know

- Draft queries surface the memo for 11 of 11 sessions that needed it (executed ground truth).
- Goal queries surface it for 1 of 14 of the previously-registered misses.
- The prior instruction moved search RATE from 0.532 to 1.000 and did not move query VOCABULARY;
  the hit rate among searchers moved 0.600 to 0.674, not significant.
- The prior task-success A/B: +0.154, CI crossing zero, at 0.674 reach.

## Confounds I can name now

1. **The instruction is longer than the control's: 1,387 characters against 1,233, a 12.5%
   increase.** Any effect could be attention rather than content. A cleaner design would
   length-match; this one does not, and states the exact gap so a reader can weigh it.
2. **Pasting a draft costs tokens in the query itself**, and the median payload is 60 characters
   but the mean is 354. Cost is reported per arm, not assumed equal.
3. **The mechanism test is mechanical and therefore strict**: an agent that paraphrases its draft
   closely will score as not-adopting. That biases stage 1 DOWN, which is the safe direction for a
   gate.
4. **Same 8 families as everything else in this lane.** A held-out confirmation is part of the
   BUILD branch rather than an afterthought.
5. **Sessions are live and non-deterministic**; pairing controls the task and the seed, not the
   model's sampling.

<!-- frozen_above -->

## Result: stage 1 GATE FAILED

Measured 2026-08-27. Adoption **0.067** on the registered `ts-*` families against a gate of 0.40,
so **stage 2 is not run**. The agent searched once at the start and composed keyword queries rather
than pasting drafts. Full result in `2026-08-27-deliberate-draft-search-stage1-result.md`.
