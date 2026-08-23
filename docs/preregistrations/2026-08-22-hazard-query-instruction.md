# Pre-registration: does the hazard-query instruction close the initiation gap?

**Date:** 2026-08-22   **Status:** predicted, not yet measured
**Run identifier:** `agent-ab-skill-001`

## The question

Does replacing the one-sentence RE-call instruction with the hazard-query instruction
(`benchmarks/agent_ab/instructions/hazard-query-v2.txt`, committed with this record) raise the rate
at which the governing memo reaches the agent, on the same eight `memory_only` tasks, corpus and
model as `agent-ab-tasksuccess-001`?

## Why this run exists

Run `agent-ab-tasksuccess-001` found a null on task success (+0.154, cluster CI crossing zero) and
located the leak precisely: the layer worked when the memo reached the agent, and the memo reached
the agent in 15 of 47 on-arm sessions (0.319), the product of a search rate of 25/47 (0.532) and a
governing-memo rate of 15/25 (0.600). The recorded misses all asked about the task rather than the
failure. The `check-memory-before-acting` skill (shipped in the plugin, PR #467) was written to
correct exactly that, and has never been measured. This run measures its distilled instruction.

The treatment changes ONE thing against the baseline run: the on arm's instruction text, passed via
the new `--instruction-file` flag. The baseline sentence is 388 formatted characters; the new
instruction is 1,233. Everything else, tasks, fixtures, checkers, corpus generation, calibration,
model, CLI version (2.1.238, identical), static bundle, is held as in run 001.

The instruction was checked for task contamination before this record was written: it contains no
vocabulary from any of the eight governing memos or task goals (an earlier draft used the
`ts-lf-rewrite` miss as its worked example and was rewritten to a dependency example for this
reason; the check is recorded in the session, and the shipped SKILL.md deliberately keeps the CRLF
example because the plugin is not the thing under measurement here).

## What I predict

Discipline from [[i-over-predict-effect-magnitudes]]: quarter-to-half of ceiling for prompt-wording
benefits; costs measured from a smoke rather than guessed, and stated high.

| # | Prediction | Ceiling arithmetic | Falsified if |
|---|---|---|---|
| 1 | **Primary: governing memo reaches the agent in 0.45 to 0.55 of on-arm `memory_only` sessions** (baseline 15/47 = 0.319) | full compliance would give roughly 0.85 × 0.75 ≈ 0.64; quarter-to-half of the +0.32 headroom is +0.13 to +0.23 | rate ≤ 0.319, or one-sided two-proportion test against the baseline 15/47 with p ≥ 0.05 |
| 2 | **Search rate 0.65 to 0.80** (baseline 25/47 = 0.532) | ceiling 1.0; quarter-to-half of +0.468 | ≤ 0.532 |
| 3 | **Governing-memo rate among searchers 0.65 to 0.78** (baseline 15/25 = 0.600) | ceiling ~0.90 (every memo qualified reachable at top-5); quarter-to-half of +0.30 | ≤ 0.600 |
| 4 | **`ts-lf-rewrite`: the governing memo reaches ≥ 2 of 6 on-arm sessions** (baseline 0 of 6, the clean miss) | operation vocabulary is the mechanism; if it works anywhere it works here | 0 of 6 again |
| 5 | **Task success delta (exploratory, NOT a falsifier): +0.10 to +0.25 per-task**, CI likely still crossing zero at 8 tasks | mechanism gain × conversion observed in run 001's three improving tasks | recorded either way; this run is not powered for it |
| 6 | **Cost: on-arm median input tokens exceed baseline's +55,959 by up to +20,000** (more searches, longer instruction) | stated high per the ×5 lesson; the smoke measures it before the run | median overhead FALLS, which would mean the instruction suppressed searching |
| 7 | **Control tasks unchanged** (both arms hold the file) | — | a significant control delta, which makes the run a harness artefact |

Prediction 1 is the headline. Predictions 2 and 3 are the two factors it decomposes into, predicted
separately so a headline move with an unmoved mechanism is visible as the fraud it would be.

## How it will be measured

```bash
docker start recall-agentab-corpus   # container exited; volume recall-agentab-corpus-data is the evidence
python -u scripts/agent_ab_run_tasks.py --run-id agent-ab-skill-001 \
  --instruction-file benchmarks/agent_ab/instructions/hazard-query-v2.txt \
  --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default \
  > benchmarks/artifacts/agent_ab/skill-001.log 2>&1
python scripts/agent_ab_analyze_tasks.py --run-id agent-ab-skill-001
```

Smoke first: `--limit 2 --reps 1` under run id `agent-ab-skill-001-smoke`, discarded from the
result, kept for the cost measurement in prediction 6.

Same design as run 001: 8 `memory_only` tasks × 6 reps + 2 controls × 4 reps = 56 pairs, 112
sessions, off arm `claude_md`, on arm `claude_md_recall`, `anthropic/claude-haiku-4.5` via
OpenRouter, admission gate as committed. **The metric names:** "search rate" is sessions with ≥ 1
`recall_search` call over admitted on-arm `memory_only` sessions; "governing-memo rate" is sessions
whose retrieved sources include the task's `governing_memo` over sessions that searched; "reached"
is their product's numerator over admitted on-arm `memory_only` sessions. The baseline comparison
is **cross-run** against the archived `agent-ab-tasksuccess-001` (`~/.claude/archive/`), not
paired; the paired structure inside this run serves predictions 5 and 7.

## What I already know

- Baseline mechanism numbers above: `~/.claude/archive/agent-ab-tasksuccess-001/analysis.json`,
  memo [[agent-ab-task-success-result-2026-08-22]].
- Every governing memo was qualified reachable at top-5 by its probe query
  (`benchmarks/agent_ab/task-qualification.json`, committed `09aa03f0`), so prediction 3's ceiling
  is a retrieval fact, not hope.
- Instruction placement matters and first position is already the baseline's placement
  (`arms.py`, the load-bearing comment): this run does not change placement, only text.
- Eleven of twelve magnitude predictions in this project have been too high; costs 5× too low.

## Confounds I can name now

1. **Length, not content.** The instruction is 3× the baseline's length; more instruction tokens
   about the tool may raise search rate regardless of what they say. Not controlled here; a
   length-matched scramble would need a third run. If predictions 2 moves but 3 does not, length
   salience is the better explanation and the record should say so.
2. **Cross-run drift.** Same CLI version (2.1.238), same model id, same corpus generation and
   calibration, but OpenRouter's serving of `claude-haiku-4.5` may have changed since 2026-08-21.
   No way to pin it; recorded, not controlled.
3. **Author overfitting.** The instruction's author knows the task set. The vocabulary check above
   removes the direct leak; the indirect one (the instruction's shape being tuned to these eight
   failure modes) is real and bounded only by the task set's diversity.
4. **Prediction 4 has n=6.** It is stated because run 001 made `ts-lf-rewrite` the clean miss, and
   a mechanism that cannot move its cleanest case is not the mechanism I think it is.

---

## Result (2026-08-23)

**Status:** measured. Run `agent-ab-skill-001`, 112 sessions, **54 pairs admitted, 2 discarded**
(both `ts-autouse-tmp-path` reps whose stdio server reported `failed`, so RE-call was never
available; the gate worked). Predictions above are untouched; this section is appended.

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| 1 | reach 0.45 to 0.55 | **31/46 = 0.674**, one-sided Fisher p = 0.0006 vs 15/47 | direction confirmed, band UNDER-predicted |
| 2 | search rate 0.65 to 0.80 | **46/46 = 1.000**, p < 0.0001 vs 25/47 | direction confirmed, band UNDER-predicted |
| 3 | memo given searched 0.65 to 0.78 | **31/46 = 0.674**, p = 0.356 vs 15/25 = 0.600 | in band, and NOT distinguishable from baseline |
| 4 | ts-lf-rewrite reach >= 2 of 6 | **0 of 6**, with 6 of 6 searching | **FALSIFIED** |
| 5 | success delta +0.10 to +0.25 (exploratory) | per-task **+0.208**, cluster CI [-0.021, +0.458], sign p = 0.375; per-pair +0.196, p = 0.022 | in band, headline still not significant, as stated |
| 6 | cost above +55,959 by up to +20,000 | median **+106,946** input tokens, +36.5 s wall | direction confirmed, band UNDER-predicted by ~2.5x |
| 7 | control unchanged | 1.000 vs 1.000, no discordant pairs | confirmed |

**What the decomposition says.** The instruction closed the initiation gap completely and did not
significantly move the formulation gap: reach doubled (0.319 to 0.674) almost entirely because
every session now searches, while the hit rate among searchers moved 0.600 to 0.674 with p = 0.356.
Confound 1 of this record (length or salience rather than content) therefore stands unresolved for
the initiation effect, and the content-specific claim, that operation vocabulary retrieves what
goal vocabulary misses, FAILED its cleanest test: all six `ts-lf-rewrite` on-arm sessions searched
and every query was still goal vocabulary ("version bump release script", "recall/version.py
format"). The agent does not perceive "a python script will edit a file" as an operation distinct
from its goal; instruction text alone did not induce that reframing where the goal's pull is
strongest. `ts-worktree-import` shows the same shape (6 of 6 searched, 1 of 6 reached).

**Task success** moved from run 001's +0.154 to +0.208 per-task, and the per-pair view crossed into
significance (p = 0.022, labelled as overstating confidence by design). The 8-task design still
cannot certify the headline, exactly as the limitation section predicted. One task went negative
(`ts-sample-covers-tail`, -0.33, reach 3/6), which run 001 did not have.

**Cost of the treatment as shipped: roughly double the baseline instruction's.** Median +106,946
input tokens and +36.5 s per session against the baseline's +55,959 and +24.5 s. The magnitude
lesson in [[i-over-predict-effect-magnitudes]] holds in both directions here: two benefit bands
under-predicted for a decision-changing intervention, and a cost band under-predicted even after
applying the times-five correction to the previous surprise.

**What follows, not measured here:** the initiation problem is solved by instruction; the
formulation problem is not, and the next lever is retrieval-side (query expansion, or indexing
memos under operation vocabulary), not more prompt text.
