# Does adding RE-call to CLAUDE.md reduce known-hazard mistakes?

**Written 2026-08-21, before any session of this run was executed.** Run identifier
`agent-ab-additive-001`.

## Why this run exists, and what was wrong with the last one

The previous run compared `claude_md` against **RE-call alone**, which measured *replacing* the
hand-written file. That is not a configuration anybody uses, and its control said so: on traps whose
fact lives in `CLAUDE.md`, the arm holding `CLAUDE.md` won by 38 points, purely because the other
arm did not have the file. The 50-point primary result stands, but it answers a question nobody
asked.

This run makes the comparison **additive**:

| arm | system prompt | memory layer |
|---|---|---|
| off (`claude_md`) | `CLAUDE.md` + `MEMORY.md` | none |
| on (`claude_md_recall`) | the **same bytes**, plus one sentence naming the tool | RE-call |

The runner asserts that the on arm's prompt begins with the off arm's byte for byte and aborts
otherwise, so the arms cannot silently differ by more than the memory layer.

## Fixed configuration

| Field | Value |
|---|---|
| Tasks | `benchmarks/agent_ab/tasks/traps-deep.jsonl`, 10 traps |
| Primary | 4 `memory_only` traps x 10 repetitions = **40 pairs** |
| Controls | 6 traps (`both`, `claude_md_only`) x 4 repetitions = **24 pairs** |
| Agent | `anthropic/claude-haiku-4.5` via OpenRouter, `--bare` |
| Judge | `openai/gpt-4.1-mini` via OpenRouter, `max_tokens=8192` |
| Corpus | calibrated generation, threshold 0.731, separability 0.980 [0.952, 1.000] |
| Transport | stdio, `RECALL_ENV=production`, strict trust |
| Database | `recall-agentab-corpus` on port 5407, **named volume** |
| Claude Code | 2.1.238 |

Three defects from the last run are fixed before this one starts, and each cost something:

1. **Records are written incrementally and fsynced.** The last runner died at 71 of 100 sessions
   and lost its index; only the transcripts survived. A dead run should cost the sessions it did
   not do, not the ones it did.
2. **Output goes to a log file, never through `grep`.** Piping the last run through `grep -v`
   swallowed every diagnostic and masked the exit code, so its cause is permanently unknown.
3. **The corpus lives in a container with a named volume.** The session container was removed by
   something outside the session three times, destroying the corpus each time.

## Endpoints and how they will be read

**Primary: hit rate on the 4 `memory_only` traps**, where the governing fact is in the memory store
and not in `CLAUDE.md`. Reported two ways, both preregistered here:

- **Per-task (headline).** One rate per trap, repetitions collapsed. ⚠️ With 4 distinct traps a
  sign test cannot reach p<0.05 **at any effect size**, so this view is reported as descriptive
  (how many traps improved, by how much) with a cluster bootstrap that resamples traps. No p-value
  will be invented for it. This limit is a property of running 4 hazards and is stated wherever the
  result appears.
- **Per-pair (consistency check).** Exact McNemar over all 40 pairs. This will produce the smaller
  p-value and it **overstates confidence**, because repetitions of one trap are correlated. It is
  secondary and labelled as such.

**Controls: the 6 traps whose fact is in `CLAUDE.md`.** Both arms hold the file, so both should
avoid these. A difference here is noise or a side effect of the memory layer, not a benefit, and
either reading weakens the primary claim.

## Predictions

Predicted before measuring, per the standing rule. Last run I **under**-predicted for the first
time; the correction here is not to simply predict bigger, because the treatment has changed: the
on arm now also carries 17.8k characters of `CLAUDE.md`, which gives it an alternative to searching
and may suppress tool use.

1. Off arm `memory_only` hit rate: **0.65**. (Measured 0.688 for the same arm last run.)
2. On arm `memory_only` hit rate: **0.30**. Worse than the 0.188 that RE-call alone achieved,
   because the added static context competes with the tool.
3. Reduction: **35 points**. Falsified if under 12 points, or if the cluster interval includes zero.
4. Per-task: **3 of 4** traps improve.
5. `recall_search` called in **65%** of on-arm `memory_only` sessions, DOWN from 81% when RE-call
   was the only memory available.
6. Governing memo in retrieved contexts in **45%** of those sessions.
7. Controls: difference within **10 points** either way, in both control loci.
8. On-arm input tokens **+8%**. Both arms now carry the static bundle, so the delta is the tool
   definitions plus retrieved text, not the file.
9. On-arm wall time **+10 s** median, of which ~9 s is stdio server startup, reported separately.
10. `recall_latency_ms` median under **400 ms**. (182 ms measured last run.)
11. Ragas `answer_correctness`: **no significant difference**, point estimate within +0.05.

## Exclusions and stopping rules

A pair is discarded, not scored, unless the on arm's `system/init` lists a `mcp__recall*` tool and
the off arm's does not. Missing measurements stay null and are never replaced with zero. No task,
arm, metric or stopping rule changes after any data is seen. Any rerun gets a new identifier and a
new record.

If the run dies again, the salvage path rebuilds records from `records.partial.jsonl` and the
transcripts, and the result says so, including how many pairs were lost.

## Stated limitations, fixed in advance

- **4 distinct hazards.** The claim generalises to the population of hazards these 4 represent, and
  no further. This was a deliberate choice to get data today rather than authoring 8 more traps.
- The corpus is this project's own memory store, so the result describes an agent working in the
  repository whose memory this is.
- The generation was built `--unverified-development`: fit for a benchmark, not for a trust claim.
- `total_cost_usd` from the CLI is not reported; it was measured about 6x wrong through the gateway.

## Amendment, 2026-08-21, after the wiring smoke and before the measurement

**Nothing above is edited.** Two changes to the treatment and the scoring, both forced by a
6-session wiring smoke, both made before the measured run started.

### 1. The RE-call instruction moves to the FRONT of the system prompt

As preregistered, the instruction was appended after the static bundle. The smoke measured what
that does: **16 RE-call tools available, zero calls in both on-arm sessions**, and the arm walked
into the `omp_threads` hazard recommending `taskset`. An instruction buried after 17,498 characters
produced a **0% search rate**, so the run would have measured prompt placement rather than
retrieval.

The instruction now leads the prompt and is more directive (419 characters, up from 293). After the
change the smoke measured **3 of 3 on-arm sessions searching**.

This was chosen on the **mechanism** (the search rate), not on any trap outcome. The off arm is
byte-for-byte unchanged; the runner asserts the static bundle appears verbatim in the on arm's
prompt and that the on arm adds no more than 2,000 characters beyond it, so the arms still differ
by the instruction and the tools alone.

**Disclosure:** the smoke also produced trap outcomes on 3 pairs, so those were seen before this
amendment was written. **Predictions 1 through 4 are unchanged** and were not revised in light of
them. The mechanism predictions that the change invalidates are restated below.

### 2. A session that does not answer is excluded, not scored as clean

Every trap detector fires on the presence of a wrong instrument, so a reply naming no instrument
avoids all of them trivially. The smoke caught it: asked for the exact commands to render an SVG,
one arm replied *"Before I give you the exact commands, I need a bit more information"* and listed
questions. It scored as having avoided the hazard. Left in, this rewards hedging and moves the
primary endpoint by whichever arm hesitates more.

`traps.answered()` now decides deterministically whether a session committed to an answer, and a
pair where **either** arm did not answer is excluded from the trap rate and counted separately per
arm. The rule is conservative: a response counts as answered unless it contains no concrete
artifact **and** asks a question.

### Restated mechanism predictions

Predictions 5 and 6 were written about the buried instruction and no longer describe the treatment.
They are superseded here, before measuring, by:

- **5a.** `recall_search` called in **85%** of on-arm `memory_only` sessions, up from the 65%
  predicted for the buried instruction and from the 0% it actually produced.
- **6a.** Governing memo in retrieved contexts in **55%** of those sessions.
- **12.** Non-answers, newly measurable: under **10%** of sessions in either arm, and **within 5
  percentage points** between the arms. A large asymmetry here would mean the memory layer changes
  how often the agent commits to an answer, which is a finding in its own right and would need
  reporting beside the trap rate rather than hidden by the exclusion.

Predictions 8 and 9 (tokens, wall time) are expected to move against RE-call now that the on arm
actually searches; they are **not** revised, and will be scored as written.

## Rerun, 2026-08-21: `agent-ab-additive-002` supersedes `agent-ab-additive-001`

**Nothing above is edited.** Run `agent-ab-additive-001` executed and is superseded, not deleted.
Its artifacts stay on disk and its outcome is stated here, because the next run is no longer blind
and pretending otherwise would be the dishonest option.

### Why 001 is being rerun

The OpenRouter account ran out of credit part-way through. **28 sessions died on `402 Insufficient
credits`** and the gate discarded 15 pairs. The failure landed on the control tasks, which sit last
in the manifest, so:

- the primary endpoint completed intact: all **40 `memory_only` pairs**;
- the controls were reduced to **4 and 5 pairs** against the 24 designed.

Finishing only the missing controls was possible and was rejected deliberately. It would produce a
result whose controls were measured an hour later than its primary, and "the controls were run
separately after we topped up" is precisely the sentence a sceptical reader pulls on. Run 002 is
the same design executed once, end to end, with credit in hand.

### ⚠️ Run 002 is NOT blind, and here is exactly what was seen

Run 001's outcome is known to me before 002 starts. Disclosed in full so a reader can discount it:

| endpoint | run 001 |
|---|---|
| `memory_only` per-pair, n=40 | on 0.100, off 0.675, delta −0.575, CI [−0.725, −0.425], p<0.0001 |
| per-task | 3 of 4 traps improved; cluster CI [−0.850, −0.175] |
| controls (underpowered) | 0.000 vs 0.000 on both loci |
| cost | input +3,410 (p=0.049), wall time −29,850 ms (p=0.078) |
| mechanism | search rate 86%, zero non-answers in either arm |

**No prediction is revised.** Predictions 1 through 12, including the amended 5a, 6a and 12, stand
exactly as committed, and will be scored against 002. They were written blind against 001; against
002 they are a **replication check**, not a blind forecast, and any writeup must say so. The
temptation this creates is to treat a confirming 002 as independent evidence. It is not: it is the
same design run twice, the second time knowing the first answer.

Nothing else changes: same tasks, same reps, same arms, same corpus generation and calibration,
same gate, same stopping rule.

### One fix landed between the runs

`scripts/agent_ab_analyze.py` was reading every record in `records.jsonl` and calling the result
"admitted pairs", so run 001's cost table averaged 15 pairs whose sessions never completed. Fixed
and regression-tested before 002. It changed two of 001's conclusions in opposite directions, which
is why the corrected 001 figures above differ from what I reported earlier: wall time went from
significant to not, and input tokens from not to marginal. The analysis code is now identical for
both runs, so 001 and 002 are comparable.

## ⚠️ Post-hoc exclusion, 2026-08-21: the `shared_db` control task is dropped

**Nothing above is edited.** This is a change to a preregistered design made **after seeing the
task misbehave**, which is the kind of decision that needs writing down before it is acted on
rather than explained afterwards. It was proposed with the alternatives and approved by the user.

### What happened

Run 002 completed its primary endpoint and two control tasks, then stalled on `trap-shared-db`.
Measured at the moment of the decision:

- 96 of 128 sessions done, **0 errors**, all 40 `memory_only` pairs complete;
- the `shared-db#r1` pair had been running **16.7 minutes with about 17 seconds of CPU** each, so
  the sessions were burning wall clock rather than working;
- a 75-minute window earlier produced **no completed session at all**;
- the same task hit the 1800-second timeout in run 001.

### Why the task, and not the run, is at fault

`shared_db` is the hazard about pointing the test suite at a database container another session
owns. **`docker` is denied in every arm by design**, so the trap can be scored from the denial
without destroying anyone's data. The agent therefore reaches for a blocked command, is refused,
and casts around instead of concluding. That failure is mechanical and has nothing to do with
memory: it is a property of the task's interaction with the tool policy, and it occurs in **both**
arms.

### What is dropped, and why this cannot flatter the result

`shared_db` is a **`claude_md_only` control**, not a primary trap. Its governing fact lives in
`CLAUDE.md`, which both arms hold, so it was predicted to show **no difference** (prediction 7) and
showed none in run 001 (0.000 against 0.000). Dropping it therefore removes a task that RE-call was
expected to draw, not one it was expected to win. The primary endpoint does not contain it and is
unaffected.

Remaining controls after the exclusion: `ruff_format` and `git_add_all` (complete, in run 002),
plus `local_master`, `stale_count` and `main_checkout` (12 pairs, to be run as a continuation).

### The cost of this choice, stated plainly

The controls are now **spliced across two runs**, which is the exact objection that motivated
restarting 001 rather than topping it up. The difference, and the reason this was still judged the
better trade: the **primary endpoint and two controls sit in one uninterrupted run**, and only 12
control pairs are appended. In 001 it was the primary that would have been spliced against
later-measured controls.

`claude_md_only` retains **one** task after the exclusion, so that locus is thin and any reading of
it is correspondingly weak. That is a real loss and is not offset by anything.

## Result

Measured 2026-08-21. **Nothing above this line has been edited.**

Runs: `agent-ab-additive-002` (primary endpoint + 2 controls, 48 pairs, uninterrupted, 0 errors,
0 gate discards) and `agent-ab-additive-002c` (12 further control pairs, 0 discards) after the
`shared_db` exclusion recorded above. Corpus `gen_f01fc522...`, calibration `cal_b40f2c6e...`
certified at threshold 0.731, separability 0.980, served `trusted` over stdio. Claude Code 2.1.238,
agent `anthropic/claude-haiku-4.5`, judge `openai/gpt-4.1-mini`, both via OpenRouter.

### Primary endpoint

| locus | n | +RE-call | CLAUDE.md | delta | 95% CI | p |
|---|---|---|---|---|---|---|
| **memory_only** | 40 | **0.000** | 0.525 | **-0.525** | [-0.675, -0.375] | **<0.0001** |

Discordant pairs: **on-only 0, off-only 21.** The RE-call arm triggered **no trap in any of the 40
pairs**, and never once hit a hazard the baseline avoided.

Per-task, the headlined view (one rate per trap, repetitions collapsed):

| trap | +RE-call | CLAUDE.md | delta |
|---|---|---|---|
| omp_threads | 0.000 | 0.900 | -0.900 |
| cairo_render | 0.000 | 0.800 | -0.800 |
| torch_install | 0.000 | 0.400 | -0.400 |
| cast_conversion | 0.000 | 0.000 | 0.000 |

**3 of 4 traps improved**, mean -0.525, cluster CI **[-0.850, -0.200]** resampling traps rather
than repetitions. As preregistered, **no p-value is reported for this view**: with 4 distinct traps
a sign test cannot reach p<0.05 at any effect size.

### Controls: the memory layer changes nothing where the file already knows

| locus | n | +RE-call | CLAUDE.md | delta | p |
|---|---|---|---|---|---|
| `both` (002c) | 12 | 0.083 | 0.083 | **0.000** | **1.0000** |
| `both` (002) | 4 | 0.000 | 0.000 | - | below the 6-pair floor |
| `claude_md_only` (002) | 4 | 0.000 | 0.000 | - | below the 6-pair floor |

Discordant pairs on the 12-pair control: **1 and 1**. This is the result that rules out "the treated
arm is simply better at everything".

### Answer quality, and the finding that was not predicted at all

| metric | n | +RE-call | CLAUDE.md | delta | p |
|---|---|---|---|---|---|
| answer_correctness | 48 | **0.337** | 0.149 | **+0.188** | **0.0002** |
| factual_correctness | 48 | **0.506** | 0.272 | **+0.234** | **<0.0001** |

Restricted to the primary tasks (n=40): answer_correctness **0.358 vs 0.144, p=0.0003**;
factual_correctness **0.559 vs 0.288, p<0.0001**. On the 12 control pairs the same metrics show
**no significant difference** (+0.119, p=0.15 and +0.074, p=0.26), so the gain is concentrated
exactly where the memory holds the fact.

⚠️ **This contradicts the superseded substitutional run, and the contradiction is the point.**
Comparing `CLAUDE.md` against **RE-call alone**, run 001 found no quality difference (+0.044,
p=0.43). Comparing `CLAUDE.md` against **`CLAUDE.md` + RE-call**, this run finds +0.188 at
p=0.0002. **Replacing the hand-written file with retrieval does nothing for answer quality; adding
retrieval to it roughly doubles correctness.** Prediction 11 expected no significant difference and
is falsified in the direction nobody predicted.

### Cost

| metric | n | +RE-call | CLAUDE.md | delta mean | delta median | p |
|---|---|---|---|---|---|---|
| input_tokens | 48 | 52,938 | 55,232 | -2,293 | **+14,904** | 0.1724 |
| output_tokens | 48 | 1,208 | 1,358 | -150 | -38 | 0.9150 |
| wall_time_ms | 48 | 23,344 | 56,379 | -33,035 | **+1,703** | 0.9473 |
| model_turns | 48 | 4 | 5 | -1 | 0 | 0.8556 |

**The mean and the median disagree in sign on three of four metrics.** Reading means alone would
claim RE-call is cheaper and faster. It is typically **dearer and slightly slower**, and
occasionally saves an enormous amount because the baseline goes exploring; the rank-based p-values
follow the median and none is significant. On the control tasks, where retrieval cannot help, it is
significantly **slower**: +11,283 ms median, p=0.0269. That is the honest cost of the layer.

### Scoring the predictions

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | off hit rate 0.65 | 0.525 | close, slightly over |
| 2 | on hit rate 0.30 | **0.000** | under-predicted |
| 3 | reduction 35 points | **52.5 points**, CI excludes zero | correct, magnitude under-predicted |
| 4 | 3 of 4 traps improve | **3 of 4** | **exact** |
| 5a | search rate >= 85% | **85%** (34/40) | **exact** |
| 6a | governing memo retrieved 55% | **82%** (33/40) | under-predicted |
| 7 | controls within 10 points | **0.000**, p=1.0000 | **correct** |
| 8 | on arm worse on `claude_md_only` by 10-30 | 0.000 vs 0.000 | not observed |
| 9 | input tokens +8% | median +14,904 (~+28%), not significant | under-predicted, unconfirmed |
| 10 | wall time +10 s median | median +1.7 s, not significant | over-predicted |
| 11 | no quality difference, within +0.05 | **+0.188, p=0.0002** | **falsified** |
| 12 | non-answers <10%, arms within 5pp | **0 and 0** | **correct** |

Six correct or exact, three under-predicted, one over-predicted, one falsified, one not observed.
The house pattern of over-predicting effect sizes did **not** hold here: predictions 2, 3, 6a and 9
were all too conservative, and the only falsified prediction was falsified by an effect that was
predicted to be absent.

### What this supports, and what it does not

**Supports:** on this corpus, task set and model, adding a calibrated retrieval memory layer to an
existing `CLAUDE.md` eliminated a class of known hazard (0 of 40, from a 52.5% baseline) and roughly
doubled answer correctness, while changing nothing on tasks whose facts the file already held.

**Does not support:** any claim of general uplift; any claim that the work itself succeeds more
often, since no task here writes code, runs a test or is scored on completion; any cost advantage;
generalisation beyond the 4 distinct hazards the primary rests on, or beyond an agent working in
the repository whose memory this is.

**Weaknesses, restated rather than buried:** 4 distinct primary traps, so the per-task view can
never reach significance; `claude_md_only` retains one task after the `shared_db` exclusion;
controls are spliced across two runs; the generation was built `--unverified-development`; and this
run was **not blind**, as recorded above.

The next benchmark, which measures whether the WORK succeeds rather than whether a hazard was
avoided, is specified in `benchmarks/agent_ab/NEXT-BENCHMARK-TASK-SUCCESS.md`.
