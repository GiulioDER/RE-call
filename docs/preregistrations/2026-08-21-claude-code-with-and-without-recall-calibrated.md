# Claude Code with and without RE-call, on a calibrated corpus

**Written 2026-08-21, before any paired session was run.** Supersedes
`2026-08-20-claude-code-with-and-without-recall.md`, which predicted against an uncalibrated
corpus and a warm HTTP transport; neither is the configuration any more, and that record is kept
unedited with the reason appended to it.

Run identifier: `agent-ab-traps-002`.

## Question

Does an agent doing ordinary work in this repository make fewer known, expensive mistakes when it
has a **calibrated** retrieval memory layer than when it has the hand-written `CLAUDE.md`, and what
does that cost in tokens and wall time?

## Measured before this was written, and therefore not predictions

| Fact | Value |
|---|---|
| Corpus | 188 sources, 981 chunks, from the project memory store |
| Generation | `gen_69eafbee8bcd459fb5e45b6eea8f6c3d`, active, built `--unverified-development` |
| Calibration | `cal_eb7c63dca4424f22b42609aa3f8491c5`, published, certified |
| Threshold | **0.731**, against an uncertified demonstration default of 0.5 |
| Separability | **0.980, 95% CI [0.952, 1.000]**, 50 answerable / 50 unanswerable |
| Score distributions | answerable median 0.789, unanswerable median 0.674; 3 of 50 answerable fall below the threshold |
| Trust state served | `trusted`, `calibrated: true`, under the **strict** policy |
| Transport | stdio, `RECALL_ENV=production`, about 9 s per session |
| Claude Code | 2.1.238 (upgraded from 2.1.220, which does not wait for a pending MCP server) |
| Static bundle | 17,499 chars; memory store 636,191 chars across 185 files, about 36:1 |
| Trap loci | 4 `memory_only`, 4 `both`, 2 `claude_md_only` |

The labelled query set (`benchmarks/agent_ab/calibration/memory-query-set.json`) shares **no query**
with any trap probe, so the threshold was not fitted on the test set.

### The abstention finding, which the design must now account for

At 0.731 the corpus **abstains on two of the four `memory_only` trap queries** (`omp_threads`,
`cairo_render`) even though the governing memo is the top hit. Hits are still returned, so the
agent can use them; abstention is a statement about confidence, not a refusal to answer. This was
invisible before calibrating, because at 0.5 nothing ever abstained. It is kept and reported rather
than tuned away: per-trap abstention is a primary diagnostic below.

## Design

Unchanged from the superseded record except for the configuration above. Paired, one task at a
time, both arms started together; three arms run as two comparisons (`recall` vs `claude_md` is the
headline, `recall` vs `bare` the ceiling); `--bare` in every arm so context arrives only by flag;
`docker` denied in every arm so a shared-container hazard is measured from the denial and never
realised. Task set `benchmarks/agent_ab/tasks/traps.jsonl`, 10 tasks, 5 repetitions.

**Admission.** A pair is discarded, not scored, unless the on arm's `system/init` lists a
`mcp__recall*` tool and the off arm's does not. A tool that was available and never called is
admitted and counted.

## Predictions

Predicted low again, and lower than the superseded record on the primary endpoint, because
abstention now fires on half the winnable traps. The house record is eleven of twelve predictions
falsified, all too high by two to four times.

**Primary: trap hit rate on the 4 `memory_only` traps, headline comparison.**

1. Off arm (`claude_md`) hit rate: **0.70**.
2. On arm (`recall`) hit rate: **0.50**.
3. Absolute reduction: **20 percentage points**. Falsified below 8 points, or if the 95% interval
   on the paired difference includes zero.

**Mechanism, predicted beside the outcome so a miss can be attributed.**

4. `recall_search` is called in at least **70%** of on-arm sessions on `memory_only` tasks.
5. The governing memo appears in the retrieved contexts in at least **50%** of those sessions.
6. On the two traps that abstain, the on-arm hit rate is **worse by 10 to 25 points** than on the
   two that do not. If retrieval fires and the memo comes back but the abstaining traps do not
   improve, the limiting factor is the agent's willingness to act on evidence the layer flagged as
   low confidence, not retrieval.

**Controls, which is why traps RE-call should lose are in the set.**

7. `both` traps: difference within **10 points** either way.
8. `claude_md_only` traps: on arm **worse by 10 to 30 points**.

**Cost of the memory layer.**

9. On-arm input tokens **+40%** against `claude_md`, revised up from the superseded record's 20%
   after a smoke pair measured 26,359 against 18,189, which is +45%. Sixteen MCP tool definitions
   enter the context whether or not a search happens.
10. On-arm wall time **+12 s** per session, median, of which about **9 s is stdio server startup**
    and the rest is the extra turn. Reported with the startup component separated, because a warm
    deployment does not pay it per session and a reader must be able to subtract it.
11. `recall_latency_ms` accounts for under **1 s** of that.

## Exclusions, stopping rules, analysis, artifacts

As in the superseded record, unchanged: no post-hoc task removal or metric change; missing values
stay null; Wilcoxon on tokens and wall time, McNemar on binary outcomes, bootstrap intervals on
paired differences; `total_cost_usd` from the CLI is not reported, because it was measured about
6x wrong through the gateway.

**Stated limitations.** The generation was built `--unverified-development`, so its manifest is not
cryptographically verified: this corpus is fit for a local benchmark, not for a trust claim. The
corpus is the project's own memory store, so the result generalises to "an agent working in the
repository whose memory this is", and no further.

**Publication gate.** Retrieved chunks land verbatim in transcripts and this corpus contains host
inventory and pointers to credential files. No transcript is published unreviewed; the preferred
route is a rerun against a filtered corpus.

## Deviation, recorded 2026-08-21 while the headline run was still executing

**Nothing above is edited.** The design section says 10 tasks and 5 repetitions for both
comparisons. The **ceiling** comparison (`recall` vs `bare`) will run at **3 repetitions**, not 5.
The **headline** comparison (`recall` vs `claude_md`) is unchanged at 5 and was already executing
when this was written.

Reason, stated before the data exists: the measured session rate is about 1.8 minutes, so the
ceiling comparison at 5 repetitions is roughly three hours of wall clock, and the two comparisons
cannot be run concurrently because **wall time is one of the endpoints** and CPU contention would
corrupt it in both. The ceiling is the secondary comparison; it bounds how much of the headline gap
is attributable to memory at all, and it does not need the headline's precision to do that.

What this costs, so the weaker number is not read as an equal one: 3 repetitions over 4
`memory_only` tasks is **12 pairs** against the headline's 20. That clears the 6-pair floor in
`benchmarks/agent_ab/stats.py`, so the exact test can still reach significance on a clean effect,
but the ceiling comparison has less power and a wider interval than the headline, and any null in
it is correspondingly weaker evidence. The `both` and `claude_md_only` controls drop to 12 pairs
each on the same terms.

No endpoint, task, arm, detector or prediction changes. Only the repetition count of the secondary
comparison, and only downward.

## Result

Measured 2026-08-21. **Nothing above this line has been edited.** Headline comparison only
(`recall` vs `claude_md`); the ceiling comparison has not been run.

### What the run actually was, including what went wrong

The runner process **died at 71 of 100 sessions** and never wrote its records. The cause is not
recoverable: the invocation was piped through `grep -v`, which swallowed every diagnostic and
masked the exit code. That is an instrumentation defect in how the run was launched, not a finding
about the system under test, and it is recorded here because the run it produced is the one being
reported.

The evidence survived, because `run_claude_case` writes each transcript to disk **before** parsing
it. `scripts/agent_ab_salvage.py` rebuilt 71 records into **35 complete pairs, 0 discarded by the
gate**, excluding 1 unpaired survivor (`trap-shared-db#r1`). So this is 35 pairs against the 50
that were preregistered, and the per-locus counts below are correspondingly smaller than planned.

Two consequences that must travel with the numbers:

- **`wall_time_ms` is not the preregistered measurement.** The runner's own timing died with it, so
  salvaged records use the session's self-reported `duration_ms`, which **excludes** process spawn
  and the ~9 s MCP server startup. Prediction 10 was written about the other quantity and cannot be
  scored against this one.
- Other sessions on this machine began four concurrent `pytest` runs and a PyInstaller build at
  11:36. **Zero measured sessions completed after that point**, so the contamination boundary is
  clean and no reported figure is affected, but the run stopped there.

Corpus: `gen_860396e395e946539b6eb1b7411ae54f`, calibration `cal_3d6fb8834b9841c4b5040314`,
threshold 0.731, separability 0.980 [0.952, 1.000], served `trusted`/`calibrated` over stdio.
Claude Code 2.1.238. Agent `anthropic/claude-haiku-4.5`, judge `openai/gpt-4.1-mini`, both through
OpenRouter.

### Primary endpoint

| locus | n | recall | claude_md | delta | 95% CI | p |
|---|---|---|---|---|---|---|
| **memory_only** | 16 | **0.188** | 0.688 | **−0.500** | [−0.750, −0.250] | **0.0078** |
| both *(control)* | 13 | 0.385 | 0.000 | +0.385 | [0.154, 0.692] | 0.0625 |
| claude_md_only *(control)* | 6 | 0.000 | 0.000 | 0.000 | — | undefined |

Discordant pairs on the primary endpoint: **on-only 0, off-only 8**. The RE-call arm never
triggered a trap the baseline avoided.

### Scoring the predictions

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| 1 | off hit rate 0.70 | **0.688** | correct |
| 2 | on hit rate 0.50 | **0.188** | **under-predicted** |
| 3 | reduction 20 points, falsified below 8 | **50 points**, CI excludes zero, p=0.0078 | correct, magnitude under-predicted 2.5x |
| 4 | `recall_search` called in >= 70% | **81%** (13/16) | correct |
| 5 | governing memo retrieved in >= 50% | **50%** (8/16) | correct, exactly at the boundary |
| 6 | abstaining traps worse by 10 to 25 points | 25% vs 0%, **25 points** | correct, at the top of the range |
| 7 | `both` within 10 points either way | **+38.5 points against RE-call** | **falsified** |
| 8 | `claude_md_only` on arm worse by 10 to 30 | both arms 0.000, no hits | not observed |
| 9 | input tokens +40% | **+19%**, p=0.65, not significant | over-predicted |
| 10 | wall time +12 s, of which ~9 s startup | not scoreable, see above | void |
| 11 | `recall_latency_ms` under 1 s | **182 ms** median | correct |

**Prediction 2 is the first time in this project's record that an effect was under-predicted.**
The standing note is eleven of twelve predictions too high by two to four times; here the baseline
was called almost exactly (0.688 against 0.70) and the treatment was called far too pessimistically.

### The falsified control, and what it means

Prediction 7 failed in the direction that matters, and the mechanism is not in doubt. The arms were
verified after the fact: the on arm receives **290 characters** of system prompt, the bare
instruction that a memory tool exists; the off arm receives **17,816 characters** of `CLAUDE.md`
plus `MEMORY.md`. This comparison therefore measures **replacing** the hand-written file with
retrieval, not adding retrieval to it. Where a fact is in `CLAUDE.md`, the arm holding `CLAUDE.md`
wins, and RE-call loses those by 38 points.

That is the honest shape of the result: **retrieval wins decisively on what the file cannot hold,
and loses on what it can.** The configuration a real user would run, `claude_md` **and** `recall`
together, was not measured and is the obvious next arm. No claim is made here about it.

### Cost and quality

| metric | n | recall | claude_md | delta | p |
|---|---|---|---|---|---|
| input tokens | 35 | 75,810 | 63,465 | +12,345 | 0.65 |
| output tokens | 35 | 1,803 | 1,751 | +52 | 0.94 |
| model turns | 35 | 8 | 8 | 0 | 0.62 |
| wall time (salvaged source) | 35 | 67.2 s | 128.1 s | −60.9 s | 0.10 |
| answer_correctness | 35 | 0.211 | 0.167 | +0.044 | 0.43 |
| factual_correctness | 35 | 0.260 | 0.236 | +0.024 | 0.65 |

**No cost or quality difference reaches significance at n=35.** In particular, RE-call did **not**
measurably improve answer correctness as judged by Ragas, and the earlier 3-pair smoke that
suggested an 83% token saving did not survive scale. Both arms score low in absolute terms (~0.2),
which is a property of scoring terse agent answers against detailed written references, not a
statement about either arm.

### What this result supports, and what it does not

It supports: on this corpus, this task set and this model, a calibrated retrieval memory layer cuts
known-hazard mistakes by 50 points where the governing fact exists only in memory, with the
retrieval mechanism confirmed (81% search rate, 50% governing-memo retrieval, 182 ms median).

It does not support: any claim about tokens, wall time or answer quality, none of which reached
significance; any claim about `claude_md` plus `recall` together, which was never run; the ceiling
comparison, which was never run; or generalisation beyond an agent working in the repository whose
memory this is.
