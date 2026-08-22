# Does adding RE-call to CLAUDE.md make the WORK succeed more often?

**Written 2026-08-21, before any session of this run was executed, and after the task set was
qualified and committed** (`09aa03f0`). Run identifier `agent-ab-tasksuccess-001`.

## The question, and why the last run could not answer it

Run `agent-ab-additive-002` measured a **52.5 point reduction in hazardous recommendations**
(0.000 against 0.525, p<0.0001) with an exactly zero difference on control tasks. It is a real
result about a real failure mode, and it says nothing about whether the agent did any work well,
because every task asked for a recommendation and every detector was a regex over the transcript.
Nothing was written, nothing was run, no test passed or failed.

Answer similarity was tried there and found nothing: Ragas `answer_correctness` gave **+0.044,
p=0.43**, both arms near 0.2. That is what happens when the endpoint measures how much of a
reference the reply echoed rather than whether the work succeeded.

So this run's endpoint is **executable**. Each task restores a repository state, asks for an
artifact, and a checker runs that artifact and returns a boolean. No judge is involved in the
primary endpoint at all.

## The design rule, stated before the result

⛔ **The oracle is not in the sandbox.** A test the agent can run is a test the agent iterates
against until green, and then both arms score 1.0 and the benchmark measures persistence rather
than knowledge. Every task here is decided by an input the session never sees: a poisoned fixture
staged after the fact, a mutation applied to the module under test, a decoy package placed on the
path, a count of planted files nobody was told.

The corollary chose the tasks: **the naive answer must fail silently.** A false zero from a mangled
search, a diff that looks like one line and is four hundred, a mean computed over fabricated
defaults. An answer that announces itself gets fixed by either arm, and the fact stops
discriminating.

## Fixed configuration

| Field | Value |
|---|---|
| Comparison | `--comparison additive`: `claude_md` against `claude_md` + RE-call |
| Tasks | `benchmarks/agent_ab/tasksuccess.py`, 10 tasks |
| Primary | 8 `memory_only` tasks x 6 repetitions = **48 pairs** |
| Controls | 2 tasks (`both`, `claude_md_only`) x 4 repetitions = **8 pairs** |
| Qualification | `benchmarks/agent_ab/task-qualification.json`, committed at `09aa03f0` |
| Agent | `anthropic/claude-haiku-4.5` via OpenRouter, `--bare` |
| Tools | `Read, Grep, Glob, Bash, Write, Edit`, identical in both arms; `docker` denied in both |
| Corpus | `recall-agentab-corpus` on 5407, tenant `default`, generation `gen_f01fc522`, calibrated, trust_state `trusted` |
| Transport | stdio, `RECALL_ENV=production`, strict trust |
| Static bundle | `CLAUDE.md`, 11,773 chars, byte-identical in both arms |
| Repository revision | `09aa03f0` on `claude/agent-memory-task-success-9d24f5` |
| Base | branched from `f4c24346`, NOT merged with `origin/master` (they have diverged 13/21) |
| Claude Code | 2.1.238 |

## The task set, and the five candidates that were dropped

Eight primary tasks, each qualified `memory_only` against the live corpus over stdio. The governing
memos are `msys-mangles-slash-patterns`, `python-write-text-crlf-churn`,
`benchmark-scripts-import-the-main-checkout`, `missing-input-becomes-a-clean-null`,
`subprocess-timeout-does-not-bound-wall-clock`, `sorted-sample-plus-early-stop-is-head-bias`,
`pasted-unicode-separators-rot` and `autouse-fixtures-must-not-write-into-tmp-path`.

Two controls: `ctl-lint-only-check` qualified `both`, `ctl-stage-by-pathspec` qualified
`claude_md_only`.

**Five candidate tasks were dropped before this document was written**, and they are recorded in
`tasksuccess.DROPPED_BEFORE_MEASUREMENT` with their evidence. Three failed the discrimination test
in `tests/test_agent_ab_tasksuccess.py`, two of them because the naive answer PASSED, which review
does not catch. Two failed qualification: their memo is in the corpus and does not come back for
the question the task provokes. The probe queries were left as first written rather than reworded
until the memo surfaced.

## Predictions

Written knowing that eleven of twelve registered predictions in this project have been too high, by
two to four times. These are deliberately low, and the mechanism metric is predicted beside the
outcome.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Primary success rate: off 0.30, on 0.50, delta +0.20** | the paired CI for the delta includes 0 |
| 2 | **Control delta 0.00**, not significant | the control delta is significant, which makes the primary a harness artefact rather than a memory effect |
| 3 | **Search rate on the on arm >= 0.80** | below 0.50, in which case this measures prompt placement again and not retrieval |
| 4 | **Governing memo in the on arm's retrieved sources on >= 0.60 of on-arm sessions** | below 0.40 |
| 5 | **>= 5 of 8 primary tasks improve** | 4 or fewer |
| 6 | **On arm dearer: median input tokens +10,000** | the median delta is negative |
| 7 | **On arm slower: median wall time +5 s** | the median delta is negative |

Prediction 1 is the headline and I expect to be wrong about its size rather than its sign. The
ceiling here is not 1.0: it is search rate times retrieval times correct application, and a session
that retrieves the memo can still write the wrong script.

## Endpoints, in the order they will be reported

1. **Primary: task success rate on the 8 `memory_only` tasks.** Reported two ways.
   - **Per-task (headline).** One rate per task, repetitions collapsed. Eight distinct tasks, so a
     sign test can reach p=0.008, which four could not reach at any effect size. Reported with a
     cluster bootstrap that resamples tasks.
   - **Per-pair (consistency check).** Exact McNemar over all 48 pairs. This produces the smaller
     p-value and **overstates confidence**, because repetitions of one task are correlated. It is
     secondary and labelled as such wherever it appears.
2. **Control success rate** on the 2 tasks whose fact is in `CLAUDE.md`.
3. **Mechanism:** search rate, governing-memo retrieval rate, RE-call call count.
4. **Cost:** input tokens, output tokens, wall time, model turns. **Median reported beside mean.**
   On run 002 they disagreed in sign on three of four cost metrics, and means alone would have
   claimed RE-call is cheaper and faster when it is typically dearer and slightly slower.

## Exclusions, and what is not allowed after seeing the numbers

- A pair is **discarded, not scored zero**, when the admission gate cannot prove the treatment was
  applied: the on arm's `system/init` must list a `mcp__recall*` tool and the off arm's must not.
  The discarded count is published beside the result.
- A pair is **discarded** when the two arms did not start from the same sandbox tree digest.
- A checker that raises scores the session as failed and is counted separately, so "the artifact
  did not work" and "the checker broke" stay distinguishable.
- **No task may be removed, no checker changed, no threshold moved, and no repetition added after
  inspecting results.** Any exploratory rerun gets a new run identifier and a new preregistration.
- Numbers in this document are never edited. Corrections are appended below the marker.

## Known limitations, stated in advance

- **Eight tasks is enough for a sign test and is not many tasks.** The per-task view is the honest
  one and it rests on eight points.
- **The tasks are Windows-specific.** Four of the eight facts are true because of MSYS, Windows
  line endings, Windows process semantics or this machine's Python. The result is about a memory
  layer carrying machine-specific facts, which is the case it should be strongest in.
- **The checkers execute model-written code.** Each is bounded by a wall-clock tree kill and runs
  with the sandbox as its working directory.
- **This is not a blind run.** I built the tasks, wrote the checkers and know which answer each
  memo produces. The protection is that the checkers are mechanical, the task set was qualified and
  committed before these predictions, and both are in git history.

<!-- frozen_above -->

## Result

Measured 2026-08-22, run `agent-ab-tasksuccess-001`. 112 sessions, **54 pairs admitted, 2
discarded** (one `api_error: Connection lost mid-response`, one MCP server that failed to start).
Zero checker errors, zero digest-parity failures, zero RE-call calls in the off arm.

The run died once at 46 of 112 sessions when the process that launched it exited. `--resume` was
added and the 23 completed pairs were carried forward; no task, checker, threshold or repetition
count changed. Nothing above the marker was edited.

### The headline: no significant uplift

| view | off | on | delta | |
|---|---|---|---|---|
| **per-task, 8 tasks (headline)** | | | **+0.154** | cluster CI **[−0.042, +0.371]**, sign test **p=1.0000** |
| per-pair, 47 pairs (overstates confidence) | 0.574 | 0.723 | +0.149 | p=0.065 |
| control, 7 pairs | 1.000 | 1.000 | 0.000 | no discordant pairs |

3 tasks improved, 2 got worse, 3 unchanged. The cluster interval includes zero and the sign test
is 1.0000. **On this task set, adding RE-call to `CLAUDE.md` did not make the work succeed
significantly more often.**

| task | off | on | delta | searched | memo retrieved |
|---|---|---|---|---|---|
| `ts-autouse-tmp-path` | 0.33 | **1.00** | +0.67 | 5/6 | **5/5** |
| `ts-false-zero-search` | 0.17 | **0.67** | +0.50 | 4/6 | 2/4 |
| `ts-bounded-runner` | 0.00 | **0.40** | +0.40 | 3/5 | **3/3** |
| `ts-separator-canary` | 1.00 | 1.00 | 0.00 | 6/6 | 4/6 |
| `ts-worktree-import` | 1.00 | 1.00 | 0.00 | 2/6 | 0/2 |
| `ts-lf-rewrite` | 0.00 | 0.00 | 0.00 | 3/6 | **0/3** |
| `ts-raise-on-missing` | 1.00 | 0.83 | −0.17 | 1/6 | 0/1 |
| `ts-sample-covers-tail` | 1.00 | 0.83 | −0.17 | 1/6 | 1/1 |

### Predictions, scored

| # | prediction | measured | verdict |
|---|---|---|---|
| 1 | off 0.30, on 0.50, delta +0.20, CI excludes 0 | off **0.574**, on 0.723, +0.149, CI **[−0.042, +0.371]** | **FALSIFIED** |
| 2 | control delta 0.00, not significant | 1.000 against 1.000, no discordant pairs | **CORRECT** |
| 3 | search rate >= 0.80 | **0.532** | **FALSIFIED** (above the 0.50 disaster line) |
| 4 | governing memo retrieved on >= 0.60 that searched | **0.600** of 25 | **CORRECT** |
| 5 | >= 5 of 8 tasks improve | **3** | **FALSIFIED** |
| 6 | median input tokens positive (predicted +10,000) | **+55,959**, p=6.4e-07 | **CORRECT**, 5.6x under-predicted |
| 7 | median wall time positive (predicted +5 s) | **+24.5 s**, p=9.9e-06 | **CORRECT**, 5x under-predicted |

Cost, median beside mean as required: input tokens mean +69,826 median +55,959; wall time mean
+32,332 ms median +24,513 ms; output tokens median +197 (p=0.11); model turns median +1 (p=0.07).
The two arms agree in sign here, unlike run 002.

⚠️ **A new error pattern.** Eleven of twelve previous predictions in this project were too high. Here
the three about *benefit* were too high again, and the two about *cost* were too LOW by a factor of
five. The prior "predict a quarter to a half of the ceiling" applies to effects, not to overheads.

### Why it is null, which is the useful part

**The layer worked when it fired, and it did not fire often.** The three tasks that improved are
the three where the governing memo actually came back: `ts-autouse-tmp-path` 5/5, `ts-bounded-runner`
3/3, `ts-false-zero-search` 2/4. Search rate over all on-arm sessions was 0.532, and of those that
searched, 60% retrieved the governing memo. So the memo reached the agent in roughly **32%** of
on-arm sessions.

**The mechanism of the miss is that you cannot look up a hazard you do not know exists.**
`ts-lf-rewrite` is the clean case: 3 of 6 sessions searched, and the memo came back **zero** times,
because all three asked about the task rather than the hazard.

| what the agent asked | what came back |
|---|---|
| "version bump script versioning conventions" | `recall-public-distribution`, `published-numbers-need-artifact-markers`, … |
| "version bumping script CI release automation" | `mutation-scripts-revert-concurrent-edits`, `read-a-pr-against-its-merge-base`, … |
| "version bump release versioning script" | `external-state-in-generated-assets`, `pytest-sessions-cannot-run-in-parallel`, … |

The qualifier asked the **symptom** question ("why does a file edited by a python script show as
modified with no content change") and `python-write-text-crlf-churn` came back, which is why the
task qualified `memory_only`. The agent asks the **task** question, and it does not. All twelve
sessions, both arms, wrote CRLF into a tree with `eol=lf`.

That gap between the probe query and the agent's query is the single largest leak in this design,
and it was invisible to a qualification step that used my query instead of theirs.

### Post-hoc, and confounded: success conditioned on retrieval

Not preregistered. **The subsets are not randomised**, and they are confounded with task
difficulty: the tasks that searched most also had the most room, and two of the three tasks that
never needed to search were already at ceiling. Reported because it is the most informative cut,
not as an effect estimate.

| on-arm sessions where… | pairs | off | on | delta |
|---|---|---|---|---|
| the memo was retrieved | 15 | 0.53 | 0.80 | +0.27 |
| it searched and missed | 10 | 0.50 | 0.60 | +0.10 |
| it never searched | 22 | 0.64 | 0.73 | +0.09 |
| all (preregistered) | 47 | 0.57 | 0.72 | +0.15 |

### Validity: half the task set had no room to move

- **Three tasks were at ceiling in both arms** (`ts-separator-canary`, `ts-worktree-import`, and
  effectively `ts-raise-on-missing` and `ts-sample-covers-tail` at off=1.00). The model writes
  better code unaided than the naive reference implementations the checkers were validated
  against, so the fact was not needed.
- **One task was at floor in both arms** (`ts-lf-rewrite`, 0/12). The checker is not at fault: the
  informed reference passes it, and the failing sessions produced a *correct* 2-line content diff
  while leaving 27 carriage returns in the file, which is precisely the hazard. `git diff --numstat`
  hid it and the byte check caught it.
- That leaves **four tasks with genuine room, of which three improved**. Stated as an observation,
  not as a headline: choosing that subset after seeing the result is the thing this document exists
  to prevent.

### What this supports, and what it does not

**Supported:** on tasks where the governing fact reached the agent, the work succeeded more often
(+0.27 in the retrieved subset, +0.40 to +0.67 on the three tasks that moved). The control is
clean at 1.000 against 1.000, so nothing here is a harness artefact. The cost is real, large and
significant: about +56,000 input tokens and +24 seconds per session, medians.

**Not supported:** any claim that installing a memory layer makes an agent's work broadly better.
The preregistered endpoint is +0.154 with an interval spanning zero, and I am reporting it as a
null. The binding constraint is not retrieval quality but **retrieval initiation and query
formulation**: the agent searches half the time, and when it does it asks about its task rather
than about the hazard it does not know it is walking into.

The obvious next experiment, which needs its own preregistration and must not be grafted onto this
one: does an instruction scoped to "before writing any script, search for hazards in what you are
about to do" close the gap, and what does it cost in tokens to search on every task?
