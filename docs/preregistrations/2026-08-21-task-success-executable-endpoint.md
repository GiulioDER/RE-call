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

To be appended after the run. Nothing above this marker may be edited.
