# Handoff: measure whether the WORK is better, not just whether the hazard was avoided

Paste the prompt at the bottom into a fresh session. Everything above it is the context that
prompt refers to, and it is written so the next session does not have to reconstruct any of it.

---

## What already exists, and what it proved

`benchmarks/agent_ab/` is a working paired A/B harness for Claude Code, measured 2026-08-21.
Run `agent-ab-additive-002` compared **CLAUDE.md alone** against **CLAUDE.md + RE-call**, with the
same static bundle byte-for-byte in both arms.

| endpoint | +RE-call | CLAUDE.md | |
|---|---|---|---|
| hazard hit rate, 40 pairs | **0.000** | 0.525 | delta −0.525, CI [−0.675, −0.375], p<0.0001 |
| per-task, 4 traps | 3 of 4 improved | | cluster CI [−0.850, −0.200] |
| control (`both` locus), 12 pairs | 0.083 | 0.083 | delta 0.000, **p=1.0000** |
| search rate, on arm | 85% | — | governing memo retrieved 82% |
| input tokens (median delta) | **+14,904** | | not significant |
| wall time (median delta) | **+1,703 ms** | | not significant |

**What it supports:** a large, reliable reduction in a *specific* expensive failure mode, on tasks
where the governing fact lives in the memory store and not in `CLAUDE.md`.

**What it does NOT support**, and this is why you are here:

- **No general uplift.** Where the fact was already in `CLAUDE.md` the difference was exactly zero,
  and RE-call was *slower* there (+11.3 s median, p=0.027). It stops a hazard; it does not make the
  agent broadly better.
- **It never measured whether the work was any good.** Every task asks for a recommendation and the
  detector scores whether the recommendation is dangerous. Nothing checks whether the agent
  actually did the job: no code was written, nothing was run, no test passed or failed.

## The question this benchmark must answer

**When an agent does real work, does a memory layer make the work succeed more often?**

Success has to be decided by something that cannot be argued with. A judge scoring prose is not
enough here, because the previous run already showed the trap that lies in wait: Ragas
`answer_correctness` found **no significant difference** (+0.044, p=0.43) with both arms near 0.2,
which measures "how much of the reference did the answer echo", not "did the work succeed".

Design the endpoint so that success is **executable**: tests pass, a script exits 0, a file
contains a value, a build completes. If a human has to adjudicate, the endpoint is wrong.

## The hard part, stated honestly

The previous benchmark worked because hazards are cheap to detect: a regex over the transcript for
a command that must not appear. Task success is not like that. You will need, per task:

1. a **real repository state** the agent works in, restored identically for both arms;
2. a **task** whose completion depends on a fact that is in the memory store and not in
   `CLAUDE.md` (otherwise both arms succeed and you measure nothing);
3. an **executable checker** that decides success without reading the transcript.

Point 2 is the one that decides whether the benchmark can show anything. `scripts/agent_ab_qualify.py`
already measures which memory holds a fact, against the real corpus and the real `CLAUDE.md`; reuse
it rather than asserting a locus by hand.

## Reuse, do not rebuild

| file | what it gives you |
|---|---|
| `benchmarks/agent_ab/runner.py` | paired execution, one task at a time |
| `benchmarks/agent_ab/arms.py` | the three arm profiles; `ArmSpec.claude_md_recall` is the additive arm, and it asserts both arms get identical static memory |
| `benchmarks/agent_ab/claude_exec.py` | Claude Code adapter, stream parsing, `SessionRecord` construction |
| `benchmarks/agent_ab/gate.py` | ⛔ admission: a pair is void unless the on arm's `system/init` lists a `mcp__recall*` tool and the off arm's does not |
| `benchmarks/agent_ab/stats.py` | exact McNemar, Wilcoxon, paired bootstrap, per-task clustering; every function returns `None` rather than a number it cannot support |
| `benchmarks/agent_ab/recall_server.py` | `StdioRecallSpec`, the only transport that serves a calibrated corpus |
| `scripts/agent_ab_run.py` | the runner; `--comparison additive` is the one you want |
| `scripts/agent_ab_analyze.py` | endpoints in preregistered order; reports median beside mean |
| `scripts/agent_ab_salvage.py` | rebuild records from transcripts when a run dies |
| `scripts/agent_ab_build_corpus.py` | build the calibrated corpus from scratch, idempotent |

## Environment facts, measured 2026-08-21

- **The agent runs through OpenRouter.** `claude -p` on the subscription token returns
  `401 OAuth access token has been revoked`. Use
  `ANTHROPIC_BASE_URL=https://openrouter.ai/api`, `ANTHROPIC_AUTH_TOKEN=$OPENROUTER_API_KEY`,
  `ANTHROPIC_API_KEY=` (empty). Verified working.
- **The corpus** is `recall-agentab-corpus` on **port 5407**, a container with a **named volume**,
  holding 1006 chunks, generation `gen_f01fc522...`, calibration certified at threshold **0.731**,
  separability 0.980 [0.952, 1.000], served `trusted`.
- **Claude Code 2.1.238.** Below 2.1.221 the CLI does not wait for a pending MCP server and the
  session runs *without the tool while reporting success*.
- **Ragas cannot run on this repo's Python.** Its `pyarrow.parquet` DLL is blocked by a Windows
  Application Control policy and only pyarrow 25 has a cp314 wheel. Use a **3.12 venv**; the pins
  are in `scripts/agent_ab_score_ragas.py`.

## Seven failures already paid for. Do not pay again.

1. **A buried instruction produced a 0% search rate.** Appended after 17,498 characters of
   `CLAUDE.md`, the RE-call instruction was ignored in every session: 16 tools available, zero
   calls. Moved to the FRONT it went to 85%. If your on arm does not search, you are measuring
   prompt placement.
2. **A non-answer scored as avoiding the hazard.** Every detector fires on the presence of a wrong
   thing, so a reply that commits to nothing avoids all of them. `traps.answered()` handles this;
   whatever endpoint you build, ask what a session that does nothing scores.
3. **`grep` in the run command destroyed the diagnostics.** A run died at 71 of 100 sessions and
   its cause is permanently unknown because the invocation was piped through `grep -v`, which
   swallowed stderr and masked the exit code. Redirect to a log file. Use `python -u`.
4. **The session container has no volume.** `scripts/session-db.sh` containers were removed by
   other sessions three times, destroying the corpus each time. That is why the corpus now lives in
   its own container with a named volume.
5. **An MCP `env` block REPLACES the environment.** Without `APPDATA` the server dies on
   `ModuleNotFoundError: anyio`; add it and without `SystemRoot` it dies in Winsock. The server also
   inherits the SESSION's working directory, not the config's `cwd`, so set `PYTHONPATH`. All of
   this is already handled in `StdioRecallSpec`.
6. **A denied tool can hang a task forever.** `docker` is denied in every arm so the shared-database
   hazard can be scored from the denial. One task reached for it, was refused, and burned 75 minutes
   at ~17 seconds of CPU. It was dropped. If a task can loop on a refusal, bound it.
7. **The analyser averaged sessions that never ran.** `records.jsonl` holds every attempted
   session, including gate-discarded ones. Use the gate's verdict from `admission.json`.

## Two statistical rules this project already learned

- **Report the median beside the mean.** On run 002 they disagreed in *sign* on three of four cost
  metrics: input tokens mean −2,293 but median **+14,904**. Means alone would have claimed RE-call
  is cheaper and faster; it is typically dearer and slightly slower, and occasionally saves an
  enormous amount when the baseline goes exploring. The rank-based p-value follows the median.
- **Repetitions of one task are not independent.** With few distinct tasks the per-task view cannot
  reach significance at any effect size (a sign test over 4 bottoms out at p=0.125). Report the
  per-task view as descriptive with a cluster bootstrap, and label the per-pair McNemar as the
  consistency check that overstates confidence. **Get more distinct tasks than four.**

## Standing rules

- **Preregister before measuring**, and commit it: a guard blocks measurement commands while
  anything under `docs/preregistrations/` is uncommitted. Use `/preregister`.
- **Never edit a number in a committed preregistration.** Append corrections underneath.
- **Work in your own worktree**: `scripts/session-space.sh new <short-name>`.
- Predictions in this project run **two to four times too high**. Predict low, and predict the
  mechanism metric beside the outcome.

---

## The prompt

> I want to measure whether adding a memory layer makes an agent's **work succeed more often**, not
> just whether it avoids known hazards.
>
> Read `benchmarks/agent_ab/NEXT-BENCHMARK-TASK-SUCCESS.md` first. It records what the existing
> benchmark measured (a 52.5 point reduction in hazard mistakes, p<0.0001, but exactly zero
> difference on control tasks and no measure of work quality at all), the harness to reuse, the
> environment, and seven failures already paid for.
>
> Build a task-success benchmark on top of `benchmarks/agent_ab/`, comparing **CLAUDE.md alone**
> against **CLAUDE.md + RE-call** as `--comparison additive`.
>
> The endpoint must be **executable**: a test passes, a script exits 0, a file holds the right
> value. Not a judge scoring prose — the previous run already showed that answer-similarity finds
> nothing (+0.044, p=0.43) because it measures echo, not success.
>
> Each task needs a repository state restored identically for both arms, and its success must
> depend on a fact that `scripts/agent_ab_qualify.py` confirms is in the memory store and NOT in
> `CLAUDE.md`. If both arms can succeed from `CLAUDE.md` alone, the task measures nothing.
>
> Aim for **at least 8 distinct tasks**; four was the weakest part of the last design and the
> per-task view could not reach significance at any effect size.
>
> Start by proposing the task set and the checker for each one, and tell me which facts you intend
> to build them on before writing any code. Preregister before measuring.
