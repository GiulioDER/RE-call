# Pre-registration: a write-time hook, and the terminal question for this lane

**Date:** 2026-08-27   **Status:** predicted, not yet built, not yet measured
**Probe:** `scripts/write_time_recall_hook.py` + `scripts/write_time_recall_hook_tests.py`,
committed with this record; driven through the existing A/B harness.

## Why this is the last shape, and why it is terminal

The chain is now measured end to end, and each link located the failure one step further along:

| question | answer | record |
|---|---|---|
| Can authoring make a memo findable? | **No.** 0 of 14, four ways | authored-surfaces |
| Can retrieval find the memo at all? | **Yes.** 11 of 11 sessions that needed it | checker-ground-truth |
| Can a threshold or fusion aim it? | **No.** 0 viable points, 0 of 12 variants | threshold / fusion frontiers |
| Can an instruction make the agent query that way? | **No.** 0.067 adoption; it searched once, and composed | deliberate-draft-search stage 1 |

So: **do it mechanically, without asking the agent to remember.** A `PreToolUse` hook on
`Write`/`Edit`/`Bash` takes the payload, queries the corpus with it, and injects any hit as
`hookSpecificOutput.additionalContext` — the same mechanism `session_start_hook.py` already uses.
Reach becomes **1.00 by construction**, which is the one thing nothing else in this lane could
guarantee.

🔑 **That makes this terminal.** If guaranteed reach does not change outcomes, then memory at write
time is dead *regardless of retrieval quality*, and the remaining loss is downstream of retrieval,
downstream of the gate, and downstream of the agent reading the memo. That is a finding about
agents rather than about retrieval, and it is publishable either way.

## What it inherits, stated plainly

The per-write design was killed for a reason this hook does not escape: **the memo is needed in 11
of 48 sessions (23%), and draft-time search fires on 29 of the 36 that do not need it.** At a median
of 10 payloads per session, the hook injects roughly ten times, most of them irrelevant.

The instruction-based design at least left the agent free to ignore its own search. A hook removes
that choice and spends context on every write. **So the honest hypothesis is two-sided**: forced
reach may rescue failures, and forced noise may cause them. This record measures both and reports
the net, rather than reporting rescues and calling it a result.

## Design

**The hook.** Fires on `Write`, `Edit`, `NotebookEdit`, `Bash`. Query = the payload
(`content` / `new_string` / `command`), truncated at `MAX_QUERY_CHARS = 4096` to match the server's
own refusal. Lexical leg, top-5, against the frozen benchmark corpus. It **never blocks**: it
returns `additionalContext` and allows. Blocking would change outcomes for reasons unrelated to
memory quality and would make the endpoint uninterpretable.

**Recorded but NOT applied**, so a follow-up can sweep offline without paying for another run
(the "collect once" discipline that closed the calibration question in seconds): for every
injection, whether the `df<=2` vocabulary trigger would have fired, the hit sources, and the
scores. A gated variant is then an offline re-analysis rather than a second A/B.

**Stage A — mechanical smoke, 4 sessions.** Does the hook fire on every write, does the context
reach the transcript, does the session still complete? Endpoints: injections per session, and
whether the injected text appears in the agent's context. **If the context does not reach the
agent, stage B is not run** — that is a plumbing failure, not a result.

**Stage B — paired A/B, 48 pairs** (8 `ts-*` families x 6 repeats). Control: the current
`hazard-query-v2.txt` instruction arm, hook off. Treatment: same instruction, hook on. Endpoint:
checker pass, McNemar on discordant pairs.

## ⚠️ Power, unchanged and still unflattering

Base failure rate 11 of 48. At 48 pairs McNemar detects **"rescues at least 6 of 11"** and nothing
smaller (6 → p≈0.03; 4 → p≈0.13, not significant). This record commits in advance: an effect of 3
or 4 is reported as **"no detectable effect at n=48"**, which is not "no effect", and nothing ships
on a non-significant positive.

## What I predict

Per `[[i-over-predict-effect-magnitudes]]`: mechanism rates are the class I estimate well; benefits
get the bottom of the band. The most relevant prior is that the task-success A/B measured **+0.154
with a CI crossing zero at 0.674 reach** — reach and effect are different things, and this record
exists to find out whether that gap closes at reach 1.00.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Stage A: injections per session 8 to 14**, matching the recorded median of ~10 payloads | outside the band |
| 2 | **Stage A: the injected text reaches the agent's context in 4 of 4 sessions.** Mechanical, so anything less is a plumbing defect | fewer than 4 |
| 3 | ⚠️ **Stage B rescues 2 to 6 of the 11 failures** — at or below the edge of detectability. Retrieval already reaches all 11; the gap is whether the agent acts | outside the band |
| 4 | **Stage B regressions 0 to 4.** Ten mostly-irrelevant injections per session is real context pressure, and this is the first design that spends it without the agent's consent | 5 or more |
| 5 | **Net (rescues minus regressions) is 0 to +5**, i.e. plausibly zero | outside the band |
| 6 | **Input tokens per treatment session rise by 40% to 120%** against control | outside the band |
| 7 | **Cost: stage A under 20 minutes; stage B roughly the prior 112-session run** | stage A over 20 minutes |

## Decision rule, cross product over rescues and regressions

| rescues \ regressions | `0-1` | `2-4` | `>= 5` |
|---|---|---|---|
| **`>= 6`** | **BUILD**: forced reach works and costs little; preregister a held-out confirmation before shipping | **BUILD WITH A GATE**: apply the recorded vocabulary trigger offline, re-derive, and register the gated variant | **KILL**: it rescues and it damages; a mechanism that must injure to help is not a memory feature |
| **`3-5`** | **UNDERPOWERED**: report as "no detectable effect at n=48" and register the larger run the power table demands. Do NOT ship | **UNDERPOWERED**, and the net is likely zero; register the larger run only if someone will pay for 4x the sessions | **KILL** |
| **`<= 2`** | ⛔ **KILL, and this is the terminal outcome.** Reach is 1.00, the memo is in front of the agent on every write, and it still falls in. The loss is downstream of memory entirely | **KILL** | **KILL** |

All three `<= 2` cells end the lane. That outcome is the most informative of the nine, because it
would be the first measurement in this project to locate the failure *after* the memo has reached
the agent — which no amount of retrieval work can address.

## What I already know

- Draft queries surface the governing memo for 11 of 11 sessions that needed it, and for 29 of 36
  that did not (executed checker ground truth, no judge).
- The vocabulary trigger moves false fires 0.806 → 0.722 while costing 1 of 11.
- The instruction arm reached the memo in 31 of 48 sessions and produced +0.154 on task success
  with the CI crossing zero.
- Median 10 payloads per session, max 24; 3 of 501 exceed the 4,096-character query limit.

## Confounds I can name now

1. **The hook spends context the control does not.** Any regression could be context pressure
   rather than bad advice, and prediction 6 exists so the two can be told apart afterwards.
2. **Injected text is not the same as attended text.** Stage A verifies the context arrives; no
   part of this design verifies the agent read it, and a null cannot distinguish "read and ignored"
   from "not attended".
3. **The hook adds latency on every write** — one retrieval per payload. Reported per arm.
4. **Same 8 families as everything else in this lane**; a held-out confirmation is inside the BUILD
   branch, not an afterthought.
5. **The hook queries the frozen benchmark corpus**, not a live memory store, so this measures the
   mechanism and not the deployed configuration.
6. **`Bash` payloads include read-only commands** (`ls`, `cat`), which cannot benefit and will
   inject noise. Not filtered, deliberately: filtering is the trigger question, already measured,
   and recorded here rather than applied.

<!-- frozen_above -->

## 🔁 Amendment appended 2026-08-27, before stage A ran: the harness cannot host this hook

**Nothing above is edited.** Two things measured after the record was committed.

### The hook itself works, verified end to end against the live corpus

`gen_f01fc522`, 1,006 chunks, the corpus the whole lane is anchored to:

| case | result | latency |
|---|---|---|
| a hazard draft, `version_file.write_text(content, encoding="utf-8")` | **injected**, top hit `python-write-text-crlf-churn` at rank 1 | 0.99s |
| an innocuous `ls -la scripts/ \| head -20` | **injected**, top hit `clean-automerge-is-the-dangerous-case` | 0.98s |
| `ls` | no output, the `MIN_QUERY_CHARS` guard held | 0.15s |

The second row is the registration's premise happening live: the hook fires on a directory listing
and injects an irrelevant memo. That is the 29-of-36 noise rate, not a defect.

Three defects were found and fixed by reviewing and smoke-testing the hook before this: an ~11s
embedder load per write (one process per tool call, so ~110s a session — replaced with `ts_rank`
in SQL), a 10s `connect_timeout` on every write when the corpus is down (now 2s, measured 10.9s →
3.0s), and raw SQL that did not bind the ACTIVE generation and so would have served retired rows.

### ⛔ But stage A cannot run on this harness as designed

`benchmarks/agent_ab/claude_exec.py` runs every arm with `--bare`, and its docstring states why:

> "`bare` defaults to True and is what makes the arms comparable: `--bare` skips hooks, plugins,
> auto memory, MCP auto-discovery and `CLAUDE.md`, so context reaches the session only through the
> flags recorded here. Without it, whatever happens to sit in the working directory or in
> `~/.claude` joins the experiment as an unrecorded variable."

**So the harness skips hooks by design, and that design is the reason its results are trustworthy.**
Running this experiment means either turning `--bare` off — which admits every hook, plugin and
MCP server on the machine into the comparison as unrecorded variables, and would invalidate the A/B
far more seriously than the hook could inform it — or teaching the harness to admit exactly one
named hook while still excluding everything else.

**Stage A is therefore BLOCKED, not failed**, and this record does not treat the blockage as a
result. The predictions above stand unscored.

### What unblocking costs, stated so the decision is informed

A selective-hook capability in `ClaudeExecConfig`: a `hooks_file` that is written into an isolated
settings file for that session only, passed alongside `--bare`, with the session's resolved hook
set recorded in `environment.json` the way `instruction_file` already is. That is real harness
work, it touches the component whose isolation guarantees every other result in this lane, and it
needs its own tests — a harness that silently admits a second hook would corrupt runs without
failing.

**It is not licensed by this record.** Whoever picks it up should note that the hook is already
built, tested (9 groups, 6 mutations killed) and verified against the live corpus, so the only
remaining work is the harness, and that the noise premise is now demonstrated rather than
predicted.
