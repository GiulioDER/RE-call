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

## 🔁 Second amendment appended 2026-08-27, before stage A ran: the block is cleared

**Nothing above is edited, including the first amendment.** The harness work the amendment above
priced out has been done, so stage A runs. Two things about it change the run condition from what
the frozen predictions assumed, and both are stated here rather than absorbed silently.

### Both arms now run under an isolated `CLAUDE_CONFIG_DIR` instead of `--bare`

Measured against CLI 2.1.238 before the field was added:

| condition | hooks | plugins |
|---|---|---|
| `--bare` alone, hook supplied via `--settings` | **skipped** | 7 loaded |
| `CLAUDE_CONFIG_DIR`, no `--bare` | **fired** | 0 loaded, 0 MCP servers |

So this is not a weakening of `--bare`'s isolation. It is stricter on plugins, and it admits
exactly what the named directory contains. `prepare_hook_config_dirs` writes one directory per
variant and installs the hook in the ON arm only; a reader can diff the two directories and find
exactly one difference. `build_configs` refuses `config_dirs` that do not cover every variant,
because one arm bare and one arm isolated differ in two ways at once and can attribute nothing.

⚠️ **This still changes the environment of the CONTROL arm** relative to every earlier result in
this lane, which ran `--bare`. The change is in the direction of less admitted context, not more,
but "less" is not "identical". Stage A therefore reads the control as well as the treatment, and
**no treatment number is read until the control is shown to behave as the `--bare` control did**.

### What the artifact records

`environment.json` gains `hook_file` and `isolation`. Without them a hooked run and an unhooked one
produce identical artifacts, which is the whole reason the first amendment asked for the resolved
hook set to be recorded the way `instruction_file` already is.

### Two hook defects found while wiring it, both fixed before the run

1. Trace lines carried no session key. Stage A's endpoint is injections **per session** and the
   whole run appends to one file, so the trace would have held a correct total from which no
   per-session number could be recovered. Every line now carries `session_id`, `cwd` and a stamp.
2. A well-formed but non-object event (`[]`, `3`, `"x"` are all valid json without a `.get`) raised
   an `AttributeError` into the session, from a hook whose design rule is that it must never break
   the session it measures.

Mutation coverage is now eight, all eight killed, including one per defect.

**The predictions above stand unscored and unedited.** Stage A is measured against them as written.

## Stage A result, 2026-08-27: BLOCKED again, and this time the block is measured

**Nothing above is edited.** That includes the second amendment's claim that a config dir is
"stricter on plugins" and "admits exactly what the named directory contains", which is **wrong**,
and is left standing because how it was wrong is the useful part: I measured plugins and MCP
servers, saw zeros, and generalised the word "isolation" from the two things I had checked to the
one I had not. `--bare` skips four things. I tested two.

### What ran

`--limit 4 --reps 1`, hook on in the ON arm only, corpus `gen_f01fc522`, 1,006 chunks.
**0 pairs admitted of 4.** One pair completed; the other three died on OpenRouter HTTP 402
("would exceed your available credits given your current in-flight requests"), in **both** arms.
That is a funding and concurrency failure, not a hook failure, and it is not the reason for the
block below.

### Three things that did work, and are worth keeping

| claim | evidence |
|---|---|
| the hook fires inside the sandbox | 2 trace lines, both from the ON arm's sandbox `cwd`, none from the OFF arm |
| the trace can answer a per-session question | every line carries `session_id`, `cwd` and a stamp |
| **PreToolUse `additionalContext` reaches the AGENT** | a hook injecting the nonce `ZQX-NONCE-8823` was quoted back verbatim by the agent, unprompted as to its content |

⚠️ **The third of those nearly went the other way, and the reason generalises.** The injected text
appears **nowhere in the captured stream-json**: not in the tool result, not as a system message,
zero occurrences of the marker or the memo name. Grepping the stream would have scored the
mechanism as broken. `stream-json` does not record system-reminders at all, so for stage B
**"injected text appears in the agent's context" cannot be measured from the transcript** and needs
an in-band probe like the nonce. Prediction 2 is therefore not scored: its endpoint as written is
not observable with the apparatus available.

### ⛔ The block: a config dir admits the developer's CLAUDE.md, and `--bare` does not

One trivial prompt, no tools, one turn, asking the agent to report whether specific documents are
in its context:

| condition | input tokens | user CLAUDE.md | project CLAUDE.md |
|---|---:|---|---|
| `--bare`, outside the repo | 7,875 | no | no |
| `--bare`, in a sandbox | 3,037 | no | no |
| config dir, outside the repo | 47,676 | **yes** | no |
| config dir, in a sandbox | 66,167 | **yes** | **yes** |

Two leaks with different causes: the project file is found by walking up from a `cwd` that lives
inside this worktree, and the user file is found some other way entirely. Two fixes were measured
and **both failed**:

| attempt | result |
|---|---|
| set `HOME` and `USERPROFILE` to an empty directory | 48,758 tokens, user file **still loaded** |
| plant a `CLAUDE.md` inside the config dir, hoping it shadows | 47,837 tokens, **both** loaded; the planted marker came back too, so the config dir's memory is ADDITIVE |

So the config dir cannot be made equivalent to `--bare`, and there is no CLI flag for it:
`--setting-sources` governs settings files, not memory discovery.

### Why this is not simply run anyway

The leak is symmetric, so the A/B stays internally valid, and a run under it would answer a real
question. It would not answer **this** one. The registered control is the `--bare` instruction arm
whose base failure rate is 11 of 48, and the whole power table is built on that number. The repo's
own `CLAUDE.md` documents several of the hazards the `ts-*` tasks are built from, so a control that
receives it is not the control that produced 11 of 48. Running stage B against a base rate measured
under different context, and reporting McNemar against a power analysis computed for the old one,
would be a worse error than not running it.

**The predictions stand unscored.** Prediction 1 (injections per session 8 to 14) has n=1 and that
session was truncated, so it is not read. Prediction 2 is unobservable as written, see above.

### What would unblock it, priced honestly

1. **Move the sandboxes outside the repository.** Closes the project half. Cheap, but it changes
   the ground every earlier result in this lane stands on, so it needs its own parity check.
2. **Run the sessions where no user `CLAUDE.md` exists** — a container, or an account without one.
   Closes the user half. This is the real cost, and it is harness-and-infrastructure work, not
   experiment work.
3. **Or re-register the experiment under the leaked condition**, which means paying for a fresh
   base-rate measurement first, since 11 of 48 does not transfer.

### Re-measure it rather than citing it

The probe is committed, because a claim about a CLI's behaviour has a short half-life and this one
already reversed once inside a day. Four sessions, about two minutes, needs `OPENROUTER_API_KEY`:

```bash
PYTHONUTF8=1 python scripts/agent_ab_isolation_probe.py
```

Run independently from the committed script the same afternoon, it printed 3,171 / 59,580 /
59,573 / 3,164 against the 7,875 / 47,676 / 66,167 / 3,037 in the table above. The absolute numbers
move with what the leaked files currently say; **the two orders of magnitude are the measurement**,
and CLEAN against LEAKED is the part that decides anything.

None of the three is licensed by this record. The hook itself is built, tested (10 groups, 8
mutations killed), verified against the live corpus, and now verified to reach the agent; what
remains is entirely the harness.

## Stage A, second attempt, 2026-08-27: PASSED its gate, and both predictions FALSIFIED

**Nothing above is edited.** The block recorded above is cleared: the `CLAUDE.md` leak was closed,
and the closure is a path rather than a variable.

### What closed it

The user `CLAUDE.md` is found by **walking up from `cwd`**, not through any environment variable.
Every `cwd` in the earlier failed attempts sat under `C:\Users\<user>\`, so the walk always
reached `.claude\CLAUDE.md` no matter what the environment said. `--work-root` puts the sandboxes
outside the profile, and the runner refuses `--hook-file` without it. Verified in-band, in the run
itself, by a real session asked what it holds:

    isolation check: 36,914 input tokens, USER=no PROJECT=no

That check runs against the **control** arm's config directory, and the run stops if it is not
clean. Its verdict is written into `environment.json` rather than only printed.

⚠️ **Still open, and it is the reason the base rate cannot simply be carried over.** A non-bare
session carries ~35,000 input tokens of system prompt against `--bare`'s ~2,900, identical in both
arms. `--disable-slash-commands` moves that by ~1,700 and stops the hook firing, so it is not the
lever. Any stage B under this apparatus reports its own control failure count beside the result.

### The run

`hook-stage-a-003`, `--limit 4 --reps 1 --pair-concurrency 1`, corpus `gen_f01fc522`, 1,006 chunks.
**4 pairs admitted, 0 discarded, 8 of 8 sessions completed.** The hook fired in the ON arm's
sandbox only, 31 times, with zero errors in the trace.

### Prediction 1: injections per session 8 to 14. FALSIFIED, 0 of 4 inside the band

| task | injections |
|---|---:|
| `ts-false-zero-search` | 2 |
| `ts-lf-rewrite` | 2 |
| `ts-raise-on-missing` | 2 |
| `ts-worktree-import` | **25** |

Not a miss in one direction: it is **bimodal**, three sessions far below the band and one far
above. The band came from a recorded median of ~10 payloads a session, and that median described a
different population. This matters beyond the prediction, because the mechanism's whole premise is
forced reach: a session that writes twice gets **two** chances, not ten, and prediction 3's
"rescues 2 to 6 of 11" was reasoned from the ten.

By tool: 22 of 31 injections came from `Bash`, 5 from `Edit`, 4 from `Write`. Median payload 81
characters. **Every injection returned exactly 5 memos**, because the hook has no relevance
threshold, and `project_index.md` was among the three most-injected documents, which is an index
file rather than a memo.

### Prediction 6: input tokens rise 40% to 120%. FALSIFIED, 1 of 4 inside the band

| task | on | off | ratio |
|---|---:|---:|---:|
| `ts-false-zero-search` | 164,111 | 144,568 | 1.14 |
| `ts-lf-rewrite` | 277,198 | 490,657 | **0.56** |
| `ts-raise-on-missing` | 277,185 | 240,994 | 1.15 |
| `ts-worktree-import` | 2,030,340 | 240,335 | **8.45** |

One arm cost **less** than its control and one cost **eight times** it. At n=4 the variance is
session length, not treatment, and the 8.45 is the 25-injection session where five memos of ~1,200
characters compound across every turn. The honest reading is that this endpoint needs the full n,
and that the tail risk is larger than the band allowed for.

### Prediction 2: not scored, because its endpoint is not observable

The injected text appears nowhere in `stream-json` (see the previous section). The **mechanism** is
verified in this exact configuration by an out-of-band nonce: a hook injecting `ZQX-NONCE-8823` from
a sandbox outside the profile had it quoted back verbatim, with the agent reporting no memory
documents. What is not verified is per-session delivery in these four sessions, and no apparatus
available here can verify it from the transcript.

### One defect the run itself exposed

`vocabulary_would_fire` was `null` on all 31 lines, because nothing passed `RECALL_HOOK_VOCAB`. The
decision rule's **BUILD WITH A GATE** cell requires that field, so the gated variant would not have
been derivable from its own evidence, and `null` is also what an untriggered injection looks like.
Fixed before any stage B: `--hook-vocab`, with `scripts/agent_ab_export_hazard_vocab.py` writing
the 3,502 terms at `df<=2`. Smoked on one pair, the field now reads `False`.

### Stage A's gate

> "If the context does not reach the agent, stage B is not run."

It reaches. **Stage B is not blocked.** Two things about it changed, though, and neither is
absorbed silently: the control arm's environment is no longer the `--bare` one that produced 11 of
48, and the per-session injection count is 2 rather than 10 for typical sessions, which is the
mechanism's reach and it is five times smaller than assumed.

## Base-rate run, registered 2026-08-27 BEFORE its result was read

**Nothing above is edited.** Stage A established that the control arm's environment is no longer
the `--bare` one that produced 11 of 48, so that number does not transfer and the power table
built on it is not valid for this apparatus. This measures the replacement first, so stage B's
power is known before stage B is paid for.

### Design

An **A/A run**: `--arms-differ-only-by-hook` with **no hook anywhere**, 8 `ts-*` families x 3
repeats = 24 pairs = **48 control sessions**, every one under exactly the apparatus stage B will
use (non-bare, config-dir isolated, sandboxes outside the user profile, `hazard-query-v2.txt`).

Two things it buys beyond the count. It is a **null control**: two identical arms should show no
systematic difference, and a systematic one would be apparatus bias that stage B would otherwise
attribute to the hook. And its per-arm disagreement rate is the **session-to-session variance**
against which any McNemar result has to be read.

### Three guards were relaxed to run it, each against a stated reason recorded in the artifact

The registered control is "the instruction arm, hook off", not the no-memory arm, and the harness
could not express two arms that differ only by a hook. Each relaxation takes a **sentence**, not a
boolean, because "the arms are identical" is otherwise indistinguishable from the commonest way to
build an experiment that measures nothing:

| guard | why it refused | what replaced it |
|---|---|---|
| the off arm must use an off-arm profile | both arms are the instruction arm by design | a stated reason, recorded in `environment.json`; the guard still holds for every other caller |
| the off arm must have no RE-call tools | the off arm carries RE-call by design | the off arm is judged by the **on-arm** rules, which is stricter than skipping: its servers must still have connected |
| `--hook-file` requires sandboxes outside the profile | unchanged | unchanged, and now also enforced for a hookless config-dir run |

### What I predict

Per `[[i-over-predict-effect-magnitudes]]`, mechanism rates I estimate well and I take the bottom
of any band that flatters a mechanism. The relevant priors are the old 11 of 48 under `--bare`, and
stage A's instruction-plus-hook arm failing 1 of 4.

| # | Prediction | Falsified by |
|---|---|---|
| B1 | **Control sessions failing: 11 to 20 of 48.** The added system prompt is generic CLI instruction, not project hazard knowledge, so it should not rescue a hazard the old apparatus fell into | outside the band |
| B2 | **The A/A arms disagree on 4 to 10 of the 24 pairs.** Identical configurations, so every disagreement is session variance, and this is the floor any McNemar effect has to clear | outside the band |
| B3 | **No systematic direction to that disagreement**: neither arm wins more than 70% of the discordant pairs | one arm takes more than 70% |
| B4 | **Failures concentrate by family rather than spreading evenly**: the worst family contributes at least 3x the best | the spread is flatter |

### What each outcome does to stage B

| control failures | consequence |
|---|---|
| **>= 11** | the registered power table roughly holds; run stage B at 48 pairs as written |
| **6 to 10** | underpowered before it starts: McNemar at 48 pairs could not detect a plausible effect, and stage B is registered at a larger n or not run |
| **<= 5** | ⛔ the apparatus has removed most of the failures the hook was meant to rescue. Stage B measures nothing and is not run; the finding is about the apparatus, not the hook |

⚠️ **B3 is the one that can invalidate everything else.** A systematic winner between two identical
arms means the harness itself favours a side, and no result from stage B could be attributed to the
hook rather than to that.

## Base-rate result, 2026-08-27: 17 of 48, and stage B is powered

**Nothing above is edited.** `base-rate-001`, A/A, no hook anywhere, 8 `ts-*` families x 3 repeats,
**24 pairs admitted, 0 discarded, 48 sessions, every one with a checker verdict.**

| # | prediction | observed | verdict |
|---|---|---|---|
| B1 | control failures 11 to 20 of 48 | **17 of 48** | **confirmed** |
| B2 | A/A arms disagree on 4 to 10 of 24 pairs | **3 of 24** | **falsified, below the band** |
| B3 | neither arm takes more than 70% of discordant pairs | 1 on / 2 off, **67%** | not falsified, but n=3 makes it weak |
| B4 | worst family at least 3x the best | **0% to 100%**, ratio unbounded | **confirmed, far past the band** |

### B1 puts stage B back on its registered footing

17 control failures under this apparatus against 11 of 48 under `--bare`, so the added system
prompt did not rescue the hazards, and the power table's arithmetic is unchanged: McNemar at these
counts still needs roughly **6 rescues with no regressions** (p about 0.03) and cannot see 4.

### B2 is the useful surprise, and it cuts in the experiment's favour

Two IDENTICAL arms disagreed on only 3 of 24 pairs. I predicted 4 to 10. So the tasks are close to
deterministic and the **noise floor a treatment has to clear is low**, which is the opposite of the
usual reason a paired agent benchmark fails to show anything. It also means a 6-rescue effect would
be plainly visible rather than buried in variance.

⚠️ The same fact bounds the ceiling: a task that fails 6 times out of 6 fails for a stable reason,
and a stable reason is either fixed by the injected memo or not. There is little room for a
partial effect.

### B4 is the finding that changes how stage B should be read

| family | control failures |
|---|---|
| `ts-lf-rewrite` | **6 of 6** |
| `ts-bounded-runner` | 5 of 6 |
| `ts-false-zero-search` | 5 of 6 |
| `ts-autouse-tmp-path` | 1 of 6 |
| `ts-raise-on-missing` | 0 of 6 |
| `ts-sample-covers-tail` | 0 of 6 |
| `ts-separator-canary` | 0 of 6 |
| `ts-worktree-import` | 0 of 6 |

**Four of the eight families never fail.** Half of stage B's 48 pairs are therefore spent on tasks
that cannot contribute a rescue, and the whole effect has to come from the other half. Stage B is
still run **exactly as registered** rather than narrowed to the failing families, because choosing
the task set after seeing which ones fail is selection, and the registered n is what the power
table describes. The consequence is stated here instead: the effective sample for detecting a
rescue is about **24 pairs, not 48**, and a null result has to be read against that.

### What was re-run, and why, stated because it could bias the number

Four attempts were needed to admit all 24 pairs. Nothing was re-run for its RESULT: every re-run
was a pair the pre-specified admission gate had already refused, and the refusals were

- **6 sessions** where the `recall-memory` MCP server failed to connect or the stream carried no
  result event, which are transient apparatus failures, and
- **3 sessions** killed by a bug in the recording layer, not the agent: `SessionRecord` refused to
  be CONSTRUCTED for an off arm that had called RE-call, which in the shared-arm design is legal.
  Those sessions had completed and produced checker verdicts, and were recorded as `the session
  did not complete`. ⛔ Note the direction: the sessions lost were exactly the ones where the agent
  **searched memory**, so leaving the bug in place would have biased the base rate in whichever
  direction memory helps. Fixed before the final attempt.

The first attempt's records are preserved beside the final ones as `records.attempt1.jsonl`.

## Stage B result, 2026-08-27: 6 rescues, 1 regression, p = 0.125, and it does NOT ship

**Nothing above is edited.** `stage-b-001`, both arms the instruction arm, hook in `recall_on`
only, sandboxes outside the user profile, isolation verified in-band (`USER=no PROJECT=no`).

⚠️ **STOPPED AT 34 OF 48 PAIRS**, on the operator's instruction, because the OpenRouter balance ran
out mid-run. The run was **not** stopped because of what the data showed, and no interim result was
read before the decision. Two families never ran at all (`ts-autouse-tmp-path`, `ts-separator-canary`)
and `ts-sample-covers-tail` got 4 pairs of 6. Everything below is a partial run and is reported as
one.

### The result

|  | control (hook off) | treatment (hook on) |
|---|---:|---:|
| failures | **17 of 34** | **12 of 34** |

| | count | tasks |
|---|---:|---|
| **rescues** (control failed, hook passed) | **6** | `ts-bounded-runner` r1 r2 r4, `ts-false-zero-search` r3 r6, `ts-sample-covers-tail` r3 |
| **regressions** (control passed, hook failed) | **1** | `ts-sample-covers-tail` r2 |

**McNemar exact, two-sided: p = 0.125** on 7 discordant pairs. Net **+5**.

### ⛔ It lands in the BUILD cell and it still does not ship

The decision table's `>= 6 rescues` x `0-1 regressions` cell reads **BUILD**. It does not ship,
because the same record committed in advance that **"nothing ships on a non-significant positive"**,
and p = 0.125 is not significant.

🔑 **The cell boundary and the significance commitment disagree, and that is a defect in the table
rather than a close call.** The `>= 6` boundary was drawn where 6 rescues and **zero** regressions
reach p about 0.03. One regression moves the identical cell to p = 0.125, and the cell alone cannot
see it. A decision rule written on counts, justified by a power calculation on a different
quantity, will sooner or later ship on evidence it was designed to refuse. The scorer now applies
the commitment explicitly and prints the override.

### Scoring the frozen predictions

| # | prediction | observed | verdict |
|---|---|---|---|
| 3 | rescues 2 to 6 | **6** (of 17 control failures) | **confirmed**, at the top edge |
| 4 | regressions 0 to 4 | **1** | **confirmed** |
| 5 | net 0 to +5 | **+5** | **confirmed**, at the top edge |
| 6 | input tokens +40% to +120% | **-1% aggregate**, median ratio 1.01, only 5 of 32 pairs in band | **falsified** |

Prediction 6 is the interesting miss. Ten mostly-irrelevant injections a session was supposed to be
real context pressure; the hook actually fired a **median of 3 times** (123 injections across 34
sessions, max 9), and five memos of about 1,200 characters against sessions of roughly 300,000
cumulative input tokens is noise. **The context cost I priced as the main risk is not the risk.**
Note this is the second time the 8-to-14 band has been falsified downward: the mechanism's reach is
persistently about a third of what the design assumed.

### Where the effect is, and where it is not

| family | pairs | control failures | hook failures |
|---|---:|---:|---:|
| `ts-bounded-runner` | 6 | 4 | **1** |
| `ts-false-zero-search` | 6 | 6 | **4** |
| `ts-lf-rewrite` | 6 | **6** | **6** |
| `ts-sample-covers-tail` | 4 | 1 | 1 |
| `ts-raise-on-missing` | 6 | 0 | 0 |
| `ts-worktree-import` | 6 | 0 | 0 |

⛔ **`ts-lf-rewrite` fails 6 of 6 in BOTH arms.** It is the family the base rate flagged as the
hardest, its governing memo is in the corpus, and the hook put memos in front of the agent on every
write. Zero rescues. That is the clearest evidence in this lane that for some failures the loss is
**downstream of the memo arriving**, which is the outcome the decision table called the most
informative of its nine.

### The gated variant, now derivable

The hook recorded `vocabulary_would_fire` on every injection without acting on it: **57 of 123
injections (46%)** would have passed the `df<=2` trigger. So the gated variant is an offline
re-analysis rather than a second A/B, exactly as intended. It is not run here, because with 7
discordant pairs a subset analysis would be noise.

### What this run does and does not license

It **does not** license shipping. It **does** license finishing: 34 pairs at this rate extrapolate
to roughly 8 or 9 rescues at the registered 48, which would be significant, and the run stopped for
budget rather than for evidence. Completing the remaining 14 pairs is the cheapest informative next
step, and it is the same registered run rather than a new one.
