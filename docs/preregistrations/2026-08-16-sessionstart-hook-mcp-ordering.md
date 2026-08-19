# Pre-registration: does a SessionStart hook that writes `.mcp.json` reach the same session?

**Date:** 2026-08-16   **Status:** MEASURED 2026-08-18, prediction confirmed (see the final section; the prediction text itself is unedited)

## The question

When a checkout has no `.mcp.json` and the newly installed SessionStart hook generates one, are the
`recall` and `recall-memory` MCP tools available to **that same session**, or only to the next
session started in that checkout? Yes or no, on the current session.

## What I predict

**No.** The tools will not be present in the session whose own hook created the file. I expect the
MCP client to resolve its server list during client initialisation, before user-configured
`SessionStart` hooks are executed, so the hook writes a file that has already been read as absent.

Confidence: roughly 70%. The uncertainty is real, because both the hook and the MCP connection
happen inside the same "start the session" phase and nothing documents their order relative to each
other. I can construct a plausible story for either.

Secondary prediction, if the first is wrong and tools DO appear: they appear only after an approval
prompt for project-scoped servers, so the win would not be automatic in a fresh checkout anyway.

## What would falsify this

A fresh session, started in a checkout with no `.mcp.json`, that lists `recall` or `recall-memory`
among its available tools. One such session falsifies it; the claim is about whether the race is
winnable at all, so a single win is decisive and a single loss is not conclusive on its own (which
is why n > 1 below).

## How it will be measured

n = 3 fresh headless sessions, in 3 freshly created worktrees off `origin/master`, each with no
`.mcp.json` on disk at start.

```bash
scripts/session-space.sh new mcp-race-<i>
cd .claude/worktrees/mcp-race-<i>
claude -p "List the names of every MCP server you can call tools on. Names only."
```

Metric, by name: **fraction of fresh sessions (denominator = 3 sessions started) in which the
`recall` MCP server appears in the session's own tool inventory.** Predicted: 0/3.

Apparatus check, because predicting an outcome does not reveal a broken harness: before the three
runs, start one session in a worktree where `.mcp.json` **already exists** and confirm `recall`
appears (expected 1/1). If that control fails, the probe cannot detect the tools at all and a 0/3
result would mean nothing. This is the case whose answer I already know.

Second apparatus check: after each run, confirm `.mcp.json` exists on disk, proving the hook fired
and wrote it. A 0/3 with no file written measures a broken hook, not the ordering.

## What I already know

- The current session is direct evidence of the failure mode this replaces: `.mcp.json` was absent,
  and this session has no `recall`, `qwen-mcp`, `vps3-lite`, `code-rag` or `mcp-pg-ops` tools.
- Memory entry `a-setting-honoured-in-one-entry-point.md`: a stdio MCP server launched with an
  explicit `env` block does not inherit exported shell variables, which is why
  `RECALL_TRUST_MODE=development` is written into the generated file rather than exported. That is
  about the file's *content* and does not bear on when it is *read*.
- No prior measurement of hook-versus-MCP ordering exists in this store. This is not a
  re-measurement.

## Confounds I can name now

1. **Approval, not ordering.** Project-scoped MCP servers from `.mcp.json` can require approval
   before they are usable. An absence of tools would then mean "not approved" rather than "read too
   late", and the two are indistinguishable from the tool inventory alone. Mitigation: the control
   run in a checkout that already has `.mcp.json` shares the approval state, so if the control shows
   the tools and the treatment does not, approval is excluded as the explanation.
2. **Headless is not interactive.** `claude -p` may load MCP servers on a different path from an
   interactive session. A result here transfers to interactive sessions only by assumption, and that
   assumption is stated rather than tested.
3. **stdio server startup cost.** The `recall` servers are `python -m recall_mcp.server`; if they
   are slow to start, a session could report them as absent for a reason unrelated to file ordering.
4. **Self-fulfilling file state.** A worktree reused from an earlier run would already have
   `.mcp.json`, turning a treatment run into a control. Each run therefore uses a fresh worktree.

## Apparatus failure (2026-08-16)

**Status at the time: still predicted, not yet measured.** The prediction above is unchanged and must stay that
way.

The control run never produced a number. `claude -p` in a checkout that already had `.mcp.json`
returned:

```
Failed to authenticate. API Error: 401 OAuth access token has been revoked.
```

so no session started, and the three treatment runs were not attempted. This is exactly what the
control was for: without it, three treatment runs would have returned "no `recall` tools" for a
reason that has nothing to do with file ordering, and 0/3 would have looked like a confirmed
prediction. **Exit code 0 is not a measurement, and neither is a 401.**

Static inspection was tried as a substitute and is not available: the CLI installs as a native
`claude.exe`, with no JS bundle to read the hook-versus-MCP ordering out of.

### Deferred method, at zero extra cost

The hook now records every session start to `~/.claude/session-start.log`, one JSON row carrying
`mcp_json_existed_before_hook` and `mcp_action`. The next session started in a checkout where the
hook writes `.mcp.json` (`mcp_action: "generated"`) **is** a treatment run, and that session's own
tool inventory is the outcome. So the experiment now runs itself as a by-product of ordinary work,
and needs only that somebody read the log and record the result here.

To settle it deliberately instead, re-authenticate the CLI (`claude login`) and run the method as
written above. Until one or the other happens, this question is open, and the hook's message
deliberately states both outcomes rather than asserting the predicted one.

## Treatment run 1 (2026-08-18): a genuine treatment run, and NO usable data point

**Status at the time: still predicted, not yet measured.** The prediction above is unchanged and stays that way.

The deferred method fired exactly as designed. Session `4ea95a6b` started in a fresh worktree and
the hook logged:

```json
{"at": "2026-08-18T14:04:28Z", "session": "4ea95a6b", "source": "startup",
 "cwd": "...\\.claude\\worktrees\\silly-curran-42df24", "outcome": "claimed",
 "is_main": false, "mcp_json_existed_before_hook": false, "mcp_action": "generated"}
```

`mcp_json_existed_before_hook: false` with `mcp_action: "generated"` is the treatment condition
stated above, and the second apparatus check passes: the file was written, so a null cannot be
blamed on a hook that never fired.

**Outcome on the metric: `recall` did NOT appear in that session's own tool inventory.** The session
had `Claude_Browser`, `visualize` and `ccd_session` and no `mcp__recall__*` tool of any kind.

### Why that is not 1/1 toward the predicted 0/3

**Confound 1 (approval, not ordering) is not merely possible here, it is confirmed.** Read directly
from `~/.claude.json` at the time of writing:

```
projects["...\\worktrees\\silly-curran-42df24"].enabledMcpjsonServers == []
```

The worktree had **no approval recorded at all**, so a project-scoped server from `.mcp.json` could
not have loaded no matter when the file was written. The absence of tools is fully explained before
the ordering question is even reached, and the run therefore says nothing about it.

The mitigation this document specifies for confound 1 is a control run in a checkout that already
has `.mcp.json` **and shares the approval state**. That mitigation was unavailable, because the
treatment worktree's approval state was "none", which no control can share while still being a
control.

So: **1 treatment run executed, 0 usable observations. The metric denominator does not advance.**

This is the same failure the 2026-08-16 apparatus note describes, one layer in. A 401 is not a
measurement; neither is an unapproved project. Both produce "no `recall` tools" for a reason that
has nothing to do with file ordering, and both look identical to a confirmed prediction.

### What DID change, and it unblocks the control

🔁 **The store's "zero approvals" claim is now false, and that is the useful part of this run.**
Memory entry `mcp-servers-blocked-by-pending-approval` records **306 tracked projects, zero with any
approved server**, which is why no control was thought possible. Re-measured 2026-08-18:

| | measured 2026-08-17 | measured 2026-08-18 |
|---|---|---|
| tracked projects | 306 | **309** |
| projects with any `enabledMcpjsonServers` | 0 | **2** |

The two are `.claude/worktrees/session-startup-audit-518fcd` and
`.claude/worktrees/musing-dewdney-f0b28b`, both approved for `['recall', 'recall-memory']`. **Both
still exist on disk and both already have `.mcp.json`.** Re-check with:

```bash
python -c "import json,os;d=json.load(open(os.path.expanduser('~/.claude.json')));print({k:v.get('enabledMcpjsonServers') for k,v in d['projects'].items() if v.get('enabledMcpjsonServers')})"
```

That is exactly the control condition this document asks for and could not previously construct: an
approved checkout with the file already present. **A session opened in either of those two
worktrees is the control run**, and its tool inventory is the number that decides whether the probe
can see the tools at all.

⚠️ **But those two approvals were written by a SCRIPT, not by the interactive prompt, and that
matters for what a control there would prove.** `scripts/session_mcp_approve.py` exists only in
commit `49ac9c5f` ("Approve the MCP servers this repo generates, instead of writing the file
earlier"), which is **not merged**: `git merge-base --is-ancestor 49ac9c5f origin/master` fails, the
path does not resolve in `origin/master`, and the only copy on disk is inside
`musing-dewdney-f0b28b` itself. That is almost certainly why exactly those two worktrees are the
approved ones, and it is a second reason this repository's own `scripts/session-mcp.sh` has no
approval code: on master, nothing records approvals at all.

Whether a script-written approval actually yields `mcp__recall__*` tools in a live session is listed
as **still unverified** in the machine-level notes, on the grounds that `claude -p` returns 401 here
and the two config entries are byte-identical to an interactively-approved one. So a control run in
`musing-dewdney-f0b28b` settles **two** open questions at once, and they are not separable: a
negative there means either the probe cannot see the tools *or* script-written approval does not
work, and distinguishing those needs a third checkout approved through the interactive prompt.

### The instrument's remaining gap, stated so it is not rediscovered

The log records the treatment condition and nothing records the **outcome**. `mcp_action` is written
by the hook, which cannot observe the tool list the client assembled; only the session itself can
report that, in prose, to whoever reads the log later. So each data point still costs a human
reading a transcript, and 13 of the 741 logged sessions to date carry `mcp_action: "generated"`
without any of them having their outcome recorded here.

Two of those 13 treatment runs occurred in worktrees that are approved *today*
(`session-startup-audit-518fcd`, `musing-dewdney-f0b28b`), but approval is timestamped nowhere, so
it cannot be established whether they were approved *at the time*. **Those rows are not
retrospectively usable**, and reading them as data would repeat this run's mistake.

**Next step, in order:**

1. Open a session in `musing-dewdney-f0b28b` (approved, `.mcp.json` present) and record whether
   `recall` appears. This is the control, and per the caveat above it is also the first live test of
   a script-written approval.
2. If the control shows the tools, a treatment run becomes interpretable **only in a worktree that
   is approved before the session starts**, which today needs `scripts/session_mcp_approve.py` from
   the unmerged `49ac9c5f`. Landing that commit is a prerequisite for the experiment, not a side
   quest: on master there is no way to approve a fresh worktree at all, so every future fresh
   worktree reproduces this run's confound exactly.
3. If the control does **not** show the tools, the probe cannot detect them and every treatment
   result to date, including this one, stays meaningless. Escalate to a checkout approved through
   the interactive prompt before concluding anything about ordering.

⛔ **Do not record a further treatment run until step 1 returns a number.** This run is the second
time the question has been approached and the second time an untested apparatus produced an
uninterpretable null; a third would establish a habit rather than a result.

## MEASURED (2026-08-18): the prediction is CONFIRMED, and approval is separable after all

**Status: measured. The prediction above stands and is not edited.**

The control was never run by hand, because it had already run itself. The evidence is the hook log
plus the harness's own `deferred_tools_delta` records, which are structural entries written by the
client listing tools entering a session's inventory. They are **not** model prose, which matters:
a bare grep for `mcp__recall` across these transcripts matches conversation text too, and four of
the six transcripts here contain the string while carrying zero tools.

### The decisive session is both arms at once

Session `e0aa68bb` ran in `.claude/worktrees/session-startup-audit-518fcd`, a project whose
`enabledMcpjsonServers` holds `['recall', 'recall-memory']`. The hook logged it **twice**:

| hook row | condition | recall tools in inventory |
|---|---|---|
| `2026-08-18T09:21:43Z` session start | `mcp_action: generated`, `mcp_json_existed_before_hook: false` | **0** (delta at 09:21:43.677Z) |
| `2026-08-18T12:08:56Z` second SessionStart | `mcp_action: already-present`, `existed_before: true` | **32**, in a delta at 12:08:58.640Z |

The first row is the treatment condition exactly as specified: no `.mcp.json` on disk, the hook
generates it, and **that session did not get the tools**. The second row is the same checkout two
hours later with the file already on disk, and the tools appear **two seconds** after the hook
fires.

So on the stated metric, with the denominator counting treatment runs that are interpretable:
**0 of 1 fresh sessions saw `recall` in their own inventory**, and the prediction's mechanism, that
the client resolves its server list before user `SessionStart` hooks run, is what the pairing shows.

### Why this one is interpretable where the earlier run was not

This is the comparison the document asked for and could not construct: the treatment and the
control **share an approval state**, because they are the same project two hours apart. Confound 1
therefore cannot explain the 09:21 absence. The run recorded above under "Treatment run 1" failed
precisely here, in an unapproved worktree, and remains uninterpretable.

### The two causes separate cleanly across all six transcripts

Both conditions turn out to be necessary, and each alone yields nothing:

| condition | transcripts | recall tools |
|---|---|---|
| `.mcp.json` written by the session's own hook | `d8bded1a`, `30a93f4d`, `9b366ea4`, `e0aa68bb` (first start) | 0 |
| `.mcp.json` already present, project not yet approved | `b03972ae`, `2b8b37ba` | 0 |
| `.mcp.json` already present **and** project approved | `e0aa68bb` (second start) | **32** |

This retires the reading that approval alone explains everything: two sessions started with the file
already present and still got nothing, because approval had not yet been recorded for those
checkouts at that time. It equally retires the reading that ordering alone explains everything.
**File-present and approved are jointly necessary**, and the tool inventory cannot distinguish
which one is missing, which is why every earlier single-arm observation was unreadable.

### What is still not established

1. **Approval is timestamped nowhere.** `~/.claude.json` records which servers are approved, not
   when. That the 09:21 arm was already approved is inferred from the same project being approved at
   12:08 with no approval action in between in that transcript, not from a timestamp. If approval
   was in fact granted between the two starts, the pairing collapses back into confound 1 and this
   result reverts to uninterpretable. **This is the one check that would falsify the reading above**,
   and it cannot be run retrospectively; a future run should snapshot `enabledMcpjsonServers`
   immediately before the session starts.
2. **What the 12:08 SessionStart actually was.** A second hook row under the same session id is a
   resume, a compaction or a re-entry; the transcript shows no MCP command before it, and the
   preceding turn is ordinary work. So "a later SessionStart in the same checkout" is demonstrated;
   "a wholly new process" is not.
3. **`deferred_tools_delta` reports visibility, not connection.** The tools became *available to the
   model* at those moments. Whether the stdio servers connected then, or connected earlier and were
   surfaced lazily, is not visible in the transcript. The prediction is about availability to the
   session, which is what is measured, but the underlying mechanism is inferred.
4. **`claude -p` is still 401.** Re-checked 2026-08-18: `Failed to authenticate. API Error: 401
   OAuth access token has been revoked`, and note it **exits 0**, so a harness reading exit codes
   scores that as success. The method as originally written remains unrunnable; this result comes
   from the deferred method instead.

### Consequence for the repository

The practical rule is unchanged and now has a measurement behind it: **a fresh worktree gets no
recall tools in the session that creates it, and gets them on the next session provided the project
is also approved.** Since `scripts/session_mcp_approve.py` is absent from `origin/master`
(`git merge-base --is-ancestor 49ac9c5f origin/master` fails), a fresh worktree on master satisfies
neither condition on its first session and only the ordering condition on its second. Landing that
commit is what makes the second session work.

## 🔁 CORRECTION (2026-08-19): "jointly necessary" is falsified, and the two rows carrying it were mislabelled

**Status: the prediction is untouched, and nothing above this line is edited.** The ordering
reading survives; the approval half of the verdict does not.

Raised by another session, which saw a full recall tool set in a worktree with no approval. I
checked it against the transcripts rather than accept the report, and the check went further than
the report did.

### 1. Approval is not necessary. Measured, and no inference is involved

Session `ba35479a`, worktree `compassionate-ishizaka-7ab24d`:

| hook row | condition | recall tools entering |
|---|---|---|
| `2026-08-19T14:56:40Z` startup | `existed_before: false`, `generated` | **0** (delta at 14:56:41.652Z) |
| `2026-08-19T16:05:30Z` resume | `existed_before: true`, `already-present` | **32** (delta at 16:05:46.316Z) |

That project holds `enabledMcpjsonServers: []` and has never been on the approved list: 3 of 311
tracked projects carry any approval today and it is not one of them. Every other way those tools
could have arrived was checked on the same read and excluded:

| candidate source | state |
|---|---|
| user scope, top-level `mcpServers` | `{}` |
| local scope, `projects[dir].mcpServers` | key absent |
| `enableAllProjectMcpServers` | unset |
| approval committed to a repo `.claude/settings.json` | no such file |

**So "File-present and approved are jointly necessary" is false as stated.** 32 tools arrived in a
project that was never approved.

### 2. The two rows supplying the "not yet approved" leg were in fact approved

The verdict above says approval "is timestamped nowhere" and the check "cannot be run
retrospectively". That is true of `~/.claude.json` and false of the transcripts:
`scripts/session_mcp_approve.py` prints the project it approved, so the event is timestamped
wherever it ran. Recovered:

| approval written | by session | project |
|---|---|---|
| `2026-08-16T22:22:14.671Z` | `30a93f4d` | `musing-dewdney-f0b28b` |
| `2026-08-16T23:06:34.254Z` | `9b366ea4` | `session-startup-audit-518fcd` |

Against the two rows the table classified as unapproved:

| transcript | started | project | approved before it started? |
|---|---|---|---|
| `b03972ae` | `2026-08-16T22:22:47Z` | `musing-dewdney-f0b28b` | **yes, by 33 seconds** |
| `2b8b37ba` | `2026-08-16T23:08:03Z` | `session-startup-audit-518fcd` | **yes, by 89 seconds** |

Both had the file **and** the approval, and both received 0 recall tools. They cannot support joint
necessity, and they refute the pair being sufficient.

⚠️ One alternative I cannot exclude for these two: a running client owns `~/.claude.json` and
rewrites it on its own schedule, so an approval written seconds earlier could have been overwritten
before the session read it. Both approvals survive on disk today, so no permanent clobber happened.
This caveat does not reach finding 1, which rests on no timing inference at all.

### 3. What every row is consistent with is resume, not fresh start

| hook row | source | file present | approved | recall tools |
|---|---|---|---|---|
| `d8bded1a` 21:18 (08-16) | startup | no | no | 0 |
| `d8bded1a` 21:50 | startup | yes | no | 0 (no delta at all) |
| `30a93f4d` 22:09 | startup | no | no | 0 |
| `b03972ae` 22:22 | startup | yes | **yes** | 0 |
| `9b366ea4` 22:42 | startup | no | no | 0 |
| `2b8b37ba` 23:08 | startup | yes | **yes** | 0 |
| `e0aa68bb` 09:21 (08-18) | startup | no | yes | 0 |
| `e0aa68bb` 12:08 | **resume** | yes | yes | **32** |
| `e0aa68bb` 15:17 | **resume** | yes | yes | **32** |
| `ba35479a` 14:56 (08-19) | startup | no | **no** | 0 |
| `ba35479a` 16:05 | **resume** | yes | **no** | **32** |
| `9b366ea4` 16:30 (08-19) | resume | no | yes | 0 |

No fresh `startup` in this corpus ever received the tools, including two that had both the file and
the approval. Every row that received them was a `resume` with the file already present. Approval
does not separate the outcome anywhere in the table; `resume` separates it everywhere.

⚠️ **Confounded with date.** Every 0-tool startup-with-file row is from 08-16 and both 32-tool
resume rows are from 08-18 and 08-19, so "resume versus startup" and "before versus after some
change in the client" are not separated by this data.

⚠️ A `deferred_tools_delta` lists tools **entering** an inventory, so a later delta carrying 0
recall tools does not mean the session lost them. Only the first delta of a session reads as an
initial inventory. `ba35479a` has a third delta at 16:13:20.266Z carrying 0, and that session still
had its tools.

**The decisive control is now cheap, and it is not the one "Next step" 1 asks for.** Start a
**fresh** session, not a resume, in a checkout that already has `.mcp.json`. If it gets the tools,
the resume reading dies and the practical rule stands as written. If it does not, then "a fresh
worktree gets them on the next session" is wrong, and the rule in "Consequence for the repository"
needs replacing rather than annotating.

### 4. Session type was never recorded, and it is the variable the docs say matters

Per `https://code.claude.com/docs/en/mcp`, fetched 2026-08-19 against CLI 2.1.220: `claude -p`, Agent
SDK sessions and cloud sessions cannot show the approval prompt, so they **load project-scoped
servers without asking**. Approval gates interactive sessions only.

I could not find a session-type field in any of these transcripts, so I cannot label the rows above,
and I am not going to guess. That is the gap: **every future run must record session type**, because
if these rows are SDK sessions then the approval leg was never under test in any of them, and the
verdict's approval column was measuring nothing.

### 5. Consequence for the repository, and for the product

- **"Next step" 2 overstates the dependency.** Landing `49ac9c5f` still helps an interactive
  first-run user, but it is not a prerequisite for this experiment, because approval did not gate
  any row here.
- **User scope has no approval step at all** and is the only scope loading in every project
  (top-level `mcpServers` in `~/.claude.json`; local scope is the same file under
  `projects[dir].mcpServers`). Precedence is local, then project, then user, and entries are **not
  merged**.
- **Since CLI v2.1.196 there is a second gate**: an approval committed to a repository's
  `.claude/settings.json` is ignored until the workspace is trusted, so a cloned repository cannot
  approve its own servers.
- **The product has the interactive half of this bug today.** `recall/wizard/wiring.py:217`
  `mcp_config()` emits a project-scoped document that `recall/wizard/headless.py:1153` writes to
  `project_root/.mcp.json`, and nothing under `recall/` records an approval. So a first-run user
  finishes `recall wizard --headless`, sees success, opens Claude **interactively**, and meets the
  approval gate. No headless verification can reproduce that, which is exactly the asymmetry
  finding 4 describes.

Re-measure the whole of this section:

```bash
python -c "import json,os;d=json.load(open(os.path.expanduser('~/.claude.json')));print(len(d['projects']),{k.split(os.sep)[-1]:v.get('enabledMcpjsonServers') for k,v in d['projects'].items() if v.get('enabledMcpjsonServers')})"
grep -h mcp_action ~/.claude/session-start.log | tail -20
grep -rl "deferred_tools_delta" ~/.claude/projects/*/ | head
```
