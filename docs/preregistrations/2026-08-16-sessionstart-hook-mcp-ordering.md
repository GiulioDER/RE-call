# Pre-registration: does a SessionStart hook that writes `.mcp.json` reach the same session?

**Date:** 2026-08-16   **Status:** predicted, not yet measured

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

**Status: still predicted, not yet measured.** The prediction above is unchanged and must stay that
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

## Result (2026-08-17): confound 1 was the operative cause, and it makes the stated metric blind

**Status: the ordering question is still open. The failure it was written to explain is solved.**
The prediction above is unchanged and stays that way.

### What was measured

The planned method is still impossible: `claude -p` returns `401 OAuth access token has been
revoked` on this machine, re-verified today. So a different instrument was used, one that does not
need a session at all. In this checkout, with `.mcp.json` present on disk in front of it:

```
$ claude mcp list
recall: python -m recall_mcp.server - ⏸ Pending approval (run `claude` to approve)
recall-memory: python -m recall_mcp.server - ⏸ Pending approval (run `claude` to approve)
```

Supporting audit of the client's own config (`~/.claude.json`), same day:

| Quantity | Value |
|---|---|
| tracked projects | 306 |
| projects with any approved `.mcp.json` server | **0** |
| projects with project-scope servers defined in the client config | 0 |
| this checkout's `hasTrustDialogAccepted` | `true` |

And an apparatus check the original method did not have: the server was driven directly over
stdio, with the exact `command`, `args`, `cwd` and `env` from the generated `.mcp.json`. It
completed the MCP handshake, listed 10 tools, and answered a real query with 5 hits and
`abstained: false`. **The server is not the problem, and neither is the file.**

### What this does to the metric

The registered metric is "fraction of fresh sessions in which the `recall` server appears in the
session's own tool inventory", predicted 0/3. That metric cannot separate the two explanations,
because an unapproved server is absent from the inventory no matter when the file was written.
0/3 was guaranteed before the first run, by a cause the experiment was not measuring.

Confound 1, **"Approval, not ordering"**, was named in advance, and its mitigation was void. The
mitigation was a control run in a checkout that already had `.mcp.json`, on the reasoning that the
control "shares the approval state". It did share it: both were **equally unapproved**, so the
control could only ever have reproduced the treatment. A control that shares the confound does not
exclude the confound. That is the transferable lesson here, and it is worth more than the result.

### Scoring the predictions honestly

- **Primary prediction (no tools in the hook's own session): outcome correct, mechanism wrong.**
  There were no tools, but not because the client read the file too early. It would equally have
  had no tools had the file been written a week earlier.
- **Secondary prediction: correct, and it was the real answer.** It reads, verbatim: "they appear
  only after an approval prompt for project-scoped servers, so the win would not be automatic in a
  fresh checkout anyway." The half-sentence hedge was the finding, and it sat unread for a day
  because the primary prediction's outcome matched and nothing forced a look at why.
- Deferred-method rows in `~/.claude/session-start.log` at time of writing: 368 rows, 4
  `generated` (treatment), 15 `already-present` (control condition). This worktree contributed one
  of each. Neither condition produced a `recall` tool in any session, which is consistent with a
  cause that is indifferent to both.

### What changed as a result

`scripts/session_mcp_approve.py`, called by `scripts/session-mcp.sh` immediately after it writes
`.mcp.json`, records the approval for that checkout in the client's config. It carries **only
server names** across that boundary, never a URL or a token, and it refuses to reverse a server
the operator has explicitly disabled. 12 tests, and the five mutants that matter were each caught
by the test that names them.

### What is still open

Whether a hook-written `.mcp.json` reaches its own session is **still unmeasured**, and is now
measurable for the first time: once a checkout is approved, tool-absence stops being overdetermined
and the registered method finally means what it says. It needs `claude login` first.
