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

## Result (2026-08-18): one treatment run, and the confound it exposes

**Status: measured once, and NOT settled.** The prediction and method above are unchanged and stay
that way. This section is appended, not edited.

### What was observed

Session `7f79562f`, in `.claude/worktrees/jolly-germain-682885`, is a treatment run produced exactly
as the deferred method intended: as a by-product of ordinary work, with nobody arranging it. Its row
in `~/.claude/session-start.log`:

```json
{"at": "2026-08-18T14:58:34Z", "session": "7f79562f", "source": "startup",
 "cwd": ".../recall/.claude/worktrees/jolly-germain-682885", "elapsed_s": 7.07,
 "outcome": "claimed", "is_main": false,
 "mcp_json_existed_before_hook": false, "mcp_action": "generated"}
```

Both apparatus checks pass. `mcp_json_existed_before_hook` is `false` and `mcp_action` is
`generated`, so the hook fired and wrote the file, and this is not a control run in disguise
(confound 4). The outcome, read off that session's own tool inventory: **no `recall` and no
`recall-memory` tools.** The session ran its whole length on `python -m recall.cli` over the CLI
instead.

So the observation is consistent with the prediction. It is not a confirmation of it, and the
denominator is 1, not the 3 the method calls for.

### Why this does not settle the question

**Confound 1 is live, and is now measured rather than merely named.** Read from `~/.claude.json`
during that session:

- 309 tracked projects, of which **2** carry any approved server:
  `.claude/worktrees/session-startup-audit-518fcd` and `.claude/worktrees/musing-dewdney-f0b28b`,
  both `['recall', 'recall-memory']`.
- This worktree **has an entry**, with `enabledMcpjsonServers: []`.

An empty approval list is a sufficient explanation for the absence of tools on its own. Ordering
never had to be involved. A 1/1 "absent" therefore measures approval state, not the race.

### The stated mitigation for confound 1 does not hold

Confound 1 above proposes: *"the control run in a checkout that already has `.mcp.json` shares the
approval state, so if the control shows the tools and the treatment does not, approval is excluded
as the explanation."*

That is false, and the reason is visible in the numbers above. Approval is recorded **per project
directory**, keyed by absolute path under `projects[<dir>].enabledMcpjsonServers`. Two worktrees are
approved and this one is not, in the same clone, at the same moment. A control run in a *different*
worktree therefore shares nothing with the treatment, and the difference between them is confounded
with approval by construction. The mitigation as written cannot exclude the explanation it was
designed to exclude.

This does not invalidate the prediction. It invalidates a piece of the apparatus, which is the
second time this question has failed on apparatus rather than on the phenomenon.

### What would actually settle it

The control and the treatment must run **in the same directory**, so that approval state is held
fixed rather than assumed shared:

1. In one worktree, record the approval so `enabledMcpjsonServers` is non-empty, and confirm it
   against `~/.claude.json` rather than `claude mcp list`, which reports `⏸ Pending approval`
   even for projects whose approval is already recorded.
2. Delete `.mcp.json` from that same worktree, leaving the approval in place.
3. Start a fresh session there and read its own tool inventory.

With approval held non-empty, an absence of tools is attributable to ordering. That is the
measurement the original method was reaching for, and it is n = 1 per worktree, so it still wants
repeating.

### One external claim this retires

`~/.claude/CLAUDE.md` records, measured 2026-08-17: *"306 tracked projects, zero with any approved
server."* As of 2026-08-18 that is **309 tracked, two approved**, so approvals do persist once
written. The claim that approval never sticks is retired; the claim that this session's tools were
blocked by approval is what replaces it.

The writer is `scripts/session_mcp_approve.py`, added in `49ac9c5f` ("Approve the MCP servers this
repo generates, instead of writing the file earlier"). **It is not on `master`**, and it is absent
from this checkout, which is level with `origin/master`. It exists only in the
`musing-dewdney-f0b28b` worktree, and that worktree is one of the two approved ones. So the approval
mechanism and the approved population are the same small thing, and a session in any other checkout
has no scripted way to reach it.
