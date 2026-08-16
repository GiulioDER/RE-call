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
