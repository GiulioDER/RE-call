# Codex RE-call integration

Codex uses the same RE-call memory corpus and memo format as Claude Code.

For a local SQLite store, follow [Lite client wiring](LITE.md#connect-another-client).
`recall setup --lite` registers Claude Code only; it does not run the automatic Codex installer
described below. Codex can connect to the same local file through a manually configured stdio MCP
server, using the interpreter and environment printed by Lite setup.

## Automatic installation

The PostgreSQL `recall setup` path detects Codex and, with the default answer, installs the RE-call
Codex integration
automatically. It writes a personal marketplace entry at `~/.agents/plugins/marketplace.json`,
copies the validated plugin bundle below that marketplace, stores the DSN in the user-owned Codex
integration directory, and merges the seven user-level hooks into `CODEX_HOME/hooks.json`.
The operation is idempotent and preserves unrelated marketplace entries and hooks. Restart Codex
after setup so it refreshes the personal marketplace and enables the plugin. The same flow works
from a wheel because the Codex plugin bundle is force-included in the package.
The bundle includes `check-memory-before-acting`, `keep-memory-current`, `re-call`,
`session-open`, and `session-close`. These portable skills ship with both client bundles;
they do not depend on another user's home-directory commands or refresh scripts.

The repository also exposes the bundle through `.agents/plugins/marketplace.json` for a repo-scoped
plugin checkout. Codex plugins require a user review of bundled hooks before trusting a changed
definition. The setup path installs equivalent user-level hooks directly, so memory enforcement
does not depend on the plugin trust prompt. The Claude bundle remains separate and continues to
invoke `recall-hooks`.

## Installed parity

The global Codex configuration at `CODEX_HOME/hooks.json` (normally `~/.codex/hooks.json`) mirrors Claude's
shared RE-call lifecycle:

* `SessionStart` restores the instruction on startup, resume, clear, and compact. Its cached
  count is not evidence of current calibration or availability; use live MCP verdicts.
* `UserPromptSubmit` runs prompt time retrieval before the response is planned.
* `PreToolUse` runs the shared Claude write time hook for write capable tools.
* `PostToolUse` sends completed shell failures to the shared general knowledge handler.
  Successful commands, interrupted calls and outputs without a proven exit status stay quiet.
* `SubagentStart` sends the shared auditor brief to recognized auditors as advisory child
  context. It neither rewrites the launch input nor makes a permission decision.
* `PreCompact` queues the docs and code refresh and serialized memory refresh; `SessionEnd` runs the
  bounded memory refresh synchronously because Codex treats that lifecycle event as synchronous.
  `SessionEnd` also invokes RE-call's repository-owned `recall_hooks.mcp_cleanup` module, closing
    only the local MCP transports descended from this Codex client, using the client process identity
    rather than a Claude-only `CLAUDE_PID` variable. Its 15-second bound covers process inspection
    and teardown of multiple transports.

For VPS2-backed MCP, each generated server command also carries `RECALL_MCP_SESSION_ID`. The value
comes from the worktree's live session claim, so a new session in the same checkout gets a new
opaque identity while a resume keeps the old one. Cleanup prefers this ID, then the client process
identity, and finally the legacy `RECALL_MCP_CLIENT` marker. The fleet sweep uses the same ID to
distinguish a live transport from an orphan; it never closes a server merely because it is old or
because its command resembles another session's.

The Codex adapter `python -m recall_hooks.codex` delegates prompt time and write time to the shared
`recall_hooks` implementation. This keeps thresholds, local project discovery, front matter
summaries, and fail-open behaviour in one implementation.

The [Codex hook contract](https://learn.chatgpt.com/docs/hooks) supplies tool output on
`PostToolUse`, but its shape depends on the tool and runtime. Structured shell output with an
integer `exit_code` is supported. When the runtime supplies raw stdout, the adapter reads at
most the last 1 MiB of the identified rollout below `CODEX_HOME/sessions`, matches the exact
`tool_use_id`, and requires a completed native `CommandExecution` record. Printed status
headers are never parsed as exit metadata. This native record path is a compatibility layer,
not a stable public rollout API; unknown records produce no failure hint.

Some named collaboration auditors start with `agent_type: default`. The adapter accepts their
role only from child session metadata whose child id, parent thread id and full agent path
agree. The startup event can name either the parent or child transcript while retaining the
parent `session_id`. Unrelated children receive no auditor brief. Codex supplies no launch
prompt here, so the shared handler uses changed files without prompt-based file hints.

Existing installations must rerun setup to refresh the installed bundle and merged hooks, then
review and trust changed hook definitions in Codex. Changing Python code alone does not add
event registrations to an older installation.

## Verify a fresh session

After installation, open a new read-only Codex chat. Check its actual skill catalog for
`session-open` and `session-close`, then read the installed skill files. Check the initial
startup context or the client's hook diagnostics for SessionStart delivery. If the deployment
records lifecycle logs, correlate the new chat's session ID with its startup record. Running
the hook manually, or finding its file on disk, does not prove automatic delivery.

Use `session-open` for the full repository and memory orientation. The automatic memory hook
does not claim a workspace, fetch origin, inspect containers, or approve knowledge proposals.
Use `session-close` to complete the project's close procedure and report each applicable
result. Queueing a refresh is not proof that the next session can retrieve a new memo.

User-level and plugin-level hooks are separate registration paths. Review an installation
that also has custom RE-call hooks before enabling equivalent handlers twice. Preserve
unrelated hooks and use the client's normal trust review for changed plugin definitions.

## Durable memo contract

Write one fact per file in the Claude project memory directory and add its pointer to
`project_index.md`. The file must contain:

```markdown
---
name: short-kebab-case-slug
description: "One-line summary used to judge relevance"
valid_from: YYYY-MM-DD
metadata:
  node_type: memory
  type: project
  originSessionId: codex-session-id
  modified: 2026-09-02T00:00:00Z
---

# What this records

The fact, why it matters, and how to apply it.
```

If the fact changes, create a new file with `supersedes: old-file.md` and retain the old file.
Add a pointer to `project_index.md`, for example `- [Title](file.md): one-line hook`.

The automatic `SessionEnd` hook handles the normal refresh path. When a session produces a durable
fact, write it in the same project memory directory and front matter format Claude uses, then add
its pointer to `project_index.md`. Verify retrieval after refresh before claiming it is served.

## Operational boundary

The Codex RE-call MCP server is launched by `recall-codex-mcp` from the protected integration
configuration. Corpus routes, refresh commands, embedding location, worker locks and maintenance
holds belong to the deployment. An audit corpus or general knowledge index is optional; this
integration does not install a particular operator's servers, tenant names or review workflow.
Keep credentials, private endpoints and machine-specific lifecycle scripts in private
configuration rather than copying them into the public plugin.

Hook failures are fail open and may be silent at the bridge boundary. They do not deny a tool or
block a session. Refresh-worker failures remain visible through the worker state and log files,
and the next startup still instructs Codex to use `recall_search`, `recall_evidence`, and
`recall_stats`.
