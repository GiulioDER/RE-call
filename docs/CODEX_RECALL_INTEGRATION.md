# Codex RE-call integration

Codex uses the same RE-call memory corpus and memo format as Claude Code.

## Automatic installation

`recall setup` detects Codex and, with the default answer, installs the RE-call Codex integration
automatically. It writes a personal marketplace entry at `~/.agents/plugins/marketplace.json`,
copies the validated plugin bundle below that marketplace, stores the DSN in the user-owned Codex
integration directory, and merges the five user-level hooks into `CODEX_HOME/hooks.json`.
The operation is idempotent and preserves unrelated marketplace entries and hooks. Restart Codex
after setup so it refreshes the personal marketplace and enables the plugin. The same flow works
from a wheel because the Codex plugin bundle is force-included in the package.

The repository also exposes the bundle through `.agents/plugins/marketplace.json` for a repo-scoped
plugin checkout. Codex plugins require a user review of bundled hooks before trusting a changed
definition. The setup path installs equivalent user-level hooks directly, so memory enforcement
does not depend on the plugin trust prompt. The Claude bundle remains separate and continues to
invoke `recall-hooks`.

## Installed parity

The global Codex configuration at `CODEX_HOME/hooks.json` (normally `~/.codex/hooks.json`) mirrors Claude's
RE-call lifecycle and additionally handles compact sessions:

* `SessionStart` restores the instruction on startup, resume, clear, and compact. Claude's
  SessionStart matcher currently covers startup, resume, and clear; compact is Codex-specific.
* `UserPromptSubmit` runs prompt time retrieval before the response is planned.
* `PreToolUse` runs the shared Claude write time hook for write capable tools.
* `PreCompact` queues the docs and code refresh and serialized memory refresh; `SessionEnd` runs the
  bounded memory refresh synchronously because Codex treats that lifecycle event as synchronous.

The Codex adapter `python -m recall_hooks.codex` delegates prompt time and write time to the shared
`recall_hooks` implementation. This keeps thresholds, local project discovery, front matter
summaries, and fail-open behaviour in one implementation.

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
Add a pointer to `project_index.md` in the form `- [Title](file.md) — one-line hook`.

The automatic `SessionEnd` hook handles the normal refresh path. When a session produces a durable
fact, write it in the same project memory directory and front matter format Claude uses, then add
its pointer to `project_index.md`; the next `SessionStart` or prompt hook will discover it.

## Operational boundary

The Codex RE-call MCP server is launched by `recall-codex-mcp` from the protected integration
configuration. The VPS2 memory, docs, and code tenants remain the deployment's configured
servers; the memory tenant uses `voyage:voyage-4`, the documentation tenant uses its active
`fastembed:BAAI/bge-large-en-v1.5` generation, and the code tenant uses `voyage:voyage-code-3`.
The workstation does not embed memory locally. The memory worker
uses the existing lock, debounce, sync, production generation, calibration, promotion, and
postcondition verification in `C:\Users\gde00\.claude\recall-vps2\refresh-memory-vps2.ps1`.

Hook failures are fail open and may be silent at the bridge boundary. They do not deny a tool or
block a session. Refresh-worker failures remain visible through the worker state and log files,
and the next startup still instructs Codex to use `recall_search`, `recall_evidence`, and
`recall_stats`.
