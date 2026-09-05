# Codex RE-call integration

Codex uses the same RE-call memory corpus and memo format as Claude Code.

## Installed parity

The global Codex configuration at `C:\Users\gde00\.codex\hooks.json` contains the same
memory events as the Claude integration:

* `SessionStart` restores the instruction on startup, resume, clear, and compact.
* `UserPromptSubmit` runs prompt time retrieval before the response is planned.
* `PreToolUse` runs the shared Claude write time hook for write capable tools.
* `PreCompact` and `SessionEnd` queue the docs and code refresh and the serialized memory refresh.

The Codex adapter at
`C:\Users\gde00\.codex\re-call\hooks\recall-hook-bridge.py` delegates prompt time and write time
to `C:\Users\gde00\.claude\hooks\recall_hooks`. This keeps thresholds, local project discovery,
front matter summaries, and fail open behaviour in one implementation.

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

Run the explicit close command from the checkout:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\gde00\.codex\re-call\session-close.ps1" -MemoPath "C:\Users\gde00\.claude\projects\<project>\memory\<memo>.md"
```

The command validates the trusted path, required front matter, and index pointer. It starts the
memory generation refresh independently from the docs and code refresh. Use `-NoMemo` only when
the session produced no durable fact.

## Operational boundary

The Codex MCP servers are configured in `C:\Users\gde00\.codex\config.toml` and use VPS2 for the
memory, docs, and code tenants. The memory tenant uses `voyage:voyage-4`, the documentation tenant
uses its active `fastembed:BAAI/bge-large-en-v1.5` generation, and the code tenant uses
`voyage:voyage-code-3`. The workstation does not embed memory locally. The memory worker
uses the existing lock, debounce, sync, production generation, calibration, promotion, and
postcondition verification in `C:\Users\gde00\.claude\recall-vps2\refresh-memory-vps2.ps1`.

Hook failures are fail open. They do not deny a tool or block a session. A failed refresh remains
visible through the worker state and log files, and the next startup still instructs Codex to use
`recall_search`, `recall_evidence`, and `recall_stats`.
