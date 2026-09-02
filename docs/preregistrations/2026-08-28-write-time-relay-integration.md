# Write-time relay integration record

**Date:** 2026-08-28  
**Status:** implemented and locally verified in the isolated worktree

## What is implemented

New hook configurations use a per-session loopback TCP relay. The relay:

- keeps one PostgreSQL connection in a detached helper process;
- binds only to `127.0.0.1` and authenticates every request with a random token;
- keeps the DSN out of process arguments and relay logs;
- fixes the tenant and retrieval options from the hook configuration for the helper lifetime;
- reconnects after a dropped database connection with a bounded backoff;
- exits on `SessionEnd`, expires after fifteen minutes of inactivity, and is removed during
  uninstall;
- restarts once when the helper socket is stale or the helper has crashed;
- returns no permission decision and fails open on every relay error.

Older project-bound hook configurations without `connection_mode` retain the cold path. The installer writes
`"connection_mode": "relay"` explicitly, and changing it to `"cold"` is the rollback switch.
The existing hook timeout, cooldown, and lexical SQL path remain unchanged. The relay requires the
client's `session_id`; if the client omits it, the hook falls back to the cold path for that call.
Malformed persisted mode values also fall back to cold, while invalid installer arguments are
rejected. A legacy configuration without `project_root` is intentionally disabled for write-time
retrieval until it is reinstalled, because its project ownership cannot be inferred safely.
The installer records the project root, and the write-time hook emits no memory context when an
event `cwd` is outside that root, preventing a user-level hook from crossing project boundaries.

## Verification

- 62 focused tests pass in the original preregistration snapshot, including installer wiring,
  session shutdown, authenticated transport,
  crash restart, unreachable-database follow-up latency, and the pre-existing hook contract.
- Ruff and Python bytecode compilation pass.
- The unreachable-database subprocess test confirms the first failure is surfaced as a relay
  failure, the next call remains below 500 ms, and the state file is removed by shutdown.
- The preceding live latency record measured 30/30 ordered hit equality and an 84.7 percent
  reduction in all-request median latency for the same SQL path.

Exact verification commands:

```powershell
python -m pytest tests/test_write_time_relay.py tests/test_write_time_hook.py tests/test_recall_hooks.py tests/test_claude_code.py -q
python -m ruff check recall_hooks/relay.py recall_hooks/write_time.py recall_hooks/__init__.py recall/claude_code.py tests/test_write_time_relay.py tests/test_write_time_hook.py tests/test_recall_hooks.py tests/test_claude_code.py
python -m compileall -q recall_hooks recall/claude_code.py
```

The latency result is recorded in
[`2026-08-28-write-time-connection-reuse.md`](2026-08-28-write-time-connection-reuse.md), with
artifact `results/write_time_connection_reuse_20260828T151400Z.json` and its remeasurement
command under **Re-measure**.

The audit hardening also bounds the complete relay search below the five second client budget,
uses the query itself as the relay liveness check, serializes per session state changes, rejects
non-loopback endpoints, limits serialized drafts to 4096 characters, and leaves a tokenized abort
marker when a detached helper misses startup.

### Audit follow-up, 2026-08-28

The audit follow-up measured 72 focused tests with the command above. This is an appended
verification result; the original 62 test claim above remains unchanged. The four local JSON files
under `results/` are pre-existing benchmark outputs and remain intentionally untracked, so they
are preserved but excluded from any release changeset.

The audit also found that the referenced latency artifact metadata is not reconcilable with the
frozen benchmark record: the artifact reports 10 payloads over 3 repetitions, while the frozen
record specifies 30 payloads over 3 repetitions, and the reported reduction uses a non-standard
upper-middle value for an even-sized sample. The original numbers above are not rewritten. That
latency claim is therefore not accepted as release evidence; a new committed preregistration is
required before remeasuring or publishing a performance result.

### Post-gate verification, 2026-08-28

The ownership cleanup fix added two regression cases. The focused suite now passes 74 tests and
the expanded suite passes 92 tests. Ruff, Python bytecode compilation, hook JSON validation, and
`git diff --check` also pass.

## Release boundary

This is code-complete for the relay integration in the isolated worktree. It is not a production
deployment: the original dirty checkout was not modified, no installer artifact was published,
and no client release was cut. A release pass still needs a clean-worktree review and supported
client smoke tests on each target platform.
