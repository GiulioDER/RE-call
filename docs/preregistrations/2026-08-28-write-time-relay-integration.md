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

Older hook configurations without `connection_mode` retain the cold path. The installer writes
`"connection_mode": "relay"` explicitly, and changing it to `"cold"` is the rollback switch.
The existing hook timeout, cooldown, and lexical SQL path remain unchanged.

## Verification

- 62 focused tests pass, including installer wiring, session shutdown, authenticated transport,
  crash restart, unreachable-database follow-up latency, and the pre-existing hook contract.
- Ruff and Python bytecode compilation pass.
- The unreachable-database subprocess test confirms the first failure is surfaced as a relay
  failure, the next call remains below 500 ms, and the state file is removed by shutdown.
- The preceding live latency record measured 30/30 ordered hit equality and an 84.7 percent
  reduction in all-request median latency for the same SQL path.

## Release boundary

This is code-complete for the relay integration in the isolated worktree. It is not a production
deployment: the original dirty checkout was not modified, no installer artifact was published,
and no client release was cut. A release pass still needs a clean-worktree review and supported
client smoke tests on each target platform.
