# Codex and RE-call smoke wiring rerun

This record is frozen before the rerun. Earlier smoke artifacts remain preserved. This run tests
whether the SSH based RE-call MCP server can be exposed to Codex when both arms use the same
controlled sandbox in an isolated worktree.

## Prediction

The `recall_on` arm should make exactly one RE-call MCP search for the specified phrase. The
`recall_off` arm should make zero RE-call MCP calls. Both arms should complete the same read-only
task successfully. Tokens and wall time remain descriptive only.

## Fixed configuration

| Field | Value |
|---|---|
| Run identifier | `smoke-2026-08-19-rerun-02` |
| Task count | `1` |
| Repetitions per task | `1` |
| Model | `gpt-5.6-luna` |
| Reasoning effort | `high` |
| Service tier | `priority` |
| Sandbox | `danger-full-access` for both arms, isolated worktree only |
| Prompt | The exact prompt is recorded in `scripts/launch_codex_ab_smoke.py`. |
| Repository revision | The current signed rerun commit. |
| Working directory | This isolated worktree. |
| Pair concurrency | `1`, with both arms started concurrently inside the pair. |
| RE-call on configuration | Temporary `CODEX_HOME` with the `recall-memory` MCP server. |
| RE-call off configuration | Temporary `CODEX_HOME` with no RE-call MCP servers. |
| Quality evaluator | None for this smoke run. |

## Primary smoke endpoints

1. Process completion in each arm.
2. RE-call MCP call count.
3. Captured input tokens, output tokens, model turns, and wall time.

The numbers are not generalizable from one task and must not be used as a product performance
claim.

## Artifact and privacy rules

The run writes raw JSONL records and a summary under `benchmarks/artifacts/codex_ab/`. The output is
private run evidence and must be reviewed for sensitive memory content before any publication.
Temporary Codex authentication copies are removed when the launcher exits.

## Launch command

```text
python scripts/launch_codex_ab_smoke.py
```
