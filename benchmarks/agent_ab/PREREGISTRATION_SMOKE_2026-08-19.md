# Codex and RE-call smoke run

This record is frozen before the launch. It is a wiring demonstration, not a performance claim.

## Prediction

For the same read-only task, both arms should complete successfully and produce the same three
requested facts. The `recall_on` arm should make at least one RE-call MCP call. The `recall_off`
arm should make zero RE-call MCP calls. Token counts and wall time are recorded descriptively only.

## Fixed configuration

| Field | Value |
|---|---|
| Run identifier | `smoke-2026-08-19` |
| Task count | `1` |
| Repetitions per task | `1` |
| Model | `gpt-5.6-luna` |
| Reasoning effort | `high` |
| Service tier | `priority` |
| Sandbox | `read-only` |
| Prompt | The exact prompt is recorded in `scripts/launch_codex_ab_smoke.py`. |
| Repository revision | The current `claude/ragas-codex-ab` worktree at launch. |
| Working directory | This worktree, shared read-only by both arms. |
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
