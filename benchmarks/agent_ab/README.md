# Paired agent benchmark

This package is the application level benchmark seam for comparing an agent with RE-call enabled
and disabled. It does not run a performance measurement by itself. A measurement must be
pre-registered and committed before the runner is invoked.

Start from [PREREGISTRATION.md](PREREGISTRATION.md) when defining a run. The Codex transport
finding that produced the admission gate is recorded in
[CONTINUATION_PLAN.md](CONTINUATION_PLAN.md).

| Module | What it does |
|---|---|
| `schema.py` | the canonical `SessionRecord`; a missing measurement stays `null`, never 0 |
| `runner.py` | paired execution, both arms of a task started together |
| `claude_exec.py` | Claude Code adapter over `claude -p --output-format stream-json`; the driver of every archived baseline, kept byte-stable |
| `sdk_exec.py` | Claude Agent SDK adapter (`--driver sdk` on the task runner); normalizes typed messages back into the stream shapes and reuses `claude_exec`'s parsing core, so the two drivers share one field mapping |
| `codex_exec.py` | Codex CLI adapter over `codex exec --json` |
| `arms.py` | the three arm profiles, and the only place they may differ |
| `recall_server.py` | one pre-warmed authenticated RE-call server for the whole run |
| `gate.py` | refuses any pair that cannot prove the treatment was applied |
| `traps.py` | deterministic hazard checks, and where each hazard's fact lives |
| `summarize.py` | paired deltas, dependency free |
| `ragas_adapter.py` | optional conversion to Ragas samples |

## Run order

```bash
python scripts/agent_ab_gate.py       # prove the on arm has RE-call and the off arm does not
python scripts/agent_ab_qualify.py    # classify each trap, and COMMIT the result
# commit the preregistration, then run the measurement
```

Both scripts need a gateway credential in the environment (`ANTHROPIC_BASE_URL` and
`ANTHROPIC_AUTH_TOKEN`) and the corpus container running.

## The two rules this package exists to enforce

**Prove the treatment reached the arm.** A `recall_on` session whose MCP server was still
`pending` runs with no RE-call tool and reports `"subtype": "success"`, `"is_error": false`, zero
denials. Averaged in, that is a null result manufactured by a wiring fault. `gate.py` requires a
`mcp__recall*` name in the session's own `system/init` tool list and discards the whole **pair**
otherwise, reporting the task id. A tool that was *available and never called* is admitted and
counted: that is a behavioural result, not a fault, and conflating the two is the original mistake.

**Decide what counts before you look.** `traps.py` classifies every hazard by where its fact is
actually reachable, from the live corpus and the static prompt, and the result is committed before
any session runs. The first trap set written here scored zero `memory_only` traps, because the
hazards anyone bothers to write into `CLAUDE.md` are the ones that fit in `CLAUDE.md`. Traps the
memory layer is expected to **lose** stay in the set on purpose.

## Codex CLI adapter

The optional adapter consumes the machine readable stream from `codex exec --json` and records
turn usage, agent messages, tool calls, thread ID, exit status, and wall time. It never chooses the
RE-call configuration. Supply separate `CODEX_HOME` values or configuration arguments for the two
arms:

```python
from benchmarks.agent_ab import (
    CodexExecConfig,
    RECALL_OFF,
    RECALL_ON,
    make_codex_runner,
    run_paired,
)

configs = {
    RECALL_ON: CodexExecConfig(env={"CODEX_HOME": "/path/to/codex-home-recall"}),
    RECALL_OFF: CodexExecConfig(env={"CODEX_HOME": "/path/to/codex-home-plain"}),
}
records = await run_paired(tasks, make_codex_runner(configs))
```

The official Codex documentation describes the JSONL event stream and its `turn.completed.usage`
fields [here](https://developers.openai.com/codex/noninteractive/). Do not include API keys in task
rows, source files, or saved artifacts.

## Contract

The runner receives one task row and one variant, either `recall_on` or `recall_off`, and returns a
`SessionRecord` or a mapping with the same fields:

```python
async def run_case(row: Mapping[str, Any], variant: str) -> SessionRecord:
    ...

records = await run_paired(tasks, run_case)
summary = summarize_pairs(records)
```

`run_paired` starts both arms of each task concurrently. `pair_concurrency=1` keeps one pair active
at a time, so the two sessions are simultaneous without turning host contention into an untracked
variable.

## Required isolation

The caller must give both arms the same model, prompt, repository snapshot, task input, and resource
limits. The only intended difference is access to RE-call. Each arm needs an isolated session state,
working directory, and memory tenant or immutable memory snapshot.

## Measurement fields

Missing measurements are represented as `null`, never zero. This keeps an unmeasured token or time
value from being reported as a successful zero cost. Model usage and RE-call latency must be recorded
by the session runner. Ragas evaluator usage is a separate cost surface.

## Ragas integration

Ragas is optional. Install it with:

```text
pip install recall-rag[ragas]
```

The adapter converts single turn records to `SingleTurnSample` objects and structured conversations
to `MultiTurnSample` objects. Use Ragas for answer quality, faithfulness, goal completion, and tool
call metrics. Keep RE-call's deterministic retrieval, trust, supersession, and abstention metrics as
the primary memory layer measures.

## Reporting

Publish the task manifest, raw JSONL records, source revision, model configuration, evaluator model,
summary code, incomplete task IDs, and all failed cases. A side by side live demo is useful evidence
for communication, but repeated paired runs are required for a performance claim.
