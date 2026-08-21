# Codex paired benchmark

This package is the application level benchmark seam for comparing an agent with RE-call enabled
and disabled. It does not run a performance measurement by itself. A future measurement must be
pre-registered and committed before the runner is invoked.

Start from [PREREGISTRATION.md](PREREGISTRATION.md) when defining a run.
The current transport finding and the next execution gates are recorded in
[CONTINUATION_PLAN.md](CONTINUATION_PLAN.md).

## Codex CLI adapter

The optional adapter consumes the machine readable stream from `codex exec --json` and records
turn usage, agent messages, tool calls, thread ID, exit status, and wall time. It never chooses the
RE-call configuration. Supply separate `CODEX_HOME` values or configuration arguments for the two
arms:

```python
from benchmarks.codex_ab import (
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

Ragas is optional, and deliberately **not** a `recall-rag` extra. Install it directly:

```text
pip install 'ragas>=0.4,<1'
```

**Why not an extra.** CI's `audit` job scans `uv export --all-extras`, so anything declared as an
extra is scanned whether or not it is installed. As of 2026-08-21 ragas 0.4.3 carries CVE-2026-6587
(SSRF through `retrieved_contexts`, exploit public) and pulls `diskcache` 5.6.3, which carries
CVE-2025-69872 (pickle deserialisation RCE). Neither has a fix version. Declaring the extra would
turn a currently green repository wide security gate red and keep it red until upstream ships
fixes, which is how a gate stops being read.

Neither vulnerability is reachable through this adapter as written: the import is lazy and
`metrics` is supplied by the caller, so the vulnerable multi modal faithfulness path runs only if
something explicitly selects that metric. That is an argument for installing ragas deliberately,
not for making every `--all-extras` resolution carry it. Revisit when a patched release exists.

The adapter converts single turn records to `SingleTurnSample` objects and structured conversations
to `MultiTurnSample` objects. Use Ragas for answer quality, faithfulness, goal completion, and tool
call metrics. Keep RE-call's deterministic retrieval, trust, supersession, and abstention metrics as
the primary memory layer measures.

## Reporting

Publish the task manifest, raw JSONL records, source revision, model configuration, evaluator model,
summary code, incomplete task IDs, and all failed cases. A side by side live demo is useful evidence
for communication, but repeated paired runs are required for a performance claim.
