# Continuation plan

The current harness is ready to run paired Codex sessions, but the latest smoke pair is a wiring
diagnostic rather than evidence for an RE-call treatment effect. Both sessions completed, while the
RE-call arm recorded zero RE-call tool calls.

## 1. Repair and verify MCP startup

The deployed stdio server starts on VPS2 but does not answer `initialize` within the diagnostic
window. Its startup path constructs the Voyage embedder and performs a provider probe before the
MCP transport becomes available. The next implementation step is to use a verified warmed RE-call
endpoint or make that provider probe lazy, without changing retrieval semantics.

The gate before measurement is:

1. An isolated client completes MCP `initialize`.
2. `tools/list` contains the expected RE-call tools.
3. A controlled `recall_search` call succeeds and its latency is recorded.
4. The Codex JSONL stream contains a RE-call tool event in the on arm and none in the off arm.

## 2. Run a preregistered paired benchmark

Create a new immutable preregistration and commit it before launching. Use a new run identifier,
the same model, prompt, repository snapshot, working directory, and resource limits for both arms,
and isolated Codex homes. Keep the two sessions concurrent within each pair.

Use a small smoke task first, then a repeated task set large enough for paired uncertainty estimates.
Record answer correctness, task completion, input tokens, output tokens, total tokens, wall time,
tool calls, RE-call latency, and failures. Do not interpret a run with missing RE-call events as an
RE-call comparison.

## 3. Add Ragas quality scoring

Feed the completed paired records into the optional Ragas adapter for answer correctness,
faithfulness, context relevance, and goal completion where the task format supports each metric.
Keep RE-call retrieval quality, trust decisions, abstentions, and supersession behavior as separate
deterministic measures. Record evaluator model and evaluator token cost independently from Codex
usage.

## 4. Produce the marketing demonstration

After the measurement gate passes, run two visibly synchronized sessions with the same task. Show
the live answer, elapsed time, token counters, tool activity, and the RE-call latency contribution.
Label this as a demonstration and place the repeated paired results beside it so the presentation
does not turn one favorable smoke pair into a general performance claim.

## 5. Report conservatively

Publish aggregate results with paired deltas and confidence intervals. Report quality first, then
token and time changes, plus the retrieval overhead. Keep raw session transcripts outside the public
repository unless they are deliberately redacted and approved for release.
