# Apparatus addendum: original model query construction

Date: 2026-08-25

This addendum freezes the execution apparatus for the prediction in
`2026-08-25-original-model-query-construction.md`. It does not alter the prediction or decision
thresholds.

## Original model

* Provider: OpenRouter chat completions.
* Model: `deepseek/deepseek-v4-pro`.
* Reasoning effort: `medium`.
* Temperature: `0`.
* Maximum output tokens: `1024`.
* Request timeout: `180` seconds.
* Maximum retries: `3`.

## RE call protocol

* Server tool: `recall_query_construction_challenge`.
* First phase: original prompt plus the frozen initial query, with no frame.
* Continuation phase: the returned frame plus the returned generation identity.
* Maximum construction rounds: `2`.
* Maximum candidates per round: `3`.
* Graph mode: `one_hop` for the graph recovery run.
* Tenant: `memory`.
* Embedder: `voyage:voyage-4`.
* Retrieval profile: `fast`.
* Index root: `/home/sentiment/recall-repos/memory`.
* Benchmark concurrency: two independent cases may run concurrently; each case remains
  sequential internally and final artifact row order is fixed to arm/input order. The bounded
  concurrency is an execution setting, not an additional treatment arm.
* Crash recovery: every completed arm/input row is fsync'd to a JSONL checkpoint sidecar. A
  restart must pass `--resume` with the same frozen input and model/retrieval settings; incomplete
  rows are rerun and completed rows are reused only after their input digest matches.

Gold labels remain in the scoring input only. They are not included in the original prompt,
challenge prompt, frame request, or MCP request.

## Artifact requirements

The runner records the input SHA256, model settings, raw challenge prompts, raw model frames,
provider usage, every MCP response, generation identity, graph diagnostics, and the final summary.
The addendum must be committed before the first smoke measurement.
