# Memory decision observability

RE-call treats retrieval as a memory decision, not only a similarity lookup. The article
[Your Agent Doesn't Have a Reasoning Problem, It Has a Memory Problem](https://dev.to/royanannya/your-agent-doesnt-have-a-reasoning-problem-it-has-a-memory-problem-49me)
describes the same boundary in three layers: durable history, deterministic state computation, and
minimal working context. Its comments add the operational requirement that memory needs its own
observability surface.

RE-call already exposes timing, freshness, calibration, provenance, validity, supersession, and
trust verdicts. With `explain=true`, the `recall_search` and `recall_evidence` MCP tools (and the
in process `recall_mcp.retrieval.search_memory` and `evidence_memory` they call) also return
`explanation.details.memory_audit`, built by `recall.explanations.memory_audit`.

## What the audit reports

Facts the retrieval layer can observe about this one decision:

| Field | Meaning |
|---|---|
| `retrieved_count`, `trusted_count`, `untrusted_count` | hits returned, hits with verdict `ok`, and the rest |
| `verdict_counts` | hits per trust verdict, for example `ok`, `superseded`, `expired`, `dependency_invalidated` |
| `distinct_source_count`, `source_diversity` | distinct source files among the hits, and that count over `retrieved_count` |
| `validity_declared_count`, `validity_coverage` | hits declaring `valid_from` or `valid_until`, and their share |
| `supersession_declared_count` | hits with a recorded successor (`superseded_by`) |
| `stale_count`, `stale_rate` | hits whose verdict is `superseded` or `expired`, and their share |
| `context` | evidence selection: `selected_count`, `selection_ratio`, `distinct_source_count`, `status` |

`stale_count` deliberately excludes `not_yet_valid` and `not_yet_known`: a memory that is not valid
yet, or was not held at a historical instant, is a temporal boundary result rather than a stale
one. `dependency_invalidated` hits appear in `verdict_counts` and `untrusted_count` but not in
`stale_count`.

`context` has the same four keys on both tools. On `recall_search` its values are `null` with
`status: not_applicable`, because a search returns the retrieval set itself. On `recall_evidence`
it has `status: selected` and describes the passages the evidence boundary admitted to the bundle.
When related expansion ran, the audited pool includes the related candidates the bundle was
selected from.

## What it does not report

`quality.retrieval_precision`, `forgetting.forget_rate`, and `influence.action_influenced` are
`null` with `status: not_measured`. Precision needs labelled queries, forgetting needs lifecycle
events, and action influence needs feedback from the consumer. A retrieval response must not
manufacture any of those values.

The context selection ratio means that the evidence boundary admitted a passage. It does not mean
that a downstream generator used that passage in its answer.

## Boundaries

The audit is opt in, contains counts and library authored status strings only (no corpus text, no
file names, no chunk identifiers), and does not change ranking, trust, abstention, or the default
response shape: without `explain=true` the response has no `explanation` field at all.

This leaves a clean path for later evaluation. Offline labelled runs can populate precision,
lifecycle instrumentation can populate forgetting, and a consumer can submit action feedback. Those
measurements should stay separate from the runtime audit until their data source and denominator
are defined.
