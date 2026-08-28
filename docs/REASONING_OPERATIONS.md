# Reasoning Operations

This document is the Session 7 control artifact for the reasoning layer. It covers integration
compatibility, operational policy, failure and outage behavior, and exported metrics.

## Integration compatibility report

Existing retrieval clients remain compatible.

* `recall_search` is unchanged.
* `recall_evidence` is unchanged.
* New MCP tools are additive: `recall_reasoning_query`, `recall_reasoning_projection`,
  `recall_reasoning_proposals`, and `recall_reasoning_audit`.
* New CLI commands are additive under `recall reasoning`: `projection`, `proposals`, `query`,
  `trace`, and `audit`.
* Reasoning is explicit opt in. No retrieval command enters reasoning mode by omission.
* Reasoning responses carry trust state, tenant id, generation id, calibration status, proposal
  status, refusal reason, and diagnostics.
* Reasoning proposals are review candidates only. They are never promoted into corpus metadata by
  the API, CLI, or MCP server.

Evidence Graph V1 is an additional opt in path. `ReasoningPolicy.graph_expansion` accepts `off`
or `one_hop`, and `ReasoningBudget.max_graph_hops` accepts only the matching value `0` or `1`.
The default is `off`, so ordinary retrieval and existing reasoning behavior do not traverse the
semantic graph.

When enabled, trusted chunks seed exact entity mentions. Authored semantic relations select
neighboring chunks, which are then evaluated again by the ordinary trust layer. Graph relations do
not promote evidence, replace authored frontmatter, or allow model generated proposals to drive
traversal. Ambiguous entities, unavailable graph rows, fingerprint mismatches, and legacy
generations fail closed with a typed graph readiness result while preserving original trusted
evidence.

Graph precision tuning is an internal admission policy for `one_hop`; it does not add a public
mode and does not affect `off`. The combined policy applies directional outgoing traversal for
positive relations, keeps `contradicts` and `same_entity` diagnostic or identity only, accumulates
distinct seed and relation corroboration, suppresses high degree entity hubs unless the query has
an exact alias, applies a relative query cosine gate, and refuses expansion when the initial
retrieval is already sufficient. The default hub threshold is 32 chunks and the default cosine
margin is 0.10. Every graph response includes a policy fingerprint and sanitized admission reason
counters so evaluation artifacts cannot mix policies. These internal evaluation variables are not
part of the public request surface:

* `RECALL_GRAPH_PRECISION_VARIANT` selects one isolated tuning arm or `combined`.
* `RECALL_GRAPH_RELATION_CONTROL` selects `none`, `shuffled`, or `removed` for evaluation only.
* `RECALL_GRAPH_HUB_DEGREE_THRESHOLD` accepts 16, 32, or 64.
* `RECALL_GRAPH_COSINE_MARGIN` accepts 0.05, 0.10, or 0.15.

All graph candidates still pass normal trust evaluation and retain their original chunk citation.
The precision evaluation protocol is recorded in
`benchmarks/PREREGISTRATION-evidence-graph-precision-tuning-v1.md`.

The core library does not require a managed database or a managed reasoning service. The core uses
typed Python APIs and provider ports. PostgreSQL is one supported durable store for RE-call
retrieval and generation serving, not a managed reasoning dependency.

## Operational policy

Strict production mode:

* Default trust policy is strict.
* Missing calibration, stale calibration, missing generation identity, lineage mismatch, or database
  unavailability produces a typed refusal or exception before an unverified answer is emitted.
* Production generation builds require immutable manifest inputs.
* Local filesystem indexing is development only.

Development exploration mode:

* Development mode requires explicit `RECALL_TRUST_MODE=development` or a direct
  `TrustPolicy.development()` object.
* Development mode may return degraded evidence for inspection, but the payload labels
  `trust_state="degraded"` and carries the failure code.
* Development mode responses must not be treated as production answers.

Human review queues:

* `requires_review` proposals enter review, not the trusted corpus.
* `review_required` policy converts any proposal assisted result with proposals into
  `needs_review`.
* Provider failures during proposal generation return `needs_review` with
  `refusal_reason="provider_failure"`.

Provider outages:

* Optional provider failures are represented as `ProviderFailure` records.
* Proposal provider failures return `needs_review` and do not invoke the answer provider.
* Retrieval expansion failures preserve the initial trusted evidence and may continue with the
  baseline answer path.
* The reasoning response carries provider id, model id, provider revision, failure kind, and
  sanitized message.

Retrieval expansion configuration:

* `RECALL_REASONING_EXPANSION=1` is required to enable the provider. It is off by default.
* `RECALL_REASONING_EXPANSION_MODEL` and `RECALL_REASONING_EXPANSION_API_KEY` are required when
  enabled.
* `RECALL_REASONING_EXPANSION_BASE_URL` defaults to OpenRouter and must be an absolute HTTP or
  HTTPS URL.
* `RECALL_REASONING_EXPANSION_TIMEOUT` defaults to 30 seconds and must be finite and positive.
* The bare `RECALL_REASONING_API_KEY`, `RECALL_REASONING_BASE_URL` and `RECALL_REASONING_TIMEOUT`
  are read as legacy fallbacks when the infixed spellings are unset. They are shared with other
  reasoning arms, which is why the infixed names exist and win.
* `RECALL_REASONING_EXPANSION_EFFORT` defaults to `minimal`.
* `RECALL_REASONING_EXPANSION_REVISION` defaults to `unpinned` and should be pinned in run records.
* `RECALL_REASONING_EXPANSION_COST_PER_1K_TOKENS` is optional nonnegative cost metadata.
* The live MCP tool must also receive `expand_retrieval=true`. Ordinary search and evidence tools
  remain unchanged.
* Depth expansion runs first. The model is called only when depth still reports an evidence gap,
  with one bounded model call and at most three generated retrieval queries.

Generation retirement:

* Reasoning projections are derived from the visible store generation.
* A generation change changes the projection identity.
* Retired generations may be inspected only when the store explicitly pins or exposes that
  generation.

Privacy erasure and rebuild:

* `recall_forget` remains the erasure path.
* Reasoning projections are rebuilt from current store visibility.
* Inferred proposals are recomputed from the rebuilt projection and are not durable trusted corpus
  metadata.
* If erasure changes the corpus fingerprint, strict production mode requires recalibration before a
  trusted answer.
* Graph rows are generation bound and are removed with their supporting chunks. Source erasure
  invalidates the graph readiness marker until the deterministic graph is rebuilt.

## Failure and outage matrix

| Condition | Outcome | Refusal reason or code | Corpus text in error |
| --- | --- | --- | --- |
| Empty query | needs clarification | `empty_query` | no |
| Retrieval only policy | abstained | `retrieval_only_policy` | citable evidence may be returned as data |
| Missing calibration in strict mode | abstained or `TrustRefusal` | `CALIBRATION_MISSING` or `uncertified_evidence` | no strict error text |
| Stale or uncertified calibration | abstained or `TrustRefusal` | `CALIBRATION_STALE` or `CALIBRATION_UNCERTIFIED` | no strict error text |
| Missing generation identity in production | refusal | `INDEX_NOT_READY` | no |
| Database outage | dependency failure | `DEPENDENCY_UNAVAILABLE` or raised store exception | no deliberate corpus echo |
| Proposal provider timeout | needs review | `provider_failure` | no |
| Proposal provider malformed output | needs review | `provider_failure` | no |
| Reasoning budget exhausted | abstained or needs review | `budget_exhausted` | trace ids only |
| Ambiguous graph evidence | needs review | `ambiguous_evidence` | trace ids only |
| Privacy erasure before rebuild | strict refusal until rebuilt and calibrated | lineage or calibration failure code | no |

Terminal output and structured output are treated separately. Human CLI summaries are terminal
safe. Structured JSON may include corpus text only in explicit evidence fields, never in advice or
error channels.

## Metric specification

Reasoning metrics are in the in process `METRICS` registry and are exposed through `recall_stats`.

Counters:

* `recall_reasoning_outcome_total{outcome,trust_state,refusal_reason}` counts answered, abstained,
  clarification, and review outcomes.
* `recall_reasoning_proposals_total` counts emitted inference proposals.
* `recall_reasoning_review_total{reason}` counts review outcomes.
* `recall_reasoning_budget_exhausted_total` counts runs that ended with the explicit
  `budget_exhausted` stop reason.
* `recall_reasoning_provider_failure_total{kind,provider_id,model_id}` counts optional provider
  failures.

Evidence Graph V1 metrics:

* `recall_graph_build_total` and `recall_graph_build_failure_total` count generation graph builds.
* `recall_graph_query_total`, `recall_graph_expansion_total`, `recall_graph_candidates_total`,
  `recall_graph_rejected_candidates_total`, and `recall_graph_diagnostics_total` describe query
  expansion work.
* `recall_graph_relations_rejected_total{reason}` and
  `recall_graph_candidates_rejected_total{reason}` count sanitized admission refusals.
* `recall_graph_gate_refused_total{reason}` counts selective gate refusals, and
  `recall_graph_policy_total{policy}` records the active policy fingerprint prefix.
* `recall_graph_latency_ms` records graph build and expansion latency.

Histograms:

* `recall_reasoning_latency_ms` records end to end reasoning latency.
* Existing retrieval histograms continue to record retrieval stage latency.
* Existing MCP tool latency histograms include the new reasoning tools with `tool` labels:
  `reasoning_query`, `reasoning_projection`, `reasoning_proposals`, and `reasoning_audit`.

Cost:

* The core library records model call budget usage in `ReasoningBudgetUsage.model_calls`.
* No managed model provider is required by the default reasoning tools, which remain deterministic.
  Retrieval expansion is separately configured through the variables above and is never the answer
  judge.
* Provider specific monetary cost should be added by provider adapters as library authored numeric
  fields or metrics, never as corpus controlled text.
