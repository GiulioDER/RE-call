# Reasoning API

Session 5 adds a typed Python reasoning entry point in `recall.reasoning`.

## Public API Specification

Primary entry point:

```python
from recall.reasoning import reason

response = reason(request)
```

Request type: `ReasoningRequest`

Fields:

1. `query`: natural language query.
2. `tenant_id`: tenant boundary the result must remain inside.
3. `generation`: `GenerationSelection` with optional `generation_id`, `pipeline_fingerprint`, and `corpus_fingerprint`.
4. `known_as_of`: optional transaction time constraint. Retriever ports should pass this through to `trusted_search(..., known_as_of=...)`.
5. `policy`: `ReasoningPolicy`.
6. `budget`: `ReasoningBudget`, shared with the planner.
7. `evidence_policy`: `EvidencePolicy`, shared with evidence assembly.
8. `providers`: `ReasoningProviderPorts`.

Provider ports:

1. `retriever`: required, returns `TrustedResult`.
2. `graph_provider`: optional, returns `ReasoningGraphProjection`.
3. `proposal_provider`: optional, returns proposals or a `ProposalProtocolReport`.
4. `answer_provider`: optional, consumes the existing evidence prompt pair and returns `AnswerEnvelope` JSON.
   The MCP server can provide a local Ollama adapter when `RECALL_REASONING_ANSWER_ENABLED=1`.
5. `expansion_provider`: optional, returns bounded untrusted retrieval proposals.
6. `expansion_retriever`: optional, executes proposals and must preserve tenant, generation, trust,
   and calibration binding.

Response type: `ReasoningResponse`

Fields include outcome, answer or clarification request, trusted evidence, inference proposals, provider failures, reasoning trace, contradictions, unsupported gaps, citations, calibration identity, generation identity, trust state, refusal reason, and diagnostics.

Retrieval expansion is depth first and bounded to two retrieval rounds, one provider call, and at
most three generated queries. A successful depth pass that no longer reports an evidence gap skips
the model provider. Provider input is bounded before serialization. Expansion proposals never
become evidence or citations until normal trusted retrieval accepts them.

Provider failures are structured records rather than exceptions in the public response. A proposal provider outage, timeout, or malformed provider report returns `outcome="needs_review"` with `refusal_reason="provider_failure"`, includes `provider_failures`, and does not invoke the answer provider.

Outcomes are distinct:

1. `answered`
2. `abstained`
3. `needs_clarification`
4. `needs_review`

Policies:

1. `retrieval_only`: returns the evidence bundle after certification checks pass and never invokes the answer provider. If certification fails, it abstains with the original trust failure.
2. `evidence_assembly`: assembles trusted evidence and may call the answer provider.

The built in local adapter uses Ollama's native `/api/chat` endpoint, including its strict JSON
schema and `think` switch. Configure it with `RECALL_REASONING_ANSWER_BASE_URL`,
`RECALL_REASONING_ANSWER_MODEL`, `RECALL_REASONING_ANSWER_TIMEOUT`,
`RECALL_REASONING_ANSWER_MAX_TOKENS`, `RECALL_REASONING_ANSWER_REVISION`, and
`RECALL_REASONING_ANSWER_THINKING`. The provider is selected with
`RECALL_REASONING_ANSWER_PROVIDER=ollama` and is disabled unless
`RECALL_REASONING_ANSWER_ENABLED=1`. Provider failures are sanitized and never promote evidence.
A local Qwen3 4B model is a suitable starting point for an 8 GB GPU; measure latency and answer
quality on the target machine before enabling it for regular use.
3. `proposal_assisted`: requires a graph provider, records proposals, and runs bounded planning.
4. `review_required`: same as proposal assisted, but returns `needs_review` when proposals are present.

## Original Model Query Construction

`recall_query_construction_challenge` is an additive, read only MCP tool. The first call accepts
`original_prompt` and the original retrieval `query`, then returns a bounded challenge prompt,
trusted retrieval context, and the current generation identity. The calling model answers with the
declared frame fields, then calls the same tool again with `frame` and
`expected_generation_id`.

The continuation frame is a proposal only. The server accepts only bounded JSON fields, limits
each query to 2,000 characters, permits two construction rounds and three candidates per round,
and refuses a continuation whose generation, pipeline, or corpus binding changed. The response
reports accepted and rejected candidates, retrieval calls, model calls, graph diagnostics, and
fallback reasons.

The `original_loop` arm retrieves the model's single revised query. The `pyramid` arm deterministically
derives up to three literal, intent, anchor, or decomposition candidates from the frame. All model
text and controller output remains a proposal. Only ordinary trusted retrieval can become evidence.
The controller permits two rounds, three candidates per round, and one challenge per round. A
generation mismatch refuses continuation before retrieval. Graph expansion is deferred until a
constructed query has produced trusted seed evidence.

The reproducible remote runner is
`scripts/run_query_construction_batch.py`. It calls the original DeepSeek model through the same
OpenRouter settings used by the earlier replay and records challenge prompts, frames, provider
metadata, tool responses, generation identities, graph diagnostics, and scoring inputs. The runner
does not send gold labels to either model or MCP.

## Compatibility Matrix

| Surface | Compatibility result |
| --- | --- |
| `trusted_search` | No signature change. Direct retrieval callers see no behavior change. |
| `TrustedResult` | No field changes. Reasoning consumes it as input. |
| `EvidenceBundle` | No field changes. Reasoning reuses `build_evidence_bundle`. |
| `generate_from_evidence` | No behavior change. Reasoning uses the same prompt rendering, answer parsing, citation normalization, and citation validation helpers without invoking the wrapper that rebuilds evidence. |
| LangChain and LlamaIndex adapters | No changes required. Existing trust state tests still pass. |
| MCP search and evidence tools | No response shape change in this session. They can consume `recall.reasoning` later. |
| Package exports | Additive exports only, sorted `__all__` retained. |

## Answer Validation Test Report

Focused command:

```powershell
python -m pytest tests/test_reasoning_api.py tests/test_evidence.py tests/test_reasoning_planner.py tests/test_integrations_agent_tool_contract.py tests/test_evidence_wiring.py tests/test_mcp_service_search.py tests/test_mcp_smoke.py
```

Result: `70 passed`.

Covered cases:

1. Valid answer requires a trusted citation.
2. Demoted memories cannot be cited.
3. Unretrieved memories cannot be cited.
4. Proposal ids cannot satisfy citation requirements.
5. Cross tenant and cross generation retrieval is refused.
6. Strict policy abstains on degraded trust state before generation.
7. `retrieval_only` does not invoke the answer provider.
8. `review_required` returns `needs_review`, not a generic failure.
9. Empty query returns `needs_clarification` before retrieval.
10. Malformed answer provider output is rejected.
11. Response serialization is strict JSON safe and round trips through `reasoning_response_from_dict`.
12. Deserialization rejects missing or mismatched nested trust state.
13. Deserialization rejects nonfinite numeric strings in diagnostics, evidence, and proposal confidence fields.

Lint command:

```powershell
python -m ruff check recall/reasoning.py tests/test_reasoning_api.py recall/__init__.py
```

Result: `All checks passed!`

Typecheck command:

```powershell
python -m mypy recall/reasoning.py
```

Result: `Success: no issues found in 1 source file`

## Security And Prompt Boundary Review

Evidence remains source only when it is present in a trusted `EvidenceBundle`. The reasoning API does not promote inference proposals into corpus metadata and does not treat proposal ids as citable evidence.

Boundary checks:

1. Retrieval tenant, generation, pipeline fingerprint, and corpus fingerprint are checked against the request when both sides specify them.
2. Graph tenant, generation, pipeline fingerprint, and corpus fingerprint are checked before proposal assisted planning.
3. Proposal `generation_id` must match the graph generation.
4. Proposal source evidence ids must resolve to graph node ids.
5. Answer generation still uses the existing fixed system prompt and JSON escaped evidence payload from `recall.evidence`.
6. Provider output is parsed by the existing strict envelope parser and validated against trusted evidence item ids only.
7. Abstain, review, and clarification outcomes are represented separately.
