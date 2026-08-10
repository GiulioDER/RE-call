# RE-call Reasoning Contract

Version: 0.1.0

Status: Session 1 baseline and control harness. Reasoning is not enabled by this document.

## Purpose

RE-call remains a retrieval system first. The planned reasoning layer may assemble and explain
relationships across retrieved memories, but it must preserve the existing retrieval, trust,
evidence, generation, calibration, and MCP contracts.

The reasoning layer will be provider neutral. Typed Python APIs are the primary interface. MCP and
CLI integrations consume those APIs and must not define separate semantics.

## Existing Contracts

Retrieval contract:

* `recall.retriever.HybridRetriever.search()` returns `RetrievalResult`.
* Results contain ranked `ScoredChunk` values, `gap_warning`, `staleness`, and retrieval diagnostics.
* Dense cosine remains the reported score basis, including sparse and fused candidates.
* Existing top k ordering and candidate pool semantics must not change when reasoning is absent.

Trust contract:

* `recall.trust.trusted_search()` is the recommended agent facing search entry point.
* `TrustedResult.hits` is ordered with verdict `ok` hits first, then demoted hits.
* Strict trust is the production default. Missing, stale, uncertified, or mismatched calibration may
  refuse before retrieval.
* Development degradation is explicit through `trust_state="degraded"` and `failure_code`.
* `Verdict` is corpus trust state, not reasoning state. Current values include `ok`, `superseded`,
  `expired`, `not_yet_valid`, `not_yet_known`, `low_confidence`, `invalid_metadata`,
  `ambiguous_supersession`, `not_entailed`, and `unverified`.

Entailment contract:

* `recall.entailment.apply_entailment()` is an optional post processing stage.
* It judges only verdict `ok` hits.
* Non entailed hits are demoted to `not_entailed`.
* The stage is off by default and must not alter retrieval behavior unless explicitly supplied.

Evidence contract:

* `recall.evidence.build_evidence_bundle()` admits only verdict `ok` hits.
* It preserves trusted retrieval order.
* It does not deduplicate semantically.
* It does not retrieve neighbors.
* `render_evidence_prompt()` returns a fixed system prompt and JSON escaped user data.
* Corpus text must never enter system instructions.
* `validate_answer()` is structural only. It checks shape and citation identity, not factual support.

Generation contract:

* `generate_from_evidence()` is generator neutral.
* It short circuits abstained or empty bundles before invoking a generator.
* It requires citations to resolve to evidence bundle chunk ids.
* It returns the evidence bundle with the generated answer.

Calibration contract:

* Calibrated means certified and exactly bound to the active generation.
* Legacy or caller supplied thresholds do not create a production trust claim.
* A default threshold is a degraded fallback, not evidence of certification.

Generation and index contract:

* Generations are immutable serving units.
* Active generation identity travels in diagnostics and evidence.
* Reasoning graphs are derived from one index generation and immutable for that generation.
* A reasoning graph must be recomputed for a new generation rather than patched in place.

MCP contract:

* MCP tools expose the typed Python semantics.
* `recall_search` returns trust evaluated retrieval and advice.
* `recall_evidence` returns the evidence boundary without selecting or invoking a generator.
* MCP advice is library authored and must label raw corpus fields as data.
* MCP must not add a reasoning shortcut that bypasses trust, evidence, calibration, or authorization.

## Reasoning Vocabulary

Claim:

A proposition that may be asked about, supported, contradicted, refined, or inferred. A claim can be
authored evidence when it appears in corpus text, or inferred structure when the reasoning layer
proposes it.

Entity:

A named or described referent such as a service, project, person, configuration, file, experiment,
or issue. Entity resolution can be ambiguous and must remain explicit.

Event:

Something that happened at a known or unknown time. Events may support temporal reasoning, but they
do not override declared validity or transaction time.

Decision:

An authored or inferred choice with a subject, rationale, effective context, and optional successor.

Memory:

An indexed corpus unit with text, source, provenance, validity fields, and optional authored
metadata. A memory is evidence only through retrieved, trusted chunks.

Question:

The caller supplied information need. Reasoning answers the question only from certified evidence
plus explicitly marked inference proposals.

Contradiction:

Two or more claims cannot all be true in the same resolved context. Contradiction detection creates
a reasoning finding, not an automatic corpus edit.

Support:

An authored evidence item or an inference step that makes a claim more likely. Support must identify
which facts it uses.

Refinement:

A narrower or more specific claim that preserves the parent claim while adding constraints.

Dependency:

A claim whose answer depends on another claim, event, decision, entity resolution, or missing edge.

Proposed supersession:

An inferred relationship that one memory likely replaces another. It is a proposal only. It must not
be represented as `supersedes`, `superseded_by`, or any other authored metadata unless a human or
corpus writer explicitly records it.

## Reasoning Outcomes

`answer`:

The system found certified evidence sufficient to answer and returns the answer, supporting facts,
inference steps, uncertainties, and citations.

`abstain`:

The system cannot answer from certified evidence and returns the reason. No unsupported answer is
produced.

`needs_clarification`:

The question is under specified, commonly because an entity, time, scope, or requested relation is
ambiguous.

`needs_review`:

The system found contradictions, likely supersession, security sensitive ambiguity, stale corpus
state, or another condition that should be handled by a human before the answer is trusted.

## Invariants

Every reasoning output must distinguish authored evidence from inferred structure.

Every supporting fact must cite a retrieved evidence item, including chunk id and source.

Every inference must cite the facts or inference steps it depends on.

Every uncertainty must be represented explicitly.

Every refusal or abstention must include a machine readable reason.

Inferred relationships are proposals. They are never written to corpus metadata automatically.

No untrusted corpus text may enter system instructions, MCP advice, tool descriptions, or developer
controlled prompt text.

Reasoning cannot promote `unverified`, `low_confidence`, `expired`, `superseded`,
`ambiguous_supersession`, `invalid_metadata`, `not_yet_known`, `not_yet_valid`, or `not_entailed`
hits into authored evidence.

Reasoning graphs are immutable per index generation.

Reasoning must preserve existing public APIs and retrieval behavior when disabled.

## Baseline Fixture Requirements

The Session 1 fixture set is frozen in `recall/eval/reasoning_session1.json`.

It must contain cases for:

* Direct answer.
* Multi hop answer.
* Near miss.
* Contradictory memories.
* Missing supersession edge.
* Ambiguous entity reference.
* Empty corpus.
* Stale corpus.

Labels must be written independently of the baseline retrieval outputs. Retrieval misses, abstains,
and unresolved relationships must remain visible as baseline behavior rather than repaired by the
fixture.

## Non Goals

Session 1 does not ship a production reasoning answerer.

Session 1 does not infer corpus metadata.

Session 1 does not change retrieval ranking, trust verdicts, calibration, generation promotion, MCP
authorization, or evidence prompt rendering.

Session 1 does not use model generated labels for the frozen fixture.

Session 1 does not claim multi hop reasoning quality from single hop retrieval metrics.

Session 1 does not make `not_entailed` mandatory.

Session 1 does not hide contradictory, ambiguous, empty, or stale states behind a best effort answer.

## Session 1 Audit Checklist

Public API compatibility:

* New Session 1 APIs are additive under `recall.eval`.
* Existing `recall`, `recall_mcp`, CLI, and MCP tool shapes are unchanged.

System instruction safety:

* Evidence rendering keeps corpus text inside JSON escaped user data.
* Reasoning fixtures and baseline metrics are data files, not prompts.
* No fixture memory text is interpolated into a system instruction.

Inferred edge safety:

* Missing supersession cases use `expected_proposals`.
* They do not use authored `supersedes` or `superseded_by` metadata.

Label audit:

* Labels are stored separately from baseline retrieved ids.
* Contradiction, ambiguity, stale, empty, and missing edge cases are not counted as direct answer
  wins.
* Baseline outputs are recorded as control observations, not as ground truth.

## Session 3 Inference Proposal Protocol

Session 3 adds the provider neutral proposal protocol in `recall.reasoning_proposals` and the
protocol specification in `docs/INFERENCE_PROPOSALS.md`.

It defines typed ports for claim extraction, entity resolution, relation proposal, contradiction
detection, and optional model backed proposal generation. It also defines `InferenceProposal`, which
records concrete source evidence ids, relation endpoints, explanation, provider identity, model
identity, pipeline identity, confidence, uncertainty, generation identity, and proposal status.

The deterministic implementation proposes only candidate or review required relationships. It
covers explicit version naming, direct textual references, repeated decision subjects, temporal
ordering, and contradictory validity windows. Authored supersession edges remain more authoritative
than inferred candidates.

Provider output is accepted only after shape, cardinality, generation, pipeline, provider identity,
relation, confidence, canonical identity, duplicate id, and evidence citation validation. Provider
batches are atomic: one malformed item rejects that provider batch without leaking earlier items.
Timeout, malformed output, wrong cardinality, and provider errors are returned in band. Rejected
provider proposals are recorded separately rather than silently dropped.

The Session 3 control artifact is `results/reasoning_session3_proposals.json`, generated by
`python -m recall.eval.reasoning_session3`. It records the proposal protocol reference, precision
and recall on synthetic missing relationships, rejected examples, provider failure matrix, and
side effect audit flags.

## Session 4 Multi Hop Evidence Planner

Session 4 adds `recall.reasoning_planner.plan_multi_hop_evidence()`, a typed, provider neutral
planner that starts from `TrustedResult` and a `ReasoningGraphProjection`. It expands only through
explicit graph operations:

* Retrieve related claims from the same projected source.
* Follow authored graph relationships.
* Compare candidate memories selected by inference proposals.
* Search for missing intermediate evidence.
* Check temporal consistency.
* Check contradiction proposals.

The planner has hard budgets for reasoning steps, graph nodes, model calls, evidence tokens, and
wall time. Budget failures return `outcome="failed_closed"` and never convert inferred proposals
into trusted evidence. Proposal assisted expansion is recorded separately in the trace as
exploration, with `trusted_evidence=False`.

The returned `ReasoningPlan` contains the initial retrieval, expansion steps, accepted evidence,
rejected evidence, unresolved gaps, proposal exploration traces, and typed budget usage. Budget usage
fields are `steps`, `graph_nodes`, `model_calls`, `evidence_tokens`, and `wall_time_ms`, with units
matching `ReasoningBudget`. Retrieval abstention, retrieval to graph binding mismatch, missing graph
support, ambiguous graph diagnostics, unsupported proposal citations, temporal inconsistency,
contradictions, and budget exhaustion all fail closed.

## Session 6 Evaluation Controls

The Session 6 fixture set is frozen in `recall/eval/reasoning_session6.json`.

It must contain cases for:

* Direct question answering.
* Multi hop composition.
* Temporal reasoning.
* Supersession recovery.
* Near miss abstention.
* Contradiction detection.
* Entity disambiguation.
* Missing evidence detection.
* Clarification decisions.

It compares:

* Current retrieval.
* Retrieval plus entailment.
* Retrieval plus authored graph traversal.
* Retrieval plus proposal assisted exploration.
* Retrieval plus full bounded planner.
* Nearest neighbor control.
* Shuffled edge control.
* Removed edge control.

It reports answer accuracy, citation precision, unsupported claim rate, correct abstention rate,
false abstention rate, proposal precision and recall, contradiction detection precision, latency,
model calls, token use, and cross generation reproducibility.

Synthetic controls and real corpus controls must remain separated in the artifact. Pre registered
thresholds must be loaded from fixture data before scoring. Per query observations must remain
available alongside aggregate metrics. Every claimed improvement must survive nearest neighbor,
shuffled edge, removed edge, and heldout controls.

The loader must reject exact expected answer facts leaked into supporting memory metadata. Labels
remain fixture data, not observation outputs. Provider and generation identities must be recorded
with every observation. Benchmark gains that come only from unsupported inference are rejected.
