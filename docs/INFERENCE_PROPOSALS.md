# RE-call Inference Proposal Protocol

Version: 0.1.0

Status: Session 3 protocol and deterministic implementation. Proposal output is not trusted corpus
metadata.

## Purpose

The inference proposal layer identifies possible relationships across projected evidence while
preserving the retrieval and trust contracts. It returns typed Python values that can be consumed by
future MCP and CLI integrations. It does not edit frontmatter, promote trust verdicts, change
retrieval ranking, or mutate a reasoning graph.

## Typed Ports

The public protocol lives in `recall.reasoning_proposals`.

Provider neutral ports:

* `ClaimExtractor`: extracts evidence bound claims.
* `EntityResolver`: proposes canonical entity groupings and ambiguity.
* `RelationProposer`: proposes relationships across claims and entities.
* `ContradictionDetector`: detects incompatible claims.
* `ModelBackedProposalProvider`: optional model backed relation provider.

The deterministic implementation is `deterministic_inference_proposals()`. The orchestration entry
point is `proposal_report()`, which returns deterministic proposals plus optional model backed
proposals and in band provider failures.

## Proposal Schema

`InferenceProposal` contains:

* `source_evidence_ids`: concrete projected node ids.
* `proposed_relation`: `supersedes`, `contradicts`, `same_entity`, `references`,
  `declares_validity`, or `declares_status`. The last two are a document asserting something
  about itself rather than a relation between two documents; `object_id` carries the asserted
  value, for example `valid_from:2026-02-01` or `status:deprecated`. They exist as their own
  relations because recording them as `references` would put a false relation into an audit
  record. Only `supersedes` proposals can become graph candidate edges.
* `subject_id` and `object_id`.
* `explanation`.
* `provider_id`, `model_id`, `provider_revision`, and `pipeline_id`.
* `confidence`.
* `uncertainty`.
* `generation_id`.
* `status`: `candidate`, `rejected`, or `requires_review`.
* `rule_id`.
* immutable metadata.

Proposal identity is deterministic over schema version, generation identity, provider identity,
pipeline identity, relation endpoints, evidence ids, rule id, and status. `PROPOSAL_SCHEMA_VERSION`
is `2`; the bump from `1` changed every proposal id, which is intended, because an id minted under
a relation vocabulary that could not express validity must not be mistaken for one that can.

## Extracted Claims

`recall.truth_extraction` turns memo prose into structured claims on the ingest path, and
`ExtractedClaimProposalProvider` replays validated claims into this protocol as a
`ModelBackedProposalProvider`. It calls nothing itself: the engine ran at ingest.

Every extracted proposal is `requires_review` with `confidence` of `None`. The measured prior is
that the rule based attempt at this problem produced four candidates on a real 792 memo corpus and
all four were wrong on review, so an extracted proposal is a question for a human, not an answer.
`metadata["quote"]` carries the verbatim body span the claim was read from, which is what makes the
review possible at all.

Direction: for a supersession claim read from file F naming target T, `subject_id` is T (the
superseded document) and `object_id` is F. Reversing it would declare the live memo stale and
demote it beneath the one it replaced.

## Deterministic Rules

Session 3 implements high precision non model rules first:

* Explicit version naming, for example `policy_v1.md` to `policy_v2.md`.
* Direct textual references such as replaces, supersedes, deprecates, updates, or revises.
* Repeated decision subjects with later validity starts.
* Contradictory validity windows with overlapping windows and opposing status language.
* Temporal ordering over the same normalized entity.

Explicit authored supersession edges remain more authoritative than inferred candidates. If an
authored edge already exists, deterministic rules do not duplicate it.

## Provider Failure Matrix

Optional providers are validated before their output is accepted. A provider cannot silently add or
drop proposals. Provider batches are atomic: if any item in a batch is malformed, no proposal from
that batch is accepted. Duplicate proposal ids are malformed rather than silently collapsed.
Failures are returned as:

* `timeout`.
* `malformed_output`.
* `wrong_cardinality`.
* `provider_error`.

Malformed output includes missing required fields, unknown relations, invalid or non-finite
confidence values, wrong generation identity, wrong pipeline identity, wrong provider identity,
absent evidence ids, duplicate proposal ids, non-canonical typed proposal ids, or citations to
unknown evidence.

## Safety Invariants

Corpus text remains data. Direct reference rules may record the matched corpus span in proposal
metadata, but explanations are library authored and do not copy adversarial instruction text.

Every accepted proposal cites at least one projected evidence id. Provider proposals citing unknown
evidence are rejected as malformed provider output.

`proposal_report()` requires either an explicit `pipeline_id` or a graph `pipeline_fingerprint`.
It does not substitute a shared sentinel pipeline identity.

Rejected model proposals are recorded in `rejected_proposals`. They are not silently dropped.

Conflicting proposals can coexist as `requires_review`. Confidence alone is not sufficient for
promotion to trusted evidence.

`proposal_to_graph_edge()` can represent a supersession proposal as an
`inferred_candidate_supersedes` graph edge. That edge remains separate from authored graph edges and
is not consumed by trust evaluation.

Deterministic text rules require graphs built with `include_text=True`. The text is projected under
a reserved metadata key and remains opt in so ordinary graph callers do not duplicate chunk bodies.

## Session 3 Control Artifact

The reproducible artifact generator is:

```powershell
.\.venv\Scripts\python.exe -m recall.eval.reasoning_session3
```

The checked in artifact is `results/reasoning_session3_proposals.json`. It contains the proposal
protocol reference, precision and recall on authored synthetic missing relationships, rejected
proposal examples, provider failure matrix, and side effect audit flags.
