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
* `proposed_relation`: `supersedes`, `contradicts`, `same_entity`, or `references`.
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
pipeline identity, relation endpoints, evidence ids, rule id, and status.

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

### How proposals are scored

`proposal_precision_recall` scores one `proposed_relation` at a time, so the four claim kinds are
never averaged into a single number. Two rules keep the score honest.

**Referrals are not assertions.** Only `candidate` proposals are scored. A `requires_review`
proposal is reported separately as `referred_proposals`, alongside `asserted_proposals` and a
`referral_rate`, and is never folded into precision. Counting referrals as predictions would let a
provider buy precision by relabelling every shaky proposal `requires_review`. The policy is the
`counted_statuses` and `referred_statuses` parameters rather than a hard coded constant; the two
must stay disjoint, and an unknown status raises rather than silently scoring zero.

Mind the two denominators. Precision divides by the deduplicated `(subject, object)` pair set,
while `asserted_proposals` and `referred_proposals` count raw proposals, because several rules can
fire on the same pair. Session 3 publishes three asserted proposals over two pairs, so
`true_positive / asserted_proposals` is 0.667 and is **not** the published precision of 1.0. The
key names carry the unit for exactly this reason.

**A rate with no data is not a score.** Precision is `NaN` when nothing was asserted and recall is
`NaN` when nothing was expected, matching `recall.eval.metrics.fraction_true`. This matters for the
rule based baseline, which proposes zero edges: precision `0.0` would read as "the rules were
wrong" when the truth is "the rules declined to answer", and recall `1.0` against an empty
expectation set is a perfect score derived from nothing.
