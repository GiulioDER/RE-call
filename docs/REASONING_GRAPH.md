# RE-call Reasoning Graph Projection

Version: 0.1.0

Status: Session 2 graph projection. Inference is not enabled by this document.

## Purpose

The reasoning graph is a derived, immutable projection over one index generation. It is a typed
Python value used by future reasoning APIs to inspect authored relationships and diagnostics
without mutating corpus metadata, active generation state, calibration state, or trust decisions.

## Schema

The public schema lives in `recall.reasoning_graph`.

`ReasoningGraphProjection` contains:

* `schema_version`.
* `graph_id`.
* `tenant_id`.
* `generation_id`.
* Optional `pipeline_fingerprint`.
* Optional `corpus_fingerprint`.
* `nodes`.
* `authored_edges`.
* `inferred_candidate_edges`.
* `diagnostics`.

`ReasoningGraphNode` represents a projected source or chunk. Every node carries tenant identity,
generation identity, source identity, optional chunk identity, file metadata, provenance, validity,
calibration metadata, and original metadata.

`ReasoningGraphEdge` separates authored supersession edges from inferred candidate edges through
the `kind` field. Session 2 builds only `authored_supersedes` edges. `inferred_candidate_edges` is
present as a separate channel so later inference proposals cannot be confused with authored corpus
metadata.

`ReasoningGraphDiagnostic` represents:

* `unresolved_reference`.
* `ambiguous_reference`.
* `cycle`.
* `conflicting_authored_claim`.
* `orphaned_node`.
* `duplicate_entity_candidate`.
* `malformed_metadata`.

## Identity Rules

Every graph identity is deterministic. Node, edge, diagnostic, and graph identifiers are derived
with `recall.lineage.canonical_sha256()` over canonical JSON payloads that include schema version,
tenant identity, generation identity, and the identity fields for that object.

Identical corpus and pipeline inputs produce identical node, authored edge, diagnostic, and graph
identities. Changing tenant or generation changes identities. Inferred candidate edges are not
folded into authored edge identity.

## Construction Rules

`build_reasoning_graph()` is pure and accepts chunks plus explicit tenant and generation identity.

`project_store_graph()` reads through the existing store APIs. If the store supports generation
snapshots, projection pins one snapshot and builds from that immutable view. The function calls
`supersession_all()` to reuse the existing authored supersession resolver and its dated candidate
map.

Graph construction does not write to the corpus, does not promote a generation, does not edit
frontmatter, and does not call trust evaluation.

## Equivalence Report

Authored graph edges are compared against `resolve_supersession_candidates()` and
`supersession_all()`. The control test
`tests/test_reasoning_graph.py::test_authored_edges_reproduce_current_supersession_resolution`
asserts that `ReasoningGraphProjection.authored_supersession_map()` equals the existing resolver's
winner map.

The projection intentionally exposes more diagnostics than retrieval. Ambiguous references,
duplicate basename candidates, cycles, conflicts, orphaned source nodes, and malformed validity
metadata are represented as graph diagnostics. They are not converted into trusted corpus metadata.

## Determinism Results

Verified locally on 2026-08-10 with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_reasoning_graph.py tests\test_supersession_edge_dating.py tests\test_trust.py tests\test_reasoning_session1.py
```

Result: 94 passed in 12.60 seconds.

Also verified:

```powershell
ruff check recall/reasoning_graph.py tests/test_reasoning_graph.py
.\.venv\Scripts\python.exe -m mypy recall\reasoning_graph.py recall\__init__.py
```

Result: ruff passed, mypy passed.

## Safety Notes

Existing trust decisions remain unchanged when the graph is absent because no retrieval or trust
entry point consumes the graph. A graph can report stale, conflicting, or ambiguous structure, but
it cannot promote a stale hit into `ok`.

Known limit: source and chunk graph projection uses chunk metadata and store supplied supersession
candidate dates. If a caller bypasses the store and calls `build_reasoning_graph()` with only bare
`Chunk` values, authored edge dates are unknown unless passed through `authored_edge_candidates`.

## Session 3 Proposal Edges

Session 3 fills the previously reserved candidate channel through `recall.reasoning_proposals`.
`proposal_to_graph_edge()` converts only `supersedes` proposals into
`inferred_candidate_supersedes` edges. Candidate edges keep proposal provenance, provider identity,
confidence, uncertainty, and explanation in edge metadata. They do not enter
`authored_supersession_map()` and are not consumed by trust evaluation.

Chunk projection carries chunk text only when a caller explicitly passes `include_text=True`.
The body text is stored under a reserved metadata key so corpus metadata cannot shadow the evidence
body. The default projection remains lightweight for graph callers that do not run proposal rules.
Node and graph identities remain derived from the same stable identity fields, not from proposal
output.
