# RE-call Reasoning Graph Projection

Version: 0.1.0

Status: Evidence Graph V1. Semantic expansion is opt in and inference is not enabled by this
document.

## Two graphs, and which one `one_hop` walks

**This document describes two separate structures with two separate relation vocabularies, and
`graph_expansion=one_hop` traverses only the second.** Read this before carrying anything from
"Purpose" or "Schema" below over to expansion; that half of the document is about the other graph.

| | Reasoning graph projection | Semantic graph (Evidence Graph V1) |
|---|---|---|
| Built by | `build_reasoning_graph()` / `project_store_graph()` | `build_semantic_graph()` |
| Edges | `authored_supersedes`, `inferred_candidate_supersedes` | `SemanticRelation` over `RelationKind` |
| Vocabulary | supersession only | `supports`, `contradicts`, `references`, `depends_on`, `caused`, `same_entity` |
| Read by | `recall_reasoning_projection`, `recall_current_state` | `one_hop` expansion |
| Traversed by `one_hop` | **no** | yes |

Three consequences that are invisible from either half alone.

**1. `supersedes` is not a semantic relation kind, so an authored supersession edge has no
representation in the semantic graph at all.** Not "not yet connected" — not expressible.
`RelationKind` in `recall/semantic_graph.py` does not contain it, and the production precision
policy narrows further to `GRAPH_DIRECTIONAL_RELATIONS`, four of the six kinds.

That is not a gap. Supersession is enforced **upstream**, by the trust layer, and enforcing it
again in expansion would be the redundancy: `recall.trust.evaluate` gives a superseded memory the verdict
`superseded`, `is_trusted` admits only `ok`, and expansion seeds exclusively from
`is_trusted(hit)`. So a superseded document cannot seed a traversal, and every chunk expansion
admits is sent back through the same trust layer before it becomes evidence. Supersession bounds
the graph stage on both sides rather than being walked inside it.

**2. A corpus can report a healthy `authored_edge_count` and still see
`graph_relations_inspected: 0`, with nothing wrong.** Those two numbers count edges in the two
different structures in the table above. This has already produced a wrong inference in the field
— eleven authored edges, zero relations inspected, read as a mis-tuned gate. The gate was fine;
those edges were never candidates for traversal.

**3. Only `references` is produced automatically, and in practice it is the only kind with any
rows.** There are exactly two extraction paths, and they are not equally reachable:

- **Links and wikilinks** are extracted automatically, always as `references`. Every corpus that
  is written in Markdown gets these for free.
- **The `recall_graph` frontmatter object** can declare any of the six kinds, but only where a
  human hand-wrote that one-line JSON. Nothing infers `supports`, `contradicts`, `depends_on`,
  `caused` or `same_entity` from prose; V1 deliberately has no model or embedding extractor.

Measured 2026-09-01 against the live serving database, **every tenant, every relation row**:

```text
 tenant_id    |  relation  |  status  | count
--------------+------------+----------+-------
 memory       | references | authored | 56536
 re-call-docs | references | authored |   395
```

Zero rows of the other five kinds anywhere. So unless a corpus authors `recall_graph` relations
deliberately, **`one_hop` is a single-relation traversal over a reference graph**, and that is the
fact that decides whether reaching for it is worth anything. Re-measure before relying on either
direction:

```sql
SELECT tenant_id, relation, status, count(*)
FROM recall_graph_relations_v1 GROUP BY 1, 2, 3 ORDER BY 1, 4 DESC;
```

## Evidence Graph V1

RE-call also projects a deterministic semantic graph into PostgreSQL migration `0016`. It is a
derived evidence structure, not a replacement for authored corpus truth. The projection contains
immutable `SemanticEntity`, `SemanticMention`, `SemanticRelation`, `SemanticGraphDiagnostic`, and
`SemanticGraphProjection` values bound to one tenant and generation.

V1 recognizes the entity kinds `person`, `project`, `service`, `file`, `decision`, `event`,
`concept`, and `unknown`. Supported authored relations are `supports`, `contradicts`, `references`,
`depends_on`, `caused`, and `same_entity` — *supported* meaning the vocabulary a `recall_graph`
declaration may name, not the vocabulary a corpus is likely to hold. Only `references` is produced
by any automatic extractor, and it is the only kind with rows on any live tenant; see **Two
graphs, and which one `one_hop` walks** above. `supersedes` is deliberately absent from this
list. Every relation has supporting chunk identifiers,
extraction method, confidence, uncertainty, tenant and generation identity, pipeline and corpus
fingerprints, and authored or candidate status.

Extraction is deterministic. It reads filenames, headings, frontmatter, explicit metadata, and
explicit relation declarations. Exact Unicode normalized aliases are resolved; fuzzy merging and
embeddings are deliberately absent. Ambiguous candidates remain separate and produce diagnostics.
Model extraction remains an ingest proposal path and cannot drive V1 graph expansion.

### Authoring deterministic relations

Markdown frontmatter can carry one namespaced, one line JSON object under `recall_graph`. The
object may contain `entities`, `aliases`, and `relations`:

```text
---
recall_graph: {"entities":[{"name":"Rate Limits V2","kind":"decision"},{"name":"API Gateway","kind":"service"}],"relations":[{"relation":"supports","subject":"Rate Limits V2","object":"API Gateway","confidence":1.0}]}
---
The decision supports the API Gateway policy.
```

The supported entity kinds and relation names are the V1 vocabularies above. Relation endpoints
must be declared in the same supporting chunk, and every accepted relation keeps that chunk as its
evidence. Invalid or incomplete declarations become graph diagnostics and never become trusted
relations. `aliases` uses the canonical name to list of exact aliases form:
`{"Rate Limits V2":["rate limits","RL v2"]}`.

Markdown links and wikilinks are also extracted conservatively as authored `references` relations.
They are accepted only when the target resolves to exactly one file in the generation. Missing and
ambiguous targets do not drive expansion. This provides useful structure without treating a
similar sentence as an authored semantic claim.

The four persistence tables are `recall_graph_entities_v1`, `recall_graph_mentions_v1`,
`recall_graph_relations_v1`, and `recall_graph_relation_evidence_v1`. They are tenant scoped,
generation scoped, protected by RLS, and linked to chunks and generations with cascading foreign
keys. Source erasure removes unsupported derived rows and invalidates the generation graph marker.

New generations build and verify the semantic graph before promotion. Existing generations remain
valid for ordinary retrieval, but graph expansion returns `GRAPH_NOT_READY` until an operator runs:

```text
recall graph rebuild --generation <generation_id>
```

Graph expansion is disabled by default. `one_hop` starts only from trusted retrieval, follows
authored semantic relations, re-evaluates every candidate through the ordinary trust layer, and
appends only trusted evidence. It cannot promote a demoted hit, bypass calibration, use model
proposals, or change ordinary `recall_search` and `recall_evidence` behavior.

### Precision admission policy

The production `one_hop` path uses the combined precision policy. Positive traversal is directional
for `supports`, `references`, `depends_on`, and `caused`. `contradicts` is retained as a diagnostic
and `same_entity` is identity resolution only. Relation evidence must intersect the trusted seed
chunks, and reverse traversal is refused. Candidate ranking uses the calibrated query cosine
first, followed by distinct trusted seed corroboration, distinct supporting relations, relation
confidence, and chunk id. Relation confidence never replaces the calibrated retrieval score.

An entity mentioned by more than 32 distinct chunks is a hub and cannot seed traversal unless the
normalized query contains an exact entity alias. A candidate must have a query cosine and be no
more than 0.10 below the strongest trusted seed cosine. Selective expansion refuses to traverse
when at least two trusted initial items exist without a retrieval gap. In every case, admitted
chunks are sent through the ordinary trust layer again.

The internal evaluation harness can isolate each policy component and run shuffled or removed
relation controls with `RECALL_GRAPH_PRECISION_VARIANT` and
`RECALL_GRAPH_RELATION_CONTROL`. The active policy fingerprint and sanitized rejection counters
are included in reasoning diagnostics. These controls are not public graph modes.

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
