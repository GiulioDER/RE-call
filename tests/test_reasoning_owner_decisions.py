"""Tests for the owner's 2026-11-06 decisions on the reasoning hunt residue (PR 1235).

CODE-008 (one diagnostic helper, ids byte-identical), BUG-006 (a claim counts once in graph
corroboration), DEPLOY-002 and DEPLOY-101 (the graph marker records its schema version; a graph
built by an older builder keeps serving, and validate rebuilds it). CODE-106 is a type change
that mypy checks, so it has no test here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest

import recall.semantic_graph._build as graph_build
import recall.semantic_graph._model as graph_model
from recall.generations import GenerationState, _semantic_graph_marker
from recall.semantic_graph import (
    SEMANTIC_GRAPH_SCHEMA_VERSION,
    SemanticRelation,
    build_semantic_graph,
    load_semantic_graph,
    read_graph_readiness,
)
from recall.types import Chunk
from recall_mcp import graph_expansion
from tests.conftest import TEST_DSN, requires_db
from tests.test_generations import _Embedder, _manifest, _pipeline, _reader, manager  # noqa: F401


def _use_schema_version(monkeypatch: pytest.MonkeyPatch, version: int) -> None:
    """Build as a builder of `version` would: every module that stamps ids reads this name."""
    monkeypatch.setattr(graph_model, "SEMANTIC_GRAPH_SCHEMA_VERSION", version)
    monkeypatch.setattr(graph_build, "SEMANTIC_GRAPH_SCHEMA_VERSION", version)


def _diagnostic_corpus() -> list[Chunk]:
    """Raises a diagnostic from every construction site of the graph builder."""

    def chunk(cid: str, file: str, text: str, **meta: Any) -> Chunk:
        return Chunk(cid, f"/r/{file}", text, {"file": file, **meta})

    def rel(**relation: Any) -> dict[str, Any]:
        return {"recall_graph": {"relations": [relation]}}

    return [
        chunk("a1", "notes/a.md", "alpha", person="Ada", project="Ada", entity_aliases={"Ada": ["AL"]}),
        chunk("a2", "notes/b.md", "beta", person="Bob", entity_aliases={"Bob": ["AL"]}),
        chunk("r1", "rel/one.md", "one", **rel(relation="bogus", subject="X", object="Y")),
        chunk("r2", "rel/two.md", "two", project="P", service="S",
              **rel(relation="supports", subject="P", object="S", confidence=7)),
        chunk("r3", "rel/three.md", "three", **rel(relation="supports", subject="Nope", object="Gone")),
        chunk("r4", "rel/four.md", "four", project="P4", service="S4",
              **rel(relation="supports", subject="P4", object="S4",
                    valid_from="2026-01-02", valid_until="2026-01-01")),
        chunk("r5", "rel/five.md", "five", project="P5", service="S5",
              **rel(relation="supports", subject="P5", object="S5", uncertainty="oops")),
        chunk("s1", "dup/x.md", "x one"),
        chunk("s2", "other/x.md", "x two"),
        chunk("s3", "sup/new.md", "new", supersedes="x.md"),
        chunk("s4", "sup/newer.md", "newer", supersedes="missing.md"),
        chunk("c1", "people/carl.md", "carl", person="Carl", entity_aliases={"Carl": ["CC"]}),
        chunk("c2", "people/cora.md", "cora", person="Cora", entity_aliases={"Cora": ["CC"]}),
        chunk("s5", "sup/dated.md", "dated", supersedes="sup/new.md", valid_from="not-a-date"),
        chunk("d1", "dep/a.md", "dep a", recall_graph={"depends_on": ["x.md", "absent.md"]}),
        chunk("d2", "dep/b.md", "dep b", recall_graph={"depends_on": "x.md"}),
        chunk("l1", "link/a.md", "see [[x]] and [[nowhere]]"),
    ]


#: Captured from the builder before the helper existed, at graph schema version 2.
GOLDEN_GRAPH_ID = "sg_graph_f3055ae7219e96df4e00bf4d"
GOLDEN_DIAGNOSTICS = [
    ("sg_diagnostic_1f6064305c4c38a0a42bde9c", "invalid_relation"),
    ("sg_diagnostic_240d391368aeb419acdaeadb", "invalid_relation"),
    ("sg_diagnostic_2e4ecfca0f509717cb99a28a", "invalid_relation"),
    ("sg_diagnostic_4e8d227bfdcebf9910dfb708", "ambiguous_entity"),
    ("sg_diagnostic_67e9e1b3890c0e9206574f7c", "ambiguous_entity"),
    ("sg_diagnostic_78933ef9485fafe9c563d729", "invalid_relation"),
    ("sg_diagnostic_914fa59aa080b6426bfc6f35", "ambiguous_entity"),
    ("sg_diagnostic_978c2747a510648d2b5011e1", "missing_evidence"),
    ("sg_diagnostic_9834c0822d16d36c71bc8e1c", "ambiguous_entity"),
    ("sg_diagnostic_be65106265614065edad7c4d", "ambiguous_entity"),
    ("sg_diagnostic_cf07db5c624f3ae221c2e27d", "invalid_relation"),
    ("sg_diagnostic_d23cd97586f9ecfe271b1f4e", "ambiguous_entity"),
    ("sg_diagnostic_ec74801ed272217864433ec6", "missing_evidence"),
    ("sg_diagnostic_ecf7d87482c5c6540db14d90", "invalid_relation"),
]


def test_code008_the_diagnostic_helper_keeps_every_id(monkeypatch) -> None:
    """CODE-008: building diagnostics through `_diagnostic` left every graph and diagnostic id.

    A pin for a refactor: the ids were captured from the pre-refactor builder at schema version 2,
    so this passes there too. Red proof is a mutation of the helper: dropping its `**identity`
    extras changed the ids and failed the diagnostic assertion.
    """
    _use_schema_version(monkeypatch, 2)
    graph = build_semantic_graph(_diagnostic_corpus(), tenant_id="t", generation_id="g")
    assert sorted((item.id, item.kind) for item in graph.diagnostics) == GOLDEN_DIAGNOSTICS
    assert graph.graph_id == GOLDEN_GRAPH_ID


def _relation(rid: str, chunk_id: str, source: str, *, obj: str = "entity-b") -> SemanticRelation:
    return SemanticRelation(
        id=rid,
        tenant_id="t",
        generation_id="g",
        subject_id="entity-a",
        object_id=obj,
        relation="supports",
        evidence_chunk_ids=(chunk_id,),
        extraction_method="explicit_relation",
        confidence=1.0,
        metadata={"source": source},
    )


def test_bug006_one_claim_repeated_across_chunks_counts_once() -> None:
    """BUG-006: one frontmatter claim, copied to three chunks of a memo, counted as three supports.

    `graph_expansion.relation_claim` names a claim by relation, endpoints and declaring source.
    The pre-fix tree has no `relation_claim`, so the red proof is a mutation: returning
    `relation.id` from it counted the three copies apart and failed the first assertion.
    """
    copies = [_relation(f"rel-{i}", f"memo-{i}", "/r/memo.md") for i in range(3)]
    other_document = _relation("rel-x", "other-0", "/r/other.md")
    other_target = _relation("rel-y", "memo-0", "/r/memo.md", obj="entity-c")
    assert len({graph_expansion.relation_claim(item) for item in copies}) == 1
    assert len({graph_expansion.relation_claim(item) for item in (*copies, other_document)}) == 2
    assert len({graph_expansion.relation_claim(item) for item in (*copies, other_target)}) == 2
    candidate = graph_expansion.GraphCandidate(
        claim_keys={graph_expansion.relation_claim(item) for item in copies}
    )
    two_documents = graph_expansion.GraphCandidate(
        claim_keys={graph_expansion.relation_claim(item) for item in (copies[0], other_document)}
    )
    assert graph_expansion.graph_corroboration(candidate) < graph_expansion.graph_corroboration(
        two_documents
    )


def _rows(graph: Any, marker: dict[str, Any]) -> tuple[Any, ...]:
    """The single row `load_semantic_graph` reads, as PostgreSQL returns it (JSON values)."""

    def jsonable(value: Any) -> Any:
        return json.loads(json.dumps(graph_model._thaw(value), default=str))

    return (
        [graph.pipeline_fingerprint, graph.corpus_fingerprint, {"semantic_graph": marker}],
        [
            [e.id, e.canonical_name, e.normalized_name, e.kind, list(e.aliases),
             e.extraction_method, e.confidence, jsonable(e.metadata)]
            for e in graph.entities
        ],
        [
            [m.id, m.entity_id, m.chunk_id, m.mention_text, m.extraction_method, m.confidence,
             jsonable(m.metadata)]
            for m in graph.mentions
        ],
        [
            [r.id, r.subject_id, r.object_id, r.relation, r.extraction_method, r.confidence,
             r.status, list(r.uncertainty), r.pipeline_fingerprint, r.corpus_fingerprint,
             jsonable(r.metadata), list(r.evidence_chunk_ids)]
            for r in graph.relations
        ],
    )


class _Conn:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self._row = row

    def execute(self, *_args: Any) -> Any:
        return SimpleNamespace(fetchone=lambda: self._row)


def _older_graph(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    older = SEMANTIC_GRAPH_SCHEMA_VERSION - 1
    _use_schema_version(monkeypatch, older)
    graph = build_semantic_graph(
        [
            Chunk("a", "/r/a.md", "alpha", {"file": "a.md", "project": "P", "service": "S",
                  "recall_graph": {"relations": [
                      {"relation": "supports", "subject": "P", "object": "S"}]}}),
            Chunk("b", "/r/b.md", "see [[a]]", {"file": "b.md"}),
        ],
        tenant_id="t",
        generation_id="g",
    )
    marker = _semantic_graph_marker(graph)
    # A marker written by that builder names its version, whether or not the code that wrote
    # it knew the field: set it explicitly so the test reads the same on either tree.
    marker["schema_version"] = older
    monkeypatch.undo()
    return graph, marker


def test_deploy002_an_older_graph_still_matches_its_marker_after_a_bump(monkeypatch) -> None:
    """DEPLOY-002: after a builder version bump, every persisted graph missed its own marker.

    `load_semantic_graph` stamped a loaded graph with the current version, and the version feeds
    the fingerprint the serving cache compares with the marker, so graph expansion turned off.
    Red on the pre-fix tree: the loaded fingerprint differed from the marker's.
    """
    graph, marker = _older_graph(monkeypatch)
    loaded = load_semantic_graph(_Conn(_rows(graph, marker)), "t", "g")
    assert loaded is not None
    assert loaded.fingerprint == marker["graph_fingerprint"]
    assert loaded.schema_version == marker["schema_version"]


def test_deploy002_readiness_names_an_older_builder() -> None:
    """DEPLOY-002: readiness could not say a graph was built by an older builder.

    The pre-fix `GraphReadiness` has no `schema_version`, so the red proof is a mutation: not
    passing `schema_version=` from `read_graph_readiness` left it None and failed the assertion.
    """
    marker = {
        "graph_id": "sg_graph_x",
        "graph_fingerprint": "f" * 64,
        "entity_count": 1,
        "mention_count": 1,
        "relation_count": 0,
        "diagnostic_count": 0,
        "ready": True,
    }
    legacy = read_graph_readiness(_Conn(({"semantic_graph": marker},)), "t", "g")
    assert legacy.ready
    assert legacy.schema_version == graph_model.LEGACY_GRAPH_SCHEMA_VERSION
    assert legacy.stale_schema
    current = read_graph_readiness(
        _Conn(({"semantic_graph": {**marker, "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION}},)),
        "t",
        "g",
    )
    assert not current.stale_schema


def test_bug001_a_graph_from_a_newer_builder_is_not_ready() -> None:
    """BUG-001: after a rollback, a graph a newer builder wrote was served silently.

    Readiness accepted any recorded version, and the loader stamped it, so the fingerprint
    matched. Red proof is a mutation: dropping the newer-version clause in
    `read_graph_readiness` left the graph ready and failed `assert not newer.ready`.
    """
    marker = {
        "graph_id": "sg_graph_x",
        "graph_fingerprint": "f" * 64,
        "entity_count": 1,
        "mention_count": 1,
        "relation_count": 0,
        "diagnostic_count": 0,
        "ready": True,
        "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION + 1,
    }
    newer = read_graph_readiness(_Conn(({"semantic_graph": marker},)), "t", "g")
    assert not newer.ready
    assert newer.reason == "GRAPH_NOT_READY"


def test_bug001_a_newer_graph_misses_its_marker_on_load(monkeypatch) -> None:
    """BUG-001: the loader stamped a newer builder's version, so its fingerprint matched.

    Red proof is a mutation: stamping `marker_schema_version(marker)` without the `min` with
    this code's version made the loaded fingerprint equal the marker's.
    """
    _use_schema_version(monkeypatch, SEMANTIC_GRAPH_SCHEMA_VERSION + 1)
    graph = build_semantic_graph(
        [Chunk("a", "/r/a.md", "alpha", {"file": "a.md", "project": "P"})],
        tenant_id="t",
        generation_id="g",
    )
    marker = _semantic_graph_marker(graph)
    marker["schema_version"] = SEMANTIC_GRAPH_SCHEMA_VERSION + 1
    monkeypatch.undo()
    loaded = load_semantic_graph(_Conn(_rows(graph, marker)), "t", "g")
    assert loaded is not None
    assert loaded.fingerprint != marker["graph_fingerprint"]


def test_code001_load_diagnostics_of_an_older_graph_use_its_version(monkeypatch) -> None:
    """CODE-001 and CODE-004: load diagnostics of an older graph were keyed to the current version.

    Both store sites are pinned against the payload they hashed before the helper existed,
    at the graph's own version. Red proof is a mutation: dropping `schema_version=` from the
    `missing_evidence` site in `_load_relation_rows` changed that id and failed the assertion;
    dropping it from the `invalid_relation` site failed the second.
    """
    graph, marker = _older_graph(monkeypatch)
    rows = _rows(graph, marker)
    relations = rows[3]
    assert len(relations) >= 2
    relations[0][11] = []
    relations[1][10] = {**relations[1][10], "valid_from": "not a date"}
    loaded = load_semantic_graph(_Conn(rows), "t", "g")
    assert loaded is not None
    older = marker["schema_version"]

    def expected(kind: str, reference: str, **identity: Any) -> str:
        return graph_model._identity(
            "diagnostic",
            {"schema_version": older, "tenant_id": "t", "generation_id": "g", "kind": kind,
             "reference": reference, **identity},
        )

    ids = {diagnostic.kind: diagnostic.id for diagnostic in loaded.diagnostics
           if diagnostic.kind in {"missing_evidence", "invalid_relation"}}
    assert ids["missing_evidence"] == expected("missing_evidence", relations[0][0])
    assert ids["invalid_relation"] == expected(
        "invalid_relation", relations[1][0], value=repr(relations[1][10])
    )


def test_deploy101_the_marker_records_the_builders_version() -> None:
    """DEPLOY-101: the marker did not say which builder wrote the graph.

    Red on the pre-fix tree: `_semantic_graph_marker` wrote no `schema_version`.
    """
    graph = build_semantic_graph([Chunk("a", "/r/a.md", "alpha", {"file": "a.md"})], tenant_id="t", generation_id="g")
    assert _semantic_graph_marker(graph).get("schema_version") == SEMANTIC_GRAPH_SCHEMA_VERSION


@requires_db
def test_deploy101_validate_upgrades_a_graph_an_older_builder_wrote(manager, monkeypatch) -> None:  # noqa: F811
    """DEPLOY-101: validating a generation built before an upgrade failed it for good.

    `_validate` rebuilt the graph with the current builder and refused the older marker as a
    fingerprint mismatch, and `rebuild_graph` refuses a validating generation. Expected red on
    the pre-fix tree: the generation ends FAILED. This runs only where a database is available
    (CI during the AML window), so its red run is not recorded here.
    """
    data = b"a memo built before the upgrade"
    manifest = _manifest(manager.tenant_id, data)
    generation = manager.create(manifest, _pipeline("model-a"))
    _use_schema_version(monkeypatch, SEMANTIC_GRAPH_SCHEMA_VERSION - 1)
    manager.build(generation.generation_id, _reader(manifest, data), _Embedder(1), lambda t: [t])
    monkeypatch.undo()

    result = manager.validate(generation.generation_id)

    assert result.state == GenerationState.READY
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        readiness = read_graph_readiness(conn, manager.tenant_id, generation.generation_id)
    assert readiness.ready
    assert readiness.schema_version == SEMANTIC_GRAPH_SCHEMA_VERSION
