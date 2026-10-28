"""View 2 and the database-backed corpus graph: `recall.dashboard.corpus` and its pages.

Invariants and the failure each one catches:
- G1 a supersession is drawn from the newer file to the one it replaces, and the replaced file is
  `superseded`; a file past its `valid_until` is `expired`.
- G2 a shared name links files only when it is uncommon: an entity in more than
  `SHARE_MAX_FILES` files draws nothing, and each file keeps at most `SHARE_PER_FILE` partners,
  the strongest first.
- G3 a Python import resolves to a file of the corpus: by full module path, by a dotted tail
  exactly one file has, `from pkg import mod` to the module file, and a relative import against
  the importer's package; a tail two files share resolves to neither.

Red proof, 2026-10-08, each mutation alone, failing in its intended assertion (JUnit XML), then
restored byte for byte and green (D1 on the test host, the rest locally):
- Q1 (G1) supersession added old to new: "a supersession was drawn the wrong way".
- Q2 (G1) validity ignored: the state map lost `expired`.
- Q3 (G2) the `SHARE_MAX_FILES` bound removed: "an entity named in too many files linked them".
- Q4 (G2) an edge kept when EITHER end ranks it: "a file kept too many, or not its strongest,
  partners" (6 against a cap of 4). That was the first implementation, caught by this test.
- Q5 (G3) a relative import resolved as its bare tail: "a relative import did not resolve". It
  SURVIVED a first version of the test, where the tail `util` named one file only and the tail
  rule rescued it; a second `other/util.py` now makes the tail ambiguous.
- Q6 (G3) an ambiguous tail accepted: "an ambiguous module tail resolved to one file".
- Q7 (P1) the cache bypassed: "the corpus graph was rebuilt inside its cache window".
- Q8 (P2) chunk text rendered without `_e`: "chunk text reached the page unescaped".
- Q9 (P3) the switch listing every tenant: "a benchmark tenant cluttered the corpus switch".
- Q10 (D1) `references` dropped from the relation filter: "the authored wiki link is missing from
  the graph".
- Q11 (D1) the import scan matching no Python file: "the Python import is missing from the graph".
- Q12 (D1) View 2 never naming the successor: the `superseded_by` assertion failed.
D1 ran on the test host against a private PostgreSQL 17 with pgvector (7 passed unmutated). Its
role there is the database owner, so row-level security is not exercised by D1; the queries set
`recall.tenant_id` regardless, which is what a restricted role needs.
- P1 `/api/graph.json?corpus=` builds the database graph for that tenant, once per cache window;
  a value that is not a tenant name falls back to the memo files.
- P2 View 2 escapes everything the database holds (chunk text, names, headings) and reads the
  current tenant; an unknown file is a 404, a database that is down the standard 503 page.
- P3 the corpus switch offers the main corpora only, plus the one being shown.
- G4 a module named like the standard library never resolves by its tail to a corpus file, and
  `from pkg import (` ending its line still links to `pkg` (CCA on #904, DAT-001 and DAT-002).
- G5 a path several roots hold is one node per copy, and a mention joins the copy that holds it
  (CODE-001); a file superseded by a forgotten memo is noted, and a file not yet valid says so.
- P4 on a lite store the corpus pages answer that they need PostgreSQL, never "did not answer".
- P5 concurrent misses on one corpus build its graph once; the cache is per tenant and expires.
- P6 View 2 names a forgotten successor without linking it, and says an ambiguous claim matches
  several files.
- R1 every corpus query runs after the connection's `recall.tenant_id` is set.
Red proofs for G4 to R1 are in each test's docstring.
- D1 against a real generation built by the generation pipeline: the graph holds the declared
  supersession, the authored wiki link and the Python import, and View 2's data names what
  replaces a file, what it imports and what imports it.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import psycopg
import pytest

from recall.dashboard import corpus
from recall.dashboard import db as dbq
from recall.dashboard.db import DashboardDB
from recall.dashboard import server as dashboard_server
from recall.dashboard.server import DashboardApp
from recall.supersession import SupersessionGraph
from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason, requires_db

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
TODAY = "2026-10-08"


def _file(name: str, *, valid_until: str | None = None) -> tuple[Any, ...]:
    return (name, 1, None, None, None, valid_until, [name.upper()])


def _graph(files: list[tuple[Any, ...]], **parts: Any) -> dict[str, Any]:
    return corpus.assemble_graph(
        files, parts.get("winner", {}), parts.get("relations", []), parts.get("mentions", []), parts.get("imports", []),
        today=TODAY, tenant="t", generation="g",
    )


def _edges(graph: dict[str, Any], kind: str) -> set[tuple[str, str]]:
    return {(e["source"], e["target"]) for e in graph["edges"] if e["kind"] == kind}


# ------------------------------------------------------------------ pure rules


def test_supersession_runs_newer_to_older_and_validity_expires() -> None:
    """G1."""
    graph = _graph([_file("old.md"), _file("new.md"), _file("gone.md", valid_until="2026-01-01")], winner={"old.md": "new.md"})
    assert _edges(graph, "supersedes") == {("new.md", "old.md")}, "a supersession was drawn the wrong way"
    states = {n["id"]: n["state"] for n in graph["nodes"]}
    assert states == {"old.md": "superseded", "new.md": "current", "gone.md": "expired"}


def test_shared_names_link_only_uncommon_entities_and_cap_partners() -> None:
    """G2."""
    many = [f"f{i:02d}.py" for i in range(corpus.SHARE_MAX_FILES + 1)]
    hub = ["hub.py", *[f"p{i}.py" for i in range(corpus.SHARE_PER_FILE + 2)]]
    mentions = [("common", name) for name in many]
    # hub.py shares one entity with each partner; p0 shares two, so it is the strongest.
    mentions += [(f"pair-{name}", "hub.py") for name in hub[1:]] + [(f"pair-{name}", name) for name in hub[1:]]
    mentions += [("extra", "hub.py"), ("extra", "p0.py")]
    graph = _graph([_file(n) for n in [*many, *hub]], mentions=mentions)
    shares = _edges(graph, "shares")
    assert not any(a in many and b in many for a, b in shares), "an entity named in too many files linked them"
    partners = {b for a, b in shares if a == "hub.py"} | {a for a, b in shares if b == "hub.py"}
    assert len(partners) <= corpus.SHARE_PER_FILE and "p0.py" in partners, "a file kept too many, or not its strongest, partners"


def test_imports_resolve_to_corpus_files() -> None:
    """G3."""
    files = {
        "src/pkg/__init__.py", "src/pkg/util.py", "src/pkg/main.py", "src/pkg/sub/deep.py",
        "a/common.py", "b/common.py",
        # A second util.py: the bare tail `util` is then ambiguous, so `..util` can resolve only
        # by walking up the importer's package, not by the tail rule.
        "other/util.py",
    }
    rows = [
        ("src/pkg/main.py", None, None, "pkg.util"),          # tail of one file's module
        ("src/pkg/main.py", "pkg", "util", None),             # from pkg import util -> util.py
        ("src/pkg/sub/deep.py", "..util", "f", None),         # relative, two levels
        ("src/pkg/main.py", ".sub.deep", "g", None),          # relative, one level
        ("src/pkg/main.py", None, None, "common"),            # two files share the tail: none
        ("src/pkg/main.py", None, None, "os"),                # not in the corpus: none
    ]
    pairs = set(corpus.resolve_imports(rows, files))
    assert ("src/pkg/main.py", "src/pkg/util.py") in pairs
    assert ("src/pkg/sub/deep.py", "src/pkg/util.py") in pairs, "a relative import did not resolve"
    assert ("src/pkg/main.py", "src/pkg/sub/deep.py") in pairs
    assert not any(target.endswith("common.py") for _, target in pairs), "an ambiguous module tail resolved to one file"


# ------------------------------------------------------------------ pages, with the queries faked


class _Fake:
    def __init__(self) -> None:
        self.graph_calls: list[str] = []

    def graph(self, db: DashboardDB, tenant: str) -> dict[str, Any]:
        self.graph_calls.append(tenant)
        return {"nodes": [], "edges": [], "counts": {"memos": 0}, "notes": [], "origin": "database", "tenant": tenant, "generation": "g"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("# A\n\nnote\n", encoding="utf-8")
    return tmp_path


def test_the_graph_api_reads_the_corpus_once_per_window(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1."""
    fake = _Fake()
    monkeypatch.setattr(corpus, "corpus_graph", fake.graph)
    app = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"))
    first = app.handle("GET", "/api/graph.json?corpus=re-call-code-gen", SIGNED)
    second = app.handle("GET", "/api/graph.json?corpus=re-call-code-gen", SIGNED)
    assert first.status == second.status == 200
    assert fake.graph_calls == ["re-call-code-gen"], "the corpus graph was rebuilt inside its cache window"
    memo = app.handle("GET", "/api/graph.json?corpus=%3Cb%3E", SIGNED)
    assert memo.status == 200 and b'"origin"' not in memo.body and fake.graph_calls == ["re-call-code-gen"]


def _detail(**over: Any) -> dict[str, Any]:
    base = {
        "tenant": "re-call-docs", "generation": "gen_x", "file": "docs/<b>x</b>.md", "source_uri": "s3://b/docs/x.md",
        "project": "p", "indexed_commit": "abc", "type": None, "valid_from": None, "valid_until": None, "state": "superseded",
        "superseded_by": "docs/y.md", "supersedes": [], "declared_supersedes": [], "unresolved_claims": [],
        "first_indexed": "2026-10-01", "last_indexed": "2026-10-02", "chunk_count": 1,
        "chunks": [{"chunk_id": "c1", "ordinal": 0, "text": "<script>alert(1)</script>", "headings": ["<img src=x>"]}],
        "entities": [{"name": "<i>n</i>", "kind": "concept", "mentions": 1}], "relations": [], "imports": [], "imported_by": [],
    }
    base.update(over)
    return base


def test_view_two_escapes_and_reads_the_current_tenant(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P2."""
    asked: list[tuple[str, str]] = []

    def detail(db: DashboardDB, tenant: str, file: str) -> dict[str, Any] | None:
        asked.append((tenant, file))
        return _detail() if file == "docs/x.md" else None

    monkeypatch.setattr(corpus, "source_detail", detail)
    app = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"))
    page = app.handle("GET", "/corpus/source?tenant=re-call-docs&file=docs/x.md", SIGNED)
    body = page.body.decode()
    assert page.status == 200 and asked == [("re-call-docs", "docs/x.md")]
    assert "<script>alert(1)</script>" not in body and "&lt;script&gt;alert(1)&lt;/script&gt;" in body, "chunk text reached the page unescaped"
    assert "<img src=x>" not in body and "<i>n</i>" not in body and "<b>x</b>" not in body
    assert "superseded" in body and "docs/y.md" in body
    assert app.handle("GET", "/corpus/source?tenant=re-call-docs&file=nope.md", SIGNED).status == 404

    def down(db: DashboardDB, tenant: str, file: str) -> None:
        raise dbq.DatabaseUnavailable("connection refused")

    monkeypatch.setattr(corpus, "source_detail", down)
    assert app.handle("GET", "/corpus/source?file=docs/x.md", SIGNED).status == 503


def test_the_corpus_switch_lists_the_main_corpora(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P3."""
    tenants = [{"tenant": name, "generation": "g"} for name in ("memory", "re-call-code-gen", "re-call-docs", "amb-bench-1", "amb-bench-2")]
    monkeypatch.setattr(dbq, "tenants", lambda db: tenants)
    app = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"))
    body = app.handle("GET", "/graph?corpus=amb-bench-2", SIGNED).body.decode()
    switch = body.split("class='corpus-switch'", 1)[1].split("</div>", 1)[0]
    assert "corpus=re-call-code-gen" in switch and "corpus=amb-bench-2" in switch
    assert "corpus=amb-bench-1" not in switch, "a benchmark tenant cluttered the corpus switch"
    assert "data-corpus='amb-bench-2'" in body


# ------------------------------------------------------------------ against a real generation


class _S3:
    def __init__(self, objects: dict[tuple[str, str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        data = self.objects[(kwargs["Bucket"], kwargs["Key"], kwargs["VersionId"])]
        return {"Body": BytesIO(data), "ContentLength": len(data), "VersionId": kwargs["VersionId"]}


class _Embedder:
    dim = 64
    name = "fixture-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float((i + len(t)) % 7) for i in range(self.dim)] for t in texts]


CORPUS = {
    "notes/old.md": ("text/markdown", b"# Old rate\n\nThe rate limit is 50 requests per second.\n"),
    "notes/new.md": ("text/markdown", b"---\nsupersedes: old\n---\n# New rate\n\nThe rate limit is 100 requests per second.\n"),
    "notes/index.md": ("text/markdown", b"# Index\n\nThe current rule is [[new]].\n"),
    "code/pkg/__init__.py": ("text/plain", b'"""The package."""\n'),
    "code/pkg/util.py": ("text/plain", b"def double(x):\n    return 2 * x\n"),
    "code/pkg/main.py": ("text/plain", b"from pkg import util\n\n\ndef run():\n    return util.double(2)\n"),
}


def _cleanup(tenant: str) -> None:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_source_tombstones WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_ingest_jobs WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


@pytest.fixture
def built() -> Any:
    if not _db_available():
        pytest.skip(db_unreachable_reason())
    from recall.generations import GenerationManager
    from recall.lineage import ChunkerIdentity, EmbedderIdentity, IndexManifestV1, ManifestObjectV1, PipelineIdentity
    from recall.manifest import S3Allowlist, S3ObjectReader

    tenant = "dash-corpus-" + uuid.uuid4().hex[:10]
    objects, entries = {}, []
    for path, (media, data) in CORPUS.items():
        key = f"corpora/{tenant}/{path}"
        objects[("approved", key, "v1")] = data
        entries.append(ManifestObjectV1(f"s3://approved/{key}", "v1", media, len(data), hashlib.sha256(data).hexdigest()))
    manifest = IndexManifestV1(tenant, "corpus-v1", tuple(entries))
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    pipeline = PipelineIdentity(
        EmbedderIdentity("fixture", "fixture-model", 64, revision="r1"),
        ChunkerIdentity("paragraph-pack", 1, {"max_chars": 800, "overlap": 80}),
        fts_configuration={"language": "english", "schema_version": 1},
    )
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    try:
        generation = manager.create(manifest, pipeline)
        manager.build(generation.generation_id, reader, _Embedder(), lambda text: [text])
        manager.validate(generation.generation_id)
        manager.promote(generation.generation_id, unsafe_development=True)
        yield tenant
    finally:
        _cleanup(tenant)


def _named(graph: dict[str, Any], suffix: str) -> str:
    return next(n["id"] for n in graph["nodes"] if n["id"].endswith(suffix))


@requires_db
def test_a_real_generation_reads_back_as_graph_and_view_two(built: str) -> None:
    """D1."""
    db = DashboardDB(TEST_DSN)
    graph = corpus.corpus_graph(db, built)
    old, new, index = _named(graph, "notes/old.md"), _named(graph, "notes/new.md"), _named(graph, "notes/index.md")
    main, util = _named(graph, "pkg/main.py"), _named(graph, "pkg/util.py")
    assert (new, old) in _edges(graph, "supersedes"), "the declared supersession is missing from the graph"
    assert (index, new) in _edges(graph, "link"), "the authored wiki link is missing from the graph"
    assert (main, util) in _edges(graph, "link"), "the Python import is missing from the graph"
    detail = corpus.source_detail(db, built, old)
    assert detail is not None and detail["state"] == "superseded" and detail["superseded_by"] == new
    code = corpus.source_detail(db, built, main)
    assert code is not None and util in code["imports"]
    assert main in (corpus.source_detail(db, built, util) or {}).get("imported_by", [])
    assert corpus.source_detail(db, built, "no/such/file.md") is None


@pytest.fixture
def built_copies() -> Any:
    if not _db_available():
        pytest.skip(db_unreachable_reason())
    from recall.generations import GenerationManager
    from recall.lineage import ChunkerIdentity, EmbedderIdentity, IndexManifestV1, ManifestObjectV1, PipelineIdentity
    from recall.manifest import S3Allowlist, S3ObjectReader

    tenant = "dash-copies-" + uuid.uuid4().hex[:10]
    sources = {
        ("approved-one", "README.md"): (
            b"---\ntype: reference\n---\n# First copy\n\nFirst paragraph.\n\nMore first copy.\n"
        ),
        ("approved-two", "README.md"): (
            b"---\ntype: feedback\nvalid_until: 2099-12-31\n---\n# Second copy\n\nOnly second copy.\n"
        ),
        ("approved-one", "CHANGES.md"): b"---\nsupersedes: README\n---\n# Changes\n\nUpdated first copy.\n",
    }
    objects, entries = {}, []
    for (bucket, path), data in sources.items():
        key = f"corpora/{tenant}/{path}"
        objects[(bucket, key, "v1")] = data
        entries.append(ManifestObjectV1(
            f"s3://{bucket}/{key}", "v1", "text/markdown", len(data), hashlib.sha256(data).hexdigest(),
        ))
    manifest = IndexManifestV1(tenant, "copies-v1", tuple(entries))
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved-one/corpora/,approved-two/corpora/"))
    pipeline = PipelineIdentity(
        EmbedderIdentity("fixture", "fixture-model", 64, revision="r1"),
        ChunkerIdentity("paragraph-pack", 1, {"max_chars": 800, "overlap": 80}),
        fts_configuration={"language": "english", "schema_version": 1},
    )
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    try:
        generation = manager.create(manifest, pipeline)
        manager.build(
            generation.generation_id, reader, _Embedder(),
            lambda text: [piece for piece in text.split("\n\n") if piece.strip()],
        )
        manager.validate(generation.generation_id)
        manager.promote(generation.generation_id, unsafe_development=True)
        yield tenant, f"s3://approved-one/corpora/{tenant}/README.md", f"s3://approved-two/corpora/{tenant}/README.md"
    finally:
        _cleanup(tenant)


@requires_db
def test_generation_graph_keeps_same_path_copies_separate(built_copies: Any) -> None:
    """Node tests/test_dashboard_corpus.py::test_generation_graph_keeps_same_path_copies_separate.

    The corpus graph must retain each root's copy and supersession edge.
    Red-proved against 50cb206e1a4e593d495fca931570d6c303016154:
    corpus_graph merged README copies and failed the node-ID assertion.
    """
    tenant, first, second = built_copies
    graph = corpus.corpus_graph(DashboardDB(TEST_DSN), tenant, today=date.fromisoformat(TODAY))
    nodes = {node["id"]: node for node in graph["nodes"]}

    assert set(nodes) == {first, second, "CHANGES.md"}, "same-path copies lost their source identities"
    assert nodes[first]["state"] == "superseded" and nodes[second]["state"] == "current"
    assert nodes[first]["chunks"] == 3 and nodes[second]["chunks"] == 2
    assert nodes[first]["folder"] == nodes[second]["folder"] == ""
    assert _edges(graph, "supersedes") == {("CHANGES.md", first)}


@requires_db
@pytest.mark.parametrize("copy", ["first", "second"])
def test_source_uri_detail_reads_only_selected_copy(built_copies: Any, copy: str) -> None:
    """Node tests/test_dashboard_corpus.py::test_source_uri_detail_reads_only_selected_copy.

    A source URI detail must contain only that copy's chunks and metadata.
    Red-proved against 50cb206e1a4e593d495fca931570d6c303016154:
    source_detail combined both copies and failed the chunk-count assertion.
    """
    tenant, first, second = built_copies
    source = first if copy == "first" else second
    expected = 3 if copy == "first" else 2
    detail = corpus.source_detail(DashboardDB(TEST_DSN), tenant, source, today=date.fromisoformat(TODAY))

    assert detail is not None
    assert detail["chunk_count"] == len(detail["chunks"]) == expected, "detail counted another root's copy"
    assert detail["file"] == detail["source_uri"] == source
    assert detail["type"] == ("reference" if copy == "first" else "feedback")
    assert detail["valid_until"] == (None if copy == "first" else "2099-12-31")
    text = "\n".join(chunk["text"] for chunk in detail["chunks"])
    assert ("First copy" if copy == "first" else "Second copy") in text
    assert ("Second copy" if copy == "first" else "First copy") not in text
    assert detail["state"] == ("superseded" if copy == "first" else "current")
    assert detail["superseded_by"] == ("CHANGES.md" if copy == "first" else None)


@requires_db
def test_ambiguous_path_detail_refuses_to_choose_copy(built_copies: Any) -> None:
    """Node tests/test_dashboard_corpus.py::test_ambiguous_path_detail_refuses_to_choose_copy.

    An ambiguous bare path must not choose or merge copies.
    Red-proved against 50cb206e1a4e593d495fca931570d6c303016154:
    source_detail returned a mixed result and failed the None assertion.
    """
    tenant, _first, _second = built_copies

    assert corpus.source_detail(DashboardDB(TEST_DSN), tenant, "README.md") is None, (
        "an ambiguous bare path chose or merged two source copies"
    )


# ------------------------------------------------------------------ CCA on #904


def test_stdlib_names_and_parenthesised_imports_resolve_as_python_does() -> None:
    """G4. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: deleting the
    `_STDLIB` check in `resolve_imports` failed at "a stdlib import resolved to a corpus file";
    restoring `+` for the names group of `_IMPORT_PATTERN` failed at "a parenthesised import
    linked nothing". Restored, passes."""
    files = {"recall/lite/inspect.py", "recall/ops/secrets.py", "recall/main.py", "pkg/__init__.py"}
    rows = [
        ("recall/main.py", None, None, "inspect"),
        ("recall/main.py", "secrets", "token_hex", None),
        ("recall/main.py", "recall.lite", "inspect", None),
    ]
    pairs = set(corpus.resolve_imports(rows, files))
    assert ("recall/main.py", "recall/ops/secrets.py") not in pairs, "a stdlib import resolved to a corpus file"
    assert pairs == {("recall/main.py", "recall/lite/inspect.py")}, "a stdlib import resolved to a corpus file"
    text = "from pkg import (\n    a,\n    b,\n)\n"
    matches = [(m.group(1), m.group(2), m.group(3)) for m in re.finditer(corpus._IMPORT_PATTERN, text, re.M)]
    found = corpus.resolve_imports([("recall/main.py", *m) for m in matches], files)
    assert ("recall/main.py", "pkg/__init__.py") in found, "a parenthesised import linked nothing"


def test_copies_of_one_path_keep_their_own_shared_names() -> None:
    """G5. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: keying mentions by
    the bare path again (`by_entity[entity].add(file)`) failed at "a mention did not join the copy
    that holds it". Restored, passes."""
    graph = SupersessionGraph(collided=frozenset({"README.md"}))
    files = [
        ("README.md", 1, None, None, None, None, ["A"], "s3://a/README.md"),
        ("README.md", 1, None, None, None, None, ["B"], "s3://b/README.md"),
        ("x.md", 1, None, None, None, None, ["X"], "s3://a/x.md"),
    ]
    mentions = [("e1", "README.md", "s3://a/README.md"), ("e1", "x.md", "s3://a/x.md")]
    built = corpus.assemble_graph(files, graph, [], mentions, [], today=TODAY, tenant="t", generation="g")
    assert {n["id"] for n in built["nodes"]} == {"s3://a/README.md", "s3://b/README.md", "x.md"}
    assert _edges(built, "shares") == {("s3://a/README.md", "x.md")}, "a mention did not join the copy that holds it"


def test_a_forgotten_successor_is_noted_and_a_future_file_is_not_yet_valid() -> None:
    """G5. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: dropping the
    forgotten-memo note failed at "a file hidden by a forgotten memo went unexplained"; dropping the
    `not_yet_valid` branch of `_state` failed at the state assertion. Restored, passes."""
    future = ("soon.md", 1, None, None, "2027-01-01", None, ["SOON"])
    built = _graph([_file("a.md"), future], winner={"a.md": "ghost.md"})
    states = {n["id"]: n["state"] for n in built["nodes"]}
    assert states == {"a.md": "superseded", "soon.md": "not_yet_valid"}
    assert any("forgotten memo" in note for note in built["notes"]), "a file hidden by a forgotten memo went unexplained"


def test_a_lite_store_says_the_corpus_pages_need_postgres(root: Path) -> None:
    """P4. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: deleting the lite
    checks in `_graph_json` and `_corpus_source_page` failed at "a lite store was told its database
    did not answer". Restored, passes."""
    app = DashboardApp(root, port=8765, token="tok", db=DashboardDB(f"sqlite:///{root / 'lite.db'}"))
    graph = app.handle("GET", "/api/graph.json?corpus=memory", SIGNED)
    page = app.handle("GET", "/corpus/source?file=a.md", SIGNED)
    assert graph.status == page.status == 501, "a lite store was told its database did not answer"
    assert b"lite" in graph.body and b"did not answer" not in page.body


def test_concurrent_misses_build_one_graph_and_the_cache_is_per_tenant(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P5. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: building outside
    the per-corpus build lock failed at "two concurrent misses each built the corpus graph"; a cache
    read that falls back to another corpus's entry failed at "one tenant's graph was served for
    another". Restored, passes."""
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def slow(db: DashboardDB, tenant: str) -> dict[str, Any]:
        calls.append(tenant)
        started.set()
        release.wait(5)
        return {"nodes": [], "edges": [], "counts": {"memos": 0}, "notes": [], "origin": "database", "tenant": tenant}

    monkeypatch.setattr(corpus, "corpus_graph", slow)
    app = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"))
    results: list[int] = []
    first = threading.Thread(target=lambda: results.append(app.handle("GET", "/api/graph.json?corpus=alpha", SIGNED).status))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=lambda: results.append(app.handle("GET", "/api/graph.json?corpus=alpha", SIGNED).status))
    second.start()
    # Give the second request time to either wait on the build lock or (unguarded) start its own
    # build; a second build shows up in `calls` at once, so stop waiting as soon as it does.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(calls) < 2:
        time.sleep(0.02)
    release.set()
    first.join(5)
    second.join(5)
    assert results == [200, 200]
    assert calls == ["alpha"], "two concurrent misses each built the corpus graph"
    beta = app.handle("GET", "/api/graph.json?corpus=beta", SIGNED)
    assert beta.status == 200 and b'"beta"' in beta.body, "one tenant's graph was served for another"
    monkeypatch.setattr(dashboard_server, "CORPUS_GRAPH_CACHE_SECONDS", 0)
    app.handle("GET", "/api/graph.json?corpus=alpha", SIGNED)
    assert calls == ["alpha", "beta", "alpha"], "an expired corpus graph was served from the cache"


def test_view_two_names_a_forgotten_successor_and_an_ambiguous_claim(
    root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P6. Red proof, recorded 2026-10-27 on branch claude/cca-pr910-descending: linking the
    successor whatever `superseded_by_forgotten` says failed at "a forgotten memo was linked as a
    page"; the old "match no file" wording failed at the ambiguous-claim assertion. Restored, passes."""
    detail = _detail(superseded_by="gone.md", superseded_by_forgotten=True, unresolved_claims=["dup"])
    monkeypatch.setattr(corpus, "source_detail", lambda db, tenant, file: detail)
    app = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"))
    body = app.handle("GET", "/corpus/source?file=docs/x.md", SIGNED).body.decode()
    assert "a forgotten memo" in body and "file=gone.md" not in body, "a forgotten memo was linked as a page"
    assert "match several files" in body and "match no file" not in body


class _RecordingDB(DashboardDB):
    statements: list[tuple[str, Any]]

    @contextmanager
    def connect(self):  # type: ignore[no-untyped-def,override]
        recorded = self.statements

        class _Cursor:
            def fetchone(self) -> None:
                return None

            def fetchall(self) -> list[Any]:
                return []

        class _Connection:
            def execute(self, sql: str, params: Any = None) -> _Cursor:
                recorded.append((sql, params))
                return _Cursor()

        yield _Connection()


def test_every_corpus_query_runs_under_the_tenant_setting() -> None:
    """R1: a role under row-level security sees only its tenant's rows because the first statement
    on every connection sets `recall.tenant_id`. Red proof, recorded 2026-10-27 on branch
    claude/cca-pr910-descending: deleting the `set_config` call in `corpus._tenant` failed at "a
    corpus query ran before the tenant was set". Restored, passes."""
    for read in (lambda db: corpus.corpus_graph(db, "acme"), lambda db: corpus.source_detail(db, "acme", "a.md")):
        db = _RecordingDB("postgresql://fake")
        object.__setattr__(db, "statements", [])
        read(db)
        assert db.statements and db.statements[0] == ("select set_config(%s, %s, false)", (corpus.TENANT_GUC, "acme")), (
            "a corpus query ran before the tenant was set"
        )
