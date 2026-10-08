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
- D1 against a real generation built by the generation pipeline: the graph holds the declared
  supersession, the authored wiki link and the Python import, and View 2's data names what
  replaces a file, what it imports and what imports it.
"""

from __future__ import annotations

import hashlib
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import psycopg
import pytest

from recall.dashboard import corpus
from recall.dashboard import db as dbq
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp
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
