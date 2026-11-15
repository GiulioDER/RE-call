"""The semantic graph on a lite store (`recall.lite.graph`): built, stored, cleared and drawn.

Invariants and the failure each one catches:
- G1 a refresh stores the graph, and what loads back from the file is exactly what
  `build_semantic_graph` builds from the same chunks: an authored link and a supersession included.
  Catches a refresh that leaves the graph unbuilt, and rows that do not round-trip.
- G2 every corpus write clears the graph in its own transaction: after a forget no graph row is
  left in the file. Catches a forgotten memo's names outliving it in derived rows.
- G3 a graph stored before a write that did not clear it (a release from before these tables)
  reads as no graph. Catches a stale graph served as the current one.
- G4 a rebuild during which another process forgets a memo stores nothing: its graph quotes a
  corpus the file no longer holds.
- G5 `recall graph rebuild` and `recall graph status` work on a lite DSN, and a `--generation`
  naming another corpus is refused by name.
- G6 the dashboard draws the store's corpus graph and opens a file's page from it, with the
  file opened read-only, from the stored graph and, when none is stored, from one built in memory.
- G7 `LiteStore.related_chunks` answers `source` and `ordinal` from the seed's rows, as
  `PgVectorStore.related_chunks` does, and `recall_related` on a lite store then returns what the
  corpus scan returned, without scanning.
- G8 on PostgreSQL, `recall graph status --generation G` reports G's readiness marker, built and
  not yet promoted. Catches the generation argument dropped, which reads the active one instead.
- G9 to G13, from the CCA audit of PR 1308 (run 20261115T105922Z): a forget by a release without
  the graph clears it (SEC-001); an unchanged refresh builds a missing graph (BUG-007); a rebuild
  builds with the store lock released (PERF-001); a file page lists only its own copy's imports
  (BUG-002); a file page keeps the in-memory graph until the corpus moves (PERF-002), and G14
  never for another file at the same path and corpus version (the regression review of that
  fix). G7 now
  pins `source` ranked as the resolver ranks it, at a `max_items` that cuts the source (BUG-001).
  Red proof, 2026-11-15 on the remote test host, each mutation alone against the fixed tree, the
  named test failing in its own assertion, then restored and the module green (17 passed, G8
  skipped without a database):
  - M14 `LiteStore.related_chunks` ordering `source` by ordinal again: G7's store test got
    `['s0', 's1', 's2']` and `test_g7_recall_related_on_lite_returns_what_the_scan_did[source-3]`
    `['s2', 's1', 's0']`, not `['s2', 's4', 's1']`.
  - M15 `lite_db.source_detail` scanning every row for imports: G12 listed `pkg/right.py` too.
  - M16 the trigger's condition made `WHEN 0`: G9 "another release's forget left the graph".
  - M17 `and store.graph_current()` removed from `refresh`: G10 "an unchanged refresh left the
    store without its graph".
  - M18 `build_graph` run under `self._lock` again: G11 "a read waited 1.51s for the graph build".
  - M19 the kept graph never reused: G13 "every file page built the whole graph". M20 the kept
    graph reused whatever the version: G13 "a graph was kept past a corpus write".
  - M21 the kept graph's stamp made the corpus version alone: G14 `[False]` "a graph kept from the
    deleted file was drawn" (the old corpus's `references` relation and generation id). M22 the
    stamp made each row's id and `indexed_at` only: G14 `[True]`, under the frozen clock, the same.
  - M4 re-run on the changed G3 (`corpus_version` comparison removed from `read_marker`): "a
    stale graph read as ready".

Red proof, 2026-11-15 on the deployment host inside the test bounds: each mutation applied alone,
the named test run and failing in its own assertion, then the file restored and the whole module
run green (37 passed with `tests/test_lite_cli_refusals.py`).
- M1 `test_g1_a_refresh_stores_the_graph_its_chunks_build`, the `rebuild_semantic_graph` call
  removed from `recall.lite.refresh._refresh_locked`: "the refresh left the graph unbuilt".
- M2 the same test, `recall.lite.graph.write_graph` writing no `graph_relation_evidence` rows: "the
  stored graph does not load back as the graph its chunks build" (it loaded as None).
- M3 `test_g2_...`, `clear_graph` removed from `LiteStore._bump`: "graph rows outlived the forget".
- M4 `test_g3_...`, the `corpus_version` comparison removed from `read_marker`: "a stale graph
  read as ready".
- M5 `test_g4_...`, the version check in `LiteStore.rebuild_semantic_graph` made `if True:`: "a
  graph of a corpus the file no longer holds was stored".
- M6 `test_g5_...`, the lite branch removed from `graph_cmd._cmd_graph`: the CLI exited
  "--generation is required on PostgreSQL" instead of rebuilding.
- M7 `test_g6_...`, the lite delegation removed from `corpus.corpus_graph`: 503 where 200 was
  asserted (psycopg refused the `sqlite:///` string). M8, the same for `corpus.source_detail`: the
  file's page was not 200. M9, `graph_of` building from no chunks: "no graph drawn when none is
  stored".
- M10 `test_g7_related_chunks_reads_the_seed_source_as_postgres_does`, the ordinal window `<= 2`
  made `<= 3`: s3 and s0, s6 came back. M11 `test_g7_recall_related_on_lite_returns_what_the_scan_did`,
  `LiteStore.related_chunks` returning None for every relation: "recall_related scanned the corpus".
- M12 `tests/test_lite_cli_refusals.py`, `graph` put back in `LITE_REFUSED`: "a command is both
  served and refused on lite".
- M13 `test_g8_graph_status_reports_a_generation_marker`, needing a database, was red-proved in CI
  (2026-11-15, run 43703934093, `floor` job, PostgreSQL 16): `graph_cmd._cmd_graph` passing None
  instead of `args.generation` to `graph_readiness` failed at its assertion, the CLI having raised
  `NoActiveGeneration` for the unpromoted generation's tenant; 1 failed, 14516 passed. Reverted
  in the next commit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import pytest

from recall.calibration import Calibration
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp
from recall.lite import LiteStore
from recall.lite.graph import build_graph, load_graph
from recall.lite.refresh import Outcome, refresh
from recall.related import trusted_related
from recall.trust_policy import TrustPolicy
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_generations import _Embedder, _manifest, _pipeline, _reader, manager  # noqa: F401  # manager: fixture
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos

GRAPH_TABLES = ("graph_entities", "graph_mentions", "graph_relations", "graph_relation_evidence")
SIGNED = {"Host": "127.0.0.1:8765", "Authorization": "Bearer tok"}


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Background memos, so a refresh can calibrate, and three that link: a -> b, b supersedes c."""
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    root = tmp_path / "memory"
    _write_memos(root, 4)
    (root / "alpha.md").write_bytes(b"# Alpha\n\nThe alpha plan builds on [the beta plan](beta.md).\n")
    (root / "beta.md").write_bytes(b"---\nsupersedes: gamma.md\n---\n# Beta\n\nThe beta plan replaces gamma.\n")
    (root / "gamma.md").write_bytes(b"# Gamma\n\nThe gamma plan, retired.\n")
    return root


@pytest.fixture
def store(tmp_path: Path, folder: Path) -> Iterator[LiteStore]:
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as opened:
        report = refresh(opened, [folder], ContentWordEmbedder(), force=True)  # type: ignore[arg-type]
        assert report.outcome is Outcome.REFRESHED, report.to_dict()
        yield opened


def _source(store: LiteStore, name: str) -> str:
    return next(c.source for c in store.iter_chunks() if Path(c.source).name == name)


def _rows(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as c:
        return {table: c.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in GRAPH_TABLES}  # noqa: S608


def _names(graph: object) -> set[tuple[str, str, str]]:
    """`(relation, subject file, object file)` of the graph's relations."""
    entities = {e.id: e.canonical_name for e in graph.entities}  # type: ignore[attr-defined]
    return {(r.relation, Path(entities[r.subject_id]).name, Path(entities[r.object_id]).name) for r in graph.relations}  # type: ignore[attr-defined]


def test_g1_a_refresh_stores_the_graph_its_chunks_build(store: LiteStore) -> None:
    """G1."""
    assert store.graph_readiness().ready, "the refresh left the graph unbuilt"
    built = build_graph(list(store.iter_chunks()), tenant=store.tenant, corpus_fingerprint=store.corpus_fingerprint())
    with store._lock:
        loaded = load_graph(store._conn, store.tenant)
    assert loaded == built, "the stored graph does not load back as the graph its chunks build"
    assert loaded.generation_id == store.lite_generation_id()
    assert {("references", "alpha.md", "beta.md"), ("supersedes", "gamma.md", "beta.md")} <= _names(loaded)


def test_g2_a_forget_clears_the_graph_in_its_transaction(store: LiteStore) -> None:
    """G2."""
    assert all(_rows(store.path).values())
    with store.erasing():
        store.delete_sources([_source(store, "gamma.md")])
    assert _rows(store.path) == dict.fromkeys(GRAPH_TABLES, 0), "graph rows outlived the forget"
    assert not store.graph_readiness().ready


def test_g3_a_graph_from_before_an_uncleared_write_is_no_graph(store: LiteStore) -> None:
    """G3. The write is the corpus version moving with the rows left: the file's clearing trigger
    (G9) dropped first, so the marker's own version check is what is tested."""
    with sqlite3.connect(store.path) as c:
        c.execute("DROP TRIGGER graph_cleared_by_corpus_write")
        c.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'corpus_version'")
        assert c.execute("SELECT count(*) FROM graph_entities").fetchone()[0], "nothing left to read as stale"
    assert not store.graph_readiness().ready, "a stale graph read as ready"
    with store._lock:
        assert load_graph(store._conn, store.tenant) is None, "a stale graph loaded as the current one"


def test_g4_a_rebuild_raced_by_a_forget_stores_nothing(store: LiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """G4."""
    import recall.lite.graph as lite_graph

    gamma = _source(store, "gamma.md")
    real = lite_graph.build_graph

    def build_then_forget(*args: object, **kwargs: object) -> object:
        graph = real(*args, **kwargs)  # type: ignore[arg-type]
        with LiteStore(store.path, dim=DIM) as other:  # another process forgets while this one builds
            other.delete_sources([gamma])
        return graph

    monkeypatch.setattr(lite_graph, "build_graph", build_then_forget)
    store.rebuild_semantic_graph()
    assert _rows(store.path) == dict.fromkeys(GRAPH_TABLES, 0), "a graph of a corpus the file no longer holds was stored"


def test_g5_the_graph_cli_serves_a_lite_store(store: LiteStore) -> None:
    """G5."""
    dsn = "sqlite:///" + store.path.as_posix()
    with store.erasing():
        store.delete_sources([_source(store, "memo-000.md")])  # clears the graph
    rebuilt = run_cli(["--dsn", dsn, "graph", "rebuild"])
    assert '"ready": true' in rebuilt and store.lite_generation_id() in rebuilt, rebuilt
    status = json.loads(run_cli(["--dsn", dsn, "graph", "status"]))
    assert status["ready"] is True and status["relation_count"] >= 2, status
    other = run_cli(["--dsn", dsn, "graph", "status", "--generation", "lite-0000000000000000"])
    assert "EXIT" in other and "recall graph status" in other and "one corpus" in other, other


def _app(store: LiteStore, root: Path) -> DashboardApp:
    return DashboardApp(root, port=8765, token="tok", db=DashboardDB("sqlite:///" + store.path.as_posix()), default_tenant=store.tenant)


def _edges(app: DashboardApp, tenant: str) -> set[tuple[str, str, str]]:
    response = app.handle("GET", f"/api/graph.json?corpus={tenant}", SIGNED)
    assert response.status == 200, response.body[:300]
    return {(Path(e["source"]).name, Path(e["target"]).name, e["kind"]) for e in json.loads(response.body)["edges"]}


def test_g6_the_dashboard_draws_the_lite_corpus_graph_read_only(store: LiteStore, folder: Path) -> None:
    """G6."""
    app = _app(store, folder)
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    assert {("alpha.md", "beta.md", "link"), ("beta.md", "gamma.md", "supersedes")} <= _edges(app, store.tenant)
    page = app.handle("GET", "/corpus/source?" + urlencode({"file": _source(store, "alpha.md")}), SIGNED)
    assert page.status == 200, page.body[:300]
    assert "references" in page.body.decode(), "the file's page does not list its relation"
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before, "the dashboard wrote the store"
    with store.erasing():
        store.delete_sources([_source(store, "memo-001.md")])  # clears the stored graph
    assert not store.graph_readiness().ready
    assert ("alpha.md", "beta.md", "link") in _edges(_app(store, folder), store.tenant), "no graph drawn when none is stored"
    assert _rows(store.path)["graph_entities"] == 0, "the read-only dashboard stored a graph"


@pytest.fixture
def numbered(tmp_path: Path) -> Iterator[LiteStore]:
    """`s.md` with chunks at ordinals 0 to 6, one unnumbered and one numbered `true`, and another source."""
    with LiteStore(tmp_path / "numbered.db", dim=4) as opened:
        chunks = [Chunk(f"s{i}", "s.md", f"part {i}", {"file": "s.md", "ord": i}) for i in range(7)]
        chunks += [Chunk("s-none", "s.md", "unnumbered", {"file": "s.md"}), Chunk("s-bool", "s.md", "bool", {"file": "s.md", "ord": True})]
        chunks.append(Chunk("t0", "t.md", "other", {"file": "t.md", "ord": 3}))
        opened.upsert(chunks, [[1.0, 0.0, 0.0, 0.0]] * len(chunks))
        yield opened


def test_g7_related_chunks_reads_the_seed_source_as_postgres_does(numbered: LiteStore) -> None:
    """G7, the store's answer: `ordinal` is the Postgres window, the two either side of the seed,
    nearest first; `source` is ranked as `recall.related` ranks it, distance from the seed with a
    boolean or missing ordinal at 0, then id, so a LIMIT keeps the seed's neighbours (CCA BUG-001:
    this asserted Postgres's first-by-ordinal `["s0", "s1", "s2"]`, the defect)."""
    seed, near = numbered.related_chunks("s3", "ordinal", 50)  # type: ignore[misc]
    assert seed.id == "s3"
    assert [c.id for c in near] == ["s2", "s4", "s1", "s5"]
    _seed, first = numbered.related_chunks("s3", "source", 3)  # type: ignore[misc]
    assert [c.id for c in first] == ["s2", "s4", "s1"]
    _seed, every = numbered.related_chunks("s3", "source", 50)  # type: ignore[misc]
    assert [c.id for c in every] == ["s2", "s4", "s1", "s5", "s-bool", "s-none", "s0", "s6"]
    assert numbered.related_chunks("s3", "supersession", 5) is None


@pytest.mark.parametrize("max_items", [50, 3])
@pytest.mark.parametrize("relation", ["source", "ordinal"])
def test_g7_recall_related_on_lite_returns_what_the_scan_did(
    numbered: LiteStore, relation: str, max_items: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G7, through `recall.related.trusted_related`, which `recall_related` calls; at 3 the
    store's LIMIT cuts the source, which is where an order other than the resolver's shows."""

    def related(store: LiteStore) -> list[str]:
        result = trusted_related(
            store, "s3", relation=relation, max_items=max_items,  # type: ignore[arg-type]
            calibration=Calibration(embedder="fixture", threshold=0.5), policy=TrustPolicy.development(),
            now=datetime(2026, 11, 14, tzinfo=UTC),
        )
        return [item.chunk.id for item in result.items]

    scanned = []
    with monkeypatch.context() as patch:
        patch.setattr(LiteStore, "related_chunks", lambda *_args: None)
        scanned = related(numbered)
    assert scanned, "the scan related nothing, so the comparison below would prove nothing"

    def no_scan(self: LiteStore, batch_size: int = 1000) -> Iterator[Chunk]:
        raise AssertionError("recall_related scanned the corpus")

    monkeypatch.setattr(LiteStore, "iter_chunks", no_scan)
    assert related(numbered) == scanned


@requires_db
def test_g8_graph_status_reports_a_generation_marker(manager) -> None:  # noqa: F811  # the fixture
    """G8."""
    data = b"# Plan\n\nThe plan builds on [the other plan](other.md).\n"
    manifest = _manifest(manager.tenant_id, data)
    generation = manager.create(manifest, _pipeline("model-a")).generation_id
    manager.build(generation, _reader(manifest, data), _Embedder(1), lambda text: [text])
    out = run_cli(["--dsn", TEST_DSN, "--tenant", manager.tenant_id, "graph", "status", "--generation", generation])
    assert '"ready": true' in out and generation in out, out


def test_g9_a_forget_by_a_release_without_the_graph_clears_it(store: LiteStore) -> None:
    """G9. The write a release from before these tables makes: chunks deleted and the corpus
    version moved, no `clear_graph`. Catches a forgotten memo's names outliving it in the graph."""
    gamma = _source(store, "gamma.md")
    assert all(_rows(store.path).values())
    with sqlite3.connect(store.path) as c:
        c.execute("DELETE FROM chunks WHERE source = ?", (gamma,))
        c.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'corpus_version'")
    assert _rows(store.path) == dict.fromkeys(GRAPH_TABLES, 0), "another release's forget left the graph"
    with sqlite3.connect(store.path) as c:
        assert not c.execute("SELECT count(*) FROM meta WHERE key = 'semantic_graph'").fetchone()[0], "the marker survived"


def test_g10_a_refresh_builds_a_missing_graph_when_no_memo_changed(store: LiteStore, folder: Path) -> None:
    """G10. A store upgraded from a release before the graph: its memos and corpus version as
    the last refresh left them, no graph. Catches the unchanged-folder shortcut skipping the build."""
    with sqlite3.connect(store.path) as c:
        for table in GRAPH_TABLES:
            c.execute(f"DELETE FROM {table}")  # noqa: S608  # a constant table name
        c.execute("DELETE FROM meta WHERE key = 'semantic_graph'")
    assert not store.graph_readiness().ready
    refresh(store, [folder], ContentWordEmbedder())  # type: ignore[arg-type]
    assert store.graph_readiness().ready, "an unchanged refresh left the store without its graph"


def test_g11_a_rebuild_builds_with_the_store_lock_released(store: LiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """G11. A read on the same store (an MCP server's search during its background refresh) waits
    for the rebuild's read, not its build. Catches the lock held across the build."""
    import threading
    import time

    import recall.lite.graph as lite_graph

    with sqlite3.connect(store.path) as c:
        c.execute("DELETE FROM meta WHERE key = 'semantic_graph'")
    real, building = lite_graph.build_graph, threading.Event()

    def slow(*args: object, **kwargs: object) -> object:
        building.set()
        time.sleep(1.5)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(lite_graph, "build_graph", slow)
    worker = threading.Thread(target=store.rebuild_semantic_graph)
    worker.start()
    assert building.wait(30), "the rebuild never reached its build"
    started = time.monotonic()
    with store.snapshot():
        pass
    waited = time.monotonic() - started
    worker.join(30)
    assert waited < 0.75, f"a read waited {waited:.2f}s for the graph build"
    assert store.graph_readiness().ready, "the rebuild stored nothing"


def test_g12_a_file_page_lists_only_its_own_copys_imports(tmp_path: Path) -> None:
    """G12. Two roots hold `pkg/mod.py`, importing different modules; one copy's View 2 lists its
    own import only, as Postgres scans the one source. Catches imports matched by the bare path."""
    from recall.dashboard import lite_db

    vector = [1.0, 0.0, 0.0, 0.0]
    with LiteStore(tmp_path / "copies.db", dim=4) as opened:
        opened.upsert(
            [
                Chunk("a0", "/a/pkg/mod.py", "import pkg.left\n", {"file": "pkg/mod.py", "ord": 0}),
                Chunk("b0", "/b/pkg/mod.py", "import pkg.right\n", {"file": "pkg/mod.py", "ord": 0}),
                Chunk("l0", "/a/pkg/left.py", "LEFT = 1\n", {"file": "pkg/left.py", "ord": 0}),
                Chunk("r0", "/b/pkg/right.py", "RIGHT = 1\n", {"file": "pkg/right.py", "ord": 0}),
            ],
            [vector] * 4,
        )
        dsn, tenant = "sqlite:///" + opened.path.as_posix(), opened.tenant
    page = lite_db.source_detail(dsn, tenant, "/a/pkg/mod.py")
    assert page is not None
    assert page["imports"] == ["pkg/left.py"], page["imports"]


def test_g13_a_file_page_keeps_the_graph_until_the_corpus_moves(numbered: LiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """G13. View 2 builds the in-memory graph once for unchanged rows, and again after a write.
    Catches a whole-graph build on every page, and a kept graph outliving a write."""
    import recall.lite.graph as lite_graph
    from recall.dashboard import lite_db

    monkeypatch.setattr(lite_db, "_GRAPHS", {})
    real, calls = lite_graph.graph_of, []

    def counted(conn: sqlite3.Connection, tenant: str) -> object:
        calls.append(tenant)
        return real(conn, tenant)

    monkeypatch.setattr(lite_graph, "graph_of", counted)
    dsn = "sqlite:///" + numbered.path.as_posix()
    for _ in range(2):
        assert lite_db.source_detail(dsn, numbered.tenant, "s.md") is not None
    assert len(calls) == 1, "every file page built the whole graph"
    numbered.upsert([Chunk("u0", "u.md", "new", {"file": "u.md", "ord": 0})], [[1.0, 0.0, 0.0, 0.0]])
    assert lite_db.source_detail(dsn, numbered.tenant, "s.md") is not None
    assert len(calls) == 2, "a graph was kept past a corpus write"


@pytest.mark.parametrize("frozen", [False, True])
def test_g14_a_kept_graph_is_not_served_for_another_file_at_the_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: bool
) -> None:
    """G14. The store deleted and indexed again at the same path, reaching the same corpus version
    with other content: its file page is the one a fresh read draws. Catches a kept graph trusted
    on the version alone, which drew the old corpus's relations; frozen, under an evaluation clock
    that holds the write time, so the same ids, times and version differ only in their text,
    which catches a kept graph trusted on ids and times."""
    from recall import eval_clock
    from recall.dashboard import lite_db

    monkeypatch.setattr(lite_db, "_GRAPHS", {})
    if frozen:
        monkeypatch.setattr(eval_clock, "_source", None)  # so the clock is uninstalled when the test ends
        eval_clock.install(lambda: datetime(2026, 11, 15, 12, tzinfo=UTC))
    path = tmp_path / "memory.db"

    def index(target: str) -> str:
        for leftover in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            leftover.unlink(missing_ok=True)
        with LiteStore(path, dim=4) as opened:
            opened.upsert(
                [
                    Chunk("a0", "a.md", f"# A\n\nThe plan builds on [the other plan]({target}).\n", {"file": "a.md", "ord": 0}),
                    Chunk("b0", "b.md", "# B\n\nOne plan.\n", {"file": "b.md", "ord": 0}),
                    Chunk("c0", "c.md", "# C\n\nAnother plan.\n", {"file": "c.md", "ord": 0}),
                ],
                [[1.0, 0.0, 0.0, 0.0]] * 3,
            )
            return opened.tenant

    tenant = index("b.md")
    dsn = "sqlite:///" + path.as_posix()
    first = lite_db.source_detail(dsn, tenant, "a.md")
    tenant = index("c.md")  # the same path, corpus version 1 again
    served = lite_db.source_detail(dsn, tenant, "a.md")
    lite_db._GRAPHS.clear()
    fresh = lite_db.source_detail(dsn, tenant, "a.md")
    assert first is not None and fresh is not None and first["relations"] != fresh["relations"], "nothing to tell apart"
    assert served == fresh, "a graph kept from the deleted file was drawn"
