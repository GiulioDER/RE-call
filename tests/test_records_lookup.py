"""`lookup_records`: audit records by exact key, returned exactly as authored (Task 1.2).

Invariants and the failure each one catches:
- K1 one record, three anchor queries, three matches: equal hash `exact`, other hash
  `symbol_changed`, other symbol in the same file `file_only`.
- K2 a record with several anchors reports each anchor's match, and its best.
- K3 every golden record (contract C4) comes back equal to its `expected_records.json` entry, with
  `name`, `description`, `valid_from` and `supersedes` as the memo declares them.
- K4 superseded, retracted and expired records are returned and marked, never hidden.
- K5 the reverse `caused` edge names the records whose `caused_by` names this one.
- K6 `ids` and `all_records` mark every match `none`; `all_records` pages with `truncated`.
- K7 more than 200 anchors, or more matches than the limit, is refused, never truncated.
- K8 a code chunk's `code_anchor` is a record of kind `code`, matched by the same rules; the
  chunks sharing one anchor (a split symbol) are one record, reported from the lowest-ordinal
  chunk with every chunk in `chunk_ids`; a null `repo` never matches a `repo` query.
- K9 on PostgreSQL the generation store reads the ACTIVE generation only (DB suite).
- K10 PostgreSQL and lite return the same records for the same memos (DB suite, through the CLI).
- K11 the source policy applies before edges are derived: a denied record is not returned, is not
  named in another record's `caused`, and a denied successor is `<withheld>`, never None (None
  would present a retracted record as current).

Red proof, recorded 2026-10-21 on the deployment host inside the test bounds, each failing in its
own assertion, then restored and green:
- K2 `...::test_multi_anchor_reports_each`: `_best` mutated from `min` to `max` over the match
  order: PERF-001~2 is no longer returned for its exact first anchor (the membership assertion).
- K2b `...::test_best_match_is_listed_first`: the sort mutated to put the worst match first.
- K6 `...::test_ids_and_all`: `offset` ignored when paging: the second page repeats the first.
- K7 `...::test_over_200_anchors_refused`: the limit mutated to `> MAX_QUERY_ANCHORS + 1`:
  201 anchors are accepted (`DID NOT RAISE`).
- K8 `...::test_code_anchor_chunks_are_code_records`: code records keyed per chunk id instead of
  per anchor: the split symbol comes back twice.
- K1 `tests/test_records_lookup.py::test_exact_symbol_changed_file_only`: `recall.records._match`
  mutated so a hash difference counts as exact (`return "exact"` for equal symbols): the
  other-hash query reports `exact`, not `symbol_changed`.
- K3 `...::test_round_trip_returns_record_unchanged`: the envelope's `supersedes` mutated to
  `None`: BUG-002's `supersedes` assertion fails.
- K4 `...::test_superseded_and_retracted_are_marked_not_hidden`: `lookup_records` mutated to skip
  a record whose file has a successor: the retracted false positive, the superseded bug and the
  retracted lesson are missing (the "were hidden" assertion).
- K5 `...::test_caused_reverse_edge`: `caused` mutated to index each record's own `caused_by`
  (the forward edge): BUG-001's `caused` is empty.
- K9 `...::test_reads_active_generation_only`: `GenerationStore.record_chunks` mutated to drop
  its `generation_id` predicate: the ready, unpromoted generation's record is returned too.
- K11 `...::test_denied_sources_do_not_leak_through_edges`: the policy filter moved back to after
  the edges are built (the first version): BUG-001's `caused` names the denied BUG-002.
- K10 `...::test_postgres_and_lite_return_the_same_records`: the legacy PostgreSQL read
  (`recall.records._record_chunks`) mutated to return chunks without their `source`
  (`source=""`): every record's `source` differs between the two stores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall.document import parse_document
from recall.lite import LiteStore
from recall.records import WITHHELD, RecordsQueryError, lookup_records
from recall.types import Chunk
from tests.audit_memory_support import EXPECTED, MEMOS, REPO, copy_memos, indexed_lite_store, lite_dsn, memo_name
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import run_cli
from tests.test_generations import (  # noqa: F401  # `manager` is a fixture, used by name
    _Embedder,
    _manifest,
    _pipeline,
    _reader,
    _ready,
    manager,
)

BUG_001 = EXPECTED["RE-call__0123456789ab__BUG-001.md"]
BUG_002 = EXPECTED["RE-call__abcdef012345__BUG-002.md"]
PERF = EXPECTED["RE-call__222222222222__PERF-001~2.md"]
FP = EXPECTED["RE-call__fp__run-20261021-9f8e7d__BUG-003.md"]
LESSON = EXPECTED["RE-call__lesson__6a3ae35062b9.md"]
P2_4 = EXPECTED["RE-call__fedcba987654__P2-4.md"]
OTHER_SHA = "0" * 64


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> LiteStore:
    return indexed_lite_store(tmp_path_factory.mktemp("golden"))


def _by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in result["records"]}


def _anchor(record: dict[str, Any], index: int = 0, **changes: Any) -> dict[str, Any]:
    return {**record["anchors"][index], **changes}


def test_exact_symbol_changed_file_only(store: LiteStore) -> None:
    """K1."""
    queries = {
        "exact": _anchor(BUG_001),
        "symbol_changed": _anchor(BUG_001, code_sha=OTHER_SHA),
        "file_only": _anchor(BUG_001, symbol="Pool.release", code_sha=OTHER_SHA),
    }
    for want, anchor in queries.items():
        found = _by_id(lookup_records(store, repo=REPO, anchors=[anchor], kinds=["bug"]))
        assert found[BUG_001["id"]]["anchor_match"] == want, (want, found[BUG_001["id"]])
        assert [m["match"] for m in found[BUG_001["id"]]["anchor_matches"]] == [want]
    unrelated = lookup_records(store, repo=REPO, anchors=[{"file": "nowhere.py", "symbol": None, "code_sha": None}])
    assert unrelated["records"] == [], "a record with no anchor in the queried file was returned"
    other_repo = lookup_records(store, repo="someone/else", anchors=[_anchor(BUG_001)])
    assert other_repo["records"] == [], "an anchor matched across repositories"


def test_multi_anchor_reports_each(store: LiteStore) -> None:
    """K2."""
    found = _by_id(lookup_records(store, repo=REPO, anchors=[_anchor(PERF, 0)]))
    assert PERF["id"] in found, "a record whose first anchor matches exactly was not returned"
    one = found[PERF["id"]]
    assert [m["match"] for m in one["anchor_matches"]] == ["exact", "none"]
    assert [m["anchor"] for m in one["anchor_matches"]] == PERF["anchors"]
    assert one["anchor_match"] == "exact"
    both = _by_id(lookup_records(store, repo=REPO, anchors=[_anchor(PERF, 1), _anchor(PERF, 0, code_sha=OTHER_SHA)]))
    assert [m["match"] for m in both[PERF["id"]]["anchor_matches"]] == ["symbol_changed", "exact"]


def test_best_match_is_listed_first(store: LiteStore) -> None:
    result = lookup_records(
        store,
        repo=REPO,
        anchors=[{"file": "recall/store.py", "symbol": "Store._with_retry_shared", "code_sha": OTHER_SHA}],
    )
    order = [r["anchor_match"] for r in result["records"]]
    assert order == sorted(order, key=["exact", "symbol_changed", "file_only"].index), order
    assert result["records"][-1]["id"] == LESSON["id"], "the lesson's module anchor matches by file only"


def test_round_trip_returns_record_unchanged(store: LiteStore) -> None:
    """K3: the contract's RE-call half, over every golden memo."""
    records = {memo_name(r["source"]): r for r in lookup_records(store, all_records=True)["records"]}
    assert set(records) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        got = records[name]
        assert got["recall_record"] == expected, name
        assert json.dumps(got["recall_record"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) == (
            json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ), name
        meta = parse_document((MEMOS / name).read_text(encoding="utf-8")).meta
        assert got["name"] == meta["name"], name
        assert got["description"] == meta["description"], name
        assert got["valid_from"] == meta["valid_from"] == expected["date"], name
        assert got["supersedes"] == (meta.get("supersedes") or None), name
        assert got["id"] == expected["id"] and got["kind"] == expected["kind"], name
    assert records["RE-call__abcdef012345__BUG-002.md"]["supersedes"] == "RE-call__fedcba987654__P2-4"


def test_superseded_and_retracted_are_marked_not_hidden(tmp_path: Path) -> None:
    """K4, with one memo made expired as well."""
    folder = copy_memos(tmp_path / "memos")
    expired = folder / "RE-call__0123456789ab__BUG-001.md"
    expired.write_bytes(expired.read_bytes().replace(b"valid_from: 2026-08-04\n", b"valid_from: 2026-08-04\nvalid_until: 2026-09-01\n"))
    store = indexed_lite_store(tmp_path, memos=folder)
    found = _by_id(lookup_records(store, repo=REPO, anchors=[_anchor(FP, 0), _anchor(FP, 1)]))
    hidden = {FP["id"], P2_4["id"], LESSON["id"], BUG_001["id"]} - set(found)
    assert not hidden, f"superseded, retracted or expired records were hidden: {sorted(hidden)}"
    assert found[FP["id"]]["superseded_by"] == "RE-call__fp__run-20261021-9f8e7d__BUG-003~retract.md"
    assert found[P2_4["id"]]["superseded_by"] == "RE-call__abcdef012345__BUG-002.md"
    assert found[LESSON["id"]]["superseded_by"] == "RE-call__lesson__6a3ae35062b9~retract.md"
    assert found[BUG_001["id"]]["valid_until"] == "2026-09-01"
    assert found[BUG_002["id"]]["superseded_by"] is None and found[BUG_002["id"]]["valid_until"] is None


def test_caused_reverse_edge(store: LiteStore) -> None:
    """K5: BUG-002 was caused by BUG-001's fix, so BUG-001 lists BUG-002 under `caused`."""
    found = _by_id(lookup_records(store, ids=[BUG_001["id"], BUG_002["id"]]))
    assert found[BUG_001["id"]]["caused"] == [BUG_002["id"]]
    assert found[BUG_002["id"]]["caused"] == []
    assert found[BUG_002["id"]]["recall_record"]["caused_by"] == [BUG_001["id"]]


def test_ids_and_all(store: LiteStore) -> None:
    """K6."""
    by_ids = lookup_records(store, ids=[FP["id"], "RE-call:missing"])
    assert [r["id"] for r in by_ids["records"]] == [FP["id"]]
    assert {m["match"] for r in by_ids["records"] for m in r["anchor_matches"]} == {"none"}
    assert by_ids["records"][0]["anchor_match"] == "none"
    everything = lookup_records(store, all_records=True)
    assert len(everything["records"]) == len(EXPECTED) and everything["truncated"] is False
    assert everything["tenant_id"] == store.tenant and everything["generation_id"] == store.generation_id
    assert {r["kind"] for r in lookup_records(store, all_records=True, kinds=["lesson", "lesson_retraction"])["records"]} == {
        "lesson",
        "lesson_retraction",
    }
    assert lookup_records(store, all_records=True, repo="someone/else")["records"] == []
    first = lookup_records(store, all_records=True, limit=4)
    rest = lookup_records(store, all_records=True, limit=4, offset=4)
    last = lookup_records(store, all_records=True, limit=4, offset=8)
    assert (first["truncated"], rest["truncated"], last["truncated"]) == (True, True, False)
    paged = [r["id"] for page in (first, rest, last) for r in page["records"]]
    assert paged == [r["id"] for r in everything["records"]]


def test_over_200_anchors_refused(store: LiteStore) -> None:
    """K7."""
    anchors = [{"file": f"f{i}.py", "symbol": None, "code_sha": None} for i in range(201)]
    with pytest.raises(RecordsQueryError, match="at most 200"):
        lookup_records(store, repo=REPO, anchors=anchors)
    with pytest.raises(RecordsQueryError, match="more than the limit of 2"):
        lookup_records(store, ids=[BUG_001["id"], BUG_002["id"], FP["id"]], limit=2)
    with pytest.raises(RecordsQueryError, match="exactly one"):
        lookup_records(store, ids=[BUG_001["id"]], all_records=True)
    with pytest.raises(RecordsQueryError, match="needs `repo`"):
        lookup_records(store, anchors=[_anchor(BUG_001)])
    with pytest.raises(RecordsQueryError, match="unknown kinds"):
        lookup_records(store, all_records=True, kinds=["rumour"])


class _DenyPolicy:
    """A source policy that refuses the named files and allows the rest."""

    def __init__(self, *denied: str) -> None:
        self.denied = set(denied)

    def decide(self, file: str, context: object) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(allowed=Path(file).name not in self.denied)


def test_denied_sources_do_not_leak_through_edges(store: LiteStore) -> None:
    """K11."""
    policy = _DenyPolicy("RE-call__abcdef012345__BUG-002.md", "RE-call__fp__run-20261021-9f8e7d__BUG-003~retract.md")
    found = _by_id(lookup_records(store, all_records=True, security_policy=policy, access_context=object()))
    assert BUG_002["id"] not in found, "a denied record was returned"
    assert found[BUG_001["id"]]["caused"] == [], "a denied record was named through `caused`"
    assert found[P2_4["id"]]["superseded_by"] == WITHHELD
    assert found[FP["id"]]["superseded_by"] == WITHHELD, "a retracted record read as current"
    assert found[LESSON["id"]]["superseded_by"] == "RE-call__lesson__6a3ae35062b9~retract.md"


class _CodeStore:
    """A code tenant's chunks, as Session 3's chunker writes them: a `code_anchor` per symbol."""

    tenant = "re-call-code"
    generation_id = "gen-code"

    def __init__(self, anchors: list[dict[str, Any]]) -> None:
        # One file, one chunk per anchor in order, so `c<i>` is ordinal i of `recall/pool.py`.
        self.chunks = [
            Chunk(f"c{i}", "recall/pool.py", "", {"file": a["file"], "ord": i, "code_anchor": a})
            for i, a in enumerate(anchors)
        ]

    def record_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def supersession_all(self) -> tuple[dict[str, str], frozenset[str], dict[str, Any]]:
        return {}, frozenset(), {}


def test_code_anchor_chunks_are_code_records() -> None:
    """K8: the code join's lookup half (contract C5), before any code chunker writes anchors."""
    lease = {"repo": REPO, **BUG_001["anchors"][0]}
    other = {"repo": REPO, "file": "recall/pool.py", "symbol": "Pool.release", "code_sha": OTHER_SHA}
    orphan = {"repo": None, **BUG_001["anchors"][0]}  # built with no --project
    store = _CodeStore([lease, dict(lease), other, orphan])  # the lease symbol was split in two chunks
    found = lookup_records(store, repo=REPO, anchors=[_anchor(BUG_001)])
    by_chunk = {r["chunk_id"]: r for r in found["records"]}
    assert set(by_chunk) == {"c0", "c2"}, "a split symbol was returned once per chunk"
    assert by_chunk["c0"]["chunk_ids"] == ["c0", "c1"] and by_chunk["c2"]["chunk_ids"] == ["c2"]
    assert by_chunk["c0"]["kind"] == "code" and by_chunk["c0"]["anchor_match"] == "exact"
    assert by_chunk["c0"]["recall_record"] == lease and by_chunk["c0"]["id"] == "c0"
    assert by_chunk["c2"]["anchor_match"] == "file_only"
    assert lookup_records(store, repo="someone/else", anchors=[_anchor(BUG_001)])["records"] == []


@requires_db
def test_reads_active_generation_only(manager) -> None:  # noqa: F811  # the imported fixture
    """K9: a built but unpromoted generation's records are not served."""
    from recall.generation_store import GenerationStore

    active_name, ready_name = "RE-call__0123456789ab__BUG-001.md", "RE-call__abcdef012345__BUG-002.md"
    pipeline = _pipeline("model-a")
    active_data = (MEMOS / active_name).read_bytes()
    active_manifest = _manifest(manager.tenant_id, active_data)
    active = _ready(manager, active_manifest, pipeline, _reader(active_manifest, active_data), _Embedder(1))
    manager.promote(active, unsafe_development=True)
    ready_data = (MEMOS / ready_name).read_bytes()
    ready_manifest = _manifest(manager.tenant_id, ready_data, version="object-v2", corpus_version="corpus-v2")
    _ready(manager, ready_manifest, pipeline, _reader(ready_manifest, ready_data), _Embedder(2))

    with GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id) as store:
        result = lookup_records(store, all_records=True)
    assert result["generation_id"] == active
    assert [r["recall_record"] for r in result["records"]] == [EXPECTED[active_name]]


@requires_db
def test_postgres_and_lite_return_the_same_records(tmp_path: Path, cli_table: str) -> None:
    """K10: the same CLI sequence on both stores prints the same records."""
    folder = copy_memos(tmp_path / "memos")
    backends = {
        "postgres": ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table],
        "lite": ["--embedder", "hashing", "--dsn", lite_dsn(tmp_path / "memory.db")],
    }
    dumps: dict[str, list[dict[str, Any]]] = {}
    for name, base in backends.items():
        out = run_cli([*base, "index", str(folder)])
        assert "indexed" in out, out
        envelope = json.loads(run_cli([*base, "records", "--all", "--json"]))
        dumps[name] = envelope["records"]
    assert len(dumps["lite"]) == len(EXPECTED)
    assert dumps["postgres"] == dumps["lite"]
