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
- K10 PostgreSQL and lite return the same records for the same memos (DB suite, through the CLI),
  and so does a generation built and promoted from that memo folder (`GenerationStore`, the store
  production serves), up to the fields a store names its own way: `source`, `chunk_id` and
  `chunk_ids` (CCA DAT-004 of the PR 977 deferred pass).
- K11 the source policy applies before edges are derived: a denied record is not returned, is not
  named in another record's `caused`, and a denied successor is `<withheld>`, never None (None
  would present a retracted record as current).
- K12 `superseded_by` follows #970's window rule through the store's supersession graph: a
  successor hides a record only inside its own validity window (the trust gate's reading).
- K13 a lookup reads only the code anchors it can match: an anchor query the asked files', a
  `kinds` without `code` none, and the lite and PostgreSQL reads select the same rows for each.
- K14 `expired` is computed at `now`, a `valid_until` date lasting through that day in UTC, the
  same through the store and through the memo-directory reader (NUM-001, owner 2026-10-27).
- K15 a `valid_until` that does not read as a date is marked expired, never current.

Red proof, recorded 2026-10-21 (K12 2026-10-22) on the deployment host inside the test bounds, each failing in its
own assertion, then restored and green:
- K2 `...::test_multi_anchor_reports_each`: `_best` mutated from `min` to `max` over the match
  order: PERF-001~2 is no longer returned for its exact first anchor (the membership assertion).
- K2b `...::test_best_match_is_listed_first`: the sort mutated to put the worst match first.
- K6 `...::test_ids_and_all` (rewritten 2026-10-29 when `offset` was removed, CCA CODE-007):
  the `all_records` cut mutated to `records[:end]` with `truncated = False`: a cut dump
  reported nothing left ("a cut all_records dump did not say it was cut").
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
- K12 `...::test_a_successor_hides_a_record_only_inside_its_own_window`: `superseded_by` read
  from the raw edge (`graph.edges.get(...)`, the pre-#970 reading) instead of
  `graph.successor(..., now=...)`: "a successor hid a record before its window".
- K13 `...::test_a_lookup_reads_only_the_code_anchors_it_can_match` (CCA run 20261026T220018Z (PR 977 audit), 2026-10-26 on the deployment host inside the test bounds): on the pre-fix
  `lookup_records`, which never narrowed: `[None, None, None]`; mutation M2 on the fix (the
  computed `code_files` overwritten with None) gives the same. Mutation M1 on the fix
  (`record_rows_condition` ignoring `code_files`): "an anchor query read another file's code
  anchors", here and in `...::test_the_postgres_reads_narrow_the_code_anchors_alike`, whose
  pre-fix run is no proof (`_record_chunks` took no `code_files`: a TypeError).
- K14 and K15 (NUM-001, 2026-10-27, on the deployment host inside the test bounds):
  `...::test_expired_is_computed_at_now_through_the_last_day` on the pre-change lookup, which had
  no `expired`: "store: at 2026-09-01T23:59:00+00:00 a record valid until 2026-09-01 read
  expired=None, expected False"; with `_expired` mutated to CCA's date-only `<=`: the same
  instant read "expired=True, expected False". `...::test_an_unreadable_valid_until_is_expired` on
  the pre-change lookup and with the unreadable branch mutated to return False: "an unreadable
  valid_until read as current".
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
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
    _S3,
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
#: An instant after every golden memo's `valid_from`, so every successor is inside its window.
AFTER_ALL = datetime(2026, 11, 1, tzinfo=UTC)


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
    found = _by_id(lookup_records(store, repo=REPO, anchors=[_anchor(FP, 0), _anchor(FP, 1)], now=AFTER_ALL))
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
    order = [r["id"] for r in everything["records"]]
    cut = lookup_records(store, all_records=True, limit=4)
    assert [r["id"] for r in cut["records"]] == order[:4]
    assert cut["truncated"] is True, "a cut all_records dump did not say it was cut"
    whole = lookup_records(store, all_records=True, limit=len(order))
    assert [r["id"] for r in whole["records"]] == order and whole["truncated"] is False


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
    found = _by_id(
        lookup_records(store, all_records=True, security_policy=policy, access_context=object(), now=AFTER_ALL)
    )
    assert BUG_002["id"] not in found, "a denied record was returned"
    assert found[BUG_001["id"]]["caused"] == [], "a denied record was named through `caused`"
    assert found[P2_4["id"]]["superseded_by"] == WITHHELD
    assert found[FP["id"]]["superseded_by"] == WITHHELD, "a retracted record read as current"
    assert found[LESSON["id"]]["superseded_by"] == "RE-call__lesson__6a3ae35062b9~retract.md"


def test_a_successor_hides_a_record_only_inside_its_own_window(store: LiteStore) -> None:
    """K12: #970's rule (owner decision 2026-10-21), read the same way the trust gate reads it.

    The lesson's retraction is valid from 2026-10-23. Before that instant the retraction is not
    yet in force, so the lesson is current; from it on, the lesson is superseded.
    """
    retraction = "RE-call__lesson__6a3ae35062b9~retract.md"
    before = _by_id(lookup_records(store, ids=[LESSON["id"]], now=datetime(2026, 10, 22, tzinfo=UTC)))
    on = _by_id(lookup_records(store, ids=[LESSON["id"]], now=datetime(2026, 10, 23, 12, tzinfo=UTC)))
    assert before[LESSON["id"]]["superseded_by"] is None, "a successor hid a record before its window"
    assert on[LESSON["id"]]["superseded_by"] == retraction


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

    def record_chunks(self, code_files: list[str] | None = None) -> list[Chunk]:
        return [c for c in self.chunks if code_files is None or c.metadata["code_anchor"]["file"] in code_files]

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
def test_postgres_and_lite_return_the_same_records(tmp_path: Path, cli_table: str, manager) -> None:  # noqa: F811  # the imported fixture
    """K10: the same CLI sequence on both stores prints the same records, and a generation built
    from the same memos returns them too."""
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

    # The generation leg: every memo of the folder as one object of a manifest under one prefix,
    # so `manifest_relative_paths` gives each its bare name, the `file` lite records too.
    from recall.generation_store import GenerationStore
    from recall.lineage import IndexManifestV1, ManifestObjectV1
    from recall.manifest import S3Allowlist, S3ObjectReader

    tenant = manager.tenant_id
    objects: dict[tuple[str, str, str], bytes] = {}
    entries = []
    for memo in sorted(folder.glob("*.md")):
        data = memo.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        objects[("approved", f"corpora/{tenant}/{memo.name}", digest)] = data
        entries.append(
            ManifestObjectV1(f"s3://approved/corpora/{tenant}/{memo.name}", digest, "text/markdown", len(data), digest)
        )
    manifest = IndexManifestV1(tenant, "corpus-v1", tuple(entries))
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    generation = _ready(manager, manifest, _pipeline("model-a"), reader, _Embedder(1))
    manager.promote(generation, unsafe_development=True)
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        served = json.loads(json.dumps(lookup_records(store, all_records=True, limit=None)["records"]))

    def comparable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # A store names its sources and chunks its own way, and this build chunks each memo whole.
        return [
            {**{k: v for k, v in r.items() if k not in ("chunk_id", "chunk_ids")}, "source": memo_name(r["source"])}
            for r in records
        ]

    assert comparable(served) == comparable(dumps["lite"]), "a generation store returned other records than lite"


def test_a_lookup_reads_only_the_code_anchors_it_can_match(tmp_path: Path) -> None:
    """K13: every chunk of a code tenant carries a `code_anchor`, so a lookup read the whole tenant
    whatever it asked. An anchor query reads the asked files' anchors, a `kinds` without `code`
    reads none, and every store's read selects the same rows for each."""
    asked: list[list[str] | None] = []

    class _Spy(_CodeStore):
        def record_chunks(self, code_files: list[str] | None = None) -> list[Chunk]:
            asked.append(None if code_files is None else sorted(code_files))
            return super().record_chunks(code_files)

    lease = {"repo": REPO, **BUG_001["anchors"][0]}
    spy = _Spy([lease])
    lookup_records(spy, repo=REPO, anchors=[_anchor(BUG_001), _anchor(BUG_001, symbol="other")])
    lookup_records(spy, ids=["c0"], kinds=["bug"])
    lookup_records(spy, all_records=True)
    assert asked == [[lease["file"]], [], None], asked

    store = LiteStore(tmp_path / "memory.db", dim=2)
    store.upsert(
        [
            Chunk("record", "memo.md", "a memo", {"recall_record": {"id": "x"}}),
            Chunk("a", "a.py", "def a(): ...", {"code_anchor": {"file": "a.py"}}),
            Chunk("b", "b.py", "def b(): ...", {"code_anchor": {"file": "b.py"}}),
            Chunk("plain", "notes.md", "nothing", {}),
        ],
        [[1.0, 0.0]] * 4,
    )

    def read(code_files: list[str] | None) -> list[str]:
        return sorted(chunk.id for chunk in store.record_chunks(code_files=code_files))

    assert read(None) == ["a", "b", "record"]
    assert read(["a.py"]) == ["a", "record"], "an anchor query read another file's code anchors"
    assert read([]) == ["record"], "a lookup that cannot match code read the code anchors"


@requires_db
def test_the_postgres_reads_narrow_the_code_anchors_alike(cli_table: str) -> None:
    """K13 on the legacy PostgreSQL read, which `recall.records` issues itself."""
    from recall.records import _record_chunks
    from recall.store import PgVectorStore

    with PgVectorStore(TEST_DSN, dim=64, table=cli_table, tenant="k13") as store:
        store.upsert(
            [
                Chunk("record", "memo.md", "a memo", {"recall_record": {"id": "x"}}),
                Chunk("a", "a.py", "def a(): ...", {"code_anchor": {"file": "a.py"}}),
                Chunk("b", "b.py", "def b(): ...", {"code_anchor": {"file": "b.py"}}),
                Chunk("plain", "notes.md", "nothing", {}),
            ],
            [[1.0] + [0.0] * 63] * 4,
        )

        def read(code_files: list[str] | None) -> list[str]:
            return sorted(chunk.id for chunk in _record_chunks(store, code_files=code_files))

        assert read(None) == ["a", "b", "record"]
        assert read(["a.py"]) == ["a", "record"], "an anchor query read another file's code anchors"
        assert read([]) == ["record"], "a lookup that cannot match code read the code anchors"


def test_expired_is_computed_at_now_through_the_last_day(tmp_path: Path) -> None:
    """K14 (NUM-001, owner decision 2026-10-27): `expired` is computed at `now` by the rule the
    trust gate reads `valid_until` with, a date lasting through that day in UTC, so no consumer
    redoes the comparison. Through the store and through the memo-directory reader alike."""
    from recall.records_files import load_memo_records, match_memo_records

    folder = copy_memos(tmp_path / "memos")
    dated = folder / "RE-call__0123456789ab__BUG-001.md"
    dated.write_bytes(dated.read_bytes().replace(b"valid_from: 2026-08-04\n", b"valid_from: 2026-08-04\nvalid_until: 2026-09-01\n"))
    store = indexed_lite_store(tmp_path, memos=folder)
    memos = load_memo_records(folder)
    last_minute = datetime(2026, 9, 1, 23, 59, tzinfo=UTC)
    next_day = datetime(2026, 9, 2, 0, 0, 1, tzinfo=UTC)
    for when, want in ((last_minute, False), (next_day, True)):
        through_store = _by_id(lookup_records(store, repo=REPO, anchors=[_anchor(BUG_001)], now=when))
        from_files = {r["id"]: r for r in match_memo_records(memos, repo=REPO, anchors=[_anchor(BUG_001)], kinds=["bug"], now=when)}
        for name, found in (("store", through_store), ("memo directory", from_files)):
            assert found[BUG_001["id"]].get("expired") is want, (
                f"{name}: at {when.isoformat()} a record valid until 2026-09-01 read expired="
                f"{found[BUG_001['id']].get('expired')!r}, expected {want}"
            )
    undated = _by_id(lookup_records(store, ids=[BUG_002["id"]], now=AFTER_ALL))
    assert undated[BUG_002["id"]].get("expired") is False, "a record with no valid_until read as expired"


def test_an_unreadable_valid_until_is_expired() -> None:
    """K15: a `valid_until` that does not read as a date is not proof the record is current, so it
    is marked expired (the indexer refuses such a memo; this pins the lookup's own answer)."""

    class _Store:
        tenant = "t"

        def record_chunks(self, code_files: list[str] | None = None) -> list[Chunk]:
            return [Chunk("c", "m.md", "", {"recall_record": BUG_002, "file": "m.md", "valid_until": "soon"})]

        def supersession_all(self) -> tuple[dict[str, str], frozenset[str], dict[str, Any]]:
            return {}, frozenset(), {}

    found = _by_id(lookup_records(_Store(), ids=[BUG_002["id"]], now=AFTER_ALL))
    assert found[BUG_002["id"]].get("expired") is True, "an unreadable valid_until read as current"
