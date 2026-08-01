"""Dating the supersession edges, so `known_as_of` can rewind supersession as well as hits.

Prior work searched 2026-08-01. `docs_search(source_type="memory")` was UNAVAILABLE (the memory
corpus is served from VPS2, which is down; the VPS3 mirror has no `docs_chunks` relation), so this
fell back to grep/Read over `~/.claude/.../memory` and the repo docs. Load-bearing:
`tests/test_bitemporal.py` and `evaluate`'s own docstring, which state the limit this closes:

    "`known_as_of` filters HITS by write time; it does not rewind supersession. Supersession
    edges carry no timestamp, so an edge added after the as-of instant still applies, and a
    memory that was current at that moment can read as `superseded` by a document the caller
    cannot see."

The edge A -> B becomes assertable the moment B is written, and B's `indexed_at` is already an
indexed column. So the timestamp the docstring says the corpus format "does not record" is in
fact derivable from data already stored, with no migration and no extraction from prose. The
framing came from a reader on the Part 4 thread: utterance time is the axis to ORDER on rather
than to filter on.

Every assertion here fails against the pre-change code: `resolve_successor` took no edge dates, so
an edge asserted after the as-of instant was applied unconditionally.

Deliberate asymmetry with hits, since a reader will notice it: a hit with NO `indexed_at` stays
visible, but an edge with NO date still APPLIES. Both are the fail-closed choice. Hiding a hit of
unknown age would silently empty result sets for stores predating the column; ignoring an edge of
unknown age would serve a memory the corpus explicitly marks as stale.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import psycopg
import pytest

from recall.embeddings import HashingEmbedder
from recall.store import PgVectorStore, resolve_edge_dates, resolve_supersession
from recall.trust import _verdict, resolve_successor
from recall.types import Chunk, ScoredChunk
from tests.conftest import TEST_DSN, requires_db

MON = datetime(2026, 3, 2, tzinfo=timezone.utc)
TUE = datetime(2026, 3, 3, tzinfo=timezone.utc)
WED = datetime(2026, 3, 4, tzinfo=timezone.utc)

DIM = 64


@pytest.fixture
def shared_table():
    name = "ed_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def _chunk(cid: str, file: str, text: str, supersedes: str | None = None) -> Chunk:
    meta: dict = {"file": file, "ord": 0}
    if supersedes:
        meta["supersedes"] = supersedes
    return Chunk(cid, file, text, meta)


def _hit(file: str = "a.md", *, indexed_at=MON):
    return ScoredChunk(
        chunk=Chunk(id="c1", source="s", text="t", metadata={"file": file}),
        score=0.9,
        indexed_at=indexed_at,
    )


def _v(*, supersession, edge_dates=None, known_as_of=None):
    return _verdict(
        _hit(), supersession, 0.5, WED, frozenset(), known_as_of, edge_dates
    )[0]


# --- the capability -----------------------------------------------------------------------

def test_an_edge_asserted_after_the_as_of_instant_does_not_apply():
    """The limit this closes. Replaying Tuesday, a revision written Wednesday must not yet have
    superseded anything, or the replay reports a fate decided after the moment being replayed."""
    verdict = _v(
        supersession={"a.md": "b.md"}, edge_dates={"a.md": WED}, known_as_of=TUE
    )
    assert verdict == "ok"


def test_an_edge_asserted_before_the_as_of_instant_still_applies():
    verdict = _v(
        supersession={"a.md": "b.md"}, edge_dates={"a.md": MON}, known_as_of=TUE
    )
    assert verdict == "superseded"


def test_an_edge_asserted_exactly_at_the_instant_applies():
    """Inclusive boundary, matching `known_as_of` on hits: written AT the instant existed then."""
    verdict = _v(
        supersession={"a.md": "b.md"}, edge_dates={"a.md": TUE}, known_as_of=TUE
    )
    assert verdict == "superseded"


# --- backward compatibility ---------------------------------------------------------------

def test_without_known_as_of_edge_dates_are_ignored():
    """Opt-in. A caller passing no as-of instant must be byte-identical to before the change."""
    verdict = _v(supersession={"a.md": "b.md"}, edge_dates={"a.md": WED})
    assert verdict == "superseded"


def test_an_edge_with_no_date_still_applies():
    """Fail closed, and the inverse of the rule for hits. See the module docstring."""
    verdict = _v(supersession={"a.md": "b.md"}, edge_dates={}, known_as_of=TUE)
    assert verdict == "superseded"


def test_no_edge_dates_at_all_behaves_as_before():
    verdict = _v(supersession={"a.md": "b.md"}, edge_dates=None, known_as_of=TUE)
    assert verdict == "superseded"


# --- chains, which is where a per-hit gate would be wrong ---------------------------------

def test_a_chain_resolves_to_the_successor_known_at_the_instant():
    """a -> b was asserted Monday, b -> c only on Wednesday. As of Tuesday the terminal successor
    is b, NOT c. Gating only the final verdict would answer c, a document that did not yet
    supersede anything."""
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "c.md"},
        edge_dates={"a.md": MON, "b.md": WED},
        known_as_of=TUE,
    )
    assert successor == "b.md"


def test_a_chain_fully_known_still_resolves_to_the_terminal_successor():
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "c.md"},
        edge_dates={"a.md": MON, "b.md": MON},
        known_as_of=TUE,
    )
    assert successor == "c.md"


def test_the_first_edge_being_too_new_hides_the_whole_chain():
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "c.md"},
        edge_dates={"a.md": WED, "b.md": WED},
        known_as_of=TUE,
    )
    assert successor is None


# --- the pre-existing contract must survive -----------------------------------------------

def test_cycle_still_does_not_hang_with_dates_present():
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "a.md"},
        edge_dates={"a.md": MON, "b.md": MON},
        known_as_of=TUE,
    )
    assert successor == "b.md"


def test_self_claim_still_ignored_with_dates_present():
    successor = resolve_successor(
        "a.md", {"a.md": "a.md"}, edge_dates={"a.md": MON}, known_as_of=TUE
    )
    assert successor is None


def test_unchanged_signature_still_works():
    """Every existing call site passes two positional arguments and must keep working."""
    assert resolve_successor("a.md", {"a.md": "b.md", "b.md": "c.md"}) == "c.md"
    assert resolve_successor("c.md", {"a.md": "b.md"}) is None


# --- the dating rule, DB-free ---------------------------------------------------------------
#
# `resolve_edge_dates` is pure for the same reason `resolve_supersession` is: the rule can be
# tested without a database. The rows it is fed by hand here are the SQL's output shape, so these
# check the RULE and not the query. The `requires_db` cases at the bottom check the query, and
# they are skipped wherever Postgres is unreachable, which is why both layers exist.

def test_an_edge_is_dated_by_the_superseding_document():
    """A -> B is assertable when B is written, because the claim lives in B's frontmatter."""
    rows = [("a.md", None, MON), ("b.md", "a.md", WED)]
    assert resolve_edge_dates(rows, {"a.md": "b.md"}) == {"a.md": WED}


def test_the_earliest_chunk_of_the_superseding_document_wins():
    """Any chunk of B existing implies its frontmatter existed, so the earliest is the date."""
    rows = [("b.md", "a.md", WED), ("b.md", "a.md", MON), ("b.md", "a.md", TUE)]
    assert resolve_edge_dates(rows, {"a.md": "b.md"}) == {"a.md": MON}


def test_an_edge_whose_superseding_file_has_no_date_is_absent():
    """Absent means `resolve_successor` applies it unconditionally, which is fail closed."""
    rows = [("b.md", "a.md", None)]
    assert resolve_edge_dates(rows, {"a.md": "b.md"}) == {}


def test_a_dangling_edge_target_is_absent_rather_than_guessed():
    """`resolve_supersession` keeps an edge keyed on a raw basename when nothing bears it. There
    is no document to date, so there is no date, and the edge keeps applying."""
    rows = [("b.md", "gone.md", MON)]
    assert resolve_edge_dates(rows, {"gone.md": "b.md"}) == {"gone.md": MON}
    assert resolve_edge_dates(rows, {"missing.md": "nowhere.md"}) == {}


def test_rows_with_no_file_are_skipped_not_crashed():
    rows = [(None, "a.md", MON), ("b.md", "a.md", TUE)]
    assert resolve_edge_dates(rows, {"a.md": "b.md"}) == {"a.md": TUE}


# --- regressions from the bug audit of this change ----------------------------------------

def test_only_claim_carrying_rows_date_an_edge():
    """Chunks of one file can disagree on `supersedes` (a direct `store.upsert` bypasses the
    Indexer's fail-fast). Dating from a chunk that does NOT assert the claim ran earlier than the
    claim, so the edge applied at instants before it was written."""
    rows = [("a.md", None, MON), ("b.md", None, MON), ("b.md", "a.md", WED)]
    assert resolve_edge_dates(rows, {"a.md": "b.md"}) == {"a.md": WED}
    assert (
        resolve_successor("a.md", {"a.md": "b.md"}, {"a.md": WED}, TUE) is None
    )


def test_a_naive_known_as_of_does_not_raise():
    """`evaluate` normalises a naive `now`; it must do the same for `known_as_of`, because both
    the hit comparison and the edge comparison put it against a tz-aware store value."""
    from recall.calibration import Calibration
    from recall.trust import evaluate
    from recall.types import RetrievalResult, StalenessReport

    result = RetrievalResult(
        query="q",
        hits=[_hit()],
        gap_warning=False,
        staleness=StalenessReport(
            stale=False, newest_indexed_at=None, age=None, max_age=None
        ),
    )
    res = evaluate(
        result,
        {"a.md": "b.md"},
        Calibration(embedder="test", threshold=0.5, scale=0.05),
        WED.replace(tzinfo=None),
        frozenset(),
        TUE.replace(tzinfo=None),  # naive: raised TypeError before the fix
        {"a.md": MON},
    )
    assert res.hits[0].verdict == "superseded"


@requires_db
@pytest.mark.xfail(
    strict=True,
    reason="OPEN DEFECT: indexed_at is the LAST write, so re-indexing a superseding document "
    "re-dates its edge. Needs a first_indexed_at column preserved across upserts. "
    "strict=True means this FAILS the suite the day it starts passing, which is the point.",
)
def test_reindexing_the_superseding_file_must_not_re_date_its_edge(shared_table):
    """The deadman for the defect that blocks this branch, asserting the CORRECT expectation.

    An earlier version of this test lived in the pure-function layer and asserted the WRONG
    answer to "pin" the bug. It could not fire: its rows were byte-identical to the happy-path
    test above, its assertions were correct for those rows, and it went on passing in the
    post-fix world. The defect is in the WRITE path's timestamping, so nothing below the store
    can see it. It has to be tested here or not at all.

    Re-indexing b.md must not move the date of the edge it asserts, because the claim was made at
    the first write and has been continuously true since.
    """
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        store.upsert([_chunk("v1", "limits_v1.md", "the rate limit is 100 rps")],
                     emb.embed(["the rate limit is 100 rps"]))
        store.upsert(
            [_chunk("v2", "limits_v2.md", "now 250 rps", "limits_v1.md")],
            emb.embed(["now 250 rps"]),
        )
        first = store.supersession_all()[2]["limits_v1.md"]

        # b.md edited and re-indexed. Its claim on a.md is unchanged and was never withdrawn.
        store.upsert(
            [_chunk("v2", "limits_v2.md", "now 250 rps (typo fixed)", "limits_v1.md")],
            emb.embed(["now 250 rps (typo fixed)"]),
        )
        after = store.supersession_all()[2]["limits_v1.md"]

        assert after == first, (
            "re-indexing the superseding document moved its edge date forward, so a replay "
            "before the re-index now drops an edge that existed then"
        )
    finally:
        store.close()


# --- regressions from the SECOND adversarial pass, over the fixes above --------------------

def test_two_divergent_claims_from_one_file_are_dated_separately():
    """Dating per FILE let an earlier claim date a later one. One file can carry several
    `supersedes` values without any authoring mistake: a corpus indexed under two roots shares
    one root-relative `file` key while `replace_sources` deletes by absolute `source`."""
    rows = [
        ("a.md", None, MON), ("c.md", None, MON),
        ("b.md", "a.md", MON), ("b.md", "c.md", WED),
    ]
    edges = {"a.md": "b.md", "c.md": "b.md"}
    assert resolve_edge_dates(rows, edges) == {"a.md": MON, "c.md": WED}
    # The claim on c.md was written WED, so as of TUE it had not been made.
    assert resolve_successor("c.md", edges, resolve_edge_dates(rows, edges), TUE) is None
    assert resolve_successor("a.md", edges, resolve_edge_dates(rows, edges), TUE) == "b.md"


def test_an_empty_supersedes_does_not_date_an_edge_it_did_not_create():
    """NOT A GUARD, and labelled so rather than left to read as one.

    A bare `supersedes:` parses to "" and builds no edge, and dating from it under an `is None`
    predicate used to re-enter the per-file bug through a second door. Keying per CLAIM closed
    that door independently: the empty row now keys `(b.md, "")`, which can never collide with
    `(b.md, "a.md")`. Verified by mutation: swapping the predicate back to `is None` leaves this
    test and the whole suite green.

    So the predicate alignment with `resolve_supersession` is defence in depth, not a fix with a
    failing case behind it, and this test documents the subsumption instead of pretending to
    protect it. Two functions consuming the same rows should agree on what a claim is regardless.
    """
    rows = [("a.md", None, MON), ("b.md", "", MON), ("b.md", "a.md", WED)]
    edges, _ = resolve_supersession([(r[0], r[1]) for r in rows])
    assert edges == {"a.md": "b.md"}
    assert resolve_edge_dates(rows, edges) == {"a.md": WED}


def test_a_claim_by_basename_dates_an_edge_between_nested_paths():
    """Edge keys are root-relative paths while the claim names a basename, which is how the edge
    was matched. The date lookup has to resolve it the same way or it silently finds nothing."""
    rows = [("x/a.md", None, MON), ("y/b.md", "a.md", WED)]
    assert resolve_edge_dates(rows, {"x/a.md": "y/b.md"}) == {"x/a.md": WED}


def test_naive_edge_dates_are_normalised_alongside_known_as_of():
    """Normalising only `known_as_of` made a self-consistently naive caller START raising, on the
    documented case of a hit with no `indexed_at`. Both operands or neither."""
    from recall.calibration import Calibration
    from recall.trust import evaluate
    from recall.types import RetrievalResult, StalenessReport

    naive_hit = ScoredChunk(
        chunk=Chunk(id="c", source="s", text="t", metadata={"file": "a.md"}),
        score=0.9,
        indexed_at=None,
    )
    result = RetrievalResult(
        query="q", hits=[naive_hit], gap_warning=False,
        staleness=StalenessReport(stale=False, newest_indexed_at=None, age=None, max_age=None),
    )
    res = evaluate(
        result,
        {"a.md": "b.md"},
        Calibration(embedder="test", threshold=0.5, scale=0.05),
        WED.replace(tzinfo=None),
        frozenset(),
        TUE.replace(tzinfo=None),
        {"a.md": MON.replace(tzinfo=None)},  # naive throughout: raised TypeError after the fix
    )
    assert res.hits[0].verdict == "superseded"


class _EdgeDatedStore:
    """Store exposing `supersession_all`, counting how it was consulted."""

    def __init__(self):
        self.all_calls = 0
        self.split_calls = 0
        self._hit = ScoredChunk(
            chunk=Chunk(id="1", source="/c/a.md", text="body", metadata={"file": "a.md"}),
            score=0.99,
            indexed_at=MON,
        )

    def query_dense(self, vector, k, source=None):
        return [self._hit]

    def query_sparse(self, query, k, source=None, vec=None):
        return []

    def newest_indexed_at(self):
        return MON

    def supersession(self):
        self.split_calls += 1
        return {"a.md": "b.md"}, frozenset()

    def supersession_all(self):
        self.all_calls += 1
        return {"a.md": "b.md"}, frozenset(), {"a.md": WED}


class _NoEdgeDatesStore(_EdgeDatedStore):
    """The pre-change read surface: `supersession()` only. Duck-typed stores like this exist."""

    supersession_all = None  # type: ignore[assignment]

    def __getattribute__(self, name):
        if name == "supersession_all":
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class _ConstantEmbedder:
    dim = 2
    name = "constant"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_trusted_search_reads_edges_and_dates_in_one_call():
    """Two accessors meant two cache validations, so a concurrent index could hand back edges
    from one scan dated by the next. Reverting to the split form left the whole suite green."""
    from recall.calibration import Calibration
    from recall.trust import trusted_search

    store = _EdgeDatedStore()
    res = trusted_search(
        store, _ConstantEmbedder(), "q", k=1,
        calibration=Calibration(embedder="constant", threshold=0.5, scale=0.05),
        known_as_of=TUE,
    )
    assert store.all_calls == 1
    assert store.split_calls == 0
    # WED edge, asked as of TUE: not yet asserted, so the hit is not superseded.
    assert res.hits[0].verdict == "ok"


def test_trusted_search_degrades_without_supersession_all_and_says_so(caplog):
    """The `getattr` fallback traded an AttributeError for a silent HALF answer. It must warn:
    hits are rewound, edges are not, and the caller asked about a past instant."""
    from recall.calibration import Calibration
    from recall.trust import _WARNED_NO_EDGE_DATES, trusted_search

    _WARNED_NO_EDGE_DATES.clear()
    store = _NoEdgeDatesStore()
    with caplog.at_level("WARNING"):
        res = trusted_search(
            store, _ConstantEmbedder(), "q", k=1,
            calibration=Calibration(embedder="constant", threshold=0.5, scale=0.05),
            known_as_of=TUE,
        )
    assert store.split_calls == 1
    assert res.hits[0].verdict == "superseded"  # undated edge applies: fail closed
    assert any("supersession_all" in r.message for r in caplog.records)


def test_no_warning_when_the_caller_never_asks_for_a_past_instant():
    """The degradation only matters to a point-in-time query. Everyone else must see nothing."""
    from recall.calibration import Calibration
    from recall.trust import _WARNED_NO_EDGE_DATES, trusted_search

    _WARNED_NO_EDGE_DATES.clear()
    store = _NoEdgeDatesStore()
    trusted_search(
        store, _ConstantEmbedder(), "q", k=1,
        calibration=Calibration(embedder="constant", threshold=0.5, scale=0.05),
    )
    assert _WARNED_NO_EDGE_DATES == set()


@requires_db
def test_supersession_all_hands_out_copies_not_the_live_cache(shared_table):
    """It is public now, and the cache is process-wide and validated rather than rebuilt, so a
    caller's mutation would survive every later cache hit and redirect other callers' verdicts."""
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        store.upsert([_chunk("v1", "limits_v1.md", "100 rps")], emb.embed(["100 rps"]))
        store.upsert(
            [_chunk("v2", "limits_v2.md", "250 rps", "limits_v1.md")], emb.embed(["250 rps"])
        )
        edges, _unresolved, dates = store.supersession_all()
        edges["limits_v1.md"] = "ATTACKER.md"
        dates.clear()

        assert store.supersession_all()[0] == {"limits_v1.md": "limits_v2.md"}
        assert store.supersession()[0] == {"limits_v1.md": "limits_v2.md"}
        assert store.supersession_all()[2] != {}
    finally:
        store.close()
