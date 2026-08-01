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
from recall.store import (
    PgVectorStore,
    resolve_supersession,
    resolve_supersession_candidates,
)
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


def _v(*, supersession, edge_candidates=None, known_as_of=None):
    return _verdict(
        _hit(), supersession, 0.5, WED, frozenset(), known_as_of, edge_candidates
    )[0]


def _cands(mapping):
    """``{superseded: (superseding, when)}`` -> the candidate-list shape.

    A convenience for the single-claim cases, where fan-in is not what is under test. The fan-in
    cases build the lists explicitly, because there the list IS the subject.
    """
    return {target: [claim] for target, claim in mapping.items()}


# --- the capability -----------------------------------------------------------------------

def test_an_edge_asserted_after_the_as_of_instant_does_not_apply():
    """The limit this closes. Replaying Tuesday, a revision written Wednesday must not yet have
    superseded anything, or the replay reports a fate decided after the moment being replayed."""
    verdict = _v(
        supersession={"a.md": "b.md"}, edge_candidates=_cands({"a.md": ("b.md", WED)}), known_as_of=TUE
    )
    assert verdict == "ok"


def test_an_edge_asserted_before_the_as_of_instant_still_applies():
    verdict = _v(
        supersession={"a.md": "b.md"}, edge_candidates=_cands({"a.md": ("b.md", MON)}), known_as_of=TUE
    )
    assert verdict == "superseded"


def test_an_edge_asserted_exactly_at_the_instant_applies():
    """Inclusive boundary, matching `known_as_of` on hits: written AT the instant existed then."""
    verdict = _v(
        supersession={"a.md": "b.md"}, edge_candidates=_cands({"a.md": ("b.md", TUE)}), known_as_of=TUE
    )
    assert verdict == "superseded"


# --- backward compatibility ---------------------------------------------------------------

def test_without_known_as_of_edge_dates_are_ignored():
    """Opt-in. A caller passing no as-of instant must be byte-identical to before the change."""
    verdict = _v(supersession={"a.md": "b.md"}, edge_candidates=_cands({"a.md": ("b.md", WED)}))
    assert verdict == "superseded"


def test_a_target_absent_from_the_candidate_map_keeps_demoting():
    """NOT the undated-edge rule, which has its own test below. An ABSENT target means the two
    maps disagree, i.e. hand-built inconsistent input, and the fail-closed answer is to keep
    demoting. A target PRESENT with an empty list means no claim, and resolves to nothing."""
    verdict = _v(supersession={"a.md": "b.md"}, edge_candidates={}, known_as_of=TUE)
    assert verdict == "superseded"


def test_no_edge_dates_at_all_behaves_as_before():
    verdict = _v(supersession={"a.md": "b.md"}, edge_candidates=None, known_as_of=TUE)
    assert verdict == "superseded"


# --- chains, which is where a per-hit gate would be wrong ---------------------------------

def test_a_chain_resolves_to_the_successor_known_at_the_instant():
    """a -> b was asserted Monday, b -> c only on Wednesday. As of Tuesday the terminal successor
    is b, NOT c. Gating only the final verdict would answer c, a document that did not yet
    supersede anything."""
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "c.md"},
        edge_candidates=_cands({"a.md": ("b.md", MON), "b.md": ("c.md", WED)}),
        known_as_of=TUE,
    )
    assert successor == "b.md"


def test_a_chain_fully_known_still_resolves_to_the_terminal_successor():
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "c.md"},
        edge_candidates=_cands({"a.md": ("b.md", MON), "b.md": ("c.md", MON)}),
        known_as_of=TUE,
    )
    assert successor == "c.md"


def test_the_first_edge_being_too_new_hides_the_whole_chain():
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "c.md"},
        edge_candidates=_cands({"a.md": ("b.md", WED), "b.md": ("c.md", WED)}),
        known_as_of=TUE,
    )
    assert successor is None


# --- the pre-existing contract must survive -----------------------------------------------

def test_cycle_still_does_not_hang_with_dates_present():
    successor = resolve_successor(
        "a.md",
        {"a.md": "b.md", "b.md": "a.md"},
        edge_candidates=_cands({"a.md": ("b.md", MON), "b.md": ("a.md", MON)}),
        known_as_of=TUE,
    )
    assert successor == "b.md"


def test_self_claim_still_ignored_with_dates_present():
    successor = resolve_successor(
        "a.md", {"a.md": "a.md"}, edge_candidates=_cands({"a.md": ("a.md", MON)}), known_as_of=TUE
    )
    assert successor is None


def test_unchanged_signature_still_works():
    """Every existing call site passes two positional arguments and must keep working."""
    assert resolve_successor("a.md", {"a.md": "b.md", "b.md": "c.md"}) == "c.md"
    assert resolve_successor("c.md", {"a.md": "b.md"}) is None


# --- the dating rule, DB-free ---------------------------------------------------------------
#
# `resolve_supersession_candidates` is pure for the same reason `resolve_supersession` is: the
# rule can be tested without a database. The rows fed by hand here are the SQL's output shape, so
# these check the RULE and not the query. The `requires_db` cases at the bottom check the query.
#
# They assert on the CANDIDATE list rather than on a winner's date. That is the point of the
# fan-in fix: keeping one winner per target threw away the information a replay needs.

def _claims(rows):
    """Just the candidate map, for tests that are only about dating."""
    return resolve_supersession_candidates(rows)[2]


def test_an_edge_is_dated_by_the_superseding_document():
    """A -> B is assertable when B is written, because the claim lives in B's frontmatter."""
    rows = [("a.md", None, MON), ("b.md", "a.md", WED)]
    assert _claims(rows) == {"a.md": [("b.md", WED)]}


def test_the_earliest_chunk_of_the_superseding_document_wins():
    """Any chunk of B existing implies its frontmatter existed, so the earliest is the date."""
    rows = [("b.md", "a.md", WED), ("b.md", "a.md", MON), ("b.md", "a.md", TUE)]
    assert _claims(rows) == {"a.md": [("b.md", MON)]}


def test_a_claim_with_no_date_is_undated_and_therefore_always_live():
    """Fail closed: unknown age keeps demoting rather than reviving a memory marked stale."""
    rows = [("b.md", "a.md", None)]
    assert _claims(rows) == {"a.md": [("b.md", None)]}
    assert resolve_successor("a.md", {"a.md": "b.md"}, _claims(rows), TUE) == "b.md"


def test_one_undated_row_makes_the_whole_claim_undated():
    """Mixing a dated and an undated row for the same claim must not let the date win: we do not
    know the claim was absent before it, so it has to keep applying."""
    rows = [("b.md", "a.md", WED), ("b.md", "a.md", None)]
    assert _claims(rows) == {"a.md": [("b.md", None)]}


def test_a_dangling_edge_target_is_still_dated():
    """`resolve_supersession` keeps an edge keyed on a raw basename when nothing bears it."""
    rows = [("b.md", "gone.md", MON)]
    assert _claims(rows) == {"gone.md": [("b.md", MON)]}


def test_rows_with_no_file_are_skipped_not_crashed():
    rows = [(None, "a.md", MON), ("b.md", "a.md", TUE)]
    assert _claims(rows) == {"a.md": [("b.md", TUE)]}


# --- regressions from the bug audit of this change ----------------------------------------

def test_only_claim_carrying_rows_date_an_edge():
    """Chunks of one file can disagree on `supersedes` (a direct `store.upsert` bypasses the
    Indexer's fail-fast). Dating from a chunk that does NOT assert the claim ran earlier than the
    claim, so the edge applied at instants before it was written."""
    rows = [("a.md", None, MON), ("b.md", None, MON), ("b.md", "a.md", WED)]
    assert _claims(rows) == {"a.md": [("b.md", WED)]}
    assert resolve_successor("a.md", {"a.md": "b.md"}, _claims(rows), TUE) is None


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
        _cands({"a.md": ("b.md", MON)}),
    )
    assert res.hits[0].verdict == "superseded"


@requires_db
def test_reindexing_the_superseding_file_must_not_re_date_its_edge(shared_table):
    """Was the deadman for the blocking defect; now the test that it stays fixed.

    It carried `xfail(strict=True, raises=AssertionError)` while `indexed_at` was the only write
    time available. `first_indexed_at` closed that, so the xfail is gone: leaving it would have
    turned an unexpected PASS into a suite failure, which is exactly what strict is for.

    An earlier version of this lived in the pure-function layer and asserted the WRONG answer to
    "pin" the bug. It could not fire: its rows were byte-identical to the happy-path test above,
    its assertions were correct for those rows, and it went on passing in the post-fix world. The
    defect is in the WRITE path's timestamping, so nothing below the store can see it. It has to
    be tested here or not at all.

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
    # DERIVED, not hand-built. The previous version supplied `edges` by hand, which is exactly
    # why it could not see that the same row shape makes b.md self-ambiguous one row away.
    edges, unresolved, claims = resolve_supersession_candidates(rows)
    assert edges == {"a.md": "b.md", "c.md": "b.md"}
    assert unresolved == frozenset(), "one file with two claims is not an ambiguous basename"
    assert claims == {"a.md": [("b.md", MON)], "c.md": [("b.md", WED)]}
    # The claim on c.md was written WED, so as of TUE it had not been made.
    assert resolve_successor("c.md", edges, claims, TUE) is None
    assert resolve_successor("a.md", edges, claims, TUE) == "b.md"


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
    edges, _unresolved, claims = resolve_supersession_candidates(rows)
    assert edges == {"a.md": "b.md"}
    assert claims == {"a.md": [("b.md", WED)]}


def test_a_claim_by_basename_dates_an_edge_between_nested_paths():
    """Edge keys are root-relative paths while the claim names a basename, which is how the edge
    was matched. The date lookup has to resolve it the same way or it silently finds nothing."""
    rows = [("x/a.md", None, MON), ("y/b.md", "a.md", WED)]
    assert _claims(rows) == {"x/a.md": [("y/b.md", WED)]}


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
        _cands({"a.md": ("b.md", MON.replace(tzinfo=None))}),  # naive: raised TypeError
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
        return {"a.md": "b.md"}, frozenset(), {"a.md": [("b.md", WED)]}


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


@pytest.fixture(autouse=True)
def _isolate_edge_date_warning():
    """`_WARNED_NO_EDGE_DATES` is process-global. Clearing it inline on entry left it populated
    for whatever ran next, so a `-k` selection or an xdist shard could see a silence produced by
    another test. Mirrors `test_uncalibrated_warning._isolated_calibration`: clear BOTH sides."""
    from recall.trust import _WARNED_NO_EDGE_DATES

    _WARNED_NO_EDGE_DATES.clear()
    yield
    _WARNED_NO_EDGE_DATES.clear()


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
    from recall.trust import trusted_search

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
        edges, _unresolved, claims = store.supersession_all()
        edges["limits_v1.md"] = "ATTACKER.md"
        # The LIST too, not just the top-level dict: a shallow `dict()` copy would share the
        # lists, so mutating one poisons the process-wide cache and the dict-only assertions
        # below would still pass.
        claims["limits_v1.md"].append(("ATTACKER.md", None))
        claims.clear()

        assert store.supersession_all()[0] == {"limits_v1.md": "limits_v2.md"}
        assert store.supersession()[0] == {"limits_v1.md": "limits_v2.md"}
        fresh = store.supersession_all()[2]
        assert fresh != {}
        assert all(
            f != "ATTACKER.md" for claimants in fresh.values() for f, _when in claimants
        ), "a caller's append reached the shared cache"
    finally:
        store.close()


# --- round three -----------------------------------------------------------------------------

def test_a_file_with_two_claims_is_not_an_ambiguous_target():
    """`rows` carries one entry per (file, supersedes) pair, so a file asserting two claims
    appeared twice and made ITSELF read as an ambiguous basename: its own incoming edge was
    dropped and it was named in `unresolved`, so the trust layer abstained and told the operator
    to disambiguate a basename that exactly one document carries. Ambiguity is two FILES sharing
    a stem, never one file carrying two claims."""
    rows = [
        ("a.md", None, MON), ("c.md", None, MON),
        ("b.md", "a.md", MON), ("b.md", "c.md", WED),
        ("d.md", "b.md", WED),          # a third document supersedes the two-claim file
    ]
    edges, unresolved = resolve_supersession([(r[0], r[1]) for r in rows])
    assert unresolved == frozenset()
    assert edges == {"a.md": "b.md", "c.md": "b.md", "b.md": "d.md"}


def test_a_bracketed_dangling_claim_is_still_dated():
    """`resolve_supersession` keys a RESOLVED edge on the target's path (matched through
    `_basename`) and a DANGLING one on the raw `rsplit`. Those differ when the claim carries both
    a path and brackets, so indexing only the normalised form left every bracketed dangling edge
    undated, which the per-file keying this replaced happened to get right."""
    rows = [("b.md", "[[dir/gone]]", WED)]
    edges, _unresolved, claims = resolve_supersession_candidates(rows)
    assert claims == {target: [("b.md", WED)] for target in edges}


# --- fan-in: the defect this used to only pin ------------------------------------------------
#
# b1.md (Monday) and b2.md (Wednesday) both supersede a.md. Keeping ONE winner per target picked
# b2.md lexicographically, so a replay of Tuesday saw a single edge dated Wednesday, dropped it,
# and answered `ok` where a.md was in fact superseded by b1.md. Renaming the two files flipped the
# answer, which is what showed the rule was keyed on the wrong axis. Now every claim is kept and
# the replay chooses the one that was live.

FAN_IN_ROWS = [("a.md", None, MON), ("b1.md", "a.md", MON), ("b2.md", "a.md", WED)]


def test_fan_in_keeps_every_claim_not_just_the_winner():
    edges, _unresolved, claims = resolve_supersession_candidates(FAN_IN_ROWS)
    assert edges == {"a.md": "b2.md"}, "the winner map is unchanged for callers who never replay"
    assert claims == {"a.md": [("b1.md", MON), ("b2.md", WED)]}


def test_fan_in_replay_uses_the_edge_live_at_the_instant():
    edges, unresolved, claims = resolve_supersession_candidates(FAN_IN_ROWS)
    verdict, validity = _verdict(_hit(), edges, 0.5, WED, unresolved, TUE, claims)
    assert verdict == "superseded", "a.md was superseded by b1.md on Tuesday"
    assert validity.superseded_by == "b1.md", "and by b1.md, not by the document written later"


def test_fan_in_replay_after_both_claims_uses_the_later_one():
    edges, unresolved, claims = resolve_supersession_candidates(FAN_IN_ROWS)
    _verdict_, validity = _verdict(_hit(), edges, 0.5, WED, unresolved, WED, claims)
    assert validity.superseded_by == "b2.md"


def test_fan_in_replay_before_any_claim_finds_no_successor():
    edges, unresolved, claims = resolve_supersession_candidates(FAN_IN_ROWS)
    # The HIT must predate the instant too, or it is `not_yet_known` and never reaches the
    # supersession branch at all.
    early = datetime(2026, 1, 2, tzinfo=timezone.utc)
    verdict, _validity = _verdict(
        _hit(indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        edges, 0.5, WED, unresolved, early, claims,
    )
    assert verdict == "ok"


def test_fan_in_answer_does_not_depend_on_the_alphabet():
    """The sharpest symptom of the old rule: renaming the two claimants flipped the answer.

    Both namings are SORTED as `ORDER BY 1, 2` sorts them before resolving, because the row order
    is derived from the names: reversing the names without re-sorting builds a row set the query
    cannot return, and the assertion then passes on unfixed code too. The invariant is that both
    answer the Monday document, whatever it happens to be called."""
    for old_name, new_name in (("b1.md", "b2.md"), ("z_old.md", "a_new.md")):
        rows = sorted(
            [("a.md", None, MON), (old_name, "a.md", MON), (new_name, "a.md", WED)],
            key=lambda r: (r[0], r[1] or ""),
        )
        edges, unresolved, claims = resolve_supersession_candidates(rows)
        _v_, validity = _verdict(_hit(), edges, 0.5, WED, unresolved, TUE, claims)
        assert validity.superseded_by == old_name, (
            f"as of Tuesday only {old_name} (Monday) had been asserted"
        )


def test_fan_in_without_an_instant_is_unchanged():
    """Callers who never replay must see exactly the pre-change winner."""
    edges, unresolved, claims = resolve_supersession_candidates(FAN_IN_ROWS)
    _v_, validity = _verdict(_hit(), edges, 0.5, WED, unresolved, None, claims)
    assert validity.superseded_by == "b2.md"


def test_an_aware_known_as_of_with_naive_edge_dates_does_not_raise():
    """The mirror of the naive case. Nesting the `edge_dates` normalisation inside
    `if known_as_of.tzinfo is None` fixed one combination and left this one raising, so the more
    careful caller was the one that crashed."""
    from recall.calibration import Calibration
    from recall.trust import evaluate
    from recall.types import RetrievalResult, StalenessReport

    hit = ScoredChunk(
        chunk=Chunk(id="c", source="s", text="t", metadata={"file": "a.md"}),
        score=0.9, indexed_at=None,
    )
    result = RetrievalResult(
        query="q", hits=[hit], gap_warning=False,
        staleness=StalenessReport(stale=False, newest_indexed_at=None, age=None, max_age=None),
    )
    res = evaluate(
        result, {"a.md": "b.md"},
        Calibration(embedder="test", threshold=0.5, scale=0.05),
        WED, frozenset(),
        TUE,                                   # AWARE
        _cands({"a.md": ("b.md", MON.replace(tzinfo=None))}),    # naive
    )
    assert res.hits[0].verdict == "superseded"


# --- first_indexed_at: the write path, which only Postgres can exercise ----------------------

@requires_db
def test_reindexing_a_hit_does_not_hide_it_from_a_past_replay(shared_table):
    """The half of the defect that predates edge dating entirely.

    `known_as_of` asks what we HELD at an instant. `indexed_at` moves forward on every re-index,
    so a memo edited today read `not_yet_known` for every instant before the edit: the store
    claimed it had never held a document it had held for months."""
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        store.upsert([_chunk("v1", "limits.md", "100 rps")], emb.embed(["100 rps"]))
        rows = store._with_retry(
            lambda conn: conn.execute(
                f"SELECT first_indexed_at FROM {shared_table} WHERE id = 'v1'"
            ).fetchall()
        )
        first = rows[0][0]

        store.upsert([_chunk("v1", "limits.md", "100 rps, typo fixed")],
                     emb.embed(["100 rps, typo fixed"]))
        after = store._with_retry(
            lambda conn: conn.execute(
                f"SELECT first_indexed_at, indexed_at FROM {shared_table} WHERE id = 'v1'"
            ).fetchall()
        )[0]
        assert after[0] == first, "re-writing a chunk moved its first-seen forward"
        assert after[1] > first, "and its LAST write should have moved, or nothing was re-indexed"
    finally:
        store.close()


@requires_db
def test_replace_sources_preserves_first_indexed_at_across_the_delete(shared_table):
    """`replace_sources` DELETEs then inserts, so `ON CONFLICT ... LEAST` never fires: there is no
    conflict to resolve. This is the path a real `recall index` takes, and the one that made the
    defect reachable in the first place."""
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        chunk = _chunk("v1", "limits.md", "100 rps")
        store.replace_sources([], [chunk], emb.embed(["100 rps"]))
        first = store._with_retry(
            lambda conn: conn.execute(
                f"SELECT first_indexed_at FROM {shared_table} WHERE id = 'v1'"
            ).fetchall()
        )[0][0]

        edited = _chunk("v1", "limits.md", "100 rps, edited")
        store.replace_sources(["limits.md"], [edited], emb.embed(["100 rps, edited"]))
        after = store._with_retry(
            lambda conn: conn.execute(
                f"SELECT first_indexed_at FROM {shared_table} WHERE id = 'v1'"
            ).fetchall()
        )[0][0]
        assert after == first, "the DELETE dropped the first-seen and the re-insert re-stamped it"
    finally:
        store.close()


@requires_db
def test_a_genuinely_new_chunk_gets_a_fresh_first_indexed_at(shared_table):
    """The preservation must not leak across ids: a chunk nobody has seen is new."""
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        store.replace_sources([], [_chunk("v1", "a.md", "one")], emb.embed(["one"]))
        store.replace_sources(["a.md"], [_chunk("v2", "a.md", "two")], emb.embed(["two"]))
        rows = dict(store._with_retry(
            lambda conn: conn.execute(
                f"SELECT id, first_indexed_at FROM {shared_table}"
            ).fetchall()
        ))
        assert set(rows) == {"v2"}, "v1 was replaced away"
        assert rows["v2"] is not None
    finally:
        store.close()


@requires_db
def test_the_migration_backfills_first_indexed_at_from_indexed_at(shared_table):
    """A table created before the column existed must NOT read as first written at upgrade time.

    Stamping every existing row with `now()` would claim the whole corpus appeared the day someone
    upgraded, and every replay before that instant would report an empty store. The migration
    therefore adds the column nullable, backfills from `indexed_at`, then sets the default."""
    emb = HashingEmbedder(dim=DIM)
    store = PgVectorStore(TEST_DSN, dim=DIM, table=shared_table)
    try:
        store.ensure_schema()
        store.upsert([_chunk("v1", "a.md", "one")], emb.embed(["one"]))

        # Simulate the pre-column world: drop it and re-run ensure_schema.
        store._with_retry(
            lambda conn: conn.execute(
                f"ALTER TABLE {shared_table} DROP COLUMN first_indexed_at"
            )
        )
        store.ensure_schema()

        row = store._with_retry(
            lambda conn: conn.execute(
                f"SELECT first_indexed_at, indexed_at FROM {shared_table} WHERE id = 'v1'"
            ).fetchall()
        )[0]
        assert row[0] == row[1], "backfilled rows must carry their indexed_at, not the upgrade time"
    finally:
        store.close()


# --- round four: the rules that ARE the fix, which nothing could fail on --------------------
#
# Every fan-in fixture above happens to list the OLDER claim first, so "latest asserted" and
# "last in scan order" coincide and cannot be told apart. Replacing the whole selection rule with
# pure scan order — the exact time-independent rule this change calls wrong — left the suite
# green. These rows put the two in conflict, which is the only shape that can distinguish them.

#: Scan order and date order DISAGREE: b1 is first but asserted later. In `ORDER BY 1, 2` order,
#: so it is a shape the production query can actually return.
FAN_IN_ANTI = [("a.md", None, MON), ("b1.md", "a.md", WED), ("b2.md", "a.md", MON)]


def test_the_latest_asserted_claim_wins_not_the_last_scanned():
    edges, _unresolved, claims = resolve_supersession_candidates(FAN_IN_ANTI)
    assert claims == {"a.md": [("b1.md", WED), ("b2.md", MON)]}
    later = datetime(2026, 3, 5, tzinfo=timezone.utc)
    assert resolve_successor("a.md", edges, claims, later) == "b1.md", (
        "b1.md asserted Wednesday, b2.md Monday: the later assertion is live"
    )


def test_only_the_claim_live_at_the_instant_counts_even_when_it_is_not_the_latest():
    """At Tuesday only b2.md (Monday) had been asserted, though b1.md wins later."""
    edges, _unresolved, claims = resolve_supersession_candidates(FAN_IN_ANTI)
    assert resolve_successor("a.md", edges, claims, TUE) == "b2.md"


def test_a_dated_claim_beats_an_undated_one_regardless_of_order():
    """The documented priority: an undated claim is of unknown age, so a known assertion decides
    where one exists. Both list orders, or the rule is really 'last wins' wearing a disguise."""
    edges = {"a.md": "dated.md"}
    forward = {"a.md": [("undated.md", None), ("dated.md", MON)]}
    reverse = {"a.md": [("dated.md", MON), ("undated.md", None)]}
    assert resolve_successor("a.md", edges, forward, TUE) == "dated.md"
    assert resolve_successor("a.md", edges, reverse, TUE) == "dated.md"


def test_an_all_undated_fan_in_falls_back_to_scan_order():
    edges = {"a.md": "b2.md"}
    claims = {"a.md": [("b1.md", None), ("b2.md", None)]}
    assert resolve_successor("a.md", edges, claims, TUE) == "b2.md"


def test_a_target_present_with_no_live_claim_has_no_successor():
    """Distinct from the target being ABSENT, which means inconsistent input and keeps demoting."""
    edges = {"a.md": "b.md"}
    assert resolve_successor("a.md", edges, {"a.md": []}, TUE) is None
    assert resolve_successor("a.md", edges, {}, TUE) == "b.md"


def test_the_last_candidate_is_always_the_winner():
    """The invariant that lets `step` break ties by scan position and match `supersession`.

    `winner` is overwritten on every row while the candidate list is appended to, so the two are
    maintained under different rules and could drift. They are kept in step by moving a repeated
    claimant to the END of the list. Without that, a tied replay answers a different document than
    a plain search, which is the drift this single pass exists to make impossible."""
    rows = [
        ("a.md", None, MON),
        ("b2.md", "a", MON), ("b1.md", "a.md", MON), ("b2.md", "[[a]]", MON),
    ]
    winner, _unresolved, claims = resolve_supersession_candidates(rows)
    for target, claimants in claims.items():
        assert claimants[-1][0] == winner[target], (
            f"{target}: candidate list ends with {claimants[-1][0]} but the winner is "
            f"{winner[target]}, so a tied replay disagrees with a plain search"
        )


def test_replay_at_now_may_differ_from_a_plain_search():
    """Documented, not accidental. Where the later-asserted claim is not the last scan row, a
    plain search follows `supersession` (scan order) and a replay follows assertion time. The
    replay answer is the better one; `supersession()` is deliberately left alone because it is
    what every existing caller already gets."""
    edges, _unresolved, claims = resolve_supersession_candidates(FAN_IN_ANTI)
    far = datetime(2099, 1, 1, tzinfo=timezone.utc)
    assert resolve_successor("a.md", edges, claims, None) == "b2.md", "scan-order winner"
    assert resolve_successor("a.md", edges, claims, far) == "b1.md", "latest asserted"
