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

from datetime import datetime, timezone

from recall.store import resolve_edge_dates
from recall.trust import _verdict, resolve_successor

MON = datetime(2026, 3, 2, tzinfo=timezone.utc)
TUE = datetime(2026, 3, 3, tzinfo=timezone.utc)
WED = datetime(2026, 3, 4, tzinfo=timezone.utc)

from recall.types import Chunk, ScoredChunk  # noqa: E402


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
# `resolve_edge_dates` is pure for the same reason `resolve_supersession` is: the rule can then
# be tested without a database. It has to be, here: there is no reachable Postgres in this
# environment, so the SQL that feeds it is NOT exercised by any test below.

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


def test_OPEN_DEFECT_reindexing_the_superseding_file_loses_its_edge():
    """PINS A KNOWN BUG so it cannot be forgotten, and fails the day it is fixed.

    `indexed_at` records the LAST write. `Indexer.index_path` skips unchanged files and
    `replace_sources` re-inserts with `now()`, so editing b.md moves its date forward while a.md
    keeps the old one. Replaying TUE then drops an edge that existed on TUE.

    The assertion below is the WRONG answer, asserted deliberately. When `first_indexed_at` lands
    this test breaks, and the correct expectation is `"b.md"`.
    """
    rows = [("a.md", None, MON), ("b.md", "a.md", WED)]  # b.md re-indexed WED, a.md untouched
    dates = resolve_edge_dates(rows, {"a.md": "b.md"})
    assert dates == {"a.md": WED}
    assert resolve_successor("a.md", {"a.md": "b.md"}, dates, TUE) is None  # truth on TUE: b.md
