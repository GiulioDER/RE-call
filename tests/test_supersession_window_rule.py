"""The supersession window rule, the shared resolver, and every reader agreeing on one corpus.

Owner decision, 2026-10-21: a successor hides its predecessor ONLY inside the successor's own
validity window (`valid_until` inclusive to the end of its day). When a time-bounded successor
expires, the predecessor is current again. Defects from the 2026-10-21 supersession audit (P1 #1,
#4, #5; P2 #8, #9, #10, #15 to #21, #27).

Tests named ``test_w*`` run at the real boundary (a `LiteStore`, `trusted_search`,
`project_current_state`) and need no new symbol, so their red proof is the pre-fix tree itself:
master ``67ffcce59`` with this file copied in. The others exercise new symbols and are red-proved
by a named mutation of the production line instead. Each docstring says which, and how it fails.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from recall.calibration import Calibration
from recall.current_state import project_current_state
from recall.embeddings import HashingEmbedder
from recall.lite import LiteStore
from recall.store import resolve_supersession_candidates
from recall.types import Chunk
from tests.conftest import dev_search, requires_db

DIM = 64
QUERY = "rotor bearing pressure"
NOW = datetime(2026, 10, 21, 12, tzinfo=UTC)


def _chunk(name: str, *, source: str | None = None, **metadata: Any) -> Chunk:
    where = source or f"/corpus/{name}"
    return Chunk(id=where, source=where, text=QUERY, metadata={"file": name, "ord": 0, **metadata})


def _lite(tmp_path: Path, chunks: list[Chunk]) -> LiteStore:
    store = LiteStore(tmp_path / "m.db", dim=DIM)
    store.upsert(chunks, HashingEmbedder(dim=DIM).embed([c.text for c in chunks]))
    return store


def _verdicts(store: Any, at: datetime) -> dict[str, tuple[str, str | None]]:
    """`trusted_search` at valid time `at`: file -> (verdict, superseded_by)."""
    result = dev_search(store, HashingEmbedder(dim=DIM), QUERY, k=20, now=at)
    return {str(h.provenance.file): (h.verdict, h.validity.superseded_by) for h in result.hits}


def _states(store: Any, at: datetime) -> dict[str, str]:
    return {r.source: r.state for r in project_current_state(store, as_of=at).records}


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(hour=12, tzinfo=UTC)


# ------------------------------------------------------------------ the window rule, at the boundary


def test_w1_a_future_dated_successor_leaves_its_predecessor_current(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w1_a_future_dated_successor_leaves_its_predecessor_current.

    Invariant: B `supersedes: a` from 2026-11-01 hides A only from then. Red on the pre-fix tree
    (``67ffcce59``): fails on "a successor not yet valid hid its predecessor", A reads
    ``superseded`` today and the search abstains (audit P1 #1).
    """
    with closing(_lite(tmp_path, [_chunk("a.md"), _chunk("b.md", supersedes="a", valid_from="2026-11-01")])) as store:
        today = _verdicts(store, NOW)
        assert today["a.md"] == ("ok", None), f"a successor not yet valid hid its predecessor: {today}"
        assert today["b.md"][0] == "not_yet_valid"
        later = _verdicts(store, _at("2026-12-01"))
        assert later["a.md"] == ("superseded", "b.md")
        assert later["b.md"][0] == "ok"
        assert _states(store, NOW)["a.md"] == "current", "current_state disagrees with the trust gate"


def test_w2_an_expired_bounded_successor_gives_its_predecessor_back(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w2_an_expired_bounded_successor_gives_its_predecessor_back.

    Invariant (the cost the owner accepted): once B's window has ended, A is current again, and
    on B's last day (`valid_until` is inclusive) A is still hidden. Red on ``67ffcce59``: fails on
    "an expired successor still hides its predecessor".
    """
    chunks = [_chunk("a.md"), _chunk("b.md", supersedes="a.md", valid_until="2026-10-01")]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        assert today["a.md"] == ("ok", None), f"an expired successor still hides its predecessor: {today}"
        assert today["b.md"][0] == "expired"
        last_day = _verdicts(store, datetime(2026, 10, 1, 23, 59, tzinfo=UTC))
        assert last_day["a.md"] == ("superseded", "b.md"), "valid_until is inclusive to the end of its day"
        assert _states(store, NOW)["a.md"] == "current"


def test_w3_a_retroactive_correction_leaves_the_present_alone(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w3_a_retroactive_correction_leaves_the_present_alone.

    Invariant: a 2020 to 2022 correction of an open-ended document hides it in 2021 and nowhere
    else. Red on ``67ffcce59``: fails on "a correction of 2020 to 2022 hid the present".
    """
    chunks = [
        _chunk("policy.md", valid_from="2020-01-01"),
        _chunk("policy-fix.md", supersedes="policy", valid_from="2020-01-01", valid_until="2021-12-31"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        assert today["policy.md"] == ("ok", None), f"a correction of 2020 to 2022 hid the present: {today}"
        then = _verdicts(store, _at("2021-06-01"))
        assert then["policy.md"] == ("superseded", "policy-fix.md")
        assert _states(store, _at("2021-06-01"))["policy.md"] == "superseded"
        assert _states(store, NOW)["policy.md"] == "current"


def test_w4_a_chain_whose_middle_member_is_temporary(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w4_a_chain_whose_middle_member_is_temporary.

    Invariant: in a -> b -> c, `a` is hidden whenever ANY member of its chain is valid, by the
    deepest valid one, and current in a gap. b holds January to March, c from June. Red on
    ``67ffcce59``: fails on "a was hidden in the gap between b and c".
    """
    chunks = [
        _chunk("a.md"),
        _chunk("b.md", supersedes="a", valid_from="2026-01-01", valid_until="2026-03-31"),
        _chunk("c.md", supersedes="b", valid_from="2026-06-01"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        gap = _verdicts(store, _at("2026-04-15"))
        assert gap["a.md"] == ("ok", None), f"a was hidden in the gap between b and c: {gap}"
        assert _verdicts(store, _at("2026-02-15"))["a.md"] == ("superseded", "b.md")
        july = _verdicts(store, _at("2026-07-15"))
        assert july["a.md"] == ("superseded", "c.md"), f"the deepest valid member names it: {july}"
        assert july["b.md"] == ("superseded", "c.md")
        assert _states(store, _at("2026-04-15"))["a.md"] == "current"
        assert _states(store, _at("2026-07-15"))["a.md"] == "superseded"


def test_w5_a_successor_whose_window_cannot_be_read_still_hides(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w5_a_successor_whose_window_cannot_be_read_still_hides.

    Invariant: an unknown window keeps today's behaviour, the edge applies (fail closed). A
    CONTROL on the pre-fix tree (green there: it pins what must not change). Red proof by
    mutation of `recall.supersession.window_contains`: return False for an unknown (None)
    window, so an unparseable successor hides nothing; fails on "an unreadable successor window
    revived its predecessor".
    """
    chunks = [_chunk("a.md"), _chunk("b.md", supersedes="a", valid_from="next week")]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        assert today["a.md"] == ("superseded", "b.md"), f"an unreadable successor window revived its predecessor: {today}"
        assert today["b.md"][0] == "invalid_metadata"


def test_w6_two_roots_same_readme_stay_two_documents(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w6_two_roots_same_readme_stay_two_documents.

    Invariant: files are keyed by root as well as path, and a claim resolves within the
    claimant's own root first, so `CHANGES.md supersedes README` in root one leaves root two's
    README served. Red on ``67ffcce59``: fails on "the other root's README was hidden too"
    (audit P1 #5).
    """
    chunks = [
        _chunk("README.md", source="/one/README.md"),
        _chunk("README.md", source="/two/README.md"),
        _chunk("CHANGES.md", source="/one/CHANGES.md", supersedes="README"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        result = dev_search(store, HashingEmbedder(dim=DIM), QUERY, k=20, now=NOW)
        by_source = {h.chunk.source: h.verdict for h in result.hits}
        assert by_source["/two/README.md"] == "ok", f"the other root's README was hidden too: {by_source}"
        assert by_source["/one/README.md"] == "superseded"
        states = _states(store, NOW)
        assert states.get("/two/README.md") == "current", f"current_state merged the two roots: {states}"


def test_w7_a_path_qualified_claim_names_one_of_two_same_named_files(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w7_a_path_qualified_claim_names_one_of_two_same_named_files.

    Invariant: an exact root-relative path is matched before the stem, and the claimant is never
    its own candidate, so `2026/notes.md supersedes 2025/notes.md` hides 2025 and leaves the
    claimant servable. Red on ``67ffcce59``: fails on "the claimant was made unservable by its
    own claim" (both read ``ambiguous_supersession``, audit P2 #8).
    """
    chunks = [
        _chunk("2025/notes.md"),
        _chunk("2026/notes.md", supersedes="2025/notes.md"),
        _chunk("other/notes.md"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        assert today["2026/notes.md"] == ("ok", None), f"the claimant was made unservable by its own claim: {today}"
        assert today["2025/notes.md"] == ("superseded", "2026/notes.md")
        assert today["other/notes.md"] == ("ok", None)


def test_w8_a_reference_matches_case_insensitively(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w8_a_reference_matches_case_insensitively.

    Invariant: `supersedes: notes` names `Notes.mdx`; matching folds case and NFC and strips a
    markdown-family extension. Red on ``67ffcce59``: fails on "a differently cased reference
    dangled" (audit P2 #9).
    """
    with closing(_lite(tmp_path, [_chunk("Notes.mdx"), _chunk("new.md", supersedes="notes")])) as store:
        today = _verdicts(store, NOW)
        assert today["Notes.mdx"] == ("superseded", "new.md"), f"a differently cased reference dangled: {today}"


def test_w9_names_that_collide_after_folding_are_ambiguous(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w9_names_that_collide_after_folding_are_ambiguous.

    Invariant: two files whose names fold to one stem are refused, never guessed at. Red on
    ``67ffcce59``: fails on "a folded collision was resolved by a guess" (the pre-fix exact
    match picked `notes.md` and served `Notes.md` as current).
    """
    chunks = [_chunk("Notes.md"), _chunk("notes.md"), _chunk("new.md", supersedes="notes")]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        verdicts = {today["Notes.md"][0], today["notes.md"][0]}
        assert verdicts == {"ambiguous_supersession"}, f"a folded collision was resolved by a guess: {today}"


def test_w10_a_self_claim_at_the_end_of_a_chain_is_dropped(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w10_a_self_claim_at_the_end_of_a_chain_is_dropped.

    Invariant: a self-claim is no claim at any step, so a -> b -> c with c claiming itself
    resolves to c, and c is current. Red on ``67ffcce59``: fails on "a resolved short of the end
    of its chain" (it answered b, audit P2 #10).
    """
    chunks = [
        _chunk("a.md"),
        _chunk("b.md", supersedes="a"),
        _chunk("c.md", supersedes=["b", "c"]),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        assert today["a.md"] == ("superseded", "c.md"), f"a resolved short of the end of its chain: {today}"
        assert today["c.md"] == ("ok", None)
        assert _states(store, NOW)["c.md"] == "current"


def test_w11_a_self_claim_cannot_unmask_a_superseded_file(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w11_a_self_claim_cannot_unmask_a_superseded_file.

    Invariant: `Z.md` claiming itself must not displace `B.md`'s real claim on it as last row.
    Red on ``67ffcce59``: fails on "a self-claim unmasked a superseded file" (Z read ``ok``).
    """
    with closing(_lite(tmp_path, [_chunk("B.md", supersedes="Z.md"), _chunk("Z.md", supersedes="Z.md")])) as store:
        today = _verdicts(store, NOW)
        assert today["Z.md"] == ("superseded", "B.md"), f"a self-claim unmasked a superseded file: {today}"


@requires_db
def test_w12_the_postgres_scan_reads_each_successor_window(make_store: Any) -> None:
    """Node tests/test_supersession_window_rule.py::test_w12_the_postgres_scan_reads_each_successor_window.

    The SQL half of the rule (`recall.supersession.supersession_rows_sql`), which no test reaches
    without a database. Red on ``67ffcce59``: fails on "the PostgreSQL scan applied a successor
    before its window" (the scan carried no window, so A read ``superseded``).
    """
    emb = HashingEmbedder(dim=DIM)
    store = make_store(DIM)
    chunks = [
        _chunk("a.md"),
        _chunk("b.md", supersedes="a", valid_from="2026-11-01"),
        _chunk("c.md"),
        _chunk("d.md", supersedes="c", valid_until="2026-01-31"),
    ]
    store.upsert(chunks, emb.embed([c.text for c in chunks]))
    today = _verdicts(store, NOW)
    assert today["a.md"] == ("ok", None), f"the PostgreSQL scan applied a successor before its window: {today}"
    assert today["c.md"] == ("ok", None)
    assert _verdicts(store, _at("2026-01-31"))["c.md"] == ("superseded", "d.md")


# ------------------------------------------------------------------ promotion, replay, the resolver


def test_an_invalidated_low_score_successor_is_never_promoted() -> None:
    """Node tests/test_supersession_window_rule.py::test_an_invalidated_low_score_successor_is_never_promoted.

    Invariant: dependency invalidation applies before promotion (audit P1 #4). A 0.9 superseded
    by B 0.3, B dependency-invalidated: B must not be served. Red on ``67ffcce59``: fails on
    "an invalidated successor was promoted to ok".
    """
    from datetime import timedelta
    from types import SimpleNamespace

    from recall.trust_gate import evaluate
    from recall.types import RetrievalResult, ScoredChunk, StalenessReport

    def hit(cid: str, file: str, score: float, **meta: Any) -> ScoredChunk:
        return ScoredChunk(Chunk(cid, file, "t", {"file": file, "ord": 0, **meta}), score, indexed_at=NOW)

    invalidation = SimpleNamespace(cause="prerequisite_expired", path=("B.md", "P.md"))
    projection = SimpleNamespace(reason_for=lambda s: invalidation if s == "B.md" else None)
    result = RetrievalResult(
        "q",
        [hit("a", "A.md", 0.9), hit("b", "B.md", 0.3, supersedes="A.md")],
        False,
        StalenessReport(False, NOW, timedelta(0), timedelta(days=2)),
    )
    trusted = evaluate(
        result, {"A.md": "B.md"}, Calibration(embedder="t", threshold=0.5, scale=0.05), NOW,
        dependency_projection=projection, dependency_mode="enforce", record_metrics=False,
    )
    verdicts = {h.provenance.file: h.verdict for h in trusted.hits}
    assert verdicts["B.md"] != "ok", f"an invalidated successor was promoted to ok: {verdicts}"
    assert verdicts["B.md"] == "dependency_invalidated"
    assert trusted.abstained


def test_replay_still_rewinds_edges_under_the_window_rule() -> None:
    """Node tests/test_supersession_window_rule.py::test_replay_still_rewinds_edges_under_the_window_rule.

    Invariant: transaction time (`known_as_of`) and valid time (`now`) compose: an edge asserted
    after the replay instant does not apply, and one asserted before applies only inside its
    successor's window. Red proof by mutation of `recall.supersession._live_claims`: drop the
    `known_as_of` filter (`when <= known_as_of`) for the primary claim; fails on "an edge
    asserted after the replay instant applied".
    """
    from recall.supersession import resolve_supersession_graph

    written = datetime(2026, 9, 1, tzinfo=UTC)
    rows = [
        ("a.md", None, datetime(2026, 1, 1, tzinfo=UTC), "/c/a.md", None, None),
        ("b.md", "a", written, "/c/b.md", "2026-06-01", None),
    ]
    graph = resolve_supersession_graph(rows)
    assert graph.successor("a.md", now=NOW, known_as_of=datetime(2026, 8, 1, tzinfo=UTC)) is None, (
        "an edge asserted after the replay instant applied"
    )
    assert graph.successor("a.md", now=NOW, known_as_of=NOW) == "b.md"
    assert graph.successor("a.md", now=_at("2026-05-01"), known_as_of=NOW) is None


def test_the_stores_three_part_accessor_is_a_projection_of_the_graph() -> None:
    """Node tests/test_supersession_window_rule.py::test_the_stores_three_part_accessor_is_a_projection_of_the_graph.

    Invariant: `resolve_supersession_candidates` (what `supersession_all()` returns) and the
    graph come from one pass, so every caller of the old shape sees the new resolver too. Red on
    ``67ffcce59`` and by mutation of `recall.supersession._resolve_rows` (resolve a ``"self"``
    match to the claimant instead of dropping it), both on "the three-part shape kept a
    self-edge" (2026-10-21).
    """
    rows = [("b.md", "a.md", NOW), ("c.md", "b.md", NOW), ("c.md", "c.md", NOW), ("a.md", None, NOW)]
    edges, unresolved, candidates = resolve_supersession_candidates(rows)
    assert "c.md" not in edges, f"the three-part shape kept a self-edge: {edges}"
    assert edges == {"a.md": "b.md", "b.md": "c.md"}
    assert not unresolved
    assert set(candidates) == {"a.md", "b.md"}


# ------------------------------------------------------------------ every reader, one corpus

#: One corpus, every shape the rule decides: a plain edge, a future-dated successor, an expired
#: one, a retroactive correction, a chain with a temporary middle, a self-claim, a case-folded
#: reference, a path-qualified reference between same-named files, an ambiguous stem, and two
#: roots holding `README.md`.
AGREEMENT_CORPUS = [
    _chunk("plain-old.md"),
    _chunk("plain-new.md", supersedes="plain-old"),
    _chunk("future-old.md"),
    _chunk("future-new.md", supersedes="future-old", valid_from="2026-11-01"),
    _chunk("expired-old.md"),
    _chunk("expired-new.md", supersedes="expired-old", valid_until="2026-03-31"),
    _chunk("retro-old.md", valid_from="2020-01-01"),
    _chunk("retro-fix.md", supersedes="retro-old", valid_from="2020-01-01", valid_until="2021-12-31"),
    _chunk("chain-a.md"),
    _chunk("chain-b.md", supersedes="chain-a", valid_from="2026-01-01", valid_until="2026-03-31"),
    _chunk("chain-c.md", supersedes="chain-b", valid_from="2026-06-01"),
    _chunk("self.md", supersedes="self"),
    _chunk("Cased.md"),
    _chunk("cased-new.md", supersedes="cased"),
    _chunk("x/dup.md"),
    _chunk("y/dup.md"),
    _chunk("dup-new.md", supersedes="x/dup.md"),
    _chunk("amb/one.md"),
    _chunk("amb2/one.md"),
    _chunk("amb-new.md", supersedes="one"),
    _chunk("README.md", source="/one/README.md"),
    _chunk("README.md", source="/two/README.md"),
    _chunk("CHANGES.md", source="/one/CHANGES.md", supersedes="README"),
]
AGREEMENT_INSTANTS = [_at(d) for d in ("2021-06-01", "2026-02-15", "2026-04-15", "2026-07-15", "2026-12-01")] + [NOW]


@pytest.mark.parametrize("at", AGREEMENT_INSTANTS, ids=lambda at: at.date().isoformat())
def test_every_reader_agrees_on_one_corpus(tmp_path: Path, at: datetime) -> None:
    """Node tests/test_supersession_window_rule.py::test_every_reader_agrees_on_one_corpus.

    One corpus, fed to every reader of supersession, at six instants. Each must agree with the
    trust gate (the served path) on WHICH documents are hidden and BY WHAT:

    - `project_current_state` (``superseded`` / ``ambiguous`` exactly where the gate says so);
    - graph expansion's filter (`recall_mcp.graph_expansion._chunk_is_superseded`);
    - the reasoning planner's lineage (`recall.reasoning_planner._lineage`) on the reasoning
      graph built from the same chunks;
    - the dashboard's state (`recall.dashboard.corpus._state`);
    - `trusted_related` with ``relation="supersession"``, which must serve the successor;
    - and, structurally, the reasoning graph's and the semantic graph's supersession edges.

    Red proof by mutation of `recall.current_state._chain`: walk only the primary claim of each
    step and ignore windows (``graph.successor_path(source)`` without ``now``), i.e. a reader
    keeping its own copy of the rule; fails on "current_state disagrees with the trust gate".
    """
    from recall.dashboard.corpus import _state
    from recall.reasoning_graph import build_reasoning_graph
    from recall.reasoning_planner import _lineage
    from recall.related import trusted_related
    from recall.semantic_graph import build_semantic_graph
    from recall_mcp.graph_expansion import _chunk_is_superseded

    from datetime import timedelta

    from recall.trust_gate import evaluate
    from recall.types import RetrievalResult, ScoredChunk, StalenessReport

    with closing(_lite(tmp_path, AGREEMENT_CORPUS)) as store:
        graph = store.supersession_graph()
        # The gate itself, over EVERY document: retrieval returns a bounded k, and the subject
        # here is the verdicts, not the ranking (`test_w*` cover trusted search end to end).
        pool = RetrievalResult(
            "q",
            [ScoredChunk(chunk, 0.9, indexed_at=NOW) for chunk in AGREEMENT_CORPUS],
            False,
            StalenessReport(False, NOW, timedelta(0), timedelta(days=2)),
        )
        result = evaluate(pool, graph, Calibration(embedder="t", threshold=0.5), at, record_metrics=False)
        gate = {graph.key(str(h.provenance.file), h.chunk.source): h for h in result.hits}
        assert len(gate) == len(AGREEMENT_CORPUS), "every document reaches the trust gate"
        hidden = {key: h.validity.superseded_by for key, h in gate.items() if h.verdict == "superseded"}
        ambiguous = {key for key, h in gate.items() if h.verdict == "ambiguous_supersession"}
        assert hidden, "the corpus must exercise the rule at every instant"

        states = {r.source: r.state for r in project_current_state(store, as_of=at).records}
        assert {k for k, s in states.items() if s == "superseded"} == set(hidden), (
            f"current_state disagrees with the trust gate at {at:%Y-%m-%d}: {states} vs {hidden}"
        )
        assert {k for k, s in states.items() if s == "ambiguous"} == ambiguous

        by_key = {graph.key(c.metadata["file"], c.source): c for c in AGREEMENT_CORPUS}
        expansion = {key for key, chunk in by_key.items() if _chunk_is_superseded(chunk, graph, at, None)}
        assert expansion == set(hidden), f"graph expansion disagrees: {expansion} vs {set(hidden)}"

        lineage = _lineage(build_reasoning_graph(list(AGREEMENT_CORPUS), tenant_id="t", generation_id="g"))
        planner = {key: lineage.successor(key, now=at) for key in by_key}
        assert {k: v for k, v in planner.items() if v is not None} == hidden, "the planner disagrees"

        today = at.date().isoformat()
        dashboard = {
            key: _state(key, graph, c.metadata.get("valid_from"), c.metadata.get("valid_until"), today)
            for key, c in by_key.items()
        }
        assert {k for k, s in dashboard.items() if s == "superseded"} == set(hidden), "the dashboard disagrees"
        assert {k for k, s in dashboard.items() if s == "ambiguous"} == ambiguous

        for old, successor in hidden.items():
            if successor not in by_key or gate[successor].verdict != "ok":
                continue
            seed = by_key[old].id
            related = trusted_related(
                store, seed, relation="supersession", max_items=5, now=at,
                calibration=Calibration(embedder="t", threshold=0.5),
            )
            served = {graph.key(str(h.provenance.file), h.chunk.source) for h in related.items}
            assert successor in served, f"related did not serve {old}'s successor {successor}: {served}"

        reasoning_edges = build_reasoning_graph(
            list(AGREEMENT_CORPUS), tenant_id="t", generation_id="g"
        ).authored_supersession_map()
        assert reasoning_edges == graph.edges, "the reasoning graph's edges disagree with the store's"

        semantic = build_semantic_graph(list(AGREEMENT_CORPUS), tenant_id="t", generation_id="g")
        names = {entity.id: entity.canonical_name for entity in semantic.entities}
        semantic_pairs = {
            (names[r.subject_id], r.metadata["source"]) for r in semantic.relations if r.relation == "supersedes"
        }
        # The semantic graph names a file entity by its path only, so it holds ONE `README.md`
        # entity for the two roots; that pair is compared by path, every other one exactly.
        store_pairs = {
            (by_key[target].metadata["file"], by_key[claimant].source)
            for target, claimant in graph.edges.items()
            if target in by_key
        }
        assert semantic_pairs == store_pairs, f"the semantic graph disagrees: {semantic_pairs ^ store_pairs}"


# ------------------------------------------------------------------ the 2026-10-21 audit of this change


def test_w13_a_bare_reference_is_not_captured_by_a_root_level_namesake(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w13_a_bare_reference_is_not_captured_by_a_root_level_namesake.

    Invariant: `supersedes: notes` names no folder, so it is matched by stem alone: with a
    root-level `notes.md` beside `archive/notes.md` it is ambiguous, and neither is served `ok`
    by a guess. Red on ``0f714299d``: fails on "a planted root-level file captured a bare
    reference" (the path stage resolved it to `notes.md` and `archive/notes.md` read `ok`, audit
    SEC-001).
    """
    chunks = [_chunk("archive/notes.md"), _chunk("new/plan.md", supersedes="notes"), _chunk("notes.md")]
    with closing(_lite(tmp_path, chunks)) as store:
        today = _verdicts(store, NOW)
        assert today["archive/notes.md"][0] == "ambiguous_supersession", (
            f"a planted root-level file captured a bare reference: {today}"
        )
        assert today["notes.md"][0] == "ambiguous_supersession"


def test_w14_the_claimants_root_breaks_a_tie_only_between_copies_of_one_path(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w14_the_claimants_root_breaks_a_tie_only_between_copies_of_one_path.

    Invariant: the claimant's own root decides only between copies of ONE root-relative path
    (w6); between different paths it is no evidence, so a file planted under the claimant's root
    cannot capture a bare reference meant for another root's file. Red on ``0f714299d``: fails on
    "a file planted in the claimant's root captured a bare reference" (`/two/archive/notes.md`
    read `ok`, audit SEC-001).
    """
    chunks = [
        _chunk("archive/notes.md", source="/two/archive/notes.md"),
        _chunk("new/plan.md", source="/one/new/plan.md", supersedes="notes"),
        _chunk("sub/notes.md", source="/one/sub/notes.md"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        result = dev_search(store, HashingEmbedder(dim=DIM), QUERY, k=20, now=NOW)
        by_source = {h.chunk.source: h.verdict for h in result.hits}
        assert by_source["/two/archive/notes.md"] == "ambiguous_supersession", (
            f"a file planted in the claimant's root captured a bare reference: {by_source}"
        )
        assert by_source["/one/sub/notes.md"] == "ambiguous_supersession"


def test_w15_a_root_level_file_naming_its_own_stem_supersedes_the_other_namesake(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w15_a_root_level_file_naming_its_own_stem_supersedes_the_other_namesake.

    Invariant: the claimant is never its own candidate, so root-level `notes.md` declaring
    `supersedes: notes` names the one other `notes`, `archive/notes.md`, exactly as the same
    claim from a subfolder does. Red on ``0f714299d``: fails on "a claim was dropped as a
    self-claim" (the path stage found only the claimant and returned self, so the claim, which
    the base resolver refused as ambiguous, vanished and both read `ok`, audit STAKES-003).
    """
    with closing(_lite(tmp_path, [_chunk("notes.md", supersedes="notes"), _chunk("archive/notes.md")])) as store:
        today = _verdicts(store, NOW)
        assert today["archive/notes.md"] == ("superseded", "notes.md"), f"a claim was dropped as a self-claim: {today}"
        assert today["notes.md"] == ("ok", None)


def test_w16_stated_facts_retire_through_the_chain_search_follows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_supersession_window_rule.py::test_w16_stated_facts_retire_through_the_chain_search_follows.

    Invariant: memo-level retirement of stated facts follows the chain search follows. `a`;
    `b`, valid only in the first half of 2025, superseding it; `c` superseding `b`: today search
    hides `a` by `c`, and `a`'s facts are retired. Red on ``0f714299d``: fails on "retirement
    asked only a's direct, expired claimant" (retired nothing, audit NUM-001).
    """
    from recall.fact_retraction import served_sources

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "mem")
    chunks = [
        _chunk("mem/a.md", source="/m/mem/a.md"),
        _chunk("mem/b.md", source="/m/mem/b.md", supersedes="a", valid_from="2025-01-01", valid_until="2025-06-30"),
        _chunk("mem/c.md", source="/m/mem/c.md", supersedes="b"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        assert store.supersession_graph().successor("mem/a.md", now=NOW) == "mem/c.md"
        served = served_sources(store)
        assert served is not None
        retired = served.superseded(frozenset({"mem/a.md"}), NOW)
        assert retired == frozenset({"mem/a.md"}), f"retirement asked only a's direct, expired claimant: {retired}"


def test_w17_a_superseded_memo_another_root_also_holds_still_retires_its_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_supersession_window_rule.py::test_w17_a_superseded_memo_another_root_also_holds_still_retires_its_facts.

    Invariant: a name two roots hold is keyed in the graph by each source, and retirement looks
    it up that way, so an upload staging a copy of a superseded memo under its own root cannot
    keep that memo's stated facts current. Red on ``0f714299d``: fails on "a collided name was
    looked up by name and never retired" (audit SEC-005).
    """
    from recall.fact_retraction import served_sources

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "mem")
    chunks = [
        _chunk("mem/user-drink.md", source="/m/mem/user-drink.md"),
        _chunk("mem/user-drink-2.md", source="/m/mem/user-drink-2.md", supersedes="user-drink"),
        _chunk("mem/user-drink.md", source="/m/uploads/default/job1/mem/user-drink.md"),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        assert "mem/user-drink.md" in store.supersession_graph().collided
        failures: list[str] = []
        served = served_sources(store, failures=failures)
        assert served is not None
        retired = served.superseded(frozenset({"mem/user-drink.md"}), NOW)
        assert retired == frozenset({"mem/user-drink.md"}), (
            f"a collided name was looked up by name and never retired: {retired}"
        )
        assert failures == []


def test_w18_a_dependent_of_an_expired_path_two_roots_hold_is_invalidated(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w18_a_dependent_of_an_expired_path_two_roots_hold_is_invalidated.

    Invariant: the dependency projection keys a document by its root-relative path, so two
    roots' copies of `README.md` are one node to it, current when any copy is (w6 keeps the
    current copy current); a dependent of a README expired in both roots is invalidated with two
    roots as with one. Red on
    ``0f714299d``: fails on "a dependent of an expired collided path read current" (the
    source-keyed state missed the path-keyed lookup, audit STAKES-004).
    """
    chunks = [
        _chunk("README.md", source="/r1/README.md", valid_until="2026-08-01"),
        _chunk("README.md", source="/r2/README.md", valid_until="2026-08-01"),
        _chunk("x.md", source="/r1/x.md", recall_graph={"depends_on": ["README.md"]}),
    ]
    with closing(_lite(tmp_path, chunks)) as store:
        states = _states(store, NOW)
        assert states["x.md"] == "dependency_invalidated", f"a dependent of an expired collided path read current: {states}"
        assert states["/r1/README.md"] == states["/r2/README.md"] == "expired"


def _records(tmp_path: Path, chunks: list[Chunk]) -> dict[str, tuple[str, tuple[str, ...]]]:
    with closing(_lite(tmp_path, chunks)) as store:
        return {r.source: (r.state, r.dependencies) for r in project_current_state(store, as_of=NOW).records}


def test_w20_a_collided_copy_reports_its_dependencies(tmp_path: Path) -> None:
    """Node tests/test_supersession_window_rule.py::test_w20_a_collided_copy_reports_its_dependencies.

    Invariant: the projection keys dependencies and authority by path, and each copy of a path
    two roots hold reads them by path, so both copies of `guide.md` declare `base.md`, and an
    expired `base.md` invalidates both through that prerequisite. Red on ``0f714299d``: fails on
    "a collided copy lost its dependencies" (read by source, they were ``()``, 2026-10-21
    architect gate, STAKES-004).
    """
    guides = [
        _chunk("guide.md", source="/r1/guide.md", recall_graph={"depends_on": ["base.md"]}),
        _chunk("guide.md", source="/r2/guide.md", recall_graph={"depends_on": ["base.md"]}),
    ]
    current = _records(tmp_path / "a", [_chunk("base.md", source="/r1/base.md"), *guides])
    assert current["/r1/guide.md"] == current["/r2/guide.md"] == ("current", ("base.md",)), (
        f"a collided copy lost its dependencies: {current}"
    )
    expired = _records(tmp_path / "b", [_chunk("base.md", source="/r1/base.md", valid_until="2026-08-01"), *guides])
    assert expired["/r1/guide.md"][0] == expired["/r2/guide.md"][0] == "dependency_invalidated"


def test_a_naive_replay_instant_is_read_as_utc() -> None:
    """Node tests/test_supersession_window_rule.py::test_a_naive_replay_instant_is_read_as_utc.

    Invariant: a naive `known_as_of`, and a naive edge date, are UTC wherever the graph is
    walked, as the trust gate reads both. Red on ``0f714299d``: fails on "a naive known_as_of was
    compared raw" (TypeError against the stores' aware edge dates, in graph expansion and the
    planner, audit NUM-002). The naive edge date half, by mutation of
    `recall.supersession._live_claims` normalising the instant only (the fix's first version):
    fails on the same message, an aware instant against a naive date (2026-10-21 review).
    """
    from recall.supersession import SupersessionGraph

    def successors(written: datetime, at: datetime) -> tuple[str | None, str | None]:
        graph = SupersessionGraph.from_parts({"a.md": "b.md"}, frozenset(), {"a.md": [("b.md", written)]})
        return graph.successor("a.md", known_as_of=at), graph.successor("a.md", known_as_of=at - timedelta(days=2))

    try:
        naive_instant = successors(datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 2))
        naive_date = successors(datetime(2026, 10, 1), datetime(2026, 10, 2, tzinfo=UTC))
    except TypeError:
        pytest.fail("a naive known_as_of was compared raw")
    assert naive_instant == naive_date == ("b.md", None)


def test_the_shared_adapter_reads_a_list_and_refuses_a_malformed_answer() -> None:
    """Node tests/test_supersession_window_rule.py::test_the_shared_adapter_reads_a_list_and_refuses_a_malformed_answer.

    Invariant: `store_supersession_graph` reads any (edges, unresolved[, candidates]) sequence
    a store's `supersession_all()` returns, and refuses any other shape loudly, since an empty
    graph would serve every superseded document as current. A new symbol, so the red proof is a
    mutation of `recall.supersession.store_supersession_graph` back to the fix's first version
    (a tuple only, anything else an empty graph): fails on "a store answering a list lost its
    edges" (2026-10-21 review of the audit fixes, CODE-001).
    """
    from recall.supersession import store_supersession_graph

    class ListStore:
        def supersession_all(self) -> list[object]:
            return [{"a.md": "b.md"}, frozenset(), {}]

    class MapStore:
        def supersession_all(self) -> dict[str, str]:
            return {"a.md": "b.md"}

    assert store_supersession_graph(ListStore()).edges == {"a.md": "b.md"}, "a store answering a list lost its edges"
    with pytest.raises(TypeError, match="supersession_all"):
        store_supersession_graph(MapStore())


def test_the_gate_refuses_a_graph_with_separate_edge_dates() -> None:
    """Node tests/test_supersession_window_rule.py::test_the_gate_refuses_a_graph_with_separate_edge_dates.

    Invariant: one verdict reads one scan, so `evaluate` given a graph refuses separate
    `edge_candidates` rather than pairing one scan's edges with another's dates. Red on
    ``0f714299d``: fails on pytest's "DID NOT RAISE" (the caller's dates were used, audit
    CODE-009).
    """
    from datetime import timedelta

    from recall.supersession import SupersessionGraph
    from recall.trust_gate import evaluate
    from recall.types import RetrievalResult, StalenessReport

    pool = RetrievalResult("q", [], False, StalenessReport(False, NOW, timedelta(0), timedelta(days=2)))
    graph = SupersessionGraph.from_parts({"a.md": "b.md"})
    with pytest.raises(ValueError, match="edge_candidates"):
        evaluate(pool, graph, Calibration(embedder="t", threshold=0.5), NOW, edge_candidates={}, record_metrics=False)


def test_the_planner_reads_an_inverted_window_as_unknown() -> None:
    """Node tests/test_supersession_window_rule.py::test_the_planner_reads_an_inverted_window_as_unknown.

    Invariant: a successor whose window is inverted (from December to January) has a malformed,
    so unknown, window, and hides its predecessor at every instant, in the planner's lineage as
    in the resolver (`parse_window`). Red on ``0f714299d``: fails on "the planner kept an
    inverted window that never holds" (`a.md` had no successor there, audit CODE-006).
    """
    from recall.reasoning_graph import build_reasoning_graph
    from recall.reasoning_planner import _lineage

    inverted = {"file": "b.md", "ord": 0, "supersedes": "a", "valid_from": "2026-12-01", "valid_until": "2026-01-01"}
    chunks = [Chunk("a", "/r/a.md", "t", {"file": "a.md", "ord": 0}), Chunk("b", "/r/b.md", "t", inverted)]
    lineage = _lineage(build_reasoning_graph(chunks, tenant_id="t", generation_id="g"))
    assert lineage.successor("a.md", now=NOW) == "b.md", "the planner kept an inverted window that never holds"
