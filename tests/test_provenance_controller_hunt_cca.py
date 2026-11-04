"""Regressions from the 2026-11-04 DEEP CCA hunt of `recall/provenance_controller.py`.

Each test names the finding it pins and how its red state was proved (run 20261104T131626Z).
"""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from recall import (
    AtomicFact,
    DecisionCode,
    EvidenceCard,
    EvidenceCardStore,
    FactApplicationRequest,
    InMemoryFactLedger,
    ProvenanceController,
)


NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def fact(value: str) -> AtomicFact:
    return AtomicFact(
        namespace="memory", subject="user", predicate="status", object=value, context={}
    )


def card(source_fact: AtomicFact, *, source: str, supersession_links=()) -> EvidenceCard:
    return EvidenceCard(
        card_id="",
        chunk_id=source,
        source=source,
        source_digest="digest",
        valid_from=None,
        valid_until=None,
        first_indexed_at=NOW - timedelta(days=1),
        indexed_at=NOW,
        tenant_id="tenant-a",
        generation_id="gen-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="cal-a",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
        supersession_links=tuple(supersession_links),
        structured_facts=(source_fact,),
    )


def apply(ledger: InMemoryFactLedger, evidence: EvidenceCard, claim: AtomicFact, request_id: str):
    store = EvidenceCardStore()
    store.put([evidence])
    return ProvenanceController(
        tenant_id="tenant-a",
        generation_id="gen-a",
        cards=store,
        ledger=ledger,
        source_digest_for=lambda _card: "digest",
        now=lambda: NOW,
        writer="test",
    ).apply_fact(FactApplicationRequest(claim, (evidence.card_id,), request_id))


def _superseding(link: str, old_source: str = "work/status.md"):
    ledger = InMemoryFactLedger()
    busy, free = fact("busy"), fact("free")
    assert apply(ledger, card(busy, source=old_source), busy, "r-old").code == DecisionCode.APPLIED
    return apply(ledger, card(free, source="home/update.md", supersession_links=(link,)), free, "r-new")


def test_stakes002_a_folder_link_does_not_retire_a_namesake_in_another_folder():
    """STAKES-002 (P1): `supersedes: home/status.md` retired the fact of `work/status.md`.

    `_superseding_fact_ids` compared only the folded file stems, so a link naming one folder
    designated every memo of that file name. The store's resolver (`recall.supersession`,
    `ReferenceIndex.match`) matches a reference that names a folder by its path. The claim must be
    refused as a contradiction nobody superseded.

    Red proof: against the pre-fix `recall/provenance_controller.py` (snapshot of run
    20261104T131626Z) this returned APPLIED and failed the assertion below.
    """
    decision = _superseding("home/status.md")
    assert decision.code == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION


def test_stakes002_control_links_that_name_the_memo_still_supersede():
    """Control: the path, the path under a root, and a bare file name all still designate it."""
    assert _superseding("work/status.md").code == DecisionCode.APPLIED
    assert _superseding("./work/Status.md").code == DecisionCode.APPLIED
    assert _superseding("work/status.md", old_source="/root/work/status.md").code == DecisionCode.APPLIED
    # A bare name is a stem link (lite names memos relative to RECALL_INDEX_ROOT, #912).
    assert _superseding("Status.md").code == DecisionCode.APPLIED
    assert _superseding("[[status]]").code == DecisionCode.APPLIED


def test_bug002_a_retry_of_an_applied_request_is_its_duplicate_after_a_later_value():
    """BUG-002 (P2): an APPLIED request sent again was validated afresh, so after a later value
    superseded it the retry came back refused (CONTRADICTION_WITHOUT_SUPERSESSION) while its
    APPLIED event stood. `replays_request`: sent again, a request is answered with its event.

    Red proof: against the pre-fix controller the retry returned allowed=False with
    CONTRADICTION_WITHOUT_SUPERSESSION and failed the first assertion after the retry.
    """
    ledger = InMemoryFactLedger()
    busy, free = fact("busy"), fact("free")
    old_card = card(busy, source="work/status.md")
    first = apply(ledger, old_card, busy, "r-1")
    assert first.code == DecisionCode.APPLIED
    later = card(free, source="work/update.md", supersession_links=(old_card.card_id,))
    assert apply(ledger, later, free, "r-2").code == DecisionCode.APPLIED
    count = len(ledger.events)
    retry = apply(ledger, old_card, busy, "r-1")
    assert retry.allowed and retry.code == DecisionCode.DUPLICATE
    assert retry.event is not None and first.event is not None
    assert retry.event.event_id == first.event.event_id
    assert len(ledger.events) == count, "the retry wrote nothing"


def test_bug002_control_a_reused_id_for_another_fact_is_still_refused():
    """Control: the replay answer needs the same fact and the same cards."""
    ledger = InMemoryFactLedger()
    busy, other = fact("busy"), fact("away")
    assert apply(ledger, card(busy, source="a.md"), busy, "r-1").code == DecisionCode.APPLIED
    count = len(ledger.events)
    reused = apply(ledger, card(other, source="b.md"), other, "r-1")
    assert reused.code == DecisionCode.REQUEST_ID_REUSED and not reused.allowed
    assert len(ledger.events) == count


class _Outbox:
    """An outbox whose claim fails, as a dropped connection does, and that logs every write."""

    def __init__(self) -> None:
        self.failed: list[tuple[str, object]] = []

    def enqueue(self, event) -> None:
        pass

    def status(self, event_id):
        return "processing"

    def claim(self, *, tenant_id, event_id=None, limit=100, now=None):
        raise ConnectionError("claim lost its connection")

    def mark_applied(self, *, tenant_id, event_id, lease_token=None) -> None:
        raise AssertionError("nothing was claimed")

    def mark_failed(self, *, tenant_id, event_id, error, lease_token=None) -> None:
        self.failed.append((event_id, lease_token))


class _Materializer:
    def materialize(self, event) -> None:
        raise AssertionError("nothing was claimed")


def test_bug003_a_claim_that_raised_releases_no_lease():
    """BUG-003 (P2): when the outbox's enqueue or claim raised, the cleanup called `mark_failed`
    with no lease token, which every outbox reads as no ownership check: it set `failed` on a
    row another worker held or had already applied.

    Red proof: against the pre-fix controller `outbox.failed` held one call with token None.
    """
    busy = fact("busy")
    evidence = card(busy, source="a.md")
    store = EvidenceCardStore()
    store.put([evidence])
    outbox = _Outbox()
    decision = ProvenanceController(
        tenant_id="tenant-a",
        generation_id="gen-a",
        cards=store,
        ledger=InMemoryFactLedger(),
        source_digest_for=lambda _card: "digest",
        materializer=_Materializer(),
        materialization_outbox=outbox,
        now=lambda: NOW,
        writer="test",
    ).apply_fact(FactApplicationRequest(busy, (evidence.card_id,), "r-1"))
    assert decision.code == DecisionCode.MATERIALIZATION_UNAVAILABLE
    assert outbox.failed == []


def _hit(metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(
        verdict="ok",
        confidence=0.9,
        chunk=SimpleNamespace(id="chunk-1", metadata=metadata, source="a.md", text="text"),
        provenance=SimpleNamespace(file="a.md", first_indexed_at=NOW, indexed_at=NOW),
        validity=SimpleNamespace(valid_from=None, valid_until=None),
    )


def _result(hit: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        hits=[hit],
        tenant_id="tenant-a",
        generation_id="gen-a",
        diagnostics=SimpleNamespace(index_generation="gen-a"),
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="cal-a",
        calibration_status="certified",
        trust_state="trusted",
    )


def test_code001_a_malformed_authored_fact_is_skipped_not_raised():
    """CODE-001 (P2): the public `recall.cards_from_trusted_result` (this module's copy) parsed
    authored facts unguarded, so one fact without a predicate raised and lost every card;
    `recall.evidence` skips it.

    Red proof: against the pre-fix controller the call raised KeyError('predicate').
    """
    from recall.provenance_controller import cards_from_trusted_result

    good = {"subject": "user", "predicate": "status", "object": "busy"}
    metadata = {"recall_graph": {"facts": [good, {"subject": "user"}]}, "content_hash": "d"}
    cards = cards_from_trusted_result(_result(_hit(metadata)))
    assert len(cards) == 1
    assert [f.object for f in cards[0].structured_facts] == ["busy"]


def _event(index: int, value: str, predicate: str, cardinality=None):
    from recall.provenance_controller import FactEvent, fact_identity

    claim = AtomicFact(
        namespace="memory", subject="user", predicate=predicate, object=value, context={},
        cardinality=cardinality,
    )
    return FactEvent(
        event_id=f"e{index}", event_type="asserted", tenant_id="tenant-a", generation_id="gen-a",
        fact=claim, fact_id=fact_identity(claim), evidence_cards=(), supersedes_fact_ids=(),
        request_id=f"r{index}", writer="test", decision_code="APPLIED", policy_version="v",
        controller_version=1, created_at=NOW + timedelta(seconds=index),
    )


def test_perf001_values_of_one_multi_valued_key_are_not_compared_pairwise(monkeypatch):
    """PERF-001 (P2): `split_contested` compared every pair of current values of one key, though
    two values of one multi-valued cardinality never conflict: k visits cost k^2 / 2 calls of
    `facts_conflict` on every current-facts listing.

    Red proof: against the pre-fix controller `calls` was 1770 (60 * 59 / 2), not 0.
    """
    import recall.provenance_controller as pc

    calls = 0
    real = pc.facts_conflict

    def counting(left, right):
        nonlocal calls
        calls += 1
        return real(left, right)

    monkeypatch.setattr(pc, "facts_conflict", counting)
    current = [_event(i, f"place-{i}", "visited") for i in range(60)]
    standing, groups = pc.split_contested(current)
    assert calls == 0
    assert len(standing) == 60 and groups == ()


def test_perf001_control_mixed_cardinalities_split_as_every_pair_would():
    """Control: on mixed groups the split equals the all-pairs rule of `facts_conflict`."""
    import recall.provenance_controller as pc

    current = [
        _event(0, "a", "status"), _event(1, "b", "status"), _event(2, "c", "status", "set"),
        _event(3, "x", "visited"), _event(4, "y", "visited"), _event(5, "z", "visited", "single"),
        _event(6, "p", "email"), _event(7, "q", "email", "event"), _event(8, "r", "email"),
        _event(9, "s", "email"), _event(10, "t1", "tag"), _event(11, "t2", "tag"),
    ]
    contested = {
        event.event_id
        for left in current
        for right in current
        if left is not right
        and left.fact is not None
        and right.fact is not None
        and pc.facts_conflict(left.fact, right.fact)
        for event in (left, right)
    }
    standing, groups = pc.split_contested(current)
    assert {e.event_id for e in standing} == {e.event_id for e in current} - contested
    assert {e.event_id for g in groups for e in g} == contested
    # Every status, visit and email value meets a single-valued or differently declared one; the
    # two tags are one set and stand.
    assert contested == {f"e{i}" for i in range(10)}
    assert {e.event_id for e in standing} == {"e10", "e11"}


def test_code013_a_bad_card_id_is_named_as_such():
    """CODE-013 (deferred pass, run 20261104T151220Z): a blank, non-string or over-long card id
    was refused with "at least one evidence card id is required", the message of no card at all.

    Red proof: against the pre-fix controller the raised message did not name the bound and
    failed the `match` below.
    """
    import pytest

    busy = fact("busy")
    for bad in (" ", "x" * 257, 7):
        with pytest.raises(ValueError, match="non-empty strings of at most 256 characters"):
            FactApplicationRequest(busy, (bad,), "r-1")
    with pytest.raises(ValueError, match="evidence card count is outside the supported bound"):
        FactApplicationRequest(busy, (), "r-1")


def test_perf002_the_claim_is_hashed_once_per_support_check(monkeypatch):
    """PERF-002 (deferred pass, run 20261104T151220Z): `_supported` hashed the claim again for
    every structured fact of every card.

    Red proof: against the pre-fix controller `claim_hashes` was 40 (40 card facts), not 1.
    """
    import recall.provenance_controller as pc

    busy = fact("busy")
    others = [fact(f"other-{i}") for i in range(39)]
    # The card states an equal fact, not the claim object itself, so only the claim is counted.
    cards = [card(f, source=f"m{i}.md") for i, f in enumerate([*others, fact("busy")])]
    claim_hashes = 0
    real = pc.fact_identity

    def counting(value):
        nonlocal claim_hashes
        if value is busy:
            claim_hashes += 1
        return real(value)

    monkeypatch.setattr(pc, "fact_identity", counting)
    assert pc.ProvenanceController._supported(busy, cards)
    assert claim_hashes == 1
