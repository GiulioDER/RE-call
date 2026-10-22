"""A fact none of whose memos the store serves is not current, and is current again when one returns.

`recall_forget` retracts a forgotten memo's facts with a permanent `retracted` event (#931). A memo
that leaves the corpus any other way (deleted from disk and pruned by a lite refresh or `recall
index`, or left out of a promoted generation) is a state rather than a decision: a restored file
or a `generation rollback` brings it back. So nothing is written: `recall_current_facts` hides a
fact whose every card, across all its applications, names a memo the store does not serve now
(`recall.provenance_controller.absent_fact_ids`), and shows it again when a memo returns.

Invariants and the failure each one catches:
- Z1 a lite refresh that prunes a fact's only memo hides the fact; restoring the file shows it.
- Z2 a fact with a memo still served stays, counting DUPLICATE applications; it goes with its last.
- Z3 a hidden fact does not block a new value of its predicate.
- Z4 a fact re-applied from a new memo while hidden is a DUPLICATE, so it is listed once, also after
  its old memo returns.
- Z5 on PostgreSQL, a promoted generation that leaves the memo out hides the fact, and
  `generation rollback` to the generation that holds it shows it again.
- Z6 the listing cites the live cards of every application of a fact (DUPLICATE ones included),
  so a pruned first memo is no longer cited while the second still is.

Red proof (2026-10-14), each against #931's tip `7a681e4f1` (pre-fix) unless a mutation is named,
then restored and green; node ids and failure lines in each test's docstring.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.lite.refresh import Outcome, refresh
from recall.provenance_controller import (
    ControllerDecision,
    DecisionCode,
    FactApplicationRequest,
    ProvenanceController,
)
from recall.types import AtomicFact
from recall_mcp.provenance import current_facts_memory
from tests.fact_fixtures import COFFEE, NOW, TEA, evidence_card
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos

DOOR = AtomicFact(namespace="memory", subject="user", predicate="door_code", object="4512")


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    _write_memos(tmp_path / "memory", 4)
    return tmp_path / "memory"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LiteStore]:
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as opened:
        yield opened


def _refresh(store: LiteStore, folder: Path) -> None:
    report = refresh(store, [folder], ContentWordEmbedder())  # type: ignore[arg-type]
    assert report.outcome is Outcome.REFRESHED, report.to_dict()


def _apply(store: LiteStore, fact: AtomicFact, memo: str, request_id: str) -> ControllerDecision:
    """Apply `fact` with one card citing the indexed memo `memo`, through the controller the MCP
    server builds (served lookup included, when this code has one)."""
    try:
        from recall.fact_retraction import served_sources
    except ImportError:  # the pre-fix tree, which red proof runs against, has no served lookup
        served_sources = None  # type: ignore[assignment]

    chunk = next(c for c in store.iter_chunks() if c.source.endswith(memo))
    name = str(chunk.metadata.get("file") or chunk.source)
    card = evidence_card(fact, tenant=store.tenant, generation="g1", chunk=chunk.id, source=name)
    cards = LiteEvidenceCardStore(store)
    cards.put((card,))
    served = None if served_sources is None else served_sources(store)
    extra = {} if served is None else {"served_sources": served}
    return ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="absent-test", **extra,
    ).apply_fact(FactApplicationRequest(fact, (card.card_id,), request_id))


def _listed(store: object) -> list[object]:
    """What `recall_current_facts` lists: each fact's object."""
    return [item["fact"]["object"] for item in current_facts_memory(store, as_of=NOW)["facts"]]  # type: ignore[arg-type, index]


def test_z1_a_pruned_memo_hides_its_fact_and_a_restored_one_shows_it(
    store: LiteStore, folder: Path, tmp_path: Path
) -> None:
    """Node tests/test_absent_memo_facts.py::test_z1_a_pruned_memo_hides_its_fact_and_a_restored_one_shows_it.

    Red against #931's tip: the projection checks no memo, so it fails on "a pruned memo's fact is
    still listed: ['4512']". Red, mutation of `served_sources` (recall/fact_retraction.py)
    returning None: the same failure.
    """
    _refresh(store, folder)
    assert _apply(store, DOOR, "memo-001.md", "r1").code == DecisionCode.APPLIED
    kept = tmp_path / "memo-001.md.kept"
    shutil.move(folder / "memo-001.md", kept)
    _refresh(store, folder)
    assert _listed(store) == [], f"a pruned memo's fact is still listed: {_listed(store)}"

    shutil.move(kept, folder / "memo-001.md")
    _refresh(store, folder)
    assert _listed(store) == ["4512"], "restoring the memo did not bring its fact back"


def test_z2_a_fact_stays_while_any_of_its_memos_is_served(store: LiteStore, folder: Path) -> None:
    """Node tests/test_absent_memo_facts.py::test_z2_a_fact_stays_while_any_of_its_memos_is_served.

    Red, mutation of `absent_fact_ids` (recall/provenance_controller.py) counting only the
    APPLIED application's cards (`asserts_fact(event)` in place of `event.event_type ==
    "asserted"`): fails on "a fact with a memo still served was hidden". Red against #931's tip:
    fails on "the fact outlived its last memo".
    """
    _refresh(store, folder)
    assert _apply(store, DOOR, "memo-001.md", "r1").code == DecisionCode.APPLIED
    assert _apply(store, DOOR, "memo-002.md", "r2").code == DecisionCode.DUPLICATE

    (folder / "memo-001.md").unlink()
    _refresh(store, folder)
    assert _listed(store) == ["4512"], "a fact with a memo still served was hidden"

    (folder / "memo-002.md").unlink()
    _refresh(store, folder)
    assert _listed(store) == [], f"the fact outlived its last memo: {_listed(store)}"


def test_z3_a_hidden_fact_does_not_block_a_new_value(store: LiteStore, folder: Path) -> None:
    """Node tests/test_absent_memo_facts.py::test_z3_a_hidden_fact_does_not_block_a_new_value.

    Red, mutation of `assertion_is_duplicate` (recall/fact_ledger.py) ignoring `absent` in its
    conflict test: the lite ledger refuses inside its append, and it fails on "a hidden value still
    contradicts the new one: CONTRADICTION_WITHOUT_SUPERSESSION". Against #931's tip the
    controller has no served lookup, so this test applies without one and fails the same way.
    """
    _refresh(store, folder)
    assert _apply(store, TEA, "memo-001.md", "r1").code == DecisionCode.APPLIED
    (folder / "memo-001.md").unlink()
    _refresh(store, folder)

    coffee = _apply(store, COFFEE, "memo-002.md", "r2")
    assert coffee.code == DecisionCode.APPLIED, f"a hidden value still contradicts the new one: {coffee.code}"
    assert _listed(store) == ["coffee"]


def test_z4_a_hidden_fact_applied_again_is_listed_once(store: LiteStore, folder: Path, tmp_path: Path) -> None:
    """Node tests/test_absent_memo_facts.py::test_z4_a_hidden_fact_applied_again_is_listed_once.

    Red, mutation of `LiteFactLedger.apply_assertion` (recall/lite/facts.py) leaving hidden facts
    out of its duplicate test (`candidates` filtered by `absent_fact_ids`): the re-application is
    APPLIED and it fails on "re-applying a hidden fact was not a duplicate". Red against #931's
    tip: the fact is never hidden, so it fails on "the fact was listed after its memo was pruned".
    """
    _refresh(store, folder)
    assert _apply(store, DOOR, "memo-001.md", "r1").code == DecisionCode.APPLIED
    kept = tmp_path / "memo-001.md.kept"
    shutil.move(folder / "memo-001.md", kept)
    _refresh(store, folder)
    assert _listed(store) == [], "the fact was listed after its memo was pruned"

    again = _apply(store, DOOR, "memo-002.md", "r2")
    assert again.code == DecisionCode.DUPLICATE, f"re-applying a hidden fact was not a duplicate: {again.code}"
    assert _listed(store) == ["4512"]
    shutil.move(kept, folder / "memo-001.md")
    _refresh(store, folder)
    assert _listed(store) == ["4512"], f"the fact is listed twice once its old memo is back: {_listed(store)}"


# --------------------------------------------------------------------------------------- PostgreSQL


class _S3:
    def __init__(self, objects: dict[tuple[str, str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs: str) -> dict[str, object]:
        data = self.objects[(kwargs["Bucket"], kwargs["Key"], kwargs["VersionId"])]
        return {"Body": BytesIO(data), "ContentLength": len(data), "VersionId": kwargs["VersionId"]}


def _corpus(tenant: str, memos: dict[str, bytes], corpus_version: str) -> tuple[object, object]:
    """A manifest of `memos` and the reader that serves them."""
    from recall.lineage import IndexManifestV1, ManifestObjectV1
    from recall.manifest import S3Allowlist, S3ObjectReader

    objects = tuple(
        ManifestObjectV1(f"s3://approved/corpora/{tenant}/{name}", "v1", "text/markdown", len(data),
                         hashlib.sha256(data).hexdigest())
        for name, data in sorted(memos.items())
    )
    store = {("approved", f"corpora/{tenant}/{name}", "v1"): data for name, data in memos.items()}
    return (
        IndexManifestV1(tenant, corpus_version, objects),
        S3ObjectReader(_S3(store), S3Allowlist.parse("approved/corpora/")),
    )


@pytest.fixture
def manager() -> Iterator[object]:
    import psycopg

    from recall.generations import GenerationManager
    from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason

    if not _db_available():
        pytest.skip(db_unreachable_reason())
    tenant = "absent-" + uuid.uuid4().hex[:10]
    yield GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for table in ("recall_source_tombstones", "recall_audit_events", "recall_ingest_jobs",
                      "recall_tenant_state", "recall_generations"):
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))  # noqa: S608


def test_z5_a_generation_without_the_memo_hides_its_fact_and_a_rollback_shows_it(manager: object) -> None:
    """Node tests/test_absent_memo_facts.py::test_z5_a_generation_without_the_memo_hides_its_fact_and_a_rollback_shows_it.

    Red against #931's tip: fails on "a memo left out of the promoted generation still lists its
    fact". Red, mutation of `GenerationStore.served_identifiers` (recall/generation_store.py)
    answering from every generation (`set(self.sources_for_identifiers(...))`): the retired
    generation still holds the memo, so the same assertion fails.
    """
    from recall.fact_ledger import PostgresFactLedger
    from recall.generation_store import GenerationStore
    from recall.provenance_cards import PostgresEvidenceCardStore
    from tests.conftest import TEST_DSN
    from tests.test_generations import _Embedder, _pipeline

    tenant = manager.tenant_id  # type: ignore[attr-defined]
    door, other = b"alpha door memo", b"beta other memo"

    def ready(memos: dict[str, bytes], version: str) -> str:
        manifest, reader = _corpus(tenant, memos, version)
        generation = manager.create(manifest, _pipeline("model-a"))  # type: ignore[attr-defined]
        manager.build(generation.generation_id, reader, _Embedder(1), lambda text: [text])  # type: ignore[attr-defined]
        manager.validate(generation.generation_id)  # type: ignore[attr-defined]
        return str(generation.generation_id)

    first = ready({"door.md": door, "other.md": other}, "corpus-v1")
    manager.promote(first, unsafe_development=True)  # type: ignore[attr-defined]
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        chunk = next(hit.chunk for hit in store.query_sparse("alpha", 5) if hit.chunk.text == door.decode())
        name = str(chunk.metadata.get("file") or chunk.source)
        card = evidence_card(DOOR, tenant=tenant, generation=first, chunk=chunk.id, source=name)
        cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
        cards.put((card,))
        decision = ProvenanceController(
            tenant_id=tenant, generation_id=first, cards=cards,
            ledger=PostgresFactLedger(TEST_DSN, tenant_id=tenant),
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="absent-test",
        ).apply_fact(FactApplicationRequest(DOOR, (card.card_id,), "r1"))
        assert decision.code == DecisionCode.APPLIED, decision
        assert _listed(store) == ["4512"]

    second = ready({"other.md": other}, "corpus-v2")
    manager.promote(second, unsafe_development=True)  # type: ignore[attr-defined]
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        assert _listed(store) == [], "a memo left out of the promoted generation still lists its fact"

    assert manager.rollback() == first  # type: ignore[attr-defined]
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        assert _listed(store) == ["4512"], "the rollback did not bring the fact back"


def test_z6_the_listing_cites_the_cards_of_every_live_memo(store: LiteStore, folder: Path) -> None:
    """Node tests/test_absent_memo_facts.py::test_z6_the_listing_cites_the_cards_of_every_live_memo.

    Invariant: a fact's `evidence_card_ids` are the live cards of all its applications, as its
    `sources` are, so with memo-001 pruned the listing cites memo-002's card, not memo-001's.
    Re-measured open on master ``52be498cc`` (2026-10-22): the listing cited the APPLIED event's
    cards. Red against ``52be498cc`` with this test copied in: fails on "the listing does not
    cite the DUPLICATE memo's card" (only memo-001's card, while both were served). Red, mutation
    of `current_facts_memory` (recall_mcp/provenance.py) citing the event's own cards: the same line.
    """
    _refresh(store, folder)
    first = _apply(store, DOOR, "memo-001.md", "r1")
    second = _apply(store, DOOR, "memo-002.md", "r2")
    assert (first.code, second.code) == (DecisionCode.APPLIED, DecisionCode.DUPLICATE)
    old, new = first.cards[0], second.cards[0]

    listed = current_facts_memory(store, as_of=NOW)["facts"]  # type: ignore[arg-type]
    assert [item["evidence_card_ids"] for item in listed] == [[old.card_id, new.card_id]], (
        f"the listing does not cite the DUPLICATE memo's card: {listed}"
    )
    (folder / "memo-001.md").unlink()
    _refresh(store, folder)
    listed = current_facts_memory(store, as_of=NOW)["facts"]  # type: ignore[arg-type]
    assert len(listed) == 1, f"the fact is not listed once: {listed}"
    assert listed[0]["evidence_card_ids"] == [new.card_id], (
        f"the listing cites a card of a memo no longer served: {listed[0]['evidence_card_ids']}"
    )
    assert listed[0]["sources"] == [new.source]
