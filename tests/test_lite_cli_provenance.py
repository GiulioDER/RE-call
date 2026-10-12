"""`recall provenance current` and `apply` on a `sqlite:///` DSN, with PostgreSQL's output.

Before this, both subcommands reached PostgreSQL on a lite DSN and died in psycopg; only the
separate `--sqlite-path` adapter worked, and it is not linked to the store's chunks or cards.

Invariants and the failure each one catches:
- P1 on a lite store, cards a trusted search mints can be applied from the CLI, and `current`
  then returns the fact.
- P2 a `--generation` that is not the lite store's own is refused, naming the right one, and
  nothing is written.
- P3 `provenance current` prints the same JSON on PostgreSQL and on lite for the same ledger.
- P4 the generation route is refused on a lite DSN by name.

Red proof: recorded in each test's docstring.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from recall.evidence import cards_from_trusted_result
from recall.fact_ledger import PostgresFactLedger
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import ensure_calibrated
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import FactApplicationRequest, ProvenanceController
from recall_mcp.evidence_cards import register_evidence_cards
from recall_mcp.retrieval import _retrieve_trusted
from tests.conftest import TEST_DSN, requires_db
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_lite_cli import _run
from tests.test_lite_fact_parity import COFFEE, GEN, NOW, TEA, card

TEA_CLAIM = {"namespace": "memory", "subject": "user", "predicate": "drink", "object": "tea"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _dsn(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


def _calibrated_store(tmp_path: Path) -> LiteStore:
    """48 memos plus one carrying a structured fact whose body states it, indexed and calibrated."""
    folder = tmp_path / "mem"
    _write_memos(folder, 48)
    graph = json.dumps({"facts": [TEA_CLAIM]})
    (folder / "user-drink.md").write_text(
        f"---\nrecall_graph: {graph}\n---\n# The user drink tea\n\nThe user drink tea, every morning.\n",
        encoding="utf-8",
    )
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    embedder = ContentWordEmbedder()
    Indexer(store, embedder, env={}).index_path(folder)
    ensure_calibrated(store, embedder)
    return store


def test_p1_apply_then_current_on_a_lite_dsn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red: baseline `c5eec57c` (the provenance command before this change) reaches PostgreSQL on a
    lite DSN and raises from psycopg, so "apply did not run on the lite store" fails."""
    store = _calibrated_store(tmp_path)
    embedder = ContentWordEmbedder()
    retrieval = _retrieve_trusted(store, embedder, "user drink tea", None, 5, None, None)
    cards = cards_from_trusted_result(retrieval.result)
    assert cards, "the trusted search minted no card for the fact memo"
    register_evidence_cards(cards, store=store)
    (tmp_path / "cards.json").write_text(json.dumps([c.to_payload() for c in cards]), encoding="utf-8")
    (tmp_path / "claim.json").write_text(json.dumps(TEA_CLAIM), encoding="utf-8")
    monkeypatch.setattr("recall.cli_commands._shared._make_embedder", lambda *_a, **_k: ContentWordEmbedder())
    base = ["--dsn", _dsn(tmp_path / "memory.db")]

    out = _run([*base, "provenance", "apply", "--claim", str(tmp_path / "claim.json"),
                "--cards", str(tmp_path / "cards.json"), "--request-id", "r1",
                "--generation", store.generation_id])
    assert out.lstrip().startswith("{"), f"apply did not run on the lite store: {out}"
    decision = json.loads(out)
    assert decision["allowed"] is True and decision["decision_code"] == "APPLIED", decision

    current = json.loads(_run([*base, "provenance", "current"]))
    assert [f["fact"]["object"] for f in current["facts"]] == ["tea"], current


def test_p2_a_foreign_generation_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red: the generation check removed from `_cmd_provenance`: the apply runs (and is decided by
    the controller) instead of stopping with the store's own generation named."""
    store = _calibrated_store(tmp_path)
    one_card = card("default", TEA, "chunk-tea", "user-drink.md").to_payload()
    (tmp_path / "cards.json").write_text(json.dumps([one_card]), encoding="utf-8")
    (tmp_path / "claim.json").write_text(json.dumps(TEA_CLAIM), encoding="utf-8")
    monkeypatch.setattr("recall.cli_commands._shared._make_embedder", lambda *_a, **_k: ContentWordEmbedder())
    out = _run(["--dsn", _dsn(tmp_path / "memory.db"), "provenance", "apply",
                "--claim", str(tmp_path / "claim.json"), "--cards", str(tmp_path / "cards.json"),
                "--request-id", "r1", "--generation", "lite-0000000000000000"])
    assert "EXIT" in out and store.generation_id in out, f"a foreign generation was not refused: {out}"
    assert LiteFactLedger(store).events == (), "the refused apply still wrote to the ledger"


def _seed(cards_store: object, ledger: object, tenant: str) -> None:
    tea = card(tenant, TEA, "chunk-tea", "tea.md")
    coffee = card(tenant, COFFEE, "chunk-coffee", "coffee.md", links=("tea.md",))
    cards_store.put((tea, coffee))  # type: ignore[attr-defined]
    for fact, evidence, rid in ((TEA, tea, "r1"), (COFFEE, coffee, "r2")):
        ProvenanceController(
            tenant_id=tenant, generation_id=GEN, cards=cards_store, ledger=ledger,  # type: ignore[arg-type]
            source_digest_for=lambda _c: "digest", now=lambda: NOW, writer="parity",
        ).apply_fact(FactApplicationRequest(fact, (evidence.card_id,), rid))


@requires_db
def test_p3_current_prints_the_same_on_postgres_and_lite(tmp_path: Path) -> None:
    """Red: the lite branch of `provenance current` reading `LiteFactLedger(store).events` (every
    event, superseded ones included) instead of `.current(...)`: the JSON differs."""
    tenant = f"cli-parity-{uuid4().hex}"
    _seed(PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant), PostgresFactLedger(TEST_DSN, tenant_id=tenant), tenant)
    lite = LiteStore(tmp_path / "parity.db", dim=64, tenant=tenant)
    _seed(LiteEvidenceCardStore(lite), LiteFactLedger(lite), tenant)
    as_of = (NOW + timedelta(hours=1)).isoformat()
    tail = ["--embedder", "hashing", "--tenant", tenant, "provenance", "current", "--as-of", as_of]
    postgres = _run(["--dsn", TEST_DSN, *tail])
    on_lite = _run(["--dsn", _dsn(tmp_path / "parity.db"), *tail])
    assert json.loads(postgres)["facts"], f"the PostgreSQL ledger was not seeded: {postgres}"
    assert on_lite == postgres, f"postgres {postgres!r} != lite {on_lite!r}"


def test_p4_the_generation_route_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red: the `lite_route_refusal` check removed from `_cmd_provenance`: no refusal by name."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path / "x.db"), "provenance", "current"])
    assert "has no generations" in out, f"the generation route was not refused: {out}"
    assert not (tmp_path / "x.db").exists(), "the refused command still created the store"
