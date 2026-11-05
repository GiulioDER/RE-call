"""A forgotten memo leaves no byte of its text anywhere the store keeps data (M5).

Invariant: after `recall forget <source> --yes` on a lite store, a canary string that only the
forgotten memo held appears in no file under the store's directory or the memo folder's sidecars:
not in a table, a WAL frame, a freed page, an FTS segment, calibration evidence, an evidence card,
a fact value or a receipt. The memo is first driven through every path that derives data from it
(indexing, the stated-fact reconcile, a calibration fit, a trusted search that mints cards, a fact
application), and the scan is of RAW BYTES, so a derived store added later is covered without
anyone registering it.

Why this shape: CCA's `race` and `data-integrity` classes hold 22 confirmed P1/P2 RE-call findings
(2026-11-04) where forget left memo text behind (calibration evidence, freed pages, outbox replay,
receipts, cards) or a concurrent writer put it back. Each was fixed and tested where it was found;
nothing checked the whole store, so each new derived store was a new place to leave residue.

The negative control is the same scan before the forget: it must find the canary, or the scan is
reading the wrong files.

Scope: a lite store, the one store a test can read byte for byte; the same paths on PostgreSQL are
not covered here. It checks what forget leaves at the end, not the ordering of a writer that runs
concurrently with it.

Red proof (2026-11-05), each restored afterwards, both failing in the final `== []` assertion with
`store/memory.db` holding the canary:

- `LiteStore.delete_sources` with its `calibration_evidence` delete made a no-op (`WHERE 0 AND`):
  the fit's questions, built from the forgotten headings, stayed.
- `recall.fact_retraction`'s `redact()` returning 0 without redacting: the forgotten fact's value
  stayed in the ledger and the current-fact tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.evidence import cards_from_trusted_result
from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import ensure_calibrated
from recall_mcp.evidence_cards import register_evidence_cards
from recall_mcp.retrieval import _retrieve_trusted
from tests.test_lite_calibration import _NOUNS, DIM, ContentWordEmbedder, _word
from tests.test_lite_cli import _run

CANARY = "zqxerasurecanary"
#: The fact the canary memo states. Its value is the canary, so every ledger row, card and current
#: fact derived from it carries the canary too.
CLAIM = {"namespace": "memory", "subject": "user", "predicate": "drink", "object": CANARY}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _holding_the_canary(roots: list[Path], skip: set[Path]) -> list[str]:
    """Every file under `roots`, other than `skip`, whose raw bytes hold the canary."""
    found = []
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.resolve() in skip:
                continue
            if CANARY.encode() in path.read_bytes():
                found.append(path.relative_to(root.parent).as_posix())
    return found


def _derive_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """49 memos holding the canary, one of which states a fact whose value is the canary: indexed,
    calibrated, searched into cards, and the fact applied. Returns the memo folder, the store file
    and the fact memo."""
    folder = tmp_path / "mem"
    folder.mkdir()
    # Every heading holds the canary, so the calibration fit's questions, built from headings,
    # carry it whichever memos the fit samples.
    for i in range(48):
        subject, part, crew = _word(i, 1), _NOUNS[i % len(_NOUNS)], _word(i, 2)
        (folder / f"memo-{i:03d}.md").write_text(
            f"# The {subject} {part} {CANARY}\n\nThe {subject} {part} is serviced by the {crew} team.\n",
            encoding="utf-8",
        )
    graph = json.dumps({"facts": [CLAIM]})
    memo = folder / "user-drink.md"
    memo.write_text(
        f"---\nrecall_graph: {graph}\n---\n# The user drink {CANARY}\n\n"
        f"The user drink {CANARY}, every morning.\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store" / "memory.db"
    store_path.parent.mkdir()
    embedder = ContentWordEmbedder()
    with LiteStore(store_path, dim=DIM) as store:
        Indexer(store, embedder, env={}).index_path(folder)
        ensure_calibrated(store, embedder)
        retrieval = _retrieve_trusted(
            store, embedder, f"user drink {CANARY}", source=None, k=5, calibration=None, policy=None
        )
        cards = cards_from_trusted_result(retrieval.result)
        assert cards, "the trusted search minted no card for the canary memo"
        register_evidence_cards(cards, store=store)
        generation = store.generation_id
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "cards.json").write_text(json.dumps([c.to_payload() for c in cards]), encoding="utf-8")
    (inputs / "claim.json").write_text(json.dumps(CLAIM), encoding="utf-8")
    monkeypatch.setattr("recall.cli_commands._shared._make_embedder", lambda *_a, **_k: ContentWordEmbedder())
    dsn = "sqlite:///" + store_path.as_posix()
    applied = _run(["--dsn", dsn, "provenance", "apply", "--claim", str(inputs / "claim.json"),
                    "--cards", str(inputs / "cards.json"), "--request-id", "r1", "--generation", generation])
    assert json.loads(applied)["decision_code"] == "APPLIED", applied
    return folder, store_path, memo


def test_a_forgotten_memo_leaves_no_byte_of_its_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder, store_path, memo = _derive_everything(tmp_path, monkeypatch)
    roots = [store_path.parent, folder]
    skip = {p.resolve() for p in folder.glob("*.md")}  # the memo files themselves, which forget keeps
    assert _holding_the_canary(roots, skip), "negative control: the scan saw no canary before the forget"

    dsn = "sqlite:///" + store_path.as_posix()
    sources = [str(path.resolve()) for path in sorted(folder.glob("*.md"))]
    done = _run(["--dsn", dsn, "--embedder", "hashing", "forget", *sources, "--yes"])
    assert f"from {len(sources)} source(s)" in done, done
    with LiteStore(store_path, dim=DIM) as store:
        assert store.checkpoint()

    assert _holding_the_canary(roots, skip) == []
