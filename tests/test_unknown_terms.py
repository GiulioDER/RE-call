"""The unknown-term gate: a question naming something the memory never contained is not answered.

Invariants and the failure each one catches:
- U1 only entity-like tokens are checked (letter and digit, all capitals, inner capital), so a
  plain word the memory lacks ("happens", "worth") can never refuse a real answer.
- U2 the lite store says which tokens occur nowhere in it, by its own FTS5 terms.
- U3 end to end on a LiteStore: a near-miss differing from an answerable question only by a name
  the memory lacks ("QQ01" for "AR25") is demoted to `unknown_term` and abstains, naming the
  token; the answerable question itself stays `ok`. This is the ARC-AGI-3 failure of 2026-10-08:
  20 of 20 such questions were trusted.
- U4 a store that cannot answer the vocabulary question is reported `unavailable` and its verdicts
  are left exactly as they were.
- U5 RECALL_UNKNOWN_TERM_GATE=0 switches the gate off, and the result says so.

Node IDs, all in `tests/test_unknown_terms.py`:
U1 `test_only_entity_like_tokens_are_checked`, U2 `test_lite_store_names_the_tokens_it_never_saw`,
U3 `test_a_question_naming_an_unknown_entity_abstains_and_the_real_one_does_not`,
U4 `test_a_store_without_a_vocabulary_is_reported_and_left_alone`,
U5 `test_the_gate_can_be_switched_off`.

Red proof, 2026-10-09, each mutation alone, the named test failing in the named assertion, then
restored byte for byte and all five green:
- M1 (U1) the letter-and-digit rule removed from `entity_like_tokens`: `build42` was missing from
  the tokens. The first version of U1 stayed green under M1, because QQ01 and ACTION3 are also
  all capitals; `build42` was added so the rule is observed on its own.
- M2 (U2, U3) `LiteStore.unknown_terms` never reporting a token: `assert [] == ['QQ01']`, and
  "a question about QQ01 was trusted".
- M3 (U3) `apply_unknown_term_gate` never demoting: "a question about QQ01 was trusted".
- M4 (U3) the call in `recall.trust._trusted_search` removed: "a question about QQ01 was trusted".
U4 and U5 are controls of the skip paths (a store without `unknown_terms`, and the switch); they
assert that nothing changes, so no mutation of the gate's positive path can turn them red.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.lite import LiteStore
from recall.types import (
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall.unknown_terms import (
    UnknownTermCheck,
    apply_unknown_term_gate,
    check_unknown_terms,
    entity_like_tokens,
)
from tests.conftest import dev_search

DIM = 64
NOTES = {
    "ar25-lesson1.md": "# AR25: lesson 1\n\nIn AR25, pressing ACTION3 rotates the blue piece one quarter turn.\n",
    "cn04-lesson1.md": "# CN04: lesson 1\n\nIn CN04, ACTION6 on a lamp toggles it and its four neighbours.\n",
}
REAL = "In AR25, pressing ACTION3 rotates the blue piece one quarter turn?"
NEAR_MISS = "In QQ01, pressing ACTION3 rotates the blue piece one quarter turn?"


def _store(tmp_path: Path) -> LiteStore:
    store = LiteStore(tmp_path / "memory.db", dim=DIM)
    root = tmp_path / "notes"
    root.mkdir()
    for name, text in NOTES.items():
        (root / name).write_text(text, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=DIM), env={}).index_path(root)
    return store


def test_only_entity_like_tokens_are_checked() -> None:
    tokens = entity_like_tokens(
        "In QQ01, what happens when I press ACTION3 on the SKU of PgVector in build42?"
    )
    # build42 is caught by the letter-and-digit rule alone: no capital anywhere.
    assert tokens == ["QQ01", "ACTION3", "SKU", "PgVector", "build42"]
    assert entity_like_tokens("What is worth doing when nothing happens?") == []


def test_lite_store_names_the_tokens_it_never_saw(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.unknown_terms(["AR25", "ACTION3", "QQ01", "CN04"]) == ["QQ01"]


def test_a_question_naming_an_unknown_entity_abstains_and_the_real_one_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECALL_UNKNOWN_TERM_GATE", raising=False)
    store = _store(tmp_path)
    embedder = HashingEmbedder(dim=DIM)

    real = dev_search(store, embedder, REAL, k=2)
    assert real.hits[0].provenance.file == "ar25-lesson1.md"
    assert real.hits[0].verdict == "ok", "the answerable question lost its trusted answer"
    assert real.abstained is False

    near = dev_search(store, embedder, NEAR_MISS, k=2)
    assert near.hits, "the near-miss retrieved nothing, so the test would prove nothing"
    assert all(h.verdict != "ok" for h in near.hits), "a question about QQ01 was trusted"
    assert near.hits[0].verdict == "unknown_term"
    assert near.abstained is True
    assert "QQ01" in near.reason
    assert near.diagnostics.unknown_terms == ("QQ01",)
    assert real.diagnostics.unknown_term_check == "checked"


def _result() -> TrustedResult:
    hit = TrustedHit(
        chunk=Chunk(id="c1", source="s", text="t"),
        cosine=0.9,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(source="s", file="f.md", ord=0, indexed_at=None),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
    )
    return TrustedResult(query="In QQ01?", hits=[hit], abstained=False, reason="",
                         gap_warning=False,
                         staleness=StalenessReport(False, None, None, timedelta(days=1)))


def test_a_store_without_a_vocabulary_is_reported_and_left_alone() -> None:
    check = check_unknown_terms(object(), "In QQ01, what happens?", env={})
    assert check == UnknownTermCheck("unavailable")
    gated = apply_unknown_term_gate(_result(), check)
    assert gated.hits[0].verdict == "ok" and gated.abstained is False
    assert gated.diagnostics.unknown_term_check == "unavailable"


def test_the_gate_can_be_switched_off(tmp_path: Path) -> None:
    store = _store(tmp_path)
    check = check_unknown_terms(store, NEAR_MISS, env={"RECALL_UNKNOWN_TERM_GATE": "0"})
    assert check == UnknownTermCheck("disabled")
    assert apply_unknown_term_gate(_result(), check).hits[0].verdict == "ok"

