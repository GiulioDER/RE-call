"""`recall_related` follows the `caused_by` regression edge between audit records (Task 1.3).

Invariants and the failure each one catches:
- E1 from the golden memos: BUG-002's record says it was caused by BUG-001's fix, so seeding at
  BUG-002 relates BUG-001 and seeding at BUG-001 relates BUG-002. Catches an edge read one way.
- E2 a relation outside the closed set is still refused.
- E3 a `caused_by` neighbour that a later memo supersedes is not returned as current evidence:
  the trust gate judges it against the supersession graph, not an empty one.
- E4 the candidates are read by id from the record chunks, not by scanning every chunk with
  its text.

Red proof, recorded 2026-10-21 on the deployment host inside the test bounds:
- E1 `tests/test_related_caused_by.py::test_caused_by_both_directions`: the reverse clause
  (`seed_record in causes`) removed from `trusted_related`: seeding at BUG-001 relates nothing and
  the assertion fails.
- E2 `...::test_unknown_relation_still_refused`: the refusal text left unchanged when the
  relation was added ("source, ordinal, or supersession"): the message no longer names
  `caused_by` and the `match` assertion fails.
- E3 `...::test_a_superseded_record_is_not_related_as_current` (CCA run 20261026T220018Z (PR 977 audit), 2026-10-26 on the deployment host inside the test bounds): on the pre-fix
  `trusted_related`, whose `caused_by` passed an empty graph: "a superseded record was related as
  current evidence".
- E4 `...::test_caused_by_reads_its_candidates_by_id` (same run): on the pre-fix full scan:
  "caused_by scanned the corpus 2 times for its candidates".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.calibration import Calibration
from recall.lite import LiteStore
from recall.related import trusted_related
from recall.trust_policy import TrustPolicy
from tests.audit_memory_support import EXPECTED, copy_memos, indexed_lite_store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> LiteStore:
    return indexed_lite_store(tmp_path_factory.mktemp("golden"))


def _seed(store: LiteStore, memo: str) -> str:
    return min(c.id for c in store.iter_chunks() if Path(c.source).name == memo)


def _related_records(store: LiteStore, memo: str) -> set[str]:
    result = trusted_related(
        store,
        _seed(store, memo),
        relation="caused_by",
        max_items=50,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=datetime(2026, 10, 21, tzinfo=UTC),
    )
    return {item.chunk.metadata["recall_record"]["id"] for item in result.items}


def test_caused_by_both_directions(store: LiteStore) -> None:
    """E1."""
    bug_001 = EXPECTED["RE-call__0123456789ab__BUG-001.md"]["id"]
    bug_002 = EXPECTED["RE-call__abcdef012345__BUG-002.md"]["id"]
    assert _related_records(store, "RE-call__abcdef012345__BUG-002.md") == {bug_001}
    assert _related_records(store, "RE-call__0123456789ab__BUG-001.md") == {bug_002}
    assert _related_records(store, "RE-call__111111111111__SEC-003.md") == set()


def test_unknown_relation_still_refused(store: LiteStore) -> None:
    """E2."""
    with pytest.raises(ValueError, match="caused_by"):
        trusted_related(store, _seed(store, "RE-call__0123456789ab__BUG-001.md"), relation="causes")  # type: ignore[arg-type]


def test_a_superseded_record_is_not_related_as_current(tmp_path: Path) -> None:
    """E3: the trust gate judges a `caused_by` neighbour against the supersession graph. With an
    empty graph, a record replaced by a later memo came back as trusted, current evidence."""
    folder = copy_memos(tmp_path / "memos")
    (folder / "RE-call__0123456789ab__BUG-001~replaced.md").write_text(
        "---\nname: bug-001-replaced\nvalid_from: 2026-10-01\nsupersedes: RE-call__0123456789ab__BUG-001\n"
        "---\n\n# BUG-001, replaced\n\nA later account of BUG-001.\n",
        encoding="utf-8",
    )
    store = indexed_lite_store(tmp_path, memos=folder)
    bug_001 = EXPECTED["RE-call__0123456789ab__BUG-001.md"]["id"]
    result = trusted_related(
        store,
        _seed(store, "RE-call__abcdef012345__BUG-002.md"),
        relation="caused_by",
        max_items=50,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=datetime(2026, 10, 21, tzinfo=UTC),
    )
    related = {item.chunk.metadata["recall_record"]["id"] for item in result.items}
    assert bug_001 not in related, "a superseded record was related as current evidence"
    assert result.rejected_count >= 1


def test_caused_by_reads_its_candidates_by_id(store: LiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """E4: the candidates are found among the record chunks, read without text, and fetched by
    id; the full scan read every chunk of the corpus, text included, to keep a handful."""
    seed = _seed(store, "RE-call__abcdef012345__BUG-002.md")
    scans: list[int] = []
    real_scan = store.iter_chunks

    def counted(batch_size: int = 1000) -> object:
        scans.append(batch_size)
        return real_scan(batch_size)

    monkeypatch.setattr(store, "iter_chunks", counted)
    # A lite store has no batch fetch by id, so the seed itself is found by the one scan.
    result = trusted_related(
        store,
        seed,
        relation="caused_by",
        max_items=50,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=datetime(2026, 10, 21, tzinfo=UTC),
    )
    assert len(scans) <= 1, f"caused_by scanned the corpus {len(scans)} times for its candidates"
    assert {item.chunk.metadata["recall_record"]["id"] for item in result.items} == {
        EXPECTED["RE-call__0123456789ab__BUG-001.md"]["id"]
    }
