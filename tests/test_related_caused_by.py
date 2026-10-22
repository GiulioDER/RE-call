"""`recall_related` follows the `caused_by` regression edge between audit records (Task 1.3).

Invariants and the failure each one catches:
- E1 from the golden memos: BUG-002's record says it was caused by BUG-001's fix, so seeding at
  BUG-002 relates BUG-001 and seeding at BUG-001 relates BUG-002. Catches an edge read one way.
- E2 a relation outside the closed set is still refused.

Red proof, recorded 2026-10-21 on the deployment host inside the test bounds:
- E1 `tests/test_related_caused_by.py::test_caused_by_both_directions`: the reverse clause
  (`seed_record in causes`) removed from `trusted_related`: seeding at BUG-001 relates nothing and
  the assertion fails.
- E2 `...::test_unknown_relation_still_refused`: the refusal text left unchanged when the
  relation was added ("source, ordinal, or supersession"): the message no longer names
  `caused_by` and the `match` assertion fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.calibration import Calibration
from recall.lite import LiteStore
from recall.related import trusted_related
from recall.trust_policy import TrustPolicy
from tests.audit_memory_support import EXPECTED, indexed_lite_store


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
