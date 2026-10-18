"""Public calibration writes must refresh the lineage of warmed SQLite stores.

Baseline: PR897 original head 01149368ed0e43618ae3990930bcd4ed20170475.
The nodes below target LiteStore._calibration_state and generation_binding:
test_backdated_write_refreshes_warm_bindings (same and other writers) fails
"backdated calibration left a stale binding"; test_strict_search_observes_changed_identity
fails "cached lineage hid the metadata change". The exact nodes are run red before repairs.

Restack note: test_strict_search_observes_changed_identity wrote a BACKDATED fit with another
model and expected the lineage to change. PR 896's audit (DAT-002) makes the stored embedder
identity follow the newest-dated calibration, the one served, so a backdated fit rightly changes
nothing. The invariant this test protects, that a warm reader's cached lineage notices a real
identity change, is kept with a NEWER-dated fit. Red proof (restack, the testbench, 2026-10-13): keying
`_calibration_state`'s cache on the corpus version alone (calibration row and identity dropped)
fails "cached lineage hid the metadata change". Dropping only the identity stays green: under
DAT-002 the identity changes only with a newer calibration row, which the key already holds.
"""
from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.calibration import auto_calibrate
from recall.trust_policy import TrustFailureCode, TrustRefusal
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _store, _strict

_NEWER = datetime(2026, 10, 7, tzinfo=UTC)
_OLDER = datetime(2026, 10, 6, tzinfo=UTC)
_LATER = datetime(2026, 10, 8, tzinfo=UTC)


@pytest.mark.parametrize("writer_kind", ["same", "other"])
def test_backdated_write_refreshes_warm_bindings(tmp_path: Path, writer_kind: str) -> None:
    with closing(_store(tmp_path, 48)) as reader:
        auto_calibrate(reader, ContentWordEmbedder(), now=_NEWER)
        previous = reader.latest_calibration_json()
        reader.generation_binding()
        writer = reader if writer_kind == "same" else LiteStore(reader.path, dim=DIM)
        try:
            writer.generation_binding()
            auto_calibrate(writer, ContentWordEmbedder(name="another-model"), now=_OLDER)
            assert reader.latest_calibration_json() == previous
            with closing(LiteStore(reader.path, dim=DIM)) as cold:
                expected = cold.generation_binding()
            assert reader.generation_binding() == expected, "backdated calibration left a stale binding"
            assert writer.generation_binding() == expected, "backdated calibration left a stale binding"
        finally:
            if writer is not reader:
                writer.close()


def test_strict_search_observes_changed_identity(tmp_path: Path) -> None:
    with closing(_store(tmp_path, 48)) as reader:
        auto_calibrate(reader, ContentWordEmbedder(), now=_NEWER)
        assert _strict(reader).trust_state == "trusted"
        with closing(LiteStore(reader.path, dim=DIM)) as writer:
            auto_calibrate(writer, ContentWordEmbedder(name="another-model"), now=_LATER)
        try:
            _strict(reader)
        except TrustRefusal as exc:
            assert exc.code is TrustFailureCode.LINEAGE_MISMATCH
        else:
            pytest.fail("cached lineage hid the metadata change")
