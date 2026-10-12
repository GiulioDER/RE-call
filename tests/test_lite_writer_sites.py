"""The write sites of a lite store take the write lane; reads do not; Postgres is untouched.

- W10 index (MCP `recall_index`), forget, CLI `recall index`, `recall setup --lite` and the explicit
  calibration run each report busy while another process holds their lane.
- W10b `write_session` on any store other than a LiteStore is a no-op that creates no files.
- W11 a search runs while the write lane is held.

Red proof: see the end of this docstring (filled in when run).
"""
from __future__ import annotations

from pathlib import Path

from recall.backends import write_session
from recall.lite import LiteStore
from recall.lite.writer_lock import lanes_dir


def test_write_session_is_a_noop_for_other_stores(tmp_path: Path) -> None:
    """W10b."""

    class NotLite:
        path = tmp_path / "x.db"

    with write_session(NotLite(), "index") as ticket:
        assert ticket is None
    assert not lanes_dir(tmp_path / "x.db").exists(), "a non-lite store got lane files"


def test_write_session_takes_the_lane_for_a_lite_store(tmp_path: Path) -> None:
    """W10 (seam)."""
    with LiteStore(tmp_path / "m.db", dim=8) as store:
        with write_session(store, "index") as ticket:
            assert ticket is not None and ticket.purpose == "index"
