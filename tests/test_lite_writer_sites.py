"""The write sites of a lite store take the write lane; reads do not; Postgres is untouched.

- W10 index (MCP `recall_index`), forget, CLI `recall index`, `recall setup --lite` and the explicit
  calibration run each report busy while another process holds their lane.
- W10b `write_session` on any store other than a LiteStore is a no-op that creates no files.
- W11 a search runs while the write lane is held.
- W13 a busy index, ingest or forget releases its idempotency reservation: nothing was written,
  so a retry with the same key must run, not answer `reconciliation_required`.

Red proof, 2026-10-12, each mutation alone, run on Linux, failing in the named assertion, then
restored byte for byte and green (W2 needed equal polling first: M2 survived the original test):
- M14 (W10 index) `index_memory`: body called without `write_session`; `test_index_and_forget_report_busy_while_the_lane_is_held`: "AssertionError: {"
- M15 (W10 forget) `forget_memory`: body called without `write_session`; `test_index_and_forget_report_busy_while_the_lane_is_held`: "AssertionError: {"
- M16 (W10 CLI) CLI `recall index`: the `write_session` block replaced by `if True:`; `test_cli_index_reports_busy`: "AssertionError: indexed 2 chunks from 2 files"
- M17 (W10 setup) `run_lite_setup`: the `lite_lane` block replaced by `if True:`; `test_setup_reports_busy`: "Failed: DID NOT RAISE LiteWriterBusy"
- M18 (W10 calibrate) `generation_admin`: the refit-lane block replaced by `if True:`; `test_calibration_run_reports_busy_on_the_refit_lane`: "AssertionError: {"
- M19 (W10b) `write_session`: takes a lane for any store with a `path`; `test_write_session_is_a_noop_for_other_stores`: "AssertionError: assert LaneTicket(lane='write', purpose='index', pid=328370, host='<host>', seq=1, acquired_at..."
- M20 (ingest routing) ingest: `lane_purpose="ingest"` dropped from the legacy call; `test_a_legacy_server_indexes_into_the_legacy_table`: "AssertionError: an ingest took the lite write lane as a plain index, so a queued refresh cannot tell them apar..."
- M21 (W11) `LiteStore.query_dense` made to take the write lane (control for W11); `test_index_and_forget_report_busy_while_the_lane_is_held`: "AssertionError: a search failed behind the write lane: RAISED LiteWriterBusy: the lite store's write lane is b..."
- W13 pre-fix server (only IndexPreflightError released, and only by index): "a busy refusal kept its
  idempotency reservation: ['k-index', 'k-ingest', 'k-forget'] (released: [])".
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from recall.backends import write_session
from recall.lite import LiteStore, LiteWriterBusy
from recall.lite.writer_lock import lanes_dir
from recall_mcp import server as srv
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos
from tests.test_lite_mcp import _env, _serve
from tests.test_lite_writer_lock import _at, _spawn, _wait_for


@pytest.fixture
def lite_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The lite MCP environment `tests/test_lite_mcp.py` uses, with the content-word embedder."""
    for name in ("RECALL_SERVING_DSN", "RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_TRUST_MODE",
                 "RECALL_PINNED_GENERATION_ID", "RECALL_MCP_TOOLS", "RECALL_DECISION_LEDGER"):
        monkeypatch.delenv(name, raising=False)
    env = _env(tmp_path)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(srv, "make_embedder", lambda *a, **k: ContentWordEmbedder())
    return env


def _hold(db: Path, tmp_path: Path, *, lane: str = "write", purpose: str = "refresh") -> subprocess.Popen[bytes]:
    """Another process holding `lane` of `db` for 30 s (killed by the caller)."""
    log = tmp_path / f"hold-{lane}.log"
    proc = _spawn(db, log, "holder", lane=lane, purpose=purpose, hold_s=30)
    _wait_for(lambda: _at(log, "holder", "acquired") is not None)
    return proc


def _db(env: dict[str, str]) -> Path:
    return Path(env["RECALL_DSN"][len("sqlite:///"):])


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


def test_index_and_forget_report_busy_while_the_lane_is_held(
    lite_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W10 (MCP index, forget) and W11 (a search runs meanwhile)."""
    _write_memos(tmp_path / "mem", 4)

    async def first(state: dict[str, Any], call: Any) -> None:
        await call("recall_index", path=str(tmp_path / "mem"))

    _serve(lite_env, first)
    monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", "0")
    holder = _hold(_db(lite_env), tmp_path)
    try:
        async def body(state: dict[str, Any], call: Any) -> None:
            indexed = await call("recall_index", path=str(tmp_path / "mem"))
            assert "busy" in indexed and "refresh" in indexed and "Retry" in indexed, indexed
            forgot = await call("recall_forget", sources=["memo-001.md"])
            assert "busy" in forgot and "refresh" in forgot, forgot
            assert state["store"].count() == 4, "a busy forget deleted something"
            # Four memos are under the calibration floor, so strict trust would refuse for reasons
            # that have nothing to do with the lane; development mode lets the search answer.
            found = await call("recall_search", query="who services the memo")
            assert "busy" not in found and "memo-0" in found, f"a search failed behind the write lane: {found}"

        monkeypatch.setenv("RECALL_TRUST_MODE", "development")
        _serve({**lite_env, "RECALL_TRUST_MODE": "development"}, body)
    finally:
        holder.kill()
        holder.wait(10)


def test_cli_index_reports_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W10 (CLI)."""
    from tests.test_lite_cli import _MEMOS, _dsn, _memos, _run

    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", "0")
    folder = _memos(tmp_path / "mem", _MEMOS)
    holder = _hold(tmp_path / "store" / "memory.db", tmp_path)
    try:
        out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "index", str(folder)])
        assert "busy" in out and "refresh" in out, out
        assert "indexed" not in out, f"the CLI indexed while another process held the lane: {out}"
    finally:
        holder.kill()
        holder.wait(10)


def test_setup_reports_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W10 (setup --lite)."""
    from recall.lite.setup import default_store_path, run_lite_setup

    project = tmp_path / "proj"
    project.mkdir()
    _write_memos(project / "notes", 3)
    monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", "0")
    holder = _hold(default_store_path(project.resolve()), tmp_path)
    try:
        with pytest.raises(LiteWriterBusy):
            run_lite_setup(project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(),
                           embedder_name="content-words", register=False)
    finally:
        holder.kill()
        holder.wait(10)


def test_calibration_run_reports_busy_on_the_refit_lane(
    lite_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W10 (the explicit calibration run takes the refit lane)."""
    monkeypatch.setenv("RECALL_LITE_REFIT_WAIT", "0")
    holder = _hold(_db(lite_env), tmp_path, lane="refit", purpose="calibrate")
    try:
        async def body(state: dict[str, Any], call: Any) -> None:
            ran = await call("recall_calibration_run")
            assert "busy" in ran and "refit" in ran, ran

        _serve(lite_env, body)
    finally:
        holder.kill()
        holder.wait(10)


def test_a_busy_write_releases_its_idempotency_reservation(
    lite_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W13: the busy refusal comes before the lane is held, so before any write. A reservation kept
    past it turns the retry the message invites into `reconciliation_required` under a limiter."""
    import base64

    _write_memos(tmp_path / "mem", 2)
    released: list[str | None] = []
    real = srv._release_mutation_reservation

    async def record(state: dict[str, object], tenant: str, idempotency_key: str | None) -> None:
        released.append(idempotency_key)
        await real(state, tenant, idempotency_key)

    monkeypatch.setattr(srv, "_release_mutation_reservation", record)
    monkeypatch.setenv("RECALL_LITE_WRITE_WAIT", "0")
    holder = _hold(_db(lite_env), tmp_path)
    try:
        async def body(state: dict[str, Any], call: Any) -> None:
            note = base64.b64encode(b"# A note\n\nThe rotor is serviced by the night team.\n").decode()
            outcomes = [
                await call("recall_index", path=str(tmp_path / "mem"), idempotency_key="k-index"),
                await call("recall_ingest", files=[{"name": "note.md", "content_b64": note}], idempotency_key="k-ingest"),
                await call("recall_forget", sources=["memo-001.md"], idempotency_key="k-forget"),
            ]
            assert all("busy" in out for out in outcomes), outcomes

        _serve(lite_env, body)
    finally:
        holder.kill()
        holder.wait(10)
    kept = [key for key in ("k-index", "k-ingest", "k-forget") if key not in released]
    assert not kept, f"a busy refusal kept its idempotency reservation: {kept} (released: {released})"
