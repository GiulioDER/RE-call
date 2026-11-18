"""Fixes from the CCA deferred pass over run 20261118T205956Z (the pre-0.16.0 Lite audit).

Invariants and the failure each one catches:
- D1 (CODE-001) every refresh handler that builds `reason` from an exception goes through `_fail`,
  so none can leave `error_type` empty and hand the exception's text to an MCP client.
- D2 (CODE-003) the state file's `last_success` no longer carries `rescued`, which nothing reads.
- D3 (CODE-004) the search warning of a failing refresh names `recall lite refresh --check` once.
- D4 (CODE-009) the uncertified status promises the quarter's wait only to a fit that recorded its
  chunk count; one without a count is refitted at its next change, and says so.
- D5 (DOC-001) while an uncertified fit waits, `recall_calibration_status` and `recall doctor` do
  not send the user to `recall_index` / `recall index`, which do not refit it then.
- D6 (STAKES-002) a `recall_calibration_run` whose rescue build fails reports the fit it stored
  instead of raising, so a retry does not store another fit.
- D7 (PERF-002) a kept scope's rows are an int64 array the search indexes with directly.
- D8 (PERF-003) the atomic rescue's parent score reuses the search's product, not a second one.

Red proof: each test's docstring names the baseline (the pre-fix code at 2a281e05b) or the
mutation it was run against and the assertion that failed. The doc-only findings of the same pass
(CODE-005, DOC-003, DOC-004, DOC-005, DOC-007, DOC-008, DOC-010) change prose only and have no
behaviour to test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite.calibration import ensure_calibrated, run_report, status_report
from recall.lite.refresh import RefreshReport, Outcome, client_reason, read_state, refresh
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos
from tests.test_lite_cca_hunt_fixes import _FlatEmbedder, _grow, _uncertified

_REFRESH = Path(__file__).resolve().parents[1] / "recall" / "lite" / "refresh.py"


def _exception_reasons_outside_fail() -> list[int]:
    """Lines of refresh.py that assign `report.reason` an f-string quoting `exc`, outside `_fail`."""
    tree = ast.parse(_REFRESH.read_text(encoding="utf-8"))
    found: list[int] = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef) or func.name == "_fail":
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t for target in node.targets for t in (target.elts if isinstance(target, ast.Tuple) else [target])]
            if not any(isinstance(t, ast.Attribute) and t.attr == "reason" for t in targets):
                continue
            quotes_exc = any(
                isinstance(v, ast.FormattedValue) and any(isinstance(n, ast.Name) and n.id == "exc" for n in ast.walk(v))
                for v in ast.walk(node.value)
            )
            if quotes_exc:
                found.append(node.lineno)
    return found


def test_d1_every_exception_reason_goes_through_fail() -> None:
    """Red against the pre-fix refresh.py (2a281e05b): its three handlers each built `reason` from
    `exc` by hand, so this failed with "a refresh handler builds reason from an exception outside
    _fail" at lines 650, 682 and 704."""
    lines = _exception_reasons_outside_fail()
    assert not lines, f"a refresh handler builds reason from an exception outside _fail: lines {lines}"


def test_d1_fail_withholds_the_exception_text_from_clients() -> None:
    """Red with `report.error_type = ...` deleted from `_fail`: "an MCP client was told the
    exception's text"."""
    from recall.lite.refresh import _fail

    report = RefreshReport(Outcome.REFRESHED, "")
    _fail(report, ValueError("key=sk-secret-123"), "atomic rescue: ")
    assert report.outcome is Outcome.FAILED
    assert "sk-secret-123" in report.reason, "the state file lost the exception's text"
    told = client_reason(report.reason, report.error_type)
    assert "sk-secret-123" not in told, "an MCP client was told the exception's text"
    assert told.startswith("atomic rescue: ValueError"), told


def test_d2_the_last_success_no_longer_records_rescued(tmp_path: Path) -> None:
    """Red against the pre-fix `_finish` (2a281e05b), which wrote `"rescued": rescued`: "the state
    file still records the unread rescued field"."""
    root = tmp_path / "m"
    _write_memos(root, 3)
    store = LiteStore(tmp_path / "memory.db", dim=ContentWordEmbedder().dim)
    try:
        report = refresh(store, [root], ContentWordEmbedder())  # type: ignore[arg-type]
        assert report.outcome is Outcome.REFRESHED, report.reason
        last = read_state(store.path).get("last_success") or {}
        assert last.get("fingerprint"), "the refresh recorded no success"
        assert "rescued" not in last, "the state file still records the unread rescued field"
    finally:
        store.close()


def test_d3_the_refresh_warning_names_the_check_command_once(tmp_path: Path) -> None:
    """Red against the pre-fix `LiteAutoRefresher.warning` (2a281e05b), which appended the pointer
    to a failure `client_reason` had already ended with it: "the warning names the command 2 times"."""
    from recall_mcp.lite_refresh import LiteAutoRefresher

    store = LiteStore(tmp_path / "memory.db", dim=ContentWordEmbedder().dim)
    try:
        refresher = LiteAutoRefresher(store, ContentWordEmbedder(), [tmp_path])  # type: ignore[arg-type]
        refresher._failure = client_reason("ValueError: key=sk-1", "ValueError")
        warning = refresher.warning() or ""
        times = warning.count("recall lite refresh --check")
        assert times == 1, f"the warning names the command {times} times: {warning}"
        # A failure the code wrote itself still gets the pointer.
        refresher._failure = "over the index budget"
        assert (refresher.warning() or "").endswith("Run `recall lite refresh --check` for details.")
    finally:
        store.close()


def test_d4_an_uncertified_fit_without_a_count_says_it_refits_at_the_next_change(tmp_path: Path) -> None:
    """Red against the pre-fix `ensure_calibrated` (2a281e05b) in its first assertion, the literal
    "a quarter" not formatted from `DEFAULT_MAX_CORPUS_DELTA`; and, with the fix's
    `if store.uncertified_fit_chunks() is None` mutated to `if False`: "a fit with no recorded
    count was promised the quarter's wait"."""
    store, _root, _fits = _uncertified(tmp_path)
    try:
        assert "25%" in ensure_calibrated(store, _FlatEmbedder()).reason
        with store._write():  # a fit recorded before the count was kept (before #1373)
            store._conn.execute("DELETE FROM meta WHERE key = 'uncertified_fit_chunks'")
        reason = ensure_calibrated(store, _FlatEmbedder()).reason
        assert "grown or shrunk" not in reason, f"a fit with no recorded count was promised the quarter's wait: {reason}"
        assert "when memory next changes" in reason, reason
    finally:
        store.close()


def test_d5_a_waiting_uncertified_store_is_not_sent_to_index(tmp_path: Path) -> None:
    """Red against the pre-fix `status_report` (2a281e05b): "recall_calibration_status advised
    recall_index, which does not refit a waiting store"; and, with the inspect half applied and the
    doctor branch removed, against the pre-fix doctor: "recall doctor advised recall index"."""
    from recall.doctor import _lite_checks

    store, root, _fits = _uncertified(tmp_path)
    try:
        _grow(store, root, 2, 30)  # 30 to 32 chunks: under a quarter, the fit waits
        message = str(status_report(store)["message"])
        assert "recall_index" not in message, f"recall_calibration_status advised recall_index, which does not refit a waiting store: {message}"
        assert "recall_calibration_run" in message, message
        check = {c.name: c for c in _lite_checks("sqlite:///" + store.path.as_posix(), tenant=store.tenant)}["calibration"]
        assert "recall index" not in (check.fix or ""), f"recall doctor advised recall index: {check.fix}"
        assert "recall_calibration_run" in (check.fix or ""), check.fix
        # Control: past the quarter, indexing refits, so doctor still names it.
        _grow(store, root, 10, 32)
        check = {c.name: c for c in _lite_checks("sqlite:///" + store.path.as_posix(), tenant=store.tenant)}["calibration"]
        assert "recall index" in (check.fix or ""), check.fix
    finally:
        store.close()


def test_d6_a_failed_rescue_build_still_reports_the_stored_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Red against the pre-fix `run_report` (2a281e05b), which let the build's exception escape:
    "a calibration run reported a failure for a fit it had stored"."""
    import recall.lite.atomic as atomic

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full at /secret/path")

    monkeypatch.setattr(atomic, "ensure_rescue_artifact", broken)
    store, _root, fits = _uncertified(tmp_path)
    try:
        try:
            report = run_report(store, _FlatEmbedder())
        except RuntimeError:
            pytest.fail("a calibration run reported a failure for a fit it had stored")
        assert len(store.calibration_rows()) == fits + 1, "the run stored no fit"
        assert report["atomic"] == "atomic rescue artifact not built: RuntimeError", report.get("atomic")
        assert "/secret/path" not in str(report), "the build failure's text reached the run's answer"
    finally:
        store.close()


def _scoped_store(tmp_path: Path) -> LiteStore:
    store = LiteStore(tmp_path / "memory.db", dim=ContentWordEmbedder().dim)
    root = tmp_path / "m"
    _write_memos(root, 6)
    Indexer(store, ContentWordEmbedder(), env={}).index_path(root)
    return store


def test_d7_a_kept_scope_is_an_int64_array(tmp_path: Path) -> None:
    """Red against the pre-fix `_allowed` (2a281e05b), which kept a list and converted it on every
    search: "a kept scope is a list, converted on every scoped search"."""
    store = _scoped_store(tmp_path)
    try:
        source = next(iter(store.iter_chunks())).source
        query = ContentWordEmbedder().embed(["serviced every spring"])[0]
        hits = store.query_dense(query, 3, source=source)
        assert hits and {hit.chunk.source for hit in hits} == {source}
        kept = list(store._load().allowed.values())
        assert len(kept) == 1
        assert isinstance(kept[0], np.ndarray) and kept[0].dtype == np.int64, (
            f"a kept scope is a {type(kept[0]).__name__}, converted on every scoped search"
        )
        assert store.query_dense(query, 3, source=source) == hits, "a kept scope changed the answer"
    finally:
        store.close()


class _CountingVectors(np.ndarray):
    products: list[int] = []

    def __matmul__(self, other: object) -> object:
        _CountingVectors.products.append(1)
        return np.asarray(self) @ other


def test_d8_the_rescue_parent_score_reuses_the_search_product(tmp_path: Path) -> None:
    """Red against the pre-fix `scored_chunk_for_query` (2a281e05b), which multiplied the whole
    corpus again: "scoring the rescued parent multiplied the corpus 2 times". The write control is
    red with the fix's `last[0] == matrix.version` check removed: "a score of the corpus before a
    write was reused after it"."""
    store = _scoped_store(tmp_path)
    try:
        matrix = store._load()
        object.__setattr__(matrix, "vectors", matrix.vectors.view(_CountingVectors))
        _CountingVectors.products = []
        query = ContentWordEmbedder().embed(["serviced every spring"])[0]
        hit = store.query_dense(query, 1)[0]
        scored = store.scored_chunk_for_query(hit.chunk.id, query)
        times = len(_CountingVectors.products)
        assert times == 1, f"scoring the rescued parent multiplied the corpus {times} times"
        assert scored is not None and scored.score == hit.score, "the reused score is not the search's"
        # Another query is not served from the cache.
        other = ContentWordEmbedder().embed(["a different question entirely"])[0]
        assert store.scored_chunk_for_query(hit.chunk.id, other) is not None
        assert len(_CountingVectors.products) == 2, "a different query reused another query's scores"
        # A write moves the corpus version, so the last search's product is not reused after it.
        _write_memos(tmp_path / "m", 1, start=6)
        Indexer(store, ContentWordEmbedder(), env={}).index_path(tmp_path / "m")
        fresh = store._load()
        assert fresh.version != matrix.version, "the write did not move the corpus version"
        object.__setattr__(fresh, "vectors", fresh.vectors.view(_CountingVectors))
        _CountingVectors.products = []
        assert store.scored_chunk_for_query(hit.chunk.id, query) is not None
        assert len(_CountingVectors.products) == 1, "a score of the corpus before a write was reused after it"
    finally:
        store.close()
