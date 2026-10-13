"""The MCP server refreshes a lite store on its own (`recall_mcp.lite_refresh`).

- R9 `before_retrieval` indexes a memo written since the last check before it returns, and a second
  call inside the interval does not look at the folder at all.
- R9b a failing check never fails the retrieval that asked for it.
- R10 the real lifespan runs a catch-up refresh at start, a search refreshes first, and
  `recall_stats` reports the last refresh.
- W1 to W3 a failing refresh is told: a warning on search answers (W2b) and refusals (W2), until a
  refresh succeeds (W1), and a `refresh` check in `recall doctor` (W3).

Red proof, 2026-10-12, on Linux: each mutation alone, the named test failing in its own assertion,
then restored and green (3 passed):
- M9a (R9) `before_retrieval`: the interval check made `if False:`: "a check inside the interval walked the folder"
- M9b (R9) `before_retrieval`: no bounded wait (`self._kick()` without `join`): "the retrieval ran before
  the new memo was indexed"
- M9c (R9b) `before_retrieval`: catches only ValueError: "the check raised into the retrieval: OSError"
- M10a (R10) lifespan: `lite_refresher.start()` made `pass`: "no catch-up refresh at start"
- M10b (R10) `_run_retrieval_tool.execute`: `before_retrieval()` made `pass`: "the search did not refresh first"
- M10c (R10) `recall_stats`: the `refresh` field not filled: "recall_stats has no refresh"

Warning tests (2026-10-13), each mutation alone on Linux, red in its own assertion, then restored:
- MW1 (W1) the failure not cleared on success: "the warning outlived a successful refresh"
- MW2 (W2) the refusal built without the warning: the ToolError payload has no refresh_warning
- MW2b (W2b) the result not given the warning: no refresh_warning in an answered search
- MW3 (W3) the doctor's refresh check not yielded: "no failed-refresh warning"
- MW4 (W4) `_refreshed_elsewhere` never adopts a person's success: "the warning outlived a person's
  successful refresh"
- MW4b (W4b) the warning not cleared on a return to the refreshed snapshot: "the warning outlived the
  folder's return to its last good state"
- MW3b (W3b) the doctor ignores the current snapshot: status `warn` for a failure left behind

Deferred pass (2026-10-13), each mutation alone on Linux, red in its own assertion:
- MD6 (D6, NUM-001) only NaN rejected: an infinite wait is kept (capped, not the default)
- MD7 (D7, ENV-003) an unrecognised switch value turns the refresh on: a refresher was built
- MD8 (D8, PERF-004) the check's snapshot not handed over: "3 walks for one check and the refresh it started"
- MD9 (D9, BUG-006) `stop` does not wait: "stop returned before the running refresh finished"
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import recall_mcp.lite_refresh as auto
from recall.lite import LiteStore
from recall.lite.refresh import Outcome, RefreshReport
from recall_mcp.lite_refresh import LiteAutoRefresher
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_lite_mcp import _serve, lite_env  # noqa: F401  # the fixture is used by name


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[LiteStore, Path, Clock]]:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    root = tmp_path / "memory"
    _write_memos(root, 3)
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as store:
        yield store, root, Clock()


def _refresher(store: LiteStore, root: Path, clock: Clock) -> LiteAutoRefresher:
    return LiteAutoRefresher(store, ContentWordEmbedder(), [root], interval_s=30, wait_s=30, clock=clock)  # type: ignore[arg-type]


def _sources(store: LiteStore) -> set[str]:
    return {Path(c.source).name for c in store.iter_chunks()}


def test_a_new_memo_is_indexed_before_the_retrieval_and_checks_are_debounced(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9."""
    store, root, clock = setup
    refresher = _refresher(store, root, clock)
    refresher.start()
    refresher._kick()[0].join(30)
    assert "memo-002.md" in _sources(store)

    _write_memos(root, 1, start=3)
    refresher.before_retrieval()
    assert "memo-003.md" in _sources(store), "the retrieval ran before the new memo was indexed"

    walks: list[int] = []
    real = auto.snapshot
    monkeypatch.setattr(auto, "snapshot", lambda roots: walks.append(1) or real(roots))
    _write_memos(root, 1, start=4)
    clock.now += 10
    refresher.before_retrieval()
    assert walks == [], "a check inside the interval walked the folder"
    assert "memo-004.md" not in _sources(store)
    clock.now += 30
    refresher.before_retrieval()
    assert walks and "memo-004.md" in _sources(store)


def test_a_failing_check_never_fails_the_retrieval(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9b."""
    store, root, clock = setup

    def broken(roots: object) -> None:
        raise OSError("the disk went away")

    monkeypatch.setattr(auto, "snapshot", broken)
    try:
        _refresher(store, root, clock).before_retrieval()
        raised = ""
    except Exception as exc:  # noqa: BLE001  # asserted below
        raised = f"{type(exc).__name__}: {exc}"
    assert raised == "", f"the check raised into the retrieval: {raised}"


def test_the_server_refreshes_at_start_and_before_a_search(lite_env: dict[str, str]) -> None:  # noqa: F811
    """R10: RECALL_INDEX_ROOT is the folder; the store sits in a subfolder of it."""
    root = Path(lite_env["RECALL_INDEX_ROOT"])
    _write_memos(root, 3)
    env = {**lite_env, "RECALL_LITE_AUTO_REFRESH": "1", "RECALL_LITE_REFRESH_INTERVAL": "0", "RECALL_LITE_REFRESH_WAIT": "30"}

    async def body(state: dict[str, Any], call: Any) -> None:
        refresher = state["lite_refresher"]
        assert refresher is not None, "a lite server with a memo folder has no refresher"
        if refresher._thread is not None:  # never `_kick()` here: that would start the catch-up itself
            refresher._thread.join(30)
        stats = json.loads(await call("recall_stats"))
        assert stats["chunks"] > 0, f"no catch-up refresh at start: {stats}"
        assert (stats.get("refresh") or {}).get("outcome") == "refreshed", f"recall_stats has no refresh: {stats}"

        _write_memos(root, 1, start=3)
        await call("recall_search", query="serviced every spring")
        stats = json.loads(await call("recall_stats"))
        assert stats["refresh"]["indexed_files"] == 1, f"the search did not refresh first: {stats['refresh']}"

    _serve(env, body)


def test_the_server_refresh_applies_the_source_policy(lite_env: dict[str, str], tmp_path: Path) -> None:  # noqa: F811
    """F1 (CCA SEC-001, P1): with RECALL_SOURCE_POLICY_FILE set, the automatic refresh refuses a
    denied memo and redacts an allowed one, as the server's own `recall_index` does.

    Red proof is at the consumer boundary (the real lifespan), because before the fix `refresh` had
    no policy parameter: the pre-fix server stored `closed/b.md` and the address unredacted.
    """
    root = Path(lite_env["RECALL_INDEX_ROOT"])
    (root / "open").mkdir()
    (root / "closed").mkdir()
    (root / "open" / "a.md").write_text("# Pump\n\nWrite to jo.pump@example.com about the pump.\n", encoding="utf-8")
    (root / "closed" / "b.md").write_text("# Valve\n\nThe valve plan is private.\n", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"rules": [{"prefix": "open", "principals": [], "purposes": ["retrieval"], "redactions": ["email"]}]}), encoding="utf-8")
    env = {**lite_env, "RECALL_LITE_AUTO_REFRESH": "1", "RECALL_SOURCE_POLICY_FILE": str(policy)}

    async def body(state: dict[str, Any], call: Any) -> None:
        refresher = state["lite_refresher"]
        if refresher._thread is not None:
            refresher._thread.join(30)
        chunks = list(state["store"].iter_chunks())
        names = {Path(c.source).name for c in chunks}
        assert "b.md" not in names, f"a memo the policy denies was indexed: {sorted(names)}"
        text = " ".join(c.text for c in chunks if Path(c.source).name == "a.md")
        assert "jo.pump@example.com" not in text, f"the address was stored unredacted: {text!r}"
        assert "[REDACTED:email]" in text, f"the allowed memo was not indexed with redaction: {text!r}"

    _serve(env, body)


def test_a_running_refresh_is_not_waited_for_again(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2 (CCA PERF-001): a check that finds a refresh already running does not join it."""
    store, root, clock = setup
    release = threading.Event()

    def slow_refresh(*args: object, **kwargs: object) -> object:
        release.wait(30)
        raise RuntimeError("released")

    monkeypatch.setattr(auto, "refresh", slow_refresh)
    refresher = LiteAutoRefresher(store, ContentWordEmbedder(), [root], interval_s=30, wait_s=5, clock=clock)  # type: ignore[arg-type]
    try:
        refresher.start()
        started = time.monotonic()
        refresher.before_retrieval()
        waited = time.monotonic() - started
    finally:
        release.set()
    assert waited < 2.0, f"the check waited {waited:.1f}s on a refresh it did not start"


def test_a_failing_refresh_backs_off(setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch) -> None:
    """F3 (CCA PERF-002): the same folder failing again is not retried on every interval."""
    store, root, clock = setup
    calls: list[int] = []

    def failing(store_: object, roots: list[Path], *args: object, **kwargs: object) -> RefreshReport:
        calls.append(1)
        return RefreshReport(Outcome.FAILED, "the embedder is down", fingerprint=auto.snapshot(roots).fingerprint)

    monkeypatch.setattr(auto, "refresh", failing)
    refresher = LiteAutoRefresher(store, ContentWordEmbedder(), [root], interval_s=30, wait_s=5, clock=clock)  # type: ignore[arg-type]
    for _ in range(4):
        refresher.before_retrieval()
        clock.now += 31
    assert len(calls) == 2, f"an unchanged failing folder was retried {len(calls)} times in 4 intervals"


def test_a_failing_refresh_warns_until_one_succeeds(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """W1: the refresher reports why refreshing fails, and stops once a refresh succeeds."""
    store, root, clock = setup
    outcomes = [Outcome.FAILED, Outcome.REFRESHED]

    def scripted(store_: object, roots: list[Path], *args: object, **kwargs: object) -> RefreshReport:
        return RefreshReport(outcomes.pop(0), "over the index budget", fingerprint=auto.snapshot(roots).fingerprint + "x")

    monkeypatch.setattr(auto, "refresh", scripted)
    refresher = LiteAutoRefresher(store, ContentWordEmbedder(), [root], interval_s=0, wait_s=5, clock=clock)  # type: ignore[arg-type]
    refresher._kick()[0].join(30)
    warning = refresher.warning()
    assert warning is not None and "over the index budget" in warning, warning
    refresher._kick()[0].join(30)
    assert refresher.warning() is None, "the warning outlived a successful refresh"


def test_the_agent_is_told_when_the_refresh_is_refused(lite_env: dict[str, str]) -> None:  # noqa: F811
    """W2: over the budget, the search answer (or its refusal) carries the reason to the agent."""
    root = Path(lite_env["RECALL_INDEX_ROOT"])
    _write_memos(root, 3)
    env = {**lite_env, "RECALL_LITE_AUTO_REFRESH": "1", "RECALL_INDEX_MAX_FILES": "1"}

    async def body(state: dict[str, Any], call: Any) -> None:
        refresher = state["lite_refresher"]
        if refresher._thread is not None:
            refresher._thread.join(30)
        answer = await call("recall_search", query="serviced every spring")
        assert "refresh_warning" in answer and "RECALL_INDEX_MAX_FILES" in answer, answer

    _serve(env, body)


def test_doctor_reports_a_failed_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W3: `recall doctor` on a lite store shows the last refresh's failure and how to look at it."""
    from recall.doctor import _lite_checks
    from recall.index_budget import IndexBudget
    from recall.lite.refresh import refresh

    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    _write_memos(tmp_path / "memory", 3)
    db = tmp_path / ".recall" / "memory.db"
    with LiteStore(db, dim=DIM) as store:
        refresh(store, [tmp_path / "memory"], ContentWordEmbedder(), budget=IndexBudget(max_files=1))  # type: ignore[arg-type]
    checks = {c.name: c for c in _lite_checks("sqlite:///" + db.as_posix(), tenant="default")}
    found = checks.get("refresh")
    assert found is not None and found.status == "warn", f"no failed-refresh warning: {sorted(checks)}"
    assert "RECALL_INDEX_MAX_FILES" in found.detail and found.fix and "recall lite refresh --check" in found.fix


def test_an_answered_search_carries_the_refresh_warning(lite_env: dict[str, str]) -> None:  # noqa: F811
    """W2b: when search answers (relaxed trust here), the warning is a field of the result."""
    root = Path(lite_env["RECALL_INDEX_ROOT"])
    _write_memos(root, 3)
    env = {**lite_env, "RECALL_LITE_AUTO_REFRESH": "1", "RECALL_INDEX_MAX_FILES": "1", "RECALL_TRUST_MODE": "development"}

    async def body(state: dict[str, Any], call: Any) -> None:
        refresher = state["lite_refresher"]
        if refresher._thread is not None:
            refresher._thread.join(30)
        answer = await call("recall_search", query="serviced every spring")
        assert not answer.startswith("RAISED"), answer
        assert "RECALL_INDEX_MAX_FILES" in (json.loads(answer).get("refresh_warning") or ""), answer

    _serve(env, body)


def test_the_warning_clears_when_a_person_refreshes_the_folder(setup: tuple[LiteStore, Path, Clock]) -> None:
    """W4 (review BUG-002): a failing server refresh, then a person's successful one: the warning goes."""
    from recall.index_budget import IndexBudget
    from recall.lite.refresh import refresh

    store, root, clock = setup
    refresher = LiteAutoRefresher(  # type: ignore[arg-type]
        store, ContentWordEmbedder(), [root], interval_s=30, wait_s=30, budget=IndexBudget(max_files=1), clock=clock
    )
    refresher.start()
    refresher._kick()[0].join(30)
    assert refresher.warning() is not None
    refresh(store, [root], ContentWordEmbedder())  # type: ignore[arg-type]  # a person, unbounded
    clock.now += 31
    refresher.before_retrieval()
    assert refresher.warning() is None, "the warning outlived a person's successful refresh"


def test_the_warning_clears_when_the_folder_returns_to_its_last_good_state(
    setup: tuple[LiteStore, Path, Clock],
) -> None:
    """W4b (review BUG-002): a memo that pushed the folder over budget is removed: the warning goes."""
    from recall.index_budget import IndexBudget

    store, root, clock = setup
    refresher = LiteAutoRefresher(  # type: ignore[arg-type]
        store, ContentWordEmbedder(), [root], interval_s=30, wait_s=30, budget=IndexBudget(max_files=3), clock=clock
    )
    refresher.start()
    refresher._kick()[0].join(30)
    assert refresher.warning() is None
    _write_memos(root, 1, start=3)
    clock.now += 31
    refresher.before_retrieval()
    assert refresher.warning() is not None, "a fourth memo over a three-file budget did not warn"
    (root / "memo-003.md").unlink()
    clock.now += 31
    refresher.before_retrieval()
    assert refresher.warning() is None, "the warning outlived the folder's return to its last good state"


def test_doctor_does_not_report_a_failure_the_folder_has_left_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """W3b (review BUG-003): refreshed, then over budget and refused, then reverted: doctor is ok."""
    from recall.doctor import _lite_checks
    from recall.index_budget import IndexBudget
    from recall.lite.refresh import refresh

    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    root = tmp_path / "memory"
    _write_memos(root, 3)
    db = tmp_path / ".recall" / "memory.db"
    with LiteStore(db, dim=DIM) as store:
        refresh(store, [root], ContentWordEmbedder())  # type: ignore[arg-type]
        _write_memos(root, 1, start=3)
        refresh(store, [root], ContentWordEmbedder(), budget=IndexBudget(max_files=3))  # type: ignore[arg-type]
    (root / "memo-003.md").unlink()
    checks = {c.name: c for c in _lite_checks("sqlite:///" + db.as_posix(), tenant="default")}
    assert checks["refresh"].status == "ok", checks["refresh"]



def test_a_non_finite_wait_falls_back_to_the_default() -> None:
    """D6 (deferred NUM-001): an infinite wait would make the bounded join raise and be skipped."""
    from recall_mcp.lite_refresh import DEFAULT_WAIT_S, _seconds

    assert _seconds({"W": "inf"}, "W", DEFAULT_WAIT_S) == DEFAULT_WAIT_S
    assert _seconds({"W": "1e300"}, "W", DEFAULT_WAIT_S) <= threading.TIMEOUT_MAX


def test_an_unreadable_switch_turns_the_refresh_off(setup: tuple[LiteStore, Path, Clock]) -> None:
    """D7 (deferred ENV-003): "disable" means off, not the default on."""
    store, root, _clock = setup
    env = {"RECALL_INDEX_ROOT": str(root), "RECALL_LITE_AUTO_REFRESH": "disable"}
    assert LiteAutoRefresher.from_env(store, ContentWordEmbedder(), env) is None  # type: ignore[arg-type]


def test_the_check_walks_the_folder_once_for_the_refresh_it_starts(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """D8 (deferred PERF-004): the refresh reuses the check's snapshot; only the in-lane one is new."""
    import recall.lite.refresh as refresh_mod

    store, root, clock = setup
    walks: list[int] = []
    real = refresh_mod.snapshot

    def counting(roots: object) -> object:
        walks.append(1)
        return real(roots)  # type: ignore[arg-type]

    monkeypatch.setattr(refresh_mod, "snapshot", counting)
    monkeypatch.setattr(auto, "snapshot", counting)
    refresher = _refresher(store, root, clock)
    refresher.before_retrieval()
    assert len(walks) == 2, f"{len(walks)} walks for one check and the refresh it started"


def test_stop_waits_for_a_running_refresh_and_starts_no_more(
    setup: tuple[LiteStore, Path, Clock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """D9 (deferred BUG-006): the lifespan stops the refresher before it closes the store."""
    store, root, clock = setup
    release = threading.Event()
    finished: list[int] = []

    def slow(*args: object, **kwargs: object) -> RefreshReport:
        release.wait(5)
        finished.append(1)
        return RefreshReport(Outcome.REFRESHED, "done", fingerprint="x")

    monkeypatch.setattr(auto, "refresh", slow)
    refresher = _refresher(store, root, clock)
    refresher.start()
    threading.Timer(0.3, release.set).start()
    refresher.stop(timeout_s=10)
    assert finished == [1], "stop returned before the running refresh finished"
    thread, started = refresher._kick()
    assert (thread, started) == (None, False), "a stopped refresher started another refresh"
