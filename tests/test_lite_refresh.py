"""`recall.lite.refresh` and `recall lite refresh`: the lite store follows its memo folder.

- R1 an unchanged folder short-circuits: no lane is taken, nothing is indexed.
- R2 an added memo, and an edit that keeps the file's size, are each refreshed.
- R3 a deleted memo is pruned and counted.
- R4 a refresh already queued for the store makes this one `coalesced`, at once.
- R5 a held lane with `wait_s=0` gives `busy`, naming the holder.
- R6 a cold start (below the calibration floor) is `refreshed`, with the calibration's own reason.
- R7 `check` writes nothing and tells current from changed.
- R8 a failed refresh keeps the last success.
- R11 the CLI's exit codes: 0 refreshed or unchanged, 3 busy; `--check` 0 current, 1 changed.
- R12 a memo folder that contains the store does not refresh forever on the store's own files.
- R13 a missing memo folder is `failed`, takes no lane and prunes nothing; R13b the same when it
  disappears while the refresh queues.
- R14 a corrupt state file reads as empty: one refresh, then unchanged.

Red proof, 2026-10-12, on Linux: each mutation alone, the named test failing in its own assertion,
then restored and green (12 passed):
- M1 (R1) `refresh`: the unchanged short-circuit disabled (`False and not force`): "an unchanged refresh took the write lane"
- M2 (R2) `snapshot`: size only, no `st_mtime_ns`: the same-size edit came back `unchanged`
- M3 (R3) `_refresh_locked`: `report.deleted += 0`: deleted 0, not 1
- M4 (R4) `refresh`: `coalesce=False`: `Outcome.BUSY ... refresh waited 2.0s`, not coalesced
- M5 (R5) `refresh`: the busy holder dropped: `AssertionError: None`
- M6 (R6) `_refresh_locked`: a `missing` calibration reported as `failed`: `Outcome.FAILED`
- M7 (R7) `check`: writes the state file back: "check wrote something"
- M8 (R8) `_refresh_locked`: a failure recorded as the last success: "a failed refresh replaced the last success"
- M11 (R11) `lite_cmd`: `raise SystemExit(EXIT_BUSY)` made `pass`: exit 0, not 3
- M12 (R12) `recall.index._DEFAULT_INDEX_EXCLUDED_SUFFIXES` without `.json`: "the store's own files moved the snapshot"
- M13 (R13) `refresh`: the missing-folder check disabled: "a refresh of a missing folder took the write lane"
  (it survived the first R13, which only asserted the outcome; the in-lane check of R13b caught it).
  Superseded by the deferred pass (BUG-004): the check is now only under the lane, so R13 no longer
  asserts that no lane is taken, and M13b covers both tests.
- M13b (R13b) `_refresh_locked`: the in-lane missing-folder check disabled: outcome `refreshed` over 0 files
- M14 (R14) `read_state`: `ValueError` not caught: "raised JSONDecodeError"

Budget tests B1 to B5 (CCA SEC-002, 2026-10-13), each mutation alone on Linux, red in its own
assertion, then restored and green:
- MB1 (B1, B1b) the budget check under the lane off: outcome `refreshed` (B1b: over 4 files). An
  earlier version also refused before the lane; review BUG-001 removed it (a state write outside
  the lane can undo a concurrent success).
- MB2 (B2) `budget_refusal` made `>=`: a folder exactly at the budget was refused
- MB3 (B3) the snapshot counts 0 bytes: "3 file(s) checked", not refused
- MB4 (B4) the server refresher passes `budget=None`: "the server indexed a folder over its budget"
- MB5 (B5) the CLI warning dropped: no "warning:" on stderr
- MB6 (B6) the minimum of 1 not enforced: "a zero file budget was accepted"
- MB7 (B7) a malformed budget stops the refresh again (SystemExit): exit -1, no warning
- MB8 (B8) the budget line rendered only when over: "within it" missing from --check

Deferred pass (CCA run 20261013T090136Z, 2026-10-13), each mutation alone on Linux, red in its own
assertion, then restored:
- MD1 (D1, STAKES-003) `refresh_purpose` returns the one shared purpose: "coalesced into another
  folder's refresh"
- MD2 (D2, BUG-004) a busy refresh written again: "a busy refresh rewrote the state file outside the lane"
- MR4 (R4, BUG-004) a coalesced refresh written again: "a coalesced refresh, holding no lane, wrote the
  state file"
- MD3 (D3, PERF-005) the unchanged check hashes the corpus again: "the unchanged check loaded the
  whole corpus"
- MD3b (D3b, PERF-005) the counter not compared: a deleted source reads `unchanged`
- MD4 (D4, BUG-007) the racy mark off: the success recorded clean
- MD4b (D4b, BUG-007 as corrected at L5.5) the window without `abs`: a future-dated memo marks the
  success racy
- MD5 (D5, STAKES-005) setup without the context policy: "the first refresh after setup re-embedded 3 memo(s)"
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import recall.lite.refresh as refresh_mod
from recall.cli import main as cli_main
from recall.lite import LiteStore
from recall.lite.refresh import Outcome, check, read_state, refresh, state_path
from recall.lite.writer_lock import lite_lane_status
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos
from tests.test_lite_writer_lock import _at, _spawn, _wait_for


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    _write_memos(tmp_path / "memory", 3)
    return tmp_path


@pytest.fixture
def store(project: Path) -> Iterator[LiteStore]:
    with LiteStore(project / ".recall" / "memory.db", dim=DIM) as opened:
        yield opened


def _run(store: LiteStore, project: Path, **kwargs: object) -> refresh_mod.RefreshReport:
    return refresh(store, [project / "memory"], ContentWordEmbedder(), **kwargs)  # type: ignore[arg-type]


def _holder(db: Path, tmp_path: Path, tag: str = "holder", purpose: str = "index") -> tuple[subprocess.Popen[bytes], Path]:
    """Another process holding the write lane until the returned release file exists."""
    log, release = tmp_path / f"{tag}.log", tmp_path / f"{tag}.release"
    proc = _spawn(db, log, tag, purpose=purpose, hold_s=60, release=release)
    _wait_for(lambda: _at(log, tag, "acquired") is not None)
    return proc, release


def _files(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_an_unchanged_folder_takes_no_lane(store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1."""
    first = _run(store, project)
    assert first.outcome is Outcome.REFRESHED, first.reason
    chunks = store.count()

    def no_lane(*args: object, **kwargs: object) -> None:
        raise AssertionError("an unchanged refresh took the write lane")

    monkeypatch.setattr(refresh_mod, "lite_lane", no_lane)
    second = _run(store, project)
    assert second.outcome is Outcome.UNCHANGED, f"{second.outcome}: {second.reason}"
    assert store.count() == chunks


def test_an_added_memo_and_a_same_size_edit_are_refreshed(store: LiteStore, project: Path) -> None:
    """R2."""
    _run(store, project)
    _write_memos(project / "memory", 1, start=3)
    added = _run(store, project)
    assert (added.outcome, added.indexed_files) == (Outcome.REFRESHED, 1), added.to_dict()
    assert any("memo-003.md" in c.source for c in store.iter_chunks())

    memo = project / "memory" / "memo-000.md"
    text = memo.read_text(encoding="utf-8")
    edited = text.replace("spring", "autumn")  # same length: only the mtime can tell
    assert len(edited) == len(text) and edited != text
    before = memo.stat().st_mtime_ns
    memo.write_text(edited, encoding="utf-8")
    os.utime(memo, ns=(before + 1_000_000, before + 1_000_000))
    after = _run(store, project)
    assert (after.outcome, after.indexed_files) == (Outcome.REFRESHED, 1), after.to_dict()


def test_a_deleted_memo_is_pruned(store: LiteStore, project: Path) -> None:
    """R3."""
    _run(store, project)
    (project / "memory" / "memo-001.md").unlink()
    report = _run(store, project)
    assert (report.outcome, report.deleted) == (Outcome.REFRESHED, 1), report.to_dict()
    assert not any("memo-001.md" in c.source for c in store.iter_chunks())


def test_a_queued_refresh_coalesces_this_one(store: LiteStore, project: Path, tmp_path: Path) -> None:
    """R4: a real holder, a real queued refresh of the SAME folder in another process, then this one.

    Changed by the deferred pass: the lane purpose names the folders (STAKES-003), and a coalesced
    refresh writes nothing, because it holds no lane (BUG-004).
    """
    from recall.lite.refresh import refresh_purpose

    db = Path(store.path)
    purpose = refresh_purpose([project / "memory"])
    holder, release = _holder(db, tmp_path)
    queued = _spawn(db, tmp_path / "q.log", "queued", purpose=purpose, wait_s=60)
    try:
        _wait_for(lambda: any(w.get("purpose") == purpose for w in lite_lane_status(db).waiters))
        started = time.monotonic()
        report = _run(store, project, wait_s=2.0)
        assert report.outcome is Outcome.COALESCED, f"{report.outcome}: {report.reason}"
        assert time.monotonic() - started < 1.5, "a coalesced refresh waited for the lane"
        assert read_state(db) == {}, "a coalesced refresh, holding no lane, wrote the state file"
    finally:
        release.touch()
        holder.wait(30)
        queued.wait(30)


def test_a_held_lane_gives_busy_and_names_the_holder(store: LiteStore, project: Path, tmp_path: Path) -> None:
    """R5."""
    holder, release = _holder(Path(store.path), tmp_path)
    try:
        report = _run(store, project, wait_s=0)
    finally:
        release.touch()
        holder.wait(30)
    assert report.outcome is Outcome.BUSY, f"{report.outcome}: {report.reason}"
    assert report.holder is not None and report.holder.get("purpose") == "index", report.holder


def test_a_cold_start_is_refreshed_with_the_calibration_reason(store: LiteStore, project: Path) -> None:
    """R6: three memos are below the calibration floor."""
    report = _run(store, project)
    assert report.outcome is Outcome.REFRESHED, f"{report.outcome}: {report.reason}"
    assert report.calibration == "missing"
    assert report.calibration_reason.startswith("Trusted answers turn on once memory holds enough text"), report.calibration_reason
    assert report.calibration_reason in report.render()
    assert read_state(store.path)["last"]["calibration_reason"] == report.calibration_reason


def test_check_writes_nothing_and_tells_current_from_changed(store: LiteStore, project: Path) -> None:
    """R7."""
    db, roots = Path(store.path), [project / "memory"]
    _run(store, project)
    before = _files(project)
    current = check(db, roots)
    assert _files(project) == before, "check wrote something"
    assert current.current, current.render()
    _write_memos(project / "memory", 1, start=3)
    before = _files(project)
    changed = check(db, roots)
    assert _files(project) == before, "check wrote something"
    assert not changed.current, changed.render()
    assert "changed since the last refresh" in changed.render()
    assert "Trusted answers turn on" in changed.render(), "the cold start reason was not reported"


def test_a_failure_keeps_the_last_success(store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R8."""
    _run(store, project)
    success = read_state(store.path)["last_success"]
    _write_memos(project / "memory", 1, start=3)

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the embedder is down")

    monkeypatch.setattr("recall.index.Indexer.index_path", broken)
    report = _run(store, project)
    assert report.outcome is Outcome.FAILED and "the embedder is down" in report.reason
    state = read_state(store.path)
    assert state["last"]["outcome"] == "failed"
    assert state["last_success"] == success, "a failed refresh replaced the last success"


def test_the_cli_exit_codes(project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """R11, with the hashing embedder the CLI can resolve by name."""
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    base = ["lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing"]

    def code(*extra: str) -> int:
        try:
            cli_main([*base, *extra])
        except SystemExit as exc:
            return int(exc.code or 0) if not isinstance(exc.code, str) else -1
        return 0

    assert code() == 0, capsys.readouterr()
    assert code() == 0 and "unchanged" in capsys.readouterr().out
    assert code("--check") == 0
    _write_memos(project / "memory", 1, start=3)
    assert code("--check") == 1
    holder, release = _holder(db, tmp_path)
    try:
        assert code("--wait", "0") == 3, capsys.readouterr()
    finally:
        release.touch()
        holder.wait(30)


def test_a_folder_holding_the_store_does_not_refresh_forever(project: Path) -> None:
    """R12: the store, its WAL, its lanes and the state file all live inside the memo folder."""
    root = project / "memory"
    with LiteStore(root / "memory.db", dim=DIM) as inside:
        first = refresh(inside, [root], ContentWordEmbedder())  # type: ignore[arg-type]
        assert first.outcome is Outcome.REFRESHED, first.reason
        assert state_path(inside.path).exists()
        second = refresh(inside, [root], ContentWordEmbedder())  # type: ignore[arg-type]
        assert second.outcome is Outcome.UNCHANGED, f"the store's own files moved the snapshot: {second.to_dict()}"


def test_a_missing_folder_fails_and_prunes_nothing(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13. Changed by the deferred pass (BUG-004): refused under the lane, where it is recorded."""
    _run(store, project)
    chunks = store.count()
    (project / "memory").rename(project / "memory-moved")
    report = _run(store, project)
    assert report.outcome is Outcome.FAILED and "memory folder not found" in report.reason, report.to_dict()
    assert store.count() == chunks
    assert read_state(store.path)["last"]["outcome"] == "failed"


def test_a_corrupt_state_file_costs_one_refresh(store: LiteStore, project: Path) -> None:
    """R14."""
    _run(store, project)
    state_path(store.path).write_text("{not json", encoding="utf-8")
    try:
        outcome: object = _run(store, project).outcome
    except ValueError as exc:  # asserted below, so a crash on the file reads as this test's failure
        outcome = f"raised {type(exc).__name__}: {exc}"
    assert outcome is Outcome.REFRESHED, outcome
    assert _run(store, project).outcome is Outcome.UNCHANGED
    assert json.loads(state_path(store.path).read_text(encoding="utf-8"))["schema"] == 1


def test_a_folder_gone_while_queued_fails_and_prunes_nothing(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13b: the folder is there for the first snapshot and gone by the one taken inside the lane."""
    _run(store, project)
    chunks = store.count()
    _write_memos(project / "memory", 1, start=3)  # so the first snapshot sees a change
    real = refresh_mod.snapshot
    calls: list[int] = []

    def moving(roots: object) -> refresh_mod.Snapshot:
        calls.append(1)
        if len(calls) == 2:
            (project / "memory").rename(project / "memory-moved")
        return real(roots)  # type: ignore[arg-type]

    monkeypatch.setattr(refresh_mod, "snapshot", moving)
    report = _run(store, project)
    assert report.outcome is Outcome.FAILED and "memory folder not found" in report.reason, report.to_dict()
    assert store.count() == chunks


def test_a_symlinked_memo_keeps_the_key_every_indexer_uses(store: LiteStore, project: Path) -> None:
    """F4 (CCA BUG-001): a memo that is a symlink is refreshed under the link's own path."""
    root = project / "memory"
    # Inside the root (confinement keeps it) but in a hidden folder the walk skips, so the link is
    # the only spelling the walk produces.
    (root / ".attic").mkdir()
    target = root / ".attic" / "elsewhere.md"
    target.write_text("# Boiler\n\nThe boiler is drained every winter by the north crew.\n", encoding="utf-8")
    link = root / "boiler.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this filesystem cannot create a symlink")
    _run(store, project)
    target.write_text("# Boiler\n\nThe boiler is now drained every autumn by the south crew.\n", encoding="utf-8")
    _run(store, project)
    texts = {c.source: c.text for c in store.iter_chunks() if Path(c.source).name in {"boiler.md", "elsewhere.md"}}
    assert set(map(lambda s: Path(s).name, texts)) == {"boiler.md"}, f"rows keyed by the link's target: {sorted(texts)}"
    assert all("autumn" in t for t in texts.values()), f"the link's rows were not refreshed: {texts}"


def test_a_failure_while_counting_is_still_recorded(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5 (CCA BUG-002): the store failing again inside the failure handler does not lose FAILED."""

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr("recall.index.Indexer.index_path", broken)
    monkeypatch.setattr(LiteStore, "count", broken)
    try:
        outcome: object = _run(store, project).outcome
    except RuntimeError as exc:  # asserted below, so a raise reads as this test's failure
        outcome = f"raised {exc}"
    assert outcome is Outcome.FAILED, outcome
    assert read_state(store.path)["last"]["outcome"] == "failed"


def test_a_locked_state_file_does_not_lose_a_refresh(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F6 (CCA BUG-003): a replace refused twice (a reader holding the file on Windows) is retried."""
    real = refresh_mod.atomic_write_bytes
    refusals: list[int] = []

    def flaky(path: Path, data: bytes) -> None:
        if len(refusals) < 2:
            refusals.append(1)
            raise PermissionError(13, "The process cannot access the file")
        real(path, data)

    monkeypatch.setattr(refresh_mod, "atomic_write_bytes", flaky)
    try:
        outcome: object = _run(store, project).outcome
    except PermissionError as exc:  # asserted below
        outcome = f"raised {exc}"
    assert outcome is Outcome.REFRESHED, outcome
    assert read_state(store.path).get("last_success"), "the success was not recorded after the retries"


def _cli(*argv: str) -> int:
    try:
        cli_main(list(argv))
    except SystemExit as exc:
        return int(exc.code or 0) if not isinstance(exc.code, str) else -1
    return 0


def test_allow_prune_clears_a_tripped_guard(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """F7 (CCA STAKES-002): a deletion the guard refuses names the flag, and the flag accepts it."""
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    _write_memos(project / "memory", 4, start=3)  # 7 memos: the guard needs at least 5 sources
    base = ["lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing"]
    assert _cli(*base) == 0, capsys.readouterr()
    for i in range(5):
        (project / "memory" / f"memo-{i:03d}.md").unlink()
    assert _cli(*base) == 1
    assert "recall lite refresh --allow-prune" in capsys.readouterr().out
    assert _cli(*base, "--allow-prune") == 0, capsys.readouterr()
    assert read_state(db)["last"]["deleted"] == 5


def test_the_cli_follows_the_folder_the_store_was_refreshed_from(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F8 (CCA ENV-002): with no --memory and no RECALL_INDEX_ROOT, the recorded folder is used."""
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    assert _cli("lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing") == 0
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("RECALL_INDEX_ROOT", raising=False)
    capsys.readouterr()
    assert _cli("lite", "refresh", "--store", str(db), "--embedder", "hashing") == 0, capsys.readouterr()
    assert "unchanged" in capsys.readouterr().out


def _policy_file(tmp_path: Path, prefix: str) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({"rules": [{"prefix": prefix, "principals": [], "purposes": ["retrieval"], "redactions": []}]}),
        encoding="utf-8",
    )
    return path


def test_the_cli_refresh_decides_the_policy_as_the_server_does(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F9 (CCA FIX-001, architect REVISE): under a policy whose rule allows the usual `retrieval`
    purpose, a CLI refresh indexes the allowed memo. Red against the first FIX-001, whose CLI used
    `recall index`'s context (purpose `indexing`) and so denied every memo."""
    _write_memos(project / "memory" / "notes", 1)
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(_policy_file(tmp_path, "notes")))
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    assert _cli("lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing") == 0, capsys.readouterr()
    last = read_state(db)["last"]
    assert last.get("denied", 0) == 3, f"the three memos outside `notes` must be denied: {last}"
    with LiteStore(db, dim=64) as opened:
        sources = {c.source for c in opened.iter_chunks()}
    assert sources and all("notes" in Path(s).parts for s in sources), f"not the allowed memo alone: {sorted(sources)}"


def test_a_refresh_the_policy_denies_entirely_is_a_failure(
    store: LiteStore, project: Path, tmp_path: Path
) -> None:
    """F10 (CCA FIX-001, architect REVISE): nothing indexed is never a success that consumes changes."""
    from recall.lite.refresh import refresh_access_context
    from recall.security_policy import load_source_policy

    policy = load_source_policy({"RECALL_SOURCE_POLICY_FILE": str(_policy_file(tmp_path, "nowhere"))})
    report = _run(store, project, security_policy=policy, security_context=refresh_access_context(store.tenant))
    assert report.outcome is Outcome.FAILED, f"{report.outcome}: {report.reason}"
    assert "denies all 3" in report.reason, report.reason
    assert read_state(store.path).get("last_success") is None


def test_an_over_budget_folder_is_refused_and_nothing_is_indexed(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1 (CCA SEC-002): three memos against a two-file budget: FAILED, no row written.

    Refused under the lane, not before it, so the refusal is written safely (review BUG-001).
    """
    from recall.index_budget import IndexBudget

    report = _run(store, project, budget=IndexBudget(max_files=2))
    assert report.outcome is Outcome.FAILED, f"{report.outcome}: {report.reason}"
    assert "RECALL_INDEX_MAX_FILES" in report.reason and "3 candidate file(s) > limit 2" in report.reason, report.reason
    assert store.count() == 0, "an over-budget folder was indexed"
    assert read_state(store.path).get("last_success") is None


def test_a_folder_exactly_at_the_budget_is_refreshed(store: LiteStore, project: Path) -> None:
    """B2 (CCA SEC-002): the limit is inclusive, as `recall_index`'s is."""
    from recall.index_budget import IndexBudget

    report = _run(store, project, budget=IndexBudget(max_files=3))
    assert report.outcome is Outcome.REFRESHED, f"{report.outcome}: {report.reason}"


def test_the_byte_budget_counts_the_folder(store: LiteStore, project: Path) -> None:
    """B3 (CCA SEC-002): the bytes the snapshot stats are what the byte budget sees."""
    from recall.index_budget import IndexBudget

    report = _run(store, project, budget=IndexBudget(max_bytes=10))
    assert report.outcome is Outcome.FAILED and "RECALL_INDEX_MAX_BYTES" in report.reason, report.reason
    assert store.count() == 0


def test_the_server_refresh_takes_the_budget_from_its_environment(store: LiteStore, project: Path) -> None:
    """B4 (CCA SEC-002): `from_env` reads RECALL_INDEX_MAX_FILES, so an unattended refresh is bounded."""
    from recall_mcp.lite_refresh import LiteAutoRefresher

    refresher = LiteAutoRefresher.from_env(
        store, ContentWordEmbedder(), {"RECALL_INDEX_ROOT": str(project / "memory"), "RECALL_INDEX_MAX_FILES": "2"}  # type: ignore[arg-type]
    )
    assert refresher is not None
    refresher.start()
    refresher._kick()[0].join(30)
    assert store.count() == 0, "the server indexed a folder over its budget"
    assert "RECALL_INDEX_MAX_FILES" in read_state(store.path)["last"]["reason"]


def test_a_person_may_refresh_past_the_budget_with_a_warning(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B5 (CCA SEC-002): the CLI indexes an over-budget folder, warns on stderr, and --check says so."""
    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "2")
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    base = ["lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing"]
    assert _cli(*base) == 0
    captured = capsys.readouterr()
    assert "warning:" in captured.err and "RECALL_INDEX_MAX_FILES" in captured.err, captured.err
    assert read_state(db)["last"]["outcome"] == "refreshed"
    _cli(*base, "--check")
    assert "budget       the memory folder" in capsys.readouterr().out


def test_a_folder_grown_past_the_budget_while_queued_is_refused(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1b (CCA SEC-002): within budget at the first snapshot, over it inside the lane: refused."""
    from recall.index_budget import IndexBudget

    real = refresh_mod.snapshot
    calls: list[int] = []

    def growing(roots: object) -> refresh_mod.Snapshot:
        calls.append(1)
        if len(calls) == 2:
            _write_memos(project / "memory", 1, start=3)
        return real(roots)  # type: ignore[arg-type]

    monkeypatch.setattr(refresh_mod, "snapshot", growing)
    report = _run(store, project, budget=IndexBudget(max_files=3))
    assert report.outcome is Outcome.FAILED and "4 candidate file(s) > limit 3" in report.reason, report.to_dict()
    assert store.count() == 0


def test_a_budget_below_one_is_refused() -> None:
    """B6 (review BUG-004): as the server's settings refuse it, so does the budget."""
    from recall.index_budget import budget_from_env

    try:
        budget_from_env({"RECALL_INDEX_MAX_FILES": "0"})
        raised = ""
    except ValueError as exc:
        raised = str(exc)
    assert "at least 1" in raised, f"a zero file budget was accepted: {raised!r}"


def test_a_malformed_budget_never_stops_a_persons_refresh(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B7 (review BUG-005): the manual refresh warns about the setting and refreshes anyway."""
    monkeypatch.setenv("RECALL_INDEX_MAX_FILES", "lots")
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    code = _cli("lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--embedder", "hashing")
    captured = capsys.readouterr()
    assert code == 0 and "RECALL_INDEX_MAX_FILES" in captured.err, (code, captured.err)


def test_check_shows_the_budget_when_within_it(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """B8 (review BUG-005): --check always states the limits, not only when over them."""
    db = project / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    _cli("lite", "refresh", "--store", str(db), "--memory", str(project / "memory"), "--check")
    assert "budget       within it: 3 of 2000 file(s)" in capsys.readouterr().out



def test_a_refresh_of_other_folders_does_not_coalesce_this_one(
    store: LiteStore, project: Path, tmp_path: Path
) -> None:
    """D1 (deferred STAKES-003): a queued refresh of another folder is not one that sees these changes."""
    from recall.lite.refresh import refresh_purpose

    db = Path(store.path)
    other = tmp_path / "other"
    other.mkdir()
    purpose = refresh_purpose([other])
    holder, release = _holder(db, tmp_path)
    queued = _spawn(db, tmp_path / "q.log", "queued", purpose=purpose, wait_s=60)
    try:
        _wait_for(lambda: any(w.get("purpose") == purpose for w in lite_lane_status(db).waiters))
        report = _run(store, project, wait_s=0.5)
        assert report.outcome is Outcome.BUSY, f"coalesced into another folder's refresh: {report.outcome}"
    finally:
        release.touch()
        holder.wait(30)
        queued.wait(30)


def test_a_busy_refresh_writes_nothing(store: LiteStore, project: Path, tmp_path: Path) -> None:
    """D2 (deferred BUG-004): outside the lane, a busy refresh leaves the state file alone."""
    _run(store, project)
    before = read_state(store.path)
    _write_memos(project / "memory", 1, start=3)
    holder, release = _holder(Path(store.path), tmp_path)
    try:
        report = _run(store, project, wait_s=0)
    finally:
        release.touch()
        holder.wait(30)
    assert report.outcome is Outcome.BUSY
    assert read_state(store.path) == before, "a busy refresh rewrote the state file outside the lane"


def test_the_unchanged_check_does_not_load_the_corpus(
    store: LiteStore, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3 (deferred PERF-005): an unchanged folder is told from the store's counter, not its contents."""
    _run(store, project)

    def heavy(self: object) -> str:
        raise AssertionError("the unchanged check loaded the whole corpus")

    monkeypatch.setattr(LiteStore, "lite_generation_id", heavy)
    assert _run(store, project).outcome is Outcome.UNCHANGED


def test_a_write_through_another_path_is_not_missed(store: LiteStore, project: Path) -> None:
    """D3b (deferred PERF-005): the counter still catches a write the folder does not show."""
    _run(store, project)
    store.delete_sources([next(iter(store.iter_chunks())).source])
    assert _run(store, project).outcome is Outcome.REFRESHED


def test_a_fresh_whole_second_mtime_is_never_recorded_clean(store: LiteStore, project: Path) -> None:
    """D4 (deferred BUG-007): a coarse mtime written just now leaves the success re-checkable."""
    memo = project / "memory" / "memo-000.md"
    whole = (time.time_ns() // 1_000_000_000) * 1_000_000_000
    os.utime(memo, ns=(whole, whole))
    first = _run(store, project)
    assert first.outcome is Outcome.REFRESHED
    assert read_state(store.path)["last_success"]["fingerprint"].endswith(":racy")
    assert _run(store, project).outcome is not Outcome.UNCHANGED, "a racily clean snapshot was trusted"


def test_setup_and_refresh_index_alike(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D5 (deferred STAKES-005): after `recall setup --lite`, the first refresh re-embeds nothing."""
    import recall.context as context
    from recall.context import ContextPolicy
    from recall.lite.setup import run_lite_setup

    monkeypatch.setattr(context, "context_policy_for_profile", lambda profile: ContextPolicy(mode="document"))
    report = run_lite_setup(
        project_root=project, memory=project / "memory", embedder=ContentWordEmbedder(),  # type: ignore[arg-type]
        embedder_name="test", register=False,
    )
    with LiteStore(report.store, dim=DIM) as opened:
        again = refresh(opened, [project / "memory"], ContentWordEmbedder())  # type: ignore[arg-type]
    assert again.indexed_files == 0, f"the first refresh after setup re-embedded {again.indexed_files} memo(s)"


def test_a_future_dated_memo_does_not_keep_every_snapshot_racy(store: LiteStore, project: Path) -> None:
    """D4b (deferred BUG-007, found at L5.5): a whole-second mtime far ahead is not "just written"."""
    memo = project / "memory" / "memo-000.md"
    ahead = (time.time_ns() // 1_000_000_000 + 86_400) * 1_000_000_000
    os.utime(memo, ns=(ahead, ahead))
    assert _run(store, project).outcome is Outcome.REFRESHED
    assert not read_state(store.path)["last_success"]["fingerprint"].endswith(":racy")
    assert _run(store, project).outcome is Outcome.UNCHANGED, "a future-dated memo forced a refresh every time"
