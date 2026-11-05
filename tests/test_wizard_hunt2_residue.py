"""Regression tests for the halves of three second-hunt fixes that lived outside its hunted files.

PR 1211 (the second CCA hunt of the first-run code) fixed DOC-022, DOC-017 and CODE-006 inside
`recall/cli_commands/` and `recall/wizard/`, and left the same defect in three files the hunt did
not cover: `recall/quickstart.py`, `recall/cli_commands/calibration_cmd.py` and
`recall/claude_code.py`. Each test names the finding it pins. Red proof was taken on the remote
test host by running each test against the unmodified files before the fix; each failed in its
assertion, and the receipt is in the pull request.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_the_quickstart_points_at_the_wizard_for_a_real_threshold() -> None:
    """DOC-022 (quickstart half). Invariant: the quickstart's closing steps send a reader who wants
    a real threshold to `recall wizard`, the command that builds and certifies a calibrated
    generation. Failure caught: it said `recall setup` fits one, but the legacy table `recall setup`
    indexes is never calibrated for search, so the reader still saw a degraded state.
    Target: `recall.quickstart.next_steps`."""
    from recall.quickstart import next_steps

    lines = next_steps("postgresql://u:p@127.0.0.1:1/recall", provisioned=True, compose_path=None)
    threshold = [line for line in lines if "fit a real threshold" in line]
    assert threshold, lines
    assert "recall wizard" in threshold[0], threshold[0]


def test_recall_calibrate_points_at_the_published_calibration_doc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """DOC-017 (calibration_cmd half). Invariant: `recall calibrate` names a page a pip user can
    open. Failure caught: it printed the relative `docs/CALIBRATION.md`, which is not in the wheel.
    Target: `recall.cli_commands.calibration_cmd._cmd_calibrate`."""
    from recall.cli import main

    calibration = SimpleNamespace(
        embedder="hashing", threshold=0.5, scale="cosine", separability=0.9,
        separability_ci=None, n_answerable=1, n_unanswerable=1, certified=True,
        certification_reason=None,
    )
    result = SimpleNamespace(
        report=SimpleNamespace(fcr_at_050=0.0, fcr_at_suggested=0.0),
        calibration=calibration,
        path=tmp_path / "calibration.json",
    )
    monkeypatch.setattr("recall.setup.calibrate_from_files", lambda **_kw: result)
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "a", "answerable": True}, {"query": "b", "answerable": False}]),
        encoding="utf-8",
    )

    main(["--dsn", "postgresql://u:pw@127.0.0.1:1/recall", "calibrate", str(queries)])

    out = capsys.readouterr().out
    assert "https://github.com/GiulioDER/RE-call/blob/master/docs/CALIBRATION.md" in out, out


def test_two_backups_in_one_second_are_two_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CODE-006 (claude_code half). Invariant: every edit of a client file `recall setup` makes is
    preceded by its own backup, named the way uninstall names its backups, never reused. Failure
    caught: the name was `<file>.recall-backup-<seconds>` and an existing one was kept, so a second
    edit inside the same second wrote no backup of the state it replaced.
    Target: `recall.claude_code._backup`."""
    import time

    from recall import claude_code

    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    settings = tmp_path / "settings.json"
    settings.write_text("first", encoding="utf-8")
    one: Any = claude_code._backup(settings)
    settings.write_text("second", encoding="utf-8")
    two: Any = claude_code._backup(settings)

    assert one != two, (one, two)
    assert one.read_text(encoding="utf-8") == "first"
    assert two.read_text(encoding="utf-8") == "second"


def test_a_backup_never_replaces_an_existing_file_and_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-001 and BUG-001 (CCA run 20261105T220222Z). Invariant: `_backup` creates its file
    exclusively, never writing over or through anything already at the chosen name, and creates it
    owner-only, because one caller backs up `~/.claude.json`, which carries MCP bearer tokens.
    Failure caught: `shutil.copy2` overwrote the uninstaller's reusable `.last` fallback name,
    followed a symlink planted there, and created the copy at the umask default.
    Target: `recall.claude_code._backup`."""
    import os
    import stat
    import sys

    from recall import claude_code
    from recall.wizard import uninstall

    settings = tmp_path / ".claude.json"
    settings.write_text("tokens", encoding="utf-8")
    taken = tmp_path / ".claude.json.recall-backup-1.last"
    taken.write_text("the earlier backup", encoding="utf-8")
    monkeypatch.setattr(uninstall, "_backup_path", lambda _path: taken)

    with pytest.raises(FileExistsError):
        claude_code._backup(settings)
    assert taken.read_text(encoding="utf-8") == "the earlier backup"

    fresh = tmp_path / ".claude.json.recall-backup-2"
    monkeypatch.setattr(uninstall, "_backup_path", lambda _path: fresh)
    claude_code._backup(settings)
    assert fresh.read_bytes() == b"tokens"
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o600
