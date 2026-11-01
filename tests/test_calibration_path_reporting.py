"""ENV-103: `recall doctor` and `recall lint` name the calibration file they resolved.

With RECALL_CALIBRATION unset the legacy file is `calibration.json` in the working directory, so
the same check run from two directories reads two files. The owner kept that default and asked for
the path to be printed instead. No database and no network: the lint scan is replaced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_doctor_names_the_calibration_file_it_looked_for(monkeypatch, tmp_path) -> None:
    """Red proof: against the pre-fix `_calibration_check` (recall/doctor.py, which reported only
    "none fitted for 'hashing'"), the detail does not contain the path, failing the assertion.
    """
    from recall.doctor import _calibration_check

    target = tmp_path / "calibration.json"
    monkeypatch.setenv("RECALL_CALIBRATION", str(target))

    check = _calibration_check("hashing", strict=True, dsn="")

    assert check.status == "warn"
    assert str(target.resolve()) in check.detail, check.detail


def test_doctor_names_the_file_a_calibration_was_read_from(monkeypatch, tmp_path) -> None:
    """Control for the found case: the threshold line names the file too."""
    from recall.calibration import Calibration, save
    from recall.doctor import _calibration_check

    target = tmp_path / "calibration.json"
    save(Calibration(embedder="hashing", threshold=0.42), target)
    monkeypatch.setenv("RECALL_CALIBRATION", str(target))

    check = _calibration_check("hashing", strict=True, dsn="")

    assert check.status == "ok"
    assert "0.420" in check.detail and str(target.resolve()) in check.detail, check.detail


@pytest.mark.parametrize("calibrated", [False, True])
def test_lint_names_the_calibration_file_behind_its_threshold(
    monkeypatch, tmp_path, capsys, calibrated: bool
) -> None:
    """Red proof: against the pre-fix `_run_semantic` (recall/cli_commands/lint_check.py, which
    printed only the embedder name as the threshold's source), the printed line does not contain
    the path, failing the assertion in both cases.
    """
    import recall.semantic_lint
    from recall.calibration import Calibration, save
    from recall.cli_commands.lint_check import _run_semantic

    target = tmp_path / "calibration.json"
    if calibrated:
        save(Calibration(embedder="hashing", threshold=0.42), target)
    monkeypatch.setenv("RECALL_CALIBRATION", str(target))
    monkeypatch.setattr(recall.semantic_lint, "semantic_lint", lambda *a, **k: [])
    args = SimpleNamespace(threshold=None, dsn="unused", path=str(tmp_path), glob="*.md", embedder="hashing")

    _run_semantic(args, SimpleNamespace(name="hashing"))

    lines = capsys.readouterr().out.splitlines()
    line = next(text for text in lines if text.startswith("semantic threshold"))
    assert str(target.resolve()) in line, line
    assert ("0.42" in line) is calibrated
