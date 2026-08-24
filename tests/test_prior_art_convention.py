from __future__ import annotations

from pathlib import Path

from tools.prior_art.convention import check_prior_work_declarations


def test_prior_work_declaration_accepts_first_module_docstring(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text('"""Prior work: [[memory-note]] established the baseline."""\n', encoding="utf-8")
    assert check_prior_work_declarations([path]) == []


def test_prior_work_declaration_rejects_missing_docstring(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    errors = check_prior_work_declarations([path])
    assert any("must start with a module docstring" in error for error in errors)


def test_prior_work_declaration_rejects_wrong_docstring_prefix(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    path.write_text('"""Experiment details."""\n', encoding="utf-8")
    errors = check_prior_work_declarations([path])
    assert any("must contain a Prior work:" in error for error in errors)
