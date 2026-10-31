"""The graphical installer consumes the engine's real report, with no Qt dependency."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.desktop import install_ui
from recall.wizard.headless import Failure, HeadlessReport, Refusal
from recall.wizard.pipeline import CorpusOutcome
from recall.wizard.wiring import SmokeResult


def _reports() -> tuple[HeadlessReport, ...]:
    return (
        HeadlessReport(failures=(Failure("acme-docs", "database connection failed"),)),
        HeadlessReport(refused=(Refusal("acme-docs", "the document root is missing"),)),
        HeadlessReport(
            outcomes=(
                CorpusOutcome(
                    "acme-code", "gen_pending", degraded_reason="calibration was not certified"
                ),
            )
        ),
        HeadlessReport(
            smoke=(
                SmokeResult(
                    "acme-docs", "a question", 0, False, "unknown", None,
                    error="the configured server is unreachable",
                ),
            )
        ),
    )


@pytest.mark.parametrize("report", _reports(), ids=("failed", "refused", "unserved", "smoke"))
def test_real_unsuccessful_report_keeps_engine_diagnostics(report: HeadlessReport) -> None:
    """BUG-001, baseline 160a23a3f: `_summarise` loses the actual engine diagnostics.

    The existing consumer returns only 'Finished.' for these real unsuccessful reports.
    The intended red assertion compares its visible text with the engine's rendered result.
    """
    assert not report.ok
    assert install_ui._summarise(report) == report.render()


def _finished_method() -> Any:
    """Compile the actual production method so its report consumer runs without PySide6.

    Only the existing `_finished` method is extracted; no GUI logic is reimplemented. Both the
    baseline and fixed file contain it. Small widget adapters observe its public widget calls,
    allowing this behavior check to run where the desktop extra is deliberately absent.
    """
    source = Path(install_ui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_finished"
    )
    namespace = dict(vars(install_ui))
    exec(compile(ast.Module(body=[method], type_ignores=[]), install_ui.__file__, "exec"), namespace)  # noqa: S102 - trusted production method, no Qt dependency
    return namespace["_finished"]


def _completion_window() -> SimpleNamespace:
    heading: list[str] = []
    progress: list[int] = []
    log: list[str] = ["provisioned the database"]
    close: list[bool] = []
    return SimpleNamespace(
        report=None,
        progress_heading=SimpleNamespace(setText=heading.append),
        progress_bar=SimpleNamespace(setRange=lambda *_args: None, setValue=progress.append),
        close_button=SimpleNamespace(setEnabled=close.append),
        _append=log.append,
        heading=heading,
        progress=progress,
        log=log,
        close=close,
    )


@pytest.mark.parametrize("report", _reports(), ids=("failed", "refused", "unserved", "smoke"))
def test_completion_does_not_label_an_unsuccessful_report_installed(
    report: HeadlessReport,
) -> None:
    """BUG-001, baseline 160a23a3f: the real `_finished` method always says 'Installed'.

    The intended red assertion observes that heading on a real report whose `ok` is false;
    source extraction avoids a missing-Qt setup failure masquerading as regression proof.
    """
    window = _completion_window()
    _finished_method()(window, report)

    assert window.heading == ["Install incomplete"]
    assert window.progress == [0]
    assert window.close == [True]
    assert window.report is report
    assert window.log == ["provisioned the database", "\n" + report.render()]


def test_successful_report_keeps_its_real_summary_and_installed_heading() -> None:
    """BUG-001 success control: the repaired consumer must preserve a completed install.

    Baseline 160a23a3f fails the summary assertion because it also discards successful outcomes.
    The heading and completed progress are controls for behavior that must remain unchanged.
    """
    report = HeadlessReport(
        outcomes=(CorpusOutcome("acme-docs", "gen_live", certified=True, promoted=True),)
    )
    assert report.ok
    assert install_ui._summarise(report) == report.render()

    window = _completion_window()
    _finished_method()(window, report)
    assert window.heading == ["Installed"]
    assert window.progress == [1]
    assert window.close == [True]


    # Existing injected-runner compatibility control in the same baseline-red node.
    report = SimpleNamespace(
        corpora=(
            SimpleNamespace(tenant="acme-docs", promoted=True),
            SimpleNamespace(tenant="acme-code", promoted=False, degraded_reason="not certified"),
        )
    )
    assert install_ui._summarise(report) == (
        "Finished.\n  acme-docs: ready\n  acme-code: not certified"
    )
    window = _completion_window()
    _finished_method()(window, report)
    assert window.heading == ["Installed"]
    assert window.progress == [1]
