"""A graphical front end for `recall wizard`, for the person who will never open a terminal.

The engine has supported an unattended install from a JSON config for a long time, and
`recall/wizard/interactive.py` made the questions answerable in a terminal. This is the third
renderer of the SAME question plan and the same engine. It is not a second installer.

Four decisions, each of which is where a graphical installer usually goes wrong:

**The questions come from `recall.wizard.questions`.** Not from this file. A form that re-typed the
defaults, the Docker/existing branch, or the rule that all three corpus roots are mandatory would be
a second installer that drifts, and every drift of that shape in this repository has been found late
and by accident. This module renders widgets for a plan it does not own.

**The answers become a config FILE, and the install runs from the file.** Exactly as the terminal
flow does. The user keeps an artefact they can re-run, hand to somebody, or put in CI, and there is
one engine rather than a graphical one that diverges. The path is shown on screen before anything is
built, because an installer that writes files without saying where is one you cannot undo.

**The install runs off the UI thread and streams progress.** `run_headless` takes minutes: it pulls
an image, applies migrations, builds a generation per corpus, calibrates each one. On the UI thread
that is a frozen window, which every user reads as a crash. See `recall/desktop/jobs.py` for the two
non-obvious properties that make the progress actually arrive.

**A default the user has edited is never silently recomputed.** The corpus roots are suggested
underneath the chosen data folder, so changing that folder should move suggestions the user has not
touched, and must not move ones they typed. Overwriting a typed path would be the same defect as a
prompt that silently applies a default over an answer, which the terminal flow refuses to do.

⚠️ **This module builds a window and nothing else: the event loop lives in
`recall/desktop/main.py`.** That split matches how `run_app` and its entry point are already
arranged, and it is what lets every test below drive the real widgets without a running application.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from recall.wizard.questions import (
    Question,
    build_config,
    default_for,
    question_plan,
    visible_questions,
)

__all__ = ["InstallerWindow"]

_qt: Any = None
try:  # pragma: no cover - exercised on environments without the desktop extra
    _qt = importlib.import_module("PySide6.QtWidgets")
    from PySide6.QtCore import Qt, QThreadPool
    # ⚠️ `QApplication` is deliberately NOT imported here; it is resolved by `getattr` below, the
    # same way `recall/desktop/ui.py` does it. Importing it as well would give the name two
    # definitions and mypy rejects the second.
    from PySide6.QtWidgets import (
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    pass

# See `recall/desktop/jobs.py`: resolved by `getattr` so no ignore is needed in either environment.
QApplication: Any = getattr(_qt, "QApplication", None)


if QApplication is not None:

    from recall.desktop.jobs import StreamingJob

    class _Field:
        """One question's widgets, and whether the person has touched it.

        ⚠️ **`edited` is the whole reason this is a class rather than a dict of widgets.** The corpus
        roots are suggested under the chosen data folder, so that folder changing should move the
        suggestions, but only the ones still showing a suggestion. A field the user typed into is
        their answer, and recomputing it would be the graphical version of a prompt that silently
        overwrites what somebody entered.
        """

        def __init__(self, question: Question, widget: Any, row: Any, label: Any) -> None:
            self.question = question
            self.widget = widget
            self.row = row
            self.label = label
            self.edited = False

        def value(self) -> str:
            if isinstance(self.widget, QComboBox):
                return str(self.widget.currentData())
            return str(self.widget.text()).strip()

        def set_value(self, value: str) -> None:
            if isinstance(self.widget, QComboBox):
                index = self.widget.findData(value)
                if index >= 0:
                    self.widget.setCurrentIndex(index)
            else:
                self.widget.setText(value)

    class InstallerWindow(QWidget):
        """The two-page installer: answer the questions, then watch it run.

        `runner`, `writer` and `prober` are injected so the whole flow can be exercised without
        provisioning anything. That is not a testing convenience bolted on: an installer whose only
        test is a real install is one that gets tested once, by hand, on the machine of whoever
        wrote it.
        """

        def __init__(
            self,
            *,
            default_root: Path | None = None,
            runner: Any = None,
            writer: Any = None,
            prober: Any = None,
        ) -> None:
            super().__init__()
            self.setWindowTitle("Install recall")
            self.resize(760, 620)

            self._default_root = default_root or (Path.home() / ".recall")
            self._plan = question_plan(default_root=self._default_root)
            self._fields: dict[str, _Field] = {}
            self._runner = runner
            self._writer = writer
            self._prober = prober
            self._jobs: list[Any] = []
            self.pool = QThreadPool(self)
            self.report: Any = None
            self.config_path: Path | None = None

            self.stack = QStackedWidget(self)
            self.stack.addWidget(self._build_form())
            self.stack.addWidget(self._build_progress())
            layout = QVBoxLayout(self)
            layout.addWidget(self.stack)

            self._refresh_visibility()
            self._refresh_defaults()

        # -- page 1 -------------------------------------------------------------------------

        def _build_form(self) -> QWidget:
            page = QWidget()
            outer = QVBoxLayout(page)
            heading = QLabel("Install recall")
            heading.setStyleSheet("font-size: 20px; font-weight: 600;")
            outer.addWidget(heading)
            outer.addWidget(
                QLabel(
                    "Your answers are saved as a configuration file, and that file is what runs. "
                    "You can re-run it later or send it to someone else."
                )
            )

            self.form = QFormLayout()
            self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for question in self._plan:
                self._add_field(question)
            outer.addLayout(self.form)

            self.form_status = QLabel("")
            self.form_status.setWordWrap(True)
            outer.addWidget(self.form_status)
            outer.addStretch(1)

            buttons = QHBoxLayout()
            buttons.addStretch(1)
            self.install_button = QPushButton("Install")
            self.install_button.setDefault(True)
            self.install_button.clicked.connect(self._start_install)
            buttons.addWidget(self.install_button)
            outer.addLayout(buttons)
            return page

        def _add_field(self, question: Question) -> None:
            """Build the widgets for one question, choosing the control from its KIND.

            A `directory` gets a Browse button and a `dsn` gets a Test button. Those are the two
            questions a person is most likely to get wrong by typing, and they are the two where
            being wrong is discovered minutes later, during a build, named as something else.
            """
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)

            widget: Any
            if question.kind == "choice":
                widget = QComboBox()
                for choice in question.choices:
                    widget.addItem(choice.label, choice.key)
                widget.currentIndexChanged.connect(self._choice_changed)
                row.addWidget(widget, 1)
            else:
                widget = QLineEdit()
                widget.textEdited.connect(lambda _text, key=question.key: self._mark_edited(key))
                row.addWidget(widget, 1)
                if question.kind == "directory":
                    browse = QPushButton("Browse...")
                    browse.clicked.connect(
                        lambda _checked=False, key=question.key: self._browse(key)
                    )
                    row.addWidget(browse)
                elif question.kind == "dsn":
                    test = QPushButton("Test")
                    test.clicked.connect(lambda _checked=False: self._test_dsn())
                    row.addWidget(test)

            label = QLabel(question.prompt + ("" if question.optional else " *"))
            label.setToolTip(question.help)
            container.setToolTip(question.help)
            self.form.addRow(label, container)
            field = _Field(question, widget, container, label)
            # The static default, applied once. A default that cannot change does not belong in
            # `_refresh_defaults`; see the reasoning there.
            if not callable(question.default):
                field.set_value(default_for(question, {}))
            self._fields[question.key] = field

        def _answers(self) -> dict[str, str]:
            """What the visible fields currently hold.

            Hidden fields are EXCLUDED, not merely ignored. A `dsn` left over from a person who
            switched to the Docker option would otherwise reach `build_config` and put both keys in
            the document, and the engine reads a document carrying a `dsn` as an existing database.
            """
            answers: dict[str, str] = {}
            for question in visible_questions(self._plan, self._raw_answers()):
                answers[question.key] = self._fields[question.key].value()
            return answers

        def _raw_answers(self) -> dict[str, str]:
            """Every field, used only to resolve `depends_on`, which reads the choice questions."""
            return {key: field.value() for key, field in self._fields.items()}

        def _choice_changed(self) -> None:
            self._refresh_visibility()
            self._refresh_defaults()

        def _mark_edited(self, key: str) -> None:
            self._fields[key].edited = True
            if key == "data_root":
                # The roots are suggested underneath this one, so a change here should move the
                # suggestions that are still suggestions.
                self._refresh_defaults()

        def _refresh_visibility(self) -> None:
            """Show the questions that apply, and take the row away for the ones that do not.

            ⚠️ **`setRowVisible`, not `setVisible` on the two widgets.** Hiding a row's label and
            field leaves the ROW in the `QFormLayout`, so the form keeps a blank gap exactly where
            the question a person did not choose used to be. It reads as a control that failed to
            load rather than as one that does not apply. Qt 6.4 and later; the desktop extra pins
            PySide6 6.11.
            """
            visible = {
                question.key for question in visible_questions(self._plan, self._raw_answers())
            }
            for key, field in self._fields.items():
                shown = key in visible
                self.form.setRowVisible(field.row, shown)
                # Kept alongside, because `isVisibleTo` on the widgets is what a test can ask, and
                # because `setRowVisible` alone leaves the widgets themselves nominally visible.
                field.row.setVisible(shown)
                field.label.setVisible(shown)

        def _refresh_defaults(self) -> None:
            """Recompute the untouched fields whose default DEPENDS on another answer.

            ⚠️ **Only the dynamic ones, and that restriction is load-bearing.** A static default
            cannot change, so rewriting it is a no-op at best and a clobber at worst: any value put
            there by something other than typing — a restored previous answer, a value set by a
            future "reuse my last install" button — would be silently reset by an unrelated change
            elsewhere in the form. Typing marks a field edited, so a person would not have seen it;
            that is precisely what makes it the kind of defect that ships.

            The static defaults are filled once, in `_add_field`, where they belong.
            """
            answers = self._raw_answers()
            for field in self._fields.values():
                if field.edited or isinstance(field.widget, QComboBox):
                    continue
                if not callable(field.question.default):
                    continue
                field.set_value(default_for(field.question, answers))

        def _browse(self, key: str) -> None:
            field = self._fields[key]
            chosen = QFileDialog.getExistingDirectory(self, field.question.prompt, field.value())
            if chosen:
                field.set_value(chosen)
                self._mark_edited(key)

        def _test_dsn(self) -> None:
            """Probe the connection string before the install, not during it.

            Taking a DSN on faith is what makes the existing-database path a trap: every way it can
            be wrong fails minutes later, inside a build, naming something other than the connection
            string. The terminal flow re-asks until the probe is clean; a form cannot block, so it
            reports and lets the person decide.
            """
            dsn = self._fields["dsn"].value()
            if not dsn:
                self.form_status.setText("Enter a connection string first.")
                return
            probe = self._prober or _default_prober()
            self.form_status.setText("Checking...")
            QApplication.processEvents()
            try:
                report = probe(
                    dsn, expected_dimension=_dimension_for(self._fields["embedder"].value())
                )
            except Exception as exc:  # noqa: BLE001 - a probe failure is a result, not a crash
                # ⛔ **Scrubbed, because a driver error quotes the DSN back verbatim.**
                # `probe_database` scrubs what it RETURNS but not what it RAISES, and the identical
                # handler in `recall/desktop/ui.py` already scrubs for a measured reason:
                # `ProgrammingError: missing "=" after "not-a-dsn://user:PASSWORD@x"`. This is the
                # surface most likely to hit it, because it is where somebody types a connection
                # string for the first time, and it is what ends up in screenshots and bug reports.
                self.form_status.setText(
                    _scrubbed(f"Could not check that database: {exc}", dsn)
                )
                return
            self.form_status.setText(_scrubbed(report.render(), dsn))

        # -- page 2 -------------------------------------------------------------------------

        def _build_progress(self) -> QWidget:
            page = QWidget()
            outer = QVBoxLayout(page)
            self.progress_heading = QLabel("Installing...")
            self.progress_heading.setStyleSheet("font-size: 20px; font-weight: 600;")
            outer.addWidget(self.progress_heading)
            self.progress_subtitle = QLabel("")
            self.progress_subtitle.setWordWrap(True)
            outer.addWidget(self.progress_subtitle)

            self.progress_bar = QProgressBar()
            # Indeterminate: the engine reports named steps, not a fraction, and inventing a
            # percentage from a step count would be a number that means nothing and stalls visibly.
            self.progress_bar.setRange(0, 0)
            outer.addWidget(self.progress_bar)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            outer.addWidget(self.log, 1)

            buttons = QHBoxLayout()
            buttons.addStretch(1)
            self.close_button = QPushButton("Close")
            self.close_button.setEnabled(False)
            self.close_button.clicked.connect(self.close)
            buttons.addWidget(self.close_button)
            outer.addLayout(buttons)
            return page

        def _start_install(self) -> None:
            """Validate, write the config, and hand the FILE to the engine on a worker thread."""
            try:
                config = build_config(self._answers())
            except ValueError as exc:
                # `build_config` raises the sentence a person can act on. Showing it here rather
                # than letting `load_config` reject the file later is the difference between a form
                # error and an installer complaining about its own output.
                self.form_status.setText(str(exc))
                return

            target = Path(str(config.get("data_root") or self._default_root)) / "wizard.json"
            try:
                self.config_path = (self._writer or _default_writer())(config, target)
            except OSError as exc:
                self.form_status.setText(f"Could not save the configuration to {target}: {exc}")
                return

            self.install_button.setEnabled(False)
            self.stack.setCurrentIndex(1)
            self.progress_subtitle.setText(
                f"Saved {self.config_path}\nRe-run this install with:\n"
                f"  recall wizard --headless --config {self.config_path}"
            )
            self._append("starting")
            self._run(config)

        def _run(self, config: dict[str, object]) -> None:
            runner = self._runner or _default_runner()
            path = self.config_path

            def _job(progress: Any) -> Any:
                return runner(config, path, progress)

            job = StreamingJob(_job)
            # Held for the length of the run. See `recall/desktop/jobs.py`: a `QRunnable` Qt has
            # already destroyed cannot deliver the queued signal it was destroyed before emitting.
            self._jobs.append(job)
            job.signals.progress.connect(self._append)
            job.signals.done.connect(self._finished)
            job.signals.failed.connect(self._failed)
            job.signals.done.connect(lambda _r=None, j=job: self._release(j))
            job.signals.failed.connect(lambda _m=None, j=job: self._release(j))
            self.pool.start(job)

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
            """Wait a bounded moment for the install, then close anyway.

            ⛔ **`QThreadPool.waitForDone()` with no argument waits FOREVER**, and that is what a
            parented pool's destructor calls. `MainWindow` was given a bounded `closeEvent` for
            exactly this reason after closing during a first install froze the application for up to
            half an hour; this window runs a strictly longer job — image pull, migrations, a
            generation and a calibration per corpus — and shipped without one.

            The bound is `CLOSE_WAIT_MS`, the same constant, chosen against a quick worker finishing
            normally rather than against `docker compose up`. Closing while work continues is both
            true and better than a window that will not close: the install runs in Docker and
            survives this process.
            """
            from recall.desktop.jobs import CLOSE_WAIT_MS

            if self._jobs and not self.pool.waitForDone(CLOSE_WAIT_MS):
                # Said out loud rather than swallowed: the user is entitled to know the install did
                # not stop just because the window did.
                print(
                    "recall-install: the install is still running in the background; "
                    "it will finish on its own.",
                    file=sys.stderr,
                )
            event.accept()

        def _release(self, job: Any) -> None:
            if job in self._jobs:
                self._jobs.remove(job)

        def _append(self, line: str) -> None:
            self.log.appendPlainText(str(line))

        def _finished(self, report: Any) -> None:
            self.report = report
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.progress_heading.setText("Installed")
            self._append("\n" + _summarise(report))
            self.close_button.setEnabled(True)

        def _failed(self, message: str) -> None:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_heading.setText("Install failed")
            # ⚠️ Appended, not substituted for the log. What ran before the failure is the only
            # thing that says HOW FAR it got, and on an installer that provisions a database and
            # applies migrations, how far it got is what decides what to do next.
            # Scrubbed against the DSN field: `run_headless` scrubs the errors it PACKAGES, but an
            # exception propagating out of it (or out of `load_config`) arrives raw, and this log is
            # what a person screenshots when an install fails.
            self._append(_scrubbed(f"\nfailed: {message}", self._fields["dsn"].value()))
            self.close_button.setEnabled(True)

else:  # pragma: no cover - no desktop extra installed

    InstallerWindow = None  # type: ignore[assignment,misc]


def _scrubbed(message: str, dsn: str) -> str:
    """`message` with `dsn`'s password removed, tolerating a blank or unparseable DSN.

    One helper rather than two call sites reaching for `scrub_dsn_secrets` independently: the
    defect this fixes was precisely a second copy of a handler that dropped the scrubbing its
    original had.
    """
    if not dsn:
        return message
    from recall.store import scrub_dsn_secrets

    try:
        return str(scrub_dsn_secrets(message, dsn))
    except Exception:  # noqa: BLE001 - a scrub that fails must not replace the error it was hiding
        # Returning the raw message here would leak; returning nothing would lose the diagnosis.
        # Say that something was suppressed instead.
        return "the error could not be displayed safely because it may contain your password"


def _default_writer() -> Any:
    from recall.wizard.interactive import write_config

    return write_config


def _default_runner() -> Any:
    """Run the install from the file that was just written.

    ⛔ **Loads the config back from disk rather than using the dict in memory.** That round trip is
    the point: it is the same `load_config` the headless path uses, so a document this form could
    produce but the engine would refuse fails HERE, with the engine's own message, instead of
    succeeding in the GUI and failing for whoever re-runs the file later.
    """

    def _run(config: dict[str, object], path: Path, progress: Any) -> Any:
        from recall.wizard.headless import load_config, run_headless

        return run_headless(load_config(path), progress=progress)

    return _run


def _default_prober() -> Any:
    from recall.wizard.database import probe_database

    return probe_database


def _dimension_for(embedder: str) -> int | None:
    """The embedder's vector width, or None when it cannot be determined without loading it."""
    try:
        from recall.embeddings import resolve_embedder

        return int(resolve_embedder(embedder).dim)
    except Exception:  # noqa: BLE001 - see the terminal flow: an unresolvable embedder is refused
        # by name later with a better message, and the dimension check degrades to "not compared",
        # which the report states explicitly.
        return None


def _summarise(report: Any) -> str:
    """One block a person can read, from whatever the engine returned.

    Defensive about the report's shape on purpose: this is the last thing shown after a long
    install, and an `AttributeError` here would replace the outcome with a stack trace.
    """
    lines: list[str] = []
    corpora = getattr(report, "corpora", None) or ()
    for outcome in corpora:
        tenant = getattr(outcome, "tenant", "?")
        if getattr(outcome, "promoted", False):
            lines.append(f"  {tenant}: ready")
        else:
            reason = getattr(outcome, "degraded_reason", None) or "not promoted"
            lines.append(f"  {tenant}: {reason}")
    if not lines:
        return "Finished."
    return "Finished.\n" + "\n".join(lines)
