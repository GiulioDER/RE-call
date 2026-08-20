"""The graphical installer, driven as a person drives it.

Every test here operates the REAL widgets offscreen and injects only the three things that would
otherwise touch the world: the config writer, the engine, and the database probe. An installer whose
tests stub the form is an installer tested once, by hand, on the machine of whoever wrote it.

The property this file exists to protect is **one installer, three renderers**. `recall wizard` in a
terminal, this window, and a saved JSON config must produce the same install. Three separate defects
fixed in this repository had exactly the shape of one rule with two implementations, so the tests
that matter most below are the ones comparing this front end against the terminal one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _installer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> Any:
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.install_ui import InstallerWindow

    QApplication.instance() or QApplication([])
    kwargs.setdefault("default_root", tmp_path / "recall")
    kwargs.setdefault("writer", lambda config, path: path)
    kwargs.setdefault("runner", lambda config, path, progress: None)
    return InstallerWindow(**kwargs)


def test_the_form_renders_every_question_in_the_shared_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ A question the plan declares and the form omits is a config key nobody can set.

    The plan is the contract. Asserted against `question_plan` rather than a list typed here,
    because a list typed here is the second definition this whole design exists to avoid.
    """
    from recall.wizard.questions import question_plan

    window = _installer(monkeypatch, tmp_path)
    try:
        expected = {question.key for question in question_plan(default_root=tmp_path / "recall")}
        assert set(window._fields) == expected
    finally:
        window.close()


def test_the_dsn_field_is_hidden_until_an_existing_database_is_chosen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both sides of the branch exist as widgets; only one applies at a time.

    Showing both would ask for a connection string on the path where the wizard PROVISIONS the
    database, which is the recommended path and therefore the common one.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        assert window._fields["data_root"].row.isVisibleTo(window)
        assert not window._fields["dsn"].row.isVisibleTo(window)

        window._fields["database"].set_value("existing")

        assert window._fields["dsn"].row.isVisibleTo(window)
        assert not window._fields["data_root"].row.isVisibleTo(window)
    finally:
        window.close()


def test_a_hidden_answer_never_reaches_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ A leftover DSN from a person who changed their mind must not reach the document.

    `load_config` reads a config carrying a `dsn` as an existing database, so a stale value in a
    hidden field would silently override the Docker choice the user actually made. Hiding a widget
    does not clear it, which is exactly why `_answers` filters rather than reads everything.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        window._fields["database"].set_value("existing")
        window._fields["dsn"].set_value("postgresql://someone@example.test/db")
        window._fields["database"].set_value("docker")

        answers = window._answers()

        assert "dsn" not in answers, "the connection string belongs to the path that was abandoned"
        assert "data_root" in answers
    finally:
        window.close()


def test_a_typed_path_is_never_overwritten_by_a_recomputed_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ Recomputing a default over an answer is the graphical form of ignoring what was typed.

    The corpus roots are suggested underneath the chosen data folder, so changing that folder must
    move the suggestions — and must not move a root the person typed. The terminal flow refuses to
    silently apply a default over an answer; a form that quietly rewrote a field would be worse,
    because the user watches it happen and cannot tell whether it will happen again.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        chosen = str(tmp_path / "my-notes")
        window._fields["docs_root"].set_value(chosen)
        window._mark_edited("docs_root")

        window._fields["data_root"].set_value(str(tmp_path / "elsewhere"))
        window._mark_edited("data_root")

        assert window._fields["docs_root"].value() == chosen, "a typed root must survive"
        assert window._fields["code_root"].value().startswith(str(tmp_path / "elsewhere")), (
            "an untouched root must follow the data folder, or the suggestion stops being useful"
        )
    finally:
        window.close()


def test_the_window_and_the_terminal_produce_the_same_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ **The property this whole design exists for.**

    Same answers in, same config document out. Not "similar": identical, key for key. Three defects
    fixed in this repository this week had the shape of one rule with two implementations — the
    certification gate keyed on the wrong environment, the desktop upload assembling its own
    pipeline identity, and a second copy of the compose file's name that made a whole feature inert.
    A second installer would be the same defect with a bigger blast radius.
    """
    from recall.wizard.interactive import Prompter, ask_config

    window = _installer(monkeypatch, tmp_path)
    try:
        window._fields["project"].set_value("acme")
        window._fields["embedder"].set_value("hashing")
        window._fields["data_root"].set_value(str(tmp_path / "store"))
        window._mark_edited("data_root")
        window._fields["project_root"].set_value(str(tmp_path / "work"))
        window._mark_edited("project_root")
        from_gui = window._answers()
    finally:
        window.close()

    typed = iter(
        [
            "acme",  # project
            "hashing",  # embedder
            "docker",  # database
            str(tmp_path / "store"),  # data_root
            str(tmp_path / "store" / "docs"),
            str(tmp_path / "store" / "code"),
            str(tmp_path / "store" / "memory"),
            str(tmp_path / "work"),  # project_root
        ]
    )
    terminal = ask_config(
        Prompter(read=lambda _p: next(typed), write=lambda _line: None),
        default_root=tmp_path / "recall",
    )

    from recall.wizard.questions import build_config

    assert build_config(from_gui, today=lambda: "2026-01-01") == {
        **terminal,
        "corpus_version": "2026-01-01",
    }


def test_the_install_runs_off_the_ui_thread_and_streams_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⚠️ **A frozen window is what every user reads as a crash.**

    `run_headless` pulls an image, applies migrations, builds a generation per corpus and calibrates
    each one. On the UI thread that is minutes of an unresponsive window with no output.

    Asserted structurally rather than by timing, following the lesson in `recall/desktop/jobs.py`:
    the delivery failure it describes is a garbage-collection RACE, so a timing test blesses the
    broken code whenever the collector is slow. These two properties hold deterministically.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        started: list[Any] = []
        window.pool.start = lambda job: started.append(job)  # type: ignore[method-assign]
        window._fields["data_root"].set_value(str(tmp_path / "store"))
        window._mark_edited("data_root")

        window._start_install()

        assert started, "the install must reach the thread pool rather than run inline"
        job = started[0]
        assert job.autoDelete() is False, (
            "a QRunnable with autoDelete on is destroyed the moment run() returns, taking its "
            "signal source with it and purging the queued progress and done callbacks"
        )
        assert job in window._jobs, (
            "the job must be held for the length of the run; dropping the reference is the same "
            "race measured at 1 of 5 deliveries in recall/desktop/jobs.py"
        )
    finally:
        window.close()


def test_progress_lines_reach_the_log_and_the_report_is_summarised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The engine reports named steps; the window must show them and then say what happened."""

    class _Outcome:
        def __init__(self, tenant: str, promoted: bool, reason: str | None = None) -> None:
            self.tenant = tenant
            self.promoted = promoted
            self.degraded_reason = reason

    class _Report:
        corpora = (_Outcome("acme-docs", True), _Outcome("acme-code", False, "not certified"))

    window = _installer(monkeypatch, tmp_path)
    try:
        window._append("provision")
        window._append("build acme-docs")
        window._finished(_Report())

        text = window.log.toPlainText()
        assert "provision" in text and "build acme-docs" in text
        assert "acme-docs: ready" in text
        assert "acme-code: not certified" in text, (
            "a corpus that did not go live must say so; reporting only the successes is how an "
            "installer claims an install that did not happen"
        )
        assert window.close_button.isEnabled()
    finally:
        window.close()


def test_a_failure_keeps_the_log_rather_than_replacing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """How far it got is what decides what to do next.

    This installer provisions a database and applies migrations before it builds anything, so a
    failure message that replaced the log would discard the only record of which of those completed.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        window._append("provision")
        window._append("schema")
        window._failed("the database never accepted a connection")

        text = window.log.toPlainText()
        assert "provision" in text and "schema" in text
        assert "the database never accepted a connection" in text
        assert window.close_button.isEnabled(), "a failed install must still be closeable"
    finally:
        window.close()


def test_an_invalid_answer_is_reported_on_the_form_not_by_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ An installer complaining about its own output is the defect the terminal flow shipped once.

    All three corpus roots are mandatory. Blanking one and pressing Install must fail HERE, with a
    sentence about the field, rather than writing a config that `load_config` rejects one step later
    with a message about JSON keys.
    """
    written: list[Path] = []
    window = _installer(
        monkeypatch, tmp_path, writer=lambda config, path: written.append(path) or path
    )
    try:
        window._fields["docs_root"].set_value("")
        window._mark_edited("docs_root")

        window._start_install()

        assert not written, "nothing may be written when the answers cannot make a valid config"
        assert window.stack.currentIndex() == 0, "the form must stay on screen to be corrected"
        assert "docs_root" in window.form_status.text()
    finally:
        window.close()


def test_the_saved_config_path_is_shown_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An installer that writes files without saying where is one you cannot undo.

    The re-run command is shown too, because the config file being reusable is the reason the
    graphical flow writes one at all instead of installing from the answers in memory.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        window.pool.start = lambda job: None  # type: ignore[method-assign]
        window._fields["data_root"].set_value(str(tmp_path / "store"))
        window._mark_edited("data_root")

        window._start_install()

        subtitle = window.progress_subtitle.text()
        assert str(tmp_path / "store") in subtitle
        assert "recall wizard --headless --config" in subtitle
    finally:
        window.close()


def test_the_probe_reports_rather_than_blocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checking the connection string before the install is the point of the button.

    Taking a DSN on faith is what makes the existing-database path a trap: every way it can be wrong
    fails minutes later, inside a build, named as something else. The terminal flow re-asks until
    the probe is clean; a form cannot block, so it reports and lets the person decide.
    """

    class _Report:
        usable = False

        def render(self) -> str:
            return "pgvector is not installed on that database"

    window = _installer(monkeypatch, tmp_path, prober=lambda dsn, expected_dimension: _Report())
    try:
        window._fields["database"].set_value("existing")
        window._fields["dsn"].set_value("postgresql://user@host/db")

        window._test_dsn()

        assert "pgvector is not installed" in window.form_status.text()
    finally:
        window.close()


def test_a_probe_that_raises_is_a_result_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreachable host raises from deep in the driver. That is an ANSWER about the database."""

    def _explode(dsn: str, expected_dimension: int | None) -> Any:
        raise OSError("no route to host")

    window = _installer(monkeypatch, tmp_path, prober=_explode)
    try:
        window._fields["database"].set_value("existing")
        window._fields["dsn"].set_value("postgresql://user@unreachable/db")

        window._test_dsn()

        assert "no route to host" in window.form_status.text()
    finally:
        window.close()


def test_the_engine_is_handed_the_file_rather_than_the_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ The round trip through `load_config` is the point, not an implementation detail.

    Running the in-memory dict would mean a document this form can produce but the engine refuses
    succeeds in the GUI and fails for whoever re-runs the saved file. Asserted on the default
    runner's source, since the injected one in every other test cannot show this.
    """
    import inspect

    from recall.desktop.install_ui import _default_runner

    source = inspect.getsource(_default_runner)
    assert "load_config(path)" in source, "the engine must read back the file that was written"
    assert "run_headless(load_config(path)" in source


def test_the_written_config_is_one_the_engine_actually_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ **The round trip, with the real writer and the real loader.**

    This is the test that caught the equivalent bug in the terminal flow, and it caught it after ten
    passing tests of the question flow had said everything was fine. The questions were right; the
    ARTEFACT was not. `load_config` names all three corpus roots as mandatory, and a document
    missing one fails to load with a message about JSON keys — an installer handing the user a
    validation error about its own output.

    Asserting on the source of `_default_runner` (the test above) proves the wiring is written down.
    This proves it works.
    """
    from recall.wizard.headless import load_config
    from recall.wizard.interactive import write_config

    loaded: list[Any] = []

    def _runner(config: dict[str, object], path: Path, progress: Any) -> Any:
        loaded.append(load_config(path))
        return None

    window = _installer(monkeypatch, tmp_path, writer=write_config, runner=_runner)
    try:
        window.pool.start = lambda job: job.run()  # type: ignore[method-assign]
        window._fields["project"].set_value("acme")
        window._fields["data_root"].set_value(str(tmp_path / "store"))
        window._mark_edited("data_root")

        window._start_install()

        assert window.config_path is not None and window.config_path.exists()
        assert loaded, "the engine never received a config"
        assert loaded[0].project == "acme"
    finally:
        window.close()


def test_the_gui_flag_does_not_also_run_the_terminal_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚠️ `recall wizard --gui` must RETURN, not fall through.

    The terminal branch below it asks every question and then installs. Falling through would ask
    the whole interview a second time, in a terminal, after the window had already installed — and
    would install twice.
    """
    import recall.desktop.main as desktop_main
    from recall import cli

    opened: list[Any] = []
    monkeypatch.setattr(desktop_main, "install_main", lambda argv: opened.append(argv) or 0)

    def _never(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - the point is it is not hit
        raise AssertionError("the terminal interview ran after the window did")

    monkeypatch.setattr("recall.wizard.interactive.ask_config", _never)

    cli.main(["wizard", "--gui"])

    assert opened == [[]], "the graphical installer must be the only thing that runs"


def test_gui_with_a_config_is_refused_rather_than_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flag that is silently discarded is how somebody installs the wrong thing.

    `--gui` asks the questions itself, so there is nothing for `--config` to do. Accepting both and
    honouring one is the shape this CLI already refuses for `--tenant` and `--table`.
    """
    from recall import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["wizard", "--gui", "--config", "anything.json"])

    assert "nothing for `--config` to do" in str(excinfo.value)
