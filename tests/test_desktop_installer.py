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


def test_the_selftest_reaches_the_engine_a_packaged_build_would_be_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ A frozen bundle's characteristic failure is `ModuleNotFoundError` at the moment of use.

    PyInstaller finds imports by static analysis. This codebase imports lazily nearly everywhere —
    `from recall.wizard.headless import run_headless` inside a function, and so on down the whole
    engine — so a bundle missing half of recall starts fine, draws its window, and dies the instant
    somebody presses Install. `--help` cannot catch that: it exits at argparse.

    So `--selftest` has to touch the modules the FIRST CLICK touches, not just the ones the window
    needs to appear. Asserted on which imports it performs, because a self-test that only proves the
    window opens is a self-test that passes on the broken bundle.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import ast
    import inspect
    import textwrap

    from recall.desktop.main import _selftest

    tree = ast.parse(textwrap.dedent(inspect.getsource(_selftest)))
    # ⛔ **`ast.Import` as well as `ast.ImportFrom`.** This collected only the `from x import y`
    # form, so the three plain `import fastembed` / `onnxruntime` / `tokenizers` lines added to
    # cover the native payload were invisible to it: all three could be deleted with the suite
    # green, and the whole stated point of that change ("a bundle missing all three passed the
    # selftest and died on that click") was unpinned.
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    imported |= {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    for engine_module in (
        "recall.wizard.headless",
        "recall.wizard.pipeline",
        "recall.wizard.uninstall",
        "recall.desktop.install_ui",
    ):
        assert engine_module in imported, (
            f"{engine_module} is reached only from inside a callback in normal use, which is "
            "exactly why a bundle can be missing it and still look healthy"
        )
    for native in ("fastembed", "onnxruntime", "tokenizers"):
        assert native in imported, (
            f"{native} ships native binaries and data files that PyInstaller's static analysis "
            "misses; importing it is what proves the payload made it into the bundle"
        )


def test_the_selftest_passes_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It runs unattended in CI, so it must neither block on an event loop nor provision anything."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)

    from recall.desktop.main import install_main

    assert install_main(["--selftest"]) == 0
    assert list(tmp_path.iterdir()) == [], "a self-test that writes files is not a self-test"


def test_the_selftest_fails_loudly_when_the_engine_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure it exists to catch must produce a non-zero exit and name the module.

    Simulated by making the engine import raise, which is what a bundle without it does.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import builtins

    real_import = builtins.__import__

    def _missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "recall.wizard.pipeline":
            raise ModuleNotFoundError("No module named 'recall.wizard.pipeline'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing)

    from recall.desktop.main import _selftest

    assert _selftest() == 1


def test_a_failing_gui_install_still_reaches_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero status must propagate; a successful one must not exit through an exception.

    ⚠️ Both halves are here because the first fix got the second one wrong. `main()` is typed to
    return None, so `return install_main(...)` is a type error — and the obvious repair, raising
    `SystemExit` unconditionally, made a SUCCESSFUL run exit through an exception, which no other
    branch of that function does. The test asserting the terminal flow does not also run caught it.
    """
    import recall.desktop.main as desktop_main
    from recall import cli

    monkeypatch.setattr(desktop_main, "install_main", lambda argv: 3)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["wizard", "--gui"])

    assert excinfo.value.code == 3, "the installer's failure must reach the shell"


#: An obvious placeholder, never a realistic-looking string. These tests assert that a value does
#: NOT reach the screen, so the value has to be greppable and unmistakably fake to whoever reads the
#: failure output.
_PLACEHOLDER_PASSWORD = "CHANGEME-placeholder-password"


def test_a_probe_error_never_shows_the_password(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ SEC-001: a driver error quotes the connection string back verbatim.

    `probe_database` scrubs what it RETURNS and not what it RAISES, and the identical handler in
    `recall/desktop/ui.py` already scrubs for a measured reason:
    `ProgrammingError: missing "=" after "not-a-dsn://user:PASSWORD@x"`. I wrote a second copy of
    that handler and left the scrubbing out. This is the surface most likely to hit it, because it
    is where somebody types a connection string for the first time and it is what gets screenshotted.
    """

    def _leaky(dsn: str, expected_dimension: int | None) -> Any:
        raise RuntimeError(f'missing "=" after "{dsn}"')

    window = _installer(monkeypatch, tmp_path, prober=_leaky)
    try:
        window._fields["database"].set_value("existing")
        window._fields["dsn"].set_value(f"postgresql://user:{_PLACEHOLDER_PASSWORD}@host/db")

        window._test_dsn()

        shown = window.form_status.text()
        assert _PLACEHOLDER_PASSWORD not in shown, f"the password reached the screen: {shown!r}"
    finally:
        window.close()


def test_an_install_failure_never_shows_the_password(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ SEC-002: `run_headless` scrubs what it packages; an exception escaping it arrives raw.

    The progress log is what a person screenshots when an install fails, and they reached that
    screen by typing a connection string a moment earlier.
    """
    window = _installer(monkeypatch, tmp_path)
    try:
        window._fields["database"].set_value("existing")
        window._fields["dsn"].set_value(f"postgresql://user:{_PLACEHOLDER_PASSWORD}@host/db")

        window._failed(
            f'connection to "postgresql://user:{_PLACEHOLDER_PASSWORD}@host/db" failed'
        )

        shown = window.log.toPlainText()
        assert _PLACEHOLDER_PASSWORD not in shown, f"the password reached the log: {shown!r}"
        assert "failed" in shown, "and the diagnosis must survive the scrubbing"
    finally:
        window.close()


def test_the_selftest_does_not_provision_a_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """⛔ **A self-test that downloads 100MB of weights is not a self-test.**

    `resolve_embedder("fastembed")` constructs a real `TextEmbedding` eagerly, so on a cold cache it
    fetches BAAI/bge-small-en-v1.5 before it can answer; measured warm at 6.76s. Adding it gave the
    test above a silent network dependency and made its `tmp_path` emptiness assertion a false
    negative, because the weights land in the HuggingFace cache rather than in `tmp_path`. This
    repository already carries one network-dependent test whose failure is indistinguishable from a
    regression, and a second was the wrong trade for a check the three imports mostly cover.

    So the resolution runs only when a cache already exists — and the skip is PRINTED, because a
    gate that was skipped and a gate that passed must never render the same.
    """
    from recall.desktop import main as desktop_main

    resolved: list[str] = []

    def never_resolve(name: str):  # pragma: no cover - the assertion is that this is not called
        resolved.append(name)
        raise AssertionError("the selftest must not construct an embedder with no cache present")

    monkeypatch.setattr(desktop_main, "_model_cache_exists", lambda: False)
    monkeypatch.setattr("recall.embeddings.resolve_embedder", never_resolve)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("PySide6")

    assert desktop_main._selftest() == 0
    assert resolved == [], "no cache means no model is provisioned"
    assert list(tmp_path.iterdir()) == [], "and nothing is written where the test can see it"


def test_the_selftest_says_which_branch_it_took(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skipped check that prints nothing is indistinguishable from a check that passed."""
    from recall.desktop import main as desktop_main

    monkeypatch.setattr(desktop_main, "_model_cache_exists", lambda: False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("PySide6")

    desktop_main._selftest()
    assert "no local model cache" in capsys.readouterr().err, (
        "the reader has to be able to tell a skipped embedder check from a passing one"
    )


def test_the_installer_window_never_waits_forever_to_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """⛔ `QThreadPool.waitForDone()` with NO argument waits forever, and a parented pool's
    destructor calls exactly that.

    `MainWindow` was given a bounded `closeEvent` after closing during a first install froze the
    application for up to half an hour. This window runs a strictly longer job — image pull,
    migrations, a generation and a calibration per corpus — and its own bounded handler shipped with
    no test, so a regression to the unbounded form would surface as FLAKINESS rather than a failure:
    the wait only exceeds a test timeout when Docker happens to be busy.

    Three properties: the wait is bounded, the bound is not absurd, and the window closes even when
    the wait itself raises.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.install_ui import InstallerWindow
    from recall.desktop.jobs import CLOSE_WAIT_MS

    QApplication.instance() or QApplication([])

    class _Event:
        def __init__(self) -> None:
            self.accepted = False

        def accept(self) -> None:
            self.accepted = True

    window = InstallerWindow()
    try:
        waits: list[object] = []
        monkeypatch.setattr(
            window.pool, "waitForDone", lambda *args: waits.append(args) or True, raising=False
        )
        window._jobs.append(object())

        event = _Event()
        window.closeEvent(event)

        assert waits and waits[0], "the wait must be given a bound; no argument waits forever"
        bound = waits[0][0]
        assert isinstance(bound, int) and 0 < bound <= 10_000, (
            f"a close that waits {bound}ms is a window that will not close"
        )
        assert bound == CLOSE_WAIT_MS, "the same constant the main window uses"
        assert event.accepted

        # And it closes even when the wait blows up: without try/finally the error path of this
        # very fix reproduces the hang it exists to prevent.
        def explode(*_args: object) -> bool:
            raise RuntimeError("the pool is gone")

        monkeypatch.setattr(window.pool, "waitForDone", explode, raising=False)
        broken = _Event()
        with pytest.raises(RuntimeError):
            window.closeEvent(broken)
        assert broken.accepted, "the window must close even when the wait itself fails"
    finally:
        # The teardown close would otherwise re-enter the still-exploding handler.
        window._jobs.clear()
        monkeypatch.setattr(window.pool, "waitForDone", lambda *args: True, raising=False)
        window.close()


def test_both_dsn_forms_have_their_password_scrubbed() -> None:
    """⛔ libpq accepts two DSN forms and the installer's field is free text; only one was covered.

    `urlsplit("host=db password=s3cret dbname=recall").password` is `None`, so the keyword form —
    which psycopg accepts, and which is what somebody pasting from a hosting provider's console
    often has — went through with the password intact while the caller's docstring said it had been
    removed. A promise that holds for one input shape and silently fails for another is worse than
    no promise, because the caller stops looking.
    """
    from recall.store import scrub_dsn_secrets

    assert scrub_dsn_secrets("failed for s3cret", "postgresql://u:s3cret@h/db") == "failed for ***"
    assert (
        scrub_dsn_secrets('missing "=" after password=s3cret x', "host=db password=s3cret dbname=r")
        == 'missing "=" after password=*** x'
    )
    # A quoted value with spaces is reachable through libpq and must be matched in full, not up to
    # the first space.
    assert (
        scrub_dsn_secrets("error: pw with spaces", "host=db password='pw with spaces' dbname=x")
        == "error: ***"
    )

    # It must never raise, on any input: this runs while REPORTING an error, and the malformed-DSN
    # case is the one that produced the leak it exists to stop.
    assert scrub_dsn_secrets("weird [dsn", "postgresql://[unbalanced") == "weird [dsn"
    assert scrub_dsn_secrets("untouched", "") == "untouched"
    assert scrub_dsn_secrets("untouched", "host=db dbname=r") == "untouched"


def test_a_scrubbed_message_still_carries_its_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """⛔ Asserting only that the password is ABSENT passes for a helper that returns nothing useful.

    `_scrubbed` has a fallback that replaces the whole message when the scrub raises. The tests for
    it asserted only `password not in shown`, which that fallback satisfies while losing every word
    of the diagnosis. An absence assertion needs a presence assertion beside it or it cannot tell
    redaction from destruction.
    """
    from recall.desktop.install_ui import _scrubbed

    dsn = "postgresql://recall:hunter2@127.0.0.1:5432/recall"
    message = "Could not check that database: password authentication failed for hunter2"

    shown = _scrubbed(message, dsn)
    assert "hunter2" not in shown
    assert "Could not check that database" in shown, "the diagnosis must survive the redaction"
    assert "password authentication failed" in shown

    # A blank DSN returns the message unchanged rather than suppressing it.
    assert _scrubbed(message, "") == message

    # And when the scrub itself fails, the message is suppressed rather than leaked.
    def explode(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("scrubber is broken")

    monkeypatch.setattr("recall.store.scrub_dsn_secrets", explode)
    suppressed = _scrubbed(message, dsn)
    assert "hunter2" not in suppressed
    assert "could not be displayed safely" in suppressed
