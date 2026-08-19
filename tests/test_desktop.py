from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

import recall.desktop.profiles as profiles
from recall.desktop.models import RuntimeMode, RuntimeProfile, SourceCategory, SourceSelection
from recall.desktop.runtime import DockerRuntime, RuntimeErrorBase, VpsMcpRuntime, create_runtime
from recall.desktop.sources import (
    classify,
    collect_files,
    default_scan_roots,
    display_type,
    scan_files,
)
from recall.desktop.updates import is_newer
from recall.desktop.uploads import stage_uploads


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if name == "recall_stats":
            return {"chunks": 3, "stale": False}
        if name == "recall_ingest":
            return {"job_id": "job-1", "state": "completed", "files": 1, "chunks": 4}
        if name == "recall_job_status":
            return {"job_id": "job-1", "state": "unknown"}
        if name == "recall_calibration_status":
            return {"status": "missing", "generation_id": None}
        if name == "recall_tenants":
            return {"tenants": ["default", "acme"]}
        return {}

    def close(self) -> None:
        return None


def test_source_categories_and_physical_tenants(tmp_path: Path) -> None:
    code = tmp_path / "app.py"
    memory = tmp_path / "fact.md"
    code.write_text("print('ok')", encoding="utf-8")
    memory.write_text("a durable fact", encoding="utf-8")

    assert classify(code) is SourceCategory.CODE
    assert classify(memory) is SourceCategory.DOCUMENTS
    assert SourceSelection(SourceCategory.CODE, (code,), "acme").physical_tenant == "acme-code"
    # 🔁 This asserted `user-docs` until 2026-08-19, and the assertion was the bug.
    # The wizard builds THREE corpora per project and memory belongs in the writable, never
    # calibrated one; mapping MEMORY onto `-docs` sent the user's memory into the corpus that is
    # production-routed, strict-trust and calibrated — the wrong destination, and the one place a
    # stray write does the most damage.
    assert (
        SourceSelection(SourceCategory.MEMORY, (memory,), "acme").physical_tenant == "acme-memory"
    )
    # The shared scope comes from the profile now, not a literal "user". A profile naming a
    # different shared scope used to ingest into `user-*` while display and calibration used the
    # configured one, so the two halves silently disagreed about which corpus was being filled.
    shared = SourceSelection(
        SourceCategory.DOCUMENTS, (memory,), "acme", True, shared_profile="everyone"
    )
    assert shared.physical_tenant == "everyone-docs"
    assert classify(tmp_path / "report.pdf") is SourceCategory.DOCUMENTS
    assert classify(tmp_path / "report.docx") is SourceCategory.DOCUMENTS
    assert classify(tmp_path / "metrics.xlsx") is SourceCategory.DOCUMENTS
    assert display_type(tmp_path / "report.pdf", SourceCategory.DOCUMENTS) == "PDF"


def test_collect_files_filters_by_category(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"x")

    assert [path.name for path in collect_files([tmp_path], SourceCategory.CODE)] == ["a.py"]
    assert [path.name for path in collect_files([tmp_path], SourceCategory.DOCUMENTS)] == ["b.md", "c.pdf"]


def test_local_scan_prunes_generated_directories_and_limits_claude_files(tmp_path: Path) -> None:
    documents = tmp_path / "Documents"
    claude = tmp_path / ".claude"
    (documents / ".git").mkdir(parents=True)
    (documents / "node_modules").mkdir()
    (documents / "notes.md").write_text("notes", encoding="utf-8")
    (documents / "app.py").write_text("print('ok')", encoding="utf-8")
    (documents / ".git" / "ignored.py").write_text("ignored", encoding="utf-8")
    (documents / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
    (claude / "projects" / "demo" / "memory").mkdir(parents=True)
    (claude / "MEMORY.md").write_text("memory", encoding="utf-8")
    (claude / "projects" / "demo" / "memory" / "notes.md").write_text(
        "memory", encoding="utf-8"
    )
    (claude / "settings.json").write_text("{}", encoding="utf-8")

    assert default_scan_roots(tmp_path) == (documents, claude)
    found = scan_files(default_scan_roots(tmp_path))

    assert [path.relative_to(tmp_path).as_posix() for path in found] == [
        "Documents/app.py",
        "Documents/notes.md",
        ".claude/MEMORY.md",
        ".claude/projects/demo/memory/notes.md",
    ]


def test_vps_runtime_uses_mcp_contract(tmp_path: Path) -> None:
    profile = RuntimeProfile(mode=RuntimeMode.VPS_MCP, endpoint="https://example.test/mcp")
    gateway = FakeGateway()
    runtime = VpsMcpRuntime(profile, gateway)
    runtime.start()

    source = tmp_path / "memo.md"
    source.write_text("memory", encoding="utf-8")
    job = runtime.start_ingest(SourceSelection(SourceCategory.MEMORY, (source,), "default"))

    assert job.job_id == "job-1"
    name, arguments = gateway.calls[-1]
    assert name == "recall_ingest"
    # 🔁 Was `default-docs`. A MEMORY selection must reach the memory corpus; see
    # `test_source_categories_and_physical_tenants` for why the old value was the defect.
    assert arguments["tenant"] == "default-memory"
    assert base64.b64decode(arguments["files"][0]["content_b64"]) == b"memory"
    assert runtime.job_status(job.job_id).state == "completed"


def test_staging_follows_the_configured_index_root_and_never_the_checkout() -> None:
    """`stage_uploads` writes under `RECALL_INDEX_ROOT`, and this suite's root is not the checkout.

    Two claims, deliberately in one test, because either alone is satisfiable by the defect:

    * that the staging root is derived from `RECALL_INDEX_ROOT` at all, the production behaviour
      documented in docs/SECURITY_MODEL.md and NOT changed here;
    * that the value this suite runs with points somewhere disposable. The variable's documented
      default is `.`, the working directory, which for a test session is the repository. Left unset,
      every test that reached `stage_uploads` decoded its upload into `uploads/<tenant>/<job_id>/`
      at the repository root and left it there, untracked, once per run.

    The second assertion is the regression guard for `tests/conftest.py::_confine_index_root`, and
    the reason it lives in a test rather than only in that fixture: an autouse fixture that stops
    doing its job breaks nothing and fails nothing. Delete the `setenv` and this goes red, naming
    the fixture.
    """
    configured = os.environ.get("RECALL_INDEX_ROOT")
    assert configured, (
        "RECALL_INDEX_ROOT is unset, so stage_uploads falls back to the working directory. "
        "tests/conftest.py::_confine_index_root is meant to set it for every test."
    )
    root = Path(configured).resolve()
    checkout = Path(__file__).resolve().parent.parent
    assert root != checkout and checkout not in root.parents, (
        f"RECALL_INDEX_ROOT resolves to {root}, inside the checkout at {checkout}; uploads staged "
        f"during the suite would land in the working tree and show up in `git status`"
    )

    payload = base64.b64encode(b"memory").decode("ascii")
    job_id, staged = stage_uploads("acme", [{"name": "memo.md", "content_b64": payload}])

    assert staged == root / "uploads" / "acme" / job_id
    assert (staged / "memo.md").read_bytes() == b"memory"


def test_runtime_factory_and_calibration_status() -> None:
    profile = RuntimeProfile(mode=RuntimeMode.VPS_MCP, endpoint="https://example.test/mcp")
    gateway = FakeGateway()
    runtime = create_runtime(profile, gateway)
    runtime.start()
    status = runtime.calibration_status("default-docs")

    assert isinstance(runtime, VpsMcpRuntime)
    assert status.status == "missing"
    assert runtime.list_tenants() == ["default", "acme"]


def test_docker_profile_requires_compose_file() -> None:
    with pytest.raises(ValueError, match="compose file"):
        RuntimeProfile(mode=RuntimeMode.DOCKER)


def test_the_desktop_and_the_wizard_agree_on_corpus_kinds() -> None:
    """The desktop's category-to-kind map must cover exactly the wizard's `CorpusKind`.

    These are two halves of one agreement written in two packages. The half that was wrong sent
    MEMORY to `-docs`, and nothing failed because nothing compared them. Pinned here so a fourth
    kind cannot be added on one side alone.
    """
    from typing import get_args

    from recall.desktop.models import _KIND_BY_CATEGORY
    from recall.wizard.corpora import CorpusKind

    assert set(_KIND_BY_CATEGORY.values()) == set(get_args(CorpusKind))
    assert set(_KIND_BY_CATEGORY) == set(SourceCategory), "every category must map to a corpus"


def test_the_shared_scope_is_offered_only_when_it_can_be_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"All projects (shared memory)" pointed at a tenant no wizard install provisions.

    The wizard builds `<project>-docs/code/memory` and never a `user-*` scope, so this permanent
    menu entry resolved to a service that does not exist and refused whenever anyone chose it. The
    legacy `docker-compose.desktop.yml` DOES define `recall-user-docs`, which is why the entry
    looked fine there and only broke on the installs the wizard produces.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.ui import MainWindow

    class UiRuntime:
        def __init__(self, servable: set[str]) -> None:
            self._servable = servable

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["default"]

        def _service_for_tenant(self, scope: str) -> str:
            if scope not in self._servable:
                raise RuntimeErrorBase(f"no service for {scope}")
            return f"recall-{scope}"

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])

    wizard_stack = MainWindow(profile, runtime=UiRuntime({"default-docs", "default-memory"}))
    try:
        labels = [wizard_stack.scope.itemText(i) for i in range(wizard_stack.scope.count())]
        assert "All projects (shared memory)" not in labels, (
            "a scope the stack cannot serve must not be offered"
        )
    finally:
        wizard_stack.close()
        app.processEvents()

    legacy = MainWindow(profile, runtime=UiRuntime({"default-docs", "user-docs", "user-code"}))
    try:
        labels = [legacy.scope.itemText(i) for i in range(legacy.scope.count())]
        assert "All projects (shared memory)" in labels, "a legacy install must keep the entry"
    finally:
        legacy.close()
        app.processEvents()


def test_the_calibration_page_reports_on_the_corpus_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Memory row used to display the DOCS tenant's calibration.

    `("Memory", "docs")` meant two rows read the same corpus under different names, so the Memory
    row showed a certification belonging to something else — and it read as reassuring, because the
    docs corpus is the one that certifies. The memory corpus is deliberately never calibrated, so
    "missing" is its honest status.

    ⚠️ This test exists because a mutation run caught the gap: reverting the Memory row and
    reverting the shared-scope guard both stayed GREEN across the whole suite, since nothing
    exercised `_calibration_targets` at all.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.ui import MainWindow

    class UiRuntime:
        def __init__(self, servable: set[str]) -> None:
            self._servable = servable

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["default"]

        def _service_for_tenant(self, scope: str) -> str:
            if scope not in self._servable:
                raise RuntimeErrorBase(f"no service for {scope}")
            return f"recall-{scope}"

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])

    window = MainWindow(profile, runtime=UiRuntime({"default-docs", "default-memory"}))
    try:
        targets = window._calibration_targets()
        by_corpus = {corpus: tenant for _label, corpus, tenant in targets}

        assert by_corpus["Memory"] == "default-memory", "the Memory row must read the memory corpus"
        assert by_corpus["Documents"] == "default-docs"
        assert by_corpus["Code"] == "default-code"
        assert len({t for _l, _c, t in targets}) == len(targets), (
            "each row must name a distinct tenant; two rows on one corpus is how the Memory row "
            "came to report another corpus's certification"
        )
        assert not any(label == "All projects" for label, _c, _t in targets), (
            "a scope the stack cannot serve must not get calibration rows either"
        )
    finally:
        window.close()
        app.processEvents()


def test_every_background_job_delivers_its_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_run`'s callbacks were arriving by luck, and every page depends on them.

    ⚠️ `_Worker` is a `QRunnable` with `autoDelete` on, so the instant `run()` returns Qt destroys
    it, taking `_WorkerSignals` with it and purging the queued cross-thread call. Measured on
    PySide6 6.11 with five identical jobs: **1 of 5 arrived**; the anti-regression reviewer measured
    **0 of 5** on the same code. It is a garbage-collection race, not a deterministic failure, which
    is why it survived: it works often enough to look correct.

    This is not specific to provisioning. Connect, ingest, calibration and the GitHub download all
    go through `_run`, so all of them could silently never report. Ten jobs rather than one, because
    a race that fires nine times out of ten is exactly what a single-job test would bless.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import time

    from PySide6.QtWidgets import QApplication

    from recall.desktop.ui import MainWindow

    class UiRuntime:
        def start(self) -> None: ...
        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["default"]

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(profile, runtime=UiRuntime())
    try:
        # ⚠️ **The structural assertion comes first, and it is the one that pins the fix.** The
        # defect is a garbage-collection RACE, so a timing test blesses the broken code whenever
        # the collector is slow: mutating the fix away and running the delivery check below passed.
        # These two properties are what stop the runnable being destroyed before Qt delivers its
        # queued signal, and they hold deterministically.
        held: list[object] = []
        window.pool.start = lambda worker: held.append(worker)  # type: ignore[method-assign]
        window._run(lambda: 1, lambda _value: None)
        assert held, "the job must reach the pool"
        assert held[0].autoDelete() is False, (
            "a QRunnable with autoDelete on is destroyed the moment run() returns, taking its "
            "signals object with it and purging the queued callback"
        )
        # Membership, not equality: the window dispatches its own connect job during construction,
        # so the list is not empty when this test starts.
        assert held[0] in window._workers, "`_run` must keep a reference to the live worker"
        window._workers.clear()
        del window.pool.start

        delivered: list[object] = []
        jobs = 10
        for index in range(jobs):
            window._run(lambda i=index: i, lambda value: delivered.append(value))

        # Waits for the release too, not just the callback. Each connection to a cross-thread
        # signal is its own queued call, so the last `append` can arrive while its matching
        # `_release` is still queued behind it — exiting on the callback alone made this test
        # flaky against correct code.
        deadline = time.time() + 10.0
        while time.time() < deadline and (len(delivered) < jobs or window._workers):
            app.processEvents()
            time.sleep(0.02)

        assert len(delivered) == jobs, (
            f"only {len(delivered)} of {jobs} background jobs reported. `_run` must keep a "
            f"reference to its worker; without one the runnable is deleted before Qt delivers "
            f"the queued signal, and every page that loads data in the background can hang."
        )
        assert window._workers == [], "a finished job must release its worker"
    finally:
        window.close()
        app.processEvents()


def test_a_project_that_was_created_but_would_not_start_is_reported_as_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating and starting are two outcomes, and the third state has to survive.

    ⚠️ Moving provisioning onto the worker pool collapsed them: any failure became "Cannot create
    {name}" and dropped the name from the selector. By then `add_project` has already written the
    compose file and the tenants exist, so that message asserts a state the code did create — and
    a retry then finds nothing to add, reports "already exists", and never starts anything, leaving
    the user with no route back to a running stack. Caught by the anti-regression review, not by a
    test, which is why this one exists.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.ui import MainWindow

    class UiRuntime:
        def start(self) -> None:
            raise RuntimeErrorBase("dependency failed to start: container is unhealthy")

        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["default"]
        def _service_for_tenant(self, scope: str) -> str:
            return f"recall-{scope}"

    class _Added:
        tenants = ("acme-docs", "acme-code", "acme-memory")
        compose_path = Path("docker-compose.recall.yml")

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(profile, runtime=UiRuntime())
    try:
        window._project_names.append("acme")

        # ⚠️ Drive `_do_provision`, NOT `_provision_done`. The first version of this test called
        # the display handler with a hand-built tuple, so it exercised the wording and nothing
        # about the routing: deleting the try/except in `_do_provision` left the whole suite green.
        # The reviewer proved that by mutation. The routing is the finding, so the routing is what
        # this asserts.
        monkeypatch.setattr(
            "recall.wizard.projects.add_project", lambda *a, **k: _Added(), raising=False
        )
        added, start_error = window._do_provision("acme", "docker-compose.desktop.yml")

        assert start_error, "a failing start must be RETURNED, not raised into _provision_failed"
        assert added is not None, "the project object must survive a start failure"

        window._provision_done("acme", (added, start_error))
        text = window.status.text()

        assert "exists" in text.lower(), f"a written project must not be reported as absent: {text}"
        assert "cannot create" not in text.lower(), (
            "the compose file was already written, so 'cannot create' is a false claim"
        )
        assert "acme" in window._project_names, "a created project must stay in the selector"
        assert window.scope.isEnabled(), "every outcome must re-enable the scope selector"
    finally:
        window.close()
        app.processEvents()


def test_provisioning_is_dispatched_to_the_pool_not_the_gui_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-inlining the provision would restore the frozen window with a fully green suite.

    `_provision_project` used to call `add_project` and `runtime.start()` directly inside a
    `currentIndexChanged` slot, so the window stopped repainting and Windows marked it
    "(Not Responding)" for the length of a compose `up --wait` — which on a first start includes
    building the image. Four auditors reported it. The sibling test drives `_do_provision`
    directly, so it says nothing about HOW `_do_provision` is reached; this asserts the dispatch.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.ui import MainWindow

    def _never(*args: object, **kwargs: object) -> object:
        raise AssertionError("add_project ran on the GUI thread; it must go through the pool")

    monkeypatch.setattr("recall.wizard.projects.add_project", _never, raising=False)

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(profile, runtime=DockerRuntime(profile))
    try:
        dispatched: list[tuple[object, ...]] = []
        window._run = lambda fn, done, failed=None: dispatched.append((fn, done, failed))  # type: ignore[method-assign]

        window._provision_project("acme")

        assert dispatched, "provisioning must be handed to the worker pool, not run inline"
        assert all(callable(part) for part in dispatched[0]), "fn, done and failed are all callables"
        assert window.scope.isEnabled() is False, (
            "the scope selector must be disabled while the job is outstanding"
        )
    finally:
        window.close()
        app.processEvents()


def test_a_retry_starts_the_stack_even_when_nothing_new_was_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stranding case: a retry after a failed start finds nothing to add.

    ⚠️ The old code returned before starting when `created_anything` was False, so a user whose
    first attempt created the project but failed to start it was told "already exists" forever,
    with the services still down and no route back. `start()` is therefore called unconditionally.
    The sibling test's fixture has three tenants, so it cannot see this: reintroducing the guard
    would leave it green.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from recall.desktop.ui import MainWindow

    class _AlreadyThere:
        tenants: tuple[str, ...] = ()
        already_present = ("acme-docs", "acme-code", "acme-memory")
        compose_path = Path("docker-compose.recall.yml")

    starts: list[int] = []

    class UiRuntime:
        def start(self) -> None:
            starts.append(1)

        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["default"]

    monkeypatch.setattr(
        "recall.wizard.projects.add_project", lambda *a, **k: _AlreadyThere(), raising=False
    )

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(profile, runtime=UiRuntime())
    try:
        # DRAIN the window's own connect job first. It calls `start()` on a worker thread and can
        # land at any moment, so neither an absolute count nor a delta measured around the call is
        # stable — both were flaky before this loop.
        import time as _time

        deadline = _time.time() + 10.0
        while _time.time() < deadline and window._workers:
            app.processEvents()
            _time.sleep(0.02)
        app.processEvents()

        before = len(starts)
        added, start_error = window._do_provision("acme", "docker-compose.recall.yml")

        assert len(starts) - before == 1, (
            "a retry must still start the stack; skipping it when nothing was created is what "
            "left the user with a created project and no way to run it"
        )
        assert not start_error
        window._provision_done("acme", (added, start_error))
        assert "already exists" in window.status.text().lower()
    finally:
        window.close()
        app.processEvents()


def test_a_compose_failure_reports_dockers_own_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """A build failure, a pull denial and an unhealthy dependency were one identical sentence.

    `str(CalledProcessError)` is only "Command '[...]' returned non-zero exit status 1", and the
    stderr had already been captured and thrown away — on exactly the paths that now build images
    and provision projects. `stack.bring_up` already extracted it; this copies that.
    """
    import subprocess as _subprocess

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="whatever.yml")
    runtime = DockerRuntime(profile)

    def failing(command, **kwargs):  # noqa: ANN001, ANN003
        raise _subprocess.CalledProcessError(
            1, command, output="", stderr="dependency failed to start: container is unhealthy"
        )

    monkeypatch.setattr(_subprocess, "run", failing)
    with pytest.raises(RuntimeErrorBase) as failure:
        runtime._compose("up", "-d", "--wait")
    assert "container is unhealthy" in str(failure.value), (
        "the captured stderr must reach the user; without it every compose failure reads alike"
    )

    def timing_out(command, **kwargs):  # noqa: ANN001, ANN003
        raise _subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(_subprocess, "run", timing_out)
    with pytest.raises(RuntimeErrorBase) as timeout:
        runtime._compose("up", "-d", "--wait")
    text = str(timeout.value)
    assert "did not finish" in text and "1800" in text, (
        "a timeout must say so, and name the budget, rather than reading as a failure"
    )


def test_an_invalid_profile_does_not_stop_the_app_opening(tmp_path: Path) -> None:
    """A file that PARSES but does not validate must fall back, not raise.

    ⚠️ `load_profile` caught `(OSError, ValueError)` around `json.loads` and then constructed the
    `RuntimeProfile` OUTSIDE the guard — and `from_dict` raises `ValueError` too: an unknown
    `RuntimeMode`, a docker profile with no compose file, an empty `default_tenant`. So an invalid
    profile escaped through the unguarded call in `main.py` and the application would not open,
    which is precisely what the sibling `load_pipelines` docstring promises a bad settings file
    never causes.
    """
    import json as _json

    bad = tmp_path / "runtime.json"

    bad.write_text(_json.dumps({"mode": "docker"}), encoding="utf-8")  # no compose_file
    assert profiles.load_profile(bad) is None, "a docker profile with no compose file must not raise"

    bad.write_text(_json.dumps({"mode": "not-a-mode"}), encoding="utf-8")
    assert profiles.load_profile(bad) is None, "an unknown runtime mode must not raise"

    bad.write_text("not json at all", encoding="utf-8")
    assert profiles.load_profile(bad) is None

    good = tmp_path / "good.json"
    good.write_text(
        _json.dumps({"mode": "docker", "compose_file": "docker-compose.desktop.yml"}),
        encoding="utf-8",
    )
    assert profiles.load_profile(good) is not None, "a valid profile must still load"


def test_a_project_name_drops_every_corpus_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """`-memory` was missing, which is the same omission `runtime.py` records making three times.

    A runtime that returns raw tenant names — the base `RuntimeManager.list_tenants` does, reading
    `recall_tenants` — then yields a phantom project called `<project>-memory`, and the calibration
    page builds `<project>-memory-docs` from it.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from recall.desktop.ui import _project_name

    assert _project_name("acme-docs") == "acme"
    assert _project_name("acme-code") == "acme"
    assert _project_name("acme-memory") == "acme", "the memory suffix must be dropped too"
    assert _project_name("acme") == "acme"


def test_a_build_gets_a_longer_budget_than_a_status_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One 120s timeout governed every compose verb, and it could not cover the work.

    The generated stack gives each tenant service a `build:` stanza and nothing in the install path
    ever builds it, so the desktop's first `up` builds LibreOffice plus a PyPI install. Separately
    the database healthcheck carries `start_period: 180s`, already longer than the old cap, so
    `up --wait` was killed while Compose was still legitimately waiting. Five auditors reached this
    from five directions.
    """
    import subprocess as _subprocess

    from recall.desktop.runtime import _QUICK_VERB_TIMEOUT, _SLOW_VERB_TIMEOUT

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="whatever.yml")
    runtime = DockerRuntime(profile)
    seen: dict[str, int] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        seen[command[command.index("whatever.yml") + 1]] = kwargs["timeout"]
        return _subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_subprocess, "run", fake_run)

    runtime._compose("up", "-d", "--wait")
    runtime._compose("config", "--services")

    assert seen["up"] == _SLOW_VERB_TIMEOUT
    assert seen["config"] == _QUICK_VERB_TIMEOUT
    assert seen["up"] > 240, (
        "an `up` must outlast the healthcheck it waits on: start_period 180s plus "
        "interval 2s x retries 30"
    )


def test_the_corpus_suffixes_match_the_wizards_kinds() -> None:
    """The desktop's scope suffixes must equal the wizard's `CorpusKind`, or services go missing.

    Pinned rather than imported: `recall.desktop` keeps no import dependency on `recall.wizard`,
    but a divergence between them is exactly the bug this branch exists to remove, so it fails a
    test instead of waiting to be noticed. The first version of the topology change spelled
    `("-docs", "-code")` inline in three places and omitted `-memory`, which every install builds,
    so `recall-<project>-memory` was excluded from the schema loop and from the project list.
    """
    from typing import get_args

    from recall.desktop.runtime import _CORPUS_SUFFIXES
    from recall.wizard.corpora import CorpusKind

    assert set(_CORPUS_SUFFIXES) == {f"-{kind}" for kind in get_args(CorpusKind)}


def test_pipeline_configuration_round_trips_without_a_gui() -> None:
    """The persistence contract, testable where CI can actually run it.

    The MainWindow-driven test beside this one is skipped in CI: no job installs the `desktop`
    extra, so `pytest.importorskip("PySide6")` skips it on every run and the regression test for
    "Configuration saved" saved nothing would only ever have run on a maintainer's machine. This
    covers the same contract with no Qt, including the layer-over-defaults merge the UI relies on.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "pipeline.json"

        assert profiles.load_pipelines(target) == {}, "absent file must not raise"

        profiles.save_pipelines({"Documents": {"embedder": "chosen", "splade": True}}, target)
        assert profiles.load_pipelines(target) == {"Documents": {"embedder": "chosen", "splade": True}}

        defaults = {"embedder": "hashing-64", "reranker": "none", "splade": False}
        defaults.update(profiles.load_pipelines(target)["Documents"])
        assert defaults == {"embedder": "chosen", "reranker": "none", "splade": True}, (
            "a key absent from the saved file must keep its default rather than vanish"
        )

        target.write_bytes(b"\xff\xfe not utf-8 at all")
        assert profiles.load_pipelines(target) == {}, (
            "a file that is not valid UTF-8 must not raise; UnicodeDecodeError is a ValueError "
            "and was not caught, so it propagated out of MainWindow.__init__ and the app would "
            "not open, which is the outcome load_pipelines documents that it prevents"
        )

        target.write_text("[1, 2, 3]", encoding="utf-8")
        assert profiles.load_pipelines(target) == {}, "a non-object document must not raise"


def test_docker_runtime_reads_its_topology_from_the_compose_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The UI could not drive the stack the wizard installs, and this is why.

    `_service_for_tenant` was a literal four-entry dict mapping `default-docs` to `recall-docs`.
    The wizard names every service `recall-<tenant>`, so it writes `recall-default-docs`. Measured
    against a generated compose document, EVERY scope missed, the default one included, so the
    mismatch was never confined to projects the user had added.

    Three claims from one fake service list, because any one alone is satisfiable by the defect:
    the wizard's naming must resolve, an unknown scope must be refused with what IS on offer, and
    `list_tenants` must report what the file contains rather than a constant. The legacy names are
    covered by `test_docker_runtime_still_serves_the_pre_wizard_compose_file`, which needs a
    different fixture — this one contains no legacy name, so it cannot assert anything about them.
    """
    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="whatever.yml")
    runtime = DockerRuntime(profile)
    monkeypatch.setattr(
        runtime,
        "_service_names",
        lambda: frozenset({"db", "recall-default-docs", "recall-default-code", "recall-acme-docs"}),
    )

    assert runtime._service_for_tenant("default-docs") == "recall-default-docs"
    assert runtime._service_for_tenant("acme-docs") == "recall-acme-docs"
    assert runtime.list_tenants() == ["acme", "default"]

    with pytest.raises(RuntimeErrorBase) as refusal:
        runtime._service_for_tenant("nothere-docs")
    # Scopes, not service names: the refusal answers in the vocabulary of the argument it refused,
    # and it is built from the same predicate the rest of the class uses, so it can never advertise
    # a sidecar the compose file happens to carry as something this call accepts.
    assert "acme-docs" in str(refusal.value), "a refusal must say what IS on offer"
    assert "recall-acme-docs" not in str(refusal.value), "and say it as a scope, not a service"
    assert "wizard" in str(refusal.value), "and what would make the scope servable"


def test_docker_start_applies_the_schema_only_where_a_tenant_lives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`start()` must not run `recall schema apply` inside every non-database service.

    The first version of this loop was `services - {"db"}`, which would reach any sidecar the
    compose file happens to carry. `_compose` runs with check=True, so one unrelated service would
    fail the whole start, and the stack would look broken because of a container nothing needed.
    """
    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="whatever.yml")
    runtime = DockerRuntime(profile)
    calls: list[tuple[str, ...]] = []

    def fake_compose(*args: str):
        calls.append(args)
        return None

    monkeypatch.setattr(runtime, "_compose", fake_compose)
    monkeypatch.setattr(
        runtime,
        "_service_names",
        lambda: frozenset(
            {"db", "recall-default-docs", "recall-default-code", "recall-default-memory",
             "otel-collector", "recall-backup"}
        ),
    )
    monkeypatch.setattr(runtime, "health", lambda: {"status": "ready"})
    monkeypatch.setattr(runtime, "_call_for", lambda *a, **k: {})

    runtime.start()

    applied = [args[2] for args in calls if "schema" in args]
    # ONCE, not once per service. `schema apply` migrates a database and every service in the stack
    # shares one migration DSN, so the second and later applies were identical redundant work that
    # made startup scale with the number of projects: roughly 3-6s per service, sequentially.
    assert len(applied) == 1, f"schema apply ran {len(applied)} times: {applied}"
    # And in a service that actually serves a tenant. `services - {"db"}` would have reached the
    # sidecars, and `_compose` runs with check=True, so one unrelated service fails the whole start.
    assert applied[0].startswith("recall-default-"), f"schema apply reached {applied}"


def test_restarting_the_stack_forgets_the_routing_it_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached gateway must not outlive the topology it was resolved from.

    `_gateways` memoises a fully resolved `docker compose exec <service> ...` argv, not a lookup.
    Clearing only `_services` left the two disagreeing: re-run the wizard while the window is open,
    press Reconnect, and the refreshed topology would be correct while every already-connected
    tenant kept writing to the service resolved before. Since the chosen service IS the corpus on
    this path, that is a silent misroute of ingest, not merely a stale name.

    Asserted through `_compose` rather than through `start()`, because `apply_update()` restarts
    the stack too and was missed the first time; the invalidation belongs to the mutating verb.
    """
    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="whatever.yml")
    runtime = DockerRuntime(profile)
    monkeypatch.setattr(runtime, "_service_names", lambda: frozenset({"db", "recall-a-docs"}))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)

    closed: list[str] = []

    class Gateway:
        def close(self) -> None:
            closed.append("closed")

    runtime._services = frozenset({"db", "recall-a-docs"})
    runtime._gateways["a-docs"] = Gateway()  # type: ignore[assignment]

    runtime._compose("up", "-d", "--wait")

    assert runtime._services is None, "the topology cache must be dropped"
    assert runtime._gateways == {}, "and so must the routing derived from it"
    assert closed == ["closed"], "a dropped gateway must be closed, not leaked"


def test_docker_start_refuses_a_stack_with_no_tenant_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying no migrations at all must fail loudly, not report ready.

    The hardcoded loop this replaced failed here by accident: `check=True` hit a missing service.
    A derived loop instead matches nothing, applies nothing, and falls through to `health()`, so a
    stack that can serve nothing would have looked like a clean start.
    """
    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="whatever.yml")
    runtime = DockerRuntime(profile)
    monkeypatch.setattr(runtime, "_compose", lambda *args: None)
    monkeypatch.setattr(runtime, "_service_names", lambda: frozenset({"db", "otel-collector"}))
    monkeypatch.setattr(runtime, "health", lambda: {"status": "ready"})

    with pytest.raises(RuntimeErrorBase, match="no tenant-serving MCP service"):
        runtime.start()


def test_docker_runtime_still_serves_the_pre_wizard_compose_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`docker-compose.desktop.yml` shipped before the wizard and names services differently.

    An existing install must keep working, so the legacy names resolve too. `recall-docs` must map
    back to the `default` project and not invent one called `docs`, which a naive prefix strip does.
    """
    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    runtime = DockerRuntime(profile)
    monkeypatch.setattr(
        runtime,
        "_service_names",
        lambda: frozenset(
            {"db", "recall-docs", "recall-code", "recall-user-docs", "recall-user-code"}
        ),
    )

    assert runtime._service_for_tenant("default-docs") == "recall-docs"
    assert runtime._service_for_tenant("user-code") == "recall-user-code"
    assert runtime.list_tenants() == ["default"]


def test_updates_never_downgrade() -> None:
    assert is_newer("0.9.4", "0.9.5")
    assert not is_newer("0.9.5", "0.9.4")


def test_page_watermarks_scope_transparent_group_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QGroupBox, QLabel

    from recall.desktop.ui import MainWindow

    app = QApplication.instance() or QApplication([])

    class UiRuntime:
        def start(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "ready"}

        def list_tenants(self) -> list[str]:
            return ["default"]

        def stop(self) -> None:
            return None

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    window = MainWindow(profile, runtime=UiRuntime())

    assert window.config_page.findChild(QGroupBox, "watermarkGroup") is not None
    assert window.settings_page.findChildren(QGroupBox, "watermarkGroup")
    assert window.github_page.findChildren(QGroupBox, "watermarkGroup") == []
    assert "QGroupBox#watermarkGroup" in window.styleSheet()
    window.resize(980, 760)
    window.show()
    app.processEvents()
    assert window.scan_button.isVisible()
    assert window.scan_button.text() == "Scan"
    assert window.scan_button.parentWidget().objectName() == "queueActions"
    format_hint = window.queue_page.findChild(QLabel, "dropHint")
    assert format_hint is not None
    assert format_hint.text().count("\n") == 1
    assert not format_hint.wordWrap()
    assert window.github_download_button.text() == "DOWNLOAD"
    assert window.github_download_button.objectName() == "downloadButton"
    assert window.github_download_button.size().toTuple() == window.main_button.size().toTuple()
    assert window.github_clear_button.size().toTuple() == window.github_download_button.size().toTuple()
    assert "rgba(215, 165, 42, 0.22)" in window.styleSheet()
    assert "QTableWidget#calibrationTable { selection-background-color: transparent; }" in window.styleSheet()
    assert "QWidget#calibrationActionsCell { background: transparent; }" in window.styleSheet()
    assert window.calibration_table.itemDelegate()._hide_selection is True

    window.close()
    app.processEvents()


def test_saving_the_pipeline_configuration_actually_persists_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The button said "Configuration saved" and saved nothing.

    Demonstrated before fixing, by driving the real window: set the embedder model, press Save,
    close it, reopen, and the field read `hashing-64` again. The status line asserted a state
    nothing had created, which is the defect class the wizard spent a day removing from its own
    report, sitting in the UI the whole time.

    `profile_path` is redirected so this writes a temp directory. An earlier test in this project
    wrote the user's REAL desktop profile and every suite stayed green, because pollution looks
    exactly like success, so this one also asserts the real location was left alone.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import recall.desktop.profiles as profiles
    from recall.desktop.ui import MainWindow

    # Snapshot BOTH real paths and assert they are unchanged, rather than asserting they are
    # absent. Two defects in the first version, found by four auditors independently:
    #   * it checked `runtime.json`, which no path in this test can write. The file this test can
    #     actually pollute is `pipeline.json`, so if the monkeypatch ever stopped taking effect the
    #     user's real settings would be overwritten and the assertion would still have passed.
    #   * `assert not real.exists()` is a claim about the developer's machine, not about this test.
    #     Anyone who has run the wizard once has a real `runtime.json`, and the suite would fail
    #     for them with a pollution message about a file it never touched.
    real_profile = profiles.profile_path()
    real_pipeline = profiles.pipeline_path()
    before = {p: (p.exists(), p.stat().st_mtime_ns if p.exists() else 0)
              for p in (real_profile, real_pipeline)}
    monkeypatch.setattr(profiles, "profile_path", lambda: tmp_path / "runtime.json")

    class UiRuntime:
        def start(self) -> None: ...
        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["default"]

    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    app = QApplication.instance() or QApplication([])

    first = MainWindow(profile, runtime=UiRuntime())
    first.model_edit.setText("a-model-the-user-chose")
    first._save_configuration()
    assert "saved" in first.status.text().lower()
    first.close()
    app.processEvents()

    second = MainWindow(profile, runtime=UiRuntime())
    try:
        assert second.model_edit.text() == "a-model-the-user-chose", (
            "the choice must survive reopening; the status line already claimed it had"
        )
    finally:
        second.close()
        app.processEvents()

    assert (tmp_path / "pipeline.json").exists()
    for path, state in before.items():
        now = (path.exists(), path.stat().st_mtime_ns if path.exists() else 0)
        assert now == state, f"a test must never write the user's real {path.name}"
