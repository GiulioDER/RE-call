from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

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
    assert SourceSelection(SourceCategory.MEMORY, (memory,), "user", True).physical_tenant == "user-docs"
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
    assert arguments["tenant"] == "default-docs"
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


def test_docker_runtime_reads_its_topology_from_the_compose_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The UI could not drive the stack the wizard installs, and this is why.

    `_service_for_tenant` was a literal four-entry dict mapping `default-docs` to `recall-docs`.
    The wizard names every service `recall-<tenant>`, so it writes `recall-default-docs`. Measured
    against a generated compose document, EVERY scope missed, the default one included, so the
    mismatch was never confined to projects the user had added.

    Both directions are asserted from one fake service list, because either alone is satisfiable by
    the defect: the wizard's naming must resolve, the legacy names must keep resolving, an unknown
    scope must be refused with what is on offer, and `list_tenants` must report what the file
    contains rather than a constant.
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
    assert "recall-acme-docs" in str(refusal.value), "a refusal must say what IS on offer"
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
        lambda: frozenset({"db", "recall-default-docs", "otel-collector", "recall-backup"}),
    )
    monkeypatch.setattr(runtime, "health", lambda: {"status": "ready"})
    monkeypatch.setattr(runtime, "_call_for", lambda *a, **k: {})

    runtime.start()

    applied = [args[2] for args in calls if "schema" in args]
    assert applied == ["recall-default-docs"], f"schema apply reached {applied}"


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

    real = profiles.profile_path()
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
    assert not real.exists(), "a test must never write the user's real desktop configuration"
