from __future__ import annotations

import base64
from pathlib import Path

import pytest

from recall.desktop.models import RuntimeMode, RuntimeProfile, SourceCategory, SourceSelection
from recall.desktop.runtime import DockerRuntime, VpsMcpRuntime, create_runtime
from recall.desktop.sources import (
    classify,
    collect_files,
    default_scan_roots,
    display_type,
    scan_files,
)
from recall.desktop.updates import is_newer


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


def test_docker_runtime_exposes_managed_project_scopes() -> None:
    profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    runtime = DockerRuntime(profile)

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
