"""PR416/419 regressions, exercised against the pre-fix beb08a605 source on Linux.

The unchanged nodes below must fail their behavior assertions on that baseline and
pass on the repaired tree. Transport and corpus building are replaced; installation,
Compose persistence, name validation and Qt selection use their real consumers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.wizard import headless as H
from recall.wizard.corpora import tenant_for
from recall.wizard.projects import ProjectRefusal, add_project
from tests.test_wizard_state import _CountingSpy, _config


def _managed_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(H, "bring_up", lambda *a, **kw: None)
    monkeypatch.setattr(H, "wait_for_database", lambda *a, **kw: None)
    config = _config(
        tmp_path, dsn=None, migration_dsn=None, data_root=str(tmp_path / "stack")
    )
    profile = tmp_path / "runtime.json"
    H.run_headless(config, services=_CountingSpy(), profile_path=profile)
    return config, profile, config.data_root / H.COMPOSE_NAME


@pytest.mark.parametrize("resume", [False, True])
def test_rerun_preserves_added_projects_at_every_compose_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    """BUG-001: both replacement writes formerly dropped the added project's services.

    Red: beb08a605 headless._provision/run_headless; the saved document lacks
    recall-acme-* at the service-preservation assertion, with and without state.
    """
    config, profile, compose = _managed_install(tmp_path, monkeypatch)
    add_project(config.data_root, "acme")
    document = json.loads(compose.read_text())
    document["services"]["recall-acme-docs"]["environment"]["RECALL_ENV"] = "production"
    document["services"]["recall-acme-docs"]["labels"] = {"operator.setting": "keep"}
    H.write_compose(compose, document)
    expected = {k: v for k, v in document["services"].items() if k.startswith("recall-acme-")}
    writes = []
    writer = H.write_compose

    def record(path, value, **kwargs):
        writes.append(json.loads(json.dumps(value)))
        return writer(path, value, **kwargs)

    monkeypatch.setattr(H, "write_compose", record)
    state = tmp_path / "state.json" if resume else None
    for _ in range(2 if resume else 1):
        report = H.run_headless(
            config, services=_CountingSpy(), profile_path=profile, state_path=state
        )
        assert report.ok
    assert len(writes) >= 2
    for written in writes:
        actual = {k: v for k, v in written["services"].items() if k.startswith("recall-acme-")}
        assert actual == expected, "every write must preserve added projects and their settings"


@pytest.mark.parametrize("failure", ["start", "readiness", "prepare"])
def test_failed_rerun_preserves_existing_generation_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """DAT-001: provisional writes must never persist over an existing serving route.

    Red: beb08a605 _provision replaces production environments before any of the
    three failing stages; the exact service comparison fails after the refusal.
    """
    config, profile, compose = _managed_install(tmp_path, monkeypatch)
    before = json.loads(compose.read_text())["services"]
    assert before["recall-default-docs"]["environment"]["RECALL_ENV"] == "production"

    def refuse(*args, **kwargs):
        raise RuntimeError("deliberate database readiness failure")

    if failure == "prepare":
        def refuse_prepare(*args, **kwargs):
            raise H.ConfigRefusal("deliberate preparation refusal")
        monkeypatch.setattr(H, "_prepare", refuse_prepare)
    else:
        monkeypatch.setattr(H, "bring_up" if failure == "start" else "wait_for_database", refuse)
    with pytest.raises(H.ConfigRefusal):
        H.run_headless(config, services=_CountingSpy(), profile_path=profile)
    assert json.loads(compose.read_text())["services"] == before, (
        "a failed rerun must retain the existing generation-serving configuration"
    )


@pytest.mark.parametrize("failure", [None, "start", "readiness", "prepare", "add"])
def test_rerun_keeps_build_recipes_consistent_with_preserved_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """BUG-001/DAT-001 repair control: preserve older images AND their recipes.

    Red: the first repair's _write_stack regenerated the shared Dockerfile at the
    running version. The original-recipe assertion fails on every rerun path.
    """
    from recall import __version__
    from recall.wizard.stack import DOCKERFILE_NAME

    config, profile, compose = _managed_install(tmp_path, monkeypatch)
    add_project(config.data_root, "acme")
    document = json.loads(compose.read_text())
    for name, service in document["services"].items():
        if name != "db":
            service["image"] = "recall-wizard:0.9.6"
            service["build"] = {"context": ".", "dockerfile": DOCKERFILE_NAME}
    H.write_compose(compose, document, dockerfile_version="0.9.6")
    recipe = compose.parent / DOCKERFILE_NAME
    before = recipe.read_bytes()
    assert b"==0.9.6" in before, "the installed recipe must match the installed image"
    assert __version__ != "0.9.6", "the fixture must exercise two different image versions"

    def refuse(*args, **kwargs):
        raise H.ConfigRefusal("deliberate rerun refusal")

    if failure and failure != "add":
        target = {"start": "bring_up", "readiness": "wait_for_database", "prepare": "_prepare"}[failure]
        monkeypatch.setattr(H, target, refuse)
        with pytest.raises(H.ConfigRefusal):
            H.run_headless(config, services=_CountingSpy(), profile_path=profile)
    else:
        assert H.run_headless(config, services=_CountingSpy(), profile_path=profile).ok
    assert recipe.read_bytes() == before, "preserved images must keep their original build recipe"
    if failure == "add":
        add_project(config.data_root, "later")
    services = json.loads(compose.read_text())["services"]
    for name, service in services.items():
        if name == "db":
            continue
        tag = service["image"].split(":", 1)[1]
        built_from = compose.parent / service["build"]["dockerfile"]
        assert f"=={tag}" in built_from.read_text(), f"{name} would build a different image version"


@pytest.mark.parametrize("failure", ["raise", "refuse"])
def test_failed_corpus_rebuild_keeps_only_preexisting_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """DAT-001: caught corpus failures must retain the installed generation route.

    Red: before the final-write correction, _write_stack deletes default-docs
    after either refusal or build exception. The missing-service assertion fails.
    The fresh-install control keeps unsuccessful new scopes without a server.
    """
    config, profile, compose = _managed_install(tmp_path, monkeypatch)
    original = json.loads(compose.read_text())["services"]["recall-default-docs"]

    class Faulty(_CountingSpy):
        def run(self, spec, *, progress=None):
            if spec.tenant == "default-docs":
                error = H.PipelineRefusal if failure == "refuse" else RuntimeError
                raise error("deliberate corpus rebuild failure")
            return super().run(spec, progress=progress)

    report = H.run_headless(config, services=Faulty(), profile_path=profile)
    assert not report.ok
    services = json.loads(compose.read_text())["services"]
    assert services.get("recall-default-docs") == original, (
        "a failed corpus rebuild must retain its preexisting service and generation route"
    )
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    fresh = _config(
        fresh_root, dsn=None, migration_dsn=None, data_root=str(fresh_root / "stack")
    )
    report = H.run_headless(fresh, services=Faulty(), profile_path=fresh_root / "runtime.json")
    assert not report.ok
    new_services = json.loads((fresh.data_root / H.COMPOSE_NAME).read_text())["services"]
    assert "recall-default-docs" not in new_services, "an unsuccessful new scope must have no server"
    assert "recall-default-code" in new_services, "independent successful corpora still get services"


@pytest.mark.parametrize("project", ["caf\u00e9", "\u6771\u4eac", "\u0661project"])
def test_non_ascii_project_is_refused_before_stack_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project: str
) -> None:
    """ENV-001: baseline tenant_for accepts names real Compose rejects.

    Red: beb08a605 tenant_for/add_project do not raise at pytest.raises; a bad
    project must be refused before rewriting an otherwise usable stack.
    """
    config, _profile, compose = _managed_install(tmp_path, monkeypatch)
    before = compose.read_bytes()
    with pytest.raises(ProjectRefusal, match="ASCII"):
        add_project(config.data_root, project)
    assert compose.read_bytes() == before
    with pytest.raises(ValueError, match="ASCII"):
        tenant_for(project, "docs")


def test_compose_project_name_contains_only_compose_characters(tmp_path: Path) -> None:
    """ENV-001: baseline compose_project_for retains non-ASCII alphanumerics."""
    name = H.compose_project_for(tmp_path, "caf\u00e9.\u6771\u4eac")
    assert name.isascii(), "Compose project names must contain ASCII characters only"
    assert all(c.isalnum() or c in "-_" for c in name)


@pytest.mark.parametrize("embedder", ["voyage", "voyage:voyage-4", "openai", "st:some-model"])
def test_managed_install_refuses_unsupported_embedder_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: str
) -> None:
    """DEPLOY-001: baseline installs unsupported backends into its fixed image.

    Red: beb08a605 run_headless completes with the corpus spy instead of refusing.
    No API or model is invoked. The guard must precede all stack and corpus work.
    """
    calls = []
    monkeypatch.setattr(H, "bring_up", lambda *a, **kw: calls.append("start"))
    monkeypatch.setattr(H, "wait_for_database", lambda *a, **kw: None)
    config = _config(
        tmp_path, dsn=None, migration_dsn=None, data_root=str(tmp_path / "stack"),
        embedder=embedder,
    )
    spy = _CountingSpy()
    with pytest.raises(H.ConfigRefusal, match="embedder") as caught:
        H.run_headless(config, services=spy, profile_path=tmp_path / "runtime.json")
    assert caught.value.keys == ("embedder", "data_root")
    assert calls == [] and spy.built == [] and spy.legacy == []
    assert not config.data_root.exists()


def test_reserved_project_refusal_keeps_selection_and_next_drop_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-002: baseline repopulation overwrites _last_scope_index with default.

    Red: beb08a605 MainWindow._scope_changed; the current-scope assertion fails
    after refusing user from acme. The next actual drop must also keep acme.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QInputDialog
    from recall.desktop.models import RuntimeMode, RuntimeProfile
    from recall.desktop.ui import MainWindow

    class Runtime:
        def stop(self):
            pass

    monkeypatch.setattr(MainWindow, "_run", lambda *a, **kw: None)
    monkeypatch.setattr("recall.desktop.ui.load_pipelines", lambda: {})
    monkeypatch.setattr("recall.desktop.profiles.profile_path", lambda: tmp_path / "profile.json")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("user", True))
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        RuntimeProfile(mode=RuntimeMode.VPS_MCP, endpoint="https://example.test/mcp"),
        runtime=Runtime(),
    )
    try:
        window._populate_scopes(["default", "acme"])
        window.scope.setCurrentIndex(window.scope.findText("acme"))
        prior = dict(window._scope_data())
        window.scope.setCurrentIndex(window.scope.count() - 1)
        assert window._scope_data() == prior, "reserved-name refusal must retain acme"
        assert "Selection unchanged" in window.status.text()
        source = tmp_path / "note.md"
        source.write_text("a note", encoding="utf-8")
        # PERF-001 collects on a worker; run that callback in this controlled fixture.
        monkeypatch.setattr(window, "_run", lambda fn, done, *_args: done(fn()))
        window._accept_drop([str(source)])
        assert window.pending_scopes[-1] == prior, "the next import must keep its original target"
    finally:
        window.close()
        app.processEvents()
