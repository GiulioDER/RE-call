"""PR402/405 regression probes, baseline 83014e2c3 before repairs.

Each test asserts the consumer invariant, with the identical node rerun after the fix.
The red receipt records the intended assertion and production boundary.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from recall import extraction
from recall.runtime_route import resolve_runtime_route
from recall.wizard.corpora import default_plan
from recall.wizard.state import WizardState, load_state
from recall.wizard.wiring import mcp_config, server_blocks, write_project_files
from recall_mcp.settings import bootstrap_settings


def test_profile_allocation_failure_returns_pool_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-001: allocation failure in _libreoffice_profile must not exhaust capacity."""
    extraction._discard_all_profiles()
    monkeypatch.setattr(extraction, "_profile_pool", None)
    monkeypatch.setenv("RECALL_LIBREOFFICE_PROFILES", "1")
    monkeypatch.setenv("RECALL_LIBREOFFICE_SHARED_PROFILE", "1")
    monkeypatch.setattr(extraction, "_libreoffice_executable", lambda: "soffice")
    original = extraction.tempfile.mkdtemp

    def allocate(*args, **kwargs):
        if kwargs.get("prefix") == "recall-office-profile-":
            raise OSError("profile allocation unavailable")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(extraction.tempfile, "mkdtemp", allocate)
        with pytest.raises(OSError, match="profile allocation unavailable"):
            extraction.extract_document(tmp_path / "input.doc", b"legacy")
    assert extraction._profiles().qsize() == 1, "allocation failure consumed the only pool token"

    def convert(command, **kwargs):
        output = Path(command[command.index("--outdir") + 1]) / "input.txt"
        output.write_text("recovered conversion", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(extraction.subprocess, "run", convert)
    try:
        assert extraction.extract_document(tmp_path / "input.doc", b"legacy").text == "recovered conversion"
    finally:
        extraction._discard_all_profiles()


@pytest.mark.parametrize("field", ["outcomes", "indexed"])
@pytest.mark.parametrize("value", [1, True])
def test_malformed_checkpoint_collections_are_ignored(tmp_path: Path, field: str, value: object) -> None:
    """BUG-002: load_state must return empty state for invalid collection shapes."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 1, "digest": "d", field: value}), encoding="utf-8")
    try:
        state = load_state(path, digest="d")
    except (TypeError, AttributeError) as exc:
        state = exc
    assert state == WizardState(digest="d"), "invalid checkpoint was trusted"


@pytest.mark.parametrize("field", ["certified", "promoted", "unverified_embedder"])
@pytest.mark.parametrize("value", ["false", 1])
def test_checkpoint_flags_require_booleans(tmp_path: Path, field: str, value: object) -> None:
    """DAT-002: persisted flags cannot acquire truth through string coercion."""
    outcome = {"tenant": "default-docs", "generation_id": "gen_never_built", "certified": True, "promoted": True}
    outcome[field] = value
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 1, "digest": "d", "outcomes": [outcome]}), encoding="utf-8")
    state = load_state(path, digest="d")
    assert state.outcome_for("default-docs") is None, "invalid flag reused an unbuilt corpus"
    assert state == WizardState(digest="d"), "invalid outcome fields were admitted"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("existing", [False, True])
def test_wizard_env_credentials_are_private(tmp_path: Path, existing: bool) -> None:
    """SEC-001: write_project_files must create and replace .env privately under umask 022."""
    root = tmp_path / "project"
    root.mkdir(mode=0o755)
    target = root / ".env"
    if existing:
        target.write_text("UNMANAGED=preserved\n", encoding="utf-8")
        target.chmod(0o644)
    previous = os.umask(0o022)
    try:
        write_project_files(project_root=root, dsn="postgresql://user:password@localhost/store", embedder="hashing", memory_dir=tmp_path / "memory")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600, "database credentials are readable by other users"
    assert "RECALL_DSN=" in target.read_text(encoding="utf-8")
    if existing:
        assert "UNMANAGED=preserved" in target.read_text(encoding="utf-8")


def _shared_blocks(tmp_path: Path):
    plan = default_plan(embedder="hashing", docs_root=tmp_path / "docs", code_root=tmp_path / "repo", memory_root=tmp_path / "memory")
    blocks, _ = server_blocks(plan, dsn="postgresql://selected/installed", promoted=frozenset({"default-docs", "default-code"}), serving=frozenset({"default-docs", "default-code", "default-memory"}))
    return blocks


def _server_environments(tmp_path: Path):
    return mcp_config(_shared_blocks(tmp_path), project_root=tmp_path)["mcpServers"]


@pytest.mark.parametrize("inherited", [{"RECALL_ENV": "production"}, {"RECALL_ENV": "production", "RECALL_INDEX_MODE": "generation"}])
def test_memory_server_route_overrides_inherited_configuration(tmp_path: Path, inherited: dict[str, str]) -> None:
    """ENV-001: emitted memory configuration must read the legacy table actually indexed."""
    servers = _server_environments(tmp_path)
    route = resolve_runtime_route({**inherited, **servers["default-memory"]["env"]})
    assert route.mode == "legacy", "inherited client environment redirected memory to generations"
    assert route.environment == "development"


@pytest.mark.parametrize("tenant", ["default-docs", "default-code", "default-memory"])
def test_server_database_overrides_inherited_serving_dsn(tmp_path: Path, tenant: str) -> None:
    """ENV-002: real settings precedence must preserve the database selected by the wizard."""
    env = {"RECALL_SERVING_DSN": "postgresql://foreign/other", **_server_environments(tmp_path)[tenant]["env"]}
    settings = bootstrap_settings(env)
    assert settings.serving_dsn == "postgresql://selected/installed", "client's serving DSN replaced the installed database"


def test_generated_container_uses_its_network_database(tmp_path: Path) -> None:
    """ENV-002 review correction: the shared block must not retain a host address in Docker."""
    from recall.wizard.stack import container_dsn, tenant_service

    for block in _shared_blocks(tmp_path):
        service = tenant_service(block.env, image="fixture")
        settings = bootstrap_settings(service["environment"])
        assert settings.serving_dsn == container_dsn(), "host serving DSN bypassed Docker's database translation"


def test_install_report_does_not_print_the_serving_password(tmp_path: Path) -> None:
    """ENV-002 review correction: adding the serving override must not add credentials to logs."""
    from recall.wizard.headless import HeadlessReport

    plan = default_plan(embedder="hashing", docs_root=tmp_path / "docs", code_root=tmp_path / "repo", memory_root=tmp_path / "memory")
    blocks, _ = server_blocks(plan, dsn="postgresql://user:fixture-password@localhost/installed", promoted=frozenset({"default-docs", "default-code"}), serving=frozenset({"default-docs", "default-code", "default-memory"}))
    assert "fixture-password" not in HeadlessReport(servers=blocks).render(), "installation report leaked the serving password"
