"""`recall setup --lite`: index, calibrate and register a project's memory in one SQLite file.

The Claude Code config is a temporary file, so nothing outside `tmp_path` is touched.

Invariants and the failure each one catches:
- E1 setup indexes the memory folder into `.recall/memory.db`, calibrates it, and registers ONE
  local-scope server whose DSN is that file and whose index root is that folder.
- E2 the registered server env carries what the server needs from the OS when a client starts
  it with only that env (APPDATA, SystemRoot and the cache and home variables, where set) and
  PYTHONSAFEPATH=1, and never PATH.
- E3 a memory folder that does not exist is refused by name, before any store is created.
- E4 setup says to add `.recall/` to .gitignore unless it is already there.

Red proof, 2026-10-08, each mutation alone, failing in its intended assertion (JUnit XML), then
restored byte for byte and green:
- R1 (E1) `ensure_calibrated` removed from `run_lite_setup`: "setup did not calibrate the store".
- R2 (E2) `server_os_env()` not merged into the server env: "the server env lacks APPDATA".
- R3 (E3) the `is_dir` check removed: "a missing memory folder was not refused by name".
- R4 (E4) the .gitignore check reading nothing: "the note repeats advice already followed".
The handshake behind E2 (the registered server, started with exactly its registered env, listing
21 tools) was run by hand on Windows the same day; CI cannot repeat it meaningfully, because an
editable install there does not live in a user site directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite.setup import SERVER_NAME, run_lite_setup
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "claude.json"
    path.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    return path


def _servers(config: Path) -> dict[str, dict[str, object]]:
    projects = json.loads(config.read_text(encoding="utf-8"))["projects"]
    return {name: server for entry in projects.values() for name, server in entry.get("mcpServers", {}).items()}


def _setup(tmp_path: Path, **kwargs: object):  # noqa: ANN202
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    _write_memos(project / "notes", 48)
    config = _config(tmp_path)
    report = run_lite_setup(
        project_root=project, memory=project / "notes", embedder=ContentWordEmbedder(), embedder_name="content-words",
        config_path=config, **kwargs,  # type: ignore[arg-type]
    )
    return project, config, report


def test_setup_indexes_calibrates_and_registers(tmp_path: Path) -> None:
    """E1."""
    project, config, report = _setup(tmp_path)
    store_path = project / ".recall" / "memory.db"
    assert report.store == store_path.resolve() and report.chunks == 48
    assert report.calibration == "certified", "setup did not calibrate the store"
    with LiteStore(store_path, dim=DIM) as store:
        assert store.count() == 48 and store.resolve_calibration().status.value == "certified"
    servers = _servers(config)
    assert list(servers) == [SERVER_NAME]
    env = servers[SERVER_NAME]["env"]
    assert isinstance(env, dict)
    assert env["RECALL_DSN"] == "sqlite:///" + store_path.resolve().as_posix()
    assert env["RECALL_INDEX_ROOT"] == str((project / "notes").resolve())


def test_the_server_env_carries_what_a_replaced_environment_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E2."""
    monkeypatch.setenv("APPDATA", "C:/fake/appdata")
    monkeypatch.setenv("SystemRoot", "C:/fake/windows")
    monkeypatch.setenv("TEMP", "C:/fake/temp")
    _project, config, _report = _setup(tmp_path)
    env = _servers(config)[SERVER_NAME]["env"]
    assert isinstance(env, dict)
    assert env.get("APPDATA") == "C:/fake/appdata", "the server env lacks APPDATA"
    assert env.get("SystemRoot") == "C:/fake/windows" and env.get("TEMP") == "C:/fake/temp"
    assert env.get("PYTHONSAFEPATH") == "1"
    assert "PATH" not in env


def test_a_missing_memory_folder_is_refused(tmp_path: Path) -> None:
    """E3."""
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(Exception) as refused:  # noqa: PT011  # which error, and its text, is the assertion
        run_lite_setup(project_root=project, memory=project / "nowhere", embedder=ContentWordEmbedder(),
                       embedder_name="content-words", register=False)
    assert isinstance(refused.value, NotADirectoryError) and "does not exist" in str(refused.value), (
        f"a missing memory folder was not refused by name: {refused.value!r}"
    )
    assert not (project / ".recall").exists()


def test_the_gitignore_note_appears_only_when_needed(tmp_path: Path) -> None:
    """E4."""
    _project, _config_path, report = _setup(tmp_path, register=False)
    assert any(".gitignore" in note for note in report.notes)
    other = tmp_path / "second"
    other.mkdir()
    (other / "project").mkdir()
    (other / "project" / ".gitignore").write_text("node_modules/\n.recall/\n", encoding="utf-8")
    _p, _c, again = _setup(other, register=False)
    assert not any(".gitignore" in note for note in again.notes), "the note repeats advice already followed"
