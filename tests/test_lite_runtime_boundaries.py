"""SQLite runtime boundaries through the CLI, MCP startup and doctor."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from recall.backends import open_legacy_store
from recall.doctor import run_checks
from recall.lite import LiteStore
from recall.types import Chunk
from recall_mcp import server as srv
from recall_mcp.settings import bootstrap_settings
from tests.test_lite_calibration import _write_memos
from tests.test_lite_cli import _run
from tests.test_lite_mcp import _serve, lite_env as _lite_env

lite_env = _lite_env  # register the shared offline fixture under its existing name


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN",
                "RECALL_TRUST_MODE", "RECALL_TENANT", "RECALL_TABLE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _dsn(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


def test_cli_custom_table_preserves_existing_corpus(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    memo = folder / "a.md"
    memo.write_text("# Original\n\nThe deployment day is Tuesday.\n", encoding="utf-8")
    db = tmp_path / "m.db"
    args = ["--embedder", "hashing", "--dsn", _dsn(db)]
    assert "indexed 1 chunks" in _run([*args, "index", str(folder)])
    with LiteStore(db, dim=64) as store:
        before = [(c.id, c.text) for c in store.iter_chunks()]
    memo.write_text("# Replacement\n\nThe deployment day is Friday.\n", encoding="utf-8")
    output = _run([*args, "--table", "scratch", "index", str(folder)])
    with LiteStore(db, dim=64) as store:
        after = [(c.id, c.text) for c in store.iter_chunks()]
    assert after == before, "custom table indexing replaced the real corpus"
    assert "one table" in output and "scratch" in output


def test_cli_custom_table_does_not_create_file(tmp_path: Path) -> None:
    db = tmp_path / "missing" / "m.db"
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "a.md").write_text("# Memo\n\nOne private memo.\n", encoding="utf-8")
    output = _run(["--embedder", "hashing", "--dsn", _dsn(db), "--table", "scratch", "index", str(folder)])
    assert not db.exists(), "custom table request created a store file"
    assert "one table" in output


def test_backend_table_guard_precedes_open(tmp_path: Path) -> None:
    db = tmp_path / "missing" / "m.db"
    error = ""
    try:
        with open_legacy_store(_dsn(db), dim=64, table="scratch"):
            pass
    except ValueError as exc:
        error = str(exc)
    assert "one table" in error and not db.exists(), "backend skipped the table guard"


def test_mcp_custom_table_is_refused_before_startup(
    lite_env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECALL_TABLE", "scratch")
    env = {**lite_env, "RECALL_TABLE": "scratch"}
    started = False

    async def body(state: dict[str, Any], call: Any) -> None:
        nonlocal started
        started = True

    error = ""
    try:
        _serve(env, body)
    except Exception as exc:  # noqa: BLE001  # assert the public startup refusal
        error = str(exc)
    assert not started and "one table" in error, "MCP ignored its custom table"
    assert not (tmp_path / "store" / "memory.db").exists()


def test_doctor_refuses_an_unsupported_table_without_opening(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    report = run_checks(dsn=_dsn(db), embedder="hashing", table="scratch")
    blocked = [c for c in report.checks if c.status == "fail" and "one table" in c.detail]
    assert blocked, "doctor accepted an unsupported table"
    assert not db.exists()


@pytest.mark.parametrize("dimension", ["bad", "0", "-1", None])
def test_doctor_reports_invalid_dimension(tmp_path: Path, dimension: str | None) -> None:
    db = tmp_path / "m.db"
    with LiteStore(db, dim=64):
        pass
    with sqlite3.connect(db) as conn:
        if dimension is None:
            conn.execute("DELETE FROM meta WHERE key='dim'")
        else:
            conn.execute("UPDATE meta SET value=? WHERE key='dim'", (dimension,))
    try:
        report = run_checks(dsn=_dsn(db), embedder="hashing")
    except ValueError as exc:
        pytest.fail(f"invalid dimension escaped the diagnostic: {exc}")
    assert any(c.name == "database" and c.status == "fail" for c in report.checks), (
        "invalid dimension was accepted by the diagnostic"
    )


def test_doctor_reports_server_tenant_divergence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "m.db"
    with LiteStore(db, dim=64):
        pass
    monkeypatch.setenv("RECALL_TENANT", "work")
    report = run_checks(dsn=_dsn(db), embedder="hashing")
    check = next((c for c in report.checks if c.name == "server config"), None)
    assert check is not None, "server tenant divergence was hidden"
    assert "'chunks'/'default'" in check.detail and "'chunks'/'work'" in check.detail
    assert check.fix is not None and "--tenant work" in check.fix


def test_calibration_description_explains_immediate_lite_activation(lite_env: dict[str, str]) -> None:
    mcp = srv.build_server(bootstrap_settings(lite_env))
    tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "recall_calibration_run")
    description = tool.description.lower()
    assert "postgres" in description and "draft" in description and "lite" in description and "immediately" in description, (
        "calibration description hides immediate Lite activation"
    )
    assert "uncertified" in description and "prior" in description


@pytest.mark.parametrize("identifiers", [["a.md"], ["missing.md"], []])
def test_source_resolution_does_not_load_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifiers: list[str],
) -> None:
    with LiteStore(tmp_path / "m.db", dim=2) as store:
        store.upsert([Chunk(id="a", text="Memo", source="/root/a.md", metadata={"file": "a.md"})], [[1.0, 0.0]])
        loads = 0
        original = store._load  # noqa: SLF001

        def counted() -> Any:
            nonlocal loads
            loads += 1
            return original()

        monkeypatch.setattr(store, "_load", counted)
        result = store.sources_for_identifiers(identifiers)
        assert loads == 0, "source resolution loaded the embedding matrix"
        assert result == ({"a.md": ["/root/a.md"]} if identifiers == ["a.md"] else {})


def test_mcp_empty_labels_are_rejected(
    lite_env: dict[str, str], tmp_path: Path,
) -> None:
    _write_memos(tmp_path / "m", 48)

    async def body(state: dict[str, Any], call: Any) -> None:
        await call("recall_index", path=str(tmp_path / "m"))
        before = state["store"].latest_calibration_json()
        assert before is not None
        result = await call("recall_calibration_run", queries=[])
        assert result == "RAISED CalibrationBindingError: query set must not be empty", (
            "empty labels were replaced by generated questions"
        )
        assert state["store"].latest_calibration_json() == before

    _serve(lite_env, body)


def test_matching_server_tenant_produces_no_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_TENANT", "default")
    monkeypatch.setenv("RECALL_TABLE", "chunks")
    report = run_checks(dsn=_dsn(tmp_path / "missing.db"), embedder="hashing")
    assert not any(c.name == "server config" for c in report.checks)


def test_source_aliases_and_large_identifier_batches(tmp_path: Path) -> None:
    with LiteStore(tmp_path / "m.db", dim=2) as store:
        store.upsert([
            Chunk(id="a", text="Memo A", source="/first/a.md", metadata={"file": "a.md"}),
            Chunk(id="b", text="Memo B", source="/second/a.md", metadata={"file": "a.md"}),
            Chunk(id="c", text="Memo C", source="/third/c.md", metadata={"file": 7}),
        ], [[1.0, 0.0]] * 3)
        identifiers = [f"missing-{i}" for i in range(1000)] + ["a.md", "/first/a.md", "/third/c.md", "7", "a.md"]
        resolved = store.sources_for_identifiers(identifiers)
        assert set(resolved["a.md"]) == {"/first/a.md", "/second/a.md"}
        assert resolved["/first/a.md"] == ["/first/a.md"]
        assert resolved["/third/c.md"] == ["/third/c.md"]
        assert set(resolved) == {"a.md", "/first/a.md", "/third/c.md"}


@pytest.mark.parametrize("file", [{"path": "a.md"}, ["a.md"]], ids=["object", "array"])
@pytest.mark.parametrize("direct", [False, True], ids=["alias-only", "with-source"])
def test_structured_file_metadata_is_not_an_erase_alias(tmp_path: Path, file: Any, direct: bool) -> None:
    source = "/root/a.md"
    identifier = json.dumps(file, separators=(",", ":"))
    with LiteStore(tmp_path / "m.db", dim=2) as store:
        store.upsert([Chunk(id="a", text="Memo", source=source, metadata={"file": file})], [[1.0, 0.0]])
        requested = [identifier, source] if direct else [identifier]
        resolved = store.sources_for_identifiers(requested)
        assert resolved == ({source: [source]} if direct else {}), "structured file metadata became an erase alias"

