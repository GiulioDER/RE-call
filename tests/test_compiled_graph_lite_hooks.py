"""The compiled graph on a real Lite store: built by the write path, erased by forget.

Invariant: with ``RECALL_COMPILED_GRAPH_MODE=active`` and an artifact root, the path every Lite
write takes (``recall.lite.atomic.ensure_rescue_artifact``) builds an artifact for the corpus in
force, with or without the atomic rescue on; ``forget_memory`` removes every artifact quoting the
forgotten source and its cache rows, and says so. The provider is a fake transport.

Red proof, recorded 2026-11-26: with the ``ensure_for_store`` call removed from
``ensure_rescue_artifact``, ``test_a_lite_write_builds_the_compiled_graph`` fails at "no artifact
was built"; with the ``erase_forgotten_sources`` block removed from
``recall_mcp.lifecycle._forget_memory_unlocked``, ``test_forget_removes_the_compiled_graph`` fails
at "an artifact quoting the forgotten memo is left". CLI (``recall.cli_commands.graph_cmd``): the
off check made ``if False`` fails ``test_cli_compile_refuses_when_off``; the Lite dry-run branch
removed fails ``test_cli_dry_run_sends_nothing``; the ``compile`` dispatch removed fails
``test_cli_compile_reports_the_artifact_in_force``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from recall.compiled_graph import service
from recall.lite.atomic import ensure_rescue_artifact
from recall_mcp.lifecycle import forget_memory
from tests.test_lite_atomic_rescue import _certified


def _compile_env(root: Path) -> dict[str, str]:
    return {
        "RECALL_COMPILED_GRAPH_MODE": "active",
        "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": str(root),
        "RECALL_COMPILED_GRAPH_API_KEY": "sk-test",
        "RECALL_COMPILED_GRAPH_MODEL": "openai/gpt-6-luna-pro",
        "RECALL_COMPILED_GRAPH_MAX_USD": "5",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT": "0.001",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT": "0.002",
    }


def _cite_first_anchor(url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
    stored = json.loads(json.loads(body)["messages"][1]["content"][len("<stored_data>") : -len("</stored_data>")])
    anchor = stored["anchors"][0]
    record = {"kind": "fact", "summary": anchor["excerpt"][:60], "detail": "", "entities": [],
              "evidence_anchor_ids": [anchor["id"]]}
    reply = {"choices": [{"message": {"content": json.dumps({"records": [record]})}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    return 200, {}, json.dumps(reply).encode()


def _artifacts(root: Path) -> list[Path]:
    return [path.parent for path in root.glob("tenant-*/cg-*/records.jsonl")]


def _sources(artifact: Path) -> set[str]:
    lines = (artifact / "records.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["source"] for line in lines if line}


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]  # yields (store, root)
    monkeypatch.setattr(service, "urllib_transport", _cite_first_anchor)
    root = tmp_path / "compiled"
    store, embedder = _certified(tmp_path)
    env = _compile_env(root)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ensure_rescue_artifact(store, embedder, env=env)
    yield store, root
    store.close()


def test_a_lite_write_builds_the_compiled_graph(built) -> None:  # type: ignore[no-untyped-def]  # fixture
    _, root = built
    artifacts = _artifacts(root)
    assert artifacts, "no artifact was built"
    rows = [json.loads(line) for line in (artifacts[0] / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["edges"] for row in rows)


def test_forget_removes_the_compiled_graph(built) -> None:  # type: ignore[no-untyped-def]  # fixture
    store, root = built
    source = next(chunk.source for chunk in store.iter_chunks() if Path(chunk.source).name == "ops-log.md")
    assert any(source in _sources(path) for path in _artifacts(root)), (
        "negative control: no artifact quotes the memo"
    )
    result = forget_memory(store, [source])
    assert result.chunks_removed > 0, result.message
    left = [path for path in _artifacts(root) if source in _sources(path)]
    assert left == [], "an artifact quoting the forgotten memo is left"
    assert "compiled graph" in result.message


def _cli(store, monkeypatch: pytest.MonkeyPatch, *flags: str) -> str:  # type: ignore[no-untyped-def]  # store is a LiteStore
    from tests.lite_cli_support import run_cli
    from tests.test_lite_atomic_rescue import ContentWordEmbedder

    monkeypatch.setattr("recall.cli_commands._shared._make_embedder", lambda name: ContentWordEmbedder())
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    return run_cli(["--dsn", "sqlite:///" + store.path.as_posix(), "--tenant", store.tenant, "graph", "compile", *flags])


def test_cli_compile_refuses_when_off(built, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]  # fixture
    store, _ = built
    monkeypatch.setenv("RECALL_COMPILED_GRAPH_MODE", "off")
    out = _cli(store, monkeypatch)
    assert "the compiled graph is off" in out and "EXIT" in out


def test_cli_dry_run_sends_nothing(built, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]  # fixture
    store, _ = built

    def refuse(*args: object) -> object:
        raise AssertionError("a dry run sent a request")

    monkeypatch.setattr(service, "urllib_transport", refuse)
    out = _cli(store, monkeypatch, "--dry-run")
    assert "0 source(s) to compile" in out and "nothing was sent" in out, out


def test_cli_compile_reports_the_artifact_in_force(built, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]  # fixture
    store, _ = built
    out = _cli(store, monkeypatch)
    assert "compiled graph in force" in out, out
