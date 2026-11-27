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
import threading
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


def _memo(store) -> str:  # type: ignore[no-untyped-def]  # store is a LiteStore
    return next(chunk.source for chunk in store.iter_chunks() if Path(chunk.source).name == "ops-log.md")


def test_a_forget_without_the_root_in_its_environment_erases_the_recorded_root(
    built, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]  # fixture
    """SEC-002: the build ran with the root set; the forget runs in a process without it."""
    store, root = built
    source = _memo(store)
    assert any(source in _sources(path) for path in _artifacts(root)), "negative control: no artifact quotes the memo"
    monkeypatch.delenv("RECALL_COMPILED_GRAPH_ARTIFACT_ROOT")
    forget_memory(store, [source])
    assert [path for path in _artifacts(root) if source in _sources(path)] == [], (
        "a forget whose environment names no root left the artifact a build recorded"
    )


def test_a_rerun_after_a_failed_erasure_erases_the_compiled_graph(built, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]  # fixture
    """SEC-001: the first forget's erasure fails after the rows went; the re-run it advises must work."""
    store, root = built
    source = _memo(store)
    real = service.erase_forgotten_sources
    failures = [OSError("locked")]

    def flaky(*args: object, **kwargs: object) -> object:
        if failures:
            raise failures.pop()
        return real(*args, **kwargs)  # type: ignore[arg-type]  # forwarding the caller's arguments

    monkeypatch.setattr(service, "erase_forgotten_sources", flaky)
    first = forget_memory(store, [source])
    assert "WARNING" in first.message and "compiled graph" in first.message, first.message
    assert any(source in _sources(path) for path in _artifacts(root)), "negative control: the failed erasure left it"
    forget_memory(store, [source])
    assert [path for path in _artifacts(root) if source in _sources(path)] == [], (
        "the re-run the warning advises left the artifact quoting the forgotten memo"
    )


def test_a_forget_racing_the_publish_leaves_no_artifact_quoting_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-001: a forget lands after the build's last presence check and before it publishes."""
    from recall.lite import LiteStore

    monkeypatch.setattr(service, "urllib_transport", _cite_first_anchor)
    root = tmp_path / "compiled"
    store, embedder = _certified(tmp_path)
    env = _compile_env(root)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    source = _memo(store)
    real_write = service.write_artifact
    real_erase = service.erase_forgotten_sources
    errors: list[BaseException] = []
    erasing = threading.Event()

    def signalled_erase(*args: object, **kwargs: object) -> object:
        erasing.set()
        return real_erase(*args, **kwargs)  # type: ignore[arg-type]  # forwarding the caller's arguments

    monkeypatch.setattr(service, "erase_forgotten_sources", signalled_erase)

    def forget_elsewhere() -> None:
        try:
            with LiteStore(store.path, dim=store.dim, tenant=store.tenant) as other:
                forget_memory(other, [source])
        except BaseException as exc:  # BROAD-CATCH: error-translation  # a thread cannot fail the test; the assertion below reports it
            errors.append(exc)

    worker = threading.Thread(target=forget_elsewhere)

    def write_after_a_forget(*args: object, **kwargs: object) -> object:
        # The forget has deleted the rows and reached its erasure before this publish; without
        # the ordering the erasure finds nothing yet, and this write then publishes the memo.
        worker.start()
        assert erasing.wait(timeout=60), "the forget never reached its compiled graph erasure"
        worker.join(timeout=1)
        return real_write(*args, **kwargs)  # type: ignore[arg-type]  # forwarding the caller's arguments

    monkeypatch.setattr(service, "write_artifact", write_after_a_forget)
    try:
        message = service.ensure_for_store(store, embedder, env)
        worker.join(timeout=60)
        assert not worker.is_alive() and not errors, errors
        assert "built" in message, message
        assert [path for path in _artifacts(root) if source in _sources(path)] == [], (
            "the build published an artifact quoting a memo the forget had already erased"
        )
    finally:
        store.close()


def test_a_forget_while_a_refit_on_the_same_store_publishes_does_not_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FA-1-4: a background refit builds on the same LiteStore object a forget holds the lock of.

    The forget takes the store's lock, then the publish mutex; a build that called the store
    while holding the mutex waited on that lock until the forget's mutex wait (30 s) ran out, and
    the forget's erasure failed, leaving the memo quoted on disk.
    """
    from contextlib import contextmanager

    monkeypatch.setattr(service, "urllib_transport", _cite_first_anchor)
    root = tmp_path / "compiled"
    store, embedder = _certified(tmp_path)
    env = _compile_env(root)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    source = _memo(store)
    holding = threading.Event()
    real_erasing = store.erasing
    results: list[object] = []

    @contextmanager
    def erasing_and_signalling(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]  # wraps the store's context manager
        with real_erasing(*args, **kwargs):
            holding.set()
            yield

    monkeypatch.setattr(store, "erasing", erasing_and_signalling)
    real_lock = service._publish_lock
    forget: threading.Thread | None = None

    def lock_then_let_the_forget_in(folder: Path):  # type: ignore[no-untyped-def]  # returns the real lock's context manager
        nonlocal forget
        lock = real_lock(folder)
        if forget is None:
            forget = threading.Thread(target=lambda: results.append(forget_memory(store, [source])))

            @contextmanager
            def held():  # type: ignore[no-untyped-def]  # a context manager around the real lock
                with lock:
                    assert forget is not None
                    forget.start()
                    assert holding.wait(timeout=30), "the forget never took the store's lock"
                    yield

            return held()
        return lock

    monkeypatch.setattr(service, "_publish_lock", lock_then_let_the_forget_in)
    try:
        service.ensure_for_store(store, embedder, env)
        assert forget is not None
        forget.join(timeout=120)
        assert not forget.is_alive()
        (result,) = results
        assert "WARNING" not in result.message, result.message  # type: ignore[attr-defined]  # a ForgetResult
        assert [path for path in _artifacts(root) if source in _sources(path)] == [], (
            "an artifact quoting the forgotten memo survived the forget"
        )
    finally:
        store.close()


def test_a_forget_whose_root_record_cannot_be_read_still_erases_the_configured_root(
    built, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]  # fixture
    """FA-1-3."""
    store, root = built
    source = _memo(store)

    def unreadable() -> list[str]:
        raise OSError("meta row unreadable")

    monkeypatch.setattr(store, "compiled_graph_roots", unreadable)
    result = forget_memory(store, [source])
    assert [path for path in _artifacts(root) if source in _sources(path)] == [], (
        "a failed read of the recorded roots skipped the configured root"
    )
    assert "could not be read" in result.message, result.message


def test_an_authenticated_forget_names_no_compiled_graph_root_paths(built, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]  # fixture
    """FA-2-4: as for the atomic rescue, host paths stay in the server log."""
    store, _ = built
    monkeypatch.setattr(store, "compiled_graph_roots", lambda: ["//fileserver/share/compiled"])
    shown = forget_memory(store, [_memo(store)], name_rescue_roots=False)
    assert "fileserver" not in shown.message and "1 root(s) the server log names" in shown.message, shown.message


def test_an_authenticated_forget_whose_erasure_fails_names_no_source_paths(built, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]  # fixture
    """L5.5 regression diff: the re-run advice lists resolved host paths only where roots may be named."""
    store, _ = built
    source = _memo(store)

    def fails(*args: object, **kwargs: object) -> object:
        raise OSError("locked")

    monkeypatch.setattr(service, "erase_forgotten_sources", fails)
    shown = forget_memory(store, [source], name_rescue_roots=False)
    assert "compiled graph failed" in shown.message, shown.message
    assert source not in shown.message and "source name(s) the server log names" in shown.message, shown.message


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
