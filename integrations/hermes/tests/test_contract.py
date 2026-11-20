"""The provider against a REAL Hermes checkout: its ABC, its directory loader and its MemoryManager.

Skipped unless HERMES_SRC names a Hermes checkout whose dependencies are importable (conftest puts
HERMES_SRC first on sys.path, in place of the fakes). The server behind the provider is still the
scripted one, so these tests check Hermes's side of the boundary only.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil

import pytest
from conftest import HERMES_SRC, ROOT
from test_provider import SEARCH_OK

pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(not HERMES_SRC, reason="set HERMES_SRC to a Hermes checkout to run the contract tests"),
]


def test_the_provider_is_a_concrete_subclass_of_the_real_abc():
    """Invariant: every abstract method of Hermes's MemoryProvider is implemented.

    Red proof (2026-11-08, CCA run 20261108T171507Z, Linux host, Hermes v0.23.0 2bd378ed, `python
    tests/mutations.py CONTRACT-ABC`): in re_call_hermes/provider.py, `def get_tool_schemas(self) ->
    list[dict[str, Any]]:` became `def _get_tool_schemas(self) -> list[dict[str, A...`; the test failed at
    `AssertionError: ['get_tool_schemas']`. Restored, green.
    """
    import agent.memory_provider as real

    from re_call_hermes.provider import ReCallMemoryProvider

    assert real.__file__.startswith(HERMES_SRC)
    assert issubclass(ReCallMemoryProvider, real.MemoryProvider)
    assert not inspect.isabstract(ReCallMemoryProvider), sorted(ReCallMemoryProvider.__abstractmethods__)
    assert ReCallMemoryProvider().name == "re-call"


def test_hermes_loads_the_directory_install_by_name(tmp_path, monkeypatch):
    """Invariant: a checkout copied to $HERMES_HOME/plugins/re-call loads as memory.provider re-call.

    Red proof (2026-11-08, CCA run 20261108T171507Z, Linux host, Hermes v0.23.0 2bd378ed, `python
    tests/mutations.py CONTRACT-LOADER`): in __init__.py, `from .re_call_hermes import ReCallMemoryProvider,
    register` became `pass`; the test failed at `assert (None is not None)`. Restored, green.
    """
    home = tmp_path / "home"
    target = home / "plugins" / "re-call"
    shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns(".git", "tests", "__pycache__", ".*cache"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    from plugins.memory import load_memory_provider

    provider = load_memory_provider("re-call", register_skills=False)
    assert provider is not None and provider.name == "re-call"
    assert type(provider).__module__.startswith("_hermes_user_memory.re-call")


def test_memory_manager_routes_prefetch_tools_and_writes(harness, monkeypatch):
    """Invariant: Hermes's MemoryManager reaches prefetch, the tools and on_memory_write.

    Red proof (2026-11-08, CCA run 20261108T171507Z, Linux host, Hermes v0.23.0 2bd378ed, `python
    tests/mutations.py CONTRACT-MANAGER`): in re_call_hermes/provider.py, `if self._writer is None: return #
    One id per write` became `return # One id per write`; the test failed at `assert False`. Restored,
    green.
    """
    from agent.memory_manager import MemoryManager

    from re_call_hermes.provider import ReCallMemoryProvider

    harness.scenario({"recall_search": [SEARCH_OK], "recall_remember": [{"memo_id": "m1.md"}]})
    (harness.data / "config.json").write_text(json.dumps(harness.config), encoding="utf-8")
    os.environ["FAKE_SCENARIO"] = str(harness.scenario_path)
    os.environ["FAKE_LOG"] = str(harness.log_path)
    # Hermes serves a manager's hooks only to callers in the profile it was bound to (#140058).
    monkeypatch.setenv("HERMES_HOME", str(harness.home))
    provider = ReCallMemoryProvider()
    harness.provider = provider
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all("s1", platform="cli", hermes_home=str(harness.home), agent_context="primary")
    from conftest import wait_for

    assert wait_for(provider._ready.is_set), provider._start_error
    assert "Works at Acme" in manager.prefetch_all("where do I work?")
    assert manager.has_tool("recall_remember")
    assert json.loads(manager.handle_tool_call("recall_remember", {"text": "Lives in Rome"})) == {"memo_id": "m1.md"}
    manager.on_memory_write("add", "user", "Has a cat")
    assert wait_for(lambda: len(harness.events("call")) == 3)
    manager.shutdown_all()
    harness.provider = None


def test_the_real_gate_refuses_a_write_nobody_can_approve(harness, monkeypatch):
    """Invariant: against Hermes's real tools/write_approval.py, with memory.write_approval on and no
    approval channel (a test run has none, so Hermes would stage), recall_remember is refused and
    never reaches RE-call; with it off, the same call goes through.

    Red proof (2026-11-20, `HERMES_SRC=... python tests/mutations.py CONTRACT-APPROVAL`): in
    re_call_hermes/provider.py, `if tool_name in WRITE_TOOLS:` became `if False:`; the test failed at
    `AssertionError: assert 'error' in {'memo_id': 'm1.md'}`. Restored, green.
    """
    monkeypatch.setenv("HERMES_HOME", str(harness.home))
    (harness.home / "config.yaml").write_text("memory:\n  write_approval: true\n", encoding="utf-8")
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]})
    provider = harness.start()
    answer = json.loads(provider.handle_tool_call("recall_remember", {"text": "Lives in Rome"}))
    assert "error" in answer
    assert [c for c in harness.events("call") if c["tool"] == "recall_remember"] == []
    (harness.home / "config.yaml").write_text("memory:\n  write_approval: false\n", encoding="utf-8")
    os.utime(harness.home / "config.yaml", (1, 1))  # a different mtime, whatever the cache keys on
    assert json.loads(provider.handle_tool_call("recall_remember", {"text": "Lives in Rome"})) == {"memo_id": "m1.md"}
