"""CCA hunt of `recall/fact_retraction.py` and `recall/fact_projection.py` (run 20261105T225204Z).

Invariants and the failure each one catches:
- ENV-001: a desktop upload records stated facts through the controller DSN the MCP server
  resolved (its runtime environment, secrets included), not through the process environment.

Red proofs are in each test's docstring.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


class _Stop(Exception):
    pass


def test_env001_a_desktop_upload_hands_its_runtime_environment_to_the_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_env001_a_desktop_upload_hands_its_runtime_environment_to_the_manager.

    Red against the pre-fix `recall_mcp/desktop_ingest.py` (base 032c205e4): the manager was
    built with only `serving_environment`, so the controller DSN resolved from a secret never
    reached the stated-fact write, and the test fails on "the upload's runtime environment did
    not reach the generation manager".
    """
    from recall_mcp import desktop_ingest

    seen: dict[str, Any] = {}

    class FakeManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)

        def tenant_ingest_lock(self) -> Any:
            raise _Stop

    monkeypatch.setattr(desktop_ingest, "GenerationManager", FakeManager)
    staged = tmp_path / "tenant" / "job"
    staged.mkdir(parents=True)
    (staged / "note.md").write_text("a note\n", encoding="utf-8")
    env = {"RECALL_FACT_WRITE_DSN": "postgresql://controller@db/recall", "RECALL_ENV": "test"}

    class FakeStore:
        _dsn = "postgresql://serving@db/recall"
        tenant = "t1"

    with pytest.raises(_Stop):
        desktop_ingest.generation_ingest(
            FakeStore(), object(), str(staged), "memory", env=env  # type: ignore[arg-type]
        )

    assert seen.get("fact_write_env") == env, "the upload's runtime environment did not reach the generation manager"


def test_env001_activation_records_stated_facts_through_the_managers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_env001_activation_records_stated_facts_through_the_managers_environment.

    The manager passes its `fact_write_env` to `reconcile_stated_facts` as `write_env`, the
    environment `fact_ledger_for` reads RECALL_FACT_WRITE_DSN from. The pre-fix manager has no
    `fact_write_env`, so a run against base 032c205e4 is not proof (TypeError); red proof is the
    mutation of `GenerationManager._record_stated_facts_after_activation` (recall/generations.py)
    back to `self.record_stated_facts(generation_id)`, which fails on "stated facts were
    recorded through the process environment".
    """
    import recall.generation_store
    import recall.stated_facts
    from recall.generations import GenerationManager

    written: list[Mapping[str, str] | None] = []

    class FakeStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeStore:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def set_fixed_generation(self, generation_id: str) -> None:
            pass

    class Report:
        enabled = True
        facts_stated = 1

    def reconcile(store: object, *, env: object = None, write_env: Mapping[str, str] | None = None) -> Report:
        written.append(write_env)
        return Report()

    monkeypatch.setattr(recall.generation_store, "GenerationStore", FakeStore)
    monkeypatch.setattr(recall.stated_facts, "reconcile_stated_facts", reconcile)
    env = {"RECALL_FACT_WRITE_DSN": "postgresql://controller@db/recall"}
    manager = GenerationManager("postgresql://serving@db/recall", "t1", environment="test", fact_write_env=env)

    assert manager._record_stated_facts_after_activation("g1") == 1
    assert written == [env], "stated facts were recorded through the process environment"
