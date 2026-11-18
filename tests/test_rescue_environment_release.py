"""Release integration of tenant modes and retained erasure roots."""

from pathlib import Path

import pytest

from recall.cli_commands.index_search import _lite_search_environment
from recall.generation_store import GenerationStore
from recall.lite.atomic import ROOT_ENV, rescue_settings


@pytest.mark.parametrize(("global_mode", "tenant_mode"), [("off", "active"), ("active", "off"), ("off", "shadow")])
def test_postgres_cli_resolves_the_tenants_rescue_mode(
    monkeypatch: pytest.MonkeyPatch, global_mode: str, tenant_mode: str
) -> None:
    """The CLI must resolve the same tenant override as promotion and MCP retrieval."""
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", global_mode)
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE_BY_TENANT", f"tenant-a:{tenant_mode}")
    # The environment consumer needs only the real store's tenant property.
    store = object.__new__(GenerationStore)
    store._tenant = "tenant-a"
    environment = _lite_search_environment(store)
    assert environment is not None and environment["RECALL_ATOMIC_RESCUE_MODE"] == tenant_mode, (
        "PostgreSQL CLI search ignored the tenant's rescue mode"
    )


@pytest.mark.parametrize("mode", ["off", "shadow", "OFF"])
def test_explicit_disable_retains_an_existing_default_erasure_root(tmp_path: Path, mode: str) -> None:
    """A wizard rerun keeps the existing root available for PostgreSQL forget."""
    root = tmp_path / "atomic"
    root.mkdir()
    settings, _ = rescue_settings(root, env={"RECALL_ATOMIC_RESCUE_MODE": mode})
    assert settings.get(ROOT_ENV) == str(root), "explicit disable lost the existing erasure root"
    assert settings["RECALL_ATOMIC_RESCUE_MODE"] == mode
    explicit = tmp_path / "custom"
    settings, _ = rescue_settings(root, env={"RECALL_ATOMIC_RESCUE_MODE": mode, ROOT_ENV: str(explicit)})
    assert settings[ROOT_ENV] == str(explicit), "the default replaced an explicit artifact root"
