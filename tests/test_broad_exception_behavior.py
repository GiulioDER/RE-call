from __future__ import annotations

import asyncio

import pytest


def test_fail_open_hook_keeps_the_prompt_alive(monkeypatch, capsys) -> None:
    from recall_hooks import prompt_time

    def explode(*_args, **_kwargs):
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(prompt_time, "settings", explode)
    assert prompt_time.user_prompt_submit({"prompt": "a prompt long enough to inspect"}) == 0
    assert capsys.readouterr().out == ""


def test_fail_closed_auth_boundary_refuses_an_unclassified_validator_error() -> None:
    from recall_mcp.oidc import OidcConfig
    from recall_mcp.server import OidcTokenVerifier

    class ExplodingValidator:
        config = OidcConfig(
            issuer="https://issuer.example",
            audience="recall",
            allowed_tenants=frozenset({"acme"}),
        )

        def validate(self, _token: str):
            raise RuntimeError("unexpected validator defect")

    assert asyncio.run(OidcTokenVerifier(ExplodingValidator()).verify_token("token")) is None


def test_cleanup_only_registry_failure_still_closes_the_shared_pool(monkeypatch) -> None:
    from recall_mcp import stores

    class FakePool:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class BrokenStore:
        def close(self) -> None:
            raise RuntimeError("store close failed")

    monkeypatch.setattr(stores, "SharedPool", FakePool)
    registry = stores.StoreRegistry(
        dsn="postgresql://example/recall",
        dim=3,
        allowed_tenants=frozenset({"acme"}),
        pool_size=1,
        statement_timeout_ms=1000,
    )
    registry._stores[("acme", "default")] = BrokenStore()

    registry.close()

    assert registry._shared.closed is True


def test_error_translation_hides_the_provider_exception_shape(monkeypatch) -> None:
    from recall.desktop import updates

    def explode(*_args, **_kwargs):
        raise OSError("provider connection details")

    monkeypatch.setattr(updates.urllib.request, "urlopen", explode)
    with pytest.raises(updates.UpdateError, match="release check failed: OSError") as excinfo:
        updates.latest_release()
    assert "provider connection details" not in str(excinfo.value)
