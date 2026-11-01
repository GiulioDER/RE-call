"""DEPLOY-003: opening a tenant's legacy store never runs DDL from the serving process."""

from __future__ import annotations

import pytest


class _Pool:
    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _registry(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, check_fails: bool = False):
    from recall_mcp import stores

    class _Store:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def check_schema(self) -> None:
            calls.append("check_schema")
            if check_fails:
                raise RuntimeError("SchemaTooOld: run `recall schema apply`")

        def ensure_schema(self) -> None:
            calls.append("ensure_schema")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(stores, "SharedPool", _Pool)
    monkeypatch.setattr(stores, "PgVectorStore", _Store)
    return stores.StoreRegistry(
        dsn="postgresql://serving@example/recall",
        dim=3,
        allowed_tenants=frozenset({"acme"}),
        pool_size=1,
        statement_timeout_ms=1000,
    )


def test_opening_a_legacy_store_checks_the_schema_and_never_migrates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant's first request only checks the schema; it never migrates.

    Red proof: on 414d56f95 `StoreRegistry` called `store.ensure_schema()` here, which runs
    `apply_migrations` with the serving DSN, so `calls` was `["ensure_schema"]` and the equality
    fails. Production line: the `self._control_plane is None` branch in
    `recall_mcp/stores.py`.
    """
    calls: list[str] = []
    registry = _registry(monkeypatch, calls)
    registry.get("acme")
    assert calls == ["check_schema"]


def test_a_failed_schema_check_closes_the_store_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending migration refuses the request and leaks no store.

    Red proof: on 414d56f95 no check ran, so nothing raised and `pytest.raises` failed with
    DID NOT RAISE: the request would have migrated the table instead of refusing.
    """
    calls: list[str] = []
    registry = _registry(monkeypatch, calls, check_fails=True)
    with pytest.raises(RuntimeError, match="SchemaTooOld"):
        registry.get("acme")
    assert calls == ["check_schema", "close"]
    assert registry.open_tenants == frozenset()
