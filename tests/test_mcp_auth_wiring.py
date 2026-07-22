"""Transport gating, the SDK verifier adapter, and per-tenant store isolation.

These cover the wiring between `recall_mcp.auth` (pure) and the running server: which transports
demand authentication, what an authenticated token turns into, and the guarantee that a principal
can only ever reach a store for its own tenant.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ, SCOPE_WRITE, AuthConfigError
from recall_mcp.server import HTTP_TRANSPORTS, RecallTokenVerifier, build_auth
from tests.test_tenancy import tenant_table  # noqa: F401  (fixture)
from recall_mcp.stores import StoreRegistry
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db

TOKEN = "t" * 40


def tokens_file(tmp_path, *, tenant="team-a", scopes=None, name="tokens.json") -> str:
    entry = {"name": "agent", "token": TOKEN, "tenant": tenant}
    if scopes is not None:
        entry["scopes"] = scopes
    p = tmp_path / name
    p.write_text(json.dumps({"principals": [entry]}), encoding="utf-8")
    return str(p)


def http_env(tmp_path, *, scopes=None, **extra) -> dict:
    # `scopes` is threaded through rather than letting callers override the env key with their
    # own tokens_file(): both writes would land on the same path and the later one would win,
    # silently testing the default scopes instead of the requested ones.
    env = {
        "RECALL_AUTH_TOKENS_FILE": tokens_file(tmp_path, scopes=scopes),
        "RECALL_AUTH_ISSUER_URL": "https://recall.example.com",
        "RECALL_AUTH_RESOURCE_URL": "https://recall.example.com",
    }
    env.update(extra)
    return env


# --------------------------------------------------------------------------- transport gating


@pytest.mark.parametrize("transport", sorted(HTTP_TRANSPORTS))
def test_http_transport_refuses_to_start_without_tokens(transport):
    """THE test in this file.

    A network listener with no authentication is the failure this whole feature exists to
    prevent, and the only reliable way to prevent it is to make the process refuse to boot —
    a warning would be emitted into a journal nobody reads while the server comes up healthy
    and world-readable.
    """
    with pytest.raises(AuthConfigError, match="authentication is required"):
        build_auth(transport, env={})


@pytest.mark.parametrize("transport", sorted(HTTP_TRANSPORTS))
def test_http_transport_refuses_to_start_without_the_metadata_urls(transport, tmp_path):
    env = {"RECALL_AUTH_TOKENS_FILE": tokens_file(tmp_path)}
    with pytest.raises(AuthConfigError, match="RECALL_AUTH_ISSUER_URL"):
        build_auth(transport, env=env)


@pytest.mark.parametrize("transport", sorted(HTTP_TRANSPORTS))
def test_fully_configured_http_transport_yields_a_verifier_and_settings(transport, tmp_path):
    verifier, settings, registry = build_auth(transport, env=http_env(tmp_path))
    assert isinstance(verifier, RecallTokenVerifier)
    assert settings is not None
    assert registry.tenants == frozenset({"team-a"})
    # No global required_scopes: a principal provisioned for exactly one capability must not be
    # rejected at the door before the per-tool check runs.
    assert settings.required_scopes == []


def test_stdio_needs_no_tokens_and_stays_unauthenticated():
    """stdio is a private pipe to one client — there is no remote caller to authenticate."""
    assert build_auth("stdio", env={}) == (None, None, None)


def test_stdio_with_tokens_configured_warns_that_they_are_unused(tmp_path, caplog):
    """Silence would let an operator believe stdio is access-controlled when it is not."""
    with caplog.at_level("WARNING"):
        verifier, settings, registry = build_auth("stdio", env=http_env(tmp_path))
    assert (verifier, settings, registry) == (None, None, None)
    assert any("tokens are unused" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- verifier adapter


def verify(verifier, token):
    return asyncio.run(verifier.verify_token(token))


def test_verifier_maps_a_good_token_to_scopes_and_a_tenant_claim(tmp_path):
    verifier, _, _ = build_auth(
        "streamable-http", env=http_env(tmp_path, scopes=[SCOPE_READ, SCOPE_WRITE])
    )
    access = verify(verifier, TOKEN)
    assert access is not None
    assert access.client_id == "agent"
    assert set(access.scopes) == {SCOPE_READ, SCOPE_WRITE}
    # The tenant travels in claims, NOT in scopes — conflating them is how a scope string ends
    # up parsed as a tenant name.
    assert access.claims["tenant"] == "team-a"
    assert SCOPE_FORGET not in access.scopes


def test_verifier_rejects_an_unknown_token_without_logging_it(tmp_path, caplog):
    verifier, _, _ = build_auth("streamable-http", env=http_env(tmp_path))
    secret = "w" * 40
    with caplog.at_level("WARNING"):
        assert verify(verifier, secret) is None
    assert caplog.records, "a rejection should be logged"
    # Even a prefix shrinks a brute-force search space, and logs travel further than expected.
    assert all(secret[:8] not in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- store isolation


def registry(tenants=("team-a", "team-b")) -> StoreRegistry:
    return StoreRegistry(
        dsn="postgresql://u:p@localhost:5432/db",
        dim=64,
        allowed_tenants=frozenset(tenants),
        pool_size=4,
        statement_timeout_ms=1000,
    )


def test_a_tenant_outside_the_provisioned_set_is_refused():
    """Reaching this means an authorisation bug upstream; it must not open a fresh namespace."""
    with pytest.raises(PermissionError, match="not provisioned"):
        registry().get("team-c")


def test_connection_budget_is_computable_before_serving_traffic():
    """Bounded by configuration, not traffic — so it can be checked against max_connections."""
    assert registry(("a", "b", "c")).max_connections() == 3 * 4


def test_nothing_is_opened_until_a_request_arrives():
    assert registry().open_tenants == frozenset()


def test_a_closed_registry_refuses_to_hand_out_stores():
    r = registry()
    r.close()
    with pytest.raises(RuntimeError, match="closed"):
        r.get("team-a")


def test_close_is_idempotent():
    r = registry()
    r.close()
    r.close()


# --------------------------------------------------------------------------- end-to-end (DB)


@requires_db
def test_two_tenants_from_one_registry_cannot_see_each_others_memory(tenant_table):
    """The composition test: `StoreRegistry` + auth must actually isolate, not just intend to.

    `test_tenancy.py` proves a single store scopes its queries and that RLS backstops a forgotten
    predicate. Neither of those would catch the failure mode THIS class introduces — a registry
    that hands the same store (and therefore the same tenant GUC) to two different principals.
    That bug would leave every existing tenancy test green while every caller shared one
    namespace, so it has to be tested at this layer.
    """
    reg = StoreRegistry(
        dsn=TEST_DSN, dim=4, allowed_tenants=frozenset({"acme", "globex"}),
        pool_size=2, statement_timeout_ms=5000, table=tenant_table,
    )
    try:
        acme, globex = reg.get("acme"), reg.get("globex")
        assert acme is not globex, "each tenant must get its own store and its own pool"

        acme.upsert(
            [Chunk(id="c1", source="acme.md", text="acme quarterly revenue", metadata={})],
            [[1.0, 0.0, 0.0, 0.0]],
        )

        assert [h.chunk.source for h in acme.search([1.0, 0.0, 0.0, 0.0], k=5)] == ["acme.md"]
        assert globex.search([1.0, 0.0, 0.0, 0.0], k=5) == []
        # Sparse too: the dense path and the full-text path filter separately, so proving one
        # says nothing about the other.
        assert globex.sparse_search("acme quarterly revenue", k=5) == []
    finally:
        reg.close()


@requires_db
def test_the_same_tenant_is_served_one_cached_store(tenant_table):
    """Otherwise every request opens a fresh pool and the connection budget is fiction."""
    reg = StoreRegistry(
        dsn=TEST_DSN, dim=4, allowed_tenants=frozenset({"acme"}),
        pool_size=2, statement_timeout_ms=5000, table=tenant_table,
    )
    try:
        assert reg.get("acme") is reg.get("acme")
        assert reg.open_tenants == frozenset({"acme"})
    finally:
        reg.close()


@requires_db
def test_concurrent_first_touch_opens_exactly_one_store(tenant_table):
    """Two threads racing a cold tenant must not each build a store and leak one of the pools."""
    reg = StoreRegistry(
        dsn=TEST_DSN, dim=4, allowed_tenants=frozenset({"acme"}),
        pool_size=2, statement_timeout_ms=5000, table=tenant_table,
    )
    try:
        seen: list = []
        barrier = threading.Barrier(6)  # 6 workers only

        def grab():
            barrier.wait()
            seen.append(reg.get("acme"))

        threads = [threading.Thread(target=grab, daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        # The main thread must NOT wait on the barrier: it is sized for the 6 workers, and a
        # seventh party blocks forever once the workers have already passed through it.
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "a worker blocked — the registry lock is not re-entrant-safe"


        assert len(seen) == 6
        assert len({id(s) for s in seen}) == 1, "the race produced more than one store"
    finally:
        reg.close()
