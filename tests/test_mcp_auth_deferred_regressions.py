"""Regressions from the deferred pass over the 2026-11-03 MCP, auth and OIDC hunt (#1164).

Each test names the finding it pins (DEF-<n>, with the hunt's original id). The registered tool
coroutines are awaited with a minimal request context, as `tests/test_mcp_auth_hunt_regressions.py`
does. Red proofs (each run against the pre-fix files, failing in its assertion, then green on the
fix) are recorded in the pull request that adds this file.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import recall_mcp.server as server
import recall_mcp.server_auth as server_auth
from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ, AuthConfigError, Principal
from recall_mcp.oidc import OidcConfig, OidcValidator, oidc_validator_from_env
from recall_mcp.server import build_server
from recall_mcp.settings import Settings

ISSUER = "https://idp.example.com"
AUDIENCE = "recall-api"


class _Token:
    def __init__(self, scopes: list[str], tenant: str = "acme") -> None:
        self.scopes = scopes
        self.claims = {"tenant": tenant, "principal": "agent-1"}
        self.client_id = "agent-1"


class _Registry:
    control_plane = None

    def get(self, tenant: str):
        return SimpleNamespace(tenant=tenant)

    def get_shadow(self, tenant: str):
        return None


def _run(name: str, state: dict, monkeypatch, token, **kwargs):
    monkeypatch.setattr(server, "get_access_token", lambda: token)
    tool = {t.name: t for t in build_server()._tool_manager.list_tools()}[name]
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))
    return asyncio.run(tool.fn(ctx=ctx, **kwargs))


def _authenticated(limiter=None) -> dict:
    return {
        "store": None,
        "stores": _Registry(),
        "limiter": limiter,
        "embedder": None,
        "calibration": None,
        "settings": Settings.from_env({}),
    }


# --- DEF-36 (ENV-003) ------------------------------------------------------------------------


def test_logging_settings_padded_with_whitespace_still_apply() -> None:
    """Invariant: a padded RECALL_LOG_LEVEL / RECALL_LOG_FORMAT passed by main() still applies.

    Failure mode caught: configure_logging stripped only its own environment fallback, so
    " json " from an EnvironmentFile fell back to text under a JSON collector.
    """
    from recall.observability import configure_logging, get_logger

    configure_logging(level=" debug ", fmt=" json ", stream=io.StringIO())
    logger = get_logger()
    try:
        assert logger.level == logging.DEBUG
        assert type(logger.handlers[0].formatter).__name__ == "_JsonFormatter"
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


# --- DEF-40 (NUM-001) ------------------------------------------------------------------------


def test_the_sdk_expiry_honours_the_configured_clock_skew() -> None:
    """Invariant: the expiry handed to the SDK carries the same skew the validator accepted.

    Failure mode caught: the validator accepted exp with 60 s leeway and the SDK middleware
    rejected `expires_at < now` with none, so a token in its last skew window was validated and
    then refused.
    """
    exp = datetime(2030, 1, 1, tzinfo=UTC)
    validator = SimpleNamespace(
        config=OidcConfig(issuer=ISSUER, audience=AUDIENCE),
        validate=lambda token: Principal(
            name="user-1", tenant="acme", scopes=frozenset({SCOPE_READ}), expires_at=exp
        ),
    )
    throttle = SimpleNamespace(reserve=lambda: True, refund=lambda: None)
    access = asyncio.run(server_auth.OidcTokenVerifier(validator, throttle).verify_token("t"))  # type: ignore[arg-type]
    assert access is not None
    assert access.expires_at == int(exp.timestamp()) + validator.config.clock_skew_s


# --- DEF-23 (PERF-006) -----------------------------------------------------------------------


def test_oidc_validation_runs_in_its_own_thread_limiter(monkeypatch) -> None:
    """Invariant: token validation does not queue behind tool bodies in anyio's default pool.

    Failure mode caught: validation shared the default limiter with long tool calls, so eight
    long calls left every new request's authentication waiting with no timeout.
    """
    import anyio.to_thread

    seen: list[object] = []

    async def run_sync(fn, *args, limiter=None, **kwargs):
        seen.append(limiter)
        return fn(*args)

    monkeypatch.setattr(anyio.to_thread, "run_sync", run_sync)
    validator = SimpleNamespace(
        config=OidcConfig(issuer=ISSUER, audience=AUDIENCE),
        validate=lambda token: Principal(name="user-1", tenant="acme", scopes=frozenset({SCOPE_READ})),
    )
    throttle = SimpleNamespace(reserve=lambda: True, refund=lambda: None)
    asyncio.run(server_auth.OidcTokenVerifier(validator, throttle).verify_token("t"))  # type: ignore[arg-type]
    limiter = getattr(server_auth, "AUTH_THREAD_LIMITER", None)
    assert seen and seen[0] is not None and seen[0] is limiter


# --- DEF-41 (NUM-003) ------------------------------------------------------------------------


def test_a_refresh_interval_over_the_stale_ceiling_names_the_variable() -> None:
    """Invariant: a refused refresh interval names the variable the operator set.

    Failure mode caught: the refusal named `max_stale_key_s`, a knob with no environment
    variable, so the operator could not tell what to change.
    """
    env = {
        "RECALL_OIDC_ISSUER": ISSUER,
        "RECALL_OIDC_AUDIENCE": AUDIENCE,
        "RECALL_OIDC_TENANTS": "acme",
        "RECALL_OIDC_TRUST_TENANT_CLAIM": "1",
        "RECALL_OIDC_JWKS_REFRESH_SECONDS": "7200",
    }
    with pytest.raises(AuthConfigError, match="RECALL_OIDC_JWKS_REFRESH_SECONDS"):
        oidc_validator_from_env(env)


# --- DEF-42 (NUM-004) ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def test_a_backward_clock_step_refreshes_the_keys(keypair) -> None:
    """Invariant: after the wall clock steps back, the cached JWKS is treated as due.

    Failure mode caught: a negative cache age read as fresh, so a withdrawn key kept verifying
    for the size of the step on top of the refresh interval.
    """
    from jwt.algorithms import RSAAlgorithm

    private, public = keypair
    jwk = json.loads(RSAAlgorithm.to_jwk(public))
    jwk.update({"kid": "key-1", "use": "sig", "alg": "RS256"})
    fetches: list[int] = []

    def fetcher() -> dict:
        fetches.append(1)
        return {"keys": [jwk]}

    now = [1_000_000.0]
    validator = OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=fetcher, _clock=lambda: now[0]
    )
    t = int(time.time())
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "iat": t, "nbf": t, "exp": t + 300, "sub": "u", "tenant": "acme"},
        private,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )
    validator.validate(token)
    now[0] -= 500  # the clock steps back
    validator.validate(token)
    assert len(fetches) == 2


# --- DEF-11 (CODE-014) -----------------------------------------------------------------------


def test_a_naive_as_of_is_read_as_utc_by_current_state(monkeypatch) -> None:
    """Invariant: every tool reads a naive `as_of` as UTC, as recall_reasoning_query does.

    Failure mode caught: recall_current_state passed a naive datetime through, so the same
    timestamp meant different instants in different tools.
    """
    seen: list[datetime] = []

    def current_state(store, *args, **kwargs):
        seen.append(kwargs.get("as_of"))
        raise RuntimeError("stop")

    monkeypatch.setattr(server, "current_state_memory", current_state)
    state = {"store": SimpleNamespace(tenant="acme"), "stores": None, "embedder": None,
             "calibration": None, "settings": Settings.from_env({})}
    with pytest.raises(RuntimeError, match="stop"):
        _run("recall_current_state", state, monkeypatch, None, as_of="2026-01-02T03:04:05")
    assert seen and seen[0] is not None and seen[0].tzinfo is not None


# --- DEF-38 (STAKES-003) ---------------------------------------------------------------------


class _ReleasingLimiter:
    def __init__(self) -> None:
        self.released: list[tuple[str, str]] = []

    def check(self, tenant, key, cost=1.0) -> None:
        pass

    async def release_idempotency_reservation(self, tenant, idempotency_key) -> None:
        self.released.append((tenant, idempotency_key))


def test_an_erasure_refused_before_any_write_releases_its_reservation(monkeypatch) -> None:
    """Invariant: a forget refused before its first write frees the idempotency key.

    Failure mode caught: only LiteWriterBusy released it, so a retry of a refused erasure was
    told the deletion "may have committed" when nothing was deleted.
    """
    limiter = _ReleasingLimiter()
    with pytest.raises(ValueError, match="non-empty"):
        _run("recall_forget", _authenticated(limiter), monkeypatch, _Token([SCOPE_FORGET]),
             sources=[], idempotency_key="key-1")
    assert limiter.released == [("acme", "key-1")]


# --- DEF-43 (DAT-004) ------------------------------------------------------------------------


def test_a_failed_durable_receipt_still_writes_the_replay_cache() -> None:
    """Invariant: when the durable receipt write fails after a commit, Redis still gets the result.

    Failure mode caught: the durable failure propagated before the Redis write, so a retry got
    reconciliation_required although the committed response was in hand.
    """
    stored: list[tuple[str, str]] = []

    class _Limiter:
        async def store_idempotency_result(self, tenant, key, result, **kwargs) -> None:
            stored.append((tenant, key))

    def failing_receipt(*args, **kwargs):
        raise OSError("receipt table unavailable")

    state = {"store": SimpleNamespace(tenant="acme", record_operation_receipt=failing_receipt),
             "limiter": _Limiter()}
    with pytest.raises(OSError, match="receipt table"):
        asyncio.run(server._record_mutation_result(state, "acme", "key-1", "{}", idempotency_operation="op"))
    assert stored == [("acme", "key-1")]


# --- DEF-44 (DAT-006) ------------------------------------------------------------------------


def test_report_stale_reads_the_index_root_from_the_settings_snapshot(monkeypatch) -> None:
    """Invariant: recall_report_stale resolves RECALL_INDEX_ROOT from the server's Settings.

    Failure mode caught: it read os.environ, so a server built with an explicit Settings
    snapshot sent reports to a different root than recall_index confines to.
    """
    seen: list[object] = []

    def report(store, **kwargs):
        seen.append(kwargs.get("env"))
        return {"queued": True}

    monkeypatch.setattr(server, "report_stale", report)
    state = {"store": SimpleNamespace(tenant="acme"), "stores": None, "embedder": None,
             "calibration": None, "settings": Settings.from_env({"RECALL_INDEX_ROOT": "/snapshot-root"})}
    _run("recall_report_stale", state, monkeypatch, None, stale_source="a.md", replacing_source="b.md",
         stale_quote="x" * 24, current_quote="y" * 24)
    assert seen and seen[0] is not None and seen[0].get("RECALL_INDEX_ROOT") == "/snapshot-root"

