"""Regressions from the 2026-11-03 whole-file CCA hunt over the MCP server, auth and OIDC.

Each test names the finding it pins. The registered tool coroutines are awaited with a minimal
request context, as `tests/test_mcp_tool_authorization.py` does, so the code under test is the
real tool body; only the service function past it and the access token are replaced.

Red proofs (each run against the pre-fix code or the named mutation of the fix, failing in its
assertion, then green on the fix) are recorded in the pull request that adds this file.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server.mcpserver.exceptions import ToolError

import recall_mcp.server as server
from recall.profiles import RetrievalOverloaded
from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ, SCOPE_WRITE, AuthConfigError
from recall_mcp.indexing import IndexPreflightError
from recall_mcp.oidc import IdentityProviderUnavailable, OidcConfig, OidcValidator, oidc_env_present
from recall_mcp.server import build_server
from recall_mcp.settings import Settings

ISSUER = "https://idp.example.com"
AUDIENCE = "recall-api"


class _Stop(Exception):
    """Raised by a stubbed service once it has recorded what it was handed."""


class _Token:
    def __init__(self, scopes: list[str], tenant: str = "acme") -> None:
        self.scopes = scopes
        self.claims = {"tenant": tenant, "principal": "agent-1"}
        self.client_id = "agent-1"


class _Registry:
    """Stands in for `StoreRegistry`; records the thread each lookup ran on."""

    def __init__(self) -> None:
        self.get_threads: list[int] = []
        self.shadow_threads: list[int] = []

    def get(self, tenant: str):
        self.get_threads.append(threading.get_ident())
        return SimpleNamespace(tenant=tenant)

    def get_shadow(self, tenant: str):
        self.shadow_threads.append(threading.get_ident())
        return None

    control_plane = None


def _tools():
    return {t.name: t for t in build_server()._tool_manager.list_tools()}


def _authenticated(registry: _Registry, limiter: object | None = None) -> dict:
    return {
        "store": None,
        "stores": registry,
        "limiter": limiter,
        "embedder": None,
        "calibration": None,
        "settings": Settings.from_env({}),
    }


def _run(name: str, state: dict, monkeypatch, token: _Token | None, **kwargs):
    monkeypatch.setattr(server, "get_access_token", lambda: token)
    tool = _tools()[name]
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))
    return asyncio.run(tool.fn(ctx=ctx, **kwargs))


# --- SEC-001 ---------------------------------------------------------------------------------


def test_report_stale_on_an_authenticated_server_reads_no_file_under_the_shared_root(
    tmp_path: Path, monkeypatch
) -> None:
    """SEC-001: the stale-report sidecar must not be a quote oracle over the process root.

    Invariant: on an authenticated (multi-tenant) server the reply cannot depend on the content
    of a file under RECALL_INDEX_ROOT, and nothing is created there. Failure mode caught: a
    principal names `.env`, sends a wrong `current_quote`, and reads which quote was refused,
    which tells it whether its guess is a verbatim substring of the secret.
    """
    root = tmp_path / "root"
    root.mkdir()
    secret = "OPENROUTER_API_KEY=sk-or-v1-a1b2c3d4e5f6\n"
    (root / ".env").write_text(secret, encoding="utf-8")
    (root / "README.md").write_text("A public readme that anyone can read in the repository.\n", encoding="utf-8")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    state = _authenticated(_Registry())
    token = _Token([SCOPE_WRITE])

    replies = []
    for guess in ("OPENROUTER_API_KEY=sk-or-v1-a", "OPENROUTER_API_KEY=sk-or-v1-z"):
        with pytest.raises(ToolError) as excinfo:
            _run(
                "recall_report_stale",
                state,
                monkeypatch,
                token,
                stale_source=".env",
                replacing_source="README.md",
                stale_quote=guess,
                current_quote="x" * 24,
            )
        replies.append(str(excinfo.value))

    assert replies[0] == replies[1], "the reply depends on the content of .env"
    assert not (root / ".recall").exists()


# --- STAKES-001 ------------------------------------------------------------------------------


class _IdempotentLimiter:
    """The shape of `RedisRateLimiter` that `_require` uses: keys are mandatory for writes."""

    requires_idempotency = True

    def __init__(self) -> None:
        self.checked: list[dict] = []
        self.stored: list[tuple[str, str]] = []

    async def check(self, tenant, key, cost=1.0, **kwargs) -> None:
        self.checked.append(kwargs)

    async def store_idempotency_result(self, tenant, idempotency_key, result, **kwargs) -> None:
        self.stored.append((tenant, idempotency_key))

    async def release_idempotency_reservation(self, tenant, idempotency_key) -> None:
        pass


def test_report_use_accepts_an_idempotency_key_where_the_limiter_requires_one(monkeypatch) -> None:
    """STAKES-001: a limiter that requires keys (production's Redis one) must not lock the tool out.

    Invariant: a key given to recall_report_use reaches the limiter, and the result is recorded
    under it. Failure mode caught: the tool calls `_require` without the key, so every call under
    the production limiter raises IdempotencyKeyRequired.
    """
    limiter = _IdempotentLimiter()
    monkeypatch.setattr(server, "report_use", lambda store, **kwargs: {"recorded": True})
    out = _run(
        "recall_report_use",
        _authenticated(_Registry(), limiter),
        monkeypatch,
        _Token([SCOPE_WRITE]),
        task="answer a question",
        effect="helped",
        idempotency_key="key-1",
    )
    assert json.loads(out) == {"recorded": True}
    assert [c.get("idempotency_key") for c in limiter.checked] == ["key-1"]
    assert limiter.stored == [("acme", "key-1")]


# --- DAT-002 ---------------------------------------------------------------------------------


def test_a_single_tenant_server_refuses_a_tenant_it_does_not_serve(monkeypatch) -> None:
    """DAT-002: stdio must refuse, not silently ignore, an explicit foreign tenant argument.

    Invariant: a request naming tenant B on a server bound to tenant A fails. Failure mode caught:
    the call runs against A and reports success, so an upload or a calibration meant for B lands
    in A.
    """
    calls: list[str] = []
    monkeypatch.setattr(server, "calibration_status", lambda store: calls.append(store.tenant) or {})
    state = {"store": SimpleNamespace(tenant="acme"), "stores": None, "embedder": None, "calibration": None}
    with pytest.raises(PermissionError, match="globex"):
        _run("recall_calibration_status", state, monkeypatch, None, tenant="globex")
    assert calls == []
    # Control: the tenant it does serve is still accepted.
    _run("recall_calibration_status", state, monkeypatch, None, tenant="acme")
    assert calls == ["acme"]


# --- BUG-001 ---------------------------------------------------------------------------------


def test_a_retrieval_overload_reaches_the_client_with_its_retry_hint(monkeypatch) -> None:
    """BUG-001: RetrievalOverloaded must cross MCP as a structured ToolError.

    Invariant: the client sees `retrieval_overloaded`, the reason and the retry hint. Failure
    mode caught: a non-ToolError exception, which the SDK replaces with a bare "Error executing
    tool", dropping the fields the tool documents.
    """

    def overloaded(*args, **kwargs):
        raise RetrievalOverloaded("busy", reason="queue_full", retry_after_seconds=1.5)

    class _Plan:
        def as_dict(self) -> dict:
            return {}

    monkeypatch.setattr(server, "prepare_federated_execution", overloaded)
    monkeypatch.setattr(server, "_retrieval_plan_for", lambda *args, **kwargs: _Plan())
    state = {
        "store": SimpleNamespace(tenant="default"),
        "embedder": object(),
        "calibration": None,
        "settings": Settings.from_env({}),
    }
    with pytest.raises(ToolError) as excinfo:
        _run("recall_search", state, monkeypatch, None, query="anything")
    payload = json.loads(str(excinfo.value))
    assert payload == {"error": "retrieval_overloaded", "reason": "queue_full", "retry_after_seconds": 1.5}


# --- BUG-002 and PERF-003 --------------------------------------------------------------------


class _ReleasingLimiter:
    def __init__(self) -> None:
        self.released: list[tuple[str, str]] = []

    def check(self, tenant, key, cost=1.0) -> None:
        pass

    async def release_idempotency_reservation(self, tenant, idempotency_key) -> None:
        self.released.append((tenant, idempotency_key))


def test_an_ingest_refused_before_any_write_releases_its_reservation(tmp_path: Path, monkeypatch) -> None:
    """BUG-002: a preflight refusal in recall_ingest must free the idempotency key.

    Invariant: when ingest is refused before writing (IndexPreflightError), the reservation is
    released. Failure mode caught: the key stays reserved, so the retry is told the mutation may
    have committed although nothing ran.
    """
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path / "index-root"))

    def refused(*args, **kwargs):
        raise IndexPreflightError("refused before any write")

    monkeypatch.setattr(server, "ingest_into_serving_store", refused)
    limiter = _ReleasingLimiter()
    with pytest.raises(IndexPreflightError):
        _run(
            "recall_ingest",
            _authenticated(_Registry(), limiter),
            monkeypatch,
            _Token([SCOPE_WRITE]),
            files=[{"name": "memo.md", "content_b64": "bWVtb3J5"}],
            idempotency_key="key-1",
        )
    assert limiter.released == [("acme", "key-1")]


def test_ingest_stages_uploads_off_the_event_loop(tmp_path: Path, monkeypatch) -> None:
    """PERF-003: decoding and writing up to 50 MiB of uploads must not run on the event loop.

    Invariant: `stage_uploads` runs on a worker thread. Failure mode caught: it runs on the loop
    thread and stalls every other request for the duration.
    """
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path / "index-root"))
    seen: list[int] = []

    def staged(tenant, files):
        seen.append(threading.get_ident())
        raise _Stop

    monkeypatch.setattr(server, "stage_uploads", staged)
    with pytest.raises(_Stop):
        _run(
            "recall_ingest",
            _authenticated(_Registry()),
            monkeypatch,
            _Token([SCOPE_WRITE]),
            files=[{"name": "memo.md", "content_b64": "bWVtb3J5"}],
        )
    assert seen and seen[0] != threading.get_ident()


# --- PERF-001 and PERF-002 -------------------------------------------------------------------


def test_the_store_registry_is_consulted_off_the_event_loop(monkeypatch) -> None:
    """PERF-001: `_require` must not run StoreRegistry.get on the event loop.

    Invariant: the registry lookup, which can do PostgreSQL work under a process-wide lock, runs
    on a worker thread. Failure mode caught: it runs on the loop thread.
    """
    registry = _Registry()
    monkeypatch.setattr(server, "calibration_status", lambda store: {})
    _run("recall_calibration_status", _authenticated(registry), monkeypatch, _Token([SCOPE_READ]))
    assert registry.get_threads and threading.get_ident() not in registry.get_threads


def test_forget_resolves_the_shadow_store_off_the_event_loop(monkeypatch) -> None:
    """PERF-002: the shadow-store lookup in recall_forget must not run on the event loop.

    Invariant: `get_shadow` runs on a worker thread. Failure mode caught: it runs on the loop
    thread, where it holds the registry lock across a control-plane read.
    """
    registry = _Registry()

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(server, "forget_memory", stop)
    with pytest.raises(_Stop):
        _run("recall_forget", _authenticated(registry), monkeypatch, _Token([SCOPE_FORGET]), sources=["memo.md"])
    assert registry.shadow_threads and threading.get_ident() not in registry.shadow_threads


# --- OIDC: STAKES-002, BUG-003, PERF-005, CODE-003 -------------------------------------------


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def _jwks(public_key) -> dict:
    from jwt.algorithms import RSAAlgorithm

    data = json.loads(RSAAlgorithm.to_jwk(public_key))
    data.update({"kid": "key-1", "use": "sig", "alg": "RS256"})
    return {"keys": [data]}


def _token(private_key, *, iss: str = ISSUER) -> str:
    now = int(time.time())
    claims = {
        "iss": iss,
        "aud": AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + 300,
        "sub": "user-1",
        "tenant": "acme",
        "scope": f"{SCOPE_READ} {SCOPE_WRITE}",
    }
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "key-1"})


def test_an_issuer_whose_iss_ends_in_a_slash_is_accepted(keypair) -> None:
    """STAKES-002: an IdP that puts a trailing slash in `iss` (Auth0) must still authenticate.

    Invariant: a genuine token with `iss` = the configured issuer plus "/" validates. Failure
    mode caught: the config strips the slash and PyJWT compares exactly, so every token is
    refused as bad_issuer while discovery looks healthy.
    """
    private, public = keypair
    jwks = _jwks(public)
    validator = OidcValidator(OidcConfig(issuer=ISSUER + "/", audience=AUDIENCE), _jwks_fetcher=lambda: jwks)
    principal = validator.validate(_token(private, iss=ISSUER + "/"))
    assert principal.tenant == "acme"


def test_cold_waiters_share_a_failed_fetch_instead_of_repeating_it(keypair) -> None:
    """BUG-003: during an IdP outage with no cached keys, N callers cost one fetch, not N.

    Invariant: callers queued behind a failing cold fetch fail with it. Failure mode caught: each
    waiter repeats the fetch in turn, so caller N waits N timeouts and holds a worker thread.
    """
    private, _public = keypair
    fetches: list[int] = []
    gate = threading.Barrier(5)

    def failing() -> dict:
        fetches.append(1)
        time.sleep(0.3)
        raise OSError("identity provider unreachable")

    validator = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=failing)
    token = _token(private)
    outcomes: list[str] = []

    def call() -> None:
        gate.wait()
        try:
            validator.validate(token)
            outcomes.append("ok")
        except IdentityProviderUnavailable:
            outcomes.append("unavailable")

    threads = [threading.Thread(target=call) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert outcomes == ["unavailable"] * 5
    assert len(fetches) == 1


def test_a_failed_refresh_backs_off_while_stale_keys_are_served(keypair) -> None:
    """PERF-005: after a failed scheduled refresh, the next request must not refetch at once.

    Invariant: inside the stale window, a failed refresh is retried no sooner than the backoff.
    Failure mode caught: every request during an outage starts a new fetch against the IdP.
    """
    private, public = keypair
    jwks = _jwks(public)
    now = [1_000_000.0]
    fetches: list[int] = []

    def fetcher() -> dict:
        fetches.append(1)
        if len(fetches) == 1:
            return jwks
        raise OSError("identity provider unreachable")

    config = OidcConfig(issuer=ISSUER, audience=AUDIENCE)
    validator = OidcValidator(config, _jwks_fetcher=fetcher, _clock=lambda: now[0])
    token = _token(private)
    validator.validate(token)
    now[0] += config.jwks_refresh_s + 1
    validator.validate(token)  # due: the refresh fails and the stale keys are served
    now[0] += 1
    validator.validate(token)  # inside the backoff: no new fetch
    assert len(fetches) == 2


def test_a_lone_oidc_tuning_key_is_a_partial_block(monkeypatch) -> None:
    """CODE-003: setting only an OIDC tuning key, with no issuer, must be refused.

    Invariant: every RECALL_OIDC_* key counts toward the partial-block guard. Failure mode
    caught: the two tuning keys were string literals the derived key list could not see, so a
    misspelt issuer next to them booted static tokens with an OIDC policy applying to nothing.
    """
    for key in ("RECALL_OIDC_JWKS_REFRESH_SECONDS", "RECALL_OIDC_MAX_TOKEN_LIFETIME_SECONDS"):
        with pytest.raises(AuthConfigError):
            oidc_env_present({key: "60"})


# --- ENV-002 ---------------------------------------------------------------------------------


def test_main_configures_logging_before_building_the_server(monkeypatch) -> None:
    """ENV-002: build_server's auth and trust warnings must reach the configured handler.

    Invariant: `configure_logging` runs before `build_server`. Failure mode caught: the order is
    reversed, so startup warnings bypass RECALL_LOG_FORMAT=json and INFO lines are dropped.
    """
    order: list[str] = []
    monkeypatch.setattr(server, "bootstrap_settings", lambda: Settings.from_env({}))
    monkeypatch.setattr(server, "configure_logging", lambda **kwargs: order.append("logging"))

    def built(settings):
        order.append("build")
        raise _Stop

    monkeypatch.setattr(server, "build_server", built)
    with pytest.raises(_Stop):
        server.main()
    assert order == ["logging", "build"]


# --- ENV-001 ---------------------------------------------------------------------------------


def test_the_documented_jwks_refresh_default_is_the_one_the_code_applies() -> None:
    """ENV-001: the schema behind .env.example and the generated docs must state the real default.

    Invariant: the documented RECALL_OIDC_JWKS_REFRESH_SECONDS default equals the value the
    validator uses when the variable is unset. Failure mode caught: the docs said 300 s while the
    code applied 900 s, so a withdrawn signing key kept verifying three times longer than an
    operator planned for. `settings.py` cannot import the constant (it would load PyJWT), so this
    test is what keeps the two from drifting.
    """
    from recall_mcp.oidc import DEFAULT_JWKS_REFRESH_S
    from recall_mcp.settings import ENVIRONMENT_SCHEMA

    (spec,) = [s for s in ENVIRONMENT_SCHEMA if s.name == "RECALL_OIDC_JWKS_REFRESH_SECONDS"]
    assert spec.default == str(DEFAULT_JWKS_REFRESH_S)
