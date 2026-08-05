"""EVERY tool is authorised and metered through `_require`, asserted at the tool body.

`_require` is the choke point: a tool cannot obtain a store without passing scope authorisation and
debiting the tenant's budget. That design was untested at the tool level, so deleting
`_require(SCOPE_READ)` from a tool body left the whole suite green.

The gap is the shape that lets a security control rot quietly:

* `tests/test_auth.py` tests `authorize()` exhaustively as a pure function, but never that a tool
  CALLS it;
* `tests/test_evidence_wiring.py` drives the service functions, which sit BELOW `_require` and never
  see a token;
* the registration test greps `inspect.getsource(server)`, which is text, not behaviour.

⚠️ The first version of this file covered the three READ tools and listed `recall_index` and
`recall_forget` in a `covered` set on the claim they were "covered in the suite that owns its
scope". No such suite existed. Downgrading `recall_forget` from `SCOPE_FORGET` to `SCOPE_READ` — a
read-only token able to permanently delete a source — left **56 passed, zero failures**. A name in
a set is a claim; a parametrised case is evidence. Every tool is now parametrised, and the scope map
below is the single source both the cases and the coverage guard derive from.
"""
from __future__ import annotations

import asyncio

import pytest
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

import recall_mcp.server as server_module
from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ, SCOPE_WRITE
from recall_mcp.limits import RateLimited
from recall_mcp.server import build_server

#: tool -> (required scope, budget key, a call that reaches its body). ONE source of truth: the
#: parametrised cases and the coverage guard both derive from it, so a tool cannot be declared
#: covered without a case existing for it.
TOOLS: dict[str, tuple[str, str, dict]] = {
    "recall_search": (SCOPE_READ, "read", {"query": "q"}),
    "recall_evidence": (SCOPE_READ, "read", {"query": "q"}),
    "recall_stats": (SCOPE_READ, "read", {}),
    "recall_index": (SCOPE_WRITE, "write", {"path": "corpus"}),
    "recall_forget": (SCOPE_FORGET, "forget", {"sources": ["f.md"]}),
}
CASES = [pytest.param(name, id=name) for name in TOOLS]

#: A scope that is never the right one for any tool, used to prove each refuses a wrong scope
#: without depending on which other scopes a token happens to carry.
_ALL_SCOPES = {SCOPE_READ, SCOPE_WRITE, SCOPE_FORGET}


@pytest.fixture(autouse=True)
def _neutral_auth_env(monkeypatch):
    """`build_server()` reads os.environ live, and this file calls it once per case.

    Without this, a developer machine with `RECALL_TRANSPORT=streamable-http` or a production auth
    block makes the whole SECURITY suite fail at collection — zero tests run. A security file that
    silently does not execute is worse than one that fails.
    """
    for var in (
        "RECALL_TRANSPORT", "RECALL_AUTH_MODE", "RECALL_AUTH_TOKENS_FILE", "RECALL_ENV",
        "RECALL_OIDC_ISSUER", "RECALL_OIDC_AUDIENCE", "RECALL_OIDC_TENANTS",
        "RECALL_OIDC_SUBJECT_TENANTS", "RECALL_OIDC_TRUST_TENANT_CLAIM",
    ):
        monkeypatch.delenv(var, raising=False)


class _Store:
    """Enough `PgVectorStore` surface for `memory_stats` to return, so one arm can assert success."""

    generation_id = "gen-test"

    def count(self) -> int:
        return 0

    def newest_indexed_at(self):  # type: ignore[no-untyped-def]
        return None

    def sources(self) -> list[str]:
        return []


class _ReachedStore(Exception):
    """`_require` completed and handed back a store. The authorisation question is answered."""


class _Registry:
    """Stands in for `StoreRegistry`, including its refusal for an unprovisioned tenant."""

    def __init__(self, allowed: set[str] | None = None, *, stop: bool = False) -> None:
        self.requested: list[str] = []
        self._allowed = {"acme"} if allowed is None else allowed
        self._stop = stop

    def get(self, tenant: str) -> _Store:
        self.requested.append(tenant)
        if self._stop and tenant in self._allowed:
            # Deliberate full stop. Letting an authorised call run on means stubbing whatever each
            # tool touches next — `get_shadow`, `control_plane`, `sources_for_identifiers`, and so
            # on — which is a growing pile of fidelity debt for an AUTHORISATION test, and every
            # one of those AttributeErrors would have to be caught by a handler broad enough to
            # hide a real refusal. The property under test is finished the moment the store is
            # handed over, so this stops exactly there.
            raise _ReachedStore(tenant)
        if tenant not in self._allowed:
            # The real one raises here (recall_mcp/stores.py). A stub that always succeeded would
            # make a cross-tenant resolution look fine in this harness.
            raise PermissionError(f"tenant {tenant!r} is not provisioned on this server")
        return _Store()

    def get_shadow(self, tenant: str):  # type: ignore[no-untyped-def]
        """The real registry has this; the write and forget tools call it after `_require`.

        Returning None (no shadow generation configured) rather than omitting the method: an
        AttributeError here would be caught by the downstream handler in the authorised arm and
        read as "something failed after authorisation", which is true but uninformative, and it
        would hide the fact that the stub is not the shape of the thing it stands in for.
        """
        return None

    #: Same reason as `get_shadow`: the write and forget tools read it straight after `_require`.
    control_plane = None


class _Limiter:
    def __init__(self, *, refuse: bool = False) -> None:
        self.debits: list[tuple[str, str]] = []
        self._refuse = refuse

    def check(self, tenant: str, key: str, cost: float = 1.0) -> None:
        self.debits.append((tenant, key))
        if self._refuse:
            raise RateLimited("budget exhausted", retry_after_seconds=1.5)


class _Token:
    def __init__(self, scopes: list[str], claims: dict | None) -> None:
        self.scopes = scopes
        self.claims = claims
        self.client_id = "test-client"


def _state(registry: _Registry | None, limiter: _Limiter | None) -> dict:
    from recall.embeddings import HashingEmbedder

    return {
        "embedder": HashingEmbedder(dim=64),
        "store": _Store(),
        "stores": registry,
        "limiter": limiter,
        "calibration": None,
    }


def _invoke(name: str, state: dict, token: _Token | None, monkeypatch, *, record: list | None = None):
    """Await the REGISTERED tool coroutine with a request context and an access token in place."""

    def _get_token():
        if record is not None:
            record.append(True)
        return token

    monkeypatch.setattr(server_module, "get_access_token", _get_token)
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = RequestContext(request_id="t", meta=None, session=None, lifespan_context=state)

    async def run():
        handle = request_ctx.set(ctx)
        try:
            return await tools[name].fn(**TOOLS[name][2])
        finally:
            request_ctx.reset(handle)

    return asyncio.run(run())


@pytest.mark.parametrize("name", CASES)
def test_a_tool_refuses_a_token_without_its_own_scope(name, monkeypatch) -> None:
    """Held to the scope the map declares, so a tool cannot be silently downgraded to a weaker one.

    The token carries every scope EXCEPT the required one, which is what makes this discriminate a
    downgrade: `recall_forget` demoted to `SCOPE_READ` passes a read-scoped token and fails here.
    """
    required, _budget, _kwargs = TOOLS[name]
    registry, limiter = _Registry(), _Limiter()
    token = _Token(sorted(_ALL_SCOPES - {required}), {"tenant": "acme"})

    with pytest.raises(PermissionError, match=required):
        _invoke(name, _state(registry, limiter), token, monkeypatch)

    assert registry.requested == [], f"{name} obtained a store without the {required} scope"
    assert limiter.debits == [], (
        "an unauthorised caller burned the tenant's budget; the debit must come AFTER authorisation"
    )


@pytest.mark.parametrize("name", CASES)
def test_a_tool_refuses_a_token_carrying_no_tenant(name, monkeypatch) -> None:
    """Fails closed rather than defaulting. Matched on the message that belongs to THIS claim.

    `match="tenant"` would also accept the registry's "tenant %r is not provisioned", i.e. a refusal
    that got past `authorize()` entirely.
    """
    required, _budget, _kwargs = TOOLS[name]
    registry = _Registry()

    with pytest.raises(PermissionError, match="carries no tenant claim"):
        _invoke(name, _state(registry, _Limiter()), _Token([required], {}), monkeypatch)

    assert registry.requested == []


@pytest.mark.parametrize("name", CASES)
def test_a_tool_refuses_an_unauthenticated_call(name, monkeypatch) -> None:
    """The token lookup is asserted to have HAPPENED, not merely to have returned None.

    Unpatched, the real `get_access_token()` also returns None outside an auth context, so this arm
    would pass identically without the harness intercepting anything. Recording the call is what
    makes it evidence about the tool rather than about the environment.
    """
    registry, consulted = _Registry(), []

    with pytest.raises(PermissionError, match="requires authentication"):
        _invoke(name, _state(registry, _Limiter()), None, monkeypatch, record=consulted)

    assert consulted, f"{name} never consulted the access token"
    assert registry.requested == []


@pytest.mark.parametrize("name", CASES)
def test_a_tool_debits_its_own_budget_for_the_callers_own_tenant(name, monkeypatch) -> None:
    """Authorised: the store is the CALLER's, and the tool's OWN budget is charged exactly once.

    The budget key is asserted per tool, so `recall_forget` metered as a `read` fails here even
    though scope authorisation would still refuse correctly.

    The registry stops the call the instant it hands the store over. An earlier version let each
    tool run on and caught the fallout with a broad `except Exception`, which swallowed an
    `AttributeError` from an under-specified stub while the test read green.
    """
    _required, budget, _kwargs = TOOLS[name]
    registry, limiter = _Registry(stop=True), _Limiter()
    token = _Token([TOOLS[name][0]], {"tenant": "acme"})

    with pytest.raises(_ReachedStore):
        _invoke(name, _state(registry, limiter), token, monkeypatch)

    assert registry.requested == ["acme"], (
        f"{name} resolved {registry.requested!r} rather than the caller's own tenant"
    )
    assert limiter.debits == [("acme", budget)], (
        f"{name} debited {limiter.debits!r}; it must charge the {budget!r} budget exactly once"
    )


@pytest.mark.parametrize("name", CASES)
def test_a_tool_propagates_the_rate_limit_refusal(name, monkeypatch) -> None:
    """A tool must not swallow `RateLimited` — the client needs `retry_after_seconds`."""
    registry = _Registry()
    token = _Token([TOOLS[name][0]], {"tenant": "acme"})

    with pytest.raises(RateLimited) as excinfo:
        _invoke(name, _state(registry, _Limiter(refuse=True)), token, monkeypatch)

    assert excinfo.value.retry_after_seconds == 1.5
    assert registry.requested == [], "the store was reached despite an exhausted budget"


def test_every_registered_tool_has_an_authorisation_case_here() -> None:
    """The guard on the guard, derived from the live registry — no allowlist by name.

    The previous version named `recall_index` and `recall_forget` in a `covered` set and asserted
    they were tested elsewhere. They were not, and a `SCOPE_FORGET` -> `SCOPE_READ` downgrade went
    unnoticed by 56 tests. `TOOLS` now drives the cases, so a tool listed here necessarily has five
    of them.
    """
    registered = {t.name for t in build_server()._tool_manager.list_tools()}

    missing = registered - set(TOOLS)
    stale = set(TOOLS) - registered
    # BOTH directions, unconditionally. `a or b` short-circuits, and a renamed tool reported only
    # the new name while staying silent that the old one had vanished — the more serious half.
    assert not missing and not stale, (
        f"registered but unauthorised here: {missing or '{}'}; "
        f"declared here but not registered: {stale or '{}'}"
    )
