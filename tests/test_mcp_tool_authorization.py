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

⚠️ Two rounds of this file were wrong in the same way, and the second is the instructive one.

Round 1 covered the three READ tools and listed `recall_index` and `recall_forget` in a `covered`
set on the claim they were tested elsewhere. No such suite existed, and downgrading `recall_forget`
from `SCOPE_FORGET` to `SCOPE_READ` left **56 passed, zero failures**.

Round 2 replaced the set with the `TOOLS` map below and claimed that "a name in a set is a claim; a
parametrised case is evidence". That was true and insufficient. The map was the single source for
both the cases and the coverage guard, so the guard could not contradict a lie told in both places
at once: editing `recall_forget` to `_require(SCOPE_READ)` **and** updating its `TOOLS` entry gave
**86 passed, 1 skipped** — a read-only token able to permanently delete a tenant's memory, again.

Round 3 — this one — closes it from two directions, because a declaration is only ever checked
against another declaration:

* **Against a published contract.** The required scope is cross-checked against the annotations
  `build_server()` PUBLISHES. `destructiveHint` implies `SCOPE_FORGET`, `readOnlyHint` implies
  `SCOPE_READ`, otherwise `SCOPE_WRITE`. A downgrade must now also lie in the contract an MCP client
  reads to decide whether to auto-approve the call.
* **Against observed BEHAVIOUR, which no declaration can edit.** A tool that mutates resolves a
  second, writable shadow store; a read-only tool must not. That record is taken from the run, so
  the three-place downgrade the annotation check alone still permitted — server body, published
  hint, and `TOOLS` entry, all edited together — is caught by `recall_index` going on resolving a
  writable store while claiming to be read-only.

⚠️ Round 3's own first draft repeated the family defect twice more, and both were found by auditing
this file rather than by writing it. Each is recorded at the test that now covers it:

* a subset assertion, `set(shadow_requested) <= {caller}`, **trivially true of an empty record** — so
  deleting the `get_shadow` call outright, or pointing the deletion path at the unauthenticated
  stdio store, left 42/42 green;
* an env-coverage guard that **could not fail**, because every key it checked matched the prefix rule
  by construction — emptying the exact-match tuple left it green while the file went back to a
  collection error and zero tests run.

Both read as protection while the defect they name was live in the tree. The lesson this file keeps
re-learning, now stated where it cannot be missed: **assert on what the run DID, and execute the path
that makes the guard fire.** A declaration can only ever be checked against another declaration.
"""
from __future__ import annotations

import asyncio
import inspect
import pathlib

import pytest
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

import recall_mcp.server as server_module
import recall_mcp.service as service_module
from recall_mcp.auth import SCOPE_FORGET, SCOPE_READ, SCOPE_WRITE
from recall_mcp.limits import RateLimited
from recall_mcp.server import build_server

#: tool -> (required scope, budget key, a call that reaches its body). ONE source of truth for the
#: parametrised cases and the coverage guard, so a tool cannot be declared covered without a case
#: existing for it. The scope column is NOT trusted on its own — see the module docstring.
TOOLS: dict[str, tuple[str, str, dict]] = {
    "recall_search": (SCOPE_READ, "read", {"query": "q"}),
    "recall_evidence": (SCOPE_READ, "read", {"query": "q"}),
    "recall_reasoning_query": (SCOPE_READ, "read", {"query": "q"}),
    "recall_reasoning_projection": (SCOPE_READ, "read", {}),
    "recall_reasoning_proposals": (SCOPE_READ, "read", {}),
    "recall_reasoning_audit": (SCOPE_READ, "read", {}),
    "recall_stats": (SCOPE_READ, "read", {}),
    "recall_index": (SCOPE_WRITE, "write", {"path": "corpus"}),
    "recall_forget": (SCOPE_FORGET, "forget", {"sources": ["f.md"]}),
}
CASES = [pytest.param(name, id=name) for name in TOOLS]

#: A scope that is never the right one for any tool, used to prove each refuses a wrong scope
#: without depending on which other scopes a token happens to carry.
_ALL_SCOPES = {SCOPE_READ, SCOPE_WRITE, SCOPE_FORGET}

#: The tenant every test authenticates as. Named so the assertion messages read as English.
_CALLER = "acme"

#: A provisioned tenant that is NOT the caller, for proving a tool never reaches sideways.
_OTHER = "globex"

#: Tools that resolve a SECOND, writable store via `registry.get_shadow()` (the shadow generation
#: dual-write). Deliberately a fixed anchor rather than a derivation — see assertion (b) in
#: `test_a_tool_resolves_every_store_it_touches_for_the_callers_own_tenant` for why a derived version
#: cannot catch the deletion it exists to catch.
_DUAL_WRITING_TOOLS = {"recall_index", "recall_forget"}


def _service_entry_points() -> tuple[str, ...]:
    """Every `recall_mcp.service` function `server.py` imported that takes a store first.

    DERIVED, not hand-listed, for the reason `recall_mcp.oidc.oidc_non_issuer_env_keys` records
    about its own list: an enumeration that must be remembered is one that will be forgotten. A
    sixth tool calling a sixth service function is picked up here without anyone editing a tuple.

    The predicate is the VALUE (same function object as the service module's) and the first
    parameter's name, not a name prefix: `make_embedder`, `make_profile_embedder` and
    `startup_retrieval_profile` are imported alongside these and must not be stubbed, because
    replacing them would raise from the lifespan rather than from a tool body.

    `__module__` is checked too, so a RE-EXPORT cannot become an "entry point". Without it, one
    plausible `from recall.readiness import check_enterprise_readiness` in `service.py` was enough to
    get that gate silently replaced by the capture stub for the duration of every test here — a
    substitution with no observable effect in this file, which is exactly why nothing would catch it.
    """
    names = []
    for name in dir(service_module):
        if name.startswith("_"):
            continue
        fn = getattr(service_module, name)
        if not inspect.isfunction(fn) or getattr(server_module, name, None) is not fn:
            continue
        if fn.__module__ != service_module.__name__:
            continue
        params = list(inspect.signature(fn).parameters)
        if params and params[0] == "store":
            names.append(name)
    return tuple(sorted(names))


class _Store:
    """A stand-in whose only job is to be IDENTIFIABLE.

    Nothing calls a method on it: every tool now stops at the service boundary, so the question
    this answers is "is this the object `_require` handed back", not "does it behave like a store".
    """

    def __init__(self, tenant: str) -> None:
        self.tenant = tenant

    def __repr__(self) -> str:  # pragma: no cover - assertion messages only
        return f"<store for {self.tenant!r}>"


class _ReachedService(Exception):
    """A tool got past `_require` and handed a store to its service function.

    This is the stop point, and it is deliberately LATER than the one round 2 used. That version
    stopped inside `registry.get`, the instant the store was handed over, on the reasoning that the
    authorisation question was finished there. It is not: `recall_index` and `recall_forget` resolve
    a SECOND, writable store from `registry.get_shadow()` immediately afterwards, for a tenant
    re-read from the token that never passes through `authorize()`. Stopping at `get` made that
    second resolution unreachable by any test in the file.

    Stopping here instead costs the fidelity the earlier note worried about — the stub registry has
    to model `get_shadow` and `control_plane` honestly — and that cost is the point. `get_shadow` is
    an authorisation surface, so a stub that returns None for everyone is not a stub of it.

    It also makes `_require`'s RETURN VALUE observable for the first time. The suite could prove
    `_require` was called and which tenant it asked for; it could not prove the tool then USED the
    store it got back, so a tool that authorised correctly and then read `state["store"]` passed
    clean.
    """

    def __init__(self, store: object) -> None:
        super().__init__(repr(store))
        self.store = store


class _Registry:
    """Stands in for `StoreRegistry`, including its refusal for an unprovisioned tenant."""

    def __init__(self, allowed: set[str] | None = None) -> None:
        self.requested: list[str] = []
        self.shadow_requested: list[str] = []
        self.handed_out: list[_Store] = []
        self._allowed = {_CALLER, _OTHER} if allowed is None else allowed

    def get(self, tenant: str) -> _Store:
        # Recorded BEFORE the refusal: a cross-tenant attempt must be visible in the record even
        # when it is correctly refused, or the test can only see the ones that succeed.
        self.requested.append(tenant)
        if tenant not in self._allowed:
            # The real one raises here (recall_mcp/stores.py). A stub that always succeeded would
            # make a cross-tenant resolution look fine in this harness.
            raise PermissionError(f"tenant {tenant!r} is not provisioned on this server")
        store = _Store(tenant)
        self.handed_out.append(store)
        return store

    def get_shadow(self, tenant: str) -> None:
        """The second store, resolved by the write and forget tools straight after `_require`.

        The real one (recall_mcp/stores.py) refuses an unprovisioned tenant EXACTLY as `get` does,
        and this reproduces that rather than returning None for everyone. Round 2's version had no
        allowlist and no recording, so repointing both tools at another tenant's shadow store gave
        49 passed — including the end-to-end byte-quota test, which cannot see it because its
        server provisions a single tenant.

        None is the return for an allowed tenant: it is what a server with no shadow generation
        configured gives back, and it keeps the tools on their non-shadow path, whose dual-write
        behaviour belongs to `test_index_quota.py` rather than to an authorisation test.
        """
        self.shadow_requested.append(tenant)
        if tenant not in self._allowed:
            raise PermissionError(f"tenant {tenant!r} is not provisioned on this server")
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
        # Tagged, and deliberately NOT a store any registry ever hands out. This is the store the
        # unauthenticated stdio path uses; a tool that ignores `_require`'s return value and reads
        # it instead fails with a message that says exactly that.
        "store": _Store("UNAUTHENTICATED-STDIO-FALLBACK"),
        "stores": registry,
        "limiter": limiter,
        "calibration": None,
    }


def _stop_at_the_service_boundary(monkeypatch) -> None:
    """Make every service entry point capture the store it was handed and stop there."""

    def _capture(store, *args, **kwargs):
        raise _ReachedService(store)

    entry_points = _service_entry_points()
    assert entry_points, "derived no service entry points; the predicate has gone stale"
    for name in entry_points:
        monkeypatch.setattr(server_module, name, _capture)


def _invoke(
    name: str,
    state: dict,
    token: _Token | None,
    monkeypatch,
    *,
    record: list | None = None,
    stop_at_service: bool = False,
):
    """Await the REGISTERED tool coroutine with a request context and an access token in place."""

    def _get_token():
        if record is not None:
            record.append(True)
        return token

    monkeypatch.setattr(server_module, "get_access_token", _get_token)
    if stop_at_service:
        _stop_at_the_service_boundary(monkeypatch)
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = RequestContext(request_id="t", meta=None, session=None, lifespan_context=state)

    async def run():
        handle = request_ctx.set(ctx)
        try:
            return await tools[name].fn(**TOOLS[name][2])
        finally:
            request_ctx.reset(handle)

    return asyncio.run(run())


def _scope_advertised_by(tool) -> str:
    """The scope a tool's PUBLISHED annotations imply, independent of anything in this file.

    `destructiveHint` implies `SCOPE_FORGET`, `readOnlyHint` implies `SCOPE_READ`, otherwise
    `SCOPE_WRITE`. Both hints must be stated: `None` means the tool declared nothing, and defaulting
    an undeclared tool to `SCOPE_WRITE` would let a new destructive tool inherit a weaker scope
    silently — the exact failure this derivation exists to stop.
    """
    ann = tool.annotations
    assert ann is not None, (
        f"{tool.name} publishes no annotations, so a client cannot tell what it does and this "
        f"file cannot cross-check the scope it enforces"
    )
    read_only, destructive = ann.readOnlyHint, ann.destructiveHint
    assert read_only is not None and destructive is not None, (
        f"{tool.name} leaves readOnlyHint={read_only!r} destructiveHint={destructive!r} "
        f"undeclared; state both"
    )
    assert not (read_only and destructive), (
        f"{tool.name} advertises itself as both read-only and destructive"
    )
    if destructive:
        return SCOPE_FORGET
    if read_only:
        return SCOPE_READ
    return SCOPE_WRITE


@pytest.mark.parametrize("name", CASES)
def test_a_tool_refuses_a_token_without_its_own_scope(name, monkeypatch) -> None:
    """Held to the scope the map declares, so a tool cannot be silently downgraded to a weaker one.

    The token carries every scope EXCEPT the required one, which is what makes this discriminate a
    downgrade: `recall_forget` demoted to `SCOPE_READ` passes a read-scoped token and fails here.
    """
    required, _budget, _kwargs = TOOLS[name]
    registry, limiter = _Registry(), _Limiter()
    token = _Token(sorted(_ALL_SCOPES - {required}), {"tenant": _CALLER})

    with pytest.raises(PermissionError, match=required):
        _invoke(name, _state(registry, limiter), token, monkeypatch)

    assert registry.requested == [], f"{name} obtained a store without the {required} scope"
    assert registry.shadow_requested == [], (
        f"{name} reached the shadow store without the {required} scope"
    )
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
    assert registry.shadow_requested == []


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
    assert registry.shadow_requested == []


@pytest.mark.parametrize("name", CASES)
def test_a_tool_debits_its_own_budget_for_the_callers_own_tenant(name, monkeypatch) -> None:
    """Authorised: the tool's OWN budget is charged exactly once, for the caller's own tenant.

    The budget key is asserted per tool, so `recall_forget` metered as a `read` fails here even
    though scope authorisation would still refuse correctly.
    """
    _required, budget, _kwargs = TOOLS[name]
    registry, limiter = _Registry(), _Limiter()
    token = _Token([TOOLS[name][0]], {"tenant": _CALLER})

    with pytest.raises(_ReachedService):
        _invoke(name, _state(registry, limiter), token, monkeypatch, stop_at_service=True)

    assert limiter.debits == [(_CALLER, budget)], (
        f"{name} debited {limiter.debits!r}; it must charge the {budget!r} budget exactly once"
    )


@pytest.mark.parametrize("name", CASES)
def test_a_tool_uses_the_store_require_handed_back(name, monkeypatch) -> None:
    """The RETURN VALUE of `_require`, not just the fact that it was called.

    Everything else in this file is satisfied by a tool that authorises impeccably and then works
    on a store it obtained some other way. `state["store"]` is exactly such a store — the
    unauthenticated stdio one — and it is present in every context here, so this is not a
    hypothetical: `store = _require(SCOPE_READ)` followed by `store = _state()["store"]` passed
    28 tests.
    """
    registry = _Registry()
    token = _Token([TOOLS[name][0]], {"tenant": _CALLER})

    with pytest.raises(_ReachedService) as excinfo:
        _invoke(name, _state(registry, _Limiter()), token, monkeypatch, stop_at_service=True)

    assert registry.handed_out, f"{name} never obtained a store from the registry"
    assert excinfo.value.store is registry.handed_out[-1], (
        f"{name} passed {excinfo.value.store!r} to its service function rather than the "
        f"{registry.handed_out[-1]!r} that `_require` authorised and handed back"
    )


@pytest.mark.parametrize("name", CASES)
def test_a_tool_resolves_every_store_it_touches_for_the_callers_own_tenant(name, monkeypatch):
    """PRIMARY and SHADOW. `_OTHER` is provisioned, so reaching it is a leak, not a lookup error.

    The shadow half is the one round 2 could not see. `recall_index` and `recall_forget` resolve a
    second, writable store via `registry.get_shadow(_current_tenant(state))`, and that tenant is
    re-read from the token rather than taken from `authorize()`'s return — so it is a resolution
    that no scope check stands in front of. Repointing both at another tenant left 49 tests green.

    The expected shadow record is EXACT, and derived from the declared scope rather than from a list
    of which tools dual-write. A mutating tool must resolve its shadow store, for its own tenant; a
    read-only tool must not resolve one at all.

    That exactness is the whole assertion. The first version of this test said
    `set(shadow_requested) <= {_CALLER}`, which is trivially true of an EMPTY record — so deleting
    `registry.get_shadow(tenant)` from `recall_forget` outright left 42 passed. A subset check over a
    collection nothing requires to be non-empty is a guard that reads as protection and cannot fire,
    which is the same defect, in the same file, as the two rounds before it.

    Deriving from the scope also anchors the WRITE tool, which the annotation cross-check alone does
    not: a three-place downgrade of `recall_index` to `SCOPE_READ` — server body, published
    `readOnlyHint`, and its `TOOLS` entry — is caught HERE, because the tool goes on resolving a
    writable shadow store while claiming to be read-only, and no edit to a declaration changes that.
    """
    required, _budget, _kwargs = TOOLS[name]
    registry = _Registry()
    token = _Token([required], {"tenant": _CALLER})

    with pytest.raises(_ReachedService):
        _invoke(name, _state(registry, _Limiter()), token, monkeypatch, stop_at_service=True)

    assert registry.requested == [_CALLER], (
        f"{name} resolved primary stores for {registry.requested!r} rather than only the caller's"
    )

    # (a) THE SECURITY ANCHOR, derived from the declared scope. A read-only tool must not touch a
    # writable store. This is the half a downgrade cannot talk its way past: to satisfy it,
    # `recall_index` demoted to SCOPE_READ would have to actually STOP dual-writing, which is a
    # change in what the server does, not in what three declarations say about it.
    assert not (required == SCOPE_READ and registry.shadow_requested), (
        f"{name} is declared {required!r} yet resolved the WRITABLE shadow store for "
        f"{registry.shadow_requested!r}. Either the scope is a downgrade or the tool is mutating "
        f"under a read-only contract."
    )

    # (b) THE HARNESS-FIDELITY ANCHOR, keyed on NAME on purpose. It is not a security claim, so the
    # objection to allowlists does not apply: its job is to fail when the harness STOPS SEEING a
    # dual-write it used to see. Deriving it from the scope conflated "may mutate" with "dual-writes
    # to a shadow generation" — a correct new SCOPE_WRITE tool with no shadow store then failed a
    # security test citing a rule it does not break. Deriving it from the tool's own source would
    # shrink in lockstep with the deletion it is meant to catch, which is how (b) went vacuous the
    # first time: `set(...) <= {_CALLER}` is trivially true of an empty record, and deleting
    # `get_shadow` from `recall_forget` outright left 42 passed.
    expected_shadow = [_CALLER] if name in _DUAL_WRITING_TOOLS else []
    assert registry.shadow_requested == expected_shadow, (
        f"{name} resolved shadow stores for {registry.shadow_requested!r}, expected "
        f"{expected_shadow!r}. Either it stopped dual-writing through the registry — in which case "
        f"the second store is no longer under this harness's eye — or it reached another tenant's."
    )


@pytest.mark.parametrize("name", CASES)
def test_a_tool_propagates_the_rate_limit_refusal(name, monkeypatch) -> None:
    """A tool must not swallow `RateLimited` — the client needs `retry_after_seconds`."""
    registry = _Registry()
    token = _Token([TOOLS[name][0]], {"tenant": _CALLER})

    with pytest.raises(RateLimited) as excinfo:
        _invoke(name, _state(registry, _Limiter(refuse=True)), token, monkeypatch)

    assert excinfo.value.retry_after_seconds == 1.5
    assert registry.requested == [], "the store was reached despite an exhausted budget"
    assert registry.shadow_requested == [], "the shadow store was reached despite an exhausted budget"


@pytest.mark.parametrize("name", CASES)
def test_the_scope_a_tool_enforces_matches_the_risk_it_advertises(name) -> None:
    """The one assertion in this file that `TOOLS` cannot satisfy by editing itself.

    Every other test derives its expectation from the `TOOLS` map, so a coordinated edit to
    `recall_mcp/server.py` AND to `TOOLS` passes all of them — 86 passed with a read-only token
    able to permanently delete a tenant's memory. This compares the map against the annotations
    `build_server()` publishes, which no edit to this file can change.

    What it closes: the two-place lie. A downgrade must now also flip the tool's published
    `destructiveHint`/`readOnlyHint`, which is the contract an MCP client reads to decide whether to
    auto-approve the call, so the lie becomes visible to every consumer rather than only to this
    repository.

    What it does NOT close, on its own: a THREE-place edit passes THIS test. Every literal here is
    editable, so the chain has to be anchored somewhere that is not a literal at all —
    `test_a_tool_resolves_every_store_it_touches_for_the_callers_own_tenant` is that anchor, and it
    catches the three-place downgrade of `recall_index` from observed behaviour. Two of the five
    tools are also pinned independently (`tests/test_mcp_tool_forget.py` for `recall_forget`,
    `tests/test_evidence_wiring.py` for `recall_evidence`).
    """
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    declared, _budget, _kwargs = TOOLS[name]

    assert _scope_advertised_by(tools[name]) == declared, (
        f"{name} is declared to require {declared!r} here, but its published annotations advertise "
        f"{_scope_advertised_by(tools[name])!r}. The scope a tool ENFORCES and the risk it "
        f"ADVERTISES must agree — a client approves the call on the annotation."
    )


def test_every_registered_tool_has_an_authorisation_case_here() -> None:
    """The guard on the guard, derived from the live registry — no allowlist by name.

    Round 1 named `recall_index` and `recall_forget` in a `covered` set and asserted they were
    tested elsewhere. They were not, and a `SCOPE_FORGET` -> `SCOPE_READ` downgrade went unnoticed
    by 56 tests. `TOOLS` now drives the cases, so a tool listed here necessarily has one of each.
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


def test_the_auth_env_neutralised_before_collection_covers_every_key_that_can_break_the_import() -> None:
    """`tests/conftest.py` clears by PREFIX; the modules that own the keys are the source of truth.

    The autouse fixture this replaces could not fire. `recall_mcp/server.py` runs `build_server()`
    at module scope, so a shell carrying `RECALL_TRANSPORT=streamable-http` raised `AuthConfigError`
    during IMPORT — a collection error in five test modules, zero tests run, this file among them.
    A fixture runs after collection, so it was never in a position to help; `server.py` also freezes
    `RECALL_TRANSPORT` into a module constant at import, which a `delenv` could not undo either.

    This asserts the prefix rule in `conftest` still covers every key the auth path actually reads,
    so a new `RECALL_AUTH_*`-shaped variable that escapes it is caught here rather than by a
    developer whose whole security suite silently stops running.

    ⚠️ On its own this assertion CANNOT FAIL, and the first version of it shipped alone. Every key in
    `owned` starts with `RECALL_AUTH_` or `RECALL_OIDC_` by construction, so the prefix branch
    satisfies it whatever `_IMPORT_TIME_ENV_EXACT` holds: emptying that tuple left this test at 42 passed
    while `RECALL_TRANSPORT=streamable-http` put the file back to a collection error with zero tests
    run — the precise failure named two paragraphs up, undetected by its own guard. It is kept for
    the coverage question it does answer, and paired with the positive control below, which executes
    the failure instead of describing it.
    """
    from recall_mcp.auth import ENV_TOKENS_FILE
    from recall_mcp.oidc import ENV_ISSUER, oidc_non_issuer_env_keys
    from tests.conftest import neutralised_env_keys

    owned = {ENV_TOKENS_FILE, ENV_ISSUER, *oidc_non_issuer_env_keys()}
    uncovered = {key for key in owned if not neutralised_env_keys(key)}

    assert not uncovered, (
        f"conftest does not neutralise {sorted(uncovered)} before collection; an operator with "
        f"those set cannot run this suite at all"
    )


#: The env a developer or CI shell can realistically be carrying. `RECALL_TRANSPORT`, either OIDC
#: key, and `RECALL_PORT` EACH make `import recall_mcp.server` raise on their own (measured, one at a
#: time). `RECALL_ENV` does NOT — it is here because `conftest` must neutralise it too, not because
#: it raises. Saying "every one of these makes the import raise" was false for that key, and an
#: overclaiming comment on a positive control is the same defect this file exists to stop.
_HOSTILE_ENV = {
    "RECALL_TRANSPORT": "streamable-http",
    "RECALL_ENV": "production",
    "RECALL_OIDC_AUDIENCE": "recall-api",
    "RECALL_OIDC_TENANTS": "acme",
    # Not auth at all: parsed by `_read_int_env` at import, raising ValueError during collection.
    "RECALL_PORT": "99999",
}

#: Stripped from the child's environment. `PYTEST_ADDOPTS` carrying an everyday `-k`/`-m` filter is
#: inherited by the child, which then deselects the tests this control greps for — so the guard fails
#: and reports a collection failure that did not happen, quoting a child that collected everything.
#: A security guard that cries wolf gets deleted; a false alarm is a defect, not caution.
_CHILD_ENV_STRIPPED_PREFIXES = ("PYTEST_", "COV_CORE_")


def test_this_file_still_collects_under_an_operators_deployment_env() -> None:
    """The POSITIVE CONTROL: run the import that used to fail, and require it to succeed.

    The assertion above compares two lists and can be satisfied by a rule that protects nothing.
    This spawns a real pytest, hands it the environment that broke collection, and asserts the
    process came back clean with this file's tests collected. A guard for "the suite still runs" has
    to run the suite; there is no arrangement of literals that establishes it.

    `--collect-only` is enough, because the defect IS a collection error: `build_server()` executes
    at `recall_mcp.server` import time, which happens when pytest imports this module. Nothing here
    reaches the network or a database.
    """
    import os
    import subprocess
    import sys

    child_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(_CHILD_ENV_STRIPPED_PREFIXES)
    }
    child_env.update(_HOSTILE_ENV)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header", __file__],
            env=child_env,
            capture_output=True,
            text=True,
            cwd=str(pathlib.Path(__file__).resolve().parent.parent),
            # Strictly BELOW the 120s `timeout` in pyproject.toml. pytest-timeout's `thread` method
            # expires by calling `os._exit(1)`, which kills the whole session with no summary and
            # ORPHANS this child — so a budget above the ini's can never be reached, and the failure
            # it produces is unreadable. This one fails as an ordinary test instead.
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - only on a pathologically slow box
        pytest.fail(f"child pytest did not finish within 60s: {exc}")

    assert result.returncode == 0, (
        f"this file does not collect when the operator's shell carries {_HOSTILE_ENV}. "
        f"Every test in it is silently not running.\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    # returncode 0 with nothing collected would be the same silence wearing a green hat.
    assert "test_a_tool_refuses_a_token_without_its_own_scope" in result.stdout, (
        f"pytest exited 0 but collected none of this file's tests:\n{result.stdout[-3000:]}"
    )
