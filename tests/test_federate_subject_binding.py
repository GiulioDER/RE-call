"""`recall:federate` widens a search to the caller's routes, never to every configured route.

Owner decision 2026-11-04, following the MCP hunt's SEC-002: the scope alone used to let a token
of tenant A name a route that serves only tenant B, and the subject binding
(`RECALL_OIDC_SUBJECT_TENANTS`) constrained the token's own tenant but not the federated legs.

`_retrieval_plan_for` is called directly with a real `RetrievalPlanResolver` and a registry that
is a `StoreRegistry` (so the authenticated branch runs) but validates nothing else. Red proofs are
recorded in the pull request that adds this file.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import recall_mcp.server as server
from recall.retrieval_plan import RetrievalPlanResolver
from recall_mcp.auth import SCOPE_FEDERATE, SCOPE_READ
from recall_mcp.stores import StoreRegistry


class _Registry(StoreRegistry):
    def __init__(self) -> None:  # noqa: D107 - skips the real pools on purpose
        pass

    def validate_retrieval_plan(self, plan):
        return plan


def _resolver() -> RetrievalPlanResolver:
    return RetrievalPlanResolver.from_env(
        {
            "RECALL_RETRIEVAL_PLANS_JSON": json.dumps(
                {
                    "version": 1,
                    "routes": [
                        {"id": "globex-only", "primary_tenant": "globex", "allowed_tenants": ["globex"]},
                        {
                            "id": "shared",
                            "primary_tenant": "globex",
                            "allowed_tenants": ["acme", "globex"],
                        },
                    ],
                }
            )
        }
    )


def _plan(monkeypatch, route_id: str, claims: dict, scopes=(SCOPE_READ, SCOPE_FEDERATE), tenant="acme"):
    token = SimpleNamespace(scopes=list(scopes), claims=claims, client_id="agent-1")
    monkeypatch.setattr(server, "get_access_token", lambda: token)
    state = {"retrieval_plan_resolver": _resolver(), "stores": _Registry()}
    return server._retrieval_plan_for(
        state,
        SimpleNamespace(tenant=tenant),
        "a question",
        scope=None,
        source=None,
        modality=None,
        route_id=route_id,
    )


def test_a_federate_token_cannot_select_a_route_without_its_own_tenant(monkeypatch) -> None:
    """Invariant: the caller's tenant must be on any federated route it selects.

    Failure mode caught: an `acme` token holding recall:federate names the `globex-only` route
    and reads globex's memory.
    """
    with pytest.raises(PermissionError, match="does not include the caller's tenant"):
        _plan(monkeypatch, "globex-only", {"tenant": "acme"})


def test_a_bound_subject_cannot_federate_into_a_tenant_it_is_not_bound_to(monkeypatch) -> None:
    """Invariant: with a subject binding, every selected leg is a bound tenant.

    Failure mode caught: a subject bound only to `acme` uses the `shared` route, which lists acme
    among its tenants, to search globex.
    """
    with pytest.raises(PermissionError, match="not bound to"):
        _plan(monkeypatch, "shared", {"tenant": "acme", "bound_tenants": ["acme"]})


def test_a_route_inside_the_callers_bound_tenants_is_served(monkeypatch) -> None:
    """Control: a route on the caller's tenant, inside its binding (or with none), still works.

    A deliberately nonbehavioral control, exempt from red proof: it passes before and after the
    change, and exists so the two refusals above cannot be satisfied by refusing everything.
    """
    bound = _plan(monkeypatch, "shared", {"tenant": "acme", "bound_tenants": ["acme", "globex"]})
    assert bound.selected_tenants == ("globex",)
    unbound = _plan(monkeypatch, "shared", {"tenant": "acme"})
    assert unbound.selected_tenants == ("globex",)


def test_a_federation_decision_writes_its_own_access_line(monkeypatch, caplog) -> None:
    """Invariant: a search that reaches other tenants records that decision, allowed or denied.

    Failure mode caught: `_require` logs `allowed` for the caller's own-tenant read, then this
    gate refuses the federation (or reaches globex) and writes nothing, so the access log shows a
    refused cross-tenant attempt as allowed and a permitted one under the caller's tenant only.
    An own-tenant plan writes no second line.

    Red proof, 2026-11-13, CCA run 20261113T103055Z (FIX-001 for SEC-001):
    `python -m cca_checks tautology verify` ran this test against the pre-fix
    `recall_mcp/server.py`, where `_retrieval_plan_for` records nothing; it failed on the
    `caplog.messages` assert (`[] == [...]`).
    """
    caplog.set_level(logging.INFO, logger="recall.mcp.access")
    acme = {"tenant": "acme", "principal": "agent"}
    _plan(monkeypatch, "globex-only", {"tenant": "globex", "principal": "agent"}, tenant="globex")
    _plan(monkeypatch, "shared", acme)
    with pytest.raises(PermissionError, match="does not include the caller's tenant"):
        _plan(monkeypatch, "globex-only", acme)
    with pytest.raises(PermissionError, match="requires the recall:federate scope"):
        _plan(monkeypatch, "shared", acme, scopes=(SCOPE_READ,))
    line = "access outcome={} principal='agent' tenant='globex' scope=recall:federate tool=- session=- client=-"
    assert caplog.messages == [line.format("allowed"), line.format("denied"), line.format("denied")]


def test_the_oidc_verifier_carries_the_subjects_bound_tenants(monkeypatch) -> None:
    """Invariant: a bound subject's token claims name its bound set, so the plan check can see it.

    Failure mode caught: the verifier drops the binding, so `_retrieval_plan_for` sees no
    `bound_tenants` and the subject check never runs.
    """
    import asyncio

    from recall_mcp.auth import Principal
    from recall_mcp.oidc import OidcConfig
    from recall_mcp.server_auth import OidcTokenVerifier

    config = OidcConfig(
        issuer="https://idp.example.com",
        audience="recall-api",
        subject_tenants={"user-1": frozenset({"acme", "globex"})},
    )
    validator = SimpleNamespace(
        config=config,
        validate=lambda token: Principal(name="user-1", tenant="acme", scopes=frozenset({SCOPE_READ})),
    )
    throttle = SimpleNamespace(reserve=lambda: True, refund=lambda: None)
    verifier = OidcTokenVerifier(validator, throttle)  # type: ignore[arg-type]
    access = asyncio.run(verifier.verify_token("a-token"))
    assert access is not None
    assert access.claims.get("bound_tenants") == ["acme", "globex"]
