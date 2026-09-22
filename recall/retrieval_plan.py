"""Deterministic request aware retrieval plans.

This module plans retrieval across physically isolated tenants without executing federation.  A
plan is the stable handoff consumed by a later executor.  It never contains scores, and it never
permits more than one bounded rescue leg.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from recall.errors import RecallError

RETRIEVAL_PLAN_SCHEMA_VERSION = 1
RETRIEVAL_PLAN_POLICY_VERSION = "request-aware-routes-v1"
DEFAULT_RETRIEVAL_ROUTE_ID = "single-tenant-compat-v1"
MAX_RETRIEVAL_FANOUT = 2
MAX_RETRIEVAL_LEG_LIMIT = 50
MAX_RETRIEVAL_LATENCY_BUDGET_MS = 60_000

Modality = Literal["text", "image", "multimodal", "unknown"]
LegRole = Literal["primary", "rescue"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class RetrievalPlanError(ValueError, RecallError):
    """A request or configured retrieval plan cannot be served safely."""


class RetrievalPlanConfigurationError(RetrievalPlanError):
    """The versioned route configuration is malformed or contradictory."""


@dataclass(frozen=True)
class TenantIdentity:
    """Identity that must travel with a planned leg.

    ``None`` means the identity has not yet been resolved by the serving control plane.  It is
    not a claim that the value is absent.  Executors must resolve it before reading corpus rows.
    """

    tenant: str
    generation: str | None = None
    embedding_profile: str | None = None
    calibration: str | None = None
    calibration_status: str | None = None
    trust_state: str | None = None
    provenance_identity: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.tenant, "tenant")
        for name in (
            "generation",
            "embedding_profile",
            "calibration",
            "calibration_status",
            "trust_state",
            "provenance_identity",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RetrievalPlanError(f"{name} identity must be a non-empty string or null")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "tenant": self.tenant,
            "generation": self.generation,
            "embedding_profile": self.embedding_profile,
            "calibration": self.calibration,
            "calibration_status": self.calibration_status,
            "trust_state": self.trust_state,
            "provenance_identity": self.provenance_identity,
        }


@dataclass(frozen=True)
class RetrievalLeg:
    """One bounded physical tenant leg in a logical retrieval plan."""

    role: LegRole
    tenant: str
    limit: int
    identity: TenantIdentity | None = None

    def __post_init__(self) -> None:
        if self.role not in {"primary", "rescue"}:
            raise RetrievalPlanError("retrieval leg role must be primary or rescue")
        _validate_identifier(self.tenant, "tenant")
        if self.limit < 1 or self.limit > MAX_RETRIEVAL_LEG_LIMIT:
            raise RetrievalPlanError(
                f"retrieval leg limit must be between 1 and {MAX_RETRIEVAL_LEG_LIMIT}"
            )
        if self.identity is not None and self.identity.tenant != self.tenant:
            raise RetrievalPlanError("retrieval leg identity tenant does not match leg tenant")

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "tenant": self.tenant,
            "limit": self.limit,
            "identity": self.identity.as_dict() if self.identity is not None else None,
        }


@dataclass(frozen=True)
class RetrievalPlan:
    """Versioned request specific route plan for a later retrieval executor."""

    route_id: str
    primary_tenant: str
    allowed_tenants: frozenset[str]
    rescue_tenant: str | None = None
    primary_limit: int = 5
    rescue_limit: int = 0
    max_fanout: int = 1
    latency_budget_ms: int = 1_000
    selection_reason: str = "configured_route"
    selected_legs: tuple[RetrievalLeg, ...] = ()
    schema_version: int = RETRIEVAL_PLAN_SCHEMA_VERSION
    policy_version: str = RETRIEVAL_PLAN_POLICY_VERSION

    def __post_init__(self) -> None:
        _validate_identifier(self.route_id, "route_id")
        _validate_identifier(self.primary_tenant, "primary_tenant")
        allowed = frozenset(self.allowed_tenants)
        if not allowed:
            raise RetrievalPlanError("allowed_tenants must not be empty")
        if self.primary_tenant not in allowed:
            raise RetrievalPlanError("primary_tenant must be in allowed_tenants")
        if self.rescue_tenant is not None:
            _validate_identifier(self.rescue_tenant, "rescue_tenant")
            if self.rescue_tenant == self.primary_tenant:
                raise RetrievalPlanError("rescue_tenant must differ from primary_tenant")
            if self.rescue_tenant not in allowed:
                raise RetrievalPlanError("rescue_tenant must be in allowed_tenants")
        if self.schema_version != RETRIEVAL_PLAN_SCHEMA_VERSION:
            raise RetrievalPlanError(
                f"unsupported retrieval plan schema version {self.schema_version!r}"
            )
        if self.primary_limit < 1 or self.primary_limit > MAX_RETRIEVAL_LEG_LIMIT:
            raise RetrievalPlanError("primary_limit is outside the bounded retrieval leg limit")
        if self.rescue_tenant is None and self.rescue_limit != 0:
            raise RetrievalPlanError("rescue_limit must be zero when no rescue tenant is selected")
        if self.rescue_tenant is not None and not 1 <= self.rescue_limit <= MAX_RETRIEVAL_LEG_LIMIT:
            raise RetrievalPlanError("rescue_limit is outside the bounded retrieval leg limit")
        if self.max_fanout not in {1, MAX_RETRIEVAL_FANOUT}:
            raise RetrievalPlanError("max_fanout must be 1 or 2")
        if self.rescue_tenant is None and self.max_fanout != 1:
            raise RetrievalPlanError("max_fanout must be 1 without a selected rescue tenant")
        if self.rescue_tenant is not None and self.max_fanout != MAX_RETRIEVAL_FANOUT:
            raise RetrievalPlanError("a rescue tenant requires max_fanout=2")
        if self.latency_budget_ms < 1 or self.latency_budget_ms > MAX_RETRIEVAL_LATENCY_BUDGET_MS:
            raise RetrievalPlanError("latency_budget_ms is outside the bounded route budget")
        if not self.selection_reason.strip():
            raise RetrievalPlanError("selection_reason must be non-empty")

        legs = self.selected_legs
        if not legs:
            primary = RetrievalLeg("primary", self.primary_tenant, self.primary_limit)
            built = [primary]
            if self.rescue_tenant is not None:
                built.append(RetrievalLeg("rescue", self.rescue_tenant, self.rescue_limit))
            object.__setattr__(self, "selected_legs", tuple(built))
            return
        if len(legs) > self.max_fanout or len(legs) > MAX_RETRIEVAL_FANOUT:
            raise RetrievalPlanError("selected retrieval legs exceed the bounded fanout")
        if legs[0].role != "primary" or legs[0].tenant != self.primary_tenant:
            raise RetrievalPlanError("the first selected leg must be the primary tenant")
        if self.rescue_tenant is None and len(legs) != 1:
            raise RetrievalPlanError("a plan without rescue_tenant may select only its primary leg")
        if self.rescue_tenant is not None:
            if len(legs) != 2 or legs[1].role != "rescue" or legs[1].tenant != self.rescue_tenant:
                raise RetrievalPlanError("selected rescue leg does not match rescue_tenant")
        if len({leg.tenant for leg in legs}) != len(legs):
            raise RetrievalPlanError("retrieval legs must use distinct physical tenants")
        if any(leg.tenant not in allowed for leg in legs):
            raise RetrievalPlanError("selected retrieval leg is outside allowed_tenants")

    @property
    def selected_tenants(self) -> tuple[str, ...]:
        return tuple(leg.tenant for leg in self.selected_legs)

    def with_identities(self, identities: Mapping[str, TenantIdentity]) -> "RetrievalPlan":
        """Return a plan carrying resolved control plane identities for every selected leg."""
        legs = tuple(
            replace(leg, identity=identities.get(leg.tenant, leg.identity))
            for leg in self.selected_legs
        )
        return replace(self, selected_legs=legs)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "route_id": self.route_id,
            "primary_tenant": self.primary_tenant,
            "rescue_tenant": self.rescue_tenant,
            "allowed_tenants": sorted(self.allowed_tenants),
            "selection_reason": self.selection_reason,
            "max_fanout": self.max_fanout,
            "latency_budget_ms": self.latency_budget_ms,
            "selected_legs": [leg.as_dict() for leg in self.selected_legs],
        }


@dataclass(frozen=True)
class _RouteSpec:
    route_id: str
    primary_tenant: str
    allowed_tenants: frozenset[str]
    rescue_tenant: str | None
    primary_limit: int
    rescue_limit: int
    latency_budget_ms: int
    scopes: frozenset[str]
    modalities: frozenset[Modality]
    source_prefixes: tuple[str, ...]
    query_any: tuple[str, ...]


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RetrievalPlanError(f"{field_name} must be a bounded identifier")


def _bounded_int(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise RetrievalPlanConfigurationError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _string_set(value: Any, name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise RetrievalPlanConfigurationError(f"{name} must be a list of non-empty strings")
    return frozenset(item.strip().casefold() for item in value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise RetrievalPlanConfigurationError(f"{name} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _modality_set(value: Any) -> frozenset[Modality]:
    values = _string_set(value, "modalities")
    invalid = values - {"text", "image", "multimodal", "unknown"}
    if invalid:
        raise RetrievalPlanConfigurationError(f"unsupported modality value(s): {sorted(invalid)}")
    return frozenset(cast(Sequence[Modality], tuple(values)))


def _required_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise RetrievalPlanConfigurationError(f"{field_name} must be a bounded identifier")
    _validate_identifier(value, field_name)
    return value


def _parse_spec(payload: Any, index: int) -> _RouteSpec:
    if not isinstance(payload, dict):
        raise RetrievalPlanConfigurationError(f"routes[{index}] must be an object")
    route_id = _required_identifier(payload.get("id"), f"routes[{index}].id")
    primary = _required_identifier(
        payload.get("primary_tenant"), f"routes[{index}].primary_tenant"
    )
    allowed_raw = payload.get("allowed_tenants")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise RetrievalPlanConfigurationError(
            f"routes[{index}].allowed_tenants must be a non-empty list"
        )
    allowed = frozenset(allowed_raw)
    if not all(isinstance(item, str) for item in allowed):
        raise RetrievalPlanConfigurationError(f"routes[{index}].allowed_tenants contains a non-string")
    for tenant in allowed:
        _validate_identifier(tenant, f"routes[{index}].allowed_tenants")
    rescue = payload.get("rescue_tenant")
    if rescue is not None:
        _validate_identifier(rescue, f"routes[{index}].rescue_tenant")
        if rescue not in allowed:
            raise RetrievalPlanConfigurationError(
                f"routes[{index}].rescue_tenant must be in allowed_tenants"
            )
    primary_limit = _bounded_int(payload.get("primary_limit", 5), "primary_limit", MAX_RETRIEVAL_LEG_LIMIT)
    rescue_limit = payload.get("rescue_limit", 2 if rescue else 0)
    if rescue is None and rescue_limit != 0:
        raise RetrievalPlanConfigurationError(
            f"routes[{index}].rescue_limit must be zero without rescue_tenant"
        )
    if rescue is not None:
        rescue_limit = _bounded_int(rescue_limit, "rescue_limit", MAX_RETRIEVAL_LEG_LIMIT)
    latency = _bounded_int(
        payload.get("latency_budget_ms", 1_000), "latency_budget_ms", MAX_RETRIEVAL_LATENCY_BUDGET_MS
    )
    return _RouteSpec(
        route_id=route_id,
        primary_tenant=primary,
        allowed_tenants=allowed,
        rescue_tenant=rescue,
        primary_limit=primary_limit,
        rescue_limit=rescue_limit,
        latency_budget_ms=latency,
        scopes=_string_set(payload.get("scopes"), "scopes"),
        modalities=_modality_set(payload.get("modalities")),
        source_prefixes=_string_tuple(payload.get("source_prefixes"), "source_prefixes"),
        query_any=tuple(_string_set(payload.get("query_any"), "query_any")),
    )


def _parse_config(raw: str | None) -> tuple[str | None, tuple[_RouteSpec, ...]]:
    if raw is None or not raw.strip():
        return None, ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RetrievalPlanConfigurationError("RECALL_RETRIEVAL_PLANS_JSON is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("version") != RETRIEVAL_PLAN_SCHEMA_VERSION:
        raise RetrievalPlanConfigurationError(
            f"retrieval plan config must be an object with version={RETRIEVAL_PLAN_SCHEMA_VERSION}"
        )
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes or len(routes) > 64:
        raise RetrievalPlanConfigurationError("routes must contain between 1 and 64 entries")
    specs = tuple(_parse_spec(item, index) for index, item in enumerate(routes))
    ids = [spec.route_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise RetrievalPlanConfigurationError("route ids must be unique")
    default_route = payload.get("default_route")
    if default_route is not None and default_route not in ids:
        raise RetrievalPlanConfigurationError("default_route must name a configured route")
    return default_route, specs


class RetrievalPlanResolver:
    """Resolve a bounded plan using only request metadata and fixed configured rules."""

    def __init__(self, *, default_route: str | None, routes: Sequence[_RouteSpec]) -> None:
        self._default_route = default_route
        self._routes = tuple(routes)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RetrievalPlanResolver":
        source = {} if env is None else env
        default_route, routes = _parse_config(source.get("RECALL_RETRIEVAL_PLANS_JSON"))
        return cls(default_route=default_route, routes=routes)

    @property
    def configured(self) -> bool:
        return bool(self._routes)

    def resolve(
        self,
        *,
        current_tenant: str,
        query: str,
        request_scope: str | None = None,
        source: str | None = None,
        modality: str | None = None,
        route_id: str | None = None,
    ) -> RetrievalPlan:
        _validate_identifier(current_tenant, "current_tenant")
        selected_modality = _normalize_modality(modality)
        normalized_query = " ".join(query.casefold().split())
        normalized_scope = request_scope.strip().casefold() if request_scope else None
        if route_id is not None:
            spec = next((item for item in self._routes if item.route_id == route_id), None)
            if spec is None:
                raise RetrievalPlanError(f"unknown retrieval route {route_id!r}")
            reason = "explicit_route_id"
            specific = True
        else:
            scored: list[tuple[tuple[int, int, int, int, str], _RouteSpec, list[str]]] = []
            for spec in self._routes:
                reasons: list[str] = []
                scope_match = bool(spec.scopes and normalized_scope in spec.scopes)
                modality_match = bool(spec.modalities and selected_modality in spec.modalities)
                source_match = max(
                    (len(prefix) for prefix in spec.source_prefixes if source and source.startswith(prefix)),
                    default=0,
                )
                query_matches = sum(token in normalized_query for token in spec.query_any)
                if spec.scopes and not scope_match:
                    continue
                if spec.modalities and not modality_match:
                    continue
                if spec.source_prefixes and source_match == 0:
                    continue
                if spec.query_any and query_matches == 0:
                    continue
                if scope_match:
                    reasons.append("request_scope")
                if modality_match:
                    reasons.append("modality")
                if source_match:
                    reasons.append("source_prefix")
                if query_matches:
                    reasons.append("query_signal")
                scored.append(((int(scope_match), int(modality_match), source_match, query_matches, spec.route_id), spec, reasons))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                _, spec, matched = scored[0]
                reason = "+".join(matched) if matched else "configured_default"
                specific = bool(matched)
            elif self._default_route is not None:
                spec = next(item for item in self._routes if item.route_id == self._default_route)
                reason = "configured_default"
                specific = False
            else:
                return _compatibility_plan(current_tenant)

        ambiguous = not specific and bool(spec.rescue_tenant)
        selected_rescue = spec.rescue_tenant if ambiguous else None
        if ambiguous:
            reason = f"{reason}+ambiguous_bounded_rescue"
        return RetrievalPlan(
            route_id=spec.route_id,
            primary_tenant=spec.primary_tenant,
            rescue_tenant=selected_rescue,
            allowed_tenants=spec.allowed_tenants,
            primary_limit=spec.primary_limit,
            rescue_limit=spec.rescue_limit if selected_rescue else 0,
            max_fanout=MAX_RETRIEVAL_FANOUT if selected_rescue else 1,
            latency_budget_ms=spec.latency_budget_ms,
            selection_reason=reason,
        )


def _normalize_modality(value: str | None) -> Modality:
    if value is None or not value.strip():
        return "unknown"
    normalized = value.strip().casefold()
    if normalized not in {"text", "image", "multimodal", "unknown"}:
        raise RetrievalPlanError("modality must be text, image, multimodal, or unknown")
    return normalized  # type: ignore[return-value]


def _compatibility_plan(current_tenant: str) -> RetrievalPlan:
    return RetrievalPlan(
        route_id=DEFAULT_RETRIEVAL_ROUTE_ID,
        primary_tenant=current_tenant,
        allowed_tenants=frozenset({current_tenant}),
        primary_limit=5,
        rescue_limit=0,
        max_fanout=1,
        selection_reason="compatibility_default",
    )


__all__ = [
    "DEFAULT_RETRIEVAL_ROUTE_ID",
    "MAX_RETRIEVAL_FANOUT",
    "MAX_RETRIEVAL_LATENCY_BUDGET_MS",
    "MAX_RETRIEVAL_LEG_LIMIT",
    "Modality",
    "RETRIEVAL_PLAN_POLICY_VERSION",
    "RETRIEVAL_PLAN_SCHEMA_VERSION",
    "RetrievalLeg",
    "RetrievalPlan",
    "RetrievalPlanConfigurationError",
    "RetrievalPlanError",
    "RetrievalPlanResolver",
    "TenantIdentity",
]
