"""Source authorization, classification, redaction, and audit context."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from recall.scope import Scope

Classification = Literal["public", "internal", "confidential", "restricted"]


@dataclass(frozen=True)
class AccessContext:
    """The caller attributes required for a source policy decision."""

    principal: str
    tenant: str
    purpose: str = "retrieval"
    clearance: Classification = "internal"
    egress_allowed: bool = False


@dataclass(frozen=True)
class SourceRule:
    """Policy for one source prefix."""

    prefix: str
    classification: Classification = "internal"
    principals: frozenset[str] = field(default_factory=frozenset)
    purposes: frozenset[str] = field(default_factory=lambda: frozenset({"retrieval"}))
    redactions: tuple[str, ...] = ()
    allow_egress: bool = False

    def __post_init__(self) -> None:
        normalized = self.prefix.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("source rule prefix must be non empty")
        object.__setattr__(self, "prefix", normalized)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    source: str
    classification: Classification | None
    redactions: tuple[str, ...]
    reason: str
    policy_digest: str


_CLEARANCE_ORDER: dict[Classification, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}
_REDACTIONS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d[\d ()-]{7,}\d)\b"),
    "secret": re.compile(r"(?i)\b(?:api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
}


class SourceSecurityPolicy:
    """A deny by default source policy that can constrain both writes and reads."""

    def __init__(self, rules: tuple[SourceRule, ...], *, default_deny: bool = True) -> None:
        if not default_deny:
            raise ValueError("source security policy must use default_deny=true")
        self._rules = tuple(rules)
        self._default_deny = default_deny
        payload = {
            "default_deny": default_deny,
            "rules": [
                {
                    "prefix": rule.prefix,
                    "classification": rule.classification,
                    "principals": sorted(rule.principals),
                    "purposes": sorted(rule.purposes),
                    "redactions": list(rule.redactions),
                    "allow_egress": rule.allow_egress,
                }
                for rule in self._rules
            ],
        }
        self.digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def rules(self) -> tuple[SourceRule, ...]:
        return self._rules

    def _matching(self, source: str) -> tuple[SourceRule, ...]:
        normalized = source.replace("\\", "/").strip("/")
        return tuple(
            rule
            for rule in self._rules
            if normalized == rule.prefix or normalized.startswith(rule.prefix + "/")
        )

    def decide(self, source: str, context: AccessContext) -> PolicyDecision:
        matches = sorted(self._matching(source), key=lambda rule: len(rule.prefix), reverse=True)
        if not matches:
            return PolicyDecision(
                False,
                source,
                None,
                (),
                "no source policy rule matched",
                self.digest,
            )
        rule = matches[0]
        allowed = (
            (not rule.principals or context.principal in rule.principals)
            and context.purpose in rule.purposes
            and _CLEARANCE_ORDER[context.clearance] >= _CLEARANCE_ORDER[rule.classification]
            and (context.egress_allowed or not rule.allow_egress)
        )
        reason = "allowed" if allowed else "principal, purpose, clearance, or egress denied"
        return PolicyDecision(
            allowed,
            source,
            rule.classification,
            rule.redactions,
            reason,
            self.digest,
        )

    def redact(self, source: str, text: str, context: AccessContext) -> tuple[str, PolicyDecision]:
        decision = self.decide(source, context)
        if not decision.allowed:
            raise PermissionError(f"source {source!r} denied: {decision.reason}")
        redacted = text
        for name in decision.redactions:
            pattern = _REDACTIONS.get(name)
            if pattern is None:
                raise ValueError(f"unknown redaction rule {name!r}")
            redacted = pattern.sub(f"[REDACTED:{name}]", redacted)
        return redacted, decision

    def scope_for(self, context: AccessContext) -> Scope:
        # A prefix scope is a conservative representation of longest-prefix policy evaluation.
        # Do not emit an allowed parent when a more specific rule denies a descendant, because
        # SQL cannot express the longest-match exception with one positive prefix predicate.
        allowed: list[str] = []
        for rule in self._rules:
            if not self.decide(rule.prefix, context).allowed:
                continue
            has_denied_descendant = any(
                other.prefix.startswith(rule.prefix + "/")
                and not self.decide(other.prefix, context).allowed
                for other in self._rules
            )
            if not has_denied_descendant:
                allowed.append(rule.prefix)
        return Scope(source_prefixes=tuple(dict.fromkeys(allowed)), security_policy_digest=self.digest)

    def constrain_scope(self, scope: Scope | None, context: AccessContext) -> Scope:
        security_scope = self.scope_for(context)
        if scope is None:
            return security_scope
        if (
            scope.security_policy_digest is not None
            and scope.security_policy_digest != self.digest
        ):
            raise ValueError("scope was created for a different source security policy")
        if scope.source_prefixes is None:
            return replace(
                scope,
                source_prefixes=security_scope.source_prefixes,
                security_policy_digest=self.digest,
            )
        combined: list[str] = []
        for left in scope.source_prefixes:
            for right in security_scope.source_prefixes:
                if left == right or left.startswith(right + "/"):
                    combined.append(left)
                elif right.startswith(left + "/"):
                    combined.append(right)
        return replace(
            scope,
            source_prefixes=tuple(dict.fromkeys(combined)),
            security_policy_digest=self.digest,
        )


def load_source_policy(env: dict[str, str] | None = None) -> SourceSecurityPolicy | None:
    """Load the optional JSON policy without importing or constructing any heavy dependency."""

    values = os.environ if env is None else env
    path = values.get("RECALL_SOURCE_POLICY_FILE", "").strip()
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("source policy must be a JSON object")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError("source policy rules must be a JSON array")
    rules: list[SourceRule] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise ValueError(f"source policy rule {index} must be an object")
        rules.append(
            SourceRule(
                prefix=str(raw["prefix"]),
                classification=raw.get("classification", "internal"),
                principals=frozenset(raw.get("principals", ())),
                purposes=frozenset(raw.get("purposes", ("retrieval",))),
                redactions=tuple(raw.get("redactions", ())),
                allow_egress=bool(raw.get("allow_egress", False)),
            )
        )
    return SourceSecurityPolicy(tuple(rules), default_deny=bool(payload.get("default_deny", True)))


def access_context_from_environment(tenant: str, env: dict[str, str] | None = None) -> AccessContext:
    """Build the non-MCP operator context used by manifest and local CLI builds."""

    values = os.environ if env is None else env
    clearance = values.get("RECALL_CLEARANCE", "internal").strip().lower()
    if clearance not in _CLEARANCE_ORDER:
        raise ValueError(
            "RECALL_CLEARANCE must be one of public, internal, confidential, restricted"
        )
    raw_egress = values.get("RECALL_EGRESS_ALLOWED", "0").strip().lower()
    return AccessContext(
        principal=values.get("RECALL_PRINCIPAL", "cli").strip() or "cli",
        tenant=tenant,
        clearance=clearance,  # type: ignore[arg-type]
        egress_allowed=raw_egress in {"1", "true", "yes", "on"},
    )


__all__ = [
    "AccessContext",
    "Classification",
    "PolicyDecision",
    "SourceRule",
    "SourceSecurityPolicy",
    "access_context_from_environment",
    "load_source_policy",
]
