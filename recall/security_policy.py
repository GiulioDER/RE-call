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


def _strict_bool(raw: object, *, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str) and raw.strip().lower() in {"0", "1", "false", "true"}:
        return raw.strip().lower() in {"1", "true"}
    raise ValueError(f"{field_name} must be a boolean")


class SourceSecurityPolicy:
    """A deny by default source policy that can constrain both writes and reads."""

    def __init__(self, rules: tuple[SourceRule, ...], *, default_deny: bool = True) -> None:
        if not default_deny:
            raise ValueError("source security policy must use default_deny=true")
        prefixes = [rule.prefix for rule in rules]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("source security policy cannot contain duplicate prefixes")
        self._rules = tuple(
            sorted(
                rules,
                key=lambda rule: (
                    rule.prefix,
                    rule.classification,
                    tuple(sorted(rule.principals)),
                    tuple(sorted(rule.purposes)),
                    tuple(sorted(rule.redactions)),
                    rule.allow_egress,
                ),
            )
        )
        self._matching_rules = tuple(
            sorted(self._rules, key=lambda rule: (len(rule.prefix), rule.prefix), reverse=True)
        )
        self._default_deny = default_deny
        payload = {
            "default_deny": default_deny,
            "rules": [
                {
                    "prefix": rule.prefix,
                    "classification": rule.classification,
                    "principals": sorted(rule.principals),
                    "purposes": sorted(rule.purposes),
                    "redactions": sorted(rule.redactions),
                    "allow_egress": rule.allow_egress,
                }
                for rule in self._rules
            ],
        }
        self.digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        content_payload = {
            "rules": [
                {"prefix": rule.prefix, "redactions": sorted(rule.redactions)}
                for rule in self._rules
            ]
        }
        self.content_digest = hashlib.sha256(
            json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def rules(self) -> tuple[SourceRule, ...]:
        return self._rules

    def _matching(self, source: str) -> tuple[SourceRule, ...]:
        normalized = source.replace("\\", "/").strip("/")
        return tuple(
            rule
            for rule in self._matching_rules
            if normalized == rule.prefix or normalized.startswith(rule.prefix + "/")
        )

    def decide(self, source: str, context: AccessContext) -> PolicyDecision:
        matches = self._matching(source)
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
        return self.redact_with_decision(text, decision)

    def redact_with_decision(self, text: str, decision: PolicyDecision) -> tuple[str, PolicyDecision]:
        if not decision.allowed:
            raise PermissionError(f"source {decision.source!r} denied: {decision.reason}")
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
        decisions = {rule.prefix: self.decide(rule.prefix, context) for rule in self._rules}
        allowed: list[str] = []
        for rule in self._rules:
            if not decisions[rule.prefix].allowed:
                continue
            has_denied_descendant = any(
                other.prefix.startswith(rule.prefix + "/")
                and not decisions[other.prefix].allowed
                for other in self._rules
            )
            if not has_denied_descendant:
                if not any(prefix == rule.prefix or rule.prefix.startswith(prefix + "/") for prefix in allowed):
                    allowed = [prefix for prefix in allowed if not prefix.startswith(rule.prefix + "/")]
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
        for left in scope.source_prefixes or ():
            for right in security_scope.source_prefixes or ():
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
        prefix = raw.get("prefix")
        classification = raw.get("classification", "internal")
        principals = raw.get("principals", ())
        purposes = raw.get("purposes", ("retrieval",))
        redactions = raw.get("redactions", ())
        if not isinstance(prefix, str):
            raise ValueError(f"source policy rule {index} prefix must be a string")
        if classification not in _CLEARANCE_ORDER:
            raise ValueError(f"source policy rule {index} classification is invalid")
        for name, value in (("principals", principals), ("purposes", purposes), ("redactions", redactions)):
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"source policy rule {index} {name} must be an array of strings")
        rules.append(
            SourceRule(
                prefix=prefix,
                classification=classification,
                principals=frozenset(principals),
                purposes=frozenset(purposes),
                redactions=tuple(redactions),
                allow_egress=_strict_bool(raw.get("allow_egress", False), field_name="allow_egress"),
            )
        )
    return SourceSecurityPolicy(
        tuple(rules),
        default_deny=_strict_bool(payload.get("default_deny", True), field_name="default_deny"),
    )


def access_context_from_environment(
    tenant: str, env: dict[str, str] | None = None, *, purpose: str = "retrieval"
) -> AccessContext:
    """Build the non-MCP operator context used by manifest and local CLI builds."""

    values = os.environ if env is None else env
    clearance = values.get("RECALL_CLEARANCE", "internal").strip().lower()
    if clearance not in _CLEARANCE_ORDER:
        raise ValueError(
            "RECALL_CLEARANCE must be one of public, internal, confidential, restricted"
        )
    egress_allowed = _strict_bool(
        values.get("RECALL_EGRESS_ALLOWED", "0"), field_name="RECALL_EGRESS_ALLOWED"
    )
    return AccessContext(
        principal=values.get("RECALL_PRINCIPAL", "cli").strip() or "cli",
        tenant=tenant,
        purpose=purpose,
        clearance=clearance,
        egress_allowed=egress_allowed,
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
