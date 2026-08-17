"""Provider execution metadata shared by reasoning and benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProviderMetadata:
    """Provider neutral identity, usage, timing, and money fields for one model call group.

    Serialized records must include ``model_revision`` and ``monetary_cost_usd`` keys even when
    the provider cannot expose those values and the value is null.
    """

    provider_id: str
    model_id: str
    model_revision: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    monetary_cost_usd: float | None = None
    prompt_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or null")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.monetary_cost_usd is not None:
            if not math.isfinite(self.monetary_cost_usd) or self.monetary_cost_usd < 0:
                raise ValueError("monetary_cost_usd must be finite and non-negative")
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "monetary_cost_usd": self.monetary_cost_usd,
            "prompt_digest": self.prompt_digest,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ProviderMetadata":
        """Deserialize artifact metadata, rejecting non-string provider identities."""

        required = {"provider_id", "model_id", "model_revision", "monetary_cost_usd"}
        missing = required - set(payload)
        if missing:
            raise ValueError("provider metadata missing required fields: " + ", ".join(sorted(missing)))
        return cls(
            provider_id=_required_str(payload["provider_id"]),
            model_id=_required_str(payload["model_id"]),
            model_revision=_optional_str(payload["model_revision"]),
            prompt_tokens=_optional_int(payload.get("prompt_tokens")),
            completion_tokens=_optional_int(payload.get("completion_tokens")),
            total_tokens=_optional_int(payload.get("total_tokens")),
            latency_ms=_optional_int(payload.get("latency_ms")),
            monetary_cost_usd=_optional_float(payload.get("monetary_cost_usd")),
            prompt_digest=_optional_str(payload.get("prompt_digest")),
        )


def validate_cost_claim(metadata: ProviderMetadata) -> None:
    """Require the fields that make a monetary benchmark claim auditable."""

    if metadata.model_revision is None or not metadata.model_revision.strip():
        raise ValueError("benchmark cost claims require model_revision")
    if metadata.monetary_cost_usd is None:
        raise ValueError("benchmark cost claims require monetary_cost_usd")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected string or null")
    return value


def _required_str(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-empty string")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected integer or null")
    if isinstance(value, (str, bytes, bytearray, int)):
        return int(value)
    raise ValueError("expected integer or null")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected number or null")
    if isinstance(value, (str, bytes, bytearray, int, float)):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("expected finite number or null")
        return parsed
    raise ValueError("expected number or null")


def provider_metadata_from_any(value: object) -> ProviderMetadata:
    if isinstance(value, ProviderMetadata):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("provider metadata must be an object")
    return ProviderMetadata.from_mapping(value)


__all__ = ["ProviderMetadata", "provider_metadata_from_any", "validate_cost_claim"]
