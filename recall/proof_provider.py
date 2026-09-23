"""Configuration boundary for the separately enabled proof obligation provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import os

from recall.answer_provider import OllamaAnswerProvider, resolve_answer_provider
from recall.proof_obligations import PROOF_PROMPT_DIGEST
from recall.provider_metadata import ProviderMetadata


class ProofProviderAdapter:
    """Give the proof controller its own receipt identity and prompt digest."""

    provider_id = "recall.reasoning.proof.openrouter"

    def __init__(self, delegate: OllamaAnswerProvider) -> None:
        self._delegate = delegate

    def __call__(self, system: str, user: str) -> str:
        return self._delegate(system, user)

    def provider_metadata(self) -> ProviderMetadata:
        metadata = self._delegate.provider_metadata()
        return replace(
            metadata,
            provider_id=self.provider_id,
            prompt_digest=PROOF_PROMPT_DIGEST,
        )


def resolve_proof_provider(
    env: Mapping[str, str] | None = None,
) -> ProofProviderAdapter | None:
    """Resolve only the explicitly configured, revision pinned OpenRouter proof provider."""
    source = env if env is not None else os.environ
    enabled = source.get("RECALL_REASONING_PROOF_ENABLED", "0").strip().lower()
    if enabled in {"", "0", "false", "no", "off"}:
        return None
    if enabled not in {"1", "true", "yes", "on"}:
        raise ValueError("RECALL_REASONING_PROOF_ENABLED must be an explicit boolean")
    provider = source.get("RECALL_REASONING_PROOF_PROVIDER", "openrouter").strip().lower()
    if provider != "openrouter":
        raise ValueError("RECALL_REASONING_PROOF_PROVIDER must be 'openrouter'")
    model = source.get("RECALL_REASONING_PROOF_MODEL", "").strip()
    if not model:
        raise ValueError("RECALL_REASONING_PROOF_MODEL is required when proof is enabled")
    revision = source.get("RECALL_REASONING_PROOF_REVISION", "").strip()
    if not revision or revision == "unpinned":
        raise ValueError("RECALL_REASONING_PROOF_REVISION must pin the configured deployment")
    mapped = dict(source)
    mapped.update(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_PROVIDER": "openrouter",
            "RECALL_REASONING_ANSWER_MODEL": model,
            "RECALL_REASONING_ANSWER_REVISION": revision,
            "RECALL_REASONING_ANSWER_BASE_URL": source.get(
                "RECALL_REASONING_PROOF_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            "RECALL_REASONING_ANSWER_TIMEOUT": source.get("RECALL_REASONING_PROOF_TIMEOUT", "60"),
            "RECALL_REASONING_ANSWER_MAX_TOKENS": source.get(
                "RECALL_REASONING_PROOF_MAX_TOKENS", "1024"
            ),
            "RECALL_REASONING_ANSWER_CONTEXT_TOKENS": source.get(
                "RECALL_REASONING_PROOF_CONTEXT_TOKENS", "4096"
            ),
            "RECALL_REASONING_ANSWER_REASONING_EFFORT": source.get(
                "RECALL_REASONING_PROOF_REASONING_EFFORT", "none"
            ),
            "RECALL_REASONING_ANSWER_COST_PER_1K_TOKENS": source.get(
                "RECALL_REASONING_PROOF_COST_PER_1K_TOKENS", ""
            ),
            "RECALL_REASONING_ANSWER_API_KEY": source.get("RECALL_REASONING_PROOF_API_KEY", ""),
        }
    )
    delegate = resolve_answer_provider(mapped)
    if delegate is None:
        raise AssertionError("enabled proof provider resolved to None")
    return ProofProviderAdapter(delegate)


__all__ = ["ProofProviderAdapter", "resolve_proof_provider"]
