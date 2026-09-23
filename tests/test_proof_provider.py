"""Configuration tests for the separately enabled proof provider.

Red proof receipt: replace the revision guard in ``resolve_proof_provider`` with ``if False``.
``test_proof_provider_requires_a_pinned_revision`` then fails because an unpinned deployment would
be accepted. Restore the guard before the green run.
"""

from __future__ import annotations

import pytest

from recall.proof_obligations import PROOF_PROMPT_DIGEST
from recall.proof_provider import resolve_proof_provider


def _env() -> dict[str, str]:
    return {
        "RECALL_REASONING_PROOF_ENABLED": "1",
        "RECALL_REASONING_PROOF_PROVIDER": "openrouter",
        "RECALL_REASONING_PROOF_MODEL": "configured/gpt-4o-mini",
        "RECALL_REASONING_PROOF_REVISION": "private-deployment-2026-09-22",
        "RECALL_REASONING_PROOF_API_KEY": "test-key",
    }


def test_proof_provider_is_disabled_by_default() -> None:
    assert resolve_proof_provider({}) is None


def test_proof_provider_requires_a_pinned_revision() -> None:
    env = _env()
    env.pop("RECALL_REASONING_PROOF_REVISION")
    with pytest.raises(ValueError, match="must pin"):
        resolve_proof_provider(env)


def test_proof_provider_has_a_separate_receipt_identity() -> None:
    provider = resolve_proof_provider(_env())

    assert provider is not None
    metadata = provider.provider_metadata()
    assert metadata.provider_id == "recall.reasoning.proof.openrouter"
    assert metadata.model_id == "configured/gpt-4o-mini"
    assert metadata.model_revision == "private-deployment-2026-09-22"
    assert metadata.prompt_digest == PROOF_PROMPT_DIGEST
