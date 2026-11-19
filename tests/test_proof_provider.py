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


def test_a_bad_answer_ceiling_is_reported_under_the_name_the_operator_set() -> None:
    """Invariant: a malformed RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN on a proof provider is
    reported under that name, the one the proof provider reads.

    Failure mode (CCA ENV-002, run 20261119T182325Z): the proof resolver rewrote every
    RECALL_REASONING_ANSWER_ in an error to RECALL_REASONING_PROOF_, so the operator was told to
    fix RECALL_REASONING_PROOF_MAX_CALLS_PER_MIN, which nothing reads.

    Red proof, 2026-11-19, against `resolve_proof_provider` at cb289667c (`str.replace` on the
    whole message): failed at the `match`, the message naming RECALL_REASONING_PROOF_MAX_CALLS_PER_MIN.
    """
    env = {**_env(), "RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": "0"}
    with pytest.raises(ValueError, match="^RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN must be"):
        resolve_proof_provider(env)
