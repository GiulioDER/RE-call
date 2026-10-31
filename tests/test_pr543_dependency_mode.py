"""Invalid safety configuration cannot silently turn enforcement off."""
from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from tests.conftest import dev_search, requires_db


@requires_db
@pytest.mark.parametrize("origin", ["argument", "environment", "store"])
def test_invalid_dependency_mode_refuses(make_store, monkeypatch, origin):
    """ENV-001: explicit invalid modes refuse on every configuration route.

    Baseline 76a0b36ea, _trusted_search downgrades the typo to off and fails the
    'silently disabled enforcement' assertion, using a real isolated store.
    """
    store = make_store(64)
    kwargs = {}
    if origin == "argument":
        kwargs["dependency_mode"] = "enforc"
    elif origin == "environment":
        monkeypatch.setenv("RECALL_DEPENDENCY_INVALIDATION", "enforc")
    else:
        monkeypatch.setattr(store, "dependency_invalidation_mode", lambda: "enforc")
    refusal = None
    try:
        dev_search(store, HashingEmbedder(dim=64), "anything", **kwargs)
    except ValueError as exc:
        refusal = exc
    assert refusal is not None and "dependency" in str(refusal).lower(), "invalid dependency mode silently disabled enforcement"
    # An explicit supported off value still overrides an environment or store default.
    dev_search(store, HashingEmbedder(dim=64), "anything", dependency_mode="off")
