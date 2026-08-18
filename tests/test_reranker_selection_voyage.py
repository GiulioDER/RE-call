"""Selecting the Voyage reranker through `RECALL_RERANK_MODEL`.

`resolve_reranker` demands `RECALL_RERANK_REVISION` whenever a model is named, because an unpinned
Hub reference is mutable and reusing the shipped pin would name the wrong artifact in every trace.
That reasoning is exactly right for a Hub model and **does not apply to a cloud API model**: there
is no Hub reference and no revision to pin. Demanding one would make the Voyage reranker
unselectable while looking like a safety check.

What is true instead, and is recorded rather than papered over: `rerank-2.5` is a name resolved on
Voyage's side, so its weights are outside our control in a way a pinned Hub revision is not. That is
a real difference in guarantee between the two rerankers, not a reason to refuse one.
"""

from __future__ import annotations

import pytest

from recall_mcp.service import resolve_reranker


def _env(**kw: str) -> dict[str, str]:
    return {"RECALL_RERANK": "1", **kw}


class TestLocalModelsAreUnchanged:
    """This is additive. An operator's existing configuration must behave identically."""

    def test_a_hub_model_still_requires_a_revision(self) -> None:
        with pytest.raises(ValueError, match="RECALL_RERANK_REVISION"):
            resolve_reranker(_env(RECALL_RERANK_MODEL="BAAI/bge-reranker-base"))

    def test_a_hub_model_with_a_revision_resolves(self) -> None:
        spec = resolve_reranker(
            _env(RECALL_RERANK_MODEL="BAAI/bge-reranker-base", RECALL_RERANK_REVISION="abc123")
        )
        assert spec == ("BAAI/bge-reranker-base", "abc123")

    def test_unset_model_still_means_the_measured_default(self) -> None:
        from recall.rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION

        assert resolve_reranker(_env()) == (DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION)

    def test_rerank_off_still_means_off(self) -> None:
        assert resolve_reranker({"RECALL_RERANK": "0"}) is None

    def test_an_unparseable_flag_is_still_refused(self) -> None:
        """Refused, not read as off: a quietly-unreranked server looks identical to a working one."""
        with pytest.raises(ValueError, match="not a boolean"):
            resolve_reranker({"RECALL_RERANK": "maybe"})


class TestVoyage:
    def test_it_resolves_without_a_revision(self) -> None:
        """The revision requirement is a Hub property. A cloud model has no Hub reference."""
        assert resolve_reranker(_env(RECALL_RERANK_MODEL="voyage:rerank-2.5")) == (
            "voyage:rerank-2.5",
            None,
        )

    def test_bare_voyage_resolves_too(self) -> None:
        assert resolve_reranker(_env(RECALL_RERANK_MODEL="voyage")) == ("voyage", None)

    def test_a_revision_alongside_voyage_is_refused(self) -> None:
        """Accepting it silently would record a pin that pins nothing.

        A trace claiming a Voyage rerank at revision `abc123` asserts a guarantee that does not
        exist, which is worse than recording no revision at all.
        """
        with pytest.raises(ValueError, match="no Hub revision"):
            resolve_reranker(
                _env(RECALL_RERANK_MODEL="voyage:rerank-2.5", RECALL_RERANK_REVISION="abc123")
            )
