"""The MCP server can turn reranking on.

Reranking is the largest retrieval gain this project has measured — LOCOMO hit@5 0.671 -> 0.777,
n=1 536, intervals disjoint through k=10 (FINDINGS §11). The MCP server is how an agent actually
consumes this library, and it had no way to enable it: `service.py` called `trusted_search` without
the `reranker` argument that call has accepted since 0.2.

Off by default, because it costs ~1 050 ms/query and a memory server that silently quadrupled every
query's latency would be choosing for the operator.
"""
from __future__ import annotations

from recall_mcp.service import resolve_reranker


def test_no_reranker_unless_asked():
    """The default must stay fast. ~1 050 ms/query is not a default anyone opted into."""
    assert resolve_reranker(env={}) is None
    assert resolve_reranker(env={"RECALL_RERANK": "0"}) is None
    assert resolve_reranker(env={"RECALL_RERANK": "false"}) is None


def test_a_truthy_flag_selects_the_shipped_cross_encoder():
    """`ms-marco-MiniLM-L-6-v2` is the measured choice, not merely the incumbent:
    `bge-reranker-base` (12x the parameters) is statistically indistinguishable at 6.3x the
    per-query cost, so task match beats model size here."""
    spec = resolve_reranker(env={"RECALL_RERANK": "1"})
    assert spec is not None
    model, revision = spec
    assert "ms-marco" in model
    assert revision, "the shipped model must stay revision-pinned — an unpinned Hub ref is mutable"


def test_a_custom_model_requires_its_own_revision():
    """The shipped pin belongs to the shipped weights. Reusing it for different weights would name
    the wrong artifact in every trace and make two deployments silently incomparable."""
    try:
        resolve_reranker(env={"RECALL_RERANK": "1", "RECALL_RERANK_MODEL": "BAAI/bge-reranker-base"})
    except ValueError as exc:
        assert "revision" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("a custom model without a revision must be refused")


def test_a_custom_model_with_a_revision_is_honoured():
    spec = resolve_reranker(env={
        "RECALL_RERANK": "1",
        "RECALL_RERANK_MODEL": "BAAI/bge-reranker-base",
        "RECALL_RERANK_REVISION": "2cfc18c9415c912f9d8155881c133215df768a70",
    })
    assert spec == ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70")


def test_a_typo_does_not_silently_disable_reranking():
    """`RECALL_RERANK=treu` must not read as 'off'. An operator who asked for reranking and got a
    fast, quiet, unreranked server would never find out — the failure looks exactly like success."""
    try:
        resolve_reranker(env={"RECALL_RERANK": "treu"})
    except ValueError as exc:
        assert "treu" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unparseable flag must be refused, not treated as off")
