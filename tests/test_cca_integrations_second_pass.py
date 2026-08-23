"""CCA second-pass integration fixes.

- INT-001: the documented ``from_store`` factory forwards ``include_untrusted`` in both adapters.
- CODE-006: both adapters build node/document metadata from one shared ``trust_metadata`` contract,
  so they can no longer diverge (they previously differed on ``chunk_id``).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.types import Chunk, Provenance, TrustedHit, Validity

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

_INDEXED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _hit(verdict: str = "ok") -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(id="f.md#0", source="memory", text="t", metadata={"file": "f.md"}),
        cosine=0.78,
        confidence=1.0,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source="memory", file="f.md", ord=0, indexed_at=_INDEXED_AT),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
    )


def test_trust_metadata_contract_includes_chunk_id() -> None:  # CODE-006
    from recall.integrations import trust_metadata

    md = trust_metadata(_hit())
    assert md["chunk_id"] == "f.md#0"
    assert md["recall_verdict"] == "ok"
    assert md["recall_cosine"] == 0.78
    assert set(md) == {
        "recall_verdict", "recall_confidence", "recall_cosine", "chunk_id",
        "source", "file", "ord", "indexed_at", "superseded_by", "valid_from", "valid_until",
    }


def test_both_adapters_route_metadata_through_the_shared_helper() -> None:  # CODE-006
    from recall.integrations import trust_metadata

    lc = pytest.importorskip("recall.integrations.langchain")
    li = pytest.importorskip("recall.integrations.llamaindex")
    assert lc.trust_metadata is trust_metadata
    assert li.trust_metadata is trust_metadata
    # the chunk_id divergence is gone: both now emit it
    assert lc._hit_to_document(_hit()).metadata["chunk_id"] == "f.md#0"
    assert li._hit_to_node(_hit()).node.metadata["chunk_id"] == "f.md#0"


def test_langchain_from_store_forwards_include_untrusted() -> None:  # INT-001
    lc = pytest.importorskip("recall.integrations.langchain")
    on = lc.RecallRetriever.from_store(object(), object(), include_untrusted=True)
    off = lc.RecallRetriever.from_store(object(), object())
    assert on.include_untrusted is True
    assert off.include_untrusted is False


def test_llamaindex_from_store_forwards_include_untrusted() -> None:  # INT-001
    li = pytest.importorskip("recall.integrations.llamaindex")
    on = li.RecallRetriever.from_store(object(), object(), include_untrusted=True)
    off = li.RecallRetriever.from_store(object(), object())
    assert on._include_untrusted is True
    assert off._include_untrusted is False
