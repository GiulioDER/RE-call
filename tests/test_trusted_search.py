"""End-to-end trusted_search against real pgvector: supersession redirect, expiry, abstention."""
from __future__ import annotations

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.trust import trusted_search

from tests.conftest import requires_db

V1 = "The API rate limit is one hundred requests per second per client key.\n"
V2 = (
    "---\nsupersedes: rate_v1.md\n---\n"
    "Rate limiting update: twenty requests per second per client key, enforced at the gateway.\n"
)
EXPIRED = "---\nvalid_until: 2020-01-01\n---\nTemporary deploy freeze for the winter release.\n"


def _index(tmp_path, store, files: dict[str, str]):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)


@requires_db
def test_superseded_memory_loses_to_successor(tmp_path, make_store):
    store = make_store(64)
    _index(tmp_path, store, {"rate_v1.md": V1, "rate_v2.md": V2})
    # query worded closer to v1 — semantic similarity alone would return the stale memory
    res = trusted_search(store, HashingEmbedder(dim=64), "API rate limit requests per second", k=5)
    files = [h.provenance.file for h in res.hits]
    assert "rate_v1.md" in files and "rate_v2.md" in files
    assert res.hits[0].provenance.file == "rate_v2.md"
    assert res.hits[0].verdict == "ok"
    stale = next(h for h in res.hits if h.provenance.file == "rate_v1.md")
    assert stale.verdict == "superseded"
    assert stale.validity.superseded_by == "rate_v2.md"
    assert res.abstained is False


@requires_db
def test_expired_only_memory_abstains(tmp_path, make_store):
    store = make_store(64)
    _index(tmp_path, store, {"freeze.md": EXPIRED})
    res = trusted_search(store, HashingEmbedder(dim=64), "deploy freeze winter release", k=5)
    assert res.abstained is True
    assert any(h.verdict == "expired" for h in res.hits)
    assert res.reason != ""


@requires_db
def test_provenance_populated_end_to_end(tmp_path, make_store):
    store = make_store(64)
    _index(tmp_path, store, {"rate_v1.md": V1})
    res = trusted_search(store, HashingEmbedder(dim=64), "API rate limit requests per second", k=3)
    h = res.hits[0]
    assert h.provenance.file == "rate_v1.md"
    assert h.provenance.ord == 0
    assert h.provenance.indexed_at is not None
    assert h.provenance.source.endswith("rate_v1.md")
