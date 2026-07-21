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


@requires_db
def test_malformed_metadata_from_direct_upsert_does_not_crash_search(make_store):
    from recall.types import Chunk

    store = make_store(64)
    store.upsert(
        [Chunk("bad", "bad.md", "deploy freeze notes",
               metadata={"file": "bad.md", "ord": 0, "valid_until": "June 2026"})],
        [[1.0] + [0.0] * 63],
    )
    res = trusted_search(store, HashingEmbedder(dim=64), "deploy freeze notes", k=3)
    bad = next(h for h in res.hits if h.provenance.file == "bad.md")
    assert bad.verdict == "invalid_metadata"  # fail closed, no ValueError


@requires_db
def test_trusted_search_rejects_nonpositive_k(make_store):
    import pytest

    store = make_store(64)
    with pytest.raises(ValueError):
        trusted_search(store, HashingEmbedder(dim=64), "anything", k=0)


@requires_db
def test_entailment_judge_is_off_by_default_and_demotes_when_passed(tmp_path, make_store):
    """Ships-OFF: without a judge nothing changes; with one, a non-entailing ok hit abstains."""

    class RejectAll:
        def judge(self, query: str, texts: list[str]) -> list[bool]:
            return [False] * len(texts)

    store = make_store(64)
    _index(tmp_path, store, {"rate_v1.md": V1})
    emb = HashingEmbedder(dim=64)
    plain = trusted_search(store, emb, "API rate limit requests per second", k=3)
    assert plain.abstained is False  # no judge -> trust layer behavior untouched

    judged = trusted_search(store, emb, "API rate limit requests per second", k=3,
                            entailment=RejectAll())
    assert judged.abstained is True
    assert judged.hits[0].verdict == "not_entailed"
    assert "entail" in judged.reason


@requires_db
def test_ambiguous_supersession_target_is_not_served_as_ok(tmp_path, make_store):
    """An unresolvable supersession edge must not read as a healthy memory.

    `supersedes:` names a basename; when two documents carry it the edge is correctly NOT
    guessed — but silently dropping it leaves the (possibly superseded) memory looking `ok`,
    which is the same wrong answer the trust layer exists to prevent. Say it is unresolvable.
    """
    store = make_store(64)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text(
        "Deploy notes: the rollout uses blue green switching.\n", encoding="utf-8"
    )
    (tmp_path / "b" / "notes.md").write_text(
        "Deploy notes: the rollout uses canary switching.\n", encoding="utf-8"
    )
    (tmp_path / "current.md").write_text(
        "---\nsupersedes: notes.md\n---\nThe rollout now uses canary switching only.\n",
        encoding="utf-8",
    )
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)

    res = trusted_search(store, HashingEmbedder(dim=64), "deploy rollout switching", k=5)
    notes = [h for h in res.hits if h.provenance.file in ("a/notes.md", "b/notes.md")]
    assert notes, "both colliding memos should still be retrievable"
    for h in notes:
        assert h.verdict == "ambiguous_supersession"
        assert h.validity.superseded_by is None  # never a guessed successor
