"""Public trust boundaries for the three confirmed PR895 findings.

Red proof baseline: fdc65bd2385f3603c7243160292d2fcbc149e703, 2026-10-13.
Every parameter of the four tests below must fail in its named behavioral assertion
before the source repair, then pass unchanged afterward. FIX001 targets the metadata
returned by LiteStore._hit and read by supersession_all. FIX002 targets the full
authorization Scope forwarded by HybridRetriever._retrieve_legs. FIX003 targets the
trusted_search snapshot hook and LiteStore.iter_chunks within that pinned view.
The saved native-red receipt records exact node IDs and intended assertion failures.
All data and vectors are synthetic; development mode explicitly reaches retrieval.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.embeddings import HashingEmbedder
from recall.lite import LiteStore
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.types import Chunk
from tests.conftest import dev_search

DIM = 64
QUERY = "rotor bearing pressure"


def _chunk(name: str, **metadata: Any) -> Chunk:
    return Chunk(id=name, source=name, text=QUERY, metadata={"file": name, **metadata})


def _put(store: LiteStore, chunks: list[Chunk]) -> None:
    store.upsert(chunks, HashingEmbedder(dim=DIM).embed([c.text for c in chunks]))


@pytest.mark.parametrize("leg", ["dense", "sparse"])
@pytest.mark.parametrize("mutation", ["add", "remove", "replace_list", "clear_list"])
def test_returned_metadata_cannot_rewrite_supersession(tmp_path: Path, leg: str, mutation: str) -> None:
    """FIX001: outbound dictionaries and nested lists cannot change persisted claims."""
    with closing(LiteStore(tmp_path / "m.db", dim=DIM)) as store:
        claim: dict[str, Any] = {} if mutation == "add" else {"supersedes": ["a.md"]}
        _put(store, [_chunk("a.md"), _chunk("b.md", **claim), _chunk("c.md")])
        vector = HashingEmbedder(dim=DIM).embed([QUERY])[0]
        hits = store.query_dense(vector, 3) if leg == "dense" else store.query_sparse(QUERY, 3, vec=vector)
        metadata = next(h.chunk.metadata for h in hits if h.chunk.id == "b.md")
        if mutation == "add":
            metadata["supersedes"] = "a.md"
        elif mutation == "remove":
            metadata.pop("supersedes")
        elif mutation == "replace_list":
            metadata["supersedes"][0] = "c.md"
        else:
            metadata["supersedes"].clear()
        expected = {} if mutation == "add" else {"a.md": "b.md"}
        assert store.supersession()[0] == expected, "returned metadata rewrote persisted supersession"
        with closing(LiteStore(store.path, dim=DIM)) as fresh:
            assert store.supersession_all() == fresh.supersession_all()


@pytest.mark.parametrize("case", ["prefix", "no_access", "different_digest", "missing_digest"])
def test_source_policy_filters_the_public_lite_trust_path(tmp_path: Path, case: str) -> None:
    """FIX002: deny by default prefixes and stored policy identity remain hard filters."""
    rules = (SourceRule("public", classification="public"),)
    policy = SourceSecurityPolicy(()) if case == "no_access" else SourceSecurityPolicy(rules)
    good = _chunk("public/good.md", security_policy_digest=policy.digest)
    denied = _chunk("private/denied.md", security_policy_digest=policy.digest)
    if case == "different_digest":
        denied = _chunk("public/denied.md", security_policy_digest="other-policy")
    elif case == "missing_digest":
        denied = _chunk("public/denied.md")
    with closing(LiteStore(tmp_path / "m.db", dim=DIM)) as store:
        _put(store, [good, denied])
        result = dev_search(
            store, HashingEmbedder(dim=DIM), QUERY, k=5,
            security_policy=policy, access_context=AccessContext("reader", "default", clearance="public"),
        )
        expected = set() if case == "no_access" else {good.id}
        assert {h.chunk.id for h in result.hits} == expected, "source policy returned denied content"


@pytest.mark.parametrize("write", ["add_successor", "delete_successor"])
@pytest.mark.parametrize("dated", [False, True])
def test_retrieval_and_graph_share_one_corpus_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: str, dated: bool,
) -> None:
    """FIX003: a real second writer cannot change this request's supersession view."""
    with closing(LiteStore(tmp_path / "m.db", dim=DIM)) as store, closing(LiteStore(store.path, dim=DIM)) as writer:
        chunks = [_chunk("a.md")]
        if write == "delete_successor":
            chunks.append(_chunk("b.md", supersedes="a.md"))
        _put(store, chunks)
        # `supersession_graph`: the one read trusted search makes now, dated or not (it carries
        # the successor windows). This patched `supersession_all` / `supersession`, the reads it
        # made before the window rule; the second writer still lands between retrieval and it.
        # Re-proved red 2026-10-21 by removing the snapshot hook in `recall.trust._trusted_search`:
        # all four parameters fail on "trust mixed retrieval and graph revisions".
        method = "supersession_graph"
        original = getattr(store, method)

        def fetch() -> Any:
            if write == "delete_successor":
                writer.delete_sources(["b.md"])
            else:
                _put(writer, [_chunk("b.md", supersedes="a.md")])
            return original()

        monkeypatch.setattr(store, method, fetch)
        result = dev_search(
            store, HashingEmbedder(dim=DIM), QUERY, k=5,
            known_as_of=datetime(2100, 1, 1, tzinfo=UTC) if dated else None,
        )
        predecessor = next(h for h in result.hits if h.provenance.file == "a.md")
        expected = "superseded" if write == "delete_successor" else "ok"
        assert predecessor.verdict == expected, "trust mixed retrieval and graph revisions"
        assert {h.chunk.id for h in result.hits} == {c.id for c in chunks}
        monkeypatch.setattr(store, method, original)
        expected_graph = {} if write == "delete_successor" else {"a.md": "b.md"}
        assert store.supersession()[0] == expected_graph, "the next reader did not see the committed write"


def test_trust_iteration_uses_the_same_pinned_view(tmp_path: Path) -> None:
    """FIX003: streaming trust consumers cannot open an independent later read view."""
    with closing(LiteStore(tmp_path / "m.db", dim=DIM)) as store, closing(LiteStore(store.path, dim=DIM)) as writer:
        _put(store, [_chunk("a.md")])

        def observe(result: Any) -> Any:
            _put(writer, [_chunk("b.md")])
            assert {c.id for c in store.iter_chunks(batch_size=1)} == {"a.md"}, "trust iteration mixed corpus revisions"
            return result

        dev_search(store, HashingEmbedder(dim=DIM), QUERY, pre_trust_transform=observe)
        assert {c.id for c in store.iter_chunks(batch_size=1)} == {"a.md", "b.md"}
