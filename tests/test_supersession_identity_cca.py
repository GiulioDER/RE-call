"""Public-store regressions for PR987 FIX-001, retained memo identity across roots."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore
from recall.types import Chunk

DIM = 8


@pytest.fixture(params=["lite", "postgres"])
def identity_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    """Use the existing isolated PostgreSQL factory or a disposable Lite file."""
    if request.param == "lite":
        with LiteStore(tmp_path / "identity.db", dim=DIM) as store:
            yield store
    else:
        factory = request.getfixturevalue("make_store")
        yield factory(DIM)


def _memo(
    chunk_id: str, source: str, file: str, *, supersedes: str | None = None,
) -> Chunk:
    metadata = {"file": file}
    if supersedes is not None:
        metadata["supersedes"] = supersedes
    return Chunk(chunk_id, source, f"Synthetic memo {chunk_id}.", metadata)


def _put(store: Any, *chunks: Chunk) -> None:
    store.upsert(list(chunks), [[1.0] + [0.0] * (DIM - 1) for _ in chunks])


def test_unrelated_root_preserves_retirement_until_the_corpus_really_moves(
    identity_store: Any,
) -> None:
    """FIX-001: an unrelated b.md cannot revive a.md while its original root is populated.

    Node: tests/test_supersession_identity_cca.py::test_unrelated_root_preserves_retirement_until_the_corpus_really_moves
    Parameters: lite and postgres.
    Original HEAD: ab1a48ba427599bd174e9d99b5280182f37b2aea.
    Target: the public store's supersession_graph, through delete_sources and upsert.
    Predicted RED: "an unrelated root silenced the forgotten memo's retirement" because
    _resolve_rows drops the /one/b.md record on seeing /two/b.md's matching file label.
    All parameter cases failed at the intended original-head assertion and passed after repair.

    The final assertions preserve the approved relocation exception: once the original
    root is empty and the corpus lives under a non-upload root, its returned b.md decides.
    """
    store = identity_store
    old_a, old_b = "/one/a.md", "/one/b.md"
    _put(
        store,
        _memo("original-a", old_a, "a.md"),
        _memo("original-b", old_b, "b.md", supersedes="a.md"),
    )
    assert store.supersession_graph().successor("a.md") == "b.md"
    assert store.delete_sources([old_b]) == 1
    assert store.supersession_graph().successor("a.md") == "b.md"

    _put(store, _memo("unrelated-b", "/two/b.md", "b.md"))
    graph = store.supersession_graph()
    assert graph.successor(graph.key("a.md", old_a)) == graph.key("b.md", old_b), (
        "an unrelated root silenced the forgotten memo's retirement"
    )
    assert store.sources_for_identifiers([old_a]) == {old_a: [old_a]}

    # Genuine relocation: /one has no live row, and /two now holds both files.
    assert store.delete_sources([old_a]) == 1
    _put(store, _memo("relocated-a", "/two/a.md", "a.md"))
    graph = store.supersession_graph()
    assert graph.successor(graph.key("a.md", "/two/a.md")) is None, (
        "an empty former root prevented a genuine non-upload relocation"
    )


@pytest.mark.parametrize("original_root_populated", [True, False], ids=["populated", "empty"])
def test_upload_name_cannot_replace_retirement_or_steal_its_chain(
    identity_store: Any, original_root_populated: bool,
) -> None:
    """FIX-001: uploads cannot displace a retained identity or capture references to it.

    Node: tests/test_supersession_identity_cca.py::test_upload_name_cannot_replace_retirement_or_steal_its_chain
    Parameters: lite/postgres and populated/empty original root.
    Original HEAD: ab1a48ba427599bd174e9d99b5280182f37b2aea.
    Target: the public store's supersession_graph after deletion and a same-named upload.
    Predicted RED: "an upload displaced the forgotten memo or stole its chain" because
    the global live_files check discards b.md's ghost, reviving a.md. Keeping the ghost
    alone is insufficient: c.md must still resolve b.md to that ghost, not to the upload.
    All parameter cases failed at the intended original-head assertion and passed after repair.
    """
    store = identity_store
    other_root = "/one" if original_root_populated else "/surviving"
    a_source, b_source, c_source = f"{other_root}/a.md", "/one/b.md", f"{other_root}/c.md"
    upload_source = f"/index/uploads/{store.tenant}/batch/b.md"
    _put(
        store,
        _memo("chain-a", a_source, "a.md"),
        _memo("chain-b", b_source, "b.md", supersedes="a.md"),
        _memo("chain-c", c_source, "c.md", supersedes="b.md"),
    )
    assert store.supersession_graph().successor("a.md") == "c.md"
    assert store.delete_sources([b_source]) == 1
    assert store.supersession_graph().successor("a.md") == "c.md"

    _put(store, _memo("uploaded-b", upload_source, "b.md"))
    graph = store.supersession_graph()
    assert graph.successor(graph.key("a.md", a_source)) == graph.key("c.md", c_source), (
        "an upload displaced the forgotten memo or stole its chain"
    )
    assert graph.successor(graph.key("b.md", b_source)) == graph.key("c.md", c_source), (
        "the upload captured a reference intended for the forgotten memo"
    )
    assert store.sources_for_identifiers([upload_source]) == {upload_source: [upload_source]}


def test_unrelated_indexed_memo_cannot_capture_a_same_root_retained_chain(identity_store: Any) -> None:
    """FIX-001 attempt2: root preference must run before discarding a same-file ghost.

    Node: test_supersession_identity_cca.py::test_unrelated_indexed_memo_cannot_capture_a_same_root_retained_chain.
    Parameters: lite and postgres. Target: ReferenceIndex._indexed_first through public graph reads.
    Predicted RED against the first repair: an unrelated non-upload b.md takes C's reference
    from B's same-root ghost, leaving A's successor at B instead of C. Original PR987 also fails.
    Both cases failed at the intended assertion against the first repair, then passed after
    preserving same-path ghosts until root selection. No assertion was relaxed.
    """
    store = identity_store
    _put(
        store,
        _memo("same-root-a", "/one/a.md", "a.md"),
        _memo("same-root-b", "/one/b.md", "b.md", supersedes="a.md"),
        _memo("same-root-c", "/one/c.md", "c.md", supersedes="b.md"),
    )
    assert store.delete_sources(["/one/b.md"]) == 1
    assert store.supersession_graph().successor("a.md") == "c.md"
    _put(store, _memo("other-root-b", "/two/b.md", "b.md"))
    graph = store.supersession_graph()
    assert graph.successor(graph.key("a.md", "/one/a.md")) == graph.key("c.md", "/one/c.md"), (
        "an unrelated root diverted a reference from the retained chain"
    )
    assert graph.successor(graph.key("b.md", "/one/b.md")) == graph.key("c.md", "/one/c.md")
    assert graph.successor(graph.key("b.md", "/two/b.md")) is None, (
        "the unrelated memo was pulled into the original chain"
    )
