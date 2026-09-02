"""`recall_inventory`: what the tenant holds, so a sync client can diff instead of re-uploading.

Without this a client has no way to ask what is already stored. It can only re-send everything,
which is how `recall_ingest` came to duplicate a corpus on every session (fixed by the stable
staging destination), and it can never notice that a file it deleted locally is still being served.

The hash these return must be the RAW content digest, so that a client which hashes file bytes
gets a match. Getting that wrong is the sharpest hazard here and it is silent: an inventory keyed
on anything derived from the embedder identity or the context policy would never match a client's
digest, so every sync would re-upload the entire corpus forever while appearing to work.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from recall_mcp.service import memory_inventory
from recall_mcp.tool_surface import ALL_TOOL_NAMES, TOOL_PRESETS

from .conftest import TEST_DSN, requires_db


def _indexed_store(tmp_path, tenant: str, files: dict[str, str]):
    """Write `files` under `tmp_path` and index them into a fresh tenant. Returns the store.

    `index_path` is the only entry point; there is no text-level indexer. Markdown is hashed as
    its decoded, newline-normalised text, so for plain ASCII bodies written with "\n" the stored
    `content_hash` equals sha256 of the file's bytes — which is exactly what a sync client can
    compute for itself, and what these tests assert.
    """
    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    from recall.store import PgVectorStore

    root = tmp_path / tenant
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8", newline="\n")
    embedder = HashingEmbedder(dim=64)
    store = PgVectorStore(TEST_DSN, dim=embedder.dim, tenant=tenant)
    store.check_schema()
    Indexer(store, embedder).index_path(root, glob="*.md")
    return store


def test_the_tool_is_registered_but_kept_out_of_the_agent_presets() -> None:
    """It belongs to a sync client, not to a model.

    `tests/test_tool_surface.py` pins the registry against `ALL_TOOL_NAMES`, so omitting it there
    would fail the moment it is registered. Keeping it OUT of `read` and `search` is the
    deliberate half: every tool in a preset costs input tokens on every turn of every session, and
    an agent has no use for a file listing.
    """
    assert "recall_inventory" in ALL_TOOL_NAMES
    for preset in ("read", "search"):
        assert "recall_inventory" not in TOOL_PRESETS[preset], (
            f"recall_inventory is in the {preset!r} preset; it is a sync-client tool and would "
            f"cost every session input tokens for a listing no agent reads"
        )


@requires_db
def test_an_inventory_reports_the_raw_digest_a_client_can_match(tmp_path) -> None:
    """The digest is of the file's bytes, so a client that hashes its own file gets a match.

    This is the assertion that would catch the silent failure: an inventory keyed on
    `index_fingerprint` (which `PgVectorStore.source_content_hashes` coalesces to first, and which
    is derived from the embedder identity and the context policy) looks perfectly well-formed and
    matches nothing a client can compute.
    """
    tenant = f"inv-{uuid.uuid4().hex[:8]}"
    body = "# Retention\n\nThe window is thirty days.\n"
    store = _indexed_store(tmp_path, tenant, {"memo.md": body})
    try:
        result = memory_inventory(store)
    finally:
        store.close()

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.source.endswith("memo.md")
    assert entry.sha256, "no digest was reported, so a client cannot diff against this"
    assert entry.sha256 == hashlib.sha256(body.encode("utf-8")).hexdigest(), (
        "the reported digest is not the raw content hash, so a client hashing its own file "
        "would never match it and would re-upload the whole corpus on every sync"
    )


@requires_db
def test_an_empty_tenant_reports_an_empty_inventory_rather_than_failing() -> None:
    """A client's first sync asks before it has ever written. That is not an error."""
    from recall.embeddings import HashingEmbedder
    from recall.store import PgVectorStore

    tenant = f"inv-empty-{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(TEST_DSN, dim=HashingEmbedder(dim=64).dim, tenant=tenant)
    store.check_schema()
    try:
        result = memory_inventory(store)
    finally:
        store.close()

    assert result.entries == []
    assert result.truncated is False


@requires_db
def test_the_listing_is_bounded_and_says_when_it_truncated(tmp_path) -> None:
    """A silently truncated listing makes a client delete everything past the cut.

    The client's diff treats "absent from the inventory" as "the server does not have it". If a
    cap could elide entries without saying so, the next sync would re-upload the tail of the
    corpus, and a client that also handled deletion would forget it. Truncation must therefore be
    reported, not merely applied.
    """
    tenant = f"inv-cap-{uuid.uuid4().hex[:8]}"
    store = _indexed_store(
        tmp_path, tenant, {f"memo{i}.md": f"Body number {i}.\n" for i in range(5)}
    )
    try:
        capped = memory_inventory(store, limit=2)
        whole = memory_inventory(store, limit=100)
    finally:
        store.close()

    assert len(capped.entries) == 2
    assert capped.truncated is True
    assert len(whole.entries) == 5
    assert whole.truncated is False


@requires_db
def test_the_listing_is_ordered_so_two_calls_agree(tmp_path) -> None:
    """An unordered LIMIT would return a different subset each call and desynchronise a client."""
    tenant = f"inv-order-{uuid.uuid4().hex[:8]}"
    store = _indexed_store(tmp_path, tenant, {f"m{i}.md": f"Body {i}.\n" for i in range(6)})
    try:
        first = memory_inventory(store, limit=3)
        second = memory_inventory(store, limit=3)
    finally:
        store.close()

    assert [e.source for e in first.entries] == [e.source for e in second.entries]
    assert [e.source for e in first.entries] == sorted(e.source for e in first.entries)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_refused(limit: int) -> None:
    """Refused rather than clamped: a caller asking for nothing has made a mistake, and returning
    an empty inventory would read to a sync client as "the server holds nothing"."""
    from recall.store import PgVectorStore

    store = PgVectorStore.__new__(PgVectorStore)  # no connection needed; the guard is first
    with pytest.raises(ValueError):
        memory_inventory(store, limit=limit)
