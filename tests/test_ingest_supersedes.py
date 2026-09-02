"""Re-ingesting a corpus must supersede it, not accumulate a second copy of it.

Measured 2026-09-01 against a real database, before the fix: the identical 25-file corpus uploaded
twice through `recall_ingest` took a tenant from 584 chunks to 1752. Every call staged into a fresh
`uuid4` directory and the manifest keys each object on its staged file's URI, so the same document
became a new object every time. Both copies were indexed and both answered, which means a memo the
user had corrected went on being returned beside its correction.

These assert the mechanism rather than the chunk count: `_carry_forward` already re-stamps an
object whose file changed and keeps ONE entry, and already drops one whose file has gone. It could
do neither, because it never saw the same URI twice. The URIs are what the manifest keys on, so the
URIs are what these pin. No database is needed for that, which is why this runs everywhere.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from recall.desktop.uploads import UploadError, discard_staging, promote_uploads, stage_uploads

from .conftest import TEST_DSN, requires_db


def _upload(name: str, body: bytes) -> dict[str, str]:
    return {"name": name, "content_b64": base64.b64encode(body).decode("ascii")}


def _staged(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_a_second_upload_of_one_corpus_lands_on_the_same_paths() -> None:
    """Two uploads of the same names produce one set of paths, so one set of manifest objects."""
    files = [_upload("notes/a.md", b"one"), _upload("b.md", b"two")]
    _j1, work1, _t1 = stage_uploads("supersede-1", files)
    root1 = promote_uploads("supersede-1", work1, "sync-memory")
    _j2, work2, _t2 = stage_uploads("supersede-1", files)
    root2 = promote_uploads("supersede-1", work2, "sync-memory")

    assert root1 == root2
    assert _staged(root2) == ["b.md", "notes/a.md"]
    assert not work1.exists() and not work2.exists(), "a spent working directory survived"


def test_an_edited_file_replaces_its_predecessor_rather_than_joining_it() -> None:
    """The corrected memo is the only one on disk, so only it can be indexed.

    This is the defect at its sharpest: before the fix both versions were staged under different
    job directories, both were pinned by the manifest, and a retracted claim kept answering beside
    its retraction.
    """
    _j1, work1, _t1 = stage_uploads("supersede-2", [_upload("memo.md", b"the old claim")])
    root = promote_uploads("supersede-2", work1, "sync-memory")
    _j2, work2, _t2 = stage_uploads("supersede-2", [_upload("memo.md", b"the corrected claim")])
    promote_uploads("supersede-2", work2, "sync-memory")

    assert _staged(root) == ["memo.md"]
    assert (root / "memo.md").read_bytes() == b"the corrected claim"


def test_a_refusal_between_staging_and_promotion_leaves_the_corpus_untouched() -> None:
    """The gate `recall_ingest` puts between decoding and committing, exercised directly.

    `recall_ingest` debits the tenant's byte quota after staging and before promoting, and its
    refusal path calls `discard_staging` on what staging returned. If that could reach the stable
    tree it would delete every file the tenant had ever uploaded — worse than the defect this
    fixes. Staging into a disposable directory is what makes it impossible rather than merely
    unlikely.
    """
    _j1, work1, _t1 = stage_uploads("supersede-3", [_upload("keep.md", b"kept")])
    root = promote_uploads("supersede-3", work1, "sync-memory")
    _j2, work2, _t2 = stage_uploads("supersede-3", [_upload("new.md", b"refused")])

    discard_staging(work2)  # exactly what the quota refusal does

    assert _staged(root) == ["keep.md"], "a refusal reached the tenant's existing corpus"
    assert not work2.exists()


def test_promotion_refuses_an_unsafe_key_before_touching_anything() -> None:
    """A key that names a directory outside the tenant's root is refused, not sanitised."""
    _j, work, _t = stage_uploads("supersede-4", [_upload("a.md", b"x")])
    with pytest.raises(UploadError):
        promote_uploads("supersede-4", work, "../escape")


def test_a_deleted_file_stays_until_it_is_forgotten() -> None:
    """Honest about what this fix does NOT do, so nobody assumes deletion is handled.

    A stable tree converges on the union of everything uploaded. A file the client stops sending
    is not removed by that, and `_carry_forward` keeps it because its bytes are still there. Only
    `recall_forget`, which unlinks the staged source, retires it. A sync client that treats absence
    as deletion needs that leg; this test exists so the gap is asserted rather than discovered.
    """
    _j1, work1, _t1 = stage_uploads(
        "supersede-5", [_upload("keep.md", b"k"), _upload("gone.md", b"g")]
    )
    root = promote_uploads("supersede-5", work1, "sync-memory")
    _j2, work2, _t2 = stage_uploads("supersede-5", [_upload("keep.md", b"k")])
    promote_uploads("supersede-5", work2, "sync-memory")

    assert _staged(root) == ["gone.md", "keep.md"], "the union is the documented behaviour"


def _active(dsn: str, tenant: str) -> tuple[int, int]:
    """Chunks and distinct sources in the tenant's ACTIVE generation.

    ⚠️ Deliberately NOT every `recall_chunks_v1` row for the tenant. Retired generations keep
    their rows until `generation gc` runs, so a second ingest always adds rows whether or not it
    duplicated anything. Measuring that instead cost one run and a wrong conclusion: it read
    32 -> 64 for a build that had in fact stayed flat, because it was counting the generation
    model rather than the defect. The served generation is the thing under test.
    """
    import psycopg

    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        row = conn.execute(
            "SELECT count(*), count(DISTINCT c.source_uri) FROM recall_chunks_v1 c "
            "JOIN recall_generations g "
            "  ON g.generation_id = c.generation_id AND g.tenant_id = c.tenant_id "
            "WHERE c.tenant_id = %s AND g.state = 'active'",
            (tenant,),
        ).fetchone()
    return (int(row[0]), int(row[1])) if row else (0, 0)


@requires_db
def test_a_re_ingest_does_not_duplicate_the_served_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measurement that found the defect, as a guard.

    Before the fix, ingesting one corpus twice left the ACTIVE generation holding two copies of
    every document: measured 2026-09-01 at 24 chunks / 8 sources going to 48 / 16 here, and at
    584 chunks going to 1752 on the 25-file corpus that first surfaced it. Both copies answered,
    so a corrected memo was returned beside the claim it corrected.
    """
    import uuid as _uuid

    from recall.embeddings import HashingEmbedder
    from recall.store import PgVectorStore
    from recall_mcp.service import generation_ingest

    # The build leg must run under development; production demands a byte-pinned embedder
    # identity that a local file cannot satisfy. See docs/OPERATING_MODES.md.
    monkeypatch.delenv("RECALL_ENV", raising=False)
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", os.environ["RECALL_INDEX_ROOT"])
    monkeypatch.setenv("RECALL_INDEX_BATCH_CHUNKS", "64")

    tenant = f"reingest-{_uuid.uuid4().hex[:8]}"
    # 64 dimensions, matching what conftest provisions, and deterministic. Not fastembed: the
    # `test` and `floor` CI jobs install without the optional extras on purpose, and a real model
    # would make this skip exactly where it is most worth running.
    embedder = HashingEmbedder(dim=64)
    store = PgVectorStore(TEST_DSN, dim=embedder.dim, tenant=tenant)
    store.check_schema()
    try:
        body = (
            "# Retention policy\n\nThe window is thirty days for operational logs.\n"
            "Archived exports are kept for one year and then deleted.\n"
        ) * 12
        files = [
            _upload(f"doc{i}.md", (body + f"\n## Section {i}\n\nTopic {i}.\n" * 6).encode())
            for i in range(4)
        ]

        seen = []
        for _ in range(2):
            _job, work, _total = stage_uploads(tenant, files)
            root = promote_uploads(tenant, work, "sync-memory")
            generation_ingest(store, embedder, str(root), "memory")
            seen.append(_active(TEST_DSN, tenant))
    finally:
        store.close()

    assert seen[0][1] == 4, f"the first ingest should hold 4 sources, got {seen[0][1]}"
    assert seen[1] == seen[0], (
        f"re-ingesting an unchanged corpus changed the served generation: "
        f"{seen[0]} -> {seen[1]} (chunks, distinct sources)"
    )
