"""Generation reuse must retain the current manifest's derived source identity."""
# Imported pytest fixtures are intentionally shadowed by their injected values.
# ruff: noqa: F811
from __future__ import annotations

import hashlib

import pytest

from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import S3Allowlist, S3ObjectReader
from recall.scope import Scope
from tests.conftest import requires_db
from tests.test_generations import _Embedder, _S3, _pipeline, manager  # noqa: F401


def _build(manager, names: tuple[str, ...], version: str):
    data = b"# Memo\n\nFirst paragraph.\n\nSecond paragraph.\n"
    manifest = IndexManifestV1(
        manager.tenant_id,
        version,
        tuple(
            ManifestObjectV1(
                f"s3://approved/corpora/{manager.tenant_id}/{name}",
                "object-v1",
                "text/markdown",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            for name in names
        ),
    )
    objects = {
        ("approved", f"corpora/{manager.tenant_id}/{name}", "object-v1"): data
        for name in names
    }
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    generation = manager.create(manifest, _pipeline("model-a"))
    stats = manager.build(
        generation.generation_id,
        reader,
        _Embedder(1),
        lambda text: text.split("\n\n"),
    )
    manager.validate(generation.generation_id)
    return generation.generation_id, stats


@requires_db
@pytest.mark.parametrize("expanding", [True, False], ids=["expand", "contract"])
def test_reuse_tracks_the_current_manifest_root(manager, expanding: bool) -> None:
    """BUG-001: source paths and folder membership follow manifest root changes.

    Exact node: tests/test_pr545_reuse_paths.py::test_reuse_tracks_the_current_manifest_root.
    Baseline 76a0b36ea, GenerationManager._reuse_source: unchanged bytes reuse stale
    metadata.file, failing the current-path assertion after expansion or contraction.
    The unchanged-root builds in this same test ensure eligible reuse remains enabled.
    Parent records the unchanged test's remote red/green execution receipt.
    """
    narrow = ("notes/a.md",)
    wide = ("notes/a.md", "other/b.md")
    initial, changed = (narrow, wide) if expanding else (wide, narrow)
    _build(manager, initial, "initial")
    _same, same_stats = _build(manager, initial, "same-root")
    assert same_stats.reused_objects == len(initial), "unchanged paths must remain reusable"

    generation_id, _stats = _build(manager, changed, "changed-root")
    uri = f"s3://approved/corpora/{manager.tenant_id}/notes/a.md"
    expected_file = "notes/a.md" if expanding else "a.md"
    with manager._connect() as conn:
        files = {
            row[0]
            for row in conn.execute(
                "SELECT metadata->>'file' FROM recall_chunks_v1 "
                "WHERE tenant_id=%s AND generation_id=%s AND source_uri=%s",
                (manager.tenant_id, generation_id, uri),
            ).fetchall()
        }
        predicate, params = Scope(folder="notes").predicate("c", source_column="source_uri")
        scoped_sources = {
            row[0]
            for row in conn.execute(
                "SELECT c.source_uri FROM recall_chunks_v1 c "
                "WHERE c.tenant_id=%(tenant)s AND c.generation_id=%(generation)s " + predicate,
                {"tenant": manager.tenant_id, "generation": generation_id, **params},
            ).fetchall()
        }
    assert files == {expected_file}, "reuse preserved a previous manifest's root-relative file"
    assert (uri in scoped_sources) is expanding, "folder filtering must use the current path"
    _same_again, same_again_stats = _build(manager, changed, "same-new-root")
    assert same_again_stats.reused_objects == len(changed), "rederived paths must reuse later"


@requires_db
def test_reuse_rejects_a_mixed_prior_file_identity(manager) -> None:
    """BUG-001: every copied chunk must agree with the current derived file identity.

    Exact node: tests/test_pr545_reuse_paths.py::test_reuse_rejects_a_mixed_prior_file_identity.
    Baseline 76a0b36ea, GenerationManager._reuse_source: a prior source with one
    stale chunk is copied wholesale, failing the no-reuse assertion. A check of only
    the candidate chunk would permit the same defect. Parent records remote red/green.
    """
    generation_id, _stats = _build(manager, ("notes/a.md",), "initial")
    with manager._connect() as conn:
        conn.execute(
            "UPDATE recall_chunks_v1 SET metadata = jsonb_set(metadata, '{file}', "
            "to_jsonb('old/a.md'::text)) WHERE tenant_id=%s AND generation_id=%s "
            "AND chunk_ordinal=1",
            (manager.tenant_id, generation_id),
        )
    rebuilt, stats = _build(manager, ("notes/a.md",), "rebuild")
    assert stats.reused_objects == 0, "a mixed prior source must be rederived as a whole"
    with manager._connect() as conn:
        files = {
            row[0]
            for row in conn.execute(
                "SELECT metadata->>'file' FROM recall_chunks_v1 "
                "WHERE tenant_id=%s AND generation_id=%s",
                (manager.tenant_id, rebuilt),
            ).fetchall()
        }
    assert files == {"a.md"}, "the stale chunk identity survived rederivation"
