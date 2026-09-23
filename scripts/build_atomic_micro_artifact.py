"""Assemble a generation-bound micro atomic artifact from the content-addressed view store.

The production memory refresh calls this after a generation receives its certified calibration
and before promotion, when ``RECALL_ATOMIC_RESCUE_MODE=active``. It is an ASSEMBLY, not a rebuild:
every chunk's micro views are looked up in ``recall.atomic_view_store`` by the embedder's profile
fingerprint and the chunk text's digest, and only chunk texts the store has never seen are
embedded. Each chunk is its own Context 4 group, so a chunk's vectors depend on nothing but its
text and survive any edit elsewhere in its memo.

Measured churn (2026-09-23, eight generations): each generation brings 8 to 249 new chunk texts
out of about 11,700, so a refresh embeds tens to low hundreds of chunks instead of all of them.

The artifact itself is unchanged in format and lineage: it binds the exact generation, certified
calibration, pipeline and corpus fingerprints, embedding profile and dimension, and the serving
path refuses it on any mismatch.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomic_view_store import AtomicViewStore, chunk_view_digest  # noqa: E402
from recall.atomizer import chunk_micro_views  # noqa: E402

EMBED_BATCH_CHUNKS = 64


def assemble(
    parents: Sequence[tuple[str, str, int, str]],
    store: AtomicViewStore,
    profile: str,
    dimension: int,
    embed_groups: Callable[[list[list[str]]], list[list[list[float]]]],
) -> tuple[Any, list[dict[str, object]], dict[str, int]]:
    """Views and vectors for every parent ``(chunk_id, source, ordinal, text)``, reusing the store.

    Returns the normalized matrix, the artifact metadata rows in parent order, and counts of
    reused and newly embedded chunks and views. Chunks whose text yields no view are skipped.
    """

    import numpy as np

    planned: list[tuple[tuple[str, str, int, str], list[str], str]] = []
    missing: list[int] = []
    for parent in parents:
        views = chunk_micro_views(parent[3])
        if not views:
            continue
        digest = chunk_view_digest(parent[3])
        if store.get(profile, digest, view_count=len(views), dimension=dimension) is None:
            missing.append(len(planned))
        planned.append((parent, views, digest))

    embedded_views = 0
    seen_digests: set[str] = set()
    for offset in range(0, len(missing), EMBED_BATCH_CHUNKS):
        batch = [
            planned[index]
            for index in missing[offset : offset + EMBED_BATCH_CHUNKS]
            if planned[index][2] not in seen_digests
        ]
        if not batch:
            continue
        vectors = embed_groups([views for _, views, _ in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("embedder returned the wrong number of view groups")
        for (_, views, digest), group in zip(batch, vectors, strict=True):
            if len(group) != len(views):
                raise RuntimeError("embedder returned the wrong number of views for a chunk")
            store.put(profile, digest, group)
            seen_digests.add(digest)
            embedded_views += len(views)
        store.commit()

    rows: list[Any] = []
    metadata: list[dict[str, object]] = []
    view_ordinal: dict[str, int] = {}
    for (chunk_id, source, ordinal, _), views, digest in planned:
        matrix = store.get(profile, digest, view_count=len(views), dimension=dimension)
        if matrix is None:
            raise RuntimeError(f"view store lacks vectors for chunk {chunk_id} after embedding")
        rows.append(matrix)
        for _ in views:
            index = view_ordinal.get(source, 0)
            view_ordinal[source] = index + 1
            metadata.append(
                {"chunk_id": chunk_id, "source": source, "parent_ordinal": ordinal, "view_ordinal": index}
            )
    if not rows:
        raise RuntimeError("no parent chunk yields a micro view")
    full: Any = np.concatenate(rows, axis=0).astype(np.float32)
    norms = np.linalg.norm(full, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise RuntimeError("a view vector has zero norm")
    full /= norms
    counts = {
        "parents": len(planned),
        "views": len(metadata),
        "reused_chunks": len(planned) - len(missing),
        "embedded_chunks": len(seen_digests),
        "embedded_views": embedded_views,
    }
    return full, metadata, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--view-store", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    if not args.dsn:
        raise SystemExit("RECALL_DSN or --dsn is required")
    if os.environ.get("RECALL_ATOMIC_SHADOW_LOCK_HELD") != "1":
        raise SystemExit("the micro artifact builder requires the host's embed.lock to be held")

    from recall.atomic_rescue import load_atomic_rescue_artifact, write_atomic_rescue_artifact
    from recall.calibration_v2 import CalibrationRepository, CalibrationStatus
    from recall.embeddings import (
        embed_document_groups,
        embedding_profile,
        embedding_profile_id,
        resolve_embedder,
    )
    from recall.generation_store import GenerationStore

    started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    profile = embedding_profile(embedder).fingerprint()
    resolution = CalibrationRepository(args.dsn, args.tenant, actor="atomic-micro-builder").resolve(
        args.generation
    )
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        raise SystemExit("REFUSED: the generation has no certified published calibration")
    calibration_id = resolution.artifact.calibration_id

    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as raw_store:
        gen_store: Any = raw_store
        gen_store.set_fixed_generation(args.generation)
        binding = gen_store.generation_binding()
        parents: list[tuple[str, str, int, str]] = []
        for chunk in gen_store.iter_chunks():
            source, ordinal = chunk.metadata.get("file"), chunk.metadata.get("ord")
            if not isinstance(source, str) or not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise SystemExit("REFUSED: a chunk lacks file or integer ord metadata")
            parents.append((chunk.id, source, ordinal, chunk.text))
    parents.sort(key=lambda item: (item[1], item[2]))

    with AtomicViewStore(args.view_store) as views_store:
        matrix, metadata, counts = assemble(
            parents,
            views_store,
            profile,
            embedder.dim,
            lambda groups: embed_document_groups(embedder, groups),
        )
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = write_atomic_rescue_artifact(
        args.artifact_directory,
        matrix=matrix,
        views=metadata,
        generation_id=args.generation,
        calibration_id=calibration_id,
        pipeline_fingerprint=str(binding["pipeline_fingerprint"]),
        corpus_fingerprint=str(binding["corpus_fingerprint"]),
        embedding_profile=embedding_profile_id(embedder),
        embedding_fingerprint=profile,
        ordinary_chunk_count=len(parents),
        source_commit=source_commit,
    )
    artifact = load_atomic_rescue_artifact(manifest)
    artifact.assert_lineage(
        generation_id=args.generation,
        calibration_id=calibration_id,
        pipeline_fingerprint=str(binding["pipeline_fingerprint"]),
        corpus_fingerprint=str(binding["corpus_fingerprint"]),
        embedder=embedder,
    )
    report = {
        "schema_version": 1,
        "atomizer": "micro-24-12-per-chunk",
        "generation_id": args.generation,
        "calibration_id": calibration_id,
        "manifest": str(manifest),
        "chunks": len(parents),
        **counts,
        "matrix_mb": round(matrix.nbytes / 1e6, 1),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "built_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
    }
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
