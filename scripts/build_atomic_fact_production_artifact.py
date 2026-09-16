"""Build the preregistered generation-bound atomic rescue production artifact."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomic_rescue import (  # noqa: E402
    load_atomic_rescue_artifact,
    write_atomic_rescue_artifact,
)
from recall.calibration_v2 import CalibrationRepository, CalibrationStatus  # noqa: E402
from recall.embeddings import (  # noqa: E402
    embed_document_groups,
    embedding_profile_id,
    resolve_embedder,
)
from recall.generation_store import GenerationStore  # noqa: E402
from scripts.run_atomic_fact_context4_pilot import (  # noqa: E402
    EXPECTED_RESOURCE_POLICY,
    _build_atomic_views,
    _source_root,
)
from scripts.run_atomic_fact_current_generation_shadow import _normalize_rows  # noqa: E402


PROTOCOL = "2026-09-16-atomic-fact-production-shadow"
EXPECTED_GENERATION = "gen_dff506e12f494965af9f109671a99e63"
EXPECTED_CALIBRATION = "cal_6171177aeb614d288baa4e28600caca1"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0"
EXPECTED_PROFILE = "voyage-context-4-v1"
EXPECTED_DIMENSION = 1024
EXPECTED_CHUNKS = 11_385
EXPECTED_VIEWS = 6_322
EXPECTED_ROOTS = frozenset(
    {
        "sentiment-agent",
        "recall",
        "ai-boost-av-safety",
        "ai-boost-cad",
        "steel",
        "cca-demos",
        "agent-memory-bench",
    }
)


def _validate_runtime() -> None:
    if os.environ.get("RECALL_ATOMIC_SHADOW_HOST") != "vps2":
        raise RuntimeError("atomic production artifact embeddings are allowed only on VPS2")
    if os.environ.get("RECALL_ATOMIC_SHADOW_LOCK_HELD") != "1":
        raise RuntimeError("atomic production artifact requires the shared embed.lock flock")
    if os.environ.get("RECALL_ATOMIC_SHADOW_RESOURCE_POLICY") != EXPECTED_RESOURCE_POLICY:
        raise RuntimeError("atomic production artifact requires the frozen resource policy")
    if os.environ.get("RECALL_EMBED_THREADS") != "4":
        raise RuntimeError("atomic production artifact requires RECALL_EMBED_THREADS=4")


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(path, 0o600)


def _parents(store: GenerationStore) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    texts: dict[tuple[str, int], str] = {}
    chunk_ids: dict[tuple[str, int], str] = {}
    for chunk in store.iter_chunks():
        source = chunk.metadata.get("file")
        ordinal = chunk.metadata.get("ord")
        if not isinstance(source, str) or not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise RuntimeError("pinned generation chunk lacks file or integer ord metadata")
        identity = source, ordinal
        if identity in texts:
            raise RuntimeError(f"duplicate pinned parent identity {identity!r}")
        texts[identity] = chunk.text
        chunk_ids[identity] = chunk.id
    if not texts:
        raise RuntimeError("pinned generation contains no chunks")
    return texts, chunk_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--expected-generation", default=EXPECTED_GENERATION)
    parser.add_argument("--expected-calibration", default=EXPECTED_CALIBRATION)
    parser.add_argument("--expected-pipeline", default=EXPECTED_PIPELINE)
    parser.add_argument("--expected-corpus", default=EXPECTED_CORPUS)
    parser.add_argument("--expected-chunks", type=int, default=EXPECTED_CHUNKS)
    parser.add_argument("--expected-views", type=int)
    args = parser.parse_args()
    if not args.dsn:
        raise RuntimeError("RECALL_DSN or --dsn is required")
    _validate_runtime()
    roots = dict(args.source_root)
    if set(roots) != EXPECTED_ROOTS or len(roots) != len(args.source_root):
        raise RuntimeError("source roots must contain each frozen production root exactly once")

    total_started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    if embedder.dim != EXPECTED_DIMENSION or embedding_profile_id(embedder) != EXPECTED_PROFILE:
        raise RuntimeError("runtime embedder is not the frozen Context 4 profile")
    repository = CalibrationRepository(args.dsn, args.tenant, actor="atomic-production-shadow")
    resolution = repository.resolve(args.expected_generation)
    calibration = resolution.artifact
    if (
        resolution.status is not CalibrationStatus.CERTIFIED
        or calibration is None
        or calibration.calibration_id != args.expected_calibration
    ):
        raise RuntimeError("frozen generation calibration is not certified and published")
    objects = repository.manifest_objects_for(args.expected_generation)
    timings: dict[str, float] = {}

    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as raw_store:
        store = cast(GenerationStore, raw_store)
        store.set_fixed_generation(args.expected_generation)
        binding = store.generation_binding()
        expected_binding = {
            "tenant_id": args.tenant,
            "generation_id": args.expected_generation,
            "pipeline_fingerprint": args.expected_pipeline,
            "corpus_fingerprint": args.expected_corpus,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise RuntimeError(f"frozen serving lineage changed: {binding}")

        started = time.perf_counter()
        parent_texts, chunk_ids = _parents(store)
        if len(parent_texts) != args.expected_chunks:
            raise RuntimeError(f"frozen generation chunk count changed: {len(parent_texts)}")
        groups, corpus_metrics = _build_atomic_views(objects, roots, parent_texts)
        views = [view for group in groups for view in group]
        if args.expected_views is not None and len(views) != args.expected_views:
            raise RuntimeError(f"frozen atomic view count changed: {len(views)}")
        timings["view_build_ms"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        vectors_by_group = embed_document_groups(
            embedder, [[view.rendered for view in group] for group in groups]
        )
        matrix = _normalize_rows(
            np.asarray([vector for group in vectors_by_group for vector in group], dtype=np.float32)
        )
        timings["document_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        if matrix.shape != (len(views), EXPECTED_DIMENSION):
            raise RuntimeError("frozen Context 4 atomic matrix shape mismatch")

    metadata: list[dict[str, object]] = []
    for view in views:
        identity = view.source, view.parent_ordinal
        chunk_id = chunk_ids.get(identity)
        if chunk_id is None:
            raise RuntimeError(f"atomic view has no generation parent {identity!r}")
        metadata.append(
            {
                "chunk_id": chunk_id,
                "source": view.source,
                "parent_ordinal": view.parent_ordinal,
                "view_ordinal": view.view_ordinal,
            }
        )

    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    started = time.perf_counter()
    manifest_path = write_atomic_rescue_artifact(
        args.artifact_directory,
        matrix=matrix,
        views=metadata,
        generation_id=args.expected_generation,
        calibration_id=args.expected_calibration,
        pipeline_fingerprint=args.expected_pipeline,
        corpus_fingerprint=args.expected_corpus,
        embedding_profile=EXPECTED_PROFILE,
        ordinary_chunk_count=args.expected_chunks,
        source_commit=source_commit,
    )
    artifact = load_atomic_rescue_artifact(manifest_path)
    timings["artifact_write_and_validation_ms"] = (time.perf_counter() - started) * 1000.0
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0
    artifact_bytes = sum(path.stat().st_size for path in manifest_path.parent.iterdir())
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "generation_id": args.expected_generation,
        "calibration_id": args.expected_calibration,
        "pipeline_fingerprint": args.expected_pipeline,
        "corpus_fingerprint": args.expected_corpus,
        "embedding_profile": EXPECTED_PROFILE,
        "dimension": EXPECTED_DIMENSION,
        "ordinary_chunks": args.expected_chunks,
        "view_count": artifact.view_count,
        "parent_count": artifact.parent_count,
        "matrix_sha256": artifact.matrix_sha256,
        "metadata_sha256": artifact.metadata_sha256,
        "artifact_bytes": artifact_bytes,
        "timings_ms": {key: round(value, 3) for key, value in timings.items()},
        "load_ms": round(artifact.load_ms, 3),
        "resident_memory_delta_bytes": artifact.resident_memory_delta_bytes,
        "corpus": corpus_metrics,
        "checks": {
            "artifact_lte_64_mib": artifact_bytes <= 64 * 1024 * 1024,
            "load_lte_2000_ms": artifact.load_ms <= 2_000.0,
            "resident_delta_lte_128_mib": (
                artifact.resident_memory_delta_bytes <= 128 * 1024 * 1024
            ),
            "view_count": (
                args.expected_views is None or artifact.view_count == args.expected_views
            ),
            "ordinary_chunk_count": artifact.ordinary_chunk_count == args.expected_chunks,
        },
    }
    result["decision"] = (
        "READY_FOR_ATOMIC_PRODUCTION_SHADOW"
        if all(result["checks"].values())
        else "STOP_ATOMIC_PRODUCTION_SHADOW"
    )
    _json(args.public_output, result)
    print(json.dumps({"decision": result["decision"], "artifact_bytes": artifact_bytes}))


if __name__ == "__main__":
    main()
