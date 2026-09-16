"""Build and benchmark an isolated atomic fact shadow for the current memory generation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - the measurement worker runs on Linux
    resource = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.calibration_v2 import CalibrationRepository, CalibrationStatus  # noqa: E402
from recall.embeddings import (  # noqa: E402
    embed_document_groups,
    embed_query,
    embedding_profile_id,
    resolve_embedder,
)
from recall.generation_store import GenerationStore  # noqa: E402
from scripts.run_atomic_fact_context4_pilot import (  # noqa: E402
    AtomicView,
    Candidate,
    EXPECTED_RESOURCE_POLICY,
    _build_atomic_views,
    _dense_candidates,
    _parent_chunks,
    _sha256,
    _source_root,
)


EXPECTED_QUERY_SHA256 = "06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f"
EXPECTED_GENERATION = "gen_83393e5c58524eecb9e390cb367e4d61"
EXPECTED_CALIBRATION = "cal_86a67f38755c45f38456d6b2445767cc"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "5357c7fb736dfd8add6182e41f97932a8bd6c103daae0f9bf8d984f113f87c42"
EXPECTED_PROFILE = "voyage-context-4-v1"
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
CUTOFFS = (1, 3, 5, 10, 20)
REPETITIONS = 5
WARMUP_QUERIES = 5


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(path, 0o600)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise RuntimeError("atomic vector matrix must be nonempty and two dimensional")
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise RuntimeError("atomic vector matrix contains a nonfinite or zero norm")
    return np.ascontiguousarray(values / norms[:, np.newaxis], dtype=np.float32)


def rank_atomic_matrix(
    views: Sequence[AtomicView],
    normalized_vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    cutoff: int = 20,
) -> list[Candidate]:
    """Rank normalized atomic vectors with the pilot's deterministic parent ordering."""

    matrix = np.asarray(normalized_vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[0] != len(views):
        raise RuntimeError("atomic view and matrix shapes differ")
    if matrix.shape[1] != query.shape[0]:
        raise RuntimeError("atomic matrix and query dimensions differ")
    query_norm = float(np.linalg.norm(query))
    if not math.isfinite(query_norm) or query_norm == 0.0:
        raise RuntimeError("query vector has a nonfinite or zero norm")
    scores = matrix @ (query / query_norm)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("atomic ranking produced a nonfinite score")
    order = sorted(
        range(len(views)),
        key=lambda index: (
            -float(scores[index]),
            views[index].source,
            views[index].parent_ordinal,
            views[index].view_ordinal,
        ),
    )
    output: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for index in order:
        view = views[index]
        if view.parent_identity in seen:
            continue
        seen.add(view.parent_identity)
        output.append(
            Candidate(
                view.source,
                view.parent_ordinal,
                view.parent_text,
                float(scores[index]),
            )
        )
        if len(output) == cutoff:
            break
    if len(output) != cutoff:
        raise RuntimeError(f"atomic retrieval returned {len(output)} unique parents")
    return output


def _validate_queries(path: Path) -> list[dict[str, Any]]:
    digest = _sha256(path)
    if digest != EXPECTED_QUERY_SHA256:
        raise RuntimeError(f"frozen query file hash changed: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 50:
        raise RuntimeError("frozen query file must contain exactly 50 rows")
    ids: list[str] = []
    for row in payload:
        if not isinstance(row, dict):
            raise RuntimeError("query row is not an object")
        query_id = str(row.get("id", ""))
        query = str(row.get("query", ""))
        relevant = row.get("relevant_files")
        if not query_id or not query or not isinstance(relevant, list) or not relevant:
            raise RuntimeError("query row lacks an id, query, or relevant files")
        if any(not isinstance(value, str) or not value for value in relevant):
            raise RuntimeError("query row has an invalid relevant file")
        ids.append(query_id)
    if len(ids) != len(set(ids)):
        raise RuntimeError("query ids are duplicated")
    return payload


def _validate_runtime() -> None:
    if os.environ.get("RECALL_ATOMIC_SHADOW_HOST") != "vps2":
        raise RuntimeError("atomic fact embeddings are allowed only on VPS2")
    if os.environ.get("RECALL_ATOMIC_SHADOW_LOCK_HELD") != "1":
        raise RuntimeError("atomic fact embeddings require the shared embed.lock flock")
    if os.environ.get("RECALL_ATOMIC_SHADOW_RESOURCE_POLICY") != EXPECTED_RESOURCE_POLICY:
        raise RuntimeError("atomic fact embeddings require the frozen resource policy")
    if os.environ.get("RECALL_EMBED_THREADS") != "4":
        raise RuntimeError("atomic fact embeddings require RECALL_EMBED_THREADS=4")


def _candidate_payload(candidates: Sequence[Candidate]) -> list[dict[str, object]]:
    return [
        {
            "source": item.source,
            "ordinal": item.ordinal,
            "score": item.score,
        }
        for item in candidates
    ]


def _views_payload(views: Sequence[AtomicView]) -> list[dict[str, object]]:
    return [
        {
            "source": view.source,
            "parent_ordinal": view.parent_ordinal,
            "view_ordinal": view.view_ordinal,
        }
        for view in views
    ]


def _views_from_payload(rows: Sequence[Mapping[str, object]]) -> list[AtomicView]:
    return [
        AtomicView(
            source=str(row["source"]),
            parent_ordinal=int(row["parent_ordinal"]),
            view_ordinal=int(row["view_ordinal"]),
            rendered="",
            parent_text="",
        )
        for row in rows
    ]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise RuntimeError("cannot calculate a percentile of an empty sequence")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _rank_worker(args: argparse.Namespace) -> None:
    if resource is None:
        raise RuntimeError("the private rank worker requires a POSIX resource module")
    baseline_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    views = _views_from_payload(metadata["views"])
    with np.load(args.vectors, allow_pickle=False) as artifact:
        vectors = np.ascontiguousarray(artifact["atomic_vectors"], dtype=np.float32)
        query_vectors = np.ascontiguousarray(artifact["query_vectors"], dtype=np.float32)
    if vectors.shape != (len(views), 1024) or query_vectors.shape != (50, 1024):
        raise RuntimeError("private vector artifact has the wrong shape")
    _ = float(vectors.sum()) + float(query_vectors.sum())
    loaded_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for query_vector in query_vectors[:WARMUP_QUERIES]:
        rank_atomic_matrix(views, vectors, query_vector)

    timings_ms: list[float] = []
    rows: list[dict[str, object]] = []
    determinism_failures = 0
    for query_id, query_vector in zip(metadata["query_ids"], query_vectors, strict=True):
        selected: list[Candidate] | None = None
        expected_identities: list[tuple[str, int]] | None = None
        for _repetition in range(REPETITIONS):
            started = time.perf_counter()
            current = rank_atomic_matrix(views, vectors, query_vector)
            timings_ms.append((time.perf_counter() - started) * 1000.0)
            identities = [item.identity for item in current]
            if expected_identities is None:
                expected_identities = identities
                selected = current
            elif identities != expected_identities:
                determinism_failures += 1
        assert selected is not None
        rows.append({"query_id": str(query_id), "atomic": _candidate_payload(selected)})

    output = {
        "schema_version": 1,
        "resident_rss_increase_bytes": max(0, loaded_rss_kib - baseline_rss_kib) * 1024,
        "latency_repetitions": len(timings_ms),
        "latency_ms": {
            "p50": _percentile(timings_ms, 50),
            "p95": _percentile(timings_ms, 95),
            "p99": _percentile(timings_ms, 99),
            "max": max(timings_ms),
        },
        "determinism_failures": determinism_failures,
        "rows": rows,
    }
    _json(args.worker_output, output)


def _gold_reach(rows: Sequence[Mapping[str, Any]], arm: str, cutoff: int) -> int:
    return sum(
        any(str(item["source"]) in row["gold_sources"] for item in row[arm][:cutoff])
        for row in rows
    )


def _overlap(rows: Sequence[Mapping[str, Any]], cutoff: int) -> dict[str, float]:
    counts = []
    for row in rows:
        dense = {(item["source"], item["ordinal"]) for item in row["dense"][:cutoff]}
        atomic = {(item["source"], item["ordinal"]) for item in row["atomic"][:cutoff]}
        counts.append(len(dense & atomic))
    return {
        "mean_count": sum(counts) / len(counts),
        "mean_fraction": sum(counts) / (len(counts) * cutoff),
        "min_count": min(counts),
        "max_count": max(counts),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_has_views: set[str],
) -> dict[str, object]:
    gold = {
        arm: {str(cutoff): _gold_reach(rows, arm, cutoff) for cutoff in CUTOFFS}
        for arm in ("dense", "atomic")
    }
    rank1_changed = sum(
        (row["dense"][0]["source"], row["dense"][0]["ordinal"])
        != (row["atomic"][0]["source"], row["atomic"][0]["ordinal"])
        for row in rows
    )
    dense_gold_1 = [row["dense"][0]["source"] in row["gold_sources"] for row in rows]
    atomic_gold_1 = [row["atomic"][0]["source"] in row["gold_sources"] for row in rows]
    return {
        "rows": len(rows),
        "rank1_changed": rank1_changed,
        "overlap": {str(cutoff): _overlap(rows, cutoff) for cutoff in (1, 5, 10, 20)},
        "consumed_set_diagnostic": {
            "gold_reach": gold,
            "rank1_gold_gains": sum(
                not dense and atomic
                for dense, atomic in zip(dense_gold_1, atomic_gold_1, strict=True)
            ),
            "rank1_gold_losses": sum(
                dense and not atomic
                for dense, atomic in zip(dense_gold_1, atomic_gold_1, strict=True)
            ),
            "all_gold_sources_zero_view": sum(
                not (set(row["gold_sources"]) & source_has_views) for row in rows
            ),
            "interpretation": "diagnostic_only_consumed_query_set",
        },
    }


def _gate_result(
    corpus: Mapping[str, int],
    *,
    artifact_bytes: int,
    worker: Mapping[str, Any],
    active_generation_unchanged: bool,
) -> dict[str, object]:
    included_sources = corpus["sources_with_views"] + corpus["zero_view_sources"]
    zero_view_fraction = corpus["zero_view_sources"] / included_sources
    checks = {
        "zero_errors": worker["determinism_failures"] == 0,
        "atomic_views_lte_7500": corpus["atomic_views"] <= 7500,
        "artifact_bytes_lte_64_mib": artifact_bytes <= 64 * 1024 * 1024,
        "rss_increase_lte_96_mib": worker["resident_rss_increase_bytes"] <= 96 * 1024 * 1024,
        "latency_p95_lte_100_ms": worker["latency_ms"]["p95"] <= 100.0,
        "latency_p99_lte_150_ms": worker["latency_ms"]["p99"] <= 150.0,
        "zero_view_fraction_lte_0_12": zero_view_fraction <= 0.12,
        "active_generation_unchanged": active_generation_unchanged,
    }
    return {
        "decision": (
            "PASS_ENGINEERING_SHADOW" if all(checks.values()) else "STOP_BEFORE_LIVE_SHADOW"
        ),
        "checks": checks,
        "zero_view_source_fraction": zero_view_fraction,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    parser.add_argument("--generation-id", default=EXPECTED_GENERATION)
    parser.add_argument("--calibration-id", default=EXPECTED_CALIBRATION)
    parser.add_argument("--pipeline-fingerprint", default=EXPECTED_PIPELINE)
    parser.add_argument("--corpus-fingerprint", default=EXPECTED_CORPUS)
    args = parser.parse_args()
    if not args.dsn:
        raise RuntimeError("RECALL_DSN or --dsn is required")
    _validate_runtime()
    queries = _validate_queries(args.queries)
    roots = dict(args.source_root)
    if set(roots) != EXPECTED_ROOTS or len(roots) != len(args.source_root):
        raise RuntimeError("source roots must contain each frozen production root exactly once")
    if any(not path.is_dir() for path in roots.values()):
        raise RuntimeError("one or more source roots do not exist")

    args.private_output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.private_output_dir, 0o700)
    vectors_path = args.private_output_dir / "atomic-fact-current-generation-vectors.npz"
    metadata_path = args.private_output_dir / "atomic-fact-current-generation-metadata.json"
    worker_path = args.private_output_dir / "atomic-fact-current-generation-worker.json"
    private_rows_path = args.private_output_dir / "atomic-fact-current-generation-rows.json"

    total_started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    if embedder.dim != 1024 or embedding_profile_id(embedder) != EXPECTED_PROFILE:
        raise RuntimeError("runtime embedder is not the frozen Context 4 profile")
    repository = CalibrationRepository(args.dsn, args.tenant, actor="atomic-fact-shadow")
    resolution = repository.resolve(args.generation_id)
    artifact = resolution.artifact
    if (
        resolution.status is not CalibrationStatus.CERTIFIED
        or artifact is None
        or artifact.calibration_id != args.calibration_id
    ):
        raise RuntimeError("frozen generation calibration is not certified and published")
    objects = repository.manifest_objects_for(args.generation_id)

    timings: dict[str, float] = {}
    dense_rows: list[dict[str, object]] = []
    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as store:
        store.set_fixed_generation(args.generation_id)
        binding = store.generation_binding()
        expected_binding = {
            "tenant_id": args.tenant,
            "generation_id": args.generation_id,
            "pipeline_fingerprint": args.pipeline_fingerprint,
            "corpus_fingerprint": args.corpus_fingerprint,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise RuntimeError(f"frozen serving lineage changed: {binding}")

        started = time.perf_counter()
        parents = _parent_chunks(store)
        groups, corpus_metrics = _build_atomic_views(objects, roots, parents)
        timings["view_build_ms"] = (time.perf_counter() - started) * 1000.0
        views = [view for group in groups for view in group]
        rendered_groups = [[view.rendered for view in group] for group in groups]

        started = time.perf_counter()
        vectors_by_group = embed_document_groups(embedder, rendered_groups)
        timings["document_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        raw_vectors = np.asarray(
            [vector for group in vectors_by_group for vector in group], dtype=np.float32
        )
        vectors = _normalize_rows(raw_vectors)
        if vectors.shape != (len(views), 1024):
            raise RuntimeError("Context 4 atomic vector shape mismatch")

        started = time.perf_counter()
        query_vectors = np.asarray(
            [embed_query(embedder, str(row["query"])) for row in queries], dtype=np.float32
        )
        timings["query_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        if query_vectors.shape != (50, 1024):
            raise RuntimeError("Context 4 query vector shape mismatch")

        started = time.perf_counter()
        for row, vector in zip(queries, query_vectors, strict=True):
            dense_rows.append(
                {
                    "query_id": str(row["id"]),
                    "gold_sources": sorted({str(value) for value in row["relevant_files"]}),
                    "dense": _candidate_payload(_dense_candidates(store, vector.tolist())),
                }
            )
        timings["dense_retrieval_ms"] = (time.perf_counter() - started) * 1000.0
        active_generation_unchanged = store.active_generation_id() == args.generation_id

    started = time.perf_counter()
    np.savez_compressed(
        vectors_path,
        atomic_vectors=vectors,
        query_vectors=query_vectors,
    )
    os.chmod(vectors_path, 0o600)
    _json(
        metadata_path,
        {
            "schema_version": 1,
            "generation_id": args.generation_id,
            "query_ids": [str(row["id"]) for row in queries],
            "views": _views_payload(views),
        },
    )
    timings["artifact_write_ms"] = (time.perf_counter() - started) * 1000.0

    del raw_vectors, vectors, query_vectors, vectors_by_group, rendered_groups, groups
    gc.collect()
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rank-worker",
            "--vectors",
            str(vectors_path),
            "--metadata",
            str(metadata_path),
            "--worker-output",
            str(worker_path),
        ],
        check=True,
    )
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    atomic_by_id = {str(row["query_id"]): row["atomic"] for row in worker["rows"]}
    measured_rows: list[dict[str, object]] = []
    for dense_row in dense_rows:
        query_id = str(dense_row["query_id"])
        measured_rows.append({**dense_row, "atomic": atomic_by_id[query_id]})

    source_has_views = {view.source for view in views}
    summary = _summarize(measured_rows, source_has_views=source_has_views)
    private_result = {
        "schema_version": 1,
        "protocol": "2026-09-16-atomic-fact-current-generation-shadow",
        "generation_id": args.generation_id,
        "summary": summary,
        "rows": measured_rows,
    }
    _json(private_rows_path, private_result)
    artifact_bytes = vectors_path.stat().st_size + metadata_path.stat().st_size
    gates = _gate_result(
        corpus_metrics,
        artifact_bytes=artifact_bytes,
        worker=worker,
        active_generation_unchanged=active_generation_unchanged,
    )
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    public_result = {
        "schema_version": 1,
        "protocol": "2026-09-16-atomic-fact-current-generation-shadow",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "query_sha256": EXPECTED_QUERY_SHA256,
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "embedding_profile": EXPECTED_PROFILE,
        "embedding_dimension": embedder.dim,
        "resource_policy": EXPECTED_RESOURCE_POLICY,
        "corpus": corpus_metrics,
        "artifact": {
            "bytes": artifact_bytes,
            "vectors_sha256": _sha256(vectors_path),
            "metadata_sha256": _sha256(metadata_path),
            "private_rows_sha256": _sha256(private_rows_path),
            "vector_count": corpus_metrics["atomic_views"],
            "resident_rss_increase_bytes": worker["resident_rss_increase_bytes"],
        },
        "timings_ms": {key: round(value, 3) for key, value in timings.items()},
        "ranking_latency_ms": {
            key: round(float(value), 3) for key, value in worker["latency_ms"].items()
        },
        "ranking_repetitions": worker["latency_repetitions"],
        "determinism_failures": worker["determinism_failures"],
        "summary": summary,
        "gates": gates,
        "quality_interpretation": "consumed_set_diagnostic_only_not_production_confirmation",
        "serving_route_changed": False,
    }
    _json(args.public_output, public_result)
    print(
        json.dumps(
            {
                "decision": gates["decision"],
                "atomic_views": corpus_metrics["atomic_views"],
                "ranking_latency_ms": public_result["ranking_latency_ms"],
                "public_output": str(args.public_output),
            }
        )
    )


def main() -> None:
    if "--rank-worker" not in sys.argv:
        _main()
        return
    parser = argparse.ArgumentParser(description="Private atomic fact rank worker")
    parser.add_argument("--rank-worker", action="store_true")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    _rank_worker(parser.parse_args())


if __name__ == "__main__":
    main()
