"""Benchmark an exact masked maximum for the atomic rescue candidate."""

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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

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
from scripts.run_atomic_fact_current_generation_shadow import (  # noqa: E402
    _normalize_rows,
    _percentile,
    _views_from_payload,
    _views_payload,
)


PROTOCOL = "2026-09-16-atomic-fact-exact-rescue-shortlist"
EXPECTED_POOL_SHA256 = "97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18"
EXPECTED_GENERATION = "gen_e5c95bffed8c41bb9c05680325fb8d60"
EXPECTED_CALIBRATION = "cal_aa2d53e5d051418c8d7b28534b862afc"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "0587af4766daf313067eb07a72f7489765ff4b0192a96bf9c0d1142b24d5c0e7"
EXPECTED_PROFILE = "voyage-context-4-v1"
EXPECTED_ROWS = 31
EXPECTED_CHUNKS = 11_376
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
REPETITIONS = 20
WARMUP_QUERIES = 5


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(path, 0o600)


def _normalized_query(query_vector: np.ndarray, expected_dimension: int) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != expected_dimension:
        raise RuntimeError("atomic matrix and query dimensions differ")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError("query vector has a nonfinite or zero norm")
    normalized: np.ndarray = np.ascontiguousarray(query / norm, dtype=np.float32)
    return normalized


def build_parent_codes(
    views: Sequence[AtomicView],
) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    """Encode every view's parent once so per-query masking stays vectorized."""

    codes: np.ndarray = np.empty(len(views), dtype=np.int32)
    code_by_identity: dict[tuple[str, int], int] = {}
    for index, view in enumerate(views):
        code = code_by_identity.setdefault(view.parent_identity, len(code_by_identity))
        codes[index] = code
    return codes, code_by_identity


def select_full_sort_reference(
    views: Sequence[AtomicView],
    normalized_vectors: np.ndarray,
    query_vector: np.ndarray,
    excluded_parents: Sequence[tuple[str, int]],
) -> Candidate:
    """Select the first nonexcluded parent from the existing full deterministic sort."""

    matrix = np.asarray(normalized_vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(views):
        raise RuntimeError("atomic view and matrix shapes differ")
    query = _normalized_query(query_vector, matrix.shape[1])
    scores = matrix @ query
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("atomic ranking produced a nonfinite score")
    excluded = set(excluded_parents)
    order = sorted(
        range(len(views)),
        key=lambda index: (
            -float(scores[index]),
            views[index].source,
            views[index].parent_ordinal,
            views[index].view_ordinal,
        ),
    )
    for index in order:
        view = views[index]
        if view.parent_identity in excluded:
            continue
        return Candidate(
            view.source,
            view.parent_ordinal,
            view.parent_text,
            float(scores[index]),
        )
    raise RuntimeError("atomic ranking has no parent outside dense top five")


def select_exact_masked_max(
    views: Sequence[AtomicView],
    normalized_vectors: np.ndarray,
    query_vector: np.ndarray,
    excluded_parents: Sequence[tuple[str, int]],
    parent_codes: np.ndarray,
    code_by_identity: Mapping[tuple[str, int], int],
) -> Candidate:
    """Select the exact best nonexcluded parent without sorting all atomic views."""

    matrix = np.asarray(normalized_vectors, dtype=np.float32)
    codes = np.asarray(parent_codes, dtype=np.int32)
    if matrix.ndim != 2 or matrix.shape[0] != len(views) or codes.shape != (len(views),):
        raise RuntimeError("atomic view, matrix, and parent code shapes differ")
    query = _normalized_query(query_vector, matrix.shape[1])
    scores = matrix @ query
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("atomic ranking produced a nonfinite score")

    excluded_mask: np.ndarray = np.zeros(len(views), dtype=np.bool_)
    for identity in excluded_parents:
        code = code_by_identity.get(identity)
        if code is not None:
            excluded_mask |= codes == code
    valid_mask = ~excluded_mask
    if not np.any(valid_mask):
        raise RuntimeError("atomic ranking has no parent outside dense top five")

    best_score = np.max(scores[valid_mask])
    tied = np.flatnonzero(valid_mask & (scores == best_score))
    winner = min(
        (int(index) for index in tied),
        key=lambda index: (
            views[index].source,
            views[index].parent_ordinal,
            views[index].view_ordinal,
        ),
    )
    view = views[winner]
    return Candidate(
        view.source,
        view.parent_ordinal,
        view.parent_text,
        float(scores[winner]),
    )


def _validate_pool(path: Path) -> list[dict[str, Any]]:
    digest = _sha256(path)
    if digest != EXPECTED_POOL_SHA256:
        raise RuntimeError(f"private pool hash changed: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise RuntimeError("private pool must contain exactly 31 query rows")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("private pool contains a nonobject row")
        query_id = str(row.get("id", ""))
        query = str(row.get("query", ""))
        if not query_id or not query:
            raise RuntimeError("private pool row lacks an id or query")
        ids.append(query_id)
    if len(ids) != len(set(ids)):
        raise RuntimeError("private pool query ids are duplicated")
    return rows


def _validate_runtime() -> None:
    if os.environ.get("RECALL_ATOMIC_SHORTLIST_HOST") != "vps2":
        raise RuntimeError("atomic shortlist embeddings are allowed only on VPS2")
    if os.environ.get("RECALL_ATOMIC_SHORTLIST_LOCK_HELD") != "1":
        raise RuntimeError("atomic shortlist embeddings require the shared embed.lock flock")
    if os.environ.get("RECALL_ATOMIC_SHORTLIST_RESOURCE_POLICY") != EXPECTED_RESOURCE_POLICY:
        raise RuntimeError("atomic shortlist embeddings require the frozen resource policy")
    if os.environ.get("RECALL_EMBED_THREADS") != "4":
        raise RuntimeError("atomic shortlist embeddings require RECALL_EMBED_THREADS=4")


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("latency sample is empty")
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
        "total": sum(values),
    }


def _timed_select(callable_: Callable[[], Candidate]) -> tuple[Candidate, float]:
    started = time.perf_counter_ns()
    candidate = callable_()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return candidate, elapsed_ms


def _rank_worker(args: argparse.Namespace) -> None:
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    views = _views_from_payload(metadata["views"])
    exclusions = [
        [(str(item["source"]), int(item["ordinal"])) for item in row]
        for row in metadata["dense_top_five"]
    ]
    with np.load(args.vectors, allow_pickle=False) as artifact:
        vectors = np.ascontiguousarray(artifact["atomic_vectors"], dtype=np.float32)
        query_vectors = np.ascontiguousarray(artifact["query_vectors"], dtype=np.float32)
    if vectors.shape != (len(views), 1024) or query_vectors.shape != (EXPECTED_ROWS, 1024):
        raise RuntimeError("private vector artifact has the wrong shape")
    if len(exclusions) != EXPECTED_ROWS or any(len(row) != 5 for row in exclusions):
        raise RuntimeError("private dense exclusions have the wrong shape")
    _ = float(vectors.sum()) + float(query_vectors.sum())
    parent_codes, code_by_identity = build_parent_codes(views)

    expected: list[Candidate] = []
    for query_vector, excluded in zip(query_vectors, exclusions, strict=True):
        expected.append(select_full_sort_reference(views, vectors, query_vector, excluded))

    for query_vector, excluded in zip(
        query_vectors[:WARMUP_QUERIES], exclusions[:WARMUP_QUERIES], strict=True
    ):
        select_full_sort_reference(views, vectors, query_vector, excluded)
        select_exact_masked_max(
            views,
            vectors,
            query_vector,
            excluded,
            parent_codes,
            code_by_identity,
        )

    timings: dict[str, list[float]] = {
        "full_sort_reference": [],
        "exact_masked_max": [],
    }
    equivalence_failures = 0
    score_failures = 0
    determinism_failures = 0
    selected_rows: list[dict[str, object]] = []
    first_candidate: list[Candidate | None] = [None] * EXPECTED_ROWS
    for repetition in range(REPETITIONS):
        for query_index, (query_vector, excluded, reference) in enumerate(
            zip(query_vectors, exclusions, expected, strict=True)
        ):
            calls: tuple[tuple[str, Callable[[], Candidate]], ...] = (
                (
                    "full_sort_reference",
                    lambda: select_full_sort_reference(
                        views, vectors, query_vector, excluded
                    ),
                ),
                (
                    "exact_masked_max",
                    lambda: select_exact_masked_max(
                        views,
                        vectors,
                        query_vector,
                        excluded,
                        parent_codes,
                        code_by_identity,
                    ),
                ),
            )
            if (repetition + query_index) % 2:
                calls = tuple(reversed(calls))
            results: dict[str, Candidate] = {}
            for name, call in calls:
                result, elapsed_ms = _timed_select(call)
                timings[name].append(elapsed_ms)
                results[name] = result
            candidate = results["exact_masked_max"]
            if candidate.identity != reference.identity:
                equivalence_failures += 1
            if candidate.score != reference.score:
                score_failures += 1
            previous = first_candidate[query_index]
            if previous is None:
                first_candidate[query_index] = candidate
            elif candidate.identity != previous.identity or candidate.score != previous.score:
                determinism_failures += 1

    for query_id, reference, selected_candidate in zip(
        metadata["query_ids"], expected, first_candidate, strict=True
    ):
        assert selected_candidate is not None
        selected_rows.append(
            {
                "query_id": str(query_id),
                "reference": {
                    "source": reference.source,
                    "ordinal": reference.ordinal,
                    "score": reference.score,
                },
                "candidate": {
                    "source": selected_candidate.source,
                    "ordinal": selected_candidate.ordinal,
                    "score": selected_candidate.score,
                },
            }
        )

    reference_latency = _latency_summary(timings["full_sort_reference"])
    candidate_latency = _latency_summary(timings["exact_masked_max"])
    output = {
        "schema_version": 1,
        "sample_count_per_arm": len(timings["full_sort_reference"]),
        "latency_ms": {
            "full_sort_reference": reference_latency,
            "exact_masked_max": candidate_latency,
        },
        "reference_to_candidate_p95_ratio": (
            reference_latency["p95"] / candidate_latency["p95"]
        ),
        "equivalence_failures": equivalence_failures,
        "score_failures": score_failures,
        "determinism_failures": determinism_failures,
        "rows": selected_rows,
    }
    _json(args.worker_output, output)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    args = parser.parse_args()
    if not args.dsn:
        raise RuntimeError("RECALL_DSN or --dsn is required")
    _validate_runtime()
    queries = _validate_pool(args.pool)
    roots = dict(args.source_root)
    if set(roots) != EXPECTED_ROOTS or len(roots) != len(args.source_root):
        raise RuntimeError("source roots must contain each frozen production root exactly once")
    if any(not path.is_dir() for path in roots.values()):
        raise RuntimeError("one or more source roots do not exist")

    args.private_output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(args.private_output_dir, 0o700)
    vectors_path = args.private_output_dir / "atomic-fact-shortlist-vectors.npz"
    metadata_path = args.private_output_dir / "atomic-fact-shortlist-metadata.json"
    worker_path = args.private_output_dir / "atomic-fact-shortlist-worker.json"

    total_started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    if embedder.dim != 1024 or embedding_profile_id(embedder) != EXPECTED_PROFILE:
        raise RuntimeError("runtime embedder is not the frozen Context 4 profile")
    repository = CalibrationRepository(args.dsn, args.tenant, actor="atomic-fact-shortlist")
    resolution = repository.resolve(EXPECTED_GENERATION)
    artifact = resolution.artifact
    if (
        resolution.status is not CalibrationStatus.CERTIFIED
        or artifact is None
        or artifact.calibration_id != EXPECTED_CALIBRATION
    ):
        raise RuntimeError("frozen generation calibration is not certified and published")
    objects = repository.manifest_objects_for(EXPECTED_GENERATION)

    timings: dict[str, float] = {}
    dense_top_five: list[list[dict[str, object]]] = []
    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as store:
        store.set_fixed_generation(EXPECTED_GENERATION)
        binding = store.generation_binding()
        expected_binding = {
            "tenant_id": args.tenant,
            "generation_id": EXPECTED_GENERATION,
            "pipeline_fingerprint": EXPECTED_PIPELINE,
            "corpus_fingerprint": EXPECTED_CORPUS,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise RuntimeError(f"frozen serving lineage changed: {binding}")

        started = time.perf_counter()
        parents = _parent_chunks(store)
        if len(parents) != EXPECTED_CHUNKS:
            raise RuntimeError(f"frozen generation chunk count changed: {len(parents)}")
        groups, corpus_metrics = _build_atomic_views(objects, roots, parents)
        timings["view_build_ms"] = (time.perf_counter() - started) * 1000.0
        views = [view for group in groups for view in group]

        started = time.perf_counter()
        vectors_by_group = embed_document_groups(
            embedder, [[view.rendered for view in group] for group in groups]
        )
        vectors = _normalize_rows(
            np.asarray([vector for group in vectors_by_group for vector in group], dtype=np.float32)
        )
        timings["document_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        if vectors.shape != (len(views), 1024):
            raise RuntimeError("Context 4 atomic vector shape mismatch")

        started = time.perf_counter()
        query_vectors = np.asarray(
            [embed_query(embedder, str(row["query"])) for row in queries], dtype=np.float32
        )
        timings["query_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        if query_vectors.shape != (EXPECTED_ROWS, 1024):
            raise RuntimeError("Context 4 query vector shape mismatch")

        started = time.perf_counter()
        for vector in query_vectors:
            dense = _dense_candidates(store, vector.tolist())[:5]
            if len(dense) != 5 or len({candidate.identity for candidate in dense}) != 5:
                raise RuntimeError("dense top five contains a missing or duplicate parent")
            dense_top_five.append(
                [{"source": item.source, "ordinal": item.ordinal} for item in dense]
            )
        timings["dense_retrieval_ms"] = (time.perf_counter() - started) * 1000.0

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
            "generation_id": EXPECTED_GENERATION,
            "query_ids": [str(row["id"]) for row in queries],
            "views": _views_payload(views),
            "dense_top_five": dense_top_five,
        },
    )
    timings["artifact_write_ms"] = (time.perf_counter() - started) * 1000.0

    del vectors_by_group, vectors, query_vectors, groups
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
    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as store:
        active_generation_unchanged = store.active_generation_id() == EXPECTED_GENERATION

    candidate_latency = worker["latency_ms"]["exact_masked_max"]
    checks = {
        "all_31_identities_equivalent": worker["equivalence_failures"] == 0,
        "all_31_scores_exact": worker["score_failures"] == 0,
        "zero_determinism_failures": worker["determinism_failures"] == 0,
        "sample_count_620_per_arm": worker["sample_count_per_arm"] == 620,
        "candidate_p95_lte_25_ms": candidate_latency["p95"] <= 25.0,
        "candidate_p99_lte_50_ms": candidate_latency["p99"] <= 50.0,
        "p95_speedup_gte_4": worker["reference_to_candidate_p95_ratio"] >= 4.0,
        "active_generation_unchanged": active_generation_unchanged,
    }
    decision = (
        "PROMISING_EXACT_RESCUE_SHORTLIST"
        if all(checks.values())
        else "STOP_EXACT_RESCUE_SHORTLIST"
    )
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    public = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "pool_sha256": EXPECTED_POOL_SHA256,
        "generation_id": EXPECTED_GENERATION,
        "calibration_id": EXPECTED_CALIBRATION,
        "pipeline_fingerprint": EXPECTED_PIPELINE,
        "corpus_fingerprint": EXPECTED_CORPUS,
        "embedding_profile": EXPECTED_PROFILE,
        "embedding_dimension": embedder.dim,
        "resource_policy": EXPECTED_RESOURCE_POLICY,
        "corpus": corpus_metrics,
        "artifact": {
            "bytes": vectors_path.stat().st_size + metadata_path.stat().st_size,
            "vectors_sha256": _sha256(vectors_path),
            "metadata_sha256": _sha256(metadata_path),
            "private_worker_sha256": _sha256(worker_path),
        },
        "timings_ms": {key: round(value, 3) for key, value in timings.items()},
        "sample_count_per_arm": worker["sample_count_per_arm"],
        "latency_ms": {
            arm: {key: round(float(value), 6) for key, value in values.items()}
            for arm, values in worker["latency_ms"].items()
        },
        "reference_to_candidate_p95_ratio": round(
            float(worker["reference_to_candidate_p95_ratio"]), 6
        ),
        "equivalence": {
            "query_rows": EXPECTED_ROWS,
            "identity_failures": worker["equivalence_failures"],
            "score_failures": worker["score_failures"],
            "determinism_failures": worker["determinism_failures"],
        },
        "checks": checks,
        "decision": decision,
        "serving_route_changed": False,
    }
    _json(args.public_output, public)
    print(
        json.dumps(
            {
                "decision": decision,
                "atomic_views": corpus_metrics["atomic_views"],
                "latency_ms": public["latency_ms"],
                "p95_speedup": public["reference_to_candidate_p95_ratio"],
                "public_output": str(args.public_output),
            }
        )
    )


def main() -> None:
    if "--rank-worker" not in sys.argv:
        _main()
        return
    parser = argparse.ArgumentParser(description="Private atomic rescue shortlist worker")
    parser.add_argument("--rank-worker", action="store_true")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    _rank_worker(parser.parse_args())


if __name__ == "__main__":
    main()
