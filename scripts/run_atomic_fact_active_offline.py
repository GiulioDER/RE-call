"""Measure the preregistered active-generation dense-six atomic rescue gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recall.atomic_rescue import load_atomic_rescue_artifact, select_atomic_rescue  # noqa: E402
from recall.calibration_v2 import CalibrationRepository, CalibrationStatus  # noqa: E402
from recall.embeddings import embed_query, embedding_profile_id, resolve_embedder  # noqa: E402
from recall.generation_store import GenerationStore  # noqa: E402
from recall.types import ScoredChunk  # noqa: E402
from scripts.run_atomic_fact_context4_pilot import Candidate, _source_root  # noqa: E402
from scripts.run_atomic_fact_release_confirmation import (  # noqa: E402
    EXPECTED_POOL_SHA256,
    EXPECTED_ROOTS,
    _candidate_payload,
    _paired,
    validate_pool,
)

PROTOCOL = "2026-09-16-atomic-fact-active-serving-offline"
EXPECTED_ROWS = 96
EXPECTED_PROFILE = "voyage-context-4-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if private:
        os.chmod(path, 0o600)


def _validate_runtime() -> None:
    if os.environ.get("RECALL_ATOMIC_ACTIVE_HOST") != "vps2":
        raise RuntimeError("active regression query embeddings are allowed only on VPS2")
    if os.environ.get("RECALL_ATOMIC_ACTIVE_LOCK_HELD") != "1":
        raise RuntimeError("active regression requires the shared embed.lock flock")
    if os.environ.get("RECALL_EMBED_THREADS") != "4":
        raise RuntimeError("active regression requires RECALL_EMBED_THREADS=4")


def _candidate(hit: ScoredChunk) -> Candidate:
    source = hit.chunk.metadata.get("file")
    ordinal = hit.chunk.metadata.get("ord")
    if not isinstance(source, str) or not isinstance(ordinal, int) or isinstance(ordinal, bool):
        raise RuntimeError("dense candidate lacks file or integer ord metadata")
    return Candidate(source, ordinal, hit.chunk.text, float(hit.score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage-context:voyage-context-4")
    args = parser.parse_args()
    if not args.dsn:
        raise RuntimeError("RECALL_DSN or --dsn is required")
    _validate_runtime()
    roots = dict(args.source_root)
    if set(roots) != EXPECTED_ROOTS or len(roots) != len(args.source_root):
        raise RuntimeError("source roots must contain each frozen production root exactly once")
    rows = validate_pool(args.pool, roots)

    started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    if embedder.dim != 1024 or embedding_profile_id(embedder) != EXPECTED_PROFILE:
        raise RuntimeError("runtime embedder is not the frozen Context 4 profile")
    artifact = load_atomic_rescue_artifact(args.artifact)
    artifact.assert_lineage(
        generation_id=args.generation,
        calibration_id=args.calibration,
        pipeline_fingerprint=args.pipeline,
        corpus_fingerprint=args.corpus,
        embedder=embedder,
    )
    repository = CalibrationRepository(args.dsn, args.tenant, actor="atomic-active-offline")
    resolution = repository.resolve(args.generation)
    if (
        resolution.status is not CalibrationStatus.CERTIFIED
        or resolution.artifact is None
        or resolution.artifact.calibration_id != args.calibration
    ):
        raise RuntimeError("active generation calibration is not certified and published")

    private_rows: list[dict[str, Any]] = []
    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as raw_store:
        store = cast(GenerationStore, raw_store)
        store.set_fixed_generation(args.generation)
        expected = {
            "tenant_id": args.tenant,
            "generation_id": args.generation,
            "pipeline_fingerprint": args.pipeline,
            "corpus_fingerprint": args.corpus,
        }
        binding = store.generation_binding()
        if any(binding.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"active serving lineage changed: {binding}")
        for row in rows:
            vector = embed_query(embedder, str(row["query"]))
            dense_hits = store.query_dense(vector, k=6)
            if len(dense_hits) != 6 or len({hit.chunk.id for hit in dense_hits}) != 6:
                raise RuntimeError("dense retrieval did not return six distinct parents")
            selected = select_atomic_rescue(artifact, vector, dense_hits)
            dense = [_candidate(hit) for hit in dense_hits]
            atomic = Candidate(selected.source, selected.parent_ordinal, "", selected.score)
            rescue = [*dense[:5], atomic]
            private_rows.append(
                {
                    "query_id": str(row["id"]),
                    "gold_source": str(row["gold_sources"][0]),
                    "gold_ordinal": int(row["gold_ordinal"]),
                    "dense": _candidate_payload(dense),
                    "dense5_atomic1": _candidate_payload(rescue),
                }
            )
        active_generation_unchanged = store.active_generation_id() == args.generation

    comparison = {label: _paired(private_rows, label) for label in ("exact", "gold")}
    checks = {
        "row_count": len(private_rows) == EXPECTED_ROWS,
        "active_generation_unchanged": active_generation_unchanged,
        "exact_net_gain_gte_8": comparison["exact"]["net"] >= 8,
        "exact_losses_lte_1": comparison["exact"]["losses"] <= 1,
        "gold_net_gain_gte_5": comparison["gold"]["net"] >= 5,
        "gold_losses_lte_1": comparison["gold"]["losses"] <= 1,
    }
    decision = (
        "PASS_ATOMIC_ACTIVE_OFFLINE"
        if all(checks.values())
        else "STOP_ATOMIC_ACTIVE_SERVING"
    )
    private = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "pool_sha256": EXPECTED_POOL_SHA256,
        "generation_id": args.generation,
        "rows": private_rows,
    }
    _json(args.private_output, private, private=True)
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    public: Mapping[str, object] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "pool_sha256": EXPECTED_POOL_SHA256,
        "private_rows_sha256": _sha256(args.private_output),
        "generation_id": args.generation,
        "calibration_id": args.calibration,
        "pipeline_fingerprint": args.pipeline,
        "corpus_fingerprint": args.corpus,
        "artifact_matrix_sha256": artifact.matrix_sha256,
        "artifact_metadata_sha256": artifact.metadata_sha256,
        "rows": len(private_rows),
        "comparison": comparison,
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "decision": decision,
        "serving_route_changed": False,
    }
    _json(args.public_output, public)
    print(json.dumps({"decision": decision, "comparison": comparison}, ensure_ascii=False))


if __name__ == "__main__":
    main()
