"""Run the preregistered 96 row atomic fact release confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, cast

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
from scripts.build_atomic_fact_blind_source_holdout import (  # noqa: E402
    _normalize,
    _source_path,
    _text_sha256,
    build_source_views,
)
from scripts.run_atomic_fact_context4_pilot import (  # noqa: E402
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
)
from scripts.run_atomic_fact_exact_rescue_shortlist import (  # noqa: E402
    build_parent_codes,
    select_exact_masked_max,
    select_full_sort_reference,
)


PROTOCOL = "2026-09-16-atomic-fact-release-confirmation-retrieval"
EXPECTED_POOL_SHA256 = "414b441fd95ccc0de7a6ede329515941c09f8aef8e9fc8e20ccbdfc1d7514f0f"
EXPECTED_GENERATION = "gen_dff506e12f494965af9f109671a99e63"
EXPECTED_CALIBRATION = "cal_6171177aeb614d288baa4e28600caca1"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0"
EXPECTED_PROFILE = "voyage-context-4-v1"
EXPECTED_ROWS = 96
EXPECTED_CHUNKS = 11_385
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


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(path, 0o600)


def _validate_runtime() -> None:
    if os.environ.get("RECALL_ATOMIC_RELEASE_HOST") != "vps2":
        raise RuntimeError("atomic release embeddings are allowed only on VPS2")
    if os.environ.get("RECALL_ATOMIC_RELEASE_LOCK_HELD") != "1":
        raise RuntimeError("atomic release embeddings require the shared embed.lock flock")
    if os.environ.get("RECALL_ATOMIC_RELEASE_RESOURCE_POLICY") != EXPECTED_RESOURCE_POLICY:
        raise RuntimeError("atomic release embeddings require the frozen resource policy")
    if os.environ.get("RECALL_EMBED_THREADS") != "4":
        raise RuntimeError("atomic release embeddings require RECALL_EMBED_THREADS=4")


def validate_pool(path: Path, roots: Mapping[str, Path]) -> list[dict[str, Any]]:
    digest = _sha256(path)
    if digest != EXPECTED_POOL_SHA256:
        raise RuntimeError(f"private pool hash changed: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise RuntimeError("private pool must contain exactly 96 query rows")
    ids: list[str] = []
    sources: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("expected_answerability") != "answerable":
            raise RuntimeError("private pool contains an invalid row")
        query_id = str(row.get("id", ""))
        query = str(row.get("query", ""))
        gold_sources = row.get("gold_sources")
        if not query_id or not query or not isinstance(gold_sources, list) or len(gold_sources) != 1:
            raise RuntimeError("private pool row lacks one gold source, id, or query")
        source = str(gold_sources[0])
        path_for_source = _source_path(source, roots)
        if (
            path_for_source is None
            or not path_for_source.is_file()
            or _sha256(path_for_source) != row.get("source_sha256")
        ):
            raise RuntimeError(f"private pool source integrity failed for {query_id}")
        answer = str(row.get("answer_span", ""))
        if not answer or _text_sha256(answer) != row.get("answer_span_sha256"):
            raise RuntimeError(f"private pool answer integrity failed for {query_id}")
        _, views = build_source_views(
            path_for_source.read_text(encoding="utf-8", errors="replace"), source
        )
        matching = [
            view
            for view in views
            if int(view["parent_ordinal"]) == int(row.get("gold_ordinal", -1))
            and _normalize(str(view["content"])) == _normalize(answer)
            and str(view["construction"]) == str(row.get("construction"))
        ]
        if len(matching) != 1:
            raise RuntimeError(f"private pool gold parent integrity failed for {query_id}")
        ids.append(query_id)
        sources.append(source)
    if len(ids) != len(set(ids)) or len(sources) != len(set(sources)):
        raise RuntimeError("private pool query ids or gold sources are duplicated")
    return rows


def _candidate_payload(candidates: Sequence[Candidate]) -> list[dict[str, object]]:
    return [
        {"source": item.source, "ordinal": item.ordinal, "score": item.score}
        for item in candidates
    ]


def _hit(row: Mapping[str, Any], arm: str, cutoff: int, label: str) -> bool:
    gold_source = str(row["gold_source"])
    gold_ordinal = int(row["gold_ordinal"])
    if label == "exact":
        return any(
            str(item["source"]) == gold_source and int(item["ordinal"]) == gold_ordinal
            for item in row[arm][:cutoff]
        )
    return any(str(item["source"]) == gold_source for item in row[arm][:cutoff])


def one_sided_paired_probability(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    return float(
        sum(math.comb(discordant, k) for k in range(gains, discordant + 1))
        / (2**discordant)
    )


def _paired(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    gains = losses = ties = 0
    root_outcomes: dict[str, Counter[str]] = {}
    for row in rows:
        dense = _hit(row, "dense", 6, label)
        rescue = _hit(row, "dense5_atomic1", 6, label)
        root = str(row["gold_source"]).partition("/")[0]
        counter = root_outcomes.setdefault(root, Counter())
        if rescue and not dense:
            gains += 1
            counter["gains"] += 1
        elif dense and not rescue:
            losses += 1
            counter["losses"] += 1
        else:
            ties += 1
            counter["ties"] += 1
    net = gains - losses
    return {
        "gains": gains,
        "losses": losses,
        "ties": ties,
        "net": net,
        "absolute_percentage_point_change": net / len(rows) * 100.0 if rows else 0.0,
        "one_sided_paired_probability": one_sided_paired_probability(gains, losses),
        "root_outcomes": {
            root: {key: counter.get(key, 0) for key in ("gains", "losses", "ties")}
            for root, counter in sorted(root_outcomes.items())
        },
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = {
        "dense": {
            label: {
                str(cutoff): sum(_hit(row, "dense", cutoff, label) for row in rows)
                for cutoff in (1, 3, 5, 6)
            }
            for label in ("exact", "gold")
        },
        "dense5_atomic1": {
            label: {"6": sum(_hit(row, "dense5_atomic1", 6, label) for row in rows)}
            for label in ("exact", "gold")
        },
    }
    comparison: dict[str, Any] = {
        "exact": _paired(rows, "exact"),
        "gold": _paired(rows, "gold"),
        "appended_differs_from_dense_rank6": sum(
            (row["dense"][5]["source"], row["dense"][5]["ordinal"])
            != (row["dense5_atomic1"][5]["source"], row["dense5_atomic1"][5]["ordinal"])
            for row in rows
        ),
    }
    exact = comparison["exact"]
    gold = comparison["gold"]
    checks = {
        "exact_net_gain_gte_6": exact["net"] >= 6,
        "exact_losses_lte_1": exact["losses"] <= 1,
        "exact_p_lte_0_05": exact["one_sided_paired_probability"] <= 0.05,
        "gold_net_gain_gte_6": gold["net"] >= 6,
        "gold_losses_lte_1": gold["losses"] <= 1,
        "gold_p_lte_0_05": gold["one_sided_paired_probability"] <= 0.05,
    }
    return {
        "rows": len(rows),
        "arms": arms,
        "comparison": comparison,
        "quality_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source-root", action="append", type=_source_root, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
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
    queries = validate_pool(args.pool, roots)

    total_started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    if embedder.dim != 1024 or embedding_profile_id(embedder) != EXPECTED_PROFILE:
        raise RuntimeError("runtime embedder is not the frozen Context 4 profile")
    repository = CalibrationRepository(args.dsn, args.tenant, actor="atomic-release-confirmation")
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
    exact_latencies: list[float] = []
    reference_latencies: list[float] = []
    private_rows: list[dict[str, Any]] = []
    equivalence_failures = 0
    score_failures = 0
    arm_identity_failures = 0

    with GenerationStore(args.dsn, embedder.dim, tenant=args.tenant) as raw_store:
        store = cast(GenerationStore, raw_store)
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
        atomic_vectors = _normalize_rows(
            np.asarray([vector for group in vectors_by_group for vector in group], dtype=np.float32)
        )
        timings["document_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        if atomic_vectors.shape != (len(views), 1024):
            raise RuntimeError("Context 4 atomic vector shape mismatch")

        started = time.perf_counter()
        query_vectors: np.ndarray = np.asarray(
            [embed_query(embedder, str(row["query"])) for row in queries], dtype=np.float32
        )
        timings["query_embedding_ms"] = (time.perf_counter() - started) * 1000.0
        if query_vectors.shape != (EXPECTED_ROWS, 1024):
            raise RuntimeError("Context 4 query vector shape mismatch")

        parent_codes, code_by_identity = build_parent_codes(views)
        started = time.perf_counter()
        for row, vector in zip(queries, query_vectors, strict=True):
            dense = _dense_candidates(store, vector.tolist())[:6]
            if len(dense) != 6 or len({candidate.identity for candidate in dense}) != 6:
                arm_identity_failures += 1
                continue
            exclusions = [candidate.identity for candidate in dense[:5]]
            reference_started = time.perf_counter_ns()
            reference = select_full_sort_reference(views, atomic_vectors, vector, exclusions)
            reference_latencies.append(
                (time.perf_counter_ns() - reference_started) / 1_000_000.0
            )
            exact_started = time.perf_counter_ns()
            atomic = select_exact_masked_max(
                views,
                atomic_vectors,
                vector,
                exclusions,
                parent_codes,
                code_by_identity,
            )
            exact_latencies.append((time.perf_counter_ns() - exact_started) / 1_000_000.0)
            if atomic.identity != reference.identity:
                equivalence_failures += 1
            if atomic.score != reference.score:
                score_failures += 1
            rescue = [*dense[:5], atomic]
            if len({candidate.identity for candidate in rescue}) != 6:
                arm_identity_failures += 1
                continue
            private_rows.append(
                {
                    "query_id": str(row["id"]),
                    "gold_source": str(row["gold_sources"][0]),
                    "gold_ordinal": int(row["gold_ordinal"]),
                    "dense": _candidate_payload(dense),
                    "dense5_atomic1": _candidate_payload(rescue),
                }
            )
        timings["retrieval_ms"] = (time.perf_counter() - started) * 1000.0
        active_generation_unchanged = store.active_generation_id() == EXPECTED_GENERATION

    summary = summarize(private_rows)
    finite_scores = all(
        np.isfinite(float(item["score"]))
        for row in private_rows
        for arm in ("dense", "dense5_atomic1")
        for item in row[arm]
    )
    integrity_checks = {
        "active_generation_unchanged": active_generation_unchanged,
        "row_count": len(private_rows) == EXPECTED_ROWS,
        "atomic_vector_count": atomic_vectors.shape[0] == corpus_metrics["atomic_views"],
        "all_candidate_scores_finite": finite_scores,
        "all_arms_have_six_distinct_parents": arm_identity_failures == 0,
        "exact_selector_identity": equivalence_failures == 0,
        "exact_selector_score": score_failures == 0,
    }
    decision = (
        "PASS_ATOMIC_RELEASE_CONFIRMATION"
        if all(integrity_checks.values()) and all(summary["quality_checks"].values())
        else "STOP_ATOMIC_RELEASE_LANE"
    )
    private_payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generation_id": EXPECTED_GENERATION,
        "pool_sha256": EXPECTED_POOL_SHA256,
        "rows": private_rows,
    }
    _json(args.private_output, private_payload)
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000.0
    selector_latency = {
        "full_sort_reference_p50": _percentile(reference_latencies, 50),
        "full_sort_reference_p95": _percentile(reference_latencies, 95),
        "full_sort_reference_p99": _percentile(reference_latencies, 99),
        "exact_masked_max_p50": _percentile(exact_latencies, 50),
        "exact_masked_max_p95": _percentile(exact_latencies, 95),
        "exact_masked_max_p99": _percentile(exact_latencies, 99),
    }
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
        "timings_ms": {key: round(value, 3) for key, value in timings.items()},
        "selector_latency_ms": {
            key: round(value, 6) for key, value in selector_latency.items()
        },
        "private_rows_sha256": _sha256(args.private_output),
        "summary": summary,
        "integrity_checks": integrity_checks,
        "decision": decision,
        "serving_route_changed": False,
    }
    _json(args.public_output, public)
    del vectors_by_group, atomic_vectors, query_vectors, groups
    gc.collect()
    print(json.dumps({"decision": decision, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
