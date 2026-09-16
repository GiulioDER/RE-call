"""Run the frozen exhaustive atomic fact Context 4 retrieval pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

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
from recall.lineage import ManifestObjectV1  # noqa: E402
from recall.manifest import local_path_for  # noqa: E402
from scripts.run_production_atomic_fact_fresh_audit import (  # noqa: E402
    SKIP_NAMES,
    _normalize,
    build_source_views,
)


EXPECTED_POOL_SHA256 = "720272504e128d861b09b7d29df8570eadfe3362865a256951c0b404bf448aa9"
EXPECTED_GENERATION = "gen_cd269b86b7364e25843ed7d7dda7c419"
EXPECTED_CALIBRATION = "cal_1f12dcbfa2b64e10a1548051cd2a5fcd"
EXPECTED_PIPELINE = "57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86"
EXPECTED_CORPUS = "5fc44dffc89b0e4f2d769bb4b52c4f3671a1d3c9ee2b30462a1b4aa0f33d24fc"
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
EXPECTED_RESOURCE_POLICY = (
    "MemoryMax=8G;MemorySwapMax=0;CPUQuota=250%;nice=15;threads=4"
)
CUTOFFS = (1, 3, 5, 10, 20)
RRF_CONSTANT = 60


@dataclass(frozen=True)
class Candidate:
    source: str
    ordinal: int
    text: str
    score: float

    @property
    def identity(self) -> tuple[str, int]:
        return self.source, self.ordinal


@dataclass(frozen=True)
class AtomicView:
    source: str
    parent_ordinal: int
    view_ordinal: int
    rendered: str
    parent_text: str

    @property
    def parent_identity(self) -> tuple[str, int]:
        return self.source, self.parent_ordinal


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_root(value: str) -> tuple[str, Path]:
    prefix, separator, raw_path = value.partition("=")
    if not separator or not prefix or not raw_path:
        raise argparse.ArgumentTypeError("source root must be PREFIX=PATH")
    return prefix, Path(raw_path)


def _manifest_object(value: Mapping[str, Any]) -> ManifestObjectV1:
    context_group = value.get("context_group_id")
    return ManifestObjectV1(
        uri=str(value.get("uri", "")),
        version_id=str(value.get("version_id", "")),
        media_type=str(value.get("media_type", "")),
        size=int(value.get("size", -1)),
        sha256=str(value.get("sha256", "")),
        context_group_id=str(context_group) if context_group is not None else None,
    )


def source_for_path(path: Path, roots: Mapping[str, Path]) -> str:
    resolved = path.resolve()
    matches: list[tuple[str, Path]] = []
    for prefix, root in roots.items():
        resolved_root = root.resolve()
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError:
            continue
        matches.append((prefix, relative))
    if len(matches) != 1:
        raise RuntimeError(
            f"manifest path {resolved} matched {len(matches)} configured source roots"
        )
    prefix, relative = matches[0]
    return f"{prefix}/{relative.as_posix()}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("cosine inputs have different or empty dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise RuntimeError("cosine input has zero norm")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def rank_atomic(
    views: Sequence[AtomicView],
    vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    *,
    cutoff: int = 20,
) -> list[Candidate]:
    if len(views) != len(vectors):
        raise RuntimeError("atomic view and vector counts differ")
    ranked = sorted(
        (
            (_cosine(vector, query_vector), view)
            for view, vector in zip(views, vectors, strict=True)
        ),
        key=lambda item: (
            -item[0],
            item[1].source,
            item[1].parent_ordinal,
            item[1].view_ordinal,
        ),
    )
    output: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for score, view in ranked:
        if view.parent_identity in seen:
            continue
        seen.add(view.parent_identity)
        output.append(
            Candidate(view.source, view.parent_ordinal, view.parent_text, score)
        )
        if len(output) == cutoff:
            break
    if len(output) != cutoff:
        raise RuntimeError(f"atomic retrieval returned {len(output)} unique parents")
    return output


def equal_rrf(
    dense: Sequence[Candidate],
    atomic: Sequence[Candidate],
    *,
    cutoff: int = 20,
    constant: int = RRF_CONSTANT,
) -> list[Candidate]:
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    dense_rank = {item.identity: rank for rank, item in enumerate(dense, 1)}
    atomic_rank = {item.identity: rank for rank, item in enumerate(atomic, 1)}
    candidates: dict[tuple[str, int], Candidate] = {}
    for item in (*dense, *atomic):
        existing = candidates.get(item.identity)
        if existing is not None and existing.text != item.text:
            raise RuntimeError(f"parent {item.identity!r} has inconsistent text")
        candidates.setdefault(item.identity, item)

    def score(identity: tuple[str, int]) -> float:
        return sum(
            1.0 / (constant + ranks[identity])
            for ranks in (dense_rank, atomic_rank)
            if identity in ranks
        )

    def key(identity: tuple[str, int]) -> tuple[float, int, int, str, int]:
        ranks = [
            ranks[identity]
            for ranks in (dense_rank, atomic_rank)
            if identity in ranks
        ]
        return (
            -score(identity),
            min(ranks),
            dense_rank.get(identity, sys.maxsize),
            identity[0],
            identity[1],
        )

    ordered = sorted(candidates, key=key)[:cutoff]
    if len(ordered) != cutoff:
        raise RuntimeError(f"equal RRF returned {len(ordered)} unique parents")
    return [
        Candidate(
            candidates[identity].source,
            candidates[identity].ordinal,
            candidates[identity].text,
            score(identity),
        )
        for identity in ordered
    ]


def _labelled_candidates(
    candidates: Sequence[Candidate],
    *,
    gold_sources: set[str],
    answer_span: str,
) -> list[dict[str, object]]:
    normalized_span = _normalize(answer_span)
    return [
        {
            "source": item.source,
            "ordinal": item.ordinal,
            "score": item.score,
            "exact": normalized_span in _normalize(item.text),
            "gold": item.source in gold_sources,
        }
        for item in candidates
    ]


def _reach(rows: Sequence[Mapping[str, Any]], arm: str, label: str, cutoff: int) -> int:
    return sum(
        any(bool(item[label]) for item in row["arms"][arm][:cutoff])
        for row in rows
    )


def _arm_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "exact_by_cutoff": {
            str(cutoff): _reach(rows, arm, "exact", cutoff) for cutoff in CUTOFFS
        },
        "gold_by_cutoff": {
            str(cutoff): _reach(rows, arm, "gold", cutoff) for cutoff in CUTOFFS
        },
    }


def _comparison(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    changed = [
        row
        for row in rows
        if (
            row["arms"]["dense"][0]["source"],
            row["arms"]["dense"][0]["ordinal"],
        )
        != (
            row["arms"][arm][0]["source"],
            row["arms"][arm][0]["ordinal"],
        )
    ]

    def gains(label: str) -> int:
        return sum(
            not bool(row["arms"]["dense"][0][label])
            and bool(row["arms"][arm][0][label])
            for row in rows
        )

    def losses(label: str) -> int:
        return sum(
            bool(row["arms"]["dense"][0][label])
            and not bool(row["arms"][arm][0][label])
            for row in rows
        )

    return {
        "changed_rank1_rows": len(changed),
        "exact_rank1_gains": gains("exact"),
        "exact_rank1_losses": losses("exact"),
        "gold_rank1_gains": gains("gold"),
        "gold_rank1_losses": losses("gold"),
        "changed_rank1_exact_precision": (
            sum(bool(row["arms"][arm][0]["exact"]) for row in changed) / len(changed)
            if changed
            else 0.0
        ),
        "changed_rank1_gold_precision": (
            sum(bool(row["arms"][arm][0]["gold"]) for row in changed) / len(changed)
            if changed
            else 0.0
        ),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {arm: _arm_metrics(rows, arm) for arm in ("dense", "atomic", "equal_rrf")}
    comparisons = {
        arm: _comparison(rows, arm) for arm in ("atomic", "equal_rrf")
    }
    qualifying: list[str] = []
    for arm in ("atomic", "equal_rrf"):
        comparison = comparisons[arm]
        exact_delta_1 = (
            metrics[arm]["exact_by_cutoff"]["1"]
            - metrics["dense"]["exact_by_cutoff"]["1"]
        )
        gold_delta_1 = (
            metrics[arm]["gold_by_cutoff"]["1"]
            - metrics["dense"]["gold_by_cutoff"]["1"]
        )
        exact_delta_20 = (
            metrics[arm]["exact_by_cutoff"]["20"]
            - metrics["dense"]["exact_by_cutoff"]["20"]
        )
        gold_delta_20 = (
            metrics[arm]["gold_by_cutoff"]["20"]
            - metrics["dense"]["gold_by_cutoff"]["20"]
        )
        comparison.update(
            {
                "exact_delta_at_1": exact_delta_1,
                "gold_delta_at_1": gold_delta_1,
                "exact_delta_at_20": exact_delta_20,
                "gold_delta_at_20": gold_delta_20,
            }
        )
        if (
            exact_delta_1 >= 2
            and gold_delta_1 >= 2
            and comparison["exact_rank1_losses"] <= 1
            and comparison["gold_rank1_losses"] <= 1
            and comparison["changed_rank1_exact_precision"] >= 0.50
            and comparison["changed_rank1_gold_precision"] >= 0.50
            and exact_delta_20 >= 0
            and gold_delta_20 >= 0
            and (exact_delta_20 >= 1 or gold_delta_20 >= 1)
        ):
            qualifying.append(arm)

    chosen: str | None = None
    if qualifying:
        chosen = sorted(
            qualifying,
            key=lambda arm: (
                -metrics[arm]["exact_by_cutoff"]["1"],
                -metrics[arm]["gold_by_cutoff"]["1"],
                -metrics[arm]["exact_by_cutoff"]["20"],
                0 if arm == "atomic" else 1,
            ),
        )[0]
    return {
        "rows": len(rows),
        "arms": metrics,
        "comparisons": comparisons,
        "qualifying_arms": qualifying,
        "chosen_arm": chosen,
        "decision": (
            "PROMISING_ATOMIC_FACT_PILOT"
            if chosen is not None
            else "STOP_ATOMIC_FACT_RETRIEVAL_PILOT"
        ),
    }


def _parent_chunks(store: GenerationStore) -> dict[tuple[str, int], str]:
    parents: dict[tuple[str, int], str] = {}
    for chunk in store.iter_chunks():
        source = chunk.metadata.get("file")
        ordinal = chunk.metadata.get("ord")
        if not isinstance(source, str) or not isinstance(ordinal, int):
            raise RuntimeError("pinned generation chunk lacks file or integer ord metadata")
        identity = source, ordinal
        if identity in parents:
            raise RuntimeError(f"duplicate pinned parent identity {identity!r}")
        parents[identity] = chunk.text
    if not parents:
        raise RuntimeError("pinned generation contains no chunks")
    return parents


def _build_atomic_views(
    objects: Iterable[Mapping[str, Any]],
    roots: Mapping[str, Path],
    parents: Mapping[tuple[str, int], str],
) -> tuple[list[list[AtomicView]], dict[str, int]]:
    groups: list[list[AtomicView]] = []
    verified_objects = 0
    excluded_index_sources = 0
    zero_view_sources = 0
    for raw_object in sorted(objects, key=lambda value: str(value.get("uri", ""))):
        entry = _manifest_object(raw_object)
        path = local_path_for(entry.uri)
        source = source_for_path(path, roots)
        if path.name in SKIP_NAMES:
            excluded_index_sources += 1
            continue
        data = path.read_bytes()
        if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.sha256:
            raise RuntimeError(f"pinned manifest bytes changed for {source}")
        verified_objects += 1
        # Decode the verified bytes directly. Path.read_text() performs universal newline
        # translation, but this frozen generation retained CRLF in its stored chunks. Translating
        # here changes the 800-character boundaries and can map a view to the wrong parent. Match
        # production's UTF-8 BOM and NUL handling before parsing; the manifest hash above remains
        # over the original bytes.
        raw = data.decode("utf-8-sig").replace("\x00", "")
        _chunks, source_views = build_source_views(raw, source)
        group: list[AtomicView] = []
        for view_ordinal, view in enumerate(source_views):
            parent_ordinal = int(view["parent_ordinal"])
            parent = parents.get((source, parent_ordinal))
            if parent is None:
                raise RuntimeError(
                    f"atomic view parent {(source, parent_ordinal)!r} is absent from the pinned generation"
                )
            if _normalize(str(view["content"])) not in _normalize(parent):
                raise RuntimeError(
                    f"atomic view content is absent from pinned parent {(source, parent_ordinal)!r}"
                )
            group.append(
                AtomicView(
                    source=source,
                    parent_ordinal=parent_ordinal,
                    view_ordinal=view_ordinal,
                    rendered=str(view["rendered"]),
                    parent_text=parent,
                )
            )
        if group:
            groups.append(group)
        else:
            zero_view_sources += 1
    return groups, {
        "manifest_objects": verified_objects,
        "excluded_index_sources": excluded_index_sources,
        "sources_with_views": len(groups),
        "zero_view_sources": zero_view_sources,
        "atomic_views": sum(len(group) for group in groups),
        "ordinary_chunks": len(parents),
    }


def _dense_candidates(store: GenerationStore, vector: list[float]) -> list[Candidate]:
    output: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for hit in store.query_dense(vector, k=20):
        source = hit.chunk.metadata.get("file")
        ordinal = hit.chunk.metadata.get("ord")
        if not isinstance(source, str) or not isinstance(ordinal, int):
            raise RuntimeError("dense candidate lacks file or integer ord metadata")
        candidate = Candidate(source, ordinal, hit.chunk.text, float(hit.score))
        if candidate.identity in seen:
            raise RuntimeError(f"dense top 20 repeated parent {candidate.identity!r}")
        seen.add(candidate.identity)
        output.append(candidate)
    if len(output) != 20:
        raise RuntimeError(f"dense retrieval returned {len(output)} unique candidates")
    return output


def _validate_pool(path: Path) -> dict[str, Any]:
    digest = _sha256(path)
    if digest != EXPECTED_POOL_SHA256:
        raise RuntimeError(f"private pool hash changed: {digest}")
    pool = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pool, dict):
        raise RuntimeError("private pool must be a JSON object")
    rows = pool.get("queries")
    if not isinstance(rows, list) or len(rows) != 22:
        raise RuntimeError("private pool must contain exactly 22 queries")
    if any(row.get("expected_answerability") != "answerable" for row in rows):
        raise RuntimeError("private pool contains a non-answerable row")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("private pool query ids are empty or duplicated")
    return pool


def _validate_runtime() -> None:
    if os.environ.get("RECALL_ATOMIC_PILOT_HOST") != "vps2":
        raise RuntimeError("atomic fact embeddings are allowed only on VPS2")
    if os.environ.get("RECALL_ATOMIC_PILOT_LOCK_HELD") != "1":
        raise RuntimeError("atomic fact embeddings require the shared embed.lock flock")
    if os.environ.get("RECALL_ATOMIC_PILOT_RESOURCE_POLICY") != EXPECTED_RESOURCE_POLICY:
        raise RuntimeError("atomic fact embeddings require the frozen resource policy")
    if os.environ.get("RECALL_EMBED_THREADS") != "4":
        raise RuntimeError("atomic fact embeddings require RECALL_EMBED_THREADS=4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
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
    roots = dict(args.source_root)
    if set(roots) != EXPECTED_ROOTS or len(roots) != len(args.source_root):
        raise RuntimeError("source roots must contain each frozen production root exactly once")
    if any(not path.is_dir() for path in roots.values()):
        raise RuntimeError("one or more source roots do not exist")
    pool = _validate_pool(args.pool)
    started = time.perf_counter()
    embedder = resolve_embedder(args.embedder)
    if embedder.dim != 1024 or embedding_profile_id(embedder) != EXPECTED_PROFILE:
        raise RuntimeError("runtime embedder is not the frozen Context 4 profile")

    repository = CalibrationRepository(args.dsn, args.tenant, actor="atomic-fact-pilot")
    resolution = repository.resolve(args.generation_id)
    artifact = resolution.artifact
    if (
        resolution.status is not CalibrationStatus.CERTIFIED
        or artifact is None
        or artifact.calibration_id != args.calibration_id
    ):
        raise RuntimeError("frozen generation calibration is not certified and published")
    objects = repository.manifest_objects_for(args.generation_id)

    args.private_output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.private_output_dir / "atomic-fact-context4-vectors.jsonl"
    private_result_path = args.private_output_dir / "atomic-fact-context4-rows.json"
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
        parents = _parent_chunks(store)
        groups, corpus_metrics = _build_atomic_views(objects, roots, parents)
        rendered_groups = [[view.rendered for view in group] for group in groups]
        vectors_by_group = embed_document_groups(embedder, rendered_groups)
        views = [view for group in groups for view in group]
        vectors = [vector for group in vectors_by_group for vector in group]
        if len(views) != len(vectors) or any(len(vector) != 1024 for vector in vectors):
            raise RuntimeError("Context 4 atomic vector shape mismatch")

        with vectors_path.open("w", encoding="utf-8", newline="\n") as vector_file:
            vector_file.write(
                json.dumps(
                    {
                        "_header": True,
                        "profile": EXPECTED_PROFILE,
                        "dimension": 1024,
                        "pool_sha256": EXPECTED_POOL_SHA256,
                        "generation_id": args.generation_id,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            for view, vector in zip(views, vectors, strict=True):
                vector_file.write(
                    json.dumps(
                        {
                            "kind": "atomic_view",
                            "source": view.source,
                            "parent_ordinal": view.parent_ordinal,
                            "view_ordinal": view.view_ordinal,
                            "vector": vector,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            measured_rows: list[dict[str, Any]] = []
            for completed, query_row in enumerate(pool["queries"], 1):
                query_vector = embed_query(embedder, str(query_row["query"]))
                if len(query_vector) != 1024:
                    raise RuntimeError("Context 4 query vector shape mismatch")
                vector_file.write(
                    json.dumps(
                        {
                            "kind": "query",
                            "query_id": str(query_row["id"]),
                            "vector": query_vector,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                dense = _dense_candidates(store, query_vector)
                atomic = rank_atomic(views, vectors, query_vector)
                fused = equal_rrf(dense, atomic)
                gold_sources = {str(value) for value in query_row["gold_sources"]}
                answer_span = str(query_row["answer_span"])
                measured_rows.append(
                    {
                        "query_id": str(query_row["id"]),
                        "arms": {
                            "dense": _labelled_candidates(
                                dense,
                                gold_sources=gold_sources,
                                answer_span=answer_span,
                            ),
                            "atomic": _labelled_candidates(
                                atomic,
                                gold_sources=gold_sources,
                                answer_span=answer_span,
                            ),
                            "equal_rrf": _labelled_candidates(
                                fused,
                                gold_sources=gold_sources,
                                answer_span=answer_span,
                            ),
                        },
                    }
                )
                print(f"atomic fact retrieval {completed}/22", flush=True)

    private_result = {
        "schema_version": 1,
        "protocol": "2026-09-16-atomic-fact-context4-exhaustive-22-pilot",
        "pool_sha256": EXPECTED_POOL_SHA256,
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "embedding_profile": EXPECTED_PROFILE,
        "summary": summarize(measured_rows),
        "rows": measured_rows,
    }
    _json(private_result_path, private_result)
    source_commit = os.environ.get("RECALL_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    public_result = {
        "schema_version": 1,
        "protocol": "2026-09-16-atomic-fact-context4-exhaustive-22-pilot",
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "preregistration_commit": os.environ.get("RECALL_POLICY_COMMIT"),
        "implementation_clarification_commit": os.environ.get(
            "RECALL_CLARIFICATION_COMMIT"
        ),
        "pool_sha256": EXPECTED_POOL_SHA256,
        "generation_id": args.generation_id,
        "calibration_id": args.calibration_id,
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "corpus_fingerprint": args.corpus_fingerprint,
        "embedding_profile": EXPECTED_PROFILE,
        "embedding_dimension": embedder.dim,
        "resource_policy": EXPECTED_RESOURCE_POLICY,
        "rrf_constant": RRF_CONSTANT,
        "corpus": corpus_metrics,
        "vectors_sha256": _sha256(vectors_path),
        "private_rows_sha256": _sha256(private_result_path),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "summary": private_result["summary"],
    }
    _json(args.public_output, public_result)
    print(
        json.dumps(
            {
                "decision": public_result["summary"]["decision"],
                "chosen_arm": public_result["summary"]["chosen_arm"],
                "corpus": corpus_metrics,
                "public_output": str(args.public_output),
            }
        )
    )


if __name__ == "__main__":
    main()
