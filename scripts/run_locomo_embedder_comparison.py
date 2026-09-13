"""Compare two explicit embedders on LOCOMO gold retrieval.

This runner uses the production generation build and retrieval path. Each arm receives its own
immutable generation, and each conversation is one contextual document group for Voyage Context.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any

from benchmarks.systems import resolve_embedder, resolve_reranker
from recall.embeddings import resolve_registered_embedder
from recall.eval.locomo import (
    ANSWERABLE_CATEGORIES,
    CATEGORY_NAMES,
    run_conversation,
    write_conversation_corpus,
)
from recall.generation_build import BuildRequest, build_generation
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import LocalObjectReader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(flags: list[bool]) -> float:
    return statistics.fmean(flags) if flags else 0.0


def _bootstrap_delta(control: list[bool], treatment: list[bool]) -> dict[str, float]:
    if len(control) != len(treatment) or not control:
        return {"mean": 0.0, "lo95": 0.0, "hi95": 0.0}
    differences = [int(b) - int(a) for a, b in zip(control, treatment)]
    rng = random.Random(20260912)
    samples = []
    for _ in range(10000):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        samples.append(statistics.fmean(sample))
    samples.sort()
    return {
        "mean": statistics.fmean(differences),
        "lo95": samples[25],
        "hi95": samples[9974],
    }


def _band(hit_by_k: dict[int, bool]) -> str:
    if hit_by_k.get(5):
        return "1-5"
    if hit_by_k.get(10):
        return "6-10"
    if hit_by_k.get(20):
        return "11-20"
    return ">20"


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


def _question_ids(conversation: dict[str, Any], qa: list[dict[str, Any]]) -> dict[int, str]:
    sample_id = str(conversation["sample_id"])
    return {index: f"{sample_id}:{index}" for index, row in enumerate(qa) if row.get("question")}


def _resolve_embedder(name: str) -> Any:
    profiles = {
        "voyage:voyage-4": "voyage-4-v1",
        "voyage-context:voyage-context-4": "voyage-context-4-v1",
    }
    profile_id = profiles.get(name)
    if profile_id is not None:
        return resolve_registered_embedder(profile_id, os.environ)
    return resolve_embedder(name)


def _run_arm(
    data: list[dict[str, Any]],
    *,
    arm: str,
    embedder_name: str,
    dsn: str,
    table: str,
    run_id: str,
    reranker_name: str = "none",
    reuse_generation_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    embedder = _resolve_embedder(embedder_name)
    reranker = resolve_reranker(reranker_name)
    workspace = Path(tempfile.mkdtemp(prefix=f"locomo-embedder-{arm}-"))
    rows: list[dict[str, Any]] = []
    generation_ids: list[str] = []
    try:
        for number, conversation in enumerate(data, start=1):
            sample_id = str(conversation["sample_id"])
            storage_arm = arm.removesuffix("-reranked") if reuse_generation_ids is not None else arm
            tenant = f"{run_id}-{storage_arm}-{sample_id}"
            qa = conversation.get("qa") or []
            corpus_dir = workspace / sample_id
            n_turns = write_conversation_corpus(conversation["conversation"], corpus_dir)
            entries = []
            for path in sorted(corpus_dir.iterdir()):
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                entries.append(
                    ManifestObjectV1(
                        uri=path.resolve().as_uri(),
                        version_id=digest,
                        media_type="text/markdown",
                        size=len(content),
                        sha256=digest,
                        context_group_id=sample_id,
                    )
                )
            manifest = IndexManifestV1(
                tenant,
                f"{run_id}-{arm}-{sample_id}",
                tuple(entries),
            )
            if reuse_generation_ids is not None:
                if len(reuse_generation_ids) != len(data):
                    raise ValueError("reuse_generation_ids must have one generation per conversation")
                generation_id = reuse_generation_ids[number - 1]
            else:
                manager = GenerationManager(
                    dsn,
                    tenant,
                    actor="locomo-production-benchmark",
                    environment="test",
                )
                generation = build_generation(
                    manager,
                    manifest,
                    LocalObjectReader([corpus_dir]),
                    embedder,
                    # Benchmark generations are isolated test-environment records. Hosted provider
                    # identity is still recorded fully, while this explicit flag satisfies the test
                    # environment gate without weakening production generation admission.
                    BuildRequest(commit_root=None, unverified=True),
                )
                manager.validate(generation.generation_id)
                generation_id = generation.generation_id
            generation_ids.append(generation_id)
            with GenerationStore(dsn, embedder.dim, tenant=tenant) as store:
                store.set_fixed_generation(generation_id)
                result = run_conversation(
                    conversation["conversation"],
                    qa,
                    store=store,
                    embedder=embedder,
                    k=5,
                    ks=[1, 3, 5, 10, 20],
                    candidate_k=20,
                    corpus_dir=workspace / sample_id,
                    reranker=reranker,
                    skip_index=True,
                )
            if n_turns is not None and result["turns"] != n_turns:
                raise RuntimeError("Context 4 pre-index turn count disagrees with scoring path")
            result_rows = result["questions"]
            aligned: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
            for result_row in result_rows:
                aligned[(int(result_row["category"]), str(result_row["question"]))].append(result_row)
            for index, question in enumerate(qa):
                if not question.get("question") or question.get("category") not in {*ANSWERABLE_CATEGORIES, 5}:
                    continue
                key = (int(question["category"]), str(question["question"]))
                if not aligned[key]:
                    continue
                current = aligned[key].pop(0)
                if question.get("category") not in ANSWERABLE_CATEGORIES:
                    continue
                if not current.get("evidence"):
                    continue
                hit_by_k = {int(key): bool(value) for key, value in current["hit_by_k"].items()}
                retrieved = list(current.get("retrieved") or [])
                gold = {str(value) for value in question["evidence"]}
                first_rank = next(
                    (rank for rank, value in enumerate(retrieved, start=1) if value in gold), None
                )
                rows.append(
                    {
                        "question_id": f"{sample_id}:{index}",
                        "category": CATEGORY_NAMES[int(question["category"])],
                        "question": str(question["question"]),
                        "gold": sorted(gold),
                        "retrieved": retrieved,
                        "hit_by_k": {str(key): value for key, value in sorted(hit_by_k.items())},
                        "first_gold_rank": first_rank,
                        "miss_band": _band(hit_by_k),
                    }
                )
            print(
                f"  {arm} [{number}/{len(data)}] {sample_id}: {result['turns']} turns "
                f"generation={generation_id}",
                flush=True,
            )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    profile = getattr(embedder, "profile", None)
    return rows, {
        "embedder": embedder.name,
        "dimension": embedder.dim,
        "profile_id": getattr(profile, "profile_id", None),
        "profile_fingerprint": profile.fingerprint() if profile is not None else None,
        "profile": asdict(profile) if profile is not None else None,
        "reranker": reranker_name,
        "generation_ids": generation_ids,
        "table_argument_unused": table,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "questions": len(rows),
        "hit_rate": {str(k): _rate([bool(row["hit_by_k"].get(str(k))) for row in rows]) for k in (1, 3, 5, 10, 20)},
        "by_category": {
            category: {
                "questions": len(category_rows),
                "hit_at_5": _rate([bool(row["hit_by_k"].get("5")) for row in category_rows]),
                "hit_at_10": _rate([bool(row["hit_by_k"].get("10")) for row in category_rows]),
                "hit_at_20": _rate([bool(row["hit_by_k"].get("20")) for row in category_rows]),
            }
            for category, category_rows in sorted(by_category.items())
        },
        "miss_attribution": {
            band: sum(row["miss_band"] == band for row in rows)
            for band in ("1-5", "6-10", "11-20", ">20")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", default="voyage:voyage-4")
    parser.add_argument("--treatment", default="voyage-context:voyage-context-4")
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"))
    parser.add_argument("--control-table", default="locomo_embedder_v3_20260912")
    parser.add_argument("--treatment-table", default="locomo_embedder_v4_20260912")
    parser.add_argument("--control-reranked-table", default="locomo_embedder_v3_rerank_20260913")
    parser.add_argument("--treatment-reranked-table", default="locomo_embedder_v4_rerank_20260913")
    parser.add_argument("--reranker", default="voyage:rerank-2.5")
    parser.add_argument(
        "--control-generation-ids",
        default=None,
        help="comma separated ready control generation IDs to reuse when resuming a run",
    )
    parser.add_argument(
        "--treatment-generation-ids",
        default=None,
        help="comma separated ready treatment generation IDs to reuse when resuming a run",
    )
    parser.add_argument("--run-id", default=None, help="reuse an existing benchmark run ID")
    args = parser.parse_args()
    _safe_identifier(args.control_table)
    _safe_identifier(args.treatment_table)
    _safe_identifier(args.control_reranked_table)
    _safe_identifier(args.treatment_reranked_table)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError("expected the frozen 10 conversation LOCOMO dataset")
    def _generation_ids(raw: str | None, option: str) -> list[str] | None:
        if not raw:
            return None
        values = [value.strip() for value in raw.split(",") if value.strip()]
        if len(values) != len(data):
            raise ValueError(f"{option} must contain one ID per conversation")
        return values

    control_generation_ids = _generation_ids(args.control_generation_ids, "--control-generation-ids")
    treatment_generation_ids = _generation_ids(args.treatment_generation_ids, "--treatment-generation-ids")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("embedder%Y%m%dT%H%M%SZ")
    control, control_meta = _run_arm(
        data, arm="control", embedder_name=args.control, dsn=args.dsn,
        table=args.control_table, run_id=run_id, reuse_generation_ids=control_generation_ids,
    )
    treatment, treatment_meta = _run_arm(
        data, arm="treatment", embedder_name=args.treatment, dsn=args.dsn,
        table=args.treatment_table, run_id=run_id, reuse_generation_ids=treatment_generation_ids,
    )
    control_reranked, control_reranked_meta = _run_arm(
        data, arm="control-reranked", embedder_name=args.control, dsn=args.dsn,
        table=args.control_reranked_table, run_id=run_id, reranker_name=args.reranker,
        reuse_generation_ids=control_meta["generation_ids"],
    )
    treatment_reranked, treatment_reranked_meta = _run_arm(
        data, arm="treatment-reranked", embedder_name=args.treatment, dsn=args.dsn,
        table=args.treatment_reranked_table, run_id=run_id, reranker_name=args.reranker,
        reuse_generation_ids=treatment_meta["generation_ids"],
    )
    def _paired(control_rows: list[dict[str, Any]], treatment_rows: list[dict[str, Any]]) -> dict[str, Any]:
        control_by_id = {row["question_id"]: row for row in control_rows}
        treatment_by_id = {row["question_id"]: row for row in treatment_rows}
        if set(control_by_id) != set(treatment_by_id):
            raise RuntimeError("paired arms did not produce the same answerable question set")
        pair_ids = sorted(control_by_id)
        return {
            "questions": len(pair_ids),
            "hit_at_5": _bootstrap_delta(
                [bool(control_by_id[key]["hit_by_k"].get("5")) for key in pair_ids],
                [bool(treatment_by_id[key]["hit_by_k"].get("5")) for key in pair_ids],
            ),
            "hit_at_10": _bootstrap_delta(
                [bool(control_by_id[key]["hit_by_k"].get("10")) for key in pair_ids],
                [bool(treatment_by_id[key]["hit_by_k"].get("10")) for key in pair_ids],
            ),
            "hit_at_20": _bootstrap_delta(
                [bool(control_by_id[key]["hit_by_k"].get("20")) for key in pair_ids],
                [bool(treatment_by_id[key]["hit_by_k"].get("20")) for key in pair_ids],
            ),
            "rescues_at_5": sum(
                not control_by_id[key]["hit_by_k"].get("5")
                and treatment_by_id[key]["hit_by_k"].get("5")
                for key in pair_ids
            ),
            "regressions_at_5": sum(
                control_by_id[key]["hit_by_k"].get("5")
                and not treatment_by_id[key]["hit_by_k"].get("5")
                for key in pair_ids
            ),
        }

    payload = {
        "protocol": "2026-09-13-voyage-context4-production-path",
        "preregistration": "docs/preregistrations/2026-09-13-voyage-context4-production-path.md",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": os.environ.get("RECALL_SOURCE_COMMIT")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_sha256": _sha256(args.data),
        "arms": {
            "voyage4": {**control_meta, "summary": _summarize(control), "rows": control},
            "context4": {**treatment_meta, "summary": _summarize(treatment), "rows": treatment},
            "voyage4_reranked": {**control_reranked_meta, "summary": _summarize(control_reranked), "rows": control_reranked},
            "context4_reranked": {**treatment_reranked_meta, "summary": _summarize(treatment_reranked), "rows": treatment_reranked},
        },
        "paired": {
            "no_reranker": _paired(control, treatment),
            "reranker": _paired(control_reranked, treatment_reranked),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "paired": payload["paired"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
