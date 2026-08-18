"""Run the preregistered EnterpriseRAG retrieval grid serially.

The grid is retrieval only. It reads question text and ids, never gold fields. Each arm receives
an isolated table and tenant name. Official scoring options are recorded in the manifest for the
later evaluator phase and are not passed to this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _values(text: str, cast: type[int] | type[str]) -> list[int] | list[str]:
    values = sorted({cast(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("grid values must not be empty")
    return values


def _safe_name(value: str, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must match {_SAFE_NAME.pattern}")
    return value


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    ks = _values(args.k, int)
    candidate_ks = _values(args.candidate_k, int)
    sparse_backends = _values(args.sparse_backend, str)
    rerankers = _values(args.reranker, str)
    arms: list[dict[str, Any]] = []
    index = 0
    for sparse_backend in sparse_backends:
        for reranker in rerankers:
            for candidate_k in candidate_ks:
                for k in ks:
                    index += 1
                    name = f"{sparse_backend}_{reranker.replace(':', '_')}_c{candidate_k}_k{k}"
                    output = args.out_dir / f"{name}.answers.jsonl"
                    arms.append(
                        {
                            "id": f"arm_{index:03d}",
                            "name": name,
                            "sparse_backend": sparse_backend,
                            "reranker": reranker,
                            "candidate_k": candidate_k,
                            "k": k,
                            "table": f"{args.table_prefix}_{index:03d}",
                            "tenant": f"{args.tenant_prefix}_{index:03d}",
                            "answers": str(output),
                        }
                    )
    return {
        "phase": "retrieval_only_runtime",
        "created_at": datetime.now(UTC).isoformat(),
        "questions": {"path": str(args.questions), "sha256": _digest(args.questions)},
        "documents": {"path": str(args.documents), "sha256": _digest(args.documents)},
        "runtime_policy": {
            "reasoning_arm": "none",
            "answer_mode": "extractive",
            "parallelism": 1,
            "gold_fields_used_at_runtime": False,
            "official_evaluator": {
                "no_correction": True,
                "skip_citation_stripping": True,
                "parallelism": 1,
            },
        },
        "grid": {
            "k": ks,
            "candidate_k": candidate_ks,
            "sparse_backend": sparse_backends,
            "reranker": rerankers,
            "retrieval_captures": args.retrieval_captures,
        },
        "arms": arms,
    }


def command(args: argparse.Namespace, arm: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.enterprise_rag",
        "--questions",
        str(args.questions),
        "--documents",
        str(args.documents),
        "--out",
        arm["answers"],
        "--dsn",
        args.dsn,
        "--table",
        arm["table"],
        "--tenant",
        arm["tenant"],
        "--embedder",
        args.embedder,
        "--k",
        str(arm["k"]),
        "--candidate-k",
        str(arm["candidate_k"]),
        "--sparse-backend",
        arm["sparse_backend"],
        "--reranker",
        arm["reranker"],
        "--question-ids-file",
        str(args.question_ids),
        "--retrieval-captures",
        str(args.retrieval_captures),
        "--answer-mode",
        "extractive",
        "--reasoning-arm",
        "none",
        "--overwrite",
    ] + (["--skip-index"] if args.skip_index else [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--question-ids", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--table-prefix", default="bench_er_nonreason")
    parser.add_argument("--tenant-prefix", default="er_nonreason")
    parser.add_argument("--embedder", default="voyage:voyage-4-large")
    parser.add_argument("--k", default="5,8,12")
    parser.add_argument("--candidate-k", default="100,200,400")
    parser.add_argument("--sparse-backend", default="lexical")
    parser.add_argument("--reranker", default="none,voyage:rerank-2.5")
    parser.add_argument("--retrieval-captures", type=int, default=3)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.table_prefix = _safe_name(args.table_prefix, "table prefix")
    args.tenant_prefix = _safe_name(args.tenant_prefix, "tenant prefix")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    plan_path = args.out_dir / "experiment.manifest.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print(f"wrote plan to {plan_path}")
        return 0
    for arm in plan["arms"]:
        argv = command(args, arm)
        print("running " + arm["id"] + ": " + " ".join(argv), flush=True)
        subprocess.run(argv, check=True)
    print(f"completed {len(plan['arms'])} arm(s); manifest {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
