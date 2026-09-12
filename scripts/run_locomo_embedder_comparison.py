"""Compare two explicit embedders on LOCOMO gold retrieval.

This runner uses the production LOCOMO indexing and retrieval path, but resolves model-specific
names through ``benchmarks.systems.resolve_embedder``. The older ``recall.eval.locomo`` CLI accepts
only the bare ``voyage`` selector and otherwise falls back to its local embedder.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from benchmarks.systems import resolve_embedder
from recall.eval.locomo import ANSWERABLE_CATEGORIES, CATEGORY_NAMES, run_conversation
from recall.store import PgVectorStore


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


def _run_arm(
    data: list[dict[str, Any]],
    *,
    arm: str,
    embedder_name: str,
    dsn: str,
    table: str,
    run_id: str,
) -> list[dict[str, Any]]:
    embedder = resolve_embedder(embedder_name)
    workspace = Path(tempfile.mkdtemp(prefix=f"locomo-embedder-{arm}-"))
    rows: list[dict[str, Any]] = []
    try:
        for number, conversation in enumerate(data, start=1):
            sample_id = str(conversation["sample_id"])
            tenant = f"{run_id}-{arm}-{sample_id}"
            qa = conversation.get("qa") or []
            with PgVectorStore(dsn, dim=embedder.dim, table=table, tenant=tenant) as store:
                store.ensure_schema()
                result = run_conversation(
                    conversation["conversation"],
                    qa,
                    store=store,
                    embedder=embedder,
                    k=5,
                    ks=[1, 3, 5, 10, 20],
                    candidate_k=20,
                    corpus_dir=workspace / sample_id,
                )
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
            print(f"  {arm} [{number}/{len(data)}] {sample_id}: {result['turns']} turns", flush=True)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return rows


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
    parser.add_argument("--control", default="voyage:voyage-3")
    parser.add_argument("--treatment", default="voyage:voyage-4")
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"))
    parser.add_argument("--control-table", default="locomo_embedder_v3_20260912")
    parser.add_argument("--treatment-table", default="locomo_embedder_v4_20260912")
    args = parser.parse_args()
    _safe_identifier(args.control_table)
    _safe_identifier(args.treatment_table)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError("expected the frozen 10 conversation LOCOMO dataset")
    run_id = datetime.now(timezone.utc).strftime("embedder%Y%m%dT%H%M%SZ")
    control = _run_arm(data, arm="control", embedder_name=args.control, dsn=args.dsn, table=args.control_table, run_id=run_id)
    treatment = _run_arm(data, arm="treatment", embedder_name=args.treatment, dsn=args.dsn, table=args.treatment_table, run_id=run_id)
    control_by_id = {row["question_id"]: row for row in control}
    treatment_by_id = {row["question_id"]: row for row in treatment}
    ids = sorted(set(control_by_id) & set(treatment_by_id))
    if len(ids) != len(control) or len(ids) != len(treatment):
        raise RuntimeError("arms did not produce the same answerable question set")
    paired = {
        "questions": len(ids),
        "hit_at_5": _bootstrap_delta(
            [bool(control_by_id[key]["hit_by_k"].get("5")) for key in ids],
            [bool(treatment_by_id[key]["hit_by_k"].get("5")) for key in ids],
        ),
        "hit_at_10": _bootstrap_delta(
            [bool(control_by_id[key]["hit_by_k"].get("10")) for key in ids],
            [bool(treatment_by_id[key]["hit_by_k"].get("10")) for key in ids],
        ),
        "hit_at_20": _bootstrap_delta(
            [bool(control_by_id[key]["hit_by_k"].get("20")) for key in ids],
            [bool(treatment_by_id[key]["hit_by_k"].get("20")) for key in ids],
        ),
        "rescues_at_5": sum(
            not control_by_id[key]["hit_by_k"].get("5") and treatment_by_id[key]["hit_by_k"].get("5")
            for key in ids
        ),
        "regressions_at_5": sum(
            control_by_id[key]["hit_by_k"].get("5") and not treatment_by_id[key]["hit_by_k"].get("5")
            for key in ids
        ),
    }
    payload = {
        "protocol": "2026-09-12-embedder-gold-retrieval-comparison",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": os.environ.get("RECALL_SOURCE_COMMIT")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_sha256": _sha256(args.data),
        "control": {"embedder": args.control, "table": args.control_table, "summary": _summarize(control), "rows": control},
        "treatment": {"embedder": args.treatment, "table": args.treatment_table, "summary": _summarize(treatment), "rows": treatment},
        "paired": paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "questions": len(ids), "paired": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
