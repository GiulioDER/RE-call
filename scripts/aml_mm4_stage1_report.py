"""MM-4 Stage 1 report: sidecar census and S versus M4r retrieval on MemEye.

Pre-registration: ``docs/preregistrations/2026-09-25-aml-c9-image-text-sidecar.md`` (amendment 2).

* ``census`` reads, for every scenario tenant a ``--keep-tenants`` Stage 1 run left, the sidecar rows
  in its ``_imgtext`` namespace (one per image read at Add), against the number of distinct images
  the scenario's dialogues hold. Tenant names come from the production ``tenant_for`` and
  ``image_text_tenant``; each query sets ``recall.tenant_id`` first, as the store does.
* ``compare`` sets S against M4r on MM-1's any-clue Recall@10 by session (``row_metrics``) and on
  MRR by session, each averaged over a question's option rotations, paired over questions with a
  10,000-resample bootstrap (seed 20260925).

    python scripts/aml_mm4_stage1_report.py census --cache-dir DIR --run-id ID --dsn DSN --table T
    python scripts/aml_mm4_stage1_report.py compare --cache-dir DIR --results S1.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aml_mm_scope_report import load_questions, row_metrics  # noqa: E402
from aml_mm_scope_stage1 import SCENARIOS  # noqa: E402

SEED = 20260925


def reciprocal_rank(items: list[dict[str, Any]], clues: set[str]) -> float:
    """1 / rank of the first item from a clue session; 0 when none is returned."""
    for rank, item in enumerate(items, start=1):
        if str(item.get("session_id", "")) in clues:
            return 1.0 / rank
    return 0.0


def per_question(rows: list[dict[str, Any]], questions: dict[tuple[str, str], dict[str, Any]],
                 arms: tuple[str, ...] = ("S", "M4r")) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Each question's metrics per arm, averaged over its option rotations."""
    grouped: dict[tuple[str, str], dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["arm"] not in arms:
            continue
        key = (row["scenario"], str(row["question_id"]))
        question = questions[key]
        grouped[key][row["arm"]].append({
            "recall_at_10": row_metrics(row["items"], question)["any_clue_recall_at_10"],
            "mrr": reciprocal_rank(row["items"], set(question["clues"])),
        })
    out = {}
    for key, by_arm in grouped.items():
        if set(by_arm) != set(arms):
            raise SystemExit(f"{key} lacks an arm: {sorted(by_arm)}")
        out[key] = {arm: {metric: statistics.fmean(r[metric] for r in rotations)
                          for metric in ("recall_at_10", "mrr")}
                    for arm, rotations in by_arm.items()}
    return out


def paired(values: dict[tuple[str, str], dict[str, dict[str, float]]], metric: str,
           a: str = "M4r", b: str = "S") -> dict[str, Any]:
    keys = sorted(values)
    diffs = [values[k][a][metric] - values[k][b][metric] for k in keys]
    rng = random.Random(SEED)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
    return {"n": len(keys), a: round(statistics.fmean(values[k][a][metric] for k in keys), 4),
            b: round(statistics.fmean(values[k][b][metric] for k in keys), 4),
            "diff": round(statistics.fmean(diffs), 4), "ci95": [round(boots[249], 4), round(boots[9_749], 4)],
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}


def compare_arms(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.cache_dir)
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").split("\n") if line.strip()]
    values = per_question(rows, questions)
    result: dict[str, Any] = {"all": {m: paired(values, m) for m in ("recall_at_10", "mrr")}, "by_scenario": {}}
    for scenario in SCENARIOS:
        subset = {k: v for k, v in values.items() if k[0] == scenario}
        if subset:
            result["by_scenario"][scenario] = {m: paired(subset, m) for m in ("recall_at_10", "mrr")}
    return result


def census(args: argparse.Namespace) -> dict[str, Any]:
    import psycopg

    from recall_aml.identity import tenant_for
    from recall_aml.image_text import image_text_tenant

    result: dict[str, Any] = {"scenarios": {}}
    lengths: list[int] = []
    with psycopg.connect(args.dsn) as conn:
        for index, scenario in enumerate(SCENARIOS):
            dataset = json.loads((args.cache_dir / f"{scenario}.json").read_text(encoding="utf-8"))
            images = {str(image) for session in dataset["multi_session_dialogues"]
                      for dialogue in session["dialogues"] for image in dialogue.get("input_image", [])}
            tenant = image_text_tenant(tenant_for(f"mms-{args.run_id}-{index}"))
            with conn.transaction():
                conn.execute("SELECT set_config('recall.tenant_id', %s, true)", (tenant,))
                rows = conn.execute(
                    f"SELECT id, text, metadata FROM {args.table} WHERE tenant_id = %s ORDER BY id",  # noqa: S608
                    (tenant,)).fetchall()
            texts = [row[1] for row in rows]
            # Kept for Stage 2: M4s is rendered offline from M4r's stored responses plus these texts,
            # and the tenants are deleted right after this census.
            result.setdefault("sidecars", {})[scenario] = [
                {"id": str(row[0]), "primary_id": str((row[2] or {}).get("primary_id", "")), "text": str(row[1])}
                for row in rows]
            words = [len(str(text).split()) for text in texts if str(text).strip()]
            lengths.extend(words)
            result["scenarios"][scenario] = {"images": len(images), "sidecars": len(texts),
                                             "non_empty": len(words)}
    total = sum(s["images"] for s in result["scenarios"].values())
    non_empty = sum(s["non_empty"] for s in result["scenarios"].values())
    result.update({"images": total, "non_empty": non_empty,
                   "non_empty_share": round(non_empty / total, 4) if total else None,
                   "median_words": statistics.median(lengths) if lengths else None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census")
    c.add_argument("--cache-dir", type=Path, required=True)
    c.add_argument("--run-id", required=True)
    c.add_argument("--dsn", required=True)
    c.add_argument("--table", required=True)
    c.set_defaults(handler=census)
    r = sub.add_parser("compare")
    r.add_argument("--cache-dir", type=Path, required=True)
    r.add_argument("--results", type=Path, required=True)
    r.set_defaults(handler=compare_arms)
    args = parser.parse_args()
    print(json.dumps(args.handler(args), indent=2))


if __name__ == "__main__":
    main()
