"""W0 R0: does the Qwen proxy's ranking loss reach the answer? LoCoMo, served C9 against the proxy.

Pre-registration: kept in the maintainer's private research log (W0 R0, 2026-09-29).

W0 stage P found the proxy (`C9_qwen8b_proxy`) losing 3.32 points of turn_hit@10 on LoCoMo, almost
all of it ranking (turn_hit@100 99.74 against 99.87). AML reads up to 100 items, so this answers the
question the retrieval rule cannot: does the loss reach the reader?

It reuses ``scripts/aml_w2w4_answers.py`` whole: its collect (C9 in process, items stored), its
answer loop (AML's LoCoMo prompt and judge, DeepSeek pinned, credit floor, resume) and its paired
bootstrap. Only two things are new. Each retrieval arm (A served, P proxy) is collected separately
and the two collects are merged per question; and the renderer picks the arm's retrieval, then
renders it exactly as W2/W4's arm B does (the served date header). A2 answers A's retrieval again,
the noise floor.

    python scripts/aml_w0_r0_answers.py collect --arm A --locomo locomo10.json --out C-A.json.gz
    python scripts/aml_w0_r0_answers.py collect --arm P --locomo locomo10.json --out C-P.json.gz
    python scripts/aml_w0_r0_answers.py merge --a C-A.json.gz --p C-P.json.gz --out C-AP.json.gz
    python scripts/aml_w0_r0_answers.py run --collected C-AP.json.gz --aml-repo AML --locomo ... --draw ... --out A.jsonl
    python scripts/aml_w0_r0_answers.py score --answers A.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aml_w2w4_answers as w2w4  # noqa: E402

from recall_aml.models import SearchItem  # noqa: E402

ARMS = ("A", "A2", "P")
#: Which collect each answered arm reads. A2 reads A's: same retrieval, answered again.
RETRIEVAL = {"A": "A", "A2": "A", "P": "P"}
VARIANTS = {"A": "C9_routed_specialists_grounded_graph_atomic", "P": "C9_qwen8b_proxy"}
GUARD_POINTS = -1.5
NOISE_BAND_POINTS = 2.0
_B_RENDER = w2w4.render


def merge(a: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """One row per question present in both collects, holding each retrieval arm's items.

    The router reads only the query, so both arms must route every question alike; a question they
    route differently is counted and kept (routes are reported, never silently repaired).
    """
    rows_p = {row["id"]: row for row in p["rows"]}
    rows: list[dict[str, Any]] = []
    route_mismatches = 0
    for row in a["rows"]:
        other = rows_p.get(row["id"])
        if other is None:
            continue
        route_mismatches += int(row["route"] != other["route"])
        rows.append({
            "id": row["id"], "category": row["category"], "route": row["route"],
            "items": row["items"], "items_by_arm": {"A": row["items"], "P": other["items"]},
        })
    return {"dataset": a["dataset"], "variants": {"A": a.get("variant"), "P": p.get("variant")},
            "route_mismatches": route_mismatches, "rows": rows}


def render(arm: str, row: dict[str, Any]) -> list[SearchItem]:
    """``arm``'s retrieval, rendered as the served C9 renders it (W2/W4's arm B)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return _B_RENDER("B", {**row, "items": row["items_by_arm"][RETRIEVAL[arm]]})


REPRODUCE_POINTS = 1.0


def turn_hit_at_10(rows: dict[str, dict[str, Any]], questions: dict[str, Any], ids: Sequence[str]) -> float:
    """turn_hit@10 in points over ``ids``, by W0's own scorer on these rows' stored items."""
    from aml_w0_embedding_compare import score_items

    hits = [score_items(rows[i]["items"], questions[i])["turn_hit@10"] for i in ids]
    return round(100 * sum(hits) / len(hits), 2)


def reproduction(
    r0: dict[str, dict[str, Any]], w0: dict[str, dict[str, Any]], questions: dict[str, Any], draw: Sequence[str]
) -> dict[str, Any]:
    """The apparatus check: on the draw's questions that carry gold turns, each arm's R0 collect
    must give turn_hit@10 within `REPRODUCE_POINTS` of the same arm's W0 collect."""
    out: dict[str, Any] = {}
    for arm in ("A", "P"):
        ids = [i for i in draw if i in r0[arm] and i in w0[arm] and questions[i].gold_turns]
        r0_hit = turn_hit_at_10(r0[arm], questions, ids)
        w0_hit = round(100 * sum(w0[arm][i]["served"]["turn_hit@10"] for i in ids) / len(ids), 2)
        out[arm] = {"n": len(ids), "r0": r0_hit, "w0": w0_hit, "ok": abs(r0_hit - w0_hit) <= REPRODUCE_POINTS}
    return out


def verdict(p_minus_a: dict[str, Any], a2_minus_a: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered reading of the two contrasts, its conditions applied in the order written.

    The two conditions overlap when the whole interval lies between -1.5 and 0: the loss is then
    real but bounded under 1.5 points. Written order makes that "does not reach"; the overlap is
    flagged as ``small_loss_detected`` so it is never hidden.
    """
    low, high = p_minus_a["ci95_points"]
    if low > GUARD_POINTS:
        reach = "does_not_reach_the_answer"
    elif high < 0:
        reach = "reaches_the_answer"
    else:
        reach = "inconclusive"
    noise_ok = abs(a2_minus_a["diff_points"]) <= NOISE_BAND_POINTS
    return {"reach": reach, "small_loss_detected": high < 0, "noise_floor_ok": noise_ok, "void": not noise_ok}


def score(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, dict[str, bool]] = {}
    unparsed: Counter[str] = Counter()
    answered: Counter[str] = Counter()
    for record in records:
        labels.setdefault(record["id"], {})[record["arm"]] = record["label"] == "CORRECT"
        unparsed[record["arm"]] += int(record["label"] == "UNPARSED")
        answered[record["arm"]] += 1
    out: dict[str, Any] = {"answered": dict(answered), "unparsed_per_arm": dict(unparsed), "contrasts": {}}
    for arm, control in (("P", "A"), ("A2", "A")):
        keys = sorted(k for k, v in labels.items() if arm in v and control in v)
        out["contrasts"][f"{arm}-vs-{control}"] = w2w4.paired([labels[k][control] for k in keys], [labels[k][arm] for k in keys])
    out["verdict"] = verdict(out["contrasts"]["P-vs-A"], out["contrasts"]["A2-vs-A"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--arm", choices=tuple(VARIANTS), required=True)
    col.add_argument("--locomo", type=Path, required=True)
    col.add_argument("--run-id", default=datetime.now().strftime("%Y%m%dT%H%M%S"))
    col.add_argument("--out", type=Path, required=True)
    mg = sub.add_parser("merge")
    mg.add_argument("--a", type=Path, required=True)
    mg.add_argument("--p", type=Path, required=True)
    mg.add_argument("--out", type=Path, required=True)
    ru = sub.add_parser("run")
    ru.add_argument("--collected", type=Path, required=True)
    ru.add_argument("--aml-repo", type=Path, required=True)
    ru.add_argument("--locomo", type=Path, required=True)
    ru.add_argument("--draw", type=Path, required=True)
    ru.add_argument("--workers", type=int, default=2)
    ru.add_argument("--max-usd", type=float, default=6.0)
    ru.add_argument("--out", type=Path, required=True)
    ck = sub.add_parser("check", help="apparatus: R0's collects reproduce W0's turn_hit@10 on the draw")
    ck.add_argument("--a", type=Path, required=True)
    ck.add_argument("--p", type=Path, required=True)
    ck.add_argument("--w0-a", type=Path, required=True)
    ck.add_argument("--w0-p", type=Path, required=True)
    ck.add_argument("--locomo", type=Path, required=True)
    ck.add_argument("--draw", type=Path, required=True)
    ck.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--answers", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "check":
        from aml_w0_embedding_compare import load_locomo

        def rows_of(path: Path, key: str) -> dict[str, dict[str, Any]]:
            return {str(r[key]): r for r in json.loads(gzip.decompress(path.read_bytes()))["rows"]}

        questions = {q.question_id: q for q in load_locomo(args.locomo, "r0-check", None).questions}
        result = reproduction(
            {"A": rows_of(args.a, "id"), "P": rows_of(args.p, "id")},
            {"A": rows_of(args.w0_a, "question_id"), "P": rows_of(args.w0_p, "question_id")},
            questions, [str(i) for i in json.loads(args.draw.read_text(encoding="utf-8"))["ids"]],
        )
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0 if all(v["ok"] for v in result.values()) else 1
    if args.mode == "collect":
        # The collect refuses any served variant but the one this arm names (its check reads w2w4.C9).
        w2w4.C9 = VARIANTS[args.arm]
        result = w2w4.collect(argparse.Namespace(dataset="locomo", locomo=args.locomo, run_id=f"r0{args.arm}-{args.run_id}",
                                                 x1_data_dir=None, x1_draw=None, out=args.out))
        args.out.write_bytes(gzip.compress(json.dumps(result).encode("utf-8")))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    elif args.mode == "merge":
        merged = merge(json.loads(gzip.decompress(args.a.read_bytes())), json.loads(gzip.decompress(args.p.read_bytes())))
        args.out.write_bytes(gzip.compress(json.dumps(merged).encode("utf-8")))
        print(json.dumps({"rows": len(merged["rows"]), "route_mismatches": merged["route_mismatches"], "variants": merged["variants"]}))
    elif args.mode == "run":
        w2w4.render = render  # the answer loop looks it up at call time
        w2w4.run(argparse.Namespace(dataset="locomo", collected=args.collected, aml_repo=args.aml_repo, locomo=args.locomo,
                                    draw=args.draw, lme_data=None, arms=",".join(ARMS), workers=args.workers,
                                    max_usd=args.max_usd, out=args.out))
    else:
        records = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score(records)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
