"""Stage 0 of the T-2 pre-registration: how often a question names a time T-2 can resolve.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-query-time-leg.md (and its amendment 1).
No model, no service, no network. For each set, the share of questions for which
``recall_aml.temporal_query.query_time_range`` returns a range, with the anchor a service would
have (the corpus's latest stored date for that user), and whether that range reaches any date the
corpus actually holds. On LongMemEval it also measures the anchor proxy against the true
``question_date``. A range is RELATIVE when moving the anchor by 400 days moves it.

BEAM's ``probing_questions`` cell is parsed by the repository's own size-capped parser
(``benchmarks.beam.dataset._parse_probing``) rather than a second copy of it.

    python scripts/aml_t2_census.py --locomo locomo10.json --beam 100K.parquet \\
        --lme longmemeval_s_cleaned.json --out census.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beam.dataset import _parse_probing  # noqa: E402
from recall_aml.temporal_query import query_time_range  # noqa: E402

_LOCOMO_DATE = re.compile(r"(\d{1,2}) ([A-Za-z]+),? (\d{4})")


def _overlaps(span: tuple[date, date], days: set[date]) -> bool:
    return any(span[0] <= day <= span[1] for day in days)


def _relative(question: str, anchor: date) -> bool:
    return query_time_range(question, anchor) != query_time_range(question, anchor + timedelta(days=400))


def _share(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ranged = [row for row in rows if row["range"] is not None]
    overlap_known = [row for row in ranged if row["overlap"] is not None]
    return {
        "n": n,
        "with_range": len(ranged),
        "share_with_range": len(ranged) / n if n else None,
        "relative": sum(row["relative"] for row in ranged),
        "range_reaches_a_stored_date": sum(row["overlap"] for row in overlap_known) if overlap_known else None,
    }


def locomo(path: Path) -> dict[str, Any]:
    rows = []
    for conversation in json.loads(path.read_text(encoding="utf-8")):
        sessions = conversation["conversation"]
        days = set()
        for key, value in sessions.items():
            if key.endswith("_date_time") and isinstance(value, str) and (m := _LOCOMO_DATE.search(value)):
                days.add(datetime.strptime(f"{m[1]} {m[2][:3]} {m[3]}", "%d %b %Y").date())
        anchor = max(days)
        for qa in conversation["qa"]:
            span = query_time_range(str(qa["question"]), anchor)
            rows.append(
                {
                    "category": qa.get("category"),
                    "range": span,
                    "relative": span is not None and _relative(str(qa["question"]), anchor),
                    "overlap": None if span is None else _overlaps(span, days),
                }
            )
    return {"all": _share(rows), "category_2": _share([row for row in rows if row["category"] == 2])}


def beam(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pq.read_table(path).to_pylist():
        anchors = [
            datetime.strptime(turn["time_anchor"], "%B-%d-%Y").date()
            for session in record["chat"]
            for turn in session
            if turn.get("time_anchor")
        ]
        anchor = max(anchors) if anchors else date(2024, 12, 31)
        for kind, items in _parse_probing(record["probing_questions"]).items():
            for item in items:
                span = query_time_range(str(item["question"]), anchor)
                rows_by_type[kind].append(
                    {
                        "range": span,
                        "relative": span is not None and _relative(str(item["question"]), anchor),
                        "overlap": None if span is None or not anchors else _overlaps(span, set(anchors)),
                    }
                )
    everything = [row for rows in rows_by_type.values() for row in rows]
    return {"all": _share(everything), "by_type": {kind: _share(rows) for kind, rows in sorted(rows_by_type.items())}}


def _lme_day(raw: str) -> date:
    return datetime.strptime(raw.split(" (")[0], "%Y/%m/%d").date()


def longmemeval(path: Path) -> dict[str, Any]:
    rows_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gaps: list[int] = []
    proxy_ok = proxy_total = 0
    for q in json.loads(path.read_bytes()):
        days = {_lme_day(value) for value in q["haystack_dates"]}
        proxy = max(days)
        truth = _lme_day(q["question_date"])
        gaps.append((truth - proxy).days)
        question = str(q["question"])
        span = query_time_range(question, proxy)
        relative = span is not None and _relative(question, proxy)
        if relative and span is not None:
            proxy_total += 1
            true_span = query_time_range(question, truth)
            proxy_ok += true_span is not None and span[0] <= true_span[0] and true_span[1] <= span[1]
        rows_by_type[q["question_type"]].append(
            {"range": span, "relative": relative, "overlap": None if span is None else _overlaps(span, days)}
        )
    everything = [row for rows in rows_by_type.values() for row in rows]
    gaps.sort()
    return {
        "all": _share(everything),
        "by_type": {kind: _share(rows) for kind, rows in sorted(rows_by_type.items())},
        "anchor_gap_days": {"median": statistics.median(gaps), "p90": gaps[int(0.9 * len(gaps))], "max": gaps[-1]},
        "relative_questions": proxy_total,
        "proxy_range_contains_true_range": proxy_ok / proxy_total if proxy_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--beam", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"locomo": locomo(args.locomo), "beam_100k": beam(args.beam), "longmemeval_s": longmemeval(args.lme)}
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
