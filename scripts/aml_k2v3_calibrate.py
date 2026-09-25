"""Fix K-2 v3's z0 on one-user LongMemEval lists, before any scoring. Voyage only.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-same-subject-adjacency-v3.md, calibration
steps 1 to 6 and apparatus checks 1 and 3.

For each of the 67 eligible knowledge-update questions, one 30-window list: the evidence windows
of its earliest and latest evidence sessions plus 28 windows drawn from its own non-evidence
sessions, each window dated by its session. Inside a list, over cross-day pairs only: z is a pair's
cosine standardised by that list's cross-day mean and standard deviation, and a pair is a mutual
nearest-neighbour (MNN) pair when each is the other's most similar cross-day partner. Negatives are
MNN pairs other than the evidence pair; z0 is the smallest value leaving at most 0.10 spurious links
per list. The evidence pair links when it is MNN and reaches z0.

    python scripts/aml_k2v3_calibrate.py --lme longmemeval_s_cleaned.json --out calibration-v3.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_k2v2_calibrate import STRIDE, WINDOW, eligible, session_day, windows_with_evidence  # noqa: E402

from recall.embeddings import resolve_registered_embedder  # noqa: E402
from recall_aml.code4 import word_windows  # noqa: E402

SEED = 20260925
LIST_SIZE = 30
SPURIOUS_PER_LIST = 0.10
MIN_POSITIVE_RECALL = 0.20
MIN_CROSS_DAY_PAIRS = 20


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def list_pairs(vectors: list[list[float]], days: list[date]) -> tuple[dict[tuple[int, int], float], set[tuple[int, int]]]:
    """z per cross-day pair of one list, and its mutual-nearest-neighbour pairs (i < j)."""
    cos = {
        (i, j): cosine(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
        if days[i] != days[j]
    }
    mean = statistics.fmean(cos.values())
    sd = statistics.pstdev(cos.values()) or 1.0
    z = {pair: (value - mean) / sd for pair, value in cos.items()}
    nearest: dict[int, int] = {}
    for i in range(len(vectors)):
        partners = [(value, j if a == i else a) for (a, j), value in cos.items() if i in (a, j)]
        if partners:
            nearest[i] = max(partners)[1]
    mutual = {tuple(sorted((i, j))) for i, j in nearest.items() if nearest.get(j) == i}
    return z, mutual  # type: ignore[return-value]


def brute_force_mutual(vectors: list[list[float]], days: list[date]) -> set[tuple[int, int]]:
    """Apparatus check 3: the MNN pairs again, by an independent loop over every item."""
    found = set()
    count = len(vectors)
    for i in range(count):
        best_i, best_i_value = None, -2.0
        for j in range(count):
            if j != i and days[j] != days[i]:
                value = cosine(vectors[i], vectors[j])
                if value > best_i_value:
                    best_i, best_i_value = j, value
        if best_i is None:
            continue
        best_back, best_back_value = None, -2.0
        for k in range(count):
            if k != best_i and days[k] != days[best_i]:
                value = cosine(vectors[best_i], vectors[k])
                if value > best_back_value:
                    best_back, best_back_value = k, value
        if best_back == i:
            found.add((min(i, best_i), max(i, best_i)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.lme.read_bytes()
    questions = [q for q in json.loads(raw) if q["question_type"] == "knowledge-update" and eligible(q)]
    rng = random.Random(SEED)

    lists: list[dict[str, Any]] = []
    for q in questions:
        sessions = list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]))
        evidence_ids = set(q["answer_session_ids"])
        evidence = sorted(
            (
                (session_day(day), sid, *windows_with_evidence(turns))
                for sid, day, turns in sessions
                if sid in evidence_ids
            ),
            key=lambda row: (row[0], row[1]),
        )
        evidence = [row for row in evidence if row[3] is not None]
        first, last = evidence[0], evidence[-1]
        pool = [
            (session_day(day), window)
            for sid, day, turns in sessions
            if sid not in evidence_ids and turns
            for window in word_windows(" ".join(str(t["content"]) for t in turns), size=WINDOW, stride=STRIDE)
        ]
        fillers = rng.sample(pool, min(LIST_SIZE - 2, len(pool)))
        texts = [first[2][first[3]], last[2][last[3]], *(window for _, window in fillers)]
        days = [first[0], last[0], *(day for day, _ in fillers)]
        cross_day = sum(days[i] != days[j] for i in range(len(days)) for j in range(i + 1, len(days)))
        lists.append({"question_id": q["question_id"], "texts": texts, "days": days, "cross_day_pairs": cross_day})

    good = [
        item
        for item in lists
        if len(item["texts"]) == LIST_SIZE and item["days"][0] != item["days"][1] and item["cross_day_pairs"] >= MIN_CROSS_DAY_PAIRS
    ]
    report: dict[str, Any] = {
        "preregistration": "docs/preregistrations/2026-09-25-aml-c9-same-subject-adjacency-v3.md",
        "lme_sha256": hashlib.sha256(raw).hexdigest(),
        "eligible_questions": len(questions),
        "lists_built": len(lists),
        "lists_passing_shape": len(good),
        "apparatus_check_1": len(good) >= 0.9 * len(questions),
    }
    if not report["apparatus_check_1"]:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit("apparatus check 1 failed; z0 not fixed")

    embedder = resolve_registered_embedder("voyage-code-4-v1", {"VOYAGE_API_KEY": os.environ["VOYAGE_API_KEY"]})
    negatives: list[float] = []
    positives: list[dict[str, Any]] = []
    check_3 = True
    for item in good:
        vectors = [list(v) for v in embedder.embed_passages(item["texts"])]
        z, mutual = list_pairs(vectors, item["days"])
        check_3 &= mutual == brute_force_mutual(vectors, item["days"])
        negatives.extend(z[pair] for pair in mutual if pair != (0, 1))
        positives.append({"question_id": item["question_id"], "mutual": (0, 1) in mutual, "z": z[(0, 1)]})

    allowed = math.floor(SPURIOUS_PER_LIST * len(good))
    ranked = sorted(negatives, reverse=True)
    z0 = (ranked[allowed] if allowed < len(ranked) else -math.inf) + 1e-9
    linked = [p for p in positives if p["mutual"] and p["z"] >= z0]
    recall = len(linked) / len(positives)
    report.update(
        {
            "apparatus_check_3_mnn_recomputed_equal": check_3,
            "negative_mnn_pairs": len(negatives),
            "negative_mnn_pairs_per_list": len(negatives) / len(good),
            "z0": round(z0, 6),
            "spurious_links_per_list_at_z0": sum(value >= z0 for value in negatives) / len(good),
            "evidence_pairs_mutual": sum(p["mutual"] for p in positives) / len(positives),
            "evidence_pairs_linked_at_z0": recall,
            "stop_rule_met": recall < MIN_POSITIVE_RECALL,
            "evidence_z_quantiles": {
                f"p{q}": round(sorted(p["z"] for p in positives)[min(len(positives) - 1, int(q / 100 * len(positives)))], 3)
                for q in (10, 50, 90)
            },
            "negative_mnn_z_quantiles": {
                f"p{q}": round(sorted(negatives)[min(len(negatives) - 1, int(q / 100 * len(negatives)))], 3)
                for q in (10, 50, 90, 99)
            },
        }
    )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
