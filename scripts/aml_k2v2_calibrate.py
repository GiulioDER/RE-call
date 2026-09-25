"""Fix K-2 v2's similarity threshold tau on LongMemEval knowledge-update pairs, before any scoring.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-same-subject-adjacency-v2.md, steps 1 to 7
and apparatus check 1.

Every session is rendered as C9 renders an Add with ``content_only_windows`` (message contents
joined by single spaces) and cut by C9's own ``word_windows`` at C9's 160/120. Windows are embedded
by the same registered profile C9's primary index uses (``voyage-code-4-v1``, document mode).
Positives pair the evidence windows of a question's earliest and latest evidence sessions;
negatives pair that earliest evidence window with random windows from other sessions on other
dates, and random windows from two different non-evidence sessions on different dates. tau is the
99.75th percentile of the negative cosines. Voyage only; no OpenRouter call.

    python scripts/aml_k2v2_calibrate.py --lme longmemeval_s_cleaned.json --out calibration.json
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall.embeddings import resolve_registered_embedder  # noqa: E402
from recall_aml.code4 import word_windows  # noqa: E402

SEED = 20260925
REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
WINDOW, STRIDE = 160, 120
NEGATIVE_LINK_RATE = 0.0025
NEGATIVES_EACH_KIND = 20
MIN_POSITIVE_RECALL = 0.20
PLANNED_POSITIVES, PLANNED_NEGATIVES = 78, 78 * 2 * NEGATIVES_EACH_KIND


def session_day(raw: str) -> date:
    """LongMemEval dates look like ``2023/05/25 (Thu) 20:21``."""
    return datetime.strptime(raw.split(" (")[0], "%Y/%m/%d").date()


def windows_with_evidence(session: list[dict[str, Any]]) -> tuple[list[str], int | None]:
    """C9's content-only windows of one session, and the index of the first evidence window."""
    words_before = 0
    evidence_start: int | None = None
    for turn in session:
        if turn.get("has_answer") and evidence_start is None:
            evidence_start = words_before
        words_before += len(str(turn["content"]).split())
    windows = word_windows(" ".join(str(turn["content"]) for turn in session), size=WINDOW, stride=STRIDE)
    if evidence_start is None:
        return windows, None
    return windows, min(evidence_start // STRIDE, len(windows) - 1)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.lme.read_bytes()
    questions = [q for q in json.loads(raw) if q["question_type"] == "knowledge-update"]
    rng = random.Random(SEED)

    texts: list[str] = []
    index: dict[str, int] = {}

    def slot(text: str) -> int:
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
        return index[text]

    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int, str]] = []
    skipped: list[str] = []
    for q in questions:
        sessions = list(zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]))
        evidence_ids = set(q["answer_session_ids"])
        evidence = sorted(
            ((session_day(day), sid, turns) for sid, day, turns in sessions if sid in evidence_ids),
            key=lambda row: (row[0], row[1]),
        )
        found = [(day, *windows_with_evidence(turns)) for day, _, turns in evidence]
        found = [(day, wins, at) for day, wins, at in found if at is not None]
        if len(found) < 2 or found[0][0] == found[-1][0]:
            skipped.append(q["question_id"])
            continue
        first_day, first_windows, first_at = found[0]
        _, last_windows, last_at = found[-1]
        anchor = slot(first_windows[first_at])
        positives.append((anchor, slot(last_windows[last_at])))
        others = [
            (session_day(day), word_windows(" ".join(str(t["content"]) for t in turns), size=WINDOW, stride=STRIDE))
            for sid, day, turns in sessions
            if sid not in evidence_ids and turns
        ]
        other_days = [(day, wins) for day, wins in others if day != first_day]
        for _ in range(NEGATIVES_EACH_KIND):
            day, wins = rng.choice(other_days)
            negatives.append((anchor, slot(rng.choice(wins)), "evidence_vs_other"))
        for _ in range(NEGATIVES_EACH_KIND):
            (day_a, wins_a), (day_b, wins_b) = rng.sample(others, 2)
            while day_a == day_b:
                (day_a, wins_a), (day_b, wins_b) = rng.sample(others, 2)
            negatives.append((slot(rng.choice(wins_a)), slot(rng.choice(wins_b)), "other_vs_other"))

    counts_ok = len(positives) >= 0.9 * PLANNED_POSITIVES and len(negatives) >= 0.9 * PLANNED_NEGATIVES
    report: dict[str, Any] = {
        "preregistration": "docs/preregistrations/2026-09-25-aml-c9-same-subject-adjacency-v2.md",
        "revision": REVISION,
        "lme_sha256": hashlib.sha256(raw).hexdigest(),
        "knowledge_update_questions": len(questions),
        "skipped_questions": skipped,
        "positives": len(positives),
        "negatives": len(negatives),
        "windows_embedded": len(texts),
        "apparatus_check_1_counts_ok": counts_ok,
    }
    if not counts_ok:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit("apparatus check 1 failed: too few pairs; tau not fixed")

    embedder = resolve_registered_embedder("voyage-code-4-v1", {"VOYAGE_API_KEY": os.environ["VOYAGE_API_KEY"]})
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        vectors.extend(embedder.embed_passages(texts[start : start + 64]))

    positive_cos = sorted(cosine(vectors[a], vectors[b]) for a, b in positives)
    negative_cos = sorted(cosine(vectors[a], vectors[b]) for a, b, _ in negatives)
    tau = negative_cos[math.ceil((1 - NEGATIVE_LINK_RATE) * len(negative_cos)) - 1]
    recall = sum(value >= tau for value in positive_cos) / len(positive_cos)

    def quantiles(values: list[float]) -> dict[str, float]:
        return {f"p{q}": round(values[min(len(values) - 1, int(q / 100 * len(values)))], 4) for q in (10, 50, 90, 99)}

    report.update(
        {
            "tau": round(tau, 6),
            "negative_link_rate_at_tau": sum(value >= tau for value in negative_cos) / len(negative_cos),
            "positive_recall_at_tau": recall,
            "stop_rule_met": recall < MIN_POSITIVE_RECALL,
            "positive_cosine": quantiles(positive_cos),
            "negative_cosine": quantiles(negative_cos),
            "negative_cosine_by_kind": {
                kind: quantiles(sorted(cosine(vectors[a], vectors[b]) for a, b, k in negatives if k == kind))
                for kind in ("evidence_vs_other", "other_vs_other")
            },
        }
    )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
