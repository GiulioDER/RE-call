"""Search for a query-time triage signal HONESTLY: discover on train, report on test.

The registered sweep found seven hand-picked scalars all inside the noise band, best AUC 0.537
against a random control of 0.472. That is a null for those seven features, not proof that no
signal exists. This module widens the search, and the widening is exactly what makes a naive
result untrustworthy: search hard enough over one dataset and something always scores well.

So the protocol is fixed here, before any number is read:

1. **Split 500 questions into train and test** by a seeded hash of the question id, roughly half
   each. The split is on the QUESTION, so no question contributes to both.
2. **Rank every feature on TRAIN only.**
3. **Report the winner's AUC on TEST**, which it has never seen. That number is the honest one.
4. The gap between the train AUC and the test AUC IS the selection inflation, and it is printed
   rather than hidden.

⛔ Nothing here is a pre-registered result. This is hypothesis generation. A feature that survives
to a good TEST AUC earns a new pre-registration and a fresh dataset; it does not earn a claim.

Three label variants, because the registered one conflates two different failures:

- `missed_any`: the registered label. Gold not fully in the top-8. Mixes "not retrieved at all"
  with "retrieved but ranked 9th".
- `recoverable`: gold is in the POOL but not the top-8. This is the one depth can fix, and it is
  the label a depth-triage should actually predict.
- `absent`: gold is not in the pool at all. Depth cannot fix these; only better retrieval can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.analyse_triage import auc


def _split(question_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()
    return "train" if int(digest[:8], 16) % 2 == 0 else "test"


def build_features(row: Mapping[str, Any], question: str, top_k: int) -> dict[str, float]:
    """Everything computable BEFORE an answer exists. Nothing may consult gold or the answer."""
    ranked = row["ranked"]
    scores = [float(h["score"]) for h in ranked]
    top = scores[:top_k]
    text = question.lower()
    words = text.split()

    def at(i: int) -> float:
        return scores[i] if i < len(scores) else 0.0

    total = sum(scores) or 1.0
    probabilities = [s / total for s in scores if s > 0]
    entropy = -sum(p * math.log(p) for p in probabilities) if probabilities else 0.0

    docs_top8 = [h["doc_id"] for h in ranked[:top_k]]
    docs_top50 = [h["doc_id"] for h in ranked[:50]]

    feats = {
        # --- shape of the score curve, beyond top-1 and decay ---
        "top1": at(0),
        "score_at_8": at(7),
        "score_at_20": at(19),
        "score_at_50": at(49),
        "score_at_199": at(len(scores) - 1),
        "decay_1_to_8": at(0) - at(7),
        "decay_8_to_50": at(7) - at(49),
        "decay_1_to_50": at(0) - at(49),
        "ratio_8_over_1": at(7) / at(0) if at(0) else 0.0,
        "ratio_50_over_8": at(49) / at(7) if at(7) else 0.0,
        "mean_top8": statistics.fmean(top) if top else 0.0,
        "stdev_top8": statistics.pstdev(top) if len(top) > 1 else 0.0,
        "entropy_pool": entropy,
        "margin_1_2": at(0) - at(1),
        # --- how concentrated the pool is on few documents ---
        "distinct_docs_top8": float(len(set(docs_top8))),
        "distinct_docs_top50": float(len(set(docs_top50))),
        "chunks_per_doc_top50": len(docs_top50) / max(1, len(set(docs_top50))),
        "top_doc_share_top50": max(
            (docs_top50.count(d) for d in set(docs_top50)), default=0
        ) / max(1, len(docs_top50)),
        # --- query shape ---
        "query_chars": float(len(question)),
        "query_words": float(len(words)),
        "n_clauses": float(text.count(",") + text.count(" and ") + text.count(";")),
        "n_question_marks": float(text.count("?")),
        "has_multi_ask": 1.0 if (" and what" in text or " and who" in text
                                 or " and when" in text or " versus " in text
                                 or " vs " in text) else 0.0,
        "n_quoted": float(text.count('"') + text.count("`")),
        "n_digits": float(sum(c.isdigit() for c in text)),
        "gap_warning": 1.0 if row.get("gap_warning") else 0.0,
    }
    # Interactions: the registered sweep tested only scalars, and the question-type spread
    # suggested the signal is not linear in any one of them.
    feats["decay_x_distinct"] = feats["decay_1_to_8"] * feats["distinct_docs_top8"]
    feats["decay_over_top1"] = feats["decay_1_to_8"] / feats["top1"] if feats["top1"] else 0.0
    feats["words_x_distinct"] = feats["query_words"] * feats["distinct_docs_top8"]
    feats["flatness"] = -feats["stdev_top8"] / feats["mean_top8"] if feats["mean_top8"] else 0.0
    return feats


def _labels(row: Mapping[str, Any], top_k: int) -> dict[str, int | None]:
    gold = set(row["expected_doc_ids"])
    if not gold:
        return {"missed_any": None, "recoverable": None, "absent": None}
    at_k = {h["doc_id"] for h in row["ranked"][:top_k]}
    pool = {h["doc_id"] for h in row["ranked"]}
    missing = gold - at_k
    return {
        "missed_any": int(bool(missing)),
        "recoverable": int(bool(missing) and missing <= pool),
        "absent": int(bool(gold - pool)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", default="missed_any",
                        choices=("missed_any", "recoverable", "absent"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = json.loads(args.retrieval.read_text(encoding="utf-8"))["evidence"]
    questions = {
        json.loads(line)["question_id"]: json.loads(line)["question"]
        for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()
    }

    data: dict[str, list[tuple[dict[str, float], int]]] = {"train": [], "test": []}
    for qid, row in rows.items():
        label = _labels(row, args.top_k)[args.label]
        if label is None:
            continue
        feats = build_features(row, questions.get(qid, ""), args.top_k)
        data[_split(qid, args.seed)].append((feats, label))

    for part, items in data.items():
        positives = sum(label for _, label in items)
        print(f"{part}: n={len(items)}  positives={positives} ({positives/len(items):.1%})")
    print(f"label = {args.label}")
    print()

    names = sorted(data["train"][0][0])
    train_auc = {
        name: auc([f[name] for f, _ in data["train"]], [y for _, y in data["train"]])
        for name in names
    }
    # Ranked by DISTANCE from 0.5, because an anti-predictive feature is a predictive one with
    # its sign flipped, and hiding that would repeat the score_decay mistake in reverse.
    ordered = sorted(train_auc.items(), key=lambda kv: -abs(kv[1] - 0.5))

    print(f"{'feature':24} {'TRAIN auc':>10} {'|dist|':>7}   {'TEST auc':>9}  {'inflation':>9}")
    for name, tr in ordered[:12]:
        te = auc([f[name] for f, _ in data["test"]], [y for _, y in data["test"]])
        print(f"{name:24} {tr:>10.4f} {abs(tr-0.5):>7.3f}   {te:>9.4f}  {abs(tr-0.5)-abs(te-0.5):>+9.3f}")

    winner = ordered[0][0]
    test_auc = auc([f[winner] for f, _ in data["test"]], [y for _, y in data["test"]])
    n = len(data["test"])
    positives = sum(y for _, y in data["test"])
    se = math.sqrt(0.25 / min(positives, n - positives)) if 0 < positives < n else float("nan")
    print()
    print(f"WINNER on train: {winner}")
    print(f"  honest TEST auc = {test_auc:.4f}   (|dist from 0.5| = {abs(test_auc-0.5):.4f})")
    print(f"  rough test SE   ~ {se:.4f}; anything within ~2 SE of 0.5 is not a signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
