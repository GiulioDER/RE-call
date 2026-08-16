"""Score the triage pre-registration from a frozen retrieval fixture. No model, no network.

Pre-registered at `results/enterprise_rag/PREREGISTRATION-retrieval-triage.md` (`c8828db`).

Computes, with the metric names the registration fixed:

- `recall_at_k` = gold documents in the top-k / gold documents
- `pool_recovery_rate` = gold documents missed at top-k but present in the pool / gold missed at top-k
- `triage_auc` = ROC AUC of one query-time feature separating questions whose gold was fully
  retrieved from questions whose gold was not

⚠️ Every feature here is computable BEFORE an answer exists. That is the whole point: a signal
that needs the answer cannot triage anything. The label comes from the retrieval fixture's own
gold membership, not from a judge, so it is mechanical and repeatable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC by rank, ties averaged. No library fitting and no threshold tuning.

    `labels` is 1 for the POSITIVE class, which here is "gold was missed", so an AUC above 0.5
    means a higher feature value predicts a miss.
    """
    # ⚠️ NaN FIRST. Tie detection uses `==`, and NaN never equals itself, so NaNs never group
    # and never raise: an all-NaN feature returns 0.0, 0.5 or 1.0 purely by row order, and prints
    # as an ordinary number. sklearn refuses the same input. A NaN cosine round-trips through JSON
    # invisibly, so this is reachable rather than theoretical.
    if any(s != s for s in scores):
        raise ValueError("NaN in scores; refusing to compute an AUC that would be row-order noise")
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        for index in range(i, j + 1):
            ranks[index] = average
        i = j + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return float("nan")
    rank_sum = sum(r for r, (_, label) in zip(ranks, pairs, strict=True) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def features(row: Mapping[str, Any], top_k: int) -> dict[str, float]:
    """Query-time features only. Nothing here may consult the answer or the gold set."""
    ranked = row["ranked"]
    scores = [float(hit["score"]) for hit in ranked[:top_k]]
    top = scores[0] if scores else 0.0
    last = scores[-1] if scores else 0.0
    # `[...]`, never `.get(..., "")`. A missing key used to become an empty string, which made
    # `conjunctions` and `question_words` constant on every row and report as a clean 0.500 null.
    text = str(row["question"]).lower()
    return {
        "neg_top1_score": -top,
        "neg_mean_topk": -(sum(scores) / len(scores)) if scores else 0.0,
        "score_decay": top - last,
        "gap_warning": 1.0 if row.get("gap_warning") else 0.0,
        "distinct_docs_topk": float(len({h["doc_id"] for h in ranked[:top_k]})),
        "query_chars": float(row.get("query_chars", 0)),
        "conjunctions": float(sum(text.count(w) for w in (" and ", " versus ", " vs "))),
        "question_words": float(sum(text.count(w) for w in ("what", "who", "when", "which", "how"))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = json.loads(args.retrieval.read_text(encoding="utf-8"))
    rows = payload["evidence"]
    print(f"questions in fixture: {len(rows)}")

    gold_total = gold_at_k = gold_in_pool = 0
    missed_at_k = missed_but_pooled = 0
    labels: list[int] = []
    feats: list[dict[str, float]] = []
    per_type: dict[str, list[int]] = {}

    for row in rows.values():
        gold = set(row["expected_doc_ids"])
        if not gold:
            continue
        ranked = row["ranked"]
        at_k = {h["doc_id"] for h in ranked[: args.top_k]}
        pool = {h["doc_id"] for h in ranked}
        gold_total += len(gold)
        gold_at_k += len(gold & at_k)
        gold_in_pool += len(gold & pool)
        missing = gold - at_k
        missed_at_k += len(missing)
        missed_but_pooled += len(missing & pool)

        label = 0 if gold <= at_k else 1
        labels.append(label)
        feats.append(features(row, args.top_k))
        per_type.setdefault(row["question_type"], []).append(label)

    if not gold_total:
        raise SystemExit("no row in the fixture has expected_doc_ids; nothing to score")
    print()
    print("=== T2 / T3 / T1 ===")
    print(f"  recall_at_{args.top_k:<4}      = {gold_at_k}/{gold_total} = {gold_at_k/gold_total:.4f}")
    print(f"  recall_at_pool     = {gold_in_pool}/{gold_total} = {gold_in_pool/gold_total:.4f}")
    print(f"  O2 lift            = {(gold_in_pool-gold_at_k)/gold_total:+.4f}  (predicted > +0.10)")
    if missed_at_k:
        print(f"  pool_recovery_rate = {missed_but_pooled}/{missed_at_k} = "
              f"{missed_but_pooled/missed_at_k:.4f}  (T1 predicted 0.45)")

    print()
    print(f"=== T4 / T5: triage AUC, positives = 'gold missed' ({sum(labels)}/{len(labels)}) ===")
    ranked_auc = []
    for name in feats[0]:
        value = auc([f[name] for f in feats], labels)
        ranked_auc.append((value, name))
    rnd = [random.Random(args.seed + i).random() for i in range(len(labels))]
    for value, name in sorted(ranked_auc, reverse=True):
        mark = "  <- shipped gap_warning" if name == "gap_warning" else ""
        print(f"  {name:22} AUC = {value:.4f}{mark}")
    print()
    print(f"  R2 random feature      AUC = {auc(rnd, labels):.4f}  (expected ~0.50)")
    print(f"  R3 label as a feature  AUC = {auc([float(x) for x in labels], labels):.4f}  (expected 1.00)")

    print()
    print("=== O3: miss rate by question type ===")
    for qtype, ls in sorted(per_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {qtype:26} n={len(ls):>3}  gold-missed {sum(ls)/len(ls)*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
