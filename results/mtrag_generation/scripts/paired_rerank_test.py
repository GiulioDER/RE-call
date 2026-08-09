"""Is the Voyage rerank lift real, and does it contradict the closed reranker null?

PAIRED per query, because the two arms answer the SAME 777 queries and an unpaired test would
throw away the pairing and inflate the variance. Bootstrap CI + permutation, per the project's
validation standards, rather than reporting a bare delta.

⚠️ Context that decides how this gets read: the reranker lever is recorded as CLOSED WITH DATA
(BGE-reranker-v2-m3 falsified; two cross-encoders 25x apart burying the same 90 vs 91 of 123
SPLADE-only gold documents), and the conclusion drawn was ARCHITECTURAL: every reranker measured
pools the (query, passage) pair before scoring. Voyage rerank-2.5 pools the pair too, so a large
lift here either falsifies that conclusion or means the earlier null was measured under a
condition this run does not share. The discriminating arm is MiniLM at the SAME candidate_k=100.
"""

import glob
import json
import math
import random
from pathlib import Path

import pytest

__test__ = False

D = "/var/tmp/mtrag_taskA_dev_20260807"
ROOT = "/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark"
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
random.seed(20260808)

if not all(
    Path(f"{ROOT}/mtrag-human/retrieval_tasks/{dom}/qrels/dev.tsv").exists()
    for dom in DOMAINS
):
    pytest.skip(
        "MTRAG qrels checkout is unavailable in this environment",
        allow_module_level=True,
    )


def load_qrels():
    import csv
    from collections import defaultdict

    q = defaultdict(set)
    for dom in DOMAINS:
        with open(f"{ROOT}/mtrag-human/retrieval_tasks/{dom}/qrels/dev.tsv", encoding="utf-8") as fh:
            next(fh)
            for row in csv.reader(fh, delimiter="\t"):
                if row and int(row[2]) > 0:
                    q[row[0]].add(row[1])
    return q


def ndcg(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def per_query(arm, qrels, k=5):
    out = {}
    with open(f"{D}/{arm}.predictions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rel = qrels.get(r["task_id"])
            if rel:
                out[r["task_id"]] = ndcg([c["document_id"] for c in r["contexts"]], rel, k)
    return out


def main() -> None:
    qrels = load_qrels()
    available = sorted(
        p.split("/")[-1].replace(".predictions.jsonl", "") for p in glob.glob(f"{D}/*.predictions.jsonl")
    )
    print("arms available:", available)

    BASE = "hybrid_splade"
    base = per_query(BASE, qrels)

    for arm in available:
        if arm == BASE or "rerank" not in arm and "voyage" not in arm:
            continue
        cur = per_query(arm, qrels)
        ids = sorted(set(base) & set(cur))
        diffs = [cur[i] - base[i] for i in ids]
        n = len(diffs)
        mean = sum(diffs) / n

        boots = []
        for _ in range(10000):
            s = [diffs[random.randrange(n)] for _ in range(n)]
            boots.append(sum(s) / n)
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

        # Permutation: under the null the sign of each paired difference is exchangeable.
        hits = 0
        trials = 10000
        for _ in range(trials):
            t = sum(d if random.random() < 0.5 else -d for d in diffs) / n
            if abs(t) >= abs(mean):
                hits += 1
        p = (hits + 1) / (trials + 1)

        better = sum(1 for d in diffs if d > 1e-12)
        worse = sum(1 for d in diffs if d < -1e-12)
        print()
        print(f"=== {arm}  vs  {BASE}   (paired, n={n}) ===")
        print(f"  mean nDCG@5 delta : {mean:+.4f}")
        print(f"  bootstrap 95% CI  : [{lo:+.4f}, {hi:+.4f}]")
        print(f"  permutation p     : {p:.5f}")
        print(f"  queries better    : {better}   worse: {worse}   unchanged: {n - better - worse}")
        print(f"  crosses zero      : {'YES (not significant)' if lo <= 0 <= hi else 'no'}")


if __name__ == "__main__":
    main()
