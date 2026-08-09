"""Independent verification of the MTRAG Task A dev run.

Recomputes nDCG@5 from the raw prediction files and the released qrels, using a scorer written
here rather than the harness's own, so an error inside `score_predictions` cannot agree with
itself. Also checks the predictions are structurally sane: right count, unique task ids, no
duplicate document ids inside one ranking, nothing empty, and every arm's file distinct.
"""

import collections
import csv
import glob
import json
import math

D = "/var/tmp/mtrag_taskA_dev_20260807"
ROOT = "/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark"
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")


def load_qrels():
    qrels = collections.defaultdict(set)
    for dom in DOMAINS:
        path = f"{ROOT}/mtrag-human/retrieval_tasks/{dom}/qrels/dev.tsv"
        with open(path, encoding="utf-8") as fh:
            next(fh)
            for row in csv.reader(fh, delimiter="\t"):
                if row and int(row[2]) > 0:
                    qrels[row[0]].add(row[1])
    return qrels


def ndcg(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def recall(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


print("=== structural checks ===")
print("arm               preds  uniq_ids  ctx_min ctx_med ctx_max  dup  empty  sha[:10]")
shas = {}
for p in sorted(glob.glob(D + "/*.predictions.jsonl")):
    name = p.split("/")[-1].replace(".predictions.jsonl", "")
    n = 0
    ids = set()
    lens = []
    dup = 0
    empty = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            ids.add(r["task_id"])
            docs = [c["document_id"] for c in r.get("contexts", [])]
            lens.append(len(docs))
            if len(docs) != len(set(docs)):
                dup += 1
            if not docs:
                empty += 1
    lens.sort()
    meta = json.load(open(p.replace(".predictions.jsonl", ".metrics.json")))
    shas[name] = meta["prediction_sha256"]
    print(
        "{:17} {:<6} {:<9} {:<7} {:<7} {:<8} {:<4} {:<6} {}".format(
            name, n, len(ids), lens[0], lens[len(lens) // 2], lens[-1], dup, empty,
            meta["prediction_sha256"][:10],
        )
    )
print("every arm's prediction file distinct:", len(set(shas.values())) == len(shas))

print()
print("=== independent rescoring (scorer written here, not the harness's) ===")
qrels = load_qrels()
print("judged queries in qrels:", len(qrels))
print("arm               n_judged  nDCG@5_mine  nDCG@5_harness  delta      R@100_mine  R@100_harness  delta")
for p in sorted(glob.glob(D + "/*.predictions.jsonl")):
    name = p.split("/")[-1].replace(".predictions.jsonl", "")
    tot5 = tot100 = 0.0
    cnt = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rel = qrels.get(r["task_id"])
            if not rel:
                continue
            ranked = [c["document_id"] for c in r["contexts"]]
            tot5 += ndcg(ranked, rel, 5)
            tot100 += recall(ranked, rel, 100)
            cnt += 1
    meta = json.load(open(p.replace(".predictions.jsonl", ".metrics.json")))
    o = meta["scores"]["overall"]
    mine5, mine100 = tot5 / cnt, tot100 / cnt
    print(
        "{:17} {:<9} {:<12.4f} {:<15.4f} {:+.6f}  {:<11.4f} {:<14.4f} {:+.6f}".format(
            name, cnt, mine5, o["nDCG@5"], mine5 - o["nDCG@5"],
            mine100, o["Recall@100"], mine100 - o["Recall@100"],
        )
    )

print()
print("=== overlap between arms (are they actually different rankings?) ===")
top5 = {}
for p in sorted(glob.glob(D + "/*.predictions.jsonl")):
    name = p.split("/")[-1].replace(".predictions.jsonl", "")
    m = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            m[r["task_id"]] = tuple(c["document_id"] for c in r["contexts"][:5])
    top5[name] = m
base = "hybrid_lexical"
for name, m in top5.items():
    if name == base:
        continue
    shared = [t for t in m if t in top5[base]]
    identical = sum(1 for t in shared if m[t] == top5[base][t])
    print("  {:17} identical top-5 vs control: {}/{} ({:.1%})".format(
        name, identical, len(shared), identical / len(shared)))
