"""X-1 LongMemEval-S baseline from K-1 Stage 1's R (C9) and R2 (C9') answers, plus evidence-session
Recall@10 from Stage B's stored items. No model call."""
import json
import sys
from collections import defaultdict

answers, stageb, data = sys.argv[1:4]
labels = defaultdict(dict)
types = {}
for line in open(answers, encoding="utf-8").read().split("\n"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r["arm"] in ("R", "R2"):
        labels[r["arm"]][r["id"]] = r["label"] == "CORRECT"
        types[r["id"]] = r["type"]
questions = {q["question_id"]: q for q in json.load(open(data, encoding="utf-8"))}
recall10 = {}
for line in open(stageb, encoding="utf-8").read().split("\n"):
    if not line.strip():
        continue
    row = json.loads(line)
    for s in row["searches"]:
        gold = set(questions[s["question_id"]]["answer_session_ids"])
        top = {i["session_id"] for i in s["items"][:10]}
        recall10[s["question_id"]] = len(gold & top) / len(gold) if gold else None
out = {"n": {a: len(v) for a, v in labels.items()}}
ids = sorted(labels["R"])
assert ids == sorted(labels["R2"]), "R and R2 answered different questions"
for arm in ("R", "R2"):
    out[f"{arm}_all"] = round(sum(labels[arm][i] for i in ids) / len(ids), 4)
by_type = defaultdict(list)
for i in ids:
    by_type[types[i]].append(i)
out["per_type"] = {
    t: {"n": len(v),
        "C9": round(sum(labels["R"][i] for i in v) / len(v), 4),
        "C9prime": round(sum(labels["R2"][i] for i in v) / len(v), 4),
        "recall10": round(sum(recall10[i] for i in v if recall10.get(i) is not None)
                          / max(1, sum(recall10.get(i) is not None for i in v)), 4)}
    for t, v in sorted(by_type.items())
}
scored = [recall10[i] for i in ids if recall10.get(i) is not None]
out["recall10_all"] = round(sum(scored) / len(scored), 4)
out["recall10_n"] = len(scored)
out["abstention_or_no_gold"] = sum(recall10.get(i) is None for i in ids)
print(json.dumps(out, indent=2))
