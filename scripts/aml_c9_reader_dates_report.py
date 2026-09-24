"""Summaries for docs/preregistrations/2026-09-24-aml-c9-reader-dates.md.

Run from the VPS3 run directory, which holds `code/` (this repository), `locomo10.json`, both
collected files and `out/` from the answer, judge and compare stages. Prints the report JSON.
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "code/scripts")
from aml_locomo_loss_diagnosis import (  # noqa: E402
    load_collected,
    qa_index,
    render_memories,
    served_items,
)

out = Path("out")


def rows(path):
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


answers = {a: rows(out / f"answers-{a}.jsonl") for a in "ABC"}
judged = {a: rows(out / f"judged-{a}.jsonl") for a in "ABC"}
qas = qa_index(json.loads(Path("locomo10.json").read_bytes()))

report = {"counts": {}, "unparsed": {}, "cost_usd": {}, "prompt_chars": {}, "accuracy": {}}
for a in "ABC":
    report["counts"][a] = (len(answers[a]), len(judged[a]))
    report["unparsed"][a] = sum(r["label"] == "UNPARSED" for r in judged[a].values())
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in answers[a].values())
    cost += sum(float((r.get("usage") or {}).get("cost") or 0) for r in judged[a].values())
    report["cost_usd"][a] = round(cost, 3)
    chars = [r["prompt_chars"] for r in answers[a].values()]
    report["prompt_chars"][a] = {"mean": round(sum(chars) / len(chars)), "max": max(chars)}
    acc = {}
    for cat in (None, 1, 2, 3, 4):
        ids = [i for i in judged[a] if cat is None or qas[i]["category"] == cat]
        acc["all" if cat is None else str(cat)] = (
            round(100 * sum(judged[a][i]["label"] == "CORRECT" for i in ids) / len(ids), 2),
            len(ids),
        )
    report["accuracy"][a] = acc
check = rows(out / "judgecheck.jsonl")
report["judgecheck"] = (sum(r["label"] == "CORRECT" for r in check.values()), len(check))
report["judgecheck_cost_usd"] = round(
    sum(float((r.get("usage") or {}).get("cost") or 0) for r in check.values()), 3
)
report["mean_prompt_chars_C_over_B"] = round(
    report["prompt_chars"]["C"]["mean"] / report["prompt_chars"]["B"]["mean"], 3
)

# Apparatus check 4: re-render 20 prompts per arm by seed 0 and look for the timestamp prefix.
collected = {"S": load_collected(Path("collected-S.json.gz")), "T": load_collected(Path("collected-T.json.gz"))}
stamp = re.compile(r"^- \[\d{4}-\d{2}-\d{2}", re.M)
views = {}
for arm, source, dated in (("A", "S", True), ("B", "S", False), ("C", "T", False)):
    by_id = {r["id"]: r for r in collected[source]["rows"]}
    sample = random.Random(0).sample(sorted(by_id), 20)
    hits = 0
    for ident in sample:
        block = render_memories(served_items(by_id[ident]["items"], drop_compiled=False), dated=dated)
        hits += bool(stamp.search(block))
    views[arm] = f"{hits}/20 prompts carry a '- [date' prefix"
report["view_check"] = views

for name in ("B-vs-A", "C-vs-B", "C-vs-A"):
    report[name] = json.loads((out / f"compare-{name}.json").read_text(encoding="utf-8"))
print(json.dumps(report, indent=1))
