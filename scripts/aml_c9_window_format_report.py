"""Summaries for docs/preregistrations/2026-09-25-aml-c9-window-format.md arm H, run on VPS3 in reader-dates/."""

import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "code/scripts")
from aml_locomo_loss_diagnosis import (  # noqa: E402
    load_collected,
    product_dated,
    qa_index,
    render_memories,
    served_items,
)

wf = Path("wf")
qas = qa_index(json.loads(Path("locomo10.json").read_bytes()))
report = {}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


for arm in ("Ap", "Bp", "H"):
    answers = lines(wf / f"answers-{arm}.jsonl")
    judged = lines(wf / f"judged-{arm}.jsonl")
    ids_a = [r["id"] for r in answers]
    ids_j = [r["id"] for r in judged]
    by_id = {r["id"]: r for r in judged}
    acc = {}
    for cat in (None, 1, 2, 3, 4):
        members = [i for i in by_id if cat is None or qas[i]["category"] == cat]
        acc["all" if cat is None else str(cat)] = (
            round(100 * sum(by_id[i]["label"] == "CORRECT" for i in members) / len(members), 2),
            len(members),
        )
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in answers + judged)
    report[arm] = {
        "answer_rows": len(answers),
        "answer_unique": len(set(ids_a)),
        "judge_rows": len(judged),
        "judge_unique": len(set(ids_j)),
        "unparsed": sum(r["label"] == "UNPARSED" for r in by_id.values()),
        "answer_models": dict(collections.Counter(r.get("model") for r in answers)),
        "judge_models": dict(collections.Counter(r.get("model") for r in judged)),
        "reader_views": dict(collections.Counter(r.get("reader_view") for r in answers)),
        "max_prompt_chars": max(r["prompt_chars"] for r in answers),
        "cost_usd": round(cost, 3),
        "accuracy": acc,
    }

check = lines(wf / "judgecheck.jsonl")
report["judgecheck"] = (sum(r["label"] == "CORRECT" for r in {r["id"]: r for r in check}.values()),
                        len({r["id"] for r in check}))

collected = load_collected(Path("collected-S.json.gz"))
by_id = {r["id"]: r for r in collected["rows"]}
sample = random.Random(0).sample(sorted(by_id), 20)
header = re.compile(r"^- \[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\] ", re.M)
iso = re.compile(r"^- \[\d{4}-\d{2}-\d{2}T", re.M)
views = {"Ap": 0, "Bp": 0, "H": 0}
for ident in sample:
    items = served_items(by_id[ident]["items"], drop_compiled=False)
    views["Ap"] += bool(iso.search(render_memories(items, dated=True)))
    views["Bp"] += bool(header.search(render_memories(items, dated=False)) or iso.search(render_memories(items, dated=False)))
    views["H"] += bool(header.search(render_memories(product_dated(items), dated=False)))
report["view_check_of_20"] = views

for name in ("H-vs-Bp", "H-vs-Ap", "Bp-vs-Ap"):
    report[name] = json.loads((wf / f"compare-{name}.json").read_text(encoding="utf-8"))
print(json.dumps(report, indent=1))
