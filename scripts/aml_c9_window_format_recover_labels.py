"""Exploratory, post hoc: recover the judge labels DeepSeek wrapped in doubled braces, recompare.

Run on VPS3 in reader-dates/. The pre-registered scoring (UNPARSED counts as WRONG) is not changed;
this only measures how much the formatting quirk moves each comparison.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "code/scripts")
from aml_locomo_loss_diagnosis import load_aml_pipeline, paired, qa_index  # noqa: E402

pipeline = load_aml_pipeline(Path("aml-official"))
qas = qa_index(json.loads(Path("locomo10.json").read_bytes()))


def recovered(row: dict) -> tuple[str, bool]:
    if row["label"] != "UNPARSED":
        return row["label"], False
    text = row["judge_response"].replace("{{", "{").replace("}}", "}")
    try:
        return pipeline.parse_judge_label(text), True
    except (ValueError, json.JSONDecodeError):
        match = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', text, re.IGNORECASE)
        return (match.group(1).upper(), True) if match else ("UNPARSED", True)


labels = {}
for arm in ("Ap", "Bp", "H"):
    parsed = [json.loads(line) for line in Path(f"wf/judged-{arm}.jsonl").read_text().splitlines() if line]
    rows = {row["id"]: row for row in parsed}
    labels[arm] = {i: recovered(r) for i, r in rows.items()}
    still = sum(1 for lab, _ in labels[arm].values() if lab == "UNPARSED")
    fixed = sum(1 for _, was in labels[arm].values() if was)
    correct = sum(1 for lab, _ in labels[arm].values() if lab == "CORRECT")
    print(arm, "recovered", fixed - still, "still unparsed", still, "accuracy %.2f" % (100 * correct / len(rows)))

for a, b in (("Bp", "H"), ("Ap", "H"), ("Ap", "Bp")):
    ids = sorted(set(labels[a]) & set(labels[b]))
    out = {"all": paired([int(labels[a][i][0] == "CORRECT") for i in ids], [int(labels[b][i][0] == "CORRECT") for i in ids])}
    for cat in (2,):
        members = [i for i in ids if qas[i]["category"] == cat]
        out[str(cat)] = paired([int(labels[a][i][0] == "CORRECT") for i in members], [int(labels[b][i][0] == "CORRECT") for i in members])
    for key, v in out.items():
        print(f"{b}-vs-{a}", key, "%+.2f [%+.2f, %+.2f]" % (v["delta_points"], v["ci95_low_points"], v["ci95_high_points"]))
