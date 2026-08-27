"""Build a BLINDED human labelling sheet, so a gold set can replace the single judge.

    python -u scripts/agent_ab_build_gold_set.py \
        --validation benchmarks/artifacts/agent_ab/validation.json \
        --trigger benchmarks/artifacts/agent_ab/trigger-screen.json \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control

Three judges on the same 46 items implied actionable recalls of 10, 9 and 6 of 14, so no model
label is currently defensible. This produces the instrument for a human to settle it. It does NOT
produce labels: a gold set is gold because a person made the calls.

Selection, and why it is not the whole 46:

- **all 20 SPLIT items**, where the judges disagree. These carry all the information about which
  model is right, and they are where a human's time is worth most.
- **10 UNANIMOUS items**, 5 where all three said yes and 5 where all three said no. These are the
  controls that catch a SHARED blind spot: if a person disagrees with a unanimous verdict, all
  three models are wrong together, which no amount of inter-model agreement would reveal.

⛔ **Blinding is the point.** The sheet carries no model verdict, no session id, no memo name, and
the items are shuffled with a fixed seed so splits are not clustered. The key lands in a separate
file that must stay closed until the answers are written.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SEED = 20260827
SPLIT_ALL = True          # every split item is included; they are the informative ones
UNANIMOUS_YES = 5
UNANIMOUS_NO = 5
#: The labelling question is the JUDGES' question verbatim. If it differs, the comparison is void.
QUESTION = (
    "Would this note's failure strike THIS code, such that the engineer should change what they "
    "are about to write?\n\n"
    "Answer **yes** only for an actionable hazard in this code. Answer **no** for a note that is "
    "merely on a related topic, about a different operation, or generally interesting."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--out", default="benchmarks/artifacts/agent_ab/gold")
    args = parser.parse_args()

    import psycopg

    validation = json.loads(Path(args.validation).expanduser().read_text(encoding="utf-8"))
    trigger = json.loads(Path(args.trigger).expanduser().read_text(encoding="utf-8"))
    full = {(r["task_id"], r["draft"][:200]): r["draft"]
            for r in trigger["records"] if r["population"] == "positive"}

    items = validation["part_a"]["items"]
    split, yes, no = [], [], []
    for item in items:
        votes = (item["haiku"], item["sonnet"], item["gemini"])
        (split if len(set(votes)) > 1 else (yes if votes[0] else no)).append(item)
    rng = random.Random(SEED)
    rng.shuffle(yes)
    rng.shuffle(no)
    chosen = split + yes[:UNANIMOUS_YES] + no[:UNANIMOUS_NO]
    rng.shuffle(chosen)
    print(f"{len(split)} split, {len(yes)} unanimous-yes, {len(no)} unanimous-no")
    print(f"selected {len(chosen)} items for labelling")

    # Pick the memo chunk to DISPLAY with `ts_rank` in SQL rather than through the retriever.
    # The retriever would load fastembed, and the embedder is not needed to choose a chunk for a
    # human to read: this is a display decision, not a measurement. It also avoids the documented
    # onnxruntime `bad allocation` on this 12 GB box, which is exactly how the first attempt died.
    def best_chunk(conn, memo: str, draft: str) -> str:
        terms = " | ".join(
            sorted({t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", draft)})[:200]
        )
        if not terms:
            terms = "the"
        row = conn.execute(
            "SELECT text FROM recall_chunks_v1 WHERE source_uri LIKE %s "
            "ORDER BY ts_rank(tsv, to_tsquery('english', %s)) DESC LIMIT 1",
            (f"%/{memo}.md", terms),
        ).fetchone()
        return row[0] if row else ""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    sheet.append("# Gold set: does this note apply to this code?\n")
    sheet.append(
        "Three models disagree on these, and every number in the draft-search work now depends on\n"
        "who is right. There are **{n} items**. For each, read the code and the note and answer\n"
        "the one question below.\n".format(n=len(chosen))
    )
    sheet.append(f"## The question\n\n{QUESTION}\n")
    sheet.append(
        "## How to answer\n\n"
        "Write your answers in `gold-answers.txt`, one per line, as `<number>: yes` or\n"
        "`<number>: no`. Add `?` instead if you genuinely cannot tell from what is shown — that is\n"
        "a real answer and it is more useful than a guess.\n\n"
        "⛔ **Do not open `gold-key.json` until you are finished.** It holds the model verdicts,\n"
        "and seeing them first would anchor the labels and waste the exercise.\n"
    )
    sheet.append("---\n")

    conn = psycopg.connect(args.dsn, connect_timeout=20)
    for number, item in enumerate(chosen, 1):
        draft = full.get((item["task_id"], item["draft_prefix"][:200]), item["draft_prefix"])
        note = best_chunk(conn, item["memo"], draft)
        sheet.append(f"## Item {number}\n")
        sheet.append("**Code about to be saved:**\n")
        sheet.append("```\n" + draft.strip()[:2500] + "\n```\n")
        sheet.append("**Retrieved note:**\n")
        sheet.append("```\n" + note.strip()[:1500] + "\n```\n")
        sheet.append(f"**Item {number} answer:** `yes` / `no` / `?`\n")
        sheet.append("---\n")
        key.append({
            "number": number, "task_id": item["task_id"], "memo": item["memo"],
            "haiku": item["haiku"], "sonnet": item["sonnet"], "gemini": item["gemini"],
            "kind": "split" if len({item["haiku"], item["sonnet"], item["gemini"]}) > 1
                    else ("unanimous_yes" if item["haiku"] else "unanimous_no"),
        })

    conn.close()
    (out / "gold-set.md").write_text("\n".join(sheet), encoding="utf-8", newline="\n")
    (out / "gold-key.json").write_text(
        json.dumps({"seed": SEED, "question": QUESTION, "items": key}, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    (out / "gold-answers.txt").write_text(
        "# One line per item: `<number>: yes` / `no` / `?`\n"
        "# `?` is a real answer: it means the item as shown does not settle it.\n\n"
        + "\n".join(f"{n}: " for n in range(1, len(chosen) + 1)) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\nwrote {out / 'gold-set.md'}          <- read and answer this")
    print(f"wrote {out / 'gold-answers.txt'}      <- fill this in")
    print(f"wrote {out / 'gold-key.json'}         <- DO NOT OPEN until answered")
    counts = {k: sum(1 for i in key if i["kind"] == k) for k in
              ("split", "unanimous_yes", "unanimous_no")}
    print(f"\ncomposition (in the key, not the sheet): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
