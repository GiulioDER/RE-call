"""Prove how MTRAG builds a dev 'lastturn' query, before constructing 65 of them myself.

The dev retrieval task files ship PRE-COMPUTED query text, and they cover only the 777 judged
queries. The 65 generation tasks with no retrieval counterpart therefore need their query built
here -- and if the construction differs by even a prefix, those 65 are retrieved from a different
string than the 777 and the two halves are not comparable.

So: reconstruct the query for every SHARED task and require an exact match against the released
text. Anything less than 777/777 means the rule is wrong.
"""

import json

R = "/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark"
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")

dev = {}
for d in DOMAINS:
    with open(f"{R}/mtrag-human/retrieval_tasks/{d}/{d}_lastturn.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                dev[row["_id"]] = row["text"]

gen = {}
with open(f"{R}/mtrag-human/generation_tasks/RAG.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            row = json.loads(line)
            gen[row["task_id"]] = row


def build(task):
    """Last USER turn, speaker-prefixed the way the release writes it."""
    for turn in reversed(task.get("input") or []):
        if str(turn.get("speaker", "")).lower() == "user":
            speaker = turn.get("speaker")
            text = turn.get("text", "")
            return "|" + str(speaker) + "|: " + str(text)
    return None


shared = [t for t in dev if t in gen]
exact = sum(1 for t in shared if build(gen[t]) == dev[t])
print(f"prefixed last-user-turn == released query: {exact}/{len(shared)}")

for t in [t for t in shared if build(gen[t]) != dev[t]][:3]:
    print("  MISMATCH", t)
    print("   released:", repr(dev[t])[:150])
    print("   built   :", repr(build(gen[t]))[:150])

missing = [t for t in gen if t not in dev]
print(f"\ngeneration tasks with no dev query: {len(missing)}")
if exact == len(shared):
    print("rule VERIFIED -- safe to build the missing queries with it")
    out = f"{R}/../missing65_queries.jsonl"
    import collections

    lab = collections.Counter((gen[t].get("Answerability") or ["?"])[0] for t in missing)
    print("their labels:", dict(lab))
    with open("/var/tmp/missing65_queries.jsonl", "w", encoding="utf-8") as fh:
        for t in missing:
            fh.write(json.dumps({
                "task_id": t,
                "Collection": gen[t].get("Collection"),
                "text": build(gen[t]),
            }, ensure_ascii=False) + "\n")
    print("wrote /var/tmp/missing65_queries.jsonl")
