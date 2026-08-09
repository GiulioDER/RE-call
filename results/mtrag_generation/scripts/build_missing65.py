"""Build the 65 missing retrieval queries, with the query format and domain DERIVED, not guessed.

Two things must match what the 777 got, and both were wrong on the first attempt:

1. QUERY TEXT. The dev retrieval files ship pre-computed query text carrying a `|user|: ` prefix,
   which `query_text` then strips. Reconstructing "the last user turn" without the prefix happens
   to land in the same place only because the strip is applied afterwards -- so the prefixed form
   is emitted here and the harness's own normalisation does the rest.

2. DOMAIN. `DOMAIN_ALIASES` maps SHORT names (clapnq/cloud/fiqa/govt, plus ibmcloud->cloud), but
   `mtrag-human/generation_tasks` labels every task with a long ELSER collection name such as
   `mt-rag-clapnq-elser-512-100-20240503`. Rather than hand-write that mapping and hope, it is
   derived from the 777 tasks that appear in BOTH files: their domain is known from WHICH dev file
   they were read out of, and their Collection is known from the generation file. A mapping that is
   not 1:1 is a hard failure rather than a coin flip.

Both rules are then asserted, because a query built differently retrieves different passages, and
65 rows that merely look like the other 777 would be worse than none.
"""

import collections
import json

R = "/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark"
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")

dev_domain = {}
dev_text = {}
for d in DOMAINS:
    with open(f"{R}/mtrag-human/retrieval_tasks/{d}/{d}_lastturn.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                dev_domain[row["_id"]] = d
                dev_text[row["_id"]] = row["text"]

gen = {}
with open(f"{R}/mtrag-human/generation_tasks/RAG.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            row = json.loads(line)
            gen[row["task_id"]] = row


def build_query(task):
    for turn in reversed(task.get("input") or []):
        if str(turn.get("speaker", "")).lower() == "user":
            return "|" + str(turn.get("speaker")) + "|: " + str(turn.get("text", ""))
    return None


shared = [t for t in dev_domain if t in gen]

# --- rule 1: query text
exact = sum(1 for t in shared if build_query(gen[t]) == dev_text[t])
print(f"query rule: {exact}/{len(shared)} exact")
assert exact == len(shared), "query construction does not reproduce the released text"

# --- rule 2: Collection -> domain, derived
pairs = collections.defaultdict(set)
for t in shared:
    pairs[str(gen[t].get("Collection"))].add(dev_domain[t])
mapping = {}
for coll, domains in sorted(pairs.items()):
    assert len(domains) == 1, f"Collection {coll!r} maps to {domains}, not 1:1"
    mapping[coll] = next(iter(domains))
    print(f"  {coll:44} -> {mapping[coll]}")

missing = [t for t in gen if t not in dev_domain]
unknown = {str(gen[t].get("Collection")) for t in missing} - set(mapping)
assert not unknown, f"missing tasks use Collections never seen in the shared set: {unknown}"

with open("/var/tmp/missing65_queries.jsonl", "w", encoding="utf-8") as fh:
    for t in missing:
        coll = str(gen[t].get("Collection"))
        fh.write(json.dumps({
            "task_id": t,
            "Collection": coll,
            "_domain": mapping[coll],
            "text": build_query(gen[t]),
            "input": gen[t].get("input", []),
        }, ensure_ascii=False) + "\n")

lab = collections.Counter((gen[t].get("Answerability") or ["?"])[0] for t in missing)
by_dom = collections.Counter(mapping[str(gen[t].get("Collection"))] for t in missing)
print(f"\nwrote {len(missing)} queries | labels {dict(lab)} | domains {dict(by_dom)}")
