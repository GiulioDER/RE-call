"""Verification, not a new claim: how much of the 14/14 draft-search recall is ACTIONABLE?

    python -u scripts/agent_ab_actionable_recall.py \
        --trigger benchmarks/artifacts/agent_ab/trigger-screen.json \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control

The judge control of `docs/preregistrations/2026-08-27-enlarged-negative-set.md` voided that run
and, in doing so, exposed a defect in the measure this whole lane has used: **"the draft retrieves
the governing memo" is not "the governing memo applies to this draft".** Measured on the committed
artifact, `ts-worktree-import#r1`'s only memo-retrieving draft is `ls -la benchmarks/`, eighteen
characters, which retrieves the memo on a shared directory name and has nothing to do with the
hazard.

This judges EVERY memo-retrieving draft (46 of them across the 14 sessions) rather than the first,
so the corrected recall is bounded from both sides instead of guessed. It re-uses the committed
judge prompt and model unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent_ab_repo_negatives import JUDGE_MODEL, JUDGE_PROMPT  # noqa: E402

TOP_K = 5
CANDIDATE_K = 200


def judge(draft: str, note: str, key: str) -> bool | None:
    body = json.dumps({
        "model": JUDGE_MODEL, "temperature": 0, "max_tokens": 5,
        "messages": [{"role": "user",
                      "content": JUDGE_PROMPT.format(draft=draft[:4000], note=note[:1500])}],
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    else:
        return None
    answer = payload["choices"][0]["message"]["content"].strip().lower()
    return True if answer.startswith("yes") else False if answer.startswith("no") else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall.retriever import HybridRetriever

    embedder = resolve_embedder("fastembed")
    store = GenerationStore(args.dsn, dim=384, tenant="default")
    store.check_schema()
    lexical = HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_dense=False)

    prior = json.loads(Path(args.trigger).expanduser().read_text(encoding="utf-8"))
    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in prior["records"]:
        if record["population"] == "positive" and record["governing_rank"] is not None:
            by_session[record["task_id"]].append(record)
    total = sum(len(v) for v in by_session.values())
    print(f"{len(by_session)} sessions, {total} memo-retrieving drafts to judge\n")

    rows = []
    for task_id, records in sorted(by_session.items()):
        verdicts = []
        for record in records:
            wanted = f"{record['memo']}.md"
            note = next(
                (
                    str(h.chunk.text)
                    for h in lexical.search(record["draft"], k=CANDIDATE_K).hits[:TOP_K]
                    if Path(str(h.chunk.source)).name == wanted
                ),
                None,
            )
            verdict = judge(record["draft"], note, key) if note else None
            verdicts.append({
                "chars": record["chars"], "rank": record["governing_rank"],
                "actionable": verdict, "draft": record["draft"][:200],
            })
        actionable = sum(1 for v in verdicts if v["actionable"] is True)
        rows.append({
            "task_id": task_id, "memo": records[0]["memo"],
            "retrieving_drafts": len(records), "actionable_drafts": actionable,
            "reached_actionable": actionable > 0, "verdicts": verdicts,
        })
        mark = "OK " if actionable else "NO "
        print(f"  {mark} {task_id:<26} {actionable}/{len(records)} retrieving drafts actionable")

    reached = sum(1 for r in rows if r["reached_actionable"])
    act_drafts = sum(r["actionable_drafts"] for r in rows)
    print(f"\nACTIONABLE recall (a session has >=1 draft where the memo truly applies): "
          f"{reached}/{len(rows)}")
    print(f"retrieval-only recall, as previously published:                          "
          f"{len(rows)}/{len(rows)}")
    print(f"memo-retrieving drafts where the memo actually applies: {act_drafts}/{total} "
          f"({act_drafts/total:.3f})")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "actionable-recall.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "judge_model": JUDGE_MODEL, "judge_prompt": JUDGE_PROMPT,
        "sessions": len(rows), "retrieving_drafts": total,
        "actionable_recall": reached, "actionable_drafts": act_drafts,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
