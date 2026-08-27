"""Measure the PRECISION half of draft-time search, on the production path.

    python -u scripts/agent_ab_draft_precision.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --archive ~/.claude/archive/agent-ab-skill-001

Preregistered in `docs/preregistrations/2026-08-27-draft-search-precision.md`. Read-only.

The direction screen measured that a draft query retrieves the governing memo 14 of 14 times. That
is recall of a document already known to be correct. This asks the three questions that decide
whether the direction is buildable:

1. does it survive the PRODUCTION path (stdio server, calibration, strict trust policy, top-5),
   rather than the lexical leg the screen's headline used;
2. what else occupies the five slots, hard-labelled and judged;
3. what happens on a draft with NO relevant memo — the case that fires on most writes.

⛔ A retrieval error is never scored as a miss, and a positive control must retrieve a memo by
quoting its own content before any number is believed. The screen that preceded this one produced
a flawless, entirely fabricated 0/14 three separate ways; see
`[[a-null-is-the-cheapest-result-to-fabricate]]`.
"""

from __future__ import annotations

import argparse
import asyncio
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

from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402

TOP_K = 5
#: `recall_mcp.service.MAX_QUERY_CHARS`. The server REFUSES a longer query rather than truncating
#: it, on the stated ground that searching a prefix answers a question the caller did not ask.
#: Measured over the archive: 3 of 501 recorded payloads exceed it, and no session loses all of
#: its. Such a payload is recorded as `refused_too_long` and excluded from every rate, never
#: silently scored as a miss.
MAX_QUERY_CHARS = 4096
JUDGE_MODEL = "anthropic/claude-haiku-4.5"
#: Actionable relevance, deliberately not topical similarity: the question is whether the note
#: would change what the author of THIS draft should do, which is the only thing worth an agent's
#: attention on every write.
JUDGE_PROMPT = (
    "An engineer is about to save the code below. A memory search returned the note below it.\n"
    "Answer strictly: would this note's failure strike THIS code, such that the engineer should "
    "change what they are about to write?\n"
    "Answer 'yes' only for an actionable hazard in this code. Answer 'no' for a note that is "
    "merely on a related topic, about a different operation, or generally interesting.\n"
    "Reply with exactly one word: yes or no.\n\n"
    "=== CODE ABOUT TO BE SAVED ===\n{draft}\n\n=== RETRIEVED NOTE ===\n{note}\n"
)


def sessions(archive: Path) -> list[dict]:
    rows: list[dict] = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("variant") != "recall_on":
            continue
        metadata = record.get("metadata") or {}
        drafts = []
        for call in record.get("tool_calls") or []:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            if name in ("Write", "Edit", "NotebookEdit"):
                payload = str(args.get("content") or args.get("new_string") or "")
                if payload.strip():
                    drafts.append(payload)
            elif name == "Bash" and args.get("command"):
                drafts.append(str(args["command"]))
        memo = metadata.get("governing_memo")
        rows.append(
            {
                "task_id": str(record.get("task_id", "")),
                "base": str(record.get("task_id", "")).split("#")[0],
                "locus": metadata.get("locus"),
                "memo": str(memo) if memo else None,
                "drafts": drafts,
            }
        )
    return rows


async def search(spec: StdioRecallSpec, queries: list[str]) -> dict[str, dict]:
    """One stdio session, every query, keeping verdicts and abstention, not just names."""

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=spec.python, args=["-m", "recall_mcp.server"], env=spec.env(), cwd=str(spec.cwd)
    )
    answers: dict[str, dict] = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for query in queries:
                result = await session.call_tool("recall_search", {"query": query, "k": TOP_K})
                text = next(
                    (getattr(b, "text", "") for b in result.content if getattr(b, "text", None)), ""
                )
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    # The server refused. That is data about the production path, never a miss.
                    answers[query] = {"refused": text[:300], "abstained": None, "hits": []}
                    continue
                if isinstance(payload, dict) and "result" in payload:
                    payload = json.loads(payload["result"])
                hits = payload.get("hits") or []
                answers[query] = {
                    "refused": None,
                    "abstained": bool(payload.get("abstained")),
                    "hits": [
                        {
                            "source": Path(str(h.get("source") or "")).name,
                            "verdict": h.get("verdict"),
                            "confidence": h.get("confidence"),
                            "text": str(h.get("text") or "")[:1200],
                        }
                        for h in hits[:TOP_K]
                    ],
                }
    return answers


def judge(draft: str, note: str, key: str) -> bool | None:
    body = json.dumps(
        {
            "model": JUDGE_MODEL,
            "temperature": 0,
            "max_tokens": 5,
            "messages": [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(draft=draft[:4000], note=note[:1500]),
                }
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
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
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = "" if args.no_judge else os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not args.no_judge and not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (or pass --no-judge)")

    rows = sessions(Path(args.archive).expanduser())
    positives = [r for r in rows if r["locus"] == "memory_only" and r["memo"]]
    negatives = [r for r in rows if r["base"] == "ctl-stage-by-pathspec"]
    all_memos = {r["memo"] for r in positives if r["memo"]}
    print(f"positives: {len(positives)} sessions, {sum(len(r['drafts']) for r in positives)} drafts")
    print(f"negatives: {len(negatives)} sessions, {sum(len(r['drafts']) for r in negatives)} drafts")
    print(f"governing memos in play: {len(all_memos)}")

    spec = StdioRecallSpec(dsn=args.dsn, cwd=REPO_ROOT, tenant="default")
    check = await spec.check()
    print(f"server: {check.get('tool_count')} tools, trust_state={check.get('trust_state')} "
          f"calibrated={check.get('calibrated')}")

    control_query = "Path.write_text on Windows injects CRLF against a tree configured eol=lf"
    queries = [control_query]
    for row in positives + negatives:
        queries.extend(row["drafts"])
    queries = list(dict.fromkeys(queries))
    print(f"{len(queries)} distinct queries to run on the production path\n")

    answers = await search(spec, queries)

    control = answers[control_query]
    names = [h["source"] for h in control["hits"]]
    if "python-write-text-crlf-churn.md" not in names:
        raise SystemExit(
            "POSITIVE CONTROL FAILED: quoting a memo's own content did not return it in the "
            f"top {TOP_K} (got {names}). The instrument is broken; nothing below is a measurement."
        )
    print(f"positive control: OK, memo at rank {names.index('python-write-text-crlf-churn.md') + 1}")

    def slots(row: dict) -> list[dict]:
        out = []
        for draft in row["drafts"]:
            answer = answers[draft]
            out.append({
                "draft": draft,
                "chars": len(draft),
                "refused": answer.get("refused"),
                "abstained": answer["abstained"],
                "hits": answer["hits"],
            })
        return out

    pos_results = []
    for row in positives:
        per_draft = slots(row)
        found = [
            (index, d) for index, d in enumerate(per_draft)
            if any(h["source"] == f"{row['memo']}.md" and h["verdict"] == "ok" for h in d["hits"])
        ]
        pos_results.append({
            "task_id": row["task_id"], "base": row["base"], "memo": row["memo"],
            "n_drafts": len(per_draft),
            "reached": bool(found),
            "first_draft_index": found[0][0] if found else None,
            "per_draft": per_draft,
        })

    neg_results = []
    for row in negatives:
        per_draft = slots(row)
        neg_results.append({
            "task_id": row["task_id"], "base": row["base"], "n_drafts": len(per_draft),
            "per_draft": per_draft,
        })

    # Hard labels: a slot is this task's memo, another task's memo (known-irrelevant), or other.
    def hard(rows_: list[dict], own: str | None) -> dict:
        own_hits = other_memo = other = 0
        for entry in rows_:
            for hit in entry["hits"]:
                stem = hit["source"][:-3] if hit["source"].endswith(".md") else hit["source"]
                if own and stem == own:
                    own_hits += 1
                elif stem in all_memos:
                    other_memo += 1
                else:
                    other += 1
        return {"own": own_hits, "other_governing": other_memo, "unlabelled": other}

    MISSED = {"ts-lf-rewrite", "ts-worktree-import", "ts-sample-covers-tail"}
    missed = [r for r in pos_results if r["base"] in MISSED]
    print(f"PRODUCTION PATH, the 14 screen misses: reached {sum(r['reached'] for r in missed)}"
          f"/{len(missed)}")
    print(f"PRODUCTION PATH, all positives:        reached "
          f"{sum(r['reached'] for r in pos_results)}/{len(pos_results)}")
    by_family: dict[str, list[int]] = defaultdict(list)
    for r in pos_results:
        by_family[r["base"]].append(int(r["reached"]))
    for base in sorted(by_family):
        v = by_family[base]
        print(f"    {base:<26} {sum(v)}/{len(v)}")

    all_slots = [d for r in pos_results + neg_results for d in r["per_draft"]]
    refused = [d for d in all_slots if d["refused"]]
    print(f"\nserver refusals (query over {MAX_QUERY_CHARS} chars): {len(refused)}/{len(all_slots)}")
    lengths = sorted(d["chars"] for d in all_slots)
    print(f"query length: median {lengths[len(lengths) // 2]}  max {lengths[-1]} characters")

    neg_queries = [d for r in neg_results for d in r["per_draft"] if not d["refused"]]
    abstained = sum(1 for d in neg_queries if d["abstained"])
    no_ok = sum(1 for d in neg_queries if not any(h["verdict"] == "ok" for h in d["hits"]))
    print(f"\nNEGATIVES ({len(neg_queries)} draft queries, {len(neg_results)} sessions):")
    print(f"    abstained:              {abstained}/{len(neg_queries)}")
    print(f"    returned no ok verdict: {no_ok}/{len(neg_queries)}")

    all_pos_slots = [d for r in pos_results for d in r["per_draft"]]
    hard_pos = hard(all_pos_slots, None)
    print(f"\nHARD LABELS over {len(all_pos_slots)} positive draft queries: {hard_pos}")

    judged = []
    if not args.no_judge:
        print("\njudging actionable relevance on the misses and the negatives...")
        targets = []
        for r in missed:
            best_draft = None
            for d in r["per_draft"]:
                if any(h["source"] == f"{r['memo']}.md" for h in d["hits"]):
                    best_draft = d
                    break
            if best_draft:
                targets.append((r["task_id"], best_draft))
        for r in neg_results:
            for d in r["per_draft"]:
                targets.append((r["task_id"], d))
        for task_id, entry in targets:
            for hit in entry["hits"]:
                verdict = judge(entry["draft"], hit["text"], key)
                judged.append({
                    "task_id": task_id, "source": hit["source"],
                    "trust_verdict": hit["verdict"], "actionable": verdict,
                })
        yes = sum(1 for j in judged if j["actionable"] is True)
        print(f"  judged {len(judged)} slots, actionable: {yes} ({yes / max(1, len(judged)):.2f})")

    payload = {
        "database": args.dsn.rsplit("/", 1)[-1],
        "top_k": TOP_K,
        "judge_model": None if args.no_judge else JUDGE_MODEL,
        "judge_prompt": None if args.no_judge else JUDGE_PROMPT,
        "positives": pos_results,
        "negatives": neg_results,
        "hard_labels_positive": hard_pos,
        "judged": judged,
    }
    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "draft-precision.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
