"""Validation: is the lane's headline an artifact of ONE judge and THREE task families?

    python -u scripts/agent_ab_validate.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --archive ~/.claude/archive/agent-ab-skill-001 \
        --actionable benchmarks/artifacts/agent_ab/actionable-recall.json \
        --trigger benchmarks/artifacts/agent_ab/trigger-screen.json

Preregistered in `docs/preregistrations/2026-08-27-judge-and-holdout-validation.md`.

Part A: re-judge the same 46 memo-retrieving drafts with a stronger same-family model and a
different-family model, and report the actionable recall each implies. A single judge produced
every corrected number in this lane and its agreement with anything has never been measured.

Part B: run the whole draft-search pipeline on the FOUR families never used for it (24 sessions,
199 drafts), and transfer the vocabulary trigger at the `df<=2` threshold FITTED ELSEWHERE rather
than refitting it here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agent_ab_repo_negatives import JUDGE_PROMPT  # noqa: E402  (the committed prompt, verbatim)

TOP_K = 5
CANDIDATE_K = 200
MAX_QUERY_CHARS = 4096
VOCAB_DF = 2  # fitted on the OTHER families; deliberately not refitted here.
HELD_OUT = ("ts-autouse-tmp-path", "ts-bounded-runner", "ts-false-zero-search",
            "ts-separator-canary")
JUDGES = {
    "haiku": "anthropic/claude-haiku-4.5",
    "sonnet": "anthropic/claude-sonnet-5",
    "gemini": "google/gemini-2.5-pro",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def ask(model: str, draft: str, note: str, key: str) -> bool | None:
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 5,
        "messages": [{"role": "user",
                      "content": JUDGE_PROMPT.format(draft=draft[:4000], note=note[:1500])}],
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    else:
        return None
    text = (payload["choices"][0]["message"].get("content") or "").strip().lower()
    return True if text.startswith("yes") else False if text.startswith("no") else None


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa; raw agreement alone flatters a skewed label distribution."""

    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else 1.0


def sessions(archive: Path, families: tuple[str, ...]) -> list[dict]:
    rows = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("variant") != "recall_on":
            continue
        metadata = record.get("metadata") or {}
        if metadata.get("locus") != "memory_only":
            continue
        base = str(record.get("task_id", "")).split("#")[0]
        if base not in families:
            continue
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
        rows.append({
            "task_id": record["task_id"], "base": base,
            "memo": str(metadata.get("governing_memo") or ""),
            "drafts": [d for d in drafts if len(d) <= MAX_QUERY_CHARS],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--actionable", required=True)
    parser.add_argument("--trigger", required=True)
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

    def top5(query: str) -> list[tuple[str, str]]:
        hits = lexical.search(query, k=CANDIDATE_K).hits[:TOP_K]
        return [(Path(str(h.chunk.source)).name, str(h.chunk.text)) for h in hits]

    control = [n for n, _ in top5(
        "Path.write_text on Windows injects CRLF against a tree configured eol=lf")]
    if "python-write-text-crlf-churn.md" not in control:
        raise SystemExit(f"POSITIVE CONTROL FAILED: {control}")
    print("positive control: OK\n")

    # ---------------- Part A: judge agreement on the SAME 46 drafts ----------------
    prior = json.loads(Path(args.actionable).expanduser().read_text(encoding="utf-8"))
    items = []
    for row in prior["rows"]:
        for entry in row["verdicts"]:
            items.append({"task_id": row["task_id"], "memo": row["memo"],
                          "draft_prefix": entry["draft"], "haiku": entry["actionable"] is True})
    print(f"PART A: re-judging {len(items)} memo-retrieving drafts with "
          f"{', '.join(k for k in JUDGES if k != 'haiku')}")

    trigger = json.loads(Path(args.trigger).expanduser().read_text(encoding="utf-8"))
    full = {(r["task_id"], r["draft"][:200]): r["draft"]
            for r in trigger["records"] if r["population"] == "positive"}
    for item in items:
        draft = full.get((item["task_id"], item["draft_prefix"][:200]))
        if draft is None:
            raise SystemExit(f"cannot recover full draft for {item['task_id']}")
        note = next((t for n, t in top5(draft) if n == f"{item['memo']}.md"), None)
        for name, model in JUDGES.items():
            if name == "haiku":
                continue
            item[name] = ask(model, draft, note, key) is True

    by_session: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        for name in JUDGES:
            by_session[item["task_id"]][name].append(item[name])
    recalls = {name: sum(1 for s in by_session.values() if any(s[name])) for name in JUDGES}
    print("\n  judge     agreement-vs-haiku   kappa   implied actionable recall")
    agree = {}
    for name in JUDGES:
        if name == "haiku":
            print(f"  {name:<9} {'—':>18}   {'—':>5}   {recalls[name]}/{len(by_session)}")
            continue
        a = [i["haiku"] for i in items]
        b = [i[name] for i in items]
        raw = round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 4)
        agree[name] = {"raw": raw, "kappa": kappa(a, b)}
        print(f"  {name:<9} {raw:>18.3f}   {agree[name]['kappa']:>5.3f}   "
              f"{recalls[name]}/{len(by_session)}")
    cross = kappa([i["sonnet"] for i in items], [i["gemini"] for i in items])
    spread = max(recalls.values()) - min(recalls.values())
    print(f"  sonnet vs gemini kappa: {cross:.3f}")
    print(f"\n  SPREAD in implied actionable recall across judges: {spread} session(s)")
    print("  " + {0: "stable", 1: "stable", 2: "soft (+/- 1 session from here on)"}.get(
        spread, "UNSTABLE: downstream numbers are provisional"))

    # ---------------- Part B: the four held-out families ----------------
    print(f"\nPART B: held-out families {HELD_OUT}")
    held = sessions(Path(args.archive).expanduser(), HELD_OUT)
    print(f"  {len(held)} sessions, {sum(len(h['drafts']) for h in held)} drafts")

    with __import__("psycopg").connect(args.dsn, connect_timeout=20) as conn:
        rows = conn.execute(
            "SELECT source_uri, string_agg(text,' ') FROM recall_chunks_v1 GROUP BY source_uri"
        ).fetchall()
    df: Counter[str] = Counter()
    for _, text in rows:
        for token in {t.lower() for t in TOKEN_RE.findall(text or "")}:
            df[token] += 1
    vocab = {t for t, c in df.items() if c <= VOCAB_DF}
    print(f"  vocabulary at df<={VOCAB_DF}: {len(vocab)} terms (fitted elsewhere, not refitted)\n")

    results = []
    for entry in held:
        wanted = f"{entry['memo']}.md"
        actionable, fired_and_actionable = [], []
        for draft in entry["drafts"]:
            hits = top5(draft)
            note = next((t for n, t in hits if n == wanted), None)
            if note is None:
                continue
            good = ask(JUDGES["haiku"], draft, note, key) is True
            if good:
                actionable.append(draft)
                if any(t.lower() in vocab for t in TOKEN_RE.findall(draft)):
                    fired_and_actionable.append(draft)
        results.append({
            "task_id": entry["task_id"], "base": entry["base"], "memo": entry["memo"],
            "drafts": len(entry["drafts"]), "actionable_drafts": len(actionable),
            "reached": bool(actionable), "trigger_covered": bool(fired_and_actionable),
        })
        print(f"  {entry['task_id']:<26} actionable {len(actionable):>2}  "
              f"{'reached' if actionable else 'MISS':<8} "
              f"{'trigger fires' if fired_and_actionable else '' if not actionable else 'TRIGGER MISSES'}")

    reached = [r for r in results if r["reached"]]
    covered = [r for r in reached if r["trigger_covered"]]
    recall_rate = round(len(reached) / len(results), 4)
    cov_rate = round(len(covered) / max(1, len(reached)), 4)
    print(f"\n  held-out actionable recall : {len(reached)}/{len(results)} = {recall_rate:.3f}"
          f"   (fitted: 10/14 = 0.714)")
    print(f"  held-out trigger coverage  : {len(covered)}/{len(reached)} = {cov_rate:.3f}"
          f"   (fitted: 9/10 = 0.900)")
    per = defaultdict(lambda: [0, 0, 0])
    for r in results:
        per[r["base"]][2] += 1
        per[r["base"]][0] += int(r["reached"])
        per[r["base"]][1] += int(r["trigger_covered"])
    print("\n  per family (counts, not rates: 6 sessions each):")
    for base, (rc, cv, n) in sorted(per.items()):
        print(f"    {base:<26} reached {rc}/{n}   trigger-covered {cv}/{n}")

    out = Path(args.out) if args.out else Path(
        "benchmarks/artifacts/agent_ab/validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "part_a": {"judges": JUDGES, "n_drafts": len(items), "agreement": agree,
                   "sonnet_vs_gemini_kappa": cross, "implied_recall": recalls,
                   "sessions": len(by_session), "spread": spread, "items": items},
        "part_b": {"families": list(HELD_OUT), "vocab_df": VOCAB_DF, "vocab_terms": len(vocab),
                   "sessions": len(results), "actionable_recall": recall_rate,
                   "trigger_coverage": cov_rate, "results": results},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
