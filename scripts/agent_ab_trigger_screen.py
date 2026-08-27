"""Can anything decide WHICH writes deserve a memory search?

    python -u scripts/agent_ab_trigger_screen.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --archive ~/.claude/archive/agent-ab-skill-001 \
        --screen ~/.claude/archive/direction-screen-2026-08-27/direction-screen.json \
        --precision benchmarks/artifacts/agent_ab/draft-precision.json

Preregistered in `docs/preregistrations/2026-08-27-search-trigger-screen.md`.

Draft-time search finds the governing memo 14 of 14 and fires just as confidently on writes with
nothing to find. No confidence threshold separates the two (threshold frontier, 0 viable points)
and no fusion weighting does either (fusion frontier, 0 of 12 variants). Both closed with: the
remaining question is a TRIGGER, not a retriever.

Candidates, and the PRE-search / POST-search distinction is the cost story:

    T0 always        —              the baseline the previous records measured
    T1 score margin  post-search    surface only if rank-1 beats rank-2 by more than delta
    T2 hazard vocab  PRE-search     search only if the draft carries a term memos distinctively use
    T3 operation     PRE-search     search only on a file write, subprocess, or VCS command
    T4 llm gate      PRE-search     one haiku call on the draft alone

⛔ This probe PERSISTS its per-draft capture. The fusion probe kept only summaries and this one
had to pay for retrieval again; every future trigger question should be free.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TOP_K = 5
CANDIDATE_K = 200
MAX_QUERY_CHARS = 4096
EMBEDDER_ID = "fastembed"
DIM = 384
CONTROL_MEMO = "python-write-text-crlf-churn"
CONTROL_QUERY = "Path.write_text on Windows injects CRLF against a tree configured eol=lf"

#: Swept rather than chosen, because no single value was registered.
VOCAB_DF_MAX = (1, 2, 3, 5, 10)
MARGIN_DELTAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)

#: T3: what a coding session physically does. Deliberately broad; prediction 1 says it is useless.
OPERATION_RE = re.compile(
    r"\b(open\(|write_text|write_bytes|writelines|subprocess|Popen|os\.system|shutil\.|"
    r"Path\(|\bgit\s|npm\s|pip\s|pytest\b|docker\s|mv\s|cp\s|rm\s|chmod\s|>>|>\s*\w+\.)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
JUDGE_PROMPT = (
    "You gate a memory lookup for a coding agent. Below is code the agent is about to save.\n"
    "Question: could this code fail in a NON-OBVIOUS way that a previously-written engineering "
    "note might warn about — a platform quirk, a silent data corruption, a tool that does not do "
    "what its name says?\n"
    "Answer 'yes' only if it is worth interrupting the agent to check notes. Answer 'no' for "
    "ordinary code whose failure modes are visible in the code itself.\n"
    "Reply with exactly one word: yes or no.\n\n=== CODE ===\n{draft}\n"
)


def sessions(archive: Path, keep: set[str]) -> list[dict]:
    rows = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("task_id") not in keep or record.get("variant") != "recall_on":
            continue
        if (record.get("metadata") or {}).get("locus") != "memory_only":
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
            "task_id": record["task_id"],
            "memo": str((record.get("metadata") or {}).get("governing_memo") or ""),
            "drafts": [d for d in drafts if len(d) <= MAX_QUERY_CHARS],
        })
    return rows


def hazard_vocabulary(store, df_max: int) -> set[str]:
    """Identifier-like tokens that few memos use, so their presence is distinctive.

    Derived from the corpus only; it never sees a draft or an outcome. Confound named in the
    record: because it comes from the same corpus the search runs against, this is closer to a
    cheap retrieval than to an independent signal, so read it as an upper bound on string matching.
    """

    import psycopg

    with psycopg.connect(store, connect_timeout=20) as conn:
        rows = conn.execute(
            "SELECT source_uri, string_agg(text, ' ') FROM recall_chunks_v1 GROUP BY source_uri"
        ).fetchall()
    df: Counter[str] = Counter()
    for _, text in rows:
        for token in {t.lower() for t in TOKEN_RE.findall(text or "")}:
            df[token] += 1
    return {token for token, count in df.items() if count <= df_max}


def judge(draft: str, key: str) -> bool | None:
    body = json.dumps({
        "model": JUDGE_MODEL, "temperature": 0, "max_tokens": 5,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(draft=draft[:4000])}],
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
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = "" if args.no_judge else os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not args.no_judge and not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (or pass --no-judge)")

    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall.retriever import HybridRetriever

    screen = json.loads(Path(args.screen).expanduser().read_text(encoding="utf-8"))
    miss_ids = {s["task_id"] for s in screen["sessions"]}
    positives = sessions(Path(args.archive).expanduser(), miss_ids)
    if len(positives) != 14:
        raise SystemExit(f"recovered {len(positives)} records for the registered 14; not run")

    precision = json.loads(Path(args.precision).expanduser().read_text(encoding="utf-8"))
    clean_negatives = [
        d["draft"] for r in precision["negatives"] for d in r["per_draft"]
        if not d.get("refused") and len(d["draft"]) <= MAX_QUERY_CHARS
    ]
    print(f"positives: 14 sessions, {sum(len(p['drafts']) for p in positives)} drafts")
    print(f"N_clean: {len(clean_negatives)} negative drafts")

    embedder = resolve_embedder(EMBEDDER_ID)
    store = GenerationStore(args.dsn, dim=DIM, tenant="default")
    store.check_schema()
    lexical = HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_dense=False)

    control = [
        Path(str(h.chunk.source)).name
        for h in lexical.search(CONTROL_QUERY, k=CANDIDATE_K).hits[:TOP_K]
    ]
    if f"{CONTROL_MEMO}.md" not in control:
        raise SystemExit(f"POSITIVE CONTROL FAILED: got {control}. Nothing below is a measurement.")
    print("positive control: OK")

    def capture(query: str, memo: str) -> dict:
        hits = lexical.search(query, k=CANDIDATE_K).hits
        top = [(Path(str(h.chunk.source)).name, float(h.score)) for h in hits[:TOP_K]]
        rank = next(
            (i for i, (n, _) in enumerate(top, 1) if memo and n == f"{memo}.md"), None
        )
        scores = [s for _, s in top]
        return {
            "chars": len(query),
            "top5": top,
            "governing_rank": rank,
            "margin": (scores[0] - scores[1]) if len(scores) >= 2 else 0.0,
            "operation": bool(OPERATION_RE.search(query)),
        }

    print("\ncapturing per-draft retrieval (persisted, so future trigger work is free)...")
    records: list[dict] = []
    for entry in positives:
        for index, draft in enumerate(entry["drafts"]):
            cap = capture(draft, entry["memo"])
            cap.update({"task_id": entry["task_id"], "memo": entry["memo"],
                        "draft_index": index, "population": "positive", "draft": draft})
            records.append(cap)
        print(f"  {entry['task_id']:<26} {len(entry['drafts'])} drafts")
    for index, draft in enumerate(clean_negatives):
        cap = capture(draft, "")
        cap.update({"task_id": "ctl-stage-by-pathspec", "memo": "", "draft_index": index,
                    "population": "n_clean", "draft": draft})
        records.append(cap)
    print(f"  N_clean {len(clean_negatives)} drafts\n")

    pos = [r for r in records if r["population"] == "positive"]
    n_clean = [r for r in records if r["population"] == "n_clean"]
    n_wide = [r for r in pos if r["governing_rank"] is None]
    bearing = [r for r in pos if r["governing_rank"] is not None]
    print(f"hazard-bearing drafts (retrieve the memo at top-5): {len(bearing)}/{len(pos)}")
    print(f"N_wide (positive-session drafts that do not): {len(n_wide)}")

    if not args.no_judge:
        print(f"\nT4: judging {len(pos) + len(n_clean)} drafts with {JUDGE_MODEL}...")
        for i, r in enumerate(records, 1):
            r["llm_gate"] = judge(r["draft"], key)
            if i % 50 == 0:
                print(f"  {i}/{len(records)}")

    def score(name: str, fires) -> dict:
        covered = sum(
            1 for e in positives
            if any(
                fires(r) and r["governing_rank"] is not None
                for r in pos if r["task_id"] == e["task_id"]
            )
        )
        ft_clean = sum(1 for r in n_clean if fires(r))
        ft_wide = sum(1 for r in n_wide if fires(r))
        fired_pos = sum(1 for r in pos if fires(r))
        return {
            "trigger": name, "coverage": covered, "of_sessions": 14,
            "ft_clean": ft_clean, "of_clean": len(n_clean),
            "ft_clean_rate": round(ft_clean / max(1, len(n_clean)), 4),
            "ft_wide": ft_wide, "of_wide": len(n_wide),
            "ft_wide_rate": round(ft_wide / max(1, len(n_wide)), 4),
            "suppression": round(1 - fired_pos / max(1, len(pos)), 4),
        }

    results = [score("T0_always", lambda r: True)]

    # ⛔ Control on this script's OWN contribution, not on its inputs: T0 fires on everything, so
    # coverage MUST be 14 and both false-trigger rates MUST be 1.0. If the scoring is wrong, this
    # is where it shows, before any candidate is read. The fusion probe lacked such a check and
    # published two invalid runs.
    t0 = results[0]
    if not (t0["coverage"] == 14 and t0["ft_clean_rate"] == 1.0 and t0["suppression"] == 0.0):
        raise SystemExit(f"TRIGGER SCORING CONTROL FAILED: T0 should be 14/1.0/0.0, got {t0}")
    print("trigger scoring control: OK (T0 reproduces the always-search baseline)\n")

    for delta in MARGIN_DELTAS:
        results.append(score(f"T1_margin_{delta}", lambda r, d=delta: r["margin"] > d))
    vocabs = {}
    for df_max in VOCAB_DF_MAX:
        vocab = hazard_vocabulary(args.dsn, df_max)
        vocabs[df_max] = len(vocab)
        results.append(score(
            f"T2_vocab_df{df_max}",
            lambda r, v=vocab: any(t.lower() in v for t in TOKEN_RE.findall(r["draft"])),
        ))
    results.append(score("T3_operation", lambda r: r["operation"]))
    if not args.no_judge:
        results.append(score("T4_llm_gate", lambda r: r.get("llm_gate") is True))

    print(f"{'trigger':<22} {'cover':>6} {'ft_clean':>9} {'ft_wide':>8} {'suppress':>9}")
    for r in results:
        print(f"{r['trigger']:<22} {r['coverage']:>3}/14  {r['ft_clean_rate']:>8.3f} "
              f"{r['ft_wide_rate']:>8.3f} {r['suppression']:>9.3f}")

    best = max(results[1:], key=lambda r: (r["coverage"], -r["ft_clean_rate"])) if len(results) > 1 else None
    if best:
        print(f"\nbest by coverage then low false-trigger: {best['trigger']} "
              f"coverage {best['coverage']}/14, ft_clean {best['ft_clean_rate']:.3f}")
    viable = [r for r in results[1:] if r["coverage"] >= 10 and r["ft_clean_rate"] <= 0.35]
    print(f"triggers with coverage >= 10 and ft_clean <= 0.35: {len(viable)}")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "trigger-screen.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "population": {"sessions": 14, "positive_drafts": len(pos),
                       "n_clean": len(n_clean), "n_wide": len(n_wide),
                       "hazard_bearing": len(bearing)},
        "vocab_sizes": vocabs, "judge_model": None if args.no_judge else JUDGE_MODEL,
        "judge_prompt": None if args.no_judge else JUDGE_PROMPT,
        "results": results, "records": records,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
