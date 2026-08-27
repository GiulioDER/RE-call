"""Build a REAL negative set from this repository's history, and re-score the triggers on it.

    python -u scripts/agent_ab_repo_negatives.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --trigger benchmarks/artifacts/agent_ab/trigger-screen.json

Preregistered in `docs/preregistrations/2026-08-27-enlarged-negative-set.md`.

Every trigger number in this lane rests on 18 draft queries from four sessions of ONE task, all
git commands, and that set already distorted one trigger's headline outright. The ideal negative
is code written in THIS repository, in its style, touching its tooling, that nonetheless has no
memo warning about it — so the population is added-line blocks from `origin/master`. Code from an
unrelated project would be an easy negative that flatters every candidate.

Ground truth is a judge, which makes the judge the instrument. ⛔ So it is measured first, against
cases whose answer is known independently: the benchmark's declared governing memos. Below the
registered bar the run is VOID and no trigger number is read.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TOP_K = 5
CANDIDATE_K = 200
SEED = 20260827
SAMPLE = 200
COMMITS = 120
MIN_CHARS, MAX_CHARS = 40, 4096
EMBEDDER_ID, DIM = "fastembed", 384
JUDGE_MODEL = "anthropic/claude-haiku-4.5"
CONTROL_FLOOR = 11  # of 14; below this the judge is untrustworthy and the run is void.

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
OPERATION_RE = re.compile(
    r"\b(open\(|write_text|write_bytes|writelines|subprocess|Popen|os\.system|shutil\.|"
    r"Path\(|\bgit\s|npm\s|pip\s|pytest\b|docker\s|mv\s|cp\s|rm\s|chmod\s|>>|>\s*\w+\.)",
    re.IGNORECASE,
)
#: Identical to the precision run's, deliberately: the same instrument, so the numbers compare.
JUDGE_PROMPT = (
    "An engineer is about to save the code below. A memory search returned the note below it.\n"
    "Answer strictly: would this note's failure strike THIS code, such that the engineer should "
    "change what they are about to write?\n"
    "Answer 'yes' only for an actionable hazard in this code. Answer 'no' for a note that is "
    "merely on a related topic, about a different operation, or generally interesting.\n"
    "Reply with exactly one word: yes or no.\n\n"
    "=== CODE ABOUT TO BE SAVED ===\n{draft}\n\n=== RETRIEVED NOTE ===\n{note}\n"
)


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


def sample_hunks() -> tuple[list[str], list[str]]:
    """Added-line blocks from master, reproducibly: fixed seed over a recorded commit list."""

    shas = subprocess.run(
        ["git", "log", "--format=%H", "-n", str(COMMITS), "origin/master"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout.split()
    blocks: list[str] = []
    for sha in shas:
        diff = subprocess.run(
            ["git", "show", sha, "--unified=0", "--format=", "--", "*.py"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        ).stdout
        current: list[str] = []
        for line in diff.splitlines():
            if line.startswith("@@"):
                if current:
                    blocks.append("\n".join(current))
                current = []
            elif line.startswith("+") and not line.startswith("+++"):
                current.append(line[1:])
        if current:
            blocks.append("\n".join(current))
    usable = [b for b in blocks if MIN_CHARS <= len(b) <= MAX_CHARS]
    random.Random(SEED).shuffle(usable)
    return usable[:SAMPLE], shas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--trigger", required=True, help="trigger-screen.json, for the control set")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall.retriever import HybridRetriever

    embedder = resolve_embedder(EMBEDDER_ID)
    store = GenerationStore(args.dsn, dim=DIM, tenant="default")
    store.check_schema()
    lexical = HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_dense=False)

    def top5(query: str) -> list[tuple[str, str]]:
        hits = lexical.search(query, k=CANDIDATE_K).hits[:TOP_K]
        return [(Path(str(h.chunk.source)).name, str(h.chunk.text)) for h in hits]

    # ⛔ CONTROL FIRST: the judge is the ground truth, so measure it against known answers.
    prior = json.loads(Path(args.trigger).expanduser().read_text(encoding="utf-8"))
    known = {}
    for record in prior["records"]:
        if record["population"] != "positive" or record["governing_rank"] is None:
            continue
        known.setdefault(record["task_id"], record)
    print(f"judge control: {len(known)} sessions with a known hazard-bearing draft")
    passed = 0
    for task_id, record in sorted(known.items()):
        wanted = f"{record['memo']}.md"
        note = next(
            (text for name, text in top5(record["draft"]) if name == wanted), None
        )
        verdict = judge(record["draft"], note, key) if note else None
        passed += bool(verdict)
        print(f"  {task_id:<26} {'actionable' if verdict else 'NOT actionable'}")
    print(f"judge control: {passed}/{len(known)} known governing memos called actionable")
    if passed < CONTROL_FLOOR:
        raise SystemExit(
            f"JUDGE CONTROL FAILED: {passed}/{len(known)}, below the registered floor of "
            f"{CONTROL_FLOOR}. The judge is the ground truth for this run, so a judge that misses "
            "known answers mislabels the population silently. VOID; no trigger number is read."
        )
    print("judge control: PASSED\n")

    hunks, shas = sample_hunks()
    print(f"sampled {len(hunks)} added-line blocks from {len(shas)} commits (seed {SEED})")

    records = []
    for index, hunk in enumerate(hunks, 1):
        hits = top5(hunk)
        verdicts = [judge(hunk, text, key) for _, text in hits]
        actionable = [n for (n, _), v in zip(hits, verdicts) if v is True]
        records.append({
            "index": index, "chars": len(hunk), "draft": hunk,
            "top5": [n for n, _ in hits], "actionable": actionable,
            "bearing": bool(actionable),
            "operation": bool(OPERATION_RE.search(hunk)),
        })
        if index % 25 == 0:
            print(f"  {index}/{len(hunks)}  bearing so far: "
                  f"{sum(1 for r in records if r['bearing'])}")

    p_repo = [r for r in records if r["bearing"]]
    n_repo = [r for r in records if not r["bearing"]]
    print(f"\nP_repo (>=1 actionable memo): {len(p_repo)}/{len(records)} "
          f"({len(p_repo)/len(records):.3f})")
    print(f"N_repo (0 actionable):        {len(n_repo)}/{len(records)}")

    import psycopg

    def vocabulary(df_max: int) -> set[str]:
        with psycopg.connect(args.dsn, connect_timeout=20) as conn:
            rows = conn.execute(
                "SELECT source_uri, string_agg(text,' ') FROM recall_chunks_v1 GROUP BY source_uri"
            ).fetchall()
        from collections import Counter
        df: Counter[str] = Counter()
        for _, text in rows:
            for token in {t.lower() for t in TOKEN_RE.findall(text or "")}:
                df[token] += 1
        return {t for t, c in df.items() if c <= df_max}

    triggers: dict[str, object] = {"T0_always": lambda r: True,
                                   "T3_operation": lambda r: r["operation"]}
    for df_max in (1, 2, 3, 5, 10):
        vocab = vocabulary(df_max)
        triggers[f"T2_vocab_df{df_max}"] = (
            lambda r, v=vocab: any(t.lower() in v for t in TOKEN_RE.findall(r["draft"]))
        )

    results = []
    for name, fires in triggers.items():
        cov = sum(1 for r in p_repo if fires(r))
        ft = sum(1 for r in n_repo if fires(r))
        results.append({
            "trigger": name,
            "coverage": cov, "of_p_repo": len(p_repo),
            "coverage_rate": round(cov / max(1, len(p_repo)), 4),
            "ft": ft, "of_n_repo": len(n_repo),
            "ft_rate": round(ft / max(1, len(n_repo)), 4),
        })
    if results[0]["coverage_rate"] != 1.0 or results[0]["ft_rate"] != 1.0:
        raise SystemExit(f"SCORING CONTROL FAILED: T0 must be 1.0/1.0, got {results[0]}")
    print("scoring control: OK (T0 fires on everything)\n")

    print(f"{'trigger':<20} {'coverage(P_repo)':>18} {'false-trigger(N_repo)':>23}")
    for r in results:
        print(f"{r['trigger']:<20} {r['coverage']:>5}/{r['of_p_repo']:<4} "
              f"{r['coverage_rate']:>6.3f}   {r['ft']:>5}/{r['of_n_repo']:<4} {r['ft_rate']:>6.3f}")

    # Registered Pareto selection, stated before any number was seen.
    frontier = [r for r in results[1:] if r["ft_rate"] <= 0.35]
    if frontier:
        chosen = max(frontier, key=lambda r: r["coverage_rate"])
        why = "max coverage subject to ft <= 0.35"
    else:
        eligible = [r for r in results[1:] if r["coverage_rate"] >= 0.60]
        if eligible:
            chosen = min(eligible, key=lambda r: r["ft_rate"])
            why = "min ft subject to coverage >= 0.60"
        else:
            chosen = max(results[1:], key=lambda r: r["coverage_rate"])
            why = "max coverage (both Pareto sets empty)"
    print(f"\nselected by the registered rule ({why}): {chosen['trigger']}  "
          f"coverage {chosen['coverage_rate']:.3f}  ft {chosen['ft_rate']:.3f}")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "repo-negatives.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": SEED, "commits": shas, "sample": len(records),
        "judge_model": JUDGE_MODEL, "judge_prompt": JUDGE_PROMPT,
        "judge_control": {"passed": passed, "of": len(known), "floor": CONTROL_FLOOR},
        "p_repo": len(p_repo), "n_repo": len(n_repo),
        "results": results, "selected": chosen, "selection_rule": why,
        "records": records,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
