"""Re-derive draft-time search on EXECUTED ground truth: no judge, no human label.

    python -u scripts/agent_ab_checker_truth.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --archive ~/.claude/archive/agent-ab-skill-001

Preregistered in `docs/preregistrations/2026-08-27-checker-ground-truth.md`.

Every "was this memo needed" label used in this lane came from a model, three models disagreed
(κ 0.33 to 0.51), and a human could not settle it. `metadata.check` — the agent's own artifact run
against the task's checker — was in the archive throughout:

- **P_exec**: the session FAILED with evidence naming the trap its memo describes. The memo was
  needed.
- **N_exec**: the session PASSED. The memo was not needed for the outcome, so surfacing it would
  have been noise. This is the first objective negative population in the lane.
- **excluded**: failed before reaching the trap, so unknowable.

The classification is a lookup, not an inference: a session is a positive only when its checker
evidence matches its family's known trap phrase, so a failure for an unrelated reason cannot be
counted as one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TOP_K = 5
CANDIDATE_K = 200
MAX_QUERY_CHARS = 4096
VOCAB_DF = 2
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CONTROL_MEMO = "python-write-text-crlf-churn"
CONTROL_QUERY = "Path.write_text on Windows injects CRLF against a tree configured eol=lf"

#: A failure counts as "fell into THIS memo's trap" only if its evidence names the trap. Anything
#: else is indeterminate and read from no endpoint — a session that failed because it never wrote
#: the script says nothing about whether the memo would have mattered.
TRAP_EVIDENCE = {
    "ts-lf-rewrite": "carriage returns",
    "ts-sample-covers-tail": "head bias",
    "ts-autouse-tmp-path": "tmp_path",
    "ts-bounded-runner": "bound",
    "ts-worktree-import": "decoy",
    "ts-false-zero-search": "zero",
    "ts-separator-canary": "separator",
    "ts-raise-on-missing": "raise",
}


def sessions(archive: Path) -> list[dict]:
    rows = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = record.get("metadata") or {}
        if record.get("variant") != "recall_on" or metadata.get("locus") != "memory_only":
            continue
        check = metadata.get("check") or {}
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
        base = str(record.get("task_id", "")).split("#")[0]
        evidence = str(check.get("evidence") or "")
        passed = bool(check.get("passed"))
        if passed:
            group = "N_exec"
        elif TRAP_EVIDENCE.get(base, "\0") in evidence.lower():
            group = "P_exec"
        else:
            group = "excluded"
        rows.append({
            "task_id": record["task_id"], "base": base,
            "memo": str(metadata.get("governing_memo") or ""),
            "passed": passed, "evidence": evidence, "group": group,
            "drafts": [d for d in drafts if len(d) <= MAX_QUERY_CHARS],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import psycopg

    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall.retriever import HybridRetriever

    rows = sessions(Path(args.archive).expanduser())
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    print(f"sessions: {len(rows)}   P_exec {len(groups['P_exec'])}   "
          f"N_exec {len(groups['N_exec'])}   excluded {len(groups['excluded'])}")
    for row in groups["excluded"]:
        print(f"  excluded: {row['task_id']}  {row['evidence'][:70]}")
    if len(groups["P_exec"]) != 11 or len(groups["N_exec"]) != 36:
        raise SystemExit(
            f"expected the registered 11 positives and 36 negatives, got "
            f"{len(groups['P_exec'])}/{len(groups['N_exec'])}; the population is not the one the "
            "record fixes, so this is not run"
        )

    # A scored session with no declared memo would make `wanted` the string ".md", which matches
    # no filename, so every draft would silently score as "never surfaced" — a clean zero from a
    # missing field. Refuse instead.
    memoless = [r["task_id"] for r in groups["P_exec"] + groups["N_exec"] if not r["memo"]]
    if memoless:
        raise SystemExit(f"scored sessions with no governing_memo: {memoless}")

    embedder = resolve_embedder("fastembed")
    store = GenerationStore(args.dsn, dim=384, tenant="default")
    store.check_schema()
    lexical = HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_dense=False)

    def top5(query: str) -> list[str]:
        return [Path(str(h.chunk.source)).name
                for h in lexical.search(query, k=CANDIDATE_K).hits[:TOP_K]]

    if f"{CONTROL_MEMO}.md" not in top5(CONTROL_QUERY):
        raise SystemExit("POSITIVE CONTROL FAILED; nothing below is a measurement")
    print("positive control: OK")

    with psycopg.connect(args.dsn, connect_timeout=20) as conn:
        docs = conn.execute(
            "SELECT source_uri, string_agg(text,' ') FROM recall_chunks_v1 GROUP BY source_uri"
        ).fetchall()
    from collections import Counter
    df: Counter[str] = Counter()
    for _, text in docs:
        for token in {t.lower() for t in TOKEN_RE.findall(text or "")}:
            df[token] += 1
    vocab = {t for t, c in df.items() if c <= VOCAB_DF}
    print(f"vocabulary at df<={VOCAB_DF}: {len(vocab)} terms (fitted elsewhere)\n")

    results = []
    for row in groups["P_exec"] + groups["N_exec"] + groups["excluded"]:
        wanted = f"{row['memo']}.md"
        surfaced = triggered = False
        for draft in row["drafts"]:
            hit = wanted in top5(draft)
            fires = any(t.lower() in vocab for t in TOKEN_RE.findall(draft))
            surfaced |= hit
            triggered |= hit and fires
        results.append({**{k: row[k] for k in ("task_id", "base", "memo", "group", "passed")},
                        "drafts": len(row["drafts"]),
                        "surfaced": surfaced, "trigger_surfaced": triggered})
        print(f"  [{row['group']:<9}] {row['task_id']:<26} "
              f"{'surfaced' if surfaced else 'no':<9} "
              f"{'trigger fires' if triggered else ''}")

    def rate(group: str, field: str) -> tuple[int, int, float]:
        subset = [r for r in results if r["group"] == group]
        n = sum(1 for r in subset if r[field])
        return n, len(subset), round(n / max(1, len(subset)), 4)

    rp, np_, recall = rate("P_exec", "surfaced")
    rn, nn, fire = rate("N_exec", "surfaced")
    tp, _, trec = rate("P_exec", "trigger_surfaced")
    tn, _, tfire = rate("N_exec", "trigger_surfaced")

    print(f"\n{'endpoint':<32} {'count':>10} {'rate':>7}")
    print(f"{'recall_exec (needed it)':<32} {f'{rp}/{np_}':>10} {recall:>7.3f}")
    print(f"{'fire_exec (did NOT need it)':<32} {f'{rn}/{nn}':>10} {fire:>7.3f}")
    print(f"{'trigger_recall':<32} {f'{tp}/{np_}':>10} {trec:>7.3f}")
    print(f"{'trigger_fire':<32} {f'{tn}/{nn}':>10} {tfire:>7.3f}")

    band_r = "<0.60" if recall < 0.60 else ("0.60-0.79" if recall < 0.80 else ">=0.80")
    band_t = "<=0.30" if tfire <= 0.30 else ("0.31-0.60" if tfire <= 0.60 else ">0.60")
    print(f"\nregistered cell: recall {band_r}  x  trigger_fire {band_t}")

    out = Path(args.out) if args.out else Path(
        "benchmarks/artifacts/agent_ab/checker-truth.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "population": {k: len(v) for k, v in groups.items()},
        "vocab_df": VOCAB_DF, "vocab_terms": len(vocab),
        "endpoints": {
            "recall_exec": {"n": rp, "of": np_, "rate": recall},
            "fire_exec": {"n": rn, "of": nn, "rate": fire},
            "trigger_recall": {"n": tp, "of": np_, "rate": trec},
            "trigger_fire": {"n": tn, "of": nn, "rate": tfire},
        },
        "cell": {"recall": band_r, "trigger_fire": band_t},
        "sessions": results,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
