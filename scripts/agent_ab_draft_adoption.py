"""Stage 1 endpoint: did the instruction change what the agent QUERIES WITH?

    python -u scripts/agent_ab_draft_adoption.py \
        --run benchmarks/artifacts/agent_ab/draft-query-v3-stage1 \
        --baseline ~/.claude/archive/agent-ab-skill-001

Preregistered in `docs/preregistrations/2026-08-27-deliberate-draft-search.md`.

The endpoint is mechanical and needs no judge: of on-arm sessions that searched, what share issued
at least one `recall_search` query that is a **verbatim substring of a payload the same session
wrote LATER**, whitespace-normalised, at least 20 characters matched.

Ordering is load-bearing. A query matching a payload the agent had ALREADY written is not the
behaviour under test — that is searching about what you just did. The instruction says search
before you save, so only payloads appearing after the query in the tool-call sequence count.

The same metric is computed on the BASELINE run, which used `hazard-query-v2.txt`, because the
registered gate is about a change in behaviour and a rate without its predecessor is not one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MIN_MATCH = 20
WS = re.compile(r"\s+")
#: The two `ctl-*` families are the benchmark's controls. The registration fixes the population as
#: the 8 `ts-*` families; the harness runs all 10, so they are scored separately, never pooled.
CONTROL_FAMILIES = ("ctl-lint-only-check", "ctl-stage-by-pathspec")


def norm(text: str) -> str:
    return WS.sub(" ", text).strip().lower()


def session_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("variant") != "recall_on":
            continue
        queries, payloads = [], []
        for index, call in enumerate(record.get("tool_calls") or []):
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            if "recall_search" in name and args.get("query"):
                queries.append((index, str(args["query"])))
            elif name in ("Write", "Edit", "NotebookEdit"):
                body = str(args.get("content") or args.get("new_string") or "")
                if body.strip():
                    payloads.append((index, body))
            elif name == "Bash" and args.get("command"):
                payloads.append((index, str(args["command"])))
        rows.append({
            "task_id": str(record.get("task_id", "")),
            "base": str(record.get("task_id", "")).split("#")[0],
            "queries": queries, "payloads": payloads,
            "searched": bool(queries),
        })
    return rows


def adopted(row: dict) -> tuple[bool, str | None]:
    """Did any query appear verbatim inside a payload written LATER in the same session?"""

    for q_index, query in row["queries"]:
        needle = norm(query)
        if len(needle) < MIN_MATCH:
            continue
        for p_index, payload in row["payloads"]:
            if p_index > q_index and needle in norm(payload):
                return True, query[:70]
    return False, None


def score(rows: list[dict], label: str) -> dict:
    ts = [r for r in rows if r["base"] not in CONTROL_FAMILIES]
    ctl = [r for r in rows if r["base"] in CONTROL_FAMILIES]
    out = {}
    for name, subset in (("ts", ts), ("ctl", ctl)):
        searched = [r for r in subset if r["searched"]]
        hits = [r for r in searched if adopted(r)[0]]
        out[name] = {
            "sessions": len(subset), "searched": len(searched), "adopted": len(hits),
            "rate": round(len(hits) / max(1, len(searched)), 4),
        }
    print(f"\n{label}")
    for name, block in out.items():
        print(f"  {name:<4} sessions {block['sessions']:>3}  searched {block['searched']:>3}  "
              f"adopted {block['adopted']:>3}  rate {block['rate']:.3f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run = session_rows(Path(args.run).expanduser() / "records.jsonl")
    base = session_rows(Path(args.baseline).expanduser() / "records.jsonl")
    print(f"treatment on-arm sessions: {len(run)}   baseline on-arm sessions: {len(base)}")

    treat = score(run, "TREATMENT (draft-query-v3)")
    control = score(base, "BASELINE (hazard-query-v2)")

    print("\nper session, treatment ts-* families:")
    for row in sorted(run, key=lambda r: r["task_id"]):
        if row["base"] in CONTROL_FAMILIES:
            continue
        ok, example = adopted(row)
        mark = "ADOPTED" if ok else ("no      " if row["searched"] else "no search")
        print(f"  {row['task_id']:<26} {mark}  q={len(row['queries'])} w={len(row['payloads'])}"
              + (f"  {example!r}" if example else ""))

    rate = treat["ts"]["rate"]
    gate = "PASS -> stage 2 licensed" if rate >= 0.40 else "FAIL -> stage 2 is NOT run"
    print(f"\nregistered gate (>= 0.40 on the ts-* families): {rate:.3f}  {gate}")
    print(f"baseline for the same metric: {control['ts']['rate']:.3f}")

    out = Path(args.out) if args.out else Path(
        "benchmarks/artifacts/agent_ab/draft-adoption.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "min_match_chars": MIN_MATCH, "control_families": list(CONTROL_FAMILIES),
        "treatment": treat, "baseline": control,
        "gate": {"threshold": 0.40, "value": rate, "passed": rate >= 0.40},
        "sessions": [
            {"task_id": r["task_id"], "base": r["base"], "searched": r["searched"],
             "queries": len(r["queries"]), "payloads": len(r["payloads"]),
             "adopted": adopted(r)[0], "example": adopted(r)[1]}
            for r in run
        ],
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
