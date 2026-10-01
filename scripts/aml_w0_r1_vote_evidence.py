"""W0 R1-Coding: the evidence file for the Task Solve replay of the served session vote.

Pre-registration: kept in the maintainer's private research log (W0 Task Solve of the session vote,
2026-10-01).

Turns R1-Coding's recorded top 100 lists into the top ``k`` windows per task for three arms, in the
schema agent-memory-bench's ``c9_retrieval_replay`` adapter reads:

- ``c9_a_replay``: C9 (Voyage) as served;
- ``c9_p_replay``: the Qwen3 proxy as served;
- ``c9_pv_replay``: the proxy re-ordered by the session vote measured offline (``L2+first``: BM25
  weight 2 over the served order, then each session's best window, sessions by summed vote).

    python scripts/aml_w0_r1_vote_evidence.py --a A.json.gz --p P.json.gz --amb-root AMB --out evidence.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXPERIMENT = "c9-session-vote-task-solve-evidence"
ARMS = {
    "c9_a_replay": "C9 served (voyage-code-4 code route)",
    "c9_p_replay": "Qwen3-Embedding-8B proxy served",
    "c9_pv_replay": "proxy, session vote L2+first",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: The third slot's content for the user-message keys run (pre-registered 2026-10-01, 309a95e).
USER_KEYS_ARM = "proxy, user-message session keys L2+K(1)"


def arm_order(arm: str, items: Sequence[dict[str, Any]], query: str, key_scores: dict[str, float] | None = None) -> list[int]:
    """The window indices an arm shows, best first: served for A and P; for the third slot,
    ``L2+first`` (session vote), or with ``key_scores`` ``L2+K(1)`` (user-message keys)."""
    from aml_w0_r1_offline import lexical_order, session_vote_order
    from aml_w0_task_keys import keyed_order

    if arm in ("c9_a_replay", "c9_p_replay"):
        return list(range(len(items)))
    sessions = [str(i["session_id"]) for i in items]
    base = lexical_order(query, [str(i["content"]) for i in items], 2.0)
    if key_scores is not None:
        return keyed_order(base, sessions, key_scores, 1.0)
    return session_vote_order(base, sessions, "first")


def windows(items: Sequence[dict[str, Any]], order: Sequence[int], k: int) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "session_id": str(items[i]["session_id"]), "text": str(items[i]["content"]), "text_sha256": sha256(str(items[i]["content"]))}
        for rank, i in enumerate(order[:k], start=1)
    ]


def build(
    rows: dict[str, dict[str, Any]], questions: dict[str, Any], k: int, user_key_scores: dict[str, dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    """Per task: the prompt hash and each arm's top ``k`` windows. A reads A's collect; P and the
    third slot P's; with ``user_key_scores`` (task -> session -> score) the third slot is the
    user-message keys arm instead of the session vote."""
    tasks = []
    for task in sorted(t for t in rows["A"] if t in rows["P"] and t in questions):
        query = questions[task].query
        arms = {}
        for arm in ARMS:
            items = rows["A" if arm == "c9_a_replay" else "P"][task]["served_facts"]["kept"]
            if len(items) < k:
                raise ValueError(f"{task} {arm}: {len(items)} items, fewer than k {k}")
            scores = None if user_key_scores is None else user_key_scores.get(task, {})
            arms[arm] = windows(items, arm_order(arm, items, query, scores), k)
        tasks.append({"task_id": task, "query_sha256": sha256(query), "arms": arms})
    return tasks


def main(argv: Sequence[str] | None = None) -> int:
    from aml_w0_r1_coding import _questions, _rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--p", type=Path, required=True)
    parser.add_argument("--amb-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user-key-scores", type=Path, help="task key scores; the third slot becomes the user-message keys arm")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the evidence file is written once")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    user_scores = None
    if args.user_key_scores:
        user_scores = json.loads(args.user_key_scores.read_text(encoding="utf-8"))["P"]["U"]
    tasks = build({"A": _rows(args.a), "P": _rows(args.p)}, _questions(args.amb_root), args.k, user_scores)
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.a, args.p)}
    artifact = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "provenance": {
            "source": "; ".join(f"{name} sha256 {digest}" for name, digest in sources.items()),
            "source_files": sources,
            "recall_commit": commit,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "configuration": {"k": args.k, "arms": {**ARMS, "c9_pv_replay": USER_KEYS_ARM} if user_scores is not None else ARMS},
        "tasks": tasks,
    }
    args.out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "k": args.k, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
