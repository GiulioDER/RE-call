"""Live activation and parity receipt for C8 atomic rescue through the hosted API.

Runs against an isolated C8 service (VPS3 only; VPS2 serves the official AML run and must not be
touched). Two subcommands:

``ingest``
    Adds every frozen CAMBench session through ``/v1/add`` with the payload shape the C7/C8
    qualification used: one user message holding the rendered transcript per session.

``search``
    Replays the reference probes (one split) and the 34 task prompts through ``/v1/search``,
    records the atomic rescue, graph and routing headers of every response, and maps each served
    item back to its window by exact content so the gold windows of the offline reference can be
    scored on the live ranking. Run once before the artifact exists (the live off control, which
    must fail closed) and once after (the live on arm).

No quality claim is made from this receipt; it proves activation, fail-closed behaviour and the
direction of the live effect.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import c8_atomizer_reference as ref  # noqa: E402
from scripts.aml_c7_qualification import HttpClient, load_frozen_corpus  # noqa: E402


HEADERS = (
    "X-Recall-Atomic-Rescue-Attempted",
    "X-Recall-Atomic-Rescue-Active",
    "X-Recall-Atomic-Rescue-Fallback",
    "X-Recall-Atomic-Rescue-Candidate-Available",
    "X-Recall-Graph-Attempted",
    "X-Recall-Graph-Fallback",
    "X-Recall-Specialist-Route",
    "X-Recall-Generation",
    "X-Recall-Corpus-SHA256",
    "X-Recall-Variant",
    "X-Recall-Served-Commit",
    "X-Recall-Search-Ms",
)


def _header(headers: Mapping[str, str], name: str) -> str:
    lowered = {key.lower(): value for key, value in headers.items()}
    return lowered.get(name.lower(), "")


def ingest(
    client: HttpClient, rendered: Mapping[str, str], user_id: str, *, workers: int = 1
) -> dict[str, Any]:
    """Add every session; one worker by default.

    Concurrent Adds for one user queue on the service's tenant advisory lock, which is held for
    the whole Add, and a wait longer than the pool's 25 s statement timeout returns 503 (measured
    2026-09-22: 11 of 60 Adds at three workers). Add is idempotent by request id, so a rerun
    retries only the gaps.
    """
    def add_one(item: tuple[str, str]) -> dict[str, Any]:
        session, text = item
        call = client.call(
            "/v1/add",
            {
                "request_id": "atomizer-c8-" + hashlib.sha256(session.encode()).hexdigest(),
                "messages": [{"role": "user", "content": text}],
                "user_id": user_id,
                "session_id": session,
            },
        )
        return {"session": session, "status": call.status, "raw_count": call.payload.get("raw_count")}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(add_one, sorted(rendered.items())))
    failed = [row for row in rows if row["status"] != 200]
    return {
        "sessions": len(rows),
        "failed": failed,
        "raw_windows": sum(int(row["raw_count"] or 0) for row in rows),
    }


def score_live(
    data: Sequence[Mapping[str, Any]],
    gold_texts: frozenset[str],
    gold_sessions: frozenset[str],
) -> tuple[int | None, int | None, list[str]]:
    """Exact-gold and source ranks of one live response, matched by window content."""

    exact: int | None = None
    source: int | None = None
    order: list[str] = []
    for rank, item in enumerate(data, start=1):
        content = item.get("content")
        order.append(str(item.get("id")))
        if exact is None and isinstance(content, str) and content in gold_texts:
            exact = rank
        if source is None and str(item.get("session_id")) in gold_sessions:
            source = rank
    return exact, source, order


def search(args: argparse.Namespace, client: HttpClient) -> dict[str, Any]:
    corpus = load_frozen_corpus(args.amb_root)
    windows = ref.build_windows(dict(corpus.rendered))
    text_of = {(window.session, window.segment): window.chunk.text for window in windows}
    queries: list[tuple[str, str, str, frozenset[str], frozenset[str]]] = []
    for line in args.probes.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] != args.split:
            continue
        queries.append(
            (
                row["probe_id"],
                "probe",
                row["question"],
                frozenset(text_of[(row["session"], segment)] for segment in row["gold_segments"]),
                frozenset({row["session"]}),
            )
        )
    for task_id, prompt, gold in corpus.tasks:
        texts = frozenset(window.chunk.text for window in windows if window.session in gold)
        queries.append((task_id, "task", prompt, texts, frozenset(gold)))

    def one(query: tuple[str, str, str, frozenset[str], frozenset[str]]) -> dict[str, Any]:
        query_id, kind, text, gold_texts, gold_sessions = query
        call = client.call("/v1/search", {"query": text, "user_id": args.user_id, "top_k": 100})
        data = call.payload.get("data", []) if call.status == 200 else []
        exact, source, order = score_live(data, gold_texts, gold_sessions)
        return {
            "query_id": query_id,
            "kind": kind,
            "status": call.status,
            "exact_rank": exact,
            "source_rank": source,
            "top8": order[:8],
            "headers": {name: _header(call.headers, name) for name in HEADERS},
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(one, queries))
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report: dict[str, Any] = {"label": args.label, "split": args.split}
    for kind in ("probe", "task"):
        selected = [row for row in rows if row["kind"] == kind]
        counts: dict[str, Any] = {"queries": len(selected)}
        counts["http_errors"] = sum(1 for row in selected if row["status"] != 200)
        for name in HEADERS[:6]:
            counts[name.removeprefix("X-Recall-").lower()] = sum(
                1 for row in selected if row["headers"][name] == "1"
            )
        routes: dict[str, int] = {}
        for row in selected:
            route = row["headers"]["X-Recall-Specialist-Route"] or "none"
            routes[route] = routes.get(route, 0) + 1
        counts["routes"] = routes
        for cutoff in ref.CUTOFFS:
            counts[f"exact@{cutoff}"] = sum(
                1 for row in selected if row["exact_rank"] is not None and row["exact_rank"] <= cutoff
            )
            counts[f"source@{cutoff}"] = sum(
                1 for row in selected if row["source_rank"] is not None and row["source_rank"] <= cutoff
            )
        counts["identities"] = sorted(
            {
                (row["headers"]["X-Recall-Generation"], row["headers"]["X-Recall-Corpus-SHA256"])
                for row in selected
            }
        )
        report[kind] = counts
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key-env", default="RECALL_AML_API_KEY")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--amb-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("ingest")
    add.add_argument("--workers", type=int, default=1)
    run = sub.add_parser("search")
    run.add_argument("--probes", type=Path, required=True)
    run.add_argument("--split", choices=("dev", "confirm"), required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")
    if "vps2" in args.url.lower():
        raise SystemExit("REFUSED: VPS2 serves the official AML run; use the VPS3 service")
    client = HttpClient(args.url, api_key)
    if args.command == "ingest":
        summary = ingest(
            client,
            dict(load_frozen_corpus(args.amb_root).rendered),
            args.user_id,
            workers=args.workers,
        )
    else:
        summary = search(args, client)
    print(json.dumps(summary, indent=2, sort_keys=True, default=list))


if __name__ == "__main__":
    main()
