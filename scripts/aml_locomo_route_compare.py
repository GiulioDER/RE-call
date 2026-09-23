"""Search-only LoCoMo comparison of C9 specialist routing: the served router, all Context4, all Code4.

Pre-registration: ``docs/preregistrations/2026-09-23-aml-c9-locomo-route-comparison.md``.

The service is built in process by ``recall_aml.__main__.build_app`` from the ``RECALL_AML_*``
environment and driven through its real ASGI app. Every LoCoMo session is added once, then each
question is searched three times with ``recall_aml.service.route_query`` replaced per arm. The
route the replacement returned is recorded for every Search, so a forced arm that did not take its
route is visible in the output rather than assumed.

    python scripts/aml_locomo_route_compare.py --data locomo10.json --out result.json
    python scripts/aml_locomo_route_compare.py --data locomo10.json --out smoke.json \
        --limit-conversations 1 --limit-questions 20
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

ARMS = ("router", "context", "code")
DEPTHS = (5, 10, 20, 100)
PRIMARY_DEPTH = 10
ANSWERABLE_CATEGORIES = frozenset({1, 2, 3, 4})
EVIDENCE_ID = re.compile(r"^D(\d+):(\d+)$")
SESSION_KEY = re.compile(r"^session_(\d+)$")
EDGE_WORDS = 12
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260923
PINNED_DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def normalize(text: str) -> str:
    return " ".join(text.split())


def session_timestamp_ms(value: str) -> int:
    """Parse LoCoMo's ``"1:56 pm on 8 May, 2023"`` as UTC Unix milliseconds."""
    parsed = datetime.strptime(normalize(value), "%I:%M %p on %d %B, %Y")
    return int(parsed.replace(tzinfo=timezone.utc).timestamp() * 1_000)


def turn_content(turn: dict[str, Any]) -> str:
    content = f"{turn['speaker']}: {turn['text']}"
    caption = turn.get("blip_caption")
    if isinstance(caption, str) and caption.strip():
        content += f" [shared image: {caption.strip()}]"
    return content


def evidence_ids(raw: Iterable[str]) -> list[str]:
    """Split packed ids such as ``"D9:1 D4:4"`` and keep only well-formed dia ids, in order."""
    found: list[str] = []
    for entry in raw:
        for fragment in re.split(r"[\s,;]+", str(entry)):
            if EVIDENCE_ID.match(fragment) and fragment not in found:
                found.append(fragment)
    return found


def turn_present(turn: str, item: str) -> bool:
    """True when an evidence turn's text survives in one returned item.

    A turn longer than ``EDGE_WORDS`` words may be split across two 160-word windows, so its first
    or last ``EDGE_WORDS`` words are enough.
    """
    turn_text, item_text = normalize(turn), normalize(item)
    if not turn_text:
        return False
    if turn_text in item_text:
        return True
    words = turn_text.split(" ")
    if len(words) <= EDGE_WORDS:
        return False
    return (
        " ".join(words[:EDGE_WORDS]) in item_text
        or " ".join(words[-EDGE_WORDS:]) in item_text
    )


def item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


def score(
    items: Sequence[dict[str, Any]],
    gold_turns: Sequence[str],
    gold_sessions: set[str],
) -> dict[str, int]:
    row: dict[str, int] = {}
    for depth in DEPTHS:
        window = items[:depth]
        row[f"turn_hit@{depth}"] = int(
            any(turn_present(turn, item_text(item)) for item in window for turn in gold_turns)
        )
        row[f"session_hit@{depth}"] = int(
            any(str(item.get("session_id", "")) in gold_sessions for item in window)
        )
    # Compiled graph records and atomic views paraphrase or slice turns, so they count here.
    row["non_raw_in_top10"] = sum(
        1 for item in items[:PRIMARY_DEPTH] if str(item.get("kind", "")) not in {"raw", "multimodal"}
    )
    return row


def paired_bootstrap(
    control: Sequence[int], treatment: Sequence[int], *, seed: int = BOOTSTRAP_SEED
) -> dict[str, float]:
    """Percentile 95% interval of mean(treatment - control) in points, resampling questions."""
    if len(control) != len(treatment) or not control:
        raise ValueError("paired samples must be non-empty and equal in length")
    deltas = [t - c for c, t in zip(control, treatment)]
    n = len(deltas)
    rng = random.Random(seed)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "delta_points": 100 * sum(deltas) / n,
        "ci95_low_points": 100 * means[int(0.025 * BOOTSTRAP_RESAMPLES)],
        "ci95_high_points": 100 * means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
        "rescues": sum(1 for d in deltas if d > 0),
        "regressions": sum(1 for d in deltas if d < 0),
    }


def build_corpus(
    data: list[dict[str, Any]], run_id: str, limit_conversations: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (add requests, questions) with gold turns and sessions resolved against ingestion."""
    adds: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for sample in data[:limit_conversations]:
        sample_id = str(sample["sample_id"])
        user_id = f"locomo-route-{run_id}-{sample_id}"
        conversation = sample["conversation"]
        turns: dict[str, tuple[str, str]] = {}
        for key in sorted(
            (k for k in conversation if SESSION_KEY.match(k)),
            key=lambda k: int(SESSION_KEY.match(k).group(1)),  # type: ignore[union-attr]
        ):
            session = conversation[key]
            if not session:
                continue
            session_id = f"{sample_id}:{key}"
            timestamp = session_timestamp_ms(conversation[f"{key}_date_time"])
            messages = []
            for turn in session:
                content = turn_content(turn)
                turns[str(turn["dia_id"])] = (content, session_id)
                messages.append({"role": "user", "content": content, "timestamp": timestamp})
            adds.append(
                {
                    "request_id": hashlib.sha256(
                        f"{run_id}:{session_id}".encode()
                    ).hexdigest()[:32],
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": messages,
                }
            )
        for index, qa in enumerate(sample["qa"]):
            if qa.get("category") not in ANSWERABLE_CATEGORIES:
                continue
            gold = [turns[dia] for dia in evidence_ids(qa.get("evidence", [])) if dia in turns]
            if not gold:
                continue
            questions.append(
                {
                    "question_id": f"{sample_id}:{index}",
                    "category": qa["category"],
                    "user_id": user_id,
                    "query": str(qa["question"]),
                    "gold_turns": [content for content, _ in gold],
                    "gold_sessions": sorted({session_id for _, session_id in gold}),
                }
            )
    return adds, questions


def run(args: argparse.Namespace) -> dict[str, Any]:
    from starlette.testclient import TestClient

    import recall_aml.service as service_module
    from recall_aml.__main__ import build_app
    from recall_aml.specialists import route_query as served_route_query

    raw = args.data.read_bytes()
    data_sha256 = hashlib.sha256(raw).hexdigest()
    adds, questions = build_corpus(json.loads(raw), args.run_id, args.limit_conversations)
    questions = questions[: args.limit_questions]
    # Canary: the verbatim text of one ingested evidence turn as the query. Every arm must find
    # it at rank 10 or better, or the harness is measuring something other than retrieval.
    canary_turn = questions[0]["gold_turns"][0]
    questions.insert(
        0,
        {
            **questions[0],
            "question_id": "canary",
            "category": 0,
            "query": canary_turn,
            "gold_turns": [canary_turn],
        },
    )

    taken: list[str] = []
    forced: dict[str, str | None] = {"route": None}

    def replacement(value: Any) -> str:
        route = forced["route"] or served_route_query(value)
        taken.append(route)
        return route

    service_module.route_query = replacement  # type: ignore[assignment]
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    failures: Counter[str] = Counter()
    started = time.perf_counter()
    with TestClient(build_app()) as client:
        version = client.get("/version", headers=headers).json()
        if version.get("variant") != args.expected_variant:
            raise SystemExit(f"served variant {version.get('variant')!r} is not {args.expected_variant!r}")
        add_fallbacks = 0
        for position, request in enumerate(adds, start=1):
            response = client.post("/v1/add", json=request, headers=headers)
            if response.status_code != 200:
                failures[f"add_{response.status_code}"] += 1
                continue
            add_fallbacks += int(bool(response.json().get("compiler_fallback")))
            if position % 25 == 0:
                print(f"added {position}/{len(adds)}", file=sys.stderr, flush=True)
        add_seconds = time.perf_counter() - started

        rows: list[dict[str, Any]] = []
        for position, question in enumerate(questions, start=1):
            row: dict[str, Any] = {
                "question_id": question["question_id"],
                "category": question["category"],
                "offline_route": served_route_query(question["query"]),
            }
            for arm in ARMS:
                forced["route"] = None if arm == "router" else arm
                before = len(taken)
                response = client.post(
                    "/v1/search",
                    json={"query": question["query"], "user_id": question["user_id"], "top_k": 100},
                    headers=headers,
                )
                row[f"{arm}_route"] = taken[before] if len(taken) > before else None
                if response.status_code != 200:
                    failures[f"search_{arm}_{response.status_code}"] += 1
                    row[arm] = None
                    continue
                items = response.json()["data"]
                row[arm] = score(items, question["gold_turns"], set(question["gold_sessions"]))
                row[f"{arm}_items"] = len(items)
            rows.append(row)
            if position % 50 == 0:
                print(f"searched {position}/{len(questions)}", file=sys.stderr, flush=True)
        for user_id in sorted({request["user_id"] for request in adds}):
            client.post("/v1/delete", json={"user_id": user_id}, headers=headers)

    canary = rows[0]
    rows = rows[1:]
    complete = [row for row in rows if all(row[arm] is not None for arm in ARMS)]
    summary: dict[str, Any] = {"n_questions": len(rows), "n_scored": len(complete)}
    for arm in ARMS:
        summary[arm] = {
            metric: sum(row[arm][metric] for row in complete) / max(1, len(complete))
            for metric in complete[0][arm]
        } if complete else {}
    for arm in ARMS[1:]:
        summary[f"{arm}_minus_router"] = {
            metric: paired_bootstrap(
                [row["router"][metric] for row in complete],
                [row[arm][metric] for row in complete],
            )
            for metric in (f"turn_hit@{PRIMARY_DEPTH}", "turn_hit@100", f"session_hit@{PRIMARY_DEPTH}")
        } if complete else {}
    return {
        "preregistration": "docs/preregistrations/2026-09-23-aml-c9-locomo-route-comparison.md",
        "run_id": args.run_id,
        "data_sha256": data_sha256,
        "data_matches_pinned": data_sha256 == PINNED_DATA_SHA256,
        "variant": version.get("variant"),
        "git_commit": version.get("git_commit") or os.environ.get("RECALL_AML_GIT_COMMIT"),
        "n_adds": len(adds),
        "add_compiler_fallbacks": add_fallbacks,
        "add_seconds": round(add_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "failures": dict(failures),
        "router_route_counts": dict(Counter(row["router_route"] for row in rows)),
        "router_matches_offline": sum(row["router_route"] == row["offline_route"] for row in rows),
        "forced_route_violations": {
            arm: sum(row[f"{arm}_route"] != arm for row in rows) for arm in ARMS[1:]
        },
        "canary_turn_hit@10": {
            arm: (canary[arm] or {}).get("turn_hit@10") for arm in ARMS
        },
        "summary": summary,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None, runner: Callable[[argparse.Namespace], dict[str, Any]] = run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument(
        "--expected-variant", default="C9_routed_specialists_grounded_graph_atomic"
    )
    args = parser.parse_args(argv)
    result = runner(args)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
