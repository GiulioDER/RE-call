"""Bound C9's Code4 or Context4 routing choice on LoCoMo, and replay a gpt-4o-mini router.

Pre-registration: docs/preregistrations/2026-09-24-c9-code4-context4-routing-headroom.md

The 2026-09-23 route comparison served every LoCoMo question three times through the real C9
app: with the keyword router, forced to Code4, and forced to Context4. Given those per-question
results, any other per-question route choice can be scored without re-running retrieval, because
choosing a route is choosing which forced arm's result the question gets. The apparatus check
verifies that premise on the served router's own choices before anything else is reported.

    python scripts/aml_c9_route_headroom.py --artifact route.json.gz --data locomo10.json \
        --llm-cache llm_routes.jsonl --out headroom.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

PINNED_DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
MODEL = "openai/gpt-4o-mini"
METRICS = ("turn_hit@5", "turn_hit@10", "turn_hit@20", "turn_hit@100", "session_hit@10")
ROUTER_PROMPT = (
    "You route a memory search query to one of two retrieval indexes built over the same "
    "conversation history. A: a contextual embedder that embeds every passage together with its "
    "surrounding conversation. It is strongest when the answer sits in a turn whose meaning "
    "depends on context: pronouns, follow ups, implicit references, events described across "
    "several turns. B: a precise embedder fused with keyword search. It is strongest when the "
    "question names specific people, objects, titles, places, numbers or distinctive words "
    "likely to appear verbatim in the answer turn. Reply with exactly one letter, A or B."
)


def load_questions(raw: bytes) -> dict[str, str]:
    questions: dict[str, str] = {}
    for sample in json.loads(raw):
        for index, qa in enumerate(sample["qa"]):
            questions[f"{sample['sample_id']}:{index}"] = qa["question"]
    return questions


def arm_for(route: str) -> str:
    """The forced arm that stands in for a route. Multimodal is served from the Code4 store."""
    return "context" if route == "context" else "code"


def apparatus_check(rows: list[dict[str, Any]]) -> dict[str, int]:
    checked = mismatched = 0
    for row in rows:
        route = row["router_route"]
        if route not in ("code", "context"):
            continue
        checked += 1
        if row["router"]["turn_hit@10"] != row[arm_for(route)]["turn_hit@10"]:
            mismatched += 1
    return {"checked": checked, "mismatched": mismatched}


def paired_bootstrap(left: list[int], right: list[int], resamples: int = 10_000) -> dict[str, Any]:
    """Left minus right, in points, with a percentile interval over questions."""
    n = len(left)
    diffs = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(0)
    samples = sorted(
        100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return {
        "delta_points": 100.0 * sum(diffs) / n,
        "ci95_low_points": samples[int(0.025 * resamples)],
        "ci95_high_points": samples[int(0.975 * resamples) - 1],
        "rescues": sum(d > 0 for d in diffs),
        "regressions": sum(d < 0 for d in diffs),
        "n": n,
    }


def llm_route(client: httpx.Client, question: str) -> str:
    for attempt in range(5):
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": MODEL,
                "temperature": 0,
                "max_tokens": 2,
                "messages": [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": question},
                ],
            },
        )
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()
    raise RuntimeError("OpenRouter kept failing")


def llm_routes(questions: dict[str, str], ids: list[str], cache: Path) -> dict[str, str]:
    replies: dict[str, str] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            replies[record["question_id"]] = record["reply"]
    missing = [qid for qid in ids if qid not in replies]
    if missing:
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        # One line per reply, flushed, so progress is visible and an interrupted run resumes.
        with httpx.Client(headers=headers, timeout=60) as client, cache.open(
            "a", encoding="utf-8"
        ) as sink, ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(llm_route, client, questions[qid]): qid for qid in missing}
            for future in as_completed(futures):
                qid = futures[future]
                replies[qid] = future.result()
                sink.write(json.dumps({"question_id": qid, "reply": replies[qid]}) + "\n")
                sink.flush()
    return replies


def reply_to_route(reply: str) -> str:
    return "context" if reply.strip().upper().startswith("A") else "code"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--llm-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    artifact = json.loads(gzip.decompress(args.artifact.read_bytes()))
    raw = args.data.read_bytes()
    data_sha256 = hashlib.sha256(raw).hexdigest()
    questions = load_questions(raw)
    rows = artifact["rows"]
    ids = [row["question_id"] for row in rows]
    if any(qid not in questions for qid in ids):
        raise SystemExit("artifact question ids do not join to the dataset")

    check = apparatus_check(rows)
    result: dict[str, Any] = {
        "preregistration": "docs/preregistrations/2026-09-24-c9-code4-context4-routing-headroom.md",
        "data_sha256": data_sha256,
        "data_matches_pinned": data_sha256 == PINNED_DATA_SHA256,
        "artifact_sha256": hashlib.sha256(gzip.decompress(args.artifact.read_bytes())).hexdigest(),
        "n": len(rows),
        "apparatus_check": check,
    }
    if check["mismatched"]:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise SystemExit(f"apparatus check failed: {check}")

    arms: dict[str, list[dict[str, int]]] = {
        "code": [row["code"] for row in rows],
        "context": [row["context"] for row in rows],
        "router": [row["router"] for row in rows],
        "oracle": [
            {m: max(row["code"][m], row["context"][m]) for m in METRICS} for row in rows
        ],
    }
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = by_category.setdefault(
            str(row["category"]), {"n": 0, "context_only": 0, "code_only": 0}
        )
        cell["n"] += 1
        cell["context_only"] += row["context"]["turn_hit@10"] > row["code"]["turn_hit@10"]
        cell["code_only"] += row["code"]["turn_hit@10"] > row["context"]["turn_hit@10"]
    result["turn_hit@10_disagreement_by_category"] = by_category

    if not args.skip_llm:
        replies = llm_routes(questions, ids, args.llm_cache)
        routes = [reply_to_route(replies[qid]) for qid in ids]
        arms["llm_router"] = [row[arm_for(route)] for row, route in zip(rows, routes, strict=True)]
        result["llm_router"] = {
            "model": MODEL,
            "routes": {r: routes.count(r) for r in ("code", "context")},
            "unparsed_replies": sum(
                replies[qid].strip().upper()[:1] not in ("A", "B") for qid in ids
            ),
            "agreement_with_keyword_router": sum(
                route == arm_for(row["router_route"])
                for row, route in zip(rows, routes, strict=True)
            ),
        }

    result["summary"] = {
        arm: {m: 100.0 * sum(r[m] for r in values) / len(values) for m in METRICS}
        for arm, values in arms.items()
    }
    comparisons = [("oracle", "code"), ("context", "code"), ("router", "code")]
    if "llm_router" in arms:
        comparisons += [("llm_router", "code"), ("llm_router", "router")]
    result["paired"] = {
        f"{left}_minus_{right}": {
            m: paired_bootstrap([r[m] for r in arms[left]], [r[m] for r in arms[right]])
            for m in ("turn_hit@5", "turn_hit@10", "session_hit@10")
        }
        for left, right in comparisons
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
