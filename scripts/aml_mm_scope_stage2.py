"""Stage 2 of the MM-1/MM-3 pre-registration: MemEye MCQ answers from the stored Stage 1 Searches.

Pre-registration: ``docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md``,
amendment 1 (reader ``deepseek/deepseek-v4.1-flash``, credit floor, no gpt-4o-mini).

Answers are built only from Stage 1's stored responses, so every arm is answered on exactly the
evidence its Search returned; images are restored byte-exact from the MemEye cache. The packer,
prompt and choice extraction are the frozen MemEye harness's (``pack_answer_content``,
``sys_prompt_mcq.txt``, ``extract_choice``). Which rows are answered, per the record:

* B, B2, P and D on questions whose B route is not ``multimodal`` (the MM-1 stratum);
* D and Dt (D with ``dated_multimodal_items`` applied) on every question (MM-3).

Usage::

    python scripts/aml_mm_scope_stage2.py --results S1.jsonl --cache-dir DIR --out S2.jsonl \\
        [--probe N] [--max-usd 15] [--workers 4]

``--probe N`` answers N rows only and prints their token use and cost, for the apparatus check.
Rows already present in ``--out`` are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recall_aml.window_format import dated_multimodal_items  # noqa: E402
from scripts.aml_mm_scope_report import image_index, load_questions, rebuild  # noqa: E402
from scripts.aml_mm_scope_stage1 import CREDIT_FLOOR_USD, credit_balance  # noqa: E402
from scripts.aml_multimodal_memeye import (  # noqa: E402
    GITHUB_RAW_BASE,
    MEMEYE_COMMIT,
    PROMPT_PATH,
    extract_choice,
    pack_answer_content,
)

READER_MODEL = "deepseek/deepseek-v4.1-flash"
MAX_OUTPUT_TOKENS = 16


def load_prompt(cache_dir: Path) -> str:
    path = cache_dir / "sys_prompt_mcq.txt"
    if not path.exists():
        request = Request(f"{GITHUB_RAW_BASE}/{MEMEYE_COMMIT}/{PROMPT_PATH}")
        with urlopen(request, timeout=60) as response:  # noqa: S310, pinned host
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def load_options(cache_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    identity = json.loads((cache_dir / "identity.json").read_text(encoding="utf-8"))
    options: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scenario in identity["scenarios"]:
        dataset = json.loads((cache_dir / f"{scenario}.json").read_text(encoding="utf-8"))
        for qa in dataset["human-annotated QAs"]:
            options[(scenario, str(qa["question_id"]))] = qa["options"]
    return options


def ask(key: str, system_prompt: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "model": READER_MODEL,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"enabled": False},
        "usage": {"include": True},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parts},
        ],
    }
    body = json.dumps(payload).encode()
    last_error = ""
    for attempt in range(1, 4):
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=600) as response:  # noqa: S310, pinned host
                result = json.loads(response.read())
            choices = result.get("choices") or []
            text = (choices[0].get("message", {}).get("content") or "") if choices else ""
            usage = result.get("usage") or {}
            return {
                "text": text,
                "attempts": attempt,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "cost_usd": float(usage.get("cost") or 0.0),
                "provider": result.get("provider"),
            }
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read()[:200]!r}"
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, OSError) as exc:
            last_error = repr(exc)
        time.sleep(2.0**attempt)
    return {"text": "", "error": last_error, "attempts": 3, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def plan(results: Path) -> tuple[list[tuple[dict[str, Any], str]], dict[tuple[str, str], str]]:
    """Which stored Searches to answer, as (row, answer arm) pairs, per the record."""
    rows: list[dict[str, Any]] = []
    with results.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    b_route = {(row["scenario"], row["question_id"]): row["route"] for row in rows if row["arm"] == "B"}
    jobs: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        key = (row["scenario"], row["question_id"])
        off_route = b_route[key] != "multimodal"
        if row["arm"] in {"B", "B2", "P"} and off_route:
            jobs.append((row, row["arm"]))
        if row["arm"] == "D":
            # D is answered on every question: MM-1 reads its off-route stratum, MM-3 all of it.
            jobs.append((row, "D"))
            jobs.append((row, "Dt"))
    return jobs, b_route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe", type=int, default=0)
    parser.add_argument("--max-usd", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY must be set")

    questions = load_questions(args.cache_dir)
    options = load_options(args.cache_dir)
    images = image_index(args.cache_dir)
    system_prompt = load_prompt(args.cache_dir)
    jobs, _ = plan(args.results)
    done: set[tuple[str, str, int, str]] = set()
    spent = 0.0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["scenario"], row["question_id"], row["rotation"], row["arm"]))
            spent += float(row.get("cost_usd") or 0.0)
    pending = [job for job in jobs if (job[0]["scenario"], job[0]["question_id"], job[0]["rotation"], job[1]) not in done]
    if args.probe:
        pending = pending[: args.probe]
    balance = credit_balance(key)
    print(json.dumps({"jobs": len(jobs), "done": len(done), "pending": len(pending), "spent_usd": spent, "balance_usd": balance}), flush=True)
    if balance < CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} below the {CREDIT_FLOOR_USD:.0f} USD floor")

    lock = threading.Lock()
    stop = threading.Event()
    counter = {"n": 0}

    def work(job: tuple[dict[str, Any], str]) -> None:
        nonlocal spent
        if stop.is_set():
            return
        row, arm = job
        question = questions[(row["scenario"], row["question_id"])]
        rotation = options[(row["scenario"], row["question_id"])][row["rotation"]]
        choice_map = {k: str(v) for k, v in rotation.items() if k != "answer"}
        items = rebuild(row["items"], images)
        if arm == "Dt":
            items = dated_multimodal_items(items)
        parts, packing = pack_answer_content(
            [item.model_dump(mode="json") for item in items], question["question"], choice_map
        )
        answer = ask(key, system_prompt, parts)
        selected = extract_choice(answer["text"], set(choice_map)) if answer["text"] else "INVALID"
        record = {
            "scenario": row["scenario"],
            "question_id": row["question_id"],
            "rotation": row["rotation"],
            "arm": arm,
            "route": row["route"],
            "selected": selected,
            "valid": selected != "INVALID",
            "em": float(selected == str(rotation["answer"]).upper()),
            "admitted_items": packing["admitted_items"],
            **answer,
        }
        with lock:
            spent += record["cost_usd"]
            with args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record) + "\n")
            counter["n"] += 1
            if args.probe:
                print(json.dumps({k: record[k] for k in ("arm", "selected", "em", "text", "prompt_tokens", "completion_tokens", "cost_usd", "admitted_items")}), flush=True)
            if spent > args.max_usd:
                stop.set()
                print(f"STOP: spend {spent:.2f} over the {args.max_usd:.2f} USD cap", flush=True)
            if counter["n"] % 100 == 0:
                balance_now = credit_balance(key)
                print(json.dumps({"answered": counter["n"], "spent_usd": round(spent, 4), "balance_usd": balance_now}), flush=True)
                if balance_now < CREDIT_FLOOR_USD:
                    stop.set()
                    print(f"STOP: balance {balance_now:.2f} below the floor", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, pending))
    print(json.dumps({"answered": counter["n"], "spent_usd": round(spent, 4), "stopped": stop.is_set()}), flush=True)


if __name__ == "__main__":
    main()
