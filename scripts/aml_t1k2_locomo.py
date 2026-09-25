"""LoCoMo half of the T-1/K-2 pre-registration: draw, mechanism metrics, answers, score.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md

Every arm reads the SAME stored C9 retrieval (``collected-S.json.gz``), rendered by the production
functions: H is ``dated_items`` (the served baseline), T1 adds ``resolve_relative_times``, K2 adds
``same_subject_adjacent``, and H2 is H again (the reader and judge noise floor). AML's own LoCoMo
Answer and judge prompts come from ``scripts/aml_locomo_loss_diagnosis.py`` and the pinned AML
checkout. For each question the four arms are answered back to back, in an order that rotates
with the question, so drift over the run lands on every arm alike.

OpenRouter use is deliberately small: two workers by default, DeepSeek V4.1 Flash with reasoning
off and a bounded output, one pinned provider, a USD cap, and a stop below the balance floor that
protects the official C9 sharing the key.

    python scripts/aml_t1k2_locomo.py draw --collected C.json.gz --data locomo10.json --out draw.json
    python scripts/aml_t1k2_locomo.py mechanism --collected C.json.gz --draw draw.json --out mech.json
    python scripts/aml_t1k2_locomo.py run --collected C.json.gz --data locomo10.json \\
        --aml-repo aml-official --draw draw.json --out answers.jsonl
    python scripts/aml_t1k2_locomo.py score --answers answers.jsonl --data locomo10.json --out score.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any, Callable

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_locomo_loss_diagnosis import load_aml_pipeline, load_collected, qa_index, render_memories  # noqa: E402

from recall_aml.conflict_order import WINDOW, same_subject_adjacent  # noqa: E402
from recall_aml.models import SearchItem  # noqa: E402
from recall_aml.temporal_render import resolve_relative_times  # noqa: E402
from recall_aml.window_format import dated_items  # noqa: E402

SEED = 20260925
MODEL = "deepseek/deepseek-v4.1-flash"
PROVIDER = "DeepInfra"
ANSWER_MAX_TOKENS = 300
JUDGE_MAX_TOKENS = 400
ARMS = ("H", "H2", "T1", "K2", "K2v2")
CREDIT_FLOOR_USD = 40.0
BRACKET = re.compile(r" \[(?:=|≈|week of|weekend of) [0-9-]+\]")


def _items(rows: list[dict[str, Any]]) -> list[SearchItem]:
    return [
        SearchItem(
            id=str(item["id"]),
            content=str(item.get("content") or ""),
            created_at=(
                datetime.fromisoformat(str(item["created_at"]).replace("Z", "+00:00"))
                if item.get("created_at")
                else None
            ),
            source="collected",
            session_id=str(item.get("session_id") or ""),
            kind=str(item.get("kind") or "raw"),
            score=float(item.get("score") or 0.0),
        )
        for item in rows
    ]


#: Item id to voyage-code-4 vector, for the K2v2 arm (``scripts/aml_k2v2_vectors.py``).
VECTORS: dict[str, list[float]] = {}


def _k2v2(items: list[SearchItem]) -> list[SearchItem]:
    base = dated_items(items)
    return same_subject_adjacent(base, vectors=[VECTORS.get(item.id) for item in base[:WINDOW]])


TRANSFORMS: dict[str, Callable[[list[SearchItem]], list[SearchItem]]] = {
    "H": lambda items: dated_items(items),
    "H2": lambda items: dated_items(items),
    "T1": lambda items: resolve_relative_times(dated_items(items)),
    "K2": lambda items: same_subject_adjacent(dated_items(items)),
    "K2v2": _k2v2,
}


def view(arm: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The items the reader sees in ``arm``, as dicts ``render_memories`` accepts."""
    return [{"content": item.content, "created_at": item.created_at} for item in TRANSFORMS[arm](_items(rows))]


def draw(args: argparse.Namespace) -> None:
    """All category 2 questions plus 400 others, stratified by category, drawn once."""
    collected = load_collected(args.collected)
    qas = qa_index(json.loads(args.data.read_bytes()))
    ids = [row["id"] for row in collected["rows"]]
    by_category: dict[int, list[str]] = defaultdict(list)
    for ident in ids:
        by_category[int(qas[ident]["category"])].append(ident)
    others = {category: members for category, members in by_category.items() if category != 2}
    total_others = sum(len(members) for members in others.values())
    rng = random.Random(SEED)
    quotas = {category: round(400 * len(members) / total_others) for category, members in others.items()}
    largest = max(quotas, key=lambda category: quotas[category])
    quotas[largest] += 400 - sum(quotas.values())
    chosen = list(by_category[2])
    for category in sorted(others):
        chosen += sorted(rng.sample(sorted(others[category]), quotas[category]))
    payload = {
        "seed": SEED,
        "collected_sha256": hashlib.sha256(args.collected.read_bytes()).hexdigest(),
        "category_counts": {str(c): sum(qas[i]["category"] == c for i in chosen) for c in sorted(by_category)},
        "ids": chosen,
    }
    payload["ids_sha256"] = hashlib.sha256("\n".join(chosen).encode()).hexdigest()
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "ids"}, indent=2))


def mechanism(args: argparse.Namespace) -> None:
    """How often each transform acts, and the render-only property on every row. No model call."""
    collected = load_collected(args.collected)
    chosen = set(json.loads(args.draw.read_text(encoding="utf-8"))["ids"])
    t1_fired = k2_moved = rows_seen = 0
    brackets: list[int] = []
    render_only = True
    for row in collected["rows"]:
        if row["id"] not in chosen:
            continue
        rows_seen += 1
        base = dated_items(_items(row["items"]))
        t1 = resolve_relative_times(base)
        k2 = same_subject_adjacent(base)
        count = sum(len(BRACKET.findall(str(item.content))) for item in t1[:20])
        brackets.append(count)
        t1_fired += count > 0
        k2_moved += [item.id for item in k2[:30]] != [item.id for item in base[:30]]
        render_only &= [item.id for item in t1] == [item.id for item in base]
        render_only &= all(BRACKET.sub("", str(a.content)) == str(b.content) for a, b in zip(t1, base))
        render_only &= sorted(item.model_dump_json() for item in k2) == sorted(item.model_dump_json() for item in base)
        render_only &= [item.id for item in k2[30:]] == [item.id for item in base[30:]]
    brackets.sort()
    result = {
        "questions": rows_seen,
        "t1_share_with_a_resolution_in_top20": t1_fired / rows_seen,
        "t1_resolutions_top20_median": brackets[len(brackets) // 2],
        "k2_share_top30_order_changed": k2_moved / rows_seen,
        "render_only_on_every_row": render_only,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


class Reader:
    """DeepSeek V4.1 Flash through one pinned provider, reasoning off, bounded output, capped spend."""

    def __init__(self, spent: float, cap: float) -> None:
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=300
        )
        self._lock = threading.Lock()
        self.spent = spent
        self.cap = cap
        self.stopped = threading.Event()

    def balance(self) -> float:
        body = self._client.get("https://openrouter.ai/api/v1/credits").json()["data"]
        return float(body["total_credits"]) - float(body["total_usage"])

    def complete(self, prompt: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
        if self.stopped.is_set():
            raise RuntimeError("stopped")
        payload = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
            "provider": {"order": [PROVIDER], "allow_fallbacks": False},
            "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True},
        }
        for attempt in range(6):
            try:
                response = self._client.post("https://openrouter.ai/api/v1/chat/completions", json=payload)
            except httpx.TransportError:
                time.sleep(2**attempt)
                continue
            if response.status_code == 402:
                self.stopped.set()
                raise RuntimeError("OpenRouter 402: out of credit")
            if response.status_code in (408, 429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            usage = {**(body.get("usage") or {}), "provider": body.get("provider"), "finish_reason": choice.get("finish_reason")}
            with self._lock:
                self.spent += float(usage.get("cost") or 0.0)
                if self.spent >= self.cap:
                    self.stopped.set()
            return str(choice["message"].get("content") or "").strip(), usage
        raise RuntimeError("OpenRouter kept failing")


def run(args: argparse.Namespace) -> None:
    pipeline = load_aml_pipeline(args.aml_repo)
    collected = load_collected(args.collected)
    qas = qa_index(json.loads(args.data.read_bytes()))
    ids = json.loads(args.draw.read_text(encoding="utf-8"))["ids"]
    arms = tuple(args.arms.split(","))
    if not set(arms) <= set(ARMS) or "H" not in arms:
        raise SystemExit(f"--arms must include H and name only {ARMS}")
    if "K2v2" in arms:
        if args.vectors is None:
            raise SystemExit("the K2v2 arm needs --vectors")
        with gzip.open(args.vectors, "rt", encoding="utf-8") as source:
            VECTORS.update(json.load(source))
    rows = {row["id"]: row for row in collected["rows"]}
    done: set[tuple[str, str]] = set()
    spent = 0.0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["id"], record["arm"]))
            spent += float(record.get("cost") or 0.0)
    reader = Reader(spent, args.max_usd)
    balance = reader.balance()
    print(json.dumps({"questions": len(ids), "done_pairs": len(done), "spent_usd": round(spent, 4), "balance_usd": round(balance, 2)}), flush=True)
    if balance < CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} below the {CREDIT_FLOOR_USD:.0f} USD floor")
    lock = threading.Lock()
    finished = {"n": 0}

    def work(index_ident: tuple[int, str]) -> None:
        index, ident = index_ident
        qa = qas[ident]
        speaker_a, speaker_b = qa["speakers"]
        order = arms[index % len(arms) :] + arms[: index % len(arms)]
        for arm in order:
            if (ident, arm) in done or reader.stopped.is_set():
                continue
            prompt = pipeline.render_answer_prompt(
                {
                    "question": qa["question"],
                    "speaker_1_name": f"{speaker_a} and {speaker_b}",
                    "speaker_1_memories": render_memories(view(arm, rows[ident]["items"]), dated=False),
                    "speaker_2_name": "(none)",
                    "speaker_2_memories": "(all memories are listed above)",
                }
            )
            try:
                generated, answer_usage = reader.complete(prompt, ANSWER_MAX_TOKENS)
                judge_prompt = pipeline.render_accuracy_prompt(
                    {"question": qa["question"], "gold_answer": str(qa["answer"])}, generated
                )
                verdict, judge_usage = reader.complete(judge_prompt, JUDGE_MAX_TOKENS)
            except RuntimeError:
                return
            try:
                label = pipeline.parse_judge_label(verdict)
            except (ValueError, json.JSONDecodeError):
                label = "UNPARSED"
            record = {
                "id": ident,
                "arm": arm,
                "category": qa["category"],
                "generated_answer": generated,
                "label": label,
                "judge_response": verdict,
                "answer_usage": answer_usage,
                "judge_usage": judge_usage,
                "cost": float(answer_usage.get("cost") or 0.0) + float(judge_usage.get("cost") or 0.0),
            }
            with lock, args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        with lock:
            finished["n"] += 1
            if finished["n"] % 50 == 0:
                current = reader.balance()
                print(json.dumps({"questions_done": finished["n"], "spent_usd": round(reader.spent, 4), "balance_usd": round(current, 2)}), flush=True)
                if current < CREDIT_FLOOR_USD:
                    reader.stopped.set()
                    print(f"STOP: balance {current:.2f} below the floor", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, enumerate(ids)))
    print(json.dumps({"spent_usd": round(reader.spent, 4), "stopped": reader.stopped.is_set()}), flush=True)


def score(args: argparse.Namespace) -> None:
    """Accuracy per arm and category, and the registered paired contrasts against H."""
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    category: dict[str, int] = {}
    for line in args.answers.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        labels[record["arm"]][record["id"]] = record["label"]
        category[record["id"]] = int(record["category"])

    def paired(arm: str, where: Callable[[str], bool], drop_unparsed: bool = False) -> dict[str, Any]:
        keys = sorted(
            ident
            for ident in set(labels["H"]) & set(labels[arm])
            if where(ident)
            and not (drop_unparsed and "UNPARSED" in (labels["H"][ident], labels[arm][ident]))
        )
        if not keys:
            return {"n": 0}
        diffs = [
            (labels[arm][k] == "CORRECT") - (labels["H"][k] == "CORRECT") for k in keys
        ]
        rng = random.Random(SEED)
        boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
        return {
            "n": len(keys),
            "arm_accuracy": 100 * sum(labels[arm][k] == "CORRECT" for k in keys) / len(keys),
            "h_accuracy": 100 * sum(labels["H"][k] == "CORRECT" for k in keys) / len(keys),
            "diff_points": 100 * sum(diffs) / len(diffs),
            "ci95_points": [100 * boots[249], 100 * boots[9_749]],
            "wins": diffs.count(1),
            "losses": diffs.count(-1),
        }

    is_cat2 = lambda ident: category[ident] == 2  # noqa: E731
    everything = lambda ident: True  # noqa: E731
    def maybe(arm: str, where: Callable[[str], bool]) -> dict[str, Any]:
        return paired(arm, where) if labels.get(arm) else {"n": 0, "not_answered": True}

    result = {
        "labels_per_arm": {arm: dict(Counter(values.values())) for arm, values in sorted(labels.items())},
        "T1_minus_H_category_2": paired("T1", is_cat2),
        "T1_minus_H_all_answered": paired("T1", everything),
        "K2_minus_H_all_answered": maybe("K2", everything),
        "K2v2_minus_H_all_answered": maybe("K2v2", everything),
        "K2v2_minus_H_category_2": maybe("K2v2", is_cat2),
        "H2_minus_H_all_answered": paired("H2", everything),
        "H2_minus_H_category_2": paired("H2", is_cat2),
        "sensitivity_unparsed_dropped": {
            "T1_minus_H_category_2": paired("T1", is_cat2, drop_unparsed=True),
            "K2_minus_H_all_answered": (
                paired("K2", everything, drop_unparsed=True) if labels.get("K2") else {"n": 0}
            ),
        },
        "spend_usd": round(
            sum(float(json.loads(line).get("cost") or 0.0) for line in args.answers.read_text(encoding="utf-8").splitlines()),
            4,
        ),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("draw")
    stage.add_argument("--collected", type=Path, required=True)
    stage.add_argument("--data", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.set_defaults(run=draw)
    stage = commands.add_parser("mechanism")
    stage.add_argument("--collected", type=Path, required=True)
    stage.add_argument("--draw", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.set_defaults(run=mechanism)
    stage = commands.add_parser("run")
    for option in ("collected", "data", "aml_repo", "draw", "out"):
        stage.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    stage.add_argument("--workers", type=int, default=2)
    stage.add_argument("--arms", default="H,H2,T1", help="comma-separated subset of " + ",".join(ARMS))
    stage.add_argument("--vectors", type=Path, default=None, help="item vectors for the K2v2 arm")
    stage.add_argument("--max-usd", type=float, default=18.0)
    stage.set_defaults(run=run)
    stage = commands.add_parser("score")
    for option in ("answers", "out"):
        stage.add_argument("--" + option, type=Path, required=True)
    stage.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
