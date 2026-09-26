"""K-1 Stage 1: answer stored LongMemEval-S retrieval with and without speaker marks.

Pre-registration: docs/preregistrations/2026-09-26-aml-c9-speaker-at-render-time.md (Stage 0
passed on all 120 questions). Arms, interleaved per question in rotating order:

* ``R``  the stored X-1 Stage B items (C9-raw, date header included), as stored;
* ``R2`` R again, for the noise floor (the record's R');
* ``K1`` R with ``[user]`` / ``[assistant]`` marks inserted where a window spans both roles
  (``speaker_render.mark_window``), the date header kept in front. A window that cannot be placed
  uniquely in its session is left as stored.

The reader and judge are AML's own LongMemEval-S prompt and binary judge from the pinned checkout,
both answered by ``deepseek/deepseek-v4.1-flash`` through one pinned provider with reasoning off
(the ``Reader`` of ``aml_t1k2_locomo.py``), under a USD cap and a balance floor.

    python scripts/aml_k1_stage1.py run --stageb out/longmemeval_s.jsonl \\
        --data longmemeval_s_cleaned.json --aml-repo aml-official --out answers.jsonl
    python scripts/aml_k1_stage1.py score --answers answers.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aml_locomo_loss_diagnosis import AML_COMMIT, render_memories  # noqa: E402
from aml_t1k2_locomo import ANSWER_MAX_TOKENS, CREDIT_FLOOR_USD, JUDGE_MAX_TOKENS, Reader  # noqa: E402

from recall_aml.models import SearchItem  # noqa: E402
from recall_aml.speaker_render import locate, mark_window, message_word_ranges  # noqa: E402
from recall_aml.temporal_render import resolve_relative_times  # noqa: E402

SEED = 20260925
ARMS = ("R", "R2", "K1")
#: Every arm the harness can answer: K-1's three, and T1 for T-1's held-out check (T-1 amendment 6).
ALL_ARMS = ("R", "R2", "K1", "T1")
DATE_HEADER = re.compile(r"^(\[[^\]]*UTC\]\s*)")
TYPES = ("single-session-assistant", "single-session-user")
#: AML's LongMemEval-S pipeline, NOT the LoCoMo one ``load_aml_pipeline`` loads: the two templates
#: differ (instructions, and a per-user memory block), so the wrong one answers a different prompt.
LME_PIPELINE = Path("data") / "longmemeval-s" / "pipeline.py"


def load_lme_pipeline(repo: Path) -> ModuleType:
    """AML's LongMemEval-S pipeline from the pinned checkout, refused at any other commit."""
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != AML_COMMIT:
        raise SystemExit(f"AML checkout is at {head}, expected {AML_COMMIT}")
    spec = importlib.util.spec_from_file_location("aml_lme_pipeline", repo / LME_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sessions_of(question: dict[str, Any]) -> dict[str, tuple[list[tuple[str, str]], list[str]]]:
    """Each haystack session's ``(role, content)`` messages and content-only word sequence."""
    out = {}
    for session_id, session in zip(question["haystack_session_ids"], question["haystack_sessions"], strict=True):
        messages = [(str(m["role"]), str(m["content"])) for m in session]
        out[str(session_id)] = (messages, " ".join(content for _, content in messages).split())
    return out


def marked(item: dict[str, Any], sessions: dict[str, tuple[list[tuple[str, str]], list[str]]]) -> dict[str, Any]:
    """``item`` with speaker marks in its window text, the date header kept; unchanged if unplaced."""
    content = str(item.get("content") or "")
    session = sessions.get(str(item.get("session_id") or ""))
    if session is None:
        return item
    match = DATE_HEADER.match(content)
    header = match.group(1) if match else ""
    window = content[len(header):]
    messages, words = session
    start = locate(window, words)
    if start is None:
        return item
    rendered = mark_window(window, start, message_word_ranges(messages))
    return item if rendered == window else {**item, "content": header + rendered}


def view(arm: str, items: list[dict[str, Any]], sessions: dict[str, Any]) -> list[dict[str, Any]]:
    if arm in ("R", "R2"):
        return items
    if arm == "K1":
        return [marked(item, sessions) for item in items]
    if arm == "T1":
        return resolved(items)
    raise ValueError(f"unknown arm {arm!r}")


def resolved(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """T-1's production ``resolve_relative_times`` on the stored items, each anchored on its own
    ``created_at``; an item it leaves alone is returned as the same object."""
    models = [
        SearchItem(
            id=str(item["id"]), content=item["content"],
            created_at=datetime.fromisoformat(str(item["created_at"]).replace("Z", "+00:00"))
            if item.get("created_at") else None,
            source=str(item.get("source") or ""), session_id=str(item.get("session_id") or ""),
            kind=str(item.get("kind") or "raw"), score=float(item.get("score") or 0.0),
        )
        for item in items
    ]
    return [item if model.content == item["content"] else {**item, "content": model.content}
            for item, model in zip(items, resolve_relative_times(models), strict=True)]


def prompt_for(pipeline: Any, question: dict[str, Any], items: list[dict[str, Any]]) -> str:
    return pipeline.render_answer_prompt(
        {
            "question": question["question"],
            "speaker_1_name": "user",
            "speaker_1_memories": render_memories(items, dated=False),
            "speaker_2_name": "(none)",
            "speaker_2_memories": "(all memories are listed above)",
        }
    )


def load(stageb: Path, data: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = {str(q["question_id"]): q for q in json.loads(data.read_text(encoding="utf-8"))}
    rows = []
    for line in stageb.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        search = record["searches"][0]
        rows.append({"id": record["tenant"], "items": [i for i in search["items"] if isinstance(i.get("content"), str)]})
    rows.sort(key=lambda row: row["id"])
    return rows, questions


def run(args: argparse.Namespace) -> None:
    pipeline = load_lme_pipeline(args.aml_repo)
    rows, questions = load(args.stageb, args.data)
    done: set[tuple[str, str]] = set()
    spent = 0.0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["id"], record["arm"]))
            spent += float(record.get("cost") or 0.0)
    reader = Reader(spent, args.max_usd)
    balance = reader.balance()
    print(json.dumps({"questions": len(rows), "done_pairs": len(done), "spent_usd": round(spent, 4),
                      "balance_usd": round(balance, 2)}), flush=True)
    if balance < CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} below the {CREDIT_FLOOR_USD:.0f} USD floor")
    lock = threading.Lock()
    arms = tuple(args.arms.split(","))
    if not set(arms) <= set(ALL_ARMS):
        raise SystemExit(f"--arms must name only {ALL_ARMS}")

    def work(index_row: tuple[int, dict[str, Any]]) -> None:
        index, row = index_row
        question = questions[row["id"]]
        sessions = sessions_of(question)
        order = arms[index % len(arms):] + arms[: index % len(arms)]
        for arm in order:
            if (row["id"], arm) in done or reader.stopped.is_set():
                continue
            items = view(arm, row["items"], sessions)
            try:
                generated, answer_usage = reader.complete(prompt_for(pipeline, question, items), ANSWER_MAX_TOKENS)
                judge_prompt = pipeline.render_accuracy_prompt(
                    {"question": question["question"], "gold_answer": str(question["answer"])}, generated
                )
                verdict, judge_usage = reader.complete(judge_prompt, JUDGE_MAX_TOKENS)
            except RuntimeError:
                return
            try:
                label = pipeline.parse_judge_label(verdict)
            except (ValueError, json.JSONDecodeError):
                label = "UNPARSED"
            record = {
                "id": row["id"], "arm": arm, "type": question["question_type"],
                "marked_items": sum(a is not b for a, b in zip(items, row["items"], strict=True)),
                "generated_answer": generated, "label": label, "judge_response": verdict,
                "answer_usage": answer_usage, "judge_usage": judge_usage,
                "cost": float(answer_usage.get("cost") or 0.0) + float(judge_usage.get("cost") or 0.0),
            }
            with lock, args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, enumerate(rows)))
    print(json.dumps({"spent_usd": round(reader.spent, 4), "stopped": reader.stopped.is_set()}), flush=True)


def paired(labels: dict[str, dict[str, str]], arm: str, keys: list[str]) -> dict[str, Any]:
    keys = [k for k in keys if k in labels[arm] and k in labels["R"]]
    if not keys:
        return {"n": 0}
    diffs = [(labels[arm][k] == "CORRECT") - (labels["R"][k] == "CORRECT") for k in keys]
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
    return {
        "n": len(keys),
        "arm_accuracy": round(sum(labels[arm][k] == "CORRECT" for k in keys) / len(keys), 4),
        "R_accuracy": round(sum(labels["R"][k] == "CORRECT" for k in keys) / len(keys), 4),
        "diff": round(sum(diffs) / len(diffs), 4),
        "ci95": [round(boots[249], 4), round(boots[9_749], 4)],
        "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
    }


def score(args: argparse.Namespace) -> None:
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    types: dict[str, str] = {}
    for path in args.answers:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["id"] in labels[record["arm"]]:
                raise SystemExit(f"{record['arm']} answered {record['id']} twice across the answer files")
            labels[record["arm"]][record["id"]] = record["label"]
            types[record["id"]] = record["type"]
    every = sorted(types)
    result: dict[str, Any] = {
        "answers_per_arm": {arm: len(labels[arm]) for arm in ALL_ARMS if labels[arm]},
        "valid_answer_rate": {arm: round(sum(v != "UNPARSED" for v in labels[arm].values()) / len(labels[arm]), 4)
                              for arm in ALL_ARMS if labels[arm]},
        "K1_minus_R_all": paired(labels, "K1", every),
        "R2_minus_R_all": paired(labels, "R2", every),
    }
    for kind in TYPES:
        result[f"K1_minus_R_{kind}"] = paired(labels, "K1", [k for k in every if types[k] == kind])
    if labels["T1"]:
        result["T1_minus_R_all"] = paired(labels, "T1", every)
        for kind in sorted(set(types.values())):
            result[f"T1_minus_R_{kind}"] = paired(labels, "T1", [k for k in every if types[k] == kind])
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stageb", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--aml-repo", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--max-usd", type=float, default=3.0)
    r.add_argument("--arms", default="R,R2,K1", help=f"comma list from {ALL_ARMS}")
    r.set_defaults(handler=run)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, nargs="+", required=True, help="one or more answer files")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(handler=score)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
