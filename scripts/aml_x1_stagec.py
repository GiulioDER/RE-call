"""X-1 Stage C (amendment 5): C9 and C9' answered on CLBench and PersonaMem-v2.

Reads X-1 Stage B's stored searches (``out/<source>.jsonl``). Arm **B** (C9) is a question's stored
items, all of them, in rank order; **B2** (C9') is the same items answered again, the reader and
judge noise floor. Every prompt and scoring rule is AML's own, loaded from the pinned checkout at
``1b8142b``:

* **CLBench** (``data/clbench/pipeline.py``): ``render_answer_prompt_clbench`` with the task's
  system prompt, ``format_structured_question`` and ``format_selected_memories``; judged with
  ``rubric_judge_prompt`` and parsed as AML parses it (strict score and requirement ratio).
* **PersonaMem-v2** (``data/personamem/pipeline_v2.py``, MCQ mode): AML's messages with the chat
  history replaced by one system message of retrieved memories (the one mapping AML does not fix),
  then the query with ``RECALL_SUFFIX`` and ``MCQ_PROMPT_TEMPLATE``; scored with
  ``extract_final_letter`` against Stage A's options.

    python scripts/aml_x1_stagec.py run --out-dir x1b/out --data-dir x1/data --draw x1/draw.json \\
        --aml-repo aml-official --answers stagec-answers.jsonl
    python scripts/aml_x1_stagec.py score --answers stagec-answers.jsonl --out stagec-score.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_locomo_loss_diagnosis import AML_COMMIT  # noqa: E402
from aml_x1_mm_answers import CREDIT_FLOOR_USD, Reader, _checked, _load, paired, rows, scorers  # noqa: E402

ARMS = ("B", "B2")
SOURCES = ("clbench", "personamem_v2")
ANSWER_MAX_TOKENS = {"clbench": 4000, "personamem_v2": 1500}
JUDGE_MAX_TOKENS = 3000
JUDGE_ATTEMPTS = 3
#: CLBench's own framing of retrieved memories, reused for PersonaMem-v2's history slot.
MEMORY_PREAMBLE = "The following memories from previous conversations may provide additional context:"


def load_pipelines(repo: Path) -> dict[str, ModuleType]:
    """AML's CLBench and PersonaMem-v2 pipelines from the pinned checkout, refused at any other."""
    _checked(repo, AML_COMMIT)
    sys.path.insert(0, str(repo))
    return {"clbench": _load(repo / "data" / "clbench" / "pipeline.py", "aml_clbench_pipeline"),
            "personamem_v2": _load(repo / "data" / "personamem" / "pipeline_v2.py", "aml_personamem_pipeline")}


def texts(items: list[dict[str, Any]]) -> list[str]:
    """The text items a reader sees, in rank order; neither source stores images."""
    return [item["content"] for item in items if isinstance(item.get("content"), str) and item["content"].strip()]


def clbench_prompt(pipeline: ModuleType, scorer: dict[str, Any], items: list[dict[str, Any]]) -> str:
    return pipeline.render_answer_prompt_clbench(
        system_prompt=str(scorer.get("system_prompt") or ""),
        memories=pipeline.format_selected_memories([{"text": text} for text in texts(items)]),
        question=pipeline.format_structured_question(question=str(scorer["question"]), qa_type="", options=[]),
    )


def clbench_judged(pipeline: ModuleType, reply: str | None) -> dict[str, Any] | None:
    """AML's parse of one judge reply: fences stripped, JSON, strict score and requirement ratio.
    ``None`` when the reply does not parse, so the caller can retry as AML does."""
    if not reply:
        return None
    text = reply.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "Overall Score" not in payload:
        return None
    statuses = pipeline._coerce_status_list(payload.get("List of Requirement Satisfaction Status", []))
    return {"score": float(pipeline._coerce_score(payload)),
            "ratio": pipeline._compute_requirement_ratio(statuses), "statuses": len(statuses)}


def personamem_messages(pipeline: ModuleType, scorer: dict[str, Any],
                        items: list[dict[str, Any]]) -> list[dict[str, str]]:
    memories = "\n".join(f"- {text}" for text in texts(items)) or "(no memories)"
    options = "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(scorer["options"]))
    return [
        {"role": "system", "content": f"{MEMORY_PREAMBLE}\n\n{memories}"},
        {"role": "user", "content": str(scorer["user_query"]) + pipeline.RECALL_SUFFIX},
        {"role": "system", "content": pipeline.MCQ_PROMPT_TEMPLATE.format(options=options)},
    ]


def personamem_correct(pipeline: ModuleType, scorer: dict[str, Any], answer: str) -> tuple[bool, str]:
    """AML's mapped-answer rule: the chosen option's TEXT must equal the correct answer."""
    letter = pipeline.extract_final_letter(answer)
    index = ord(letter) - 65 if letter else -1
    chosen = scorer["options"][index] if 0 <= index < len(scorer["options"]) else None
    return chosen == str(scorer["correct_answer"]), letter


def run(args: argparse.Namespace) -> None:
    pipelines = load_pipelines(args.aml_repo)
    draw = json.loads(args.draw.read_text(encoding="utf-8"))
    key = os.environ["OPENROUTER_API_KEY"].strip()
    done: set[tuple[str, str]] = set()
    spent = 0.0
    if args.answers.exists():
        for line in args.answers.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                record = json.loads(line)
                done.add((record["id"], record["arm"]))
                spent += float(record.get("cost") or 0.0)
    reader = Reader(key, spent, args.max_usd)
    balance = reader.balance()
    print(json.dumps({"done_pairs": len(done), "spent_usd": round(spent, 4), "balance_usd": round(balance, 2)}),
          flush=True)
    if balance < CREDIT_FLOOR_USD:
        raise SystemExit(f"balance {balance:.2f} below the {CREDIT_FLOOR_USD:.0f} USD floor")
    lock = threading.Lock()
    for source in args.sources:
        pipeline = pipelines[source]
        found = rows(args.out_dir, source)
        meta = scorers(source, args.data_dir, draw)
        print(json.dumps({"source": source, "questions": len(found)}), flush=True)

        def work(index_row: tuple[int, dict[str, Any]], source: str = source, pipeline: ModuleType = pipeline,
                 meta: dict[str, Any] = meta) -> None:
            index, row = index_row
            scorer = meta[str(row["question_id"])]
            ident = f"{source}:{row['question_id']}"
            for arm in ARMS[index % 2:] + ARMS[: index % 2]:
                if (ident, arm) in done or reader.stopped.is_set():
                    continue
                items = list(row["items"])
                cost = 0.0
                try:
                    if source == "clbench":
                        answer, usage = reader.complete(
                            [{"role": "user", "content": clbench_prompt(pipeline, scorer, items)}],
                            ANSWER_MAX_TOKENS[source])
                        cost += float(usage.get("cost") or 0.0)
                        normalized = pipeline._normalize_model_output(answer)
                        rubrics = pipeline._normalize_rubrics(scorer["rubrics"])
                        judged, attempts = None, 0
                        if normalized and rubrics:
                            prompt = pipeline.rubric_judge_prompt(
                                rubrics_text=pipeline._build_rubrics_text(rubrics), model_output=normalized)
                            while judged is None and attempts < JUDGE_ATTEMPTS:
                                attempts += 1
                                reply, judge_usage = reader.complete(
                                    [{"role": "user", "content": prompt}], JUDGE_MAX_TOKENS)
                                cost += float(judge_usage.get("cost") or 0.0)
                                judged = clbench_judged(pipeline, reply)
                        detail = {"score": judged["score"] if judged else 0.0,
                                  "ratio": judged["ratio"] if judged else 0.0,
                                  "judge_attempts": attempts,
                                  "zero_reason": None if judged else
                                  ("empty answer" if not normalized else
                                   "no rubrics" if not rubrics else "judge reply never parsed")}
                        correct = detail["score"]
                    else:
                        answer, usage = reader.complete(personamem_messages(pipeline, scorer, items),
                                                        ANSWER_MAX_TOKENS[source])
                        cost += float(usage.get("cost") or 0.0)
                        hit, letter = personamem_correct(pipeline, scorer, answer)
                        correct = float(hit)
                        detail = {"letter": letter, "gold": scorer["correct_letter"]}
                except RuntimeError:
                    return
                record = {"id": ident, "source": source, "arm": arm, "category": row["category"],
                          "items": len(texts(items)), "answer": answer[:4000], "correct": correct, **detail,
                          "finish_reason": usage.get("finish_reason"), "provider": usage.get("provider"),
                          "cost": round(cost, 6)}
                with lock, args.answers.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(work, enumerate(found)))
        if reader.stopped.is_set():
            break
    print(json.dumps({"spent_usd": round(reader.spent, 4), "stopped": reader.stopped.is_set()}), flush=True)


def score(args: argparse.Namespace) -> None:
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    ratios: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    categories: dict[str, dict[str, str]] = defaultdict(dict)
    notes: dict[str, int] = defaultdict(int)
    for line in args.answers.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        source, arm, ident = record["source"], record["arm"], record["id"]
        if ident in scores[source][arm]:
            raise SystemExit(f"{arm} answered {ident} twice")
        scores[source][arm][ident] = float(record["correct"])
        categories[source][ident] = str(record["category"])
        if source == "clbench":
            ratios[source][arm][ident] = float(record["ratio"])
            if record.get("zero_reason"):
                notes[f"clbench:{arm}:{record['zero_reason']}"] += 1
        elif not record.get("letter"):
            notes[f"personamem_v2:{arm}:no letter"] += 1
        if record.get("finish_reason") == "length":
            notes[f"{source}:{arm}:answer cut at max tokens"] += 1
    result: dict[str, Any] = {"notes": dict(notes)}
    for source, by_arm in scores.items():
        every = sorted(by_arm["B"])
        entry: dict[str, Any] = {
            "answers_per_arm": {arm: len(by_arm[arm]) for arm in ARMS},
            "C9": round(sum(by_arm["B"][k] for k in every) / len(every), 4) if every else None,
            "C9prime_minus_C9": paired(by_arm, "B2", "B", every),
            "by_category": {},
        }
        if source == "clbench":
            entry["C9_rubric_share"] = round(sum(ratios[source]["B"][k] for k in every) / len(every), 4)
            entry["C9prime_minus_C9_rubric_share"] = paired(ratios[source], "B2", "B", every)
        for category in sorted(set(categories[source].values())):
            keys = [k for k in every if categories[source][k] == category]
            entry["by_category"][category] = {"n": len(keys),
                                              "C9": round(sum(by_arm["B"][k] for k in keys) / len(keys), 4),
                                              "C9prime": round(sum(by_arm["B2"].get(k, 0.0) for k in keys)
                                                               / len(keys), 4)}
        result[source] = entry
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("run")
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--data-dir", type=Path, required=True)
    a.add_argument("--draw", type=Path, required=True)
    a.add_argument("--aml-repo", type=Path, required=True)
    a.add_argument("--answers", type=Path, required=True)
    a.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--max-usd", type=float, default=5.0)
    a.set_defaults(handler=run)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(handler=score)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
