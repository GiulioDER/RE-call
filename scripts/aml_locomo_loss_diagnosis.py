"""Where does C9 lose LoCoMo answers when the evidence is already in what it returns?

Pre-registration: docs/preregistrations/2026-09-24-aml-c9-locomo-loss-diagnosis.md

Four stages, each resumable and each writing its own file:

    collect   (VPS3) Add every LoCoMo session to the served C9 app, Search every question once
              with the served router at top_k 100, keep the returned items, delete the users.
    answer    AML's own LoCoMo-Refined Answer prompt over those items, one call per question.
    judge     AML's own LoCoMo-Refined accuracy prompt over each generated answer.
    classify  every WRONG answer: retrieval miss and abstention by fixed rule, the rest by one
              fixed classifier prompt.
    report    the tables the pre-registration names.

The AML prompts are imported from a checkout of github.com/AML-memory/agent-memory-leaderboard
at the pinned commit rather than copied, so this file carries none of their text.

    python scripts/aml_locomo_loss_diagnosis.py collect --data locomo10.json --out collected.json.gz
    python scripts/aml_locomo_loss_diagnosis.py answer --collected collected.json.gz \
        --data locomo10.json --aml-repo aml-official --out answers.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aml_locomo_route_compare import (  # noqa: E402
    PINNED_DATA_SHA256,
    build_corpus,
    item_text,
    score,
    turn_present,
)

AML_COMMIT = "1b8142bfe0f20f1c5218d6b554aa0012de34e504"
ANSWER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"
CLASSIFIER_MODEL = "openai/gpt-4.1"
COST_CAP_USD = 15.0
WORKERS = 8
BUCKETS = (
    "RETRIEVAL_MISS",
    "ABSTAINED",
    "TEMPORAL",
    "STALE",
    "LIST",
    "DISTRACTOR",
    "INFERENCE",
    "JUDGE",
    "GOLD_ISSUE",
    "OTHER",
)
LLM_BUCKETS = BUCKETS[2:]
ABSTENTION = re.compile(
    r"\b(?:not (?:mentioned|specified|stated|provided|available|known|clear)|"
    r"no (?:information|mention|memory|memories|record|details)|unknown|"
    r"cannot (?:be )?determined?|can't determine|unable to (?:determine|answer)|"
    r"does not (?:say|mention|specify|state)|isn't mentioned|not enough information)\b",
    re.IGNORECASE,
)
CLASSIFIER_PROMPT = """You audit why a memory-based question answering system got a question wrong.
The system read a list of retrieved conversation memories and answered. A judge marked the answer
WRONG against the gold answer. The gold evidence turns are the conversation turns the dataset
says support the gold answer; they were present in what the system read.

Choose the ONE label that best explains the error:
TEMPORAL: the question asks for a date, time, duration or order, and the answer's time is wrong,
  at a different granularity than the gold, or relative where the gold is absolute or the reverse.
STALE: the memories hold an earlier and a later state of the same fact and the answer reports the
  earlier one.
LIST: the gold has several items and the answer misses at least one or adds items.
DISTRACTOR: the answer names a different person, object, place or detail that appears elsewhere in
  the conversation instead of the gold one.
INFERENCE: the gold needs combining several memories or commonsense reasoning over them, and the
  answer fails that reasoning.
JUDGE: the generated answer conveys the gold content; the WRONG label is a judging mistake.
GOLD_ISSUE: the gold answer is wrong or is not supported by the gold evidence turns.
OTHER: none of the above.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}
Gold evidence turns (with the session date):
{evidence}

Reply with a JSON object only: {{"label": "<one label>", "reason": "<one sentence>"}}"""


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {record["id"]: record for record in records}


def load_aml_pipeline(repo: Path) -> ModuleType:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != AML_COMMIT:
        raise SystemExit(f"AML checkout is at {head}, expected {AML_COMMIT}")
    path = repo / "data" / "locomo-refined" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("aml_locomo_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenRouter:
    """Chat completions at temperature 0, with a hard cumulative cost cap across all stages."""

    def __init__(self, spent: float) -> None:
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=180
        )
        self._lock = threading.Lock()
        self.spent = spent

    def complete(self, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
        with self._lock:
            if self.spent >= COST_CAP_USD:
                raise RuntimeError(f"cost cap reached: {self.spent:.2f} USD")
        for attempt in range(6):
            response = self._client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                    "usage": {"include": True},
                },
            )
            if response.status_code in (408, 429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") or {}
            with self._lock:
                self.spent += float(usage.get("cost") or 0.0)
            return str(body["choices"][0]["message"]["content"]).strip(), usage
        raise RuntimeError("OpenRouter kept failing")


def spent_so_far(*paths: Path) -> float:
    return sum(
        float((record.get("usage") or {}).get("cost") or 0.0)
        for path in paths
        for record in read_jsonl(path).values()
    )


def run_parallel(
    todo: list[str], work: Any, sink_path: Path, label: str
) -> None:
    with sink_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(work, ident): ident for ident in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            sink.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            sink.flush()
            if done % 100 == 0:
                print(f"{label} {done}/{len(todo)}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- collect (VPS3)


def collect(args: argparse.Namespace) -> None:
    from starlette.testclient import TestClient

    from recall_aml.__main__ import build_app

    raw = args.data.read_bytes()
    adds, questions = build_corpus(json.loads(raw), args.run_id, None)
    canary_turn = questions[0]["gold_turns"][0]
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    failures: Counter[str] = Counter()
    retries: Counter[str] = Counter()

    def post(client: Any, path: str, body: dict[str, Any]) -> Any:
        for attempt in range(4):
            response = client.post(path, json=body, headers=headers)
            if response.status_code < 500 or attempt == 3:
                return response
            retries[path] += 1
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with TestClient(build_app()) as client:
        version = client.get("/version", headers=headers).json()
        if version.get("variant") != args.expected_variant:
            raise SystemExit(f"served variant {version.get('variant')!r}")
        fallbacks = 0
        for position, request in enumerate(adds, start=1):
            response = post(client, "/v1/add", request)
            if response.status_code != 200:
                failures[f"add_{response.status_code}"] += 1
                continue
            fallbacks += int(bool(response.json().get("compiler_fallback")))
            if position % 25 == 0:
                print(f"added {position}/{len(adds)}", file=sys.stderr, flush=True)
        add_seconds = time.perf_counter() - started
        canary = post(
            client,
            "/v1/search",
            {"query": canary_turn, "user_id": questions[0]["user_id"], "top_k": 100},
        ).json()["data"]
        for position, question in enumerate(questions, start=1):
            response = post(
                client,
                "/v1/search",
                {"query": question["query"], "user_id": question["user_id"], "top_k": 100},
            )
            if response.status_code != 200:
                failures[f"search_{response.status_code}"] += 1
                continue
            body = response.json()
            items = body["data"]
            first_gold = next(
                (
                    rank
                    for rank, item in enumerate(items, start=1)
                    if any(turn_present(turn, item_text(item)) for turn in question["gold_turns"])
                ),
                None,
            )
            rows.append(
                {
                    "id": question["question_id"],
                    "category": question["category"],
                    "route": body.get("specialist_route"),
                    "hits": score(items, question["gold_turns"], set(question["gold_sessions"])),
                    "first_gold_rank": first_gold,
                    "items": [
                        {
                            "id": item["id"],
                            "content": item_text(item),
                            "created_at": item.get("created_at"),
                            "session_id": item.get("session_id"),
                            "kind": item.get("kind"),
                        }
                        for item in items
                    ],
                }
            )
            if position % 100 == 0:
                print(f"searched {position}/{len(questions)}", file=sys.stderr, flush=True)
        for user_id in sorted({request["user_id"] for request in adds}):
            client.post("/v1/delete", json={"user_id": user_id}, headers=headers)
    result = {
        "preregistration": "docs/preregistrations/2026-09-24-aml-c9-locomo-loss-diagnosis.md",
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "data_matches_pinned": hashlib.sha256(raw).hexdigest() == PINNED_DATA_SHA256,
        "variant": version.get("variant"),
        "version": version,
        "n_adds": len(adds),
        "add_compiler_fallbacks": fallbacks,
        "add_seconds": round(add_seconds, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "failures": dict(failures),
        "retries": dict(retries),
        "canary_turn_hit@10": score(canary, [canary_turn], set())["turn_hit@10"],
        "rows": rows,
    }
    args.out.write_bytes(gzip.compress(json.dumps(result).encode("utf-8")))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, default=str))


# ---------------------------------------------------------------- answer and judge (local)


def load_collected(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(gzip.decompress(path.read_bytes()))
    return loaded


def qa_index(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for sample in data:
        conversation = sample["conversation"]
        for position, qa in enumerate(sample["qa"]):
            index[f"{sample['sample_id']}:{position}"] = {
                **qa,
                "sample_id": sample["sample_id"],
                "speakers": (conversation["speaker_a"], conversation["speaker_b"]),
            }
    return index


def render_memories(items: list[dict[str, Any]]) -> str:
    """AML's own timestamped block format (``format_selected_memories`` in its CLBench pipeline)."""
    lines = []
    for item in items:
        stamp = str(item.get("created_at") or "").strip()
        text = str(item.get("content") or "").strip()
        if text:
            lines.append(f"- [{stamp}] {text}" if stamp else f"- {text}")
    return "\n".join(lines)


def answer(args: argparse.Namespace) -> None:
    pipeline = load_aml_pipeline(args.aml_repo)
    collected = load_collected(args.collected)
    qas = qa_index(json.loads(args.data.read_bytes()))
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.out))
    rows = {row["id"]: row for row in collected["rows"]}

    def work(ident: str) -> dict[str, Any]:
        row, qa = rows[ident], qas[ident]
        speaker_a, speaker_b = qa["speakers"]
        prompt = pipeline.render_answer_prompt(
            {
                "question": qa["question"],
                "speaker_1_name": f"{speaker_a} and {speaker_b}",
                "speaker_1_memories": render_memories(row["items"]),
                "speaker_2_name": "(none)",
                "speaker_2_memories": "(all memories are listed above)",
            }
        )
        generated, usage = router.complete(ANSWER_MODEL, prompt)
        return {
            "id": ident,
            "generated_answer": generated,
            "prompt_chars": len(prompt),
            "usage": usage,
        }

    run_parallel([i for i in rows if i not in done], work, args.out, "answered")


def judge(args: argparse.Namespace) -> None:
    pipeline = load_aml_pipeline(args.aml_repo)
    qas = qa_index(json.loads(args.data.read_bytes()))
    answers = read_jsonl(args.answers)
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.answers, args.out))

    def work(ident: str) -> dict[str, Any]:
        qa = qas[ident]
        prompt = pipeline.render_accuracy_prompt(
            {"question": qa["question"], "gold_answer": str(qa["answer"])},
            answers[ident]["generated_answer"],
        )
        response, usage = router.complete(JUDGE_MODEL, prompt)
        try:
            label = pipeline.parse_judge_label(response)
        except (ValueError, json.JSONDecodeError):
            label = "UNPARSED"
        return {"id": ident, "label": label, "judge_response": response, "usage": usage}

    run_parallel([i for i in answers if i not in done], work, args.out, "judged")


def judgecheck(args: argparse.Namespace) -> None:
    """Known answer: every 15th question, judged with its own gold answer as the generated one."""
    pipeline = load_aml_pipeline(args.aml_repo)
    qas = qa_index(json.loads(args.data.read_bytes()))
    ids = sorted(i for i, qa in qas.items() if qa.get("category") in (1, 2, 3, 4))[::15]
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.out))

    def work(ident: str) -> dict[str, Any]:
        gold = str(qas[ident]["answer"])
        prompt = pipeline.render_accuracy_prompt(
            {"question": qas[ident]["question"], "gold_answer": gold}, gold
        )
        response, usage = router.complete(JUDGE_MODEL, prompt)
        try:
            label = pipeline.parse_judge_label(response)
        except (ValueError, json.JSONDecodeError):
            label = "UNPARSED"
        return {"id": ident, "label": label, "judge_response": response, "usage": usage}

    run_parallel([i for i in ids if i not in done], work, args.out, "self-judged")
    labels = [record["label"] for record in read_jsonl(args.out).values()]
    print(json.dumps({"n": len(labels), "correct": labels.count("CORRECT")}))


# ---------------------------------------------------------------- classify and report (local)


def evidence_block(qa: dict[str, Any], data: list[dict[str, Any]]) -> str:
    conversation = next(
        sample["conversation"] for sample in data if sample["sample_id"] == qa["sample_id"]
    )
    lines = []
    for fragment in re.split(r"[\s,;]+", " ".join(map(str, qa.get("evidence", [])))):
        match = re.match(r"^D(\d+):(\d+)$", fragment)
        if not match:
            continue
        session = f"session_{match.group(1)}"
        for turn in conversation.get(session, []):
            if turn["dia_id"] == fragment:
                date = conversation.get(f"{session}_date_time", "")
                lines.append(f"[{date}] {turn['speaker']}: {turn['text']}")
    return "\n".join(lines) or "(none resolved)"


def deterministic_bucket(row: dict[str, Any], generated: str) -> str | None:
    if not row["hits"]["turn_hit@100"]:
        return "RETRIEVAL_MISS"
    if ABSTENTION.search(generated):
        return "ABSTAINED"
    return None


def classify(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_bytes())
    qas = qa_index(data)
    rows = {row["id"]: row for row in load_collected(args.collected)["rows"]}
    answers = read_jsonl(args.answers)
    labels = read_jsonl(args.judged)
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.answers, args.judged, args.out))
    wrong = [i for i, record in labels.items() if record["label"] != "CORRECT"]

    def work(ident: str) -> dict[str, Any]:
        generated = answers[ident]["generated_answer"]
        bucket = deterministic_bucket(rows[ident], generated)
        if bucket is not None:
            return {"id": ident, "bucket": bucket, "source": "rule", "usage": {}}
        qa = qas[ident]
        prompt = CLASSIFIER_PROMPT.format(
            question=qa["question"],
            gold=qa["answer"],
            generated=generated,
            evidence=evidence_block(qa, data),
        )
        response, usage = router.complete(CLASSIFIER_MODEL, prompt)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        label = "OTHER"
        reason = response
        if match:
            try:
                payload = json.loads(match.group(0))
                label = str(payload.get("label", "OTHER")).upper()
                reason = str(payload.get("reason", ""))
            except json.JSONDecodeError:
                pass
        if label not in LLM_BUCKETS:
            label = "OTHER"
        return {"id": ident, "bucket": label, "reason": reason, "source": "llm", "usage": usage}

    run_parallel([i for i in wrong if i not in done], work, args.out, "classified")


def report(args: argparse.Namespace) -> None:
    collected = load_collected(args.collected)
    rows = {row["id"]: row for row in collected["rows"]}
    answers = read_jsonl(args.answers)
    labels = read_jsonl(args.judged)
    buckets = read_jsonl(args.classified)
    ids = sorted(rows)
    if set(ids) != set(answers) or set(ids) != set(labels):
        raise SystemExit("collected, answered and judged ids differ")
    correct = {i for i in ids if labels[i]["label"] == "CORRECT"}
    wrong = [i for i in ids if i not in correct]
    if set(wrong) != set(buckets):
        raise SystemExit("classified ids are not exactly the WRONG answers")

    def rate(subset: list[str]) -> dict[str, Any]:
        hit = sum(i in correct for i in subset)
        return {"n": len(subset), "correct": hit, "accuracy": hit / len(subset) if subset else None}

    counts = Counter(buckets[i]["bucket"] for i in wrong)
    bands = {
        "gold_rank_1_10": [i for i in ids if (rows[i]["first_gold_rank"] or 999) <= 10],
        "gold_rank_11_100": [i for i in ids if 10 < (rows[i]["first_gold_rank"] or 999) <= 100],
        "gold_absent": [i for i in ids if rows[i]["first_gold_rank"] is None],
    }
    result = {
        "n": len(ids),
        "accuracy": rate(ids),
        "unparsed_judge_labels": sum(labels[i]["label"] == "UNPARSED" for i in ids),
        "by_category": {
            str(c): rate([i for i in ids if rows[i]["category"] == c])
            for c in sorted({rows[i]["category"] for i in ids})
        },
        "by_route": {
            str(r): rate([i for i in ids if rows[i]["route"] == r])
            for r in sorted({str(rows[i]["route"]) for i in ids})
        },
        "by_first_gold_rank": {band: rate(members) for band, members in bands.items()},
        "wrong_buckets": {
            bucket: {"count": counts.get(bucket, 0), "share_of_wrong": counts.get(bucket, 0) / len(wrong)}
            for bucket in BUCKETS
        },
        "wrong_buckets_by_category": {
            str(c): dict(Counter(buckets[i]["bucket"] for i in wrong if rows[i]["category"] == c))
            for c in sorted({rows[i]["category"] for i in ids})
        },
        "max_prompt_chars": max(answers[i]["prompt_chars"] for i in ids),
        "spend_usd": {
            "answer": spent_so_far(args.answers),
            "judge": spent_so_far(args.judged),
            "classify": spent_so_far(args.classified),
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("collect")
    stage.add_argument("--data", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.add_argument("--run-id", default="lossdiag1")
    stage.add_argument(
        "--expected-variant", default="C9_routed_specialists_grounded_graph_atomic"
    )
    stage.set_defaults(run=collect)
    for name, function, inputs in (
        ("answer", answer, ("collected", "data", "aml_repo")),
        ("judge", judge, ("answers", "data", "aml_repo")),
        ("judgecheck", judgecheck, ("data", "aml_repo")),
        ("classify", classify, ("collected", "answers", "judged", "data")),
        ("report", report, ("collected", "answers", "judged", "classified")),
    ):
        stage = commands.add_parser(name)
        for option in inputs:
            stage.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
        stage.add_argument("--out", type=Path, required=True)
        stage.set_defaults(run=function)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
