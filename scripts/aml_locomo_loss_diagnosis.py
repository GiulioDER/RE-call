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
        payload: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True},
        }
        # Two runs of the same evidence drifted by 2.4 points (compiled-records counterfactual,
        # 2026-09-24). Pinning one upstream removes one candidate cause of that drift.
        if provider := os.environ.get("AML_DIAG_PROVIDER"):
            payload["provider"] = {"order": [provider], "allow_fallbacks": False}
        for attempt in range(6):
            try:
                response = self._client.post(
                    "https://openrouter.ai/api/v1/chat/completions", json=payload
                )
            except httpx.TransportError:
                # A reset connection killed two answer processes on 2026-09-24; retry it like a 5xx.
                time.sleep(2**attempt)
                continue
            if response.status_code == 402:
                raise SystemExit("OpenRouter returned 402 Payment Required: the account is out of credit")
            if response.status_code in (408, 429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            body = response.json()
            usage = {
                **(body.get("usage") or {}),
                "provider": body.get("provider"),
                "system_fingerprint": body.get("system_fingerprint"),
            }
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


def adds_by_user(adds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One ordered lane per user, in first-seen user order.

    Users may be added in parallel, but one user's sessions stay in their conversation order, as
    AML sends them, so a lane is never split across workers.
    """
    lanes: dict[str, list[dict[str, Any]]] = {}
    for request in adds:
        lanes.setdefault(request["user_id"], []).append(request)
    return list(lanes.values())


def collect(args: argparse.Namespace) -> None:
    from starlette.testclient import TestClient

    from recall_aml.__main__ import build_app

    raw = args.data.read_bytes()
    adds, questions = build_corpus(json.loads(raw), args.run_id, args.limit_conversations)
    canary_turn = questions[0]["gold_turns"][0]
    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    failures: Counter[str] = Counter()
    retries: Counter[str] = Counter()
    tally = threading.Lock()

    def post(client: Any, path: str, body: dict[str, Any]) -> Any:
        for attempt in range(4):
            response = client.post(path, json=body, headers=headers)
            if response.status_code < 500 or attempt == 3:
                return response
            with tally:
                retries[path] += 1
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    expected_renderer = "message-content-only-v1"
    if args.timestamped_windows:
        import recall_aml.__main__ as hosted_main

        served_variant = hosted_main.variant
        hosted_main.variant = lambda name: timestamped_windows(served_variant(name))  # type: ignore[assignment]
        expected_renderer = "timestamp-role-content-v1"

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with TestClient(build_app()) as client:
        version = client.get("/version", headers=headers).json()
        if version.get("variant") != args.expected_variant:
            raise SystemExit(f"served variant {version.get('variant')!r}")
        if version.get("window_renderer_profile") != expected_renderer:
            raise SystemExit(f"served renderer {version.get('window_renderer_profile')!r}")
        fallbacks = 0
        added = 0

        def add_lane(lane: list[dict[str, Any]]) -> None:
            nonlocal fallbacks, added
            for request in lane:
                response = post(client, "/v1/add", request)
                with tally:
                    added += 1
                    if response.status_code != 200:
                        failures[f"add_{response.status_code}"] += 1
                    else:
                        fallbacks += int(bool(response.json().get("compiler_fallback")))
                    if added % 25 == 0:
                        print(f"added {added}/{len(adds)}", file=sys.stderr, flush=True)

        with ThreadPoolExecutor(args.workers) as pool:
            for future in [pool.submit(add_lane, lane) for lane in adds_by_user(adds)]:
                future.result()
        add_seconds = time.perf_counter() - started
        canary = post(
            client,
            "/v1/search",
            {"query": canary_turn, "user_id": questions[0]["user_id"], "top_k": 100},
        ).json()["data"]

        def search(question: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            return question, post(
                client,
                "/v1/search",
                {"query": question["query"], "user_id": question["user_id"], "top_k": 100},
            )

        with ThreadPoolExecutor(args.workers) as pool:
            answered = list(pool.map(search, questions))
        for position, (question, response) in enumerate(answered, start=1):
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
                    # The service keeps diagnostics out of the contract body and sends them as
                    # headers; reading the body for them silently yields None for every row.
                    "route": response.headers["X-Recall-Specialist-Route"],
                    "diagnostics": {
                        name: response.headers[f"X-Recall-{name}"]
                        for name in (
                            "Graph-Attempted",
                            "Graph-Fallback",
                            "Graph-Promoted",
                            "Graph-Top10-Order-Changed",
                            "Atomic-Rescue-Attempted",
                            "Atomic-Rescue-Active",
                        )
                    },
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
        "preregistration": args.preregistration,
        "timestamped_windows": bool(args.timestamped_windows),
        "workers": args.workers,
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


def render_memories(items: list[dict[str, Any]], *, dated: bool = True) -> str:
    """AML's own timestamped block format (``format_selected_memories`` in its CLBench pipeline).

    With ``dated=False`` each item is its ``content`` alone. AML's API guide promises only that
    ``content`` reaches the Answer model, and ``created_at`` is optional, so this is the view a
    reader gets if the platform never renders the timestamp
    (docs/preregistrations/2026-09-24-aml-c9-reader-dates.md).
    """
    lines = []
    for item in items:
        stamp = str(item.get("created_at") or "").strip() if dated else ""
        text = str(item.get("content") or "").strip()
        if text:
            lines.append(f"- [{stamp}] {text}" if stamp else f"- {text}")
    return "\n".join(lines)


def timestamped_windows(behavior: Any) -> Any:
    """The same variant with the timestamp, role and content window renderer.

    C9 ships ``content_only_windows=True``, which drops every message timestamp and role from the
    stored window text. Flipping only that flag selects the renderer the service already has,
    ``timestamp-role-content-v1``, and changes nothing else about the variant.
    """
    import dataclasses

    return dataclasses.replace(behavior, content_only_windows=False)


def route_of(question: str) -> str:
    """The served router's route. It is a pure function of the question text."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from recall_aml.specialists import route_query

    return str(route_query(question))


def served_items(items: list[dict[str, Any]], *, drop_compiled: bool) -> list[dict[str, Any]]:
    """The counterfactual arm keeps only raw windows, in their served rank order."""
    if not drop_compiled:
        return items
    return [item for item in items if item["kind"] == "raw"]


def answer(args: argparse.Namespace) -> None:
    pipeline = load_aml_pipeline(args.aml_repo)
    collected = load_collected(args.collected)
    qas = qa_index(json.loads(args.data.read_bytes()))
    done = read_jsonl(args.out)
    router = OpenRouter(spent_so_far(args.out))
    rows = {row["id"]: row for row in collected["rows"]}
    if args.route is not None:
        rows = {i: row for i, row in rows.items() if route_of(qas[i]["question"]) == args.route}
    if args.category is not None:
        rows = {i: row for i, row in rows.items() if qas[i]["category"] == args.category}

    def work(ident: str) -> dict[str, Any]:
        row, qa = rows[ident], qas[ident]
        speaker_a, speaker_b = qa["speakers"]
        prompt = pipeline.render_answer_prompt(
            {
                "question": qa["question"],
                "speaker_1_name": f"{speaker_a} and {speaker_b}",
                "speaker_1_memories": render_memories(
                    served_items(row["items"], drop_compiled=args.drop_compiled),
                    dated=args.reader_view == "dated",
                ),
                "speaker_2_name": "(none)",
                "speaker_2_memories": "(all memories are listed above)",
            }
        )
        generated, usage = router.complete(ANSWER_MODEL, prompt)
        return {
            "id": ident,
            "generated_answer": generated,
            "prompt_chars": len(prompt),
            "reader_view": args.reader_view,
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


def paired(control: list[int], treatment: list[int], seed: int = 0) -> dict[str, Any]:
    """Treatment minus control in points, with a 10,000 resample percentile interval."""
    import random

    n = len(control)
    diffs = [t - c for c, t in zip(control, treatment, strict=True)]
    rng = random.Random(seed)
    means = sorted(
        100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(10_000)
    )
    return {
        "n": n,
        "control_accuracy": sum(control) / n,
        "treatment_accuracy": sum(treatment) / n,
        "delta_points": 100.0 * sum(diffs) / n,
        "ci95_low_points": means[250],
        "ci95_high_points": means[9_749],
        "wrong_to_right": sum(d > 0 for d in diffs),
        "right_to_wrong": sum(d < 0 for d in diffs),
    }


def compare(args: argparse.Namespace) -> None:
    """Paired accuracy of two judged arms over the questions both answered."""
    qas = qa_index(json.loads(args.data.read_bytes()))
    control = read_jsonl(args.judged_a)
    treatment = read_jsonl(args.judged_b)
    ids = sorted(set(control) & set(treatment))
    if not ids:
        raise SystemExit(f"no question is judged in both {args.judged_a} and {args.judged_b}")

    def correct(labels: dict[str, dict[str, Any]], members: list[str]) -> list[int]:
        return [int(labels[i]["label"] == "CORRECT") for i in members]

    result: dict[str, Any] = {
        "all": paired(correct(control, ids), correct(treatment, ids)),
        "by_category": {},
    }
    for category in sorted({qas[i]["category"] for i in ids}):
        members = [i for i in ids if qas[i]["category"] == category]
        result["by_category"][str(category)] = paired(
            correct(control, members), correct(treatment, members)
        )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


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
    stage.add_argument("--limit-conversations", type=int, default=None)
    stage.add_argument(
        "--expected-variant", default="C9_routed_specialists_grounded_graph_atomic"
    )
    stage.add_argument(
        "--timestamped-windows",
        action="store_true",
        help="build the variant with the timestamp, role and content window renderer",
    )
    stage.add_argument(
        "--preregistration",
        default="docs/preregistrations/2026-09-24-aml-c9-locomo-loss-diagnosis.md",
    )
    stage.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent requests: users are added in parallel lanes, questions searched in parallel",
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
    answer_stage = commands.choices["answer"]
    answer_stage.add_argument("--route", choices=("code", "context", "multimodal"), default=None)
    answer_stage.add_argument("--drop-compiled", action="store_true")
    answer_stage.add_argument("--category", type=int, choices=(1, 2, 3, 4), default=None)
    answer_stage.add_argument(
        "--reader-view",
        choices=("dated", "content"),
        default="dated",
        help="dated: '- [created_at] content' as before; content: the content field alone",
    )
    stage = commands.add_parser("compare")
    for option in ("judged_a", "judged_b", "data"):
        stage.add_argument("--" + option.replace("_", "-"), type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
