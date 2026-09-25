"""Answer BEAM through a live C9 exactly as AML's public BEAM pipeline would, and score it.

Pre-registration: ``docs/preregistrations/2026-09-24-c9-beam-ability-probe.md``.

Why: the official Textual smoke scored 0 on B2 (causal chain and intermediate steps) and F1
(summarization and long-history synthesis), and 20 on C1 (dates), on one to three questions each.
AML forbids analysing its evaluation data, so the diagnosis runs on the PUBLIC BEAM release, whose
ability types cover the same ground (summarization, multi_session_reasoning, event_ordering,
temporal_reasoning), through the live service's own Add and Search.

What is copied from AML's public pipeline (``AML-memory/agent-memory-leaderboard`` at
``1b8142b``, ``data/beam/pipeline.py``): the answer prompt, the batch rubric judge prompt and its
0 / 0.5 / 1 scale, the event-ordering alignment F1 times normalised Kendall tau, the default model
for both roles (Qwen3-14B, thinking off, temperature 0, 512 answer tokens). What is NOT known and
is assumed: how the platform joins ``data[].content`` into the answer context (here: in returned
order, separated by blank lines, content only, no ``created_at``) and which judge model the
platform really configures.

Two arms answer from the SAME retrieved 100 items, so they differ only in order:
``returned`` (C9's rank order, what AML sees) and ``chronological`` (batch order, then position in
the batch). A third pass, ``coverage``, asks the judge whether each rubric point is present in the
retrieved context at all, which separates a retrieval miss from a reader miss.

Phases, each resumable, all writing under ``--out``:

    ingest -> retrieve -> answer --arm A -> judge --arm A -> coverage -> report -> cleanup

The service key comes from ``RECALL_AML_API_KEY``, the model key from ``OPENROUTER_API_KEY``.
Stdlib only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524})
ADD_ATTEMPTS = 32
SEARCH_ATTEMPTS = 3
TOP_K = 100
MODEL = "qwen/qwen3-14b"
#: USD per token for MODEL on OpenRouter, read 2026-09-24 from /api/v1/models.
PRICE_IN = 0.12e-6
PRICE_OUT = 0.24e-6
#: A stronger model, used only to QUOTE evidence for the quote-verified coverage pass.
QUOTE_MODEL = "openai/gpt-4.1-mini"
#: The model C9 already runs at Add (its compiler), used to write one summary per session.
SUMMARY_MODEL = "openai/gpt-4o-mini"
#: USD per token (in, out) per model, read 2026-09-25 from OpenRouter's /api/v1/models.
PRICES = {MODEL: (PRICE_IN, PRICE_OUT), QUOTE_MODEL: (0.4e-6, 1.6e-6),
          SUMMARY_MODEL: (0.15e-6, 0.6e-6)}
ARMS = ("returned", "chronological", "top10", "top20", "top40", "summaries")
#: Arms that answer from only the first N returned items, in returned order.
TOP_ARMS = {"top10": 10, "top20": 20, "top40": 40}

# Verbatim from AML's data/beam/pipeline.py (itself verbatim from BEAM src/prompts.py).
ANSWER_PROMPT = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below.

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation

RESPONSE:
"""

JUDGE_BASE = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
"""

BATCH_OUTPUT_FORMAT = """## OUTPUT FORMAT:
Return one independent evaluation for every indexed rubric criterion in JSON:

{
  "scores": [
    {"index": 0, "score": 1.0, "reason": "detailed justification"}
  ]
}

Include every index exactly once. Each score must be 1.0, 0.5, or 0.0.
NOTE: ONLY output the json object, without any explanation before or after that
"""

EQUIVALENCE_SYSTEM = """You are a binary classifier.
If the TWO snippets describe the SAME event/fact, reply **YES**
Otherwise reply **NO**. No extra words.
DO NOT provide any exaplanation."""

# Ours, not AML's: asks whether the retrieved CONTEXT holds what each rubric point needs.
COVERAGE_PROMPT = """You check whether a retrieved CONTEXT contains the information needed to satisfy each
RUBRIC CRITERION for the QUESTION. Do not judge any answer; judge only the context.

QUESTION: <question>

RUBRIC CRITERIA:
<rubric_item>

CONTEXT:
<context>

Score each criterion 1.0 if the context contains all the information needed to satisfy it, 0.5 if it
contains part of it, 0.0 if it does not contain it.

""" + BATCH_OUTPUT_FORMAT


# ------------------------------------------------------------------------------------------
# Transport
# ------------------------------------------------------------------------------------------


def http_json(
    url: str, payload: dict[str, Any] | None, headers: dict[str, str], timeout: float
) -> tuple[int, dict[str, Any], dict[str, str]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=body, method="POST" if body is not None else "GET",
                      headers={"Content-Type": "application/json", **headers})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
            return (response.status, json.loads(raw) if raw else {},
                    {k.casefold(): v for k, v in response.headers.items()})
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw[:300]}
        return exc.code, parsed if isinstance(parsed, dict) else {}, {}
    except (URLError, TimeoutError, OSError) as exc:
        return 599, {"error": type(exc).__name__}, {}


class Service:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def call(self, path: str, payload: dict[str, Any] | None, *, attempts: int, timeout: float,
             sleep: Callable[[float], None] = time.sleep) -> tuple[int, dict, dict, int]:
        status, body, headers = 599, {}, {}  # type: tuple[int, dict[str, Any], dict[str, str]]
        for attempt in range(attempts):
            status, body, headers = http_json(self.base + path, payload, self.headers, timeout)
            if status not in RETRYABLE:
                return status, body, headers, attempt + 1
            if attempt + 1 < attempts:
                sleep(min(30.0, 2.0 * 2**attempt))
        return status, body, headers, attempts


class Spend:
    """Model spend so far, from the usage every OpenRouter response reports."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.usd = 0.0
        self.calls = 0
        self.lock = threading.Lock()

    def add(self, usage: dict[str, Any], model: str = MODEL) -> None:
        price_in, price_out = PRICES[model]
        with self.lock:
            self.calls += 1
            self.usd += (usage.get("prompt_tokens", 0) * price_in
                         + usage.get("completion_tokens", 0) * price_out)

    def check(self) -> None:
        if self.usd > self.cap:
            raise SystemExit(f"model spend ${self.usd:.2f} passed the ${self.cap:.2f} cap")


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def complete(spend: Spend, messages: list[dict[str, str]], max_tokens: int,
             json_mode: bool = False, model: str = MODEL) -> str:
    spend.check()
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
    }
    if model == MODEL:
        # AML's pipeline turns Qwen3 thinking off (``enable_thinking: False``).
        payload["reasoning"] = {"enabled": False}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    key = os.environ["OPENROUTER_API_KEY"].strip()
    for attempt in range(6):
        status, body, _ = http_json("https://openrouter.ai/api/v1/chat/completions", payload,
                                    {"Authorization": f"Bearer {key}"}, 180)
        if status == 200 and body.get("choices"):
            spend.add(body.get("usage", {}), model)
            content = body["choices"][0]["message"].get("content") or ""
            return _THINK.sub("", content).strip()
        time.sleep(min(60, 3 * 2**attempt))
    raise RuntimeError(f"model call failed with status {status}: {str(body)[:200]}")


# ------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Appender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def pool(workers: int, jobs: list[Callable[[], Any]]) -> list[Any]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return [future.result() for future in [executor.submit(job) for job in jobs]]


# ------------------------------------------------------------------------------------------
# Phases
# ------------------------------------------------------------------------------------------


def state_of(out: Path) -> dict[str, Any]:
    path = out / "state.json"
    if not path.exists():
        path.write_text(json.dumps({"nonce": uuid4().hex[:10]}), encoding="utf-8")
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return state


def user_of(state: dict[str, Any], conversation: int) -> str:
    return f"beam-probe-{state['nonce']}-conv-{conversation}"


def session_of(user: str, batch: int) -> str:
    return f"{user}/batch-{batch:02d}"


def ingest(service: Service, data: list[dict], out: Path, workers: int) -> None:
    state = state_of(out)
    log = Appender(out / "adds.jsonl")
    done = {(r["conversation"], r["batch"]) for r in read_jsonl(out / "adds.jsonl") if r["status"] == 200}

    def one(conversation: dict) -> None:
        user = user_of(state, conversation["conversation"])
        for index, batch in enumerate(conversation["batches"]):  # sequential within a user
            if (conversation["conversation"], index) in done:
                continue
            body = {"request_id": f"{user}-add-{index:02d}", "user_id": user,
                    "session_id": session_of(user, index), "messages": batch["messages"]}
            started = time.perf_counter()
            status, payload, _, attempts = service.call("/v1/add", body, attempts=ADD_ATTEMPTS,
                                                        timeout=1800)
            log.write({"conversation": conversation["conversation"], "batch": index,
                       "status": status, "attempts": attempts,
                       "seconds": round(time.perf_counter() - started, 3),
                       "words": sum(len(m["content"].split()) for m in batch["messages"]),
                       "raw_count": payload.get("raw_count"),
                       "compiled_count": payload.get("compiled_count"),
                       "compiler_fallback": payload.get("compiler_fallback")})

    pool(workers, [partial(one, c) for c in data])


def retrieve(service: Service, data: list[dict], out: Path, workers: int) -> None:
    state = state_of(out)
    log = Appender(out / "retrieval.jsonl")
    done = {r["id"] for r in read_jsonl(out / "retrieval.jsonl") if r["status"] == 200}
    keep = ("x-recall-specialist-route", "x-recall-graph-attempted", "x-recall-graph-fallback",
            "x-recall-graph-promoted", "x-recall-atomic-rescue-active",
            "x-recall-atomic-rescue-fallback", "x-recall-search-ms")

    def one(conversation: dict, question: dict) -> None:
        body = {"query": question["question"], "user_id": user_of(state, conversation["conversation"]),
                "top_k": TOP_K}
        status, payload, headers, attempts = service.call("/v1/search", body,
                                                          attempts=SEARCH_ATTEMPTS, timeout=120)
        log.write({"id": question["id"], "conversation": conversation["conversation"],
                   "type": question["type"], "status": status, "attempts": attempts,
                   "items": payload.get("data", []),
                   "headers": {k: v for k, v in headers.items() if k in keep}})

    pool(workers, [partial(one, c, q) for c in data for q in c["questions"]
                   if q["id"] not in done])


def _words(text: str) -> str:
    return " ".join(text.split())


def chronological(items: list[dict], conversation: dict, user: str) -> list[dict]:
    """The same items in batch order, then by their position in the batch's text.

    A raw window's content is a word window of the batch's messages joined by spaces, so its
    first dozen words locate it. An item that cannot be located (a compiled record) sorts after
    the located items of its batch, keeping its returned order among its peers.
    """
    batch_text = [_words(" ".join(m["content"] for m in b["messages"])) for b in conversation["batches"]]
    prefix = f"{user}/batch-"

    def key(pair: tuple[int, dict]) -> tuple[int, float, int]:
        rank, item = pair
        session = str(item.get("session_id", ""))
        if not session.startswith(prefix):
            return (10**6, math.inf, rank)
        batch = int(session[len(prefix):])
        head = " ".join(_words(str(item.get("content", ""))).split()[:12])
        position = batch_text[batch].find(head) if head and batch < len(batch_text) else -1
        return (batch, position if position >= 0 else math.inf, rank)

    return [item for _, item in sorted(enumerate(items), key=key)]


def arm_items(arm: str, items: list[dict], conversation: dict, user: str) -> list[dict]:
    """The items an arm answers from: all of them, re-ordered, or the first N as returned."""
    if arm == "chronological":
        return chronological(items, conversation, user)
    if arm in TOP_ARMS:
        return items[: TOP_ARMS[arm]]
    return items


#: v1 put this instruction above the session as a user message; on sessions of tens of thousands of
#: words gpt-4o-mini lost it and continued the dialogue instead (61 of 90 over 300 words, 21 read as
#: replies, 1 empty). v2 makes it a system message, fences the session as data, and repeats the task
#: after it.
SUMMARY_SYSTEM = (
    "You write memory summaries of recorded conversations. The user message contains one recorded "
    "session between a user and an assistant, fenced as data. Never reply to it or continue it. "
    "Summarize it: the main topics, every decision, plan, problem and result, and the events in "
    "the order they happened. Keep names, numbers and technical terms exactly. Include a date only "
    "if the session states it; never invent one. At most 250 words of plain prose, no preamble."
)
SUMMARY_REMINDER = (
    "Now write the summary of the recorded session above, following the system instructions: "
    "at most 250 words, third person, no reply to its content."
)


def summary_block(conversation: int, summaries: list[dict]) -> str:
    """This conversation's session summaries, in session order, as one context block."""
    own = sorted((s for s in summaries if s["conversation"] == conversation),
                 key=lambda s: s["batch"])
    return "\n\n".join(f"Session {s['batch'] + 1} summary: {s['summary']}" for s in own)


def summarize(data: list[dict], out: Path, workers: int, spend: Spend) -> None:
    log = Appender(out / "summaries.jsonl")
    done = {(r["conversation"], r["batch"]) for r in read_jsonl(out / "summaries.jsonl")}

    def one(conversation: int, batch: int, messages: list[dict]) -> None:
        session = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text = ""
        for _ in range(3):
            text = complete(spend, [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": f"<recorded_session>\n{session}\n</recorded_session>"
                                            f"\n\n{SUMMARY_REMINDER}"},
            ], 600, model=SUMMARY_MODEL)
            if text.strip():
                break
        log.write({"conversation": conversation, "batch": batch, "summary": text})

    pool(workers, [partial(one, c["conversation"], b, batch["messages"])
                   for c in data for b, batch in enumerate(c["batches"])
                   if (c["conversation"], b) not in done])


def context_of(items: list[dict]) -> str:
    return "\n\n".join(str(item.get("content", "")) for item in items)


def answer(data: list[dict], out: Path, arm: str, types: set[str] | None, workers: int,
           spend: Spend) -> None:
    state = state_of(out)
    by_conv = {c["conversation"]: c for c in data}
    questions = {q["id"]: q for c in data for q in c["questions"]}
    log = Appender(out / f"answers-{arm}.jsonl")
    done = {r["id"] for r in read_jsonl(out / f"answers-{arm}.jsonl")}
    summaries = read_jsonl(out / "summaries.jsonl")
    if arm == "summaries" and len(summaries) != sum(len(c["batches"]) for c in data):
        raise SystemExit("the summaries arm needs every session summarised first")

    def one(record: dict) -> None:
        items = arm_items(arm, record["items"], by_conv[record["conversation"]],
                          user_of(state, record["conversation"]))
        question = questions[record["id"]]
        context = context_of(items)
        if arm == "summaries":
            context = summary_block(record["conversation"], summaries) + "\n\n" + context
        prompt = (ANSWER_PROMPT.replace("<context>", context)
                  .replace("<question>", question["question"]))
        text = complete(spend, [{"role": "user", "content": prompt}], 512)
        log.write({"id": record["id"], "type": record["type"], "arm": arm, "answer": text})

    records = [r for r in read_jsonl(out / "retrieval.jsonl") if r["status"] == 200
               and r["id"] not in done and (types is None or r["type"] in types)]
    pool(workers, [partial(one, r) for r in records])


def _scores(response: str, count: int) -> list[float]:
    candidate = response.strip()
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    payload = json.loads(match.group(0) if match else candidate)
    raw = {int(item["index"]): float(item["score"]) for item in payload["scores"]}
    if set(raw) != set(range(count)) or any(v not in (0.0, 0.5, 1.0) for v in raw.values()):
        raise ValueError("judge did not score every index on the 0 / 0.5 / 1 scale")
    return [raw[index] for index in range(count)]


def _judged(spend: Spend, prompt: str, count: int) -> list[float] | None:
    for _ in range(3):
        try:
            return _scores(complete(spend, [{"role": "user", "content": prompt}], 1024, True), count)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


def kendall_tau_b(first: list[int], second: list[int]) -> float:
    concordant = discordant = first_ties = second_ties = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            a, b = first[left] - first[right], second[left] - second[right]
            if a == 0 and b == 0:
                continue
            if a == 0:
                first_ties += 1
            elif b == 0:
                second_ties += 1
            elif a * b > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + first_ties)
                            * (concordant + discordant + second_ties))
    return (concordant - discordant) / denominator if denominator else 0.0


def event_ordering_score(spend: Spend, reference: list[str], lines: list[str]) -> float:
    used: set[int] = set()
    system: list[str] = []
    for candidate in lines:
        matched = None
        for index, expected in enumerate(reference):
            if index in used:
                continue
            reply = complete(spend, [{"role": "system", "content": EQUIVALENCE_SYSTEM},
                                     {"role": "user", "content": f"First snippet: {expected}\n\n"
                                                                  f"Second snippet: {candidate}"}], 8)
            if "yes" in reply.casefold():
                matched = index
                break
        if matched is None:
            system.append(candidate)
        else:
            system.append(reference[matched])
            used.add(matched)
    tp = len(set(reference) & set(system))
    fp = len([s for s in system if s not in reference])
    fn = len([r for r in reference if r not in system])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = list(dict.fromkeys(reference + system))

    def ranks(sequence: list[str]) -> list[int]:
        positions = {item: index + 1 for index, item in enumerate(sequence)}
        return [positions.get(item, len(union) + 1) for item in union]

    return (kendall_tau_b(ranks(reference), ranks(system)) + 1) / 2 * f1


def judge(data: list[dict], out: Path, arm: str, workers: int, spend: Spend) -> None:
    questions = {q["id"]: q for c in data for q in c["questions"]}
    log = Appender(out / f"judged-{arm}.jsonl")
    done = {r["id"] for r in read_jsonl(out / f"judged-{arm}.jsonl")}

    def one(record: dict) -> None:
        question = questions[record["id"]]
        rubric = question["rubric"]
        criteria = "\n".join(f"[{i}] {r}" for i, r in enumerate(rubric))
        prompt = ("Evaluate every indexed RUBRIC CRITERION independently. Apply the complete "
                  "protocol below separately to each criterion; do not let one criterion affect "
                  "another.\n\n"
                  + JUDGE_BASE.replace("<question>", question["question"])
                  .replace("<rubric_item>", criteria).replace("<llm_response>", record["answer"])
                  .split("## OUTPUT FORMAT:")[0] + BATCH_OUTPUT_FORMAT)
        scores = _judged(spend, prompt, len(rubric)) if rubric else None
        result = {"id": record["id"], "type": record["type"], "arm": arm, "scores": scores,
                  "score": sum(scores) / len(scores) if scores else None}
        if record["type"] == "event_ordering" and rubric:
            result["event_ordering"] = event_ordering_score(spend, rubric, record["answer"].split("\n"))
        log.write(result)

    pool(workers, [partial(one, r) for r in read_jsonl(out / f"answers-{arm}.jsonl")
                   if r["id"] not in done])


def coverage(data: list[dict], out: Path, types: set[str] | None, workers: int,
             spend: Spend, top: int | None = None) -> None:
    questions = {q["id"]: q for c in data for q in c["questions"]}
    name = "coverage.jsonl" if top is None else f"coverage-top{top}.jsonl"
    log = Appender(out / name)
    done = {r["id"] for r in read_jsonl(out / name)}

    def one(record: dict) -> None:
        question = questions[record["id"]]
        rubric = question["rubric"]
        criteria = "\n".join(f"[{i}] {r}" for i, r in enumerate(rubric))
        prompt = (COVERAGE_PROMPT.replace("<question>", question["question"])
                  .replace("<rubric_item>", criteria)
                  .replace("<context>", context_of(record["items"][:top])))
        scores = _judged(spend, prompt, len(rubric)) if rubric else None
        log.write({"id": record["id"], "type": record["type"], "scores": scores,
                   "score": sum(scores) / len(scores) if scores else None})

    records = [r for r in read_jsonl(out / "retrieval.jsonl") if r["status"] == 200
               and r["id"] not in done and (types is None or r["type"] in types)]
    pool(workers, [partial(one, r) for r in records])


QUOTE_PROMPT = """For each RUBRIC CRITERION, find the passage in the CONTEXT that contains the information
needed to satisfy it, and copy it VERBATIM: the exact characters, at most 300 of them, no paraphrase,
no ellipsis. If the context does not contain that information, return an empty quote. Several
criteria may quote the same passage.

QUESTION: <question>

RUBRIC CRITERIA:
<rubric_item>

CONTEXT:
<context>

Return JSON only: {"quotes": [{"index": 0, "quote": "..."}]} with every index exactly once."""


def _normalised(text: str) -> str:
    return " ".join(text.split()).casefold()


def verified_quotes(context: str, quotes: dict[int, str], count: int) -> tuple[list[float], int]:
    """1.0 for a criterion whose quote occurs verbatim in the context, else 0.0.

    Whitespace and case are normalised, nothing else: a paraphrase, an invented passage or an
    ellipsis-joined quote does not count. Also returns how many non-empty quotes failed the check,
    which is the judge inventing evidence.
    """
    haystack = _normalised(context)
    scores: list[float] = []
    invented = 0
    for index in range(count):
        quote = _normalised(quotes.get(index, ""))
        if quote and quote in haystack:
            scores.append(1.0)
        else:
            scores.append(0.0)
            invented += bool(quote)
    return scores, invented


def quote_coverage(data: list[dict], out: Path, types: set[str] | None, workers: int,
                   spend: Spend) -> None:
    questions = {q["id"]: q for c in data for q in c["questions"]}
    log = Appender(out / "coverage-quoted.jsonl")
    done = {r["id"] for r in read_jsonl(out / "coverage-quoted.jsonl")}

    def one(record: dict) -> None:
        question = questions[record["id"]]
        rubric = question["rubric"]
        context = context_of(record["items"])
        criteria = "\n".join(f"[{i}] {r}" for i, r in enumerate(rubric))
        prompt = (QUOTE_PROMPT.replace("<question>", question["question"])
                  .replace("<rubric_item>", criteria).replace("<context>", context))
        result: dict[str, Any] = {"id": record["id"], "type": record["type"], "scores": None,
                                  "score": None, "invented": None}
        for _ in range(3):
            try:
                reply = complete(spend, [{"role": "user", "content": prompt}], 2048, True,
                                 model=QUOTE_MODEL)
                match = re.search(r"\{.*\}", reply, re.DOTALL)
                payload = json.loads(match.group(0) if match else reply)
                quotes = {int(q["index"]): str(q.get("quote") or "") for q in payload["quotes"]}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            scores, invented = verified_quotes(context, quotes, len(rubric))
            result.update(scores=scores, score=sum(scores) / len(scores) if scores else None,
                          invented=invented)
            break
        log.write(result)

    records = [r for r in read_jsonl(out / "retrieval.jsonl") if r["status"] == 200
               and r["id"] not in done and (types is None or r["type"] in types)
               and questions[r["id"]]["rubric"]]
    pool(workers, [partial(one, r) for r in records])


def report(out: Path, spend: Spend) -> dict[str, Any]:
    adds = read_jsonl(out / "adds.jsonl")
    retrieval = read_jsonl(out / "retrieval.jsonl")

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 4) if values else None

    per_type: dict[str, dict[str, Any]] = {}
    for name in ["coverage", "coverage-quoted",
                 *(f"coverage-top{n}" for n in TOP_ARMS.values())]:
        for record in read_jsonl(out / f"{name}.jsonl"):
            per_type.setdefault(record["type"], {}).setdefault(name.replace("-", "_"), []).append(
                record["score"])
    for arm in ARMS:
        for record in read_jsonl(out / f"judged-{arm}.jsonl"):
            row = per_type.setdefault(record["type"], {})
            row.setdefault(arm, []).append(record["score"])
            if "event_ordering" in record:
                row.setdefault(f"{arm}_kendall_f1", []).append(record["event_ordering"])
    table = {
        kind: {name: {"mean": mean([v for v in values if v is not None]), "n": len(values),
                      "unscored": sum(1 for v in values if v is None)}
               for name, values in row.items()}
        for kind, row in sorted(per_type.items())
    }
    summary = {
        "adds": {"count": len(adds), "ok": sum(1 for a in adds if a["status"] == 200),
                 "compiler_fallbacks": sum(1 for a in adds if a.get("compiler_fallback")),
                 "seconds_max": max((a["seconds"] for a in adds), default=0.0),
                 "retried": sum(1 for a in adds if a["attempts"] > 1)},
        "searches": {"count": len(retrieval),
                     "ok": sum(1 for r in retrieval if r["status"] == 200),
                     "items_mean": mean([len(r["items"]) for r in retrieval]),
                     "graph_fallbacks": sum(1 for r in retrieval
                                            if r["headers"].get("x-recall-graph-fallback") == "1"),
                     "atomic_fallbacks": sum(1 for r in retrieval
                                             if r["headers"].get("x-recall-atomic-rescue-fallback") == "1")},
        "by_type": table,
        "model_spend_usd_this_process": round(spend.usd, 4),
        "model_calls_this_process": spend.calls,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def cleanup(service: Service, data: list[dict], out: Path) -> dict[str, int]:
    state = state_of(out)
    result = {}
    for conversation in data:
        user = user_of(state, conversation["conversation"])
        status, body, _, _ = service.call("/v1/delete", {"user_id": user}, attempts=3, timeout=300)
        result[user] = body.get("deleted_count", -1) if status == 200 else -status
    (out / "cleanup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["ingest", "retrieve", "summarize", "answer", "judge",
                                          "coverage", "quote-coverage", "report", "cleanup"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18015")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arm", choices=ARMS, default="returned")
    parser.add_argument("--types", help="comma separated question types (default: all)")
    parser.add_argument("--conversations", help="comma separated indices (default: all)")
    parser.add_argument("--top", type=int, help="coverage over the first N items only")
    parser.add_argument("--cap-usd", type=float, default=3.0, help="model spend cap per process")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(args.data)
    if args.conversations:
        wanted = {int(v) for v in args.conversations.split(",")}
        data = [c for c in data if c["conversation"] in wanted]
    types = set(args.types.split(",")) if args.types else None
    spend = Spend(args.cap_usd)
    service = Service(args.base_url, os.environ.get("RECALL_AML_API_KEY", "").strip())
    if args.phase == "ingest":
        ingest(service, data, args.out, args.workers)
    elif args.phase == "retrieve":
        retrieve(service, data, args.out, args.workers)
    elif args.phase == "summarize":
        summarize(data, args.out, args.workers, spend)
    elif args.phase == "answer":
        answer(data, args.out, args.arm, types, args.workers, spend)
    elif args.phase == "judge":
        judge(data, args.out, args.arm, args.workers, spend)
    elif args.phase == "coverage":
        coverage(data, args.out, types, args.workers, spend, args.top)
    elif args.phase == "quote-coverage":
        quote_coverage(data, args.out, types, args.workers, spend)
    elif args.phase == "cleanup":
        print(json.dumps(cleanup(service, data, args.out), indent=2))
        return
    print(json.dumps(report(args.out, spend), indent=2))


if __name__ == "__main__":
    main()
