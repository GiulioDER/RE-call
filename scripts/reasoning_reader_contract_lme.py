"""RC-1: does Hindsight's reader contract help `recall_reasoning_query` answer LongMemEval-S?

Pre-registration: kept in the maintainer's private research log (RC-1 reasoning reader contract,
2026-09-29).

vectorize-io/hindsight (read at 26981c6b) and its benchmark harness (vectorize-io/agent-memory-
benchmark) hand the reader three things RE-call's answer boundary does not: the QUESTION'S DATE,
each memory's DATE under a name that says what it is, and date-aware instructions (the timeline,
the most recent statement as the current fact, list-then-count, both answers when two remain,
date arithmetic only when both dates are known). In their same-harness comparison nearly all of
the gap to a plain hybrid baseline is temporal (+52) and preference (+43) questions.

This holds the evidence FIXED and changes only the prompt at the library's answer boundary. Each of
120 LongMemEval-S questions gets the top 10 items of RE-call's stored served retrieval (X-1 Stage B,
C9 at 3eb447c4), rebuilt as `recall.evidence.EvidenceItem`s: text without C9's date header,
`indexed_at` = the session's timestamp (what a live deployment writing each session as it happened
records). Every arm goes through the library's own provider (`resolve_answer_provider`), envelope
parser, citation normaliser and validator, exactly as `recall.reasoning._answer_from_evidence`
does; a response the validator rejects is scored as the tool's refusal.

Arms:

* ``R``  the shipped boundary, `render_evidence_prompt(bundle)` unchanged;
* ``R2`` R again, the noise floor;
* ``Q``  R plus the question's date in the payload (the request's `as_of`, which today never
  reaches the prompt), system prompt unchanged;
* ``H``  Q plus each item's date under the name ``date`` and Hindsight's GENERIC reader contract
  appended to the unchanged system prompt;
* ``HB`` H plus the benchmark-shaped rules of Hindsight's LongMemEval prompt (recommendations as
  a description of what the user would prefer, detail-maximal answers). Reported, never a product
  candidate: those rules are written to the LongMemEval rubric.

Judge: LongMemEval's own per-type prompts (``xiaowu0162/LongMemEval`` `evaluate_qa.py`), gpt-4o
through OpenRouter, temperature 0, 10 tokens, ``yes`` in the reply is correct, as the paper did.

    python scripts/reasoning_reader_contract_lme.py run --stageb longmemeval_s.jsonl \\
        --data longmemeval_s_cleaned.json --out answers.jsonl
    python scripts/reasoning_reader_contract_lme.py score --answers answers.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall.evidence import (  # noqa: E402
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    SYSTEM_PROMPT,
    EvidenceBundle,
    EvidenceItem,
    _encode,
    _item_payload,
    normalize_citations,
    parse_answer_envelope,
    render_evidence_prompt,
    validate_answer,
)

SEED = 20260929
TOP_K = 10
ARMS = ("R", "R2", "Q", "H", "HB")
JUDGE_MODEL = "openai/gpt-4o"
DATE_HEADER = re.compile(r"^\[[^\]]*UTC\]\s*")
REFUSAL = "I don't have enough information in my memory to answer that."

#: Hindsight's reader contract, generic half: what its reflect prompt and its LongMemEval reader
#: prompt ask of any question, restated. Appended AFTER the unchanged system prompt, so the safety
#: contract (untrusted data, citations, the envelope) still comes first.
GENERIC_CONTRACT = """

How to read the evidence:
1. Each evidence item has a `date`: when it was said. `question_date` is when the question is asked.
   First work out what happened and in what order.
2. When items disagree about the same thing, the one with the LATEST date is the current fact; an
   earlier value is history. Mention the earlier value only when it helps explain the answer.
3. For "how many" questions, list every distinct item you find across ALL the evidence, then count
   the list. Do not count the same item twice.
4. If two different answers remain possible, give both and say why.
5. For questions about time ("when", "how long ago", "how many days between"), compare the items'
   dates with each other and with `question_date`. Compute a difference only when both dates are
   known; otherwise say what is missing.
6. If the question names a specific thing and the evidence is about a different one (another sport,
   another person, a show rather than a podcast), do not substitute it: say the evidence does not
   cover it.
7. For a comparison, answer only when the evidence covers every side of it."""

#: Hindsight's LongMemEval reader prompt, benchmark-shaped half (restated). Its rules follow the
#: LongMemEval rubric (a preference question is judged on whether the answer uses the user's
#: preferences, not on a recommendation), so this arm is reported and never a product candidate.
BENCHMARK_CONTRACT = """
8. For a request for recommendations, suggestions or tips, do not recommend specific items:
   describe what KIND of thing this user would prefer, and what they would not, from what the
   evidence says about them.
9. For a request for help or instructions, use what the evidence says about the user's current
   situation (recent purchases, the specific models they use, earlier conversations).
10. For a question about a number or value, give the most recent value and say which earlier values
   there were and why they are no longer current.
11. Make answers detailed: include every relevant detail the evidence gives, from every item."""


def _stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_bundle(query: str, items: Sequence[Mapping[str, Any]]) -> EvidenceBundle:
    """The top `TOP_K` stored items as the library's evidence bundle (text without C9's header,
    `indexed_at` the session's timestamp)."""
    evidence = tuple(
        EvidenceItem(
            chunk_id=str(item["id"]), text=DATE_HEADER.sub("", str(item["content"]), count=1),
            source=str(item.get("source") or ""), ordinal=rank,
            indexed_at=_stamp(str(item["created_at"])) if item.get("created_at") else None,
            valid_from=None, valid_until=None,
            cosine=float(item.get("score") or 0.0), confidence=float(item.get("score") or 0.0),
        )
        for rank, item in enumerate(items[:TOP_K])
        if isinstance(item.get("content"), str)
    )
    return EvidenceBundle(
        query=query, decision="answer", reason_code=None, decision_state="supported", calibrated=True,
        stale=False, embedding_profile="c9-stored", retrieval_profile="c9-stored",
        index_generation="x1-stage-b", items=evidence,
    )


def _schema() -> dict[str, str]:
    return {"answer": "string or null", "citations": "array of chunk_id strings",
            "insufficient_evidence": "boolean"}


def render(arm: str, bundle: EvidenceBundle, question_date: str) -> tuple[str, str]:
    """(system, user) for ``arm``. R and R2 are the library's own rendering, byte for byte."""
    if arm in ("R", "R2"):
        return render_evidence_prompt(bundle)
    if arm == "Q":
        data: dict[str, object] = {"query": bundle.query, "question_date": question_date,
                                   "evidence": [_item_payload(i) for i in bundle.items], "answer_schema": _schema()}
        return SYSTEM_PROMPT, f"{EVIDENCE_OPEN}{_encode(data)}{EVIDENCE_CLOSE}"
    if arm in ("H", "HB"):
        evidence = []
        for item in bundle.items:
            payload = _item_payload(item)
            payload["date"] = payload["indexed_at"]
            evidence.append(payload)
        data = {"query": bundle.query, "question_date": question_date, "evidence": evidence,
                "answer_schema": _schema()}
        system = SYSTEM_PROMPT + GENERIC_CONTRACT + (BENCHMARK_CONTRACT if arm == "HB" else "")
        return system, f"{EVIDENCE_OPEN}{_encode(data)}{EVIDENCE_CLOSE}"
    raise ValueError(f"unknown arm {arm!r}")


def hypothesis(raw: str | None, bundle: EvidenceBundle) -> tuple[str, str]:
    """(text the judge reads, outcome), through the library's parser, normaliser and validator.

    A rejected envelope is what `recall_reasoning_query` would refuse, so the judge reads the
    refusal, not the rejected text."""
    if raw is None:
        return REFUSAL, "provider_error"
    try:
        envelope = normalize_citations(parse_answer_envelope(raw))
    except Exception:  # BROAD-CATCH: an unparseable envelope is the tool's refusal
        return REFUSAL, "unparseable"
    if not validate_answer(envelope, bundle).valid:
        return REFUSAL, "invalid"
    if envelope.insufficient_evidence:
        return REFUSAL, "insufficient"
    return str(envelope.answer), "answered"


def judge_prompt(question: Mapping[str, Any], response: str) -> str:
    """LongMemEval's `get_anscheck_prompt`, verbatim in substance, per type and for `_abs` ids."""
    q, a = question["question"], question["answer"]
    head = ("I will give you a question, a correct answer, and a response from a model. Please answer yes "
            "if the response contains the correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the intermediate steps to get the correct "
            "answer, you should also answer yes. If the response only contains a subset of the "
            "information required by the answer, answer no. ")
    if "_abs" in str(question["question_id"]):
        return ("I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly identifies the question as unanswerable. The "
                "model could say that the information is incomplete, or some other information is given "
                "but the asked information is not.\n\n"
                f"Question: {q}\n\nExplanation: {a}\n\nModel Response: {response}\n\n"
                "Does the model correctly identify the question as unanswerable? Answer yes or no only.")
    kind = question["question_type"]
    if kind in ("single-session-user", "single-session-assistant", "multi-session"):
        body = head
    elif kind == "temporal-reasoning":
        body = head + ("In addition, do not penalize off-by-one errors for the number of days. If the "
                       "question asks for the number of days/weeks/months, etc., and the model makes "
                       "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
                       "response is still correct. ")
    elif kind == "knowledge-update":
        body = ("I will give you a question, a correct answer, and a response from a model. Please answer "
                "yes if the response contains the correct answer. Otherwise, answer no. If the response "
                "contains some previous information along with an updated answer, the response should be "
                "considered as correct as long as the updated answer is the required answer.")
    elif kind == "single-session-preference":
        return ("I will give you a question, a rubric for desired personalized response, and a response "
                "from a model. Please answer yes if the response satisfies the desired response. "
                "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal "
                "information correctly.\n\n"
                f"Question: {q}\n\nRubric: {a}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only.")
    else:
        raise ValueError(f"unknown question type {kind!r}")
    return (f"{body.rstrip()}\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only.")


# ------------------------------------------------------------------------------------------- run


class Spend:
    def __init__(self, cap: float) -> None:
        self.cap, self.usd, self.lock, self.stopped = cap, 0.0, threading.Lock(), threading.Event()

    def add(self, usd: float) -> None:
        with self.lock:
            self.usd += usd
            if self.usd > self.cap:
                self.stopped.set()


def _pricing(model: str) -> tuple[float, float]:
    import httpx

    for entry in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]:
        if entry["id"] == model:
            return float(entry["pricing"]["prompt"]), float(entry["pricing"]["completion"])
    raise SystemExit(f"no OpenRouter price for {model}")


def _judge(prompt: str) -> tuple[str, float]:
    import httpx

    body = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
              "max_tokens": 10, "usage": {"include": True}},
        timeout=120,
    ).json()
    return str(body["choices"][0]["message"]["content"]).strip(), float((body.get("usage") or {}).get("cost") or 0.0)


def load(stageb: Path, data: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = {str(q["question_id"]): q for q in json.loads(data.read_text(encoding="utf-8"))}
    rows = []
    for line in stageb.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            search = record["searches"][0]
            rows.append({"id": str(search["question_id"]), "items": search["items"]})
    rows.sort(key=lambda row: row["id"])
    return rows, questions


def run(args: argparse.Namespace) -> None:
    from recall.answer_provider import resolve_answer_provider

    provider = resolve_answer_provider(os.environ)
    if provider is None:
        raise SystemExit("RECALL_REASONING_ANSWER_ENABLED is not set")
    price_in, price_out = _pricing(os.environ["RECALL_REASONING_ANSWER_MODEL"])
    rows, questions = load(args.stageb, args.data)
    done: set[tuple[str, str]] = set()
    spend = Spend(args.max_usd)
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["id"], record["arm"]))
            spend.add(float(record.get("cost") or 0.0))
    lock = threading.Lock()
    local = threading.local()
    print(json.dumps({"questions": len(rows), "done": len(done), "spent_usd": round(spend.usd, 4),
                      "reader": os.environ["RECALL_REASONING_ANSWER_MODEL"], "judge": JUDGE_MODEL}), flush=True)

    def answer(system: str, user: str) -> tuple[str | None, float, dict[str, Any]]:
        try:
            raw = provider(system, user)
        except Exception as exc:  # BROAD-CATCH: a provider failure is the tool's refusal, recorded
            local.error = f"{type(exc).__name__}: {exc}"[:200]
            raw = None
        meta = provider.provider_metadata()
        usage = {"prompt_tokens": meta.prompt_tokens, "completion_tokens": meta.completion_tokens}
        cost = ((meta.prompt_tokens or 0) * price_in) + ((meta.completion_tokens or 0) * price_out)
        return raw, cost, usage

    def work(index_row: tuple[int, dict[str, Any]]) -> None:
        index, row = index_row
        question = questions[row["id"]]
        bundle = build_bundle(str(question["question"]), row["items"])
        order = ARMS[index % len(ARMS):] + ARMS[: index % len(ARMS)]
        for arm in order:
            if (row["id"], arm) in done or spend.stopped.is_set():
                continue
            local.error = None
            system, user = render(arm, bundle, str(question["question_date"]))
            raw, answer_cost, usage = answer(system, user)
            text, outcome = hypothesis(raw, bundle)
            try:
                verdict, judge_cost = _judge(judge_prompt(question, text))
            except Exception as exc:  # BROAD-CATCH: an unjudged answer is dropped and retried on resume
                print(json.dumps({"judge_error": row["id"], "arm": arm, "error": str(exc)[:200]}), flush=True)
                continue
            spend.add(answer_cost + judge_cost)
            record = {"id": row["id"], "arm": arm, "type": question["question_type"],
                      "abstention": "_abs" in row["id"], "outcome": outcome, "hypothesis": text, "raw": raw,
                      "provider_error": local.error, "verdict": verdict, "correct": "yes" in verdict.lower(),
                      "usage": usage, "cost": round(answer_cost + judge_cost, 6)}
            with lock, args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, enumerate(rows)))
    print(json.dumps({"spent_usd": round(spend.usd, 4), "stopped": spend.stopped.is_set()}), flush=True)


# ----------------------------------------------------------------------------------------- score


def paired(correct: Mapping[str, Mapping[str, bool]], arm: str, base: str, keys: Sequence[str]) -> dict[str, Any]:
    keys = [k for k in keys if k in correct[arm] and k in correct[base]]
    if not keys:
        return {"n": 0}
    diffs = [int(correct[arm][k]) - int(correct[base][k]) for k in keys]
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000))
    return {"n": len(keys), "arm": round(sum(correct[arm][k] for k in keys) / len(keys), 4),
            "base": round(sum(correct[base][k] for k in keys) / len(keys), 4),
            "diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(boots[249], 4), round(boots[9_749], 4)],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}


def score(args: argparse.Namespace) -> None:
    correct: dict[str, dict[str, bool]] = defaultdict(dict)
    outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    types: dict[str, str] = {}
    cost = 0.0
    for line in args.answers.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] in correct[record["arm"]]:
            raise SystemExit(f"{record['arm']} answered {record['id']} twice")
        correct[record["arm"]][record["id"]] = bool(record["correct"])
        outcomes[record["arm"]][record["outcome"]] += 1
        types[record["id"]] = "abstention" if record["abstention"] else record["type"]
        cost += float(record.get("cost") or 0.0)
    every = sorted(types)
    result: dict[str, Any] = {
        "answers": {arm: len(correct[arm]) for arm in ARMS},
        "accuracy": {arm: round(sum(correct[arm].values()) / len(correct[arm]), 4) for arm in ARMS if correct[arm]},
        "outcomes": {arm: dict(outcomes[arm]) for arm in ARMS},
        "cost_usd": round(cost, 4),
        "noise_R2_minus_R": paired(correct, "R2", "R", every),
        "Q_minus_R": paired(correct, "Q", "R", every),
        "H_minus_R": paired(correct, "H", "R", every),
        "HB_minus_H": paired(correct, "HB", "H", every),
        "HB_minus_R": paired(correct, "HB", "R", every),
        "per_type": {},
    }
    for kind in sorted(set(types.values())):
        keys = [k for k in every if types[k] == kind]
        result["per_type"][kind] = {
            "n": len(keys),
            **{arm: round(sum(correct[arm].get(k, False) for k in keys) / len(keys), 4) for arm in ARMS},
            "H_minus_R": paired(correct, "H", "R", keys),
        }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stageb", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-usd", type=float, default=4.0)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"run": run, "score": score}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
