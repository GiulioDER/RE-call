"""RC-3: Hindsight's four benchmark-shaped reader rules, each on its own, over the shipped dated prompt.

Pre-registration: kept in the maintainer's private research log (RC-3, 2026-09-29).

RC-1 measured the four rules only together (arm HB): +0.100 [+0.033, +0.167] over the dated
contract on 120 LongMemEval-S questions, mostly multi-session (0.32 to 0.58), and could not say which
rule did it. This separates them. Every arm starts from the dated profile exactly as master ships it
(`recall.evidence.render_dated_evidence_prompt`, `DATED_SYSTEM_PROMPT`, with #814's "JSON"):

* ``D``   the shipped default;  ``D2`` D again, the noise floor;
* ``R8``  D + rule 8 alone (recommendations answered as a description of what the user prefers);
* ``R9``  D + rule 9 alone (help requests use the user's current situation);
* ``R10`` D + rule 10 alone (a value question gets the latest value and the earlier ones, explained);
* ``R11`` D + rule 11 alone (detailed answers, every relevant detail from every item);
* ``HB``  D + all four, byte for byte RC-1's `BENCHMARK_CONTRACT`.

A single rule is renumbered 8 so it reads as the next rule of the contract; its words are unchanged.

Stage 1 is LongMemEval-S (RC-1's 120 questions and evidence, DeepSeek v4.1 flash reader,
LongMemEval's own per-type judge on gpt-4o). Stage 2 is LoCoMo (RC-2's 720 questions and evidence,
Gemini 2.5 Flash reader, `benchmarks.pipeline`'s judge on DeepSeek), for the arms Stage 1 advances.
Answers go through the library's provider, parser, normaliser and validator, as in RC-1 and RC-2.

    python scripts/reasoning_hb_rules.py run --dataset lme --stageb S.jsonl --data lme.json --out a.jsonl
    python scripts/reasoning_hb_rules.py run --dataset locomo --collected C.json.gz --data locomo10.json \\
        --draw draw.json --arms D,D2,R11 --out b.jsonl
    python scripts/reasoning_hb_rules.py score --answers a.jsonl --out score.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pipeline import JUDGE_SYSTEM_PROMPT  # noqa: E402
from recall.evidence import (  # noqa: E402
    DATED_SYSTEM_PROMPT,
    EvidenceBundle,
    EvidenceItem,
    normalize_citations,
    parse_answer_envelope,
    render_dated_evidence_prompt,
    validate_answer,
)

SEED = 20260929
TOP_K = 10
ALL_ARMS = ("D", "D2", "R8", "R9", "R10", "R11", "HB")
REFUSAL = "I don't have enough information in my memory to answer that."
DATE_HEADER = re.compile(r"^\[[^\]]*UTC\]\s*")
_SESSION = re.compile(r"^session_(\d+)_date_time$")

#: RC-1's `BENCHMARK_CONTRACT`, byte for byte (the HB arm).
BENCHMARK_CONTRACT = """
8. For a request for recommendations, suggestions or tips, do not recommend specific items:
   describe what KIND of thing this user would prefer, and what they would not, from what the
   evidence says about them.
9. For a request for help or instructions, use what the evidence says about the user's current
   situation (recent purchases, the specific models they use, earlier conversations).
10. For a question about a number or value, give the most recent value and say which earlier values
   there were and why they are no longer current.
11. Make answers detailed: include every relevant detail the evidence gives, from every item."""

#: Each rule alone, renumbered 8 (the next number after the dated contract's 7); words unchanged.
SINGLE_RULES = {
    "R8": """
8. For a request for recommendations, suggestions or tips, do not recommend specific items:
   describe what KIND of thing this user would prefer, and what they would not, from what the
   evidence says about them.""",
    "R9": """
8. For a request for help or instructions, use what the evidence says about the user's current
   situation (recent purchases, the specific models they use, earlier conversations).""",
    "R10": """
8. For a question about a number or value, give the most recent value and say which earlier values
   there were and why they are no longer current.""",
    "R11": """
8. Make answers detailed: include every relevant detail the evidence gives, from every item.""",
}


def system_prompt(arm: str) -> str:
    if arm in ("D", "D2"):
        return DATED_SYSTEM_PROMPT
    if arm == "HB":
        return DATED_SYSTEM_PROMPT + BENCHMARK_CONTRACT
    if arm in SINGLE_RULES:
        return DATED_SYSTEM_PROMPT + SINGLE_RULES[arm]
    raise ValueError(f"unknown arm {arm!r}")


def render(arm: str, bundle: EvidenceBundle, question_date: str) -> tuple[str, str]:
    """(system, user): the shipped dated rendering, with the arm's rules appended to its system."""
    shipped_system, user = render_dated_evidence_prompt(bundle, question_date)
    assert shipped_system is DATED_SYSTEM_PROMPT
    return system_prompt(arm), user


def _stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_bundle(query: str, items: Sequence[Mapping[str, Any]], *, strip_header: bool) -> EvidenceBundle:
    evidence = tuple(
        EvidenceItem(
            chunk_id=str(item["id"]),
            text=DATE_HEADER.sub("", str(item["content"]), count=1) if strip_header else str(item["content"]),
            source=str(item.get("source") or item.get("session_id") or ""), ordinal=rank,
            indexed_at=_stamp(str(item["created_at"])) if item.get("created_at") else None,
            valid_from=None, valid_until=None, cosine=0.0, confidence=0.0,
        )
        for rank, item in enumerate(items[:TOP_K])
        if isinstance(item.get("content"), str)
    )
    return EvidenceBundle(
        query=query, decision="answer", reason_code=None, decision_state="supported", calibrated=True,
        stale=False, embedding_profile="c9-stored", retrieval_profile="c9-stored", index_generation="stored",
        items=evidence,
    )


def hypothesis(raw: str | None, bundle: EvidenceBundle) -> tuple[str, str]:
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


# ------------------------------------------------------------------------------------ datasets


def lme_judge_prompt(question: Mapping[str, Any], response: str) -> str:
    """LongMemEval's `get_anscheck_prompt`, per type and for `_abs` ids (as RC-1)."""
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


def last_session_time(conversation: Mapping[str, Any]) -> datetime:
    numbered = sorted((int(m.group(1)), key) for key in conversation if (m := _SESSION.match(key)))
    raw = str(conversation[numbered[-1][1]])
    return datetime.strptime(raw, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def load_lme(stageb: Path, data: Path) -> list[dict[str, Any]]:
    """RC-1's questions: each with its question date (raw, as RC-1 sent it) and the judge prompt."""
    questions = {str(q["question_id"]): q for q in json.loads(data.read_text(encoding="utf-8"))}
    out = []
    for line in stageb.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        search = record["searches"][0]
        q = questions[str(search["question_id"])]
        out.append({"id": str(q["question_id"]), "group": "abstention" if "_abs" in str(q["question_id"])
                    else str(q["question_type"]), "question": str(q["question"]),
                    "question_date": str(q["question_date"]),
                    "bundle": build_bundle(str(q["question"]), search["items"], strip_header=True),
                    "judge": ("lme", q)})
    return sorted(out, key=lambda row: row["id"])


def load_locomo(collected: Path, data: Path, draw: Path) -> list[dict[str, Any]]:
    """RC-2's questions: question date = the conversation's last session, as ISO."""
    rows = {r["id"]: r for r in json.load(gzip.open(collected))["rows"]}
    conversations = {c["sample_id"]: c for c in json.loads(data.read_text(encoding="utf-8"))}
    out = []
    for ident in json.loads(draw.read_text(encoding="utf-8"))["ids"]:
        sample, index = ident.rsplit(":", 1)
        conv = conversations[sample]
        qa = conv["qa"][int(index)]
        out.append({"id": ident, "group": str(qa["category"]), "question": str(qa["question"]),
                    "question_date": last_session_time(conv["conversation"]).isoformat(),
                    "bundle": build_bundle(str(qa["question"]), rows[ident]["items"], strip_header=False),
                    "judge": ("locomo", {"question": str(qa["question"]), "gold": str(qa.get("answer", ""))})})
    return out


# ------------------------------------------------------------------------------------------- run


def _pricing(model: str) -> tuple[float, float]:
    import httpx

    for entry in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]:
        if entry["id"] == model:
            return float(entry["pricing"]["prompt"]), float(entry["pricing"]["completion"])
    raise SystemExit(f"no OpenRouter price for {model}")


def _judge(kind: str, payload: Mapping[str, Any], response: str) -> tuple[bool, str, float]:
    import httpx

    if kind == "lme":
        body_json: dict[str, Any] = {"model": "openai/gpt-4o", "temperature": 0, "max_tokens": 10,
                                     "messages": [{"role": "user", "content": lme_judge_prompt(payload, response)}]}
    else:
        user = f"Question: {payload['question']}\nGold answer: {payload['gold']}\nPredicted answer: {response}\nCorrect?"
        body_json = {"model": "deepseek/deepseek-v4.1-flash", "temperature": 0, "max_tokens": 10,
                     "reasoning": {"effort": "none"},
                     "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}]}
    body_json["usage"] = {"include": True}
    body = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      json=body_json, timeout=120).json()
    verdict = str(body["choices"][0]["message"]["content"] or "").strip()
    correct = ("yes" in verdict.lower()) if kind == "lme" else verdict.casefold().startswith("yes")
    return correct, verdict, float((body.get("usage") or {}).get("cost") or 0.0)


def run(args: argparse.Namespace) -> None:
    from recall.answer_provider import resolve_answer_provider

    provider = resolve_answer_provider(os.environ)
    if provider is None:
        raise SystemExit("RECALL_REASONING_ANSWER_ENABLED is not set")
    arms = tuple(args.arms.split(","))
    if not set(arms) <= set(ALL_ARMS):
        raise SystemExit(f"--arms must name only {ALL_ARMS}")
    price_in, price_out = _pricing(os.environ["RECALL_REASONING_ANSWER_MODEL"])
    rows = (load_lme(args.stageb, args.data) if args.dataset == "lme"
            else load_locomo(args.collected, args.data, args.draw))
    if args.limit:
        rows = rows[: args.limit]
    done: set[tuple[str, str]] = set()
    spent = [0.0]
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["id"], record["arm"]))
            spent[0] += float(record.get("cost") or 0.0)
    lock, stop, local = threading.Lock(), threading.Event(), threading.local()
    print(json.dumps({"dataset": args.dataset, "questions": len(rows), "arms": arms, "done": len(done),
                      "spent_usd": round(spent[0], 4), "reader": os.environ["RECALL_REASONING_ANSWER_MODEL"]}),
          flush=True)

    def work(index_row: tuple[int, dict[str, Any]]) -> None:
        index, row = index_row
        for arm in arms[index % len(arms):] + arms[: index % len(arms)]:
            if (row["id"], arm) in done or stop.is_set():
                continue
            local.error = None
            system, user = render(arm, row["bundle"], row["question_date"])
            try:
                raw = provider(system, user)
            except Exception as exc:  # BROAD-CATCH: a provider failure is the tool's refusal, recorded
                local.error, raw = f"{type(exc).__name__}: {exc}"[:200], None
            meta = provider.provider_metadata()
            answer_cost = (meta.prompt_tokens or 0) * price_in + (meta.completion_tokens or 0) * price_out
            text, outcome = hypothesis(raw, row["bundle"])
            try:
                correct, verdict, judge_cost = _judge(row["judge"][0], row["judge"][1], text)
            except Exception as exc:  # BROAD-CATCH: an unjudged answer is dropped and retried on resume
                print(json.dumps({"judge_error": row["id"], "arm": arm, "error": str(exc)[:200]}), flush=True)
                continue
            record = {"id": row["id"], "arm": arm, "group": row["group"], "outcome": outcome, "hypothesis": text,
                      "raw": raw, "provider_error": local.error, "verdict": verdict, "correct": correct,
                      "completion_tokens": meta.completion_tokens, "cost": round(answer_cost + judge_cost, 6)}
            with lock:
                spent[0] += record["cost"]
                if spent[0] > args.max_usd:
                    stop.set()
                with args.out.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, enumerate(rows)))
    print(json.dumps({"spent_usd": round(spent[0], 4), "stopped": stop.is_set()}), flush=True)


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
    tokens: dict[str, list[int]] = defaultdict(list)
    answered: dict[str, list[bool]] = defaultdict(list)
    groups: dict[str, str] = {}
    cost = 0.0
    for line in args.answers.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] in correct[record["arm"]]:
            raise SystemExit(f"{record['arm']} answered {record['id']} twice")
        correct[record["arm"]][record["id"]] = bool(record["correct"])
        outcomes[record["arm"]][record["outcome"]] += 1
        groups[record["id"]] = str(record["group"])
        if record.get("completion_tokens"):
            tokens[record["arm"]].append(int(record["completion_tokens"]))
        if record["outcome"] == "answered":
            answered[record["arm"]].append(bool(record["correct"]))
        cost += float(record.get("cost") or 0.0)
    arms = [a for a in ALL_ARMS if correct[a]]
    every = sorted(groups)
    result: dict[str, Any] = {
        "answers": {a: len(correct[a]) for a in arms},
        "accuracy": {a: round(sum(correct[a].values()) / len(correct[a]), 4) for a in arms},
        "outcomes": {a: dict(outcomes[a]) for a in arms},
        "correct_among_answered": {a: round(sum(v) / len(v), 4) for a, v in answered.items() if v},
        "median_completion_tokens": {a: statistics.median(v) for a, v in tokens.items() if v},
        "cost_usd": round(cost, 4),
        "minus_D": {a: paired(correct, a, "D", every) for a in arms if a != "D"},
        "per_group": {
            g: {"n": sum(groups[k] == g for k in every),
                **{a: round(sum(correct[a].get(k, False) for k in every if groups[k] == g)
                            / max(1, sum(groups[k] == g for k in every)), 4) for a in arms},
                "minus_D": {a: paired(correct, a, "D", [k for k in every if groups[k] == g]) for a in arms if a != "D"}}
            for g in sorted(set(groups.values()))
        },
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("accuracy", "outcomes", "correct_among_answered",
                                              "median_completion_tokens", "cost_usd")}, indent=1))
    print(json.dumps({a: (r["diff"], r["ci95"], r["wins"], r["losses"]) for a, r in result["minus_D"].items()}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", choices=("lme", "locomo"), required=True)
    r.add_argument("--stageb", type=Path)
    r.add_argument("--collected", type=Path)
    r.add_argument("--draw", type=Path)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--arms", default=",".join(ALL_ARMS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-usd", type=float, default=3.0)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"run": run, "score": score}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
