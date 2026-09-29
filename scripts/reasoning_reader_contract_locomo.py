"""RC-2: the `dated` answer profile on LoCoMo with a second reader, before it can become a default.

Pre-registration: kept in the maintainer's private research log (RC-2, 2026-09-29).

RC-1 measured the profile on LongMemEval-S with one reader (DeepSeek v4.1 flash): +0.108 [+0.033,
+0.183]. Its own rule says a default change needs a second dataset AND a second reader. This is
both: LoCoMo, answered by gpt-4o-mini, judged by RE-call's own LoCoMo judge
(`benchmarks.pipeline.JUDGE_SYSTEM_PROMPT`, the head-to-head benchmark's) on DeepSeek v4.1 flash, so
the judge is not the reader.

The evidence is RE-call's stored served retrieval for LoCoMo (the T-1 collection, C9), on T-1's
drawn 720 questions: the first 10 items in served order, `indexed_at` the session timestamp. The
arms render with the SHIPPED code on the PR branch this runs from:

* ``R``  `recall.evidence.render_evidence_prompt`, the plain default;
* ``R2`` R again, the noise floor;
* ``H``  `recall.evidence.render_dated_evidence_prompt`, with the question's date the conversation's
  last session time (LoCoMo's questions are asked after it) as ISO, the form `reason` sends.

Answers go through the library's provider, parser, normaliser and validator; a rejected envelope,
an unparseable one and `insufficient_evidence` are judged as the tool's refusal, which the judge
marks wrong on an answerable question, as `benchmarks.pipeline` does.

    python scripts/reasoning_reader_contract_locomo.py run --collected collected-S.json.gz \\
        --data locomo10.json --draw locomo-draw.json --out answers.jsonl
    python scripts/reasoning_reader_contract_locomo.py score --answers answers.jsonl --out score.json
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
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pipeline import JUDGE_SYSTEM_PROMPT  # noqa: E402
from recall.evidence import (  # noqa: E402
    EvidenceBundle,
    EvidenceItem,
    normalize_citations,
    parse_answer_envelope,
    render_dated_evidence_prompt,
    render_evidence_prompt,
    validate_answer,
)

SEED = 20260929
TOP_K = 10
ARMS = ("R", "R2", "H")
JUDGE_MODEL = "deepseek/deepseek-v4.1-flash"
REFUSAL = "I don't have enough information in my memory to answer that."
_SESSION = re.compile(r"^session_(\d+)_date_time$")


def _stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def last_session_time(conversation: Mapping[str, Any]) -> datetime:
    """The latest session's time, sorted by session NUMBER (as text, session_10 < session_2)."""
    numbered = sorted((int(m.group(1)), key) for key in conversation if (m := _SESSION.match(key)))
    raw = str(conversation[numbered[-1][1]])
    return datetime.strptime(raw, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def build_bundle(query: str, items: Sequence[Mapping[str, Any]]) -> EvidenceBundle:
    evidence = tuple(
        EvidenceItem(
            chunk_id=str(item["id"]), text=str(item["content"]), source=str(item.get("session_id") or ""),
            ordinal=rank, indexed_at=_stamp(str(item["created_at"])) if item.get("created_at") else None,
            valid_from=None, valid_until=None, cosine=0.0, confidence=0.0,
        )
        for rank, item in enumerate(items[:TOP_K])
        if isinstance(item.get("content"), str)
    )
    return EvidenceBundle(
        query=query, decision="answer", reason_code=None, decision_state="supported", calibrated=True,
        stale=False, embedding_profile="c9-stored", retrieval_profile="c9-stored",
        index_generation="t1-collected-S", items=evidence,
    )


def render(arm: str, bundle: EvidenceBundle, asked: datetime) -> tuple[str, str]:
    if arm in ("R", "R2"):
        return render_evidence_prompt(bundle)
    if arm == "H":
        return render_dated_evidence_prompt(bundle, asked.isoformat())
    raise ValueError(f"unknown arm {arm!r}")


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


def judge_user(question: str, gold: str, answer: str) -> str:
    """`benchmarks.pipeline.judge_correct`'s user message, verbatim."""
    return f"Question: {question}\nGold answer: {gold}\nPredicted answer: {answer}\nCorrect?"


def load(collected: Path, data: Path, draw: Path) -> list[dict[str, Any]]:
    rows = {r["id"]: r for r in json.load(gzip.open(collected))["rows"]}
    conversations = {c["sample_id"]: c for c in json.loads(data.read_text(encoding="utf-8"))}
    out = []
    for ident in json.loads(draw.read_text(encoding="utf-8"))["ids"]:
        sample, index = ident.rsplit(":", 1)
        conv = conversations[sample]
        qa = conv["qa"][int(index)]
        out.append({"id": ident, "category": int(qa["category"]), "question": str(qa["question"]),
                    "gold": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                    "asked": last_session_time(conv["conversation"]), "items": rows[ident]["items"]})
    return out


# ------------------------------------------------------------------------------------------- run


def _pricing(model: str) -> tuple[float, float]:
    import httpx

    for entry in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]:
        if entry["id"] == model:
            return float(entry["pricing"]["prompt"]), float(entry["pricing"]["completion"])
    raise SystemExit(f"no OpenRouter price for {model}")


def _judge(user: str) -> tuple[str, float]:
    import httpx

    body = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 10, "reasoning": {"effort": "none"},
              "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}],
              "usage": {"include": True}},
        timeout=120,
    ).json()
    return str(body["choices"][0]["message"]["content"] or "").strip(), float((body.get("usage") or {}).get("cost") or 0.0)


def run(args: argparse.Namespace) -> None:
    from recall.answer_provider import resolve_answer_provider

    provider = resolve_answer_provider(os.environ)
    if provider is None:
        raise SystemExit("RECALL_REASONING_ANSWER_ENABLED is not set")
    price_in, price_out = _pricing(os.environ["RECALL_REASONING_ANSWER_MODEL"])
    questions = load(args.collected, args.data, args.draw)
    if args.limit:
        questions = questions[: args.limit]
    done: set[tuple[str, str]] = set()
    spent = 0.0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["id"], record["arm"]))
            spent += float(record.get("cost") or 0.0)
    lock, stop, local = threading.Lock(), threading.Event(), threading.local()
    total = [spent]
    print(json.dumps({"questions": len(questions), "done": len(done), "spent_usd": round(spent, 4),
                      "reader": os.environ["RECALL_REASONING_ANSWER_MODEL"], "judge": JUDGE_MODEL}), flush=True)

    def work(index_q: tuple[int, dict[str, Any]]) -> None:
        index, q = index_q
        bundle = build_bundle(q["question"], q["items"])
        for arm in ARMS[index % len(ARMS):] + ARMS[: index % len(ARMS)]:
            if (q["id"], arm) in done or stop.is_set():
                continue
            local.error = None
            system, user = render(arm, bundle, q["asked"])
            try:
                raw = provider(system, user)
            except Exception as exc:  # BROAD-CATCH: a provider failure is the tool's refusal, recorded
                local.error, raw = f"{type(exc).__name__}: {exc}"[:200], None
            meta = provider.provider_metadata()
            answer_cost = (meta.prompt_tokens or 0) * price_in + (meta.completion_tokens or 0) * price_out
            text, outcome = hypothesis(raw, bundle)
            try:
                verdict, judge_cost = _judge(judge_user(q["question"], q["gold"], text))
            except Exception as exc:  # BROAD-CATCH: an unjudged answer is dropped and retried on resume
                print(json.dumps({"judge_error": q["id"], "arm": arm, "error": str(exc)[:200]}), flush=True)
                continue
            record = {"id": q["id"], "arm": arm, "category": q["category"], "outcome": outcome, "hypothesis": text,
                      "raw": raw, "provider_error": local.error, "verdict": verdict,
                      "correct": verdict.strip().casefold().startswith("yes"),
                      "usage": {"prompt_tokens": meta.prompt_tokens, "completion_tokens": meta.completion_tokens},
                      "cost": round(answer_cost + judge_cost, 6)}
            with lock:
                total[0] += record["cost"]
                if total[0] > args.max_usd:
                    stop.set()
                with args.out.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, enumerate(questions)))
    print(json.dumps({"spent_usd": round(total[0], 4), "stopped": stop.is_set()}), flush=True)


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
    categories: dict[str, int] = {}
    precision: dict[str, list[bool]] = defaultdict(list)
    cost = 0.0
    for line in args.answers.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] in correct[record["arm"]]:
            raise SystemExit(f"{record['arm']} answered {record['id']} twice")
        correct[record["arm"]][record["id"]] = bool(record["correct"])
        outcomes[record["arm"]][record["outcome"]] += 1
        categories[record["id"]] = int(record["category"])
        if record["outcome"] == "answered":
            precision[record["arm"]].append(bool(record["correct"]))
        cost += float(record.get("cost") or 0.0)
    every = sorted(categories)
    result: dict[str, Any] = {
        "answers": {arm: len(correct[arm]) for arm in ARMS},
        "accuracy": {arm: round(sum(correct[arm].values()) / len(correct[arm]), 4) for arm in ARMS if correct[arm]},
        "outcomes": {arm: dict(outcomes[arm]) for arm in ARMS},
        "correct_among_answered": {arm: round(sum(v) / len(v), 4) for arm, v in precision.items() if v},
        "cost_usd": round(cost, 4),
        "noise_R2_minus_R": paired(correct, "R2", "R", every),
        "H_minus_R": paired(correct, "H", "R", every),
        "per_category": {
            str(c): {"n": sum(categories[k] == c for k in every),
                     **{arm: round(sum(correct[arm].get(k, False) for k in every if categories[k] == c)
                                   / max(1, sum(categories[k] == c for k in every)), 4) for arm in ARMS},
                     "H_minus_R": paired(correct, "H", "R", [k for k in every if categories[k] == c])}
            for c in sorted(set(categories.values()))
        },
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--collected", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--draw", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-usd", type=float, default=4.0)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"run": run, "score": score}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
