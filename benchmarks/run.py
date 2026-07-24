"""Run one arm of the head-to-head benchmark over LOCOMO and dump a results artifact.

::

    python -m benchmarks.run --arm recall --conversations 1
    python -m benchmarks.run --arm mem0  --conversations 1 --model openai/gpt-4o-mini

One process runs ONE arm. The arms share the generator, the judge, the question list and the
scoring code (`benchmarks.pipeline`) — the only thing that differs between them is which
`MemorySystem` supplies the context — so the comparison is between memory systems rather than
between harnesses. Splitting the arms across processes also keeps a crash or a rate-limit in one
arm from destroying the other arm's already-paid-for results.

Every run costs money (generator + judge calls on OpenRouter), so the raw per-question dump is
written for every run and treated as the publishable artifact: it carries the retrieved context,
the generated answer and the verdict for each question, which is what lets a reader re-score the
run under different rules — or catch the harness lying — without paying for it again.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.pipeline import Outcome, aggregate, run_question
from benchmarks.systems import MemorySystem, Mem0System, RecallSystem
from recall.eval.locomo import (
    ADVERSARIAL_CATEGORY,
    ANSWERABLE_CATEGORIES,
    CATEGORY_NAMES,
    DEFAULT_DSN,
)


def run_arm(
    system: MemorySystem, completer: Completer, questions: list[dict[str, Any]]
) -> tuple[list[Outcome], dict[str, Any]]:
    """Score `questions` against whatever `system` currently has ingested.

    Deliberately does NOT ingest: `MemorySystem.ingest` is per-conversation and stateful (both
    adapters point their tenant/user at the LAST conversation ingested), so the caller ingests one
    conversation and then calls this with that conversation's questions only. `main` pools the
    per-conversation outcomes and re-aggregates over the pool rather than averaging the per-call
    aggregates, because conversations carry different question counts and a mean of rates would
    weight a 150-question conversation the same as a 250-question one.
    """
    outcomes = [run_question(system.retrieve, completer, q) for q in questions]
    return outcomes, aggregate(outcomes)


def _sample_id(conversation: dict[str, Any], position: int) -> str:
    """Stable identity for one LOCOMO item, mirroring `recall.eval.locomo.run`'s fallback."""
    raw = conversation.get("sample_id")
    return str(raw) if raw else f"conv{position}"


def _load(
    data_path: Path, limit: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load LOCOMO into (outer conversation items, flat question list).

    The conversations are returned as the OUTER items — `sample_id` plus the nested `conversation`
    object — because that is what `MemorySystem.ingest` takes on both adapters. Handing over the
    inner object instead finds zero `session_` keys, indexes nothing, and yields a perfect,
    entirely false abstention score.

    Each question is translated into the shape `benchmarks.pipeline.run_question` consumes:

    - `adversarial` is derived from the category, not from the presence of an `answer` key. Two of
      LOCOMO's category-5 questions do carry an `answer` alongside their `adversarial_answer`, so
      keying off the field would misclassify them as answerable and score a refusal as a failure.
    - `answer` is the empty string for adversarials. `adversarial_answer` is the plausible-looking
      distractor the dataset supplies, NOT a gold answer — feeding it to the judge would grade a
      system for reproducing the trap.
    - Answerable questions without a usable gold answer are dropped rather than counted as wrong;
      scoring a label gap would report a dataset defect as a system property (the same choice
      `run_conversation` makes for questions with no `evidence`).
    - `question_id` is `{sample_id}:{position-in-qa}`, so it stays stable when the rules above skip
      a neighbouring row, and stays joinable back to the source file.
    """
    conversations: list[dict[str, Any]] = json.loads(data_path.read_text(encoding="utf-8"))
    if limit is not None:
        conversations = conversations[:limit]

    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, conversation in enumerate(conversations):
        sample_id = _sample_id(conversation, position)
        for index, qa in enumerate(conversation.get("qa") or []):
            category = qa.get("category")
            if category != ADVERSARIAL_CATEGORY and category not in ANSWERABLE_CATEGORIES:
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            adversarial = category == ADVERSARIAL_CATEGORY
            raw_answer = qa.get("answer")
            # str(): a handful of LOCOMO gold answers are ints, and the judge prompt is text.
            answer = "" if raw_answer is None else str(raw_answer).strip()
            if not adversarial and not answer:
                continue
            question_id = f"{sample_id}:{index}"
            if question_id in seen:
                raise ValueError(
                    f"duplicate question_id {question_id!r} — the results artifact is keyed by it"
                )
            seen.add(question_id)
            questions.append(
                {
                    "question_id": question_id,
                    "sample_id": sample_id,
                    "category": CATEGORY_NAMES.get(category, f"cat{category}"),
                    "adversarial": adversarial,
                    "question": question,
                    "answer": "" if adversarial else answer,
                }
            )
    return conversations, questions


def _build_system(arm: str, model: str, openrouter_key: str) -> MemorySystem:
    """Construct the arm under test. `model` is shared across arms so only memory differs."""
    if arm == "recall":
        # RECALL_TEST_DSN first: it is the DSN the repo's own integration tests already point at,
        # so a machine set up to run them can run the benchmark with no extra configuration.
        dsn = os.environ.get("RECALL_TEST_DSN") or os.environ.get("RECALL_DSN") or DEFAULT_DSN
        return RecallSystem(dsn)
    if arm == "mem0":
        return Mem0System(openrouter_key, model)
    if arm == "mem0-default":
        # The ablation arm: Mem0 as shipped, on its documented OpenAI embedder rather than the
        # local one the fairness-controlled `mem0` arm uses.
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("--arm mem0-default needs OPENAI_API_KEY (its embedder is OpenAI)")
        return Mem0System(openrouter_key, model, embedder="openai", openai_key=openai_key)
    raise ValueError(f"unknown arm {arm!r}")


def _results_payload(
    arm: str,
    model: str,
    conversations: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    outcomes: list[Outcome],
    aggregate_: dict[str, Any],
) -> dict[str, Any]:
    """The publishable artifact: run identity, the aggregate, and every per-question record.

    The question TEXT is joined back in here (an `Outcome` carries only the id), so the file can be
    read and re-scored without also holding the LOCOMO source alongside it.
    """
    text_by_id = {q["question_id"]: q["question"] for q in questions}
    return {
        "arm": arm,
        "model": model,
        "conversations": len(conversations),
        "questions": len(outcomes),
        "aggregate": aggregate_,
        "outcomes": [
            {**asdict(o), "question": text_by_id.get(o.question_id, "")} for o in outcomes
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Run one arm of the memory head-to-head benchmark over LOCOMO.",
    )
    p.add_argument("--arm", choices=["recall", "mem0", "mem0-default"], required=True)
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--data", type=Path, default=Path("locomo10.json"))
    p.add_argument("--conversations", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args(argv)

    if not args.data.exists():
        p.error(
            f"{args.data} not found. Fetch it with:\n"
            "  curl -sLO https://raw.githubusercontent.com/snap-research/locomo/main/data/"
            "locomo10.json"
        )
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        # Checked BEFORE ingestion: ingesting the Mem0 arm burns LLM calls, and discovering the
        # missing generator key afterwards would waste them.
        # ASCII only: argparse writes this to stderr, and a Windows cp1252 console mangles
        # anything else.
        p.error("OPENROUTER_API_KEY is not set - the generator and judge both need it")

    llm = OpenRouterLLM(model=args.model, api_key=key)
    completer: Completer = llm.complete

    convs, questions = _load(args.data, args.conversations)
    system = _build_system(args.arm, args.model, key)

    outcomes: list[Outcome] = []
    for position, conv in enumerate(convs):
        sample_id = _sample_id(conv, position)
        conv_questions = [q for q in questions if q["sample_id"] == sample_id]
        # Ingest THEN score, one conversation at a time: both adapters scope retrieval to the last
        # conversation ingested, so ingesting all of them up front would answer every question out
        # of the final conversation's memory.
        system.ingest(conv)
        conv_outcomes, _ = run_arm(system, completer, conv_questions)
        outcomes.extend(conv_outcomes)
        print(
            f"  [{position + 1}/{len(convs)}] {sample_id}: {len(conv_outcomes)} questions scored",
            flush=True,
        )

    agg = aggregate(outcomes)
    payload = _results_payload(args.arm, args.model, convs, questions, outcomes, agg)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = f"{args.arm}_{args.model.replace('/', '-')}_{len(convs)}conv"
    path = args.out / f"{stamp}.json"
    # `aggregate` already sanitises its empty rate blocks to None, so this never emits the bare
    # `NaN` token that no non-Python JSON parser accepts.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    print(f"full results -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
