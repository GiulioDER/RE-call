"""Run one arm of the head-to-head benchmark over LOCOMO and dump a results artifact.

::

    python -m benchmarks.run --arm recall --conversations 1
    python -m benchmarks.run --arm mem0  --conversations 1 --model openai/gpt-4o-mini
    python -m benchmarks.run --arm recall --conversations 1 --k 10   # wider retrieval budget

One process runs ONE arm. The arms share the generator, the judge, the question list and the
scoring code (`benchmarks.pipeline`) — the only thing that differs between them is which
`MemorySystem` supplies the context — so the comparison is between memory systems rather than
between harnesses. Splitting the arms across processes also keeps a crash or a rate-limit in one
arm from destroying the other arm's already-paid-for results.

Every run costs money (generator + judge calls on OpenRouter), so the raw per-question dump is
written for every run and treated as the publishable artifact: it carries the retrieved context,
the generated answer and the verdict for each question, which is what lets a reader re-score the
run under different rules — or catch the harness lying — without paying for it again.

Two properties of the output exist because the money is real:

- results are written INCREMENTALLY, one JSONL line per scored question, appended after each
  conversation. A rate limit or a Ctrl-C on conversation 7 of 10 then costs the conversation in
  flight, not the six already paid for.
- the filename carries a UTC timestamp, so a second run never overwrites the first one's artifact.

The artifact also carries the run's full `config` block — retrieval budget, per-arm embedder,
`mem0ai` version, Mem0's vector-store settings, both system prompts verbatim, temperature and the
model string — plus the count of `qa` rows the loader could not score. Between them, a reader can
tell which settings produced the numbers and why n is what it is, from the file alone.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.pipeline import (
    GEN_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    Outcome,
    aggregate,
    run_question,
)
from benchmarks.systems import (
    DEFAULT_K,
    MemorySystem,
    Mem0System,
    RecallSystem,
    mem0ai_version,
    sample_id_of,
)
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


def _load(
    data_path: Path, limit: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load LOCOMO into (outer conversation items, flat question list, skipped-row report).

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

    Every skip is COUNTED, by reason, and the counts are returned as the third element. LOCOMO's
    published shape is 1,540 answerable + 446 adversarial; a run that reports any other n is
    either measuring a different dataset or dropping rows, and until the drops were counted there
    was no way for a reader — or for the author — to tell those two apart from the artifact. The
    report is printed and published, so a headline n that differs from the canonical one comes
    with its own explanation instead of an accusation.
    """
    conversations: list[dict[str, Any]] = json.loads(data_path.read_text(encoding="utf-8"))
    if limit is not None:
        conversations = conversations[:limit]

    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for conversation in conversations:
        sample_id = sample_id_of(conversation)
        for index, qa in enumerate(conversation.get("qa") or []):
            category = qa.get("category")
            if category != ADVERSARIAL_CATEGORY and category not in ANSWERABLE_CATEGORIES:
                _skip("category_not_scored")
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                _skip("blank_question")
                continue
            adversarial = category == ADVERSARIAL_CATEGORY
            raw_answer = qa.get("answer")
            # str(): a handful of LOCOMO gold answers are ints, and the judge prompt is text.
            answer = "" if raw_answer is None else str(raw_answer).strip()
            if not adversarial and not answer:
                _skip("no_gold_answer")
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
    return conversations, questions, {"total": sum(skipped.values()), "by_reason": skipped}


def _build_system(arm: str, model: str, openrouter_key: str, k: int, run_id: str) -> MemorySystem:
    """Construct the arm under test. `model` and `k` are shared so only the memory system differs.

    `run_id` is the run's unique stamp; the Mem0 arms carry it into their vector-store path and
    collection name so a rerun cannot reopen a previous run's accumulated store.
    """
    if arm == "recall":
        # RECALL_TEST_DSN first: it is the DSN the repo's own integration tests already point at,
        # so a machine set up to run them can run the benchmark with no extra configuration.
        dsn = os.environ.get("RECALL_TEST_DSN") or os.environ.get("RECALL_DSN") or DEFAULT_DSN
        return RecallSystem(dsn, k=k)
    if arm == "mem0":
        return Mem0System(openrouter_key, model, k=k, run_id=run_id)
    if arm == "mem0-default":
        # The ablation arm: Mem0 as shipped, on its documented OpenAI embedder rather than the
        # local one the fairness-controlled `mem0` arm uses.
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("--arm mem0-default needs OPENAI_API_KEY (its embedder is OpenAI)")
        return Mem0System(
            openrouter_key,
            model,
            embedder="openai",
            openai_key=openai_key,
            k=k,
            run_id=run_id,
        )
    raise ValueError(f"unknown arm {arm!r}")


def _text_by_id(questions: list[dict[str, Any]]) -> dict[str, str]:
    """Question text keyed by `question_id`, so an `Outcome` (which carries only the id) can be
    written out as a self-contained, re-scorable record."""
    return {str(q["question_id"]): str(q["question"]) for q in questions}


def _gold_by_id(questions: list[dict[str, Any]]) -> dict[str, str]:
    """Gold answer keyed by `question_id` (empty string for adversarials, which have none)."""
    return {str(q["question_id"]): str(q["answer"]) for q in questions}


def _outcome_record(
    outcome: Outcome, text_by_id: dict[str, str], gold_by_id: dict[str, str]
) -> dict[str, Any]:
    """One per-question record: the scored `Outcome` plus the question text and gold answer.

    The same shape is used for the incremental JSONL lines and for the `outcomes` array of the
    final artifact, so a run that died half-way is re-scored by exactly the code that reads a run
    that finished.

    `gold` is in the record because the file claims to be re-scorable WITHOUT the LOCOMO source,
    and re-scoring means re-running the judge — which needs the gold answer. Without it the claim
    held only for re-reading the verdicts already paid for, not for producing new ones, and a
    reader who wanted to grade the run under a different judge had to fetch and re-join
    `locomo10.json` by `question_id`. It is the empty string for an adversarial, which is the
    honest record: there is no gold answer to be correct about.
    """
    return {
        **asdict(outcome),
        "question": text_by_id.get(outcome.question_id, ""),
        "gold": gold_by_id.get(outcome.question_id, ""),
    }


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Append one conversation's records to the incremental JSONL sidecar and close the file.

    Append-and-close per conversation rather than one handle held open for the whole run: the
    point is that the bytes have reached the OS by the time the next conversation starts, so a
    crash, a rate limit or a Ctrl-C loses at most the conversation in flight. JSONL rather than
    rewriting a JSON array each time because appending cannot corrupt what is already there.
    """
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _run_stamp(arm: str, model: str, conversations: int, now: datetime) -> str:
    """The filename stem for one run's artifacts — unique per run because of the timestamp.

    Without it the stem is `{arm}_{model}_{N}conv`, so re-running the same arm/model/slice
    silently overwrites the previous run's results file: an artifact that cost real money and may
    already have been published and linked. `now` is passed in rather than read from the clock in
    here so a test can pin the name.
    """
    stamped = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{arm}_{model.replace('/', '-')}_{conversations}conv_{stamped}"


def _run_config(
    arm: str, model: str, k: int, llm: OpenRouterLLM, system: MemorySystem
) -> dict[str, Any]:
    """Everything a reader needs to identify and reproduce this run, in the artifact itself.

    The design promises "pinned versions, full configs, both prompts, seed" (§5.4), and the
    artifact was shipping the arm name and the model string alone. Every remaining number in the
    file is a function of the settings below — change `k`, the embedder, the generator prompt or
    the Mem0 release and the same code produces different results — so an artifact without them
    documents a run nobody can reproduce, including its author six months later.

    Per-arm configuration comes from the adapter's own `describe()` (duck-typed: a stub system in
    a test simply has none), which redacts API keys — this file is meant to be published.
    Temperature stands in for the seed: the OpenAI-compatible API exposes no seed for these
    models, and temperature 0.0 is the determinism knob that does exist.
    """
    describe = getattr(system, "describe", None)
    return {
        "k": k,
        "arm": arm,
        "model": model,
        "temperature": llm.temperature,
        "base_url": llm.base_url,
        "max_attempts": llm.max_attempts,
        "mem0ai_version": mem0ai_version(),
        "system": describe() if callable(describe) else {},
        "gen_system_prompt": GEN_SYSTEM_PROMPT,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
    }


def _results_payload(
    arm: str,
    model: str,
    conversations: list[dict[str, Any]],
    text_by_id: dict[str, str],
    gold_by_id: dict[str, str],
    outcomes: list[Outcome],
    aggregate_: dict[str, Any],
    config: dict[str, Any],
    skipped: dict[str, Any],
) -> dict[str, Any]:
    """The publishable artifact: run identity, config, the aggregate, and every per-question record.

    The question TEXT and the GOLD answer are joined back in here (an `Outcome` carries only the
    id), so the file can be read and re-scored — judge included — without also holding the LOCOMO
    source alongside it.
    """
    return {
        "arm": arm,
        "model": model,
        "config": config,
        "conversations": len(conversations),
        "questions": len(outcomes),
        "skipped_questions": skipped,
        "aggregate": aggregate_,
        "outcomes": [_outcome_record(o, text_by_id, gold_by_id) for o in outcomes],
    }


def _positive_int(raw: str) -> int:
    """argparse type for a count that must be at least 1.

    `--conversations 0` used to slice the dataset to nothing and produce a complete-looking
    artifact with n=0 in every cell; a negative value sliced from the END of the list, silently
    benchmarking a different subset than the one named. Both are rejected at the parser now.
    """
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def main(argv: list[str] | None = None, now: datetime | None = None) -> int:
    """Run one arm. `now` stamps the output filenames; it defaults to the wall clock in UTC.

    It is a parameter rather than a `datetime.now()` buried in the filename logic so a test can
    pin the artifact paths without freezing time process-wide.
    """
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Run one arm of the memory head-to-head benchmark over LOCOMO.",
    )
    p.add_argument("--arm", choices=["recall", "mem0", "mem0-default"], required=True)
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--data", type=Path, default=Path("locomo10.json"))
    p.add_argument("--conversations", type=_positive_int, default=1)
    p.add_argument(
        "--k",
        type=_positive_int,
        default=DEFAULT_K,
        help=(
            f"retrieval budget passed to BOTH arms (default {DEFAULT_K}). Recorded in the results "
            "artifact; the arms do not retrieve comparable volumes at the same k, so the artifact "
            "also reports each arm's retrieved-context size"
        ),
    )
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

    convs, questions, skipped = _load(args.data, args.conversations)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = _run_stamp(args.arm, args.model, len(convs), now or datetime.now(timezone.utc))
    # The stamp is built BEFORE the system so the Mem0 arms can name their vector-store path and
    # collection after this run, and never reopen an earlier run's accumulated store.
    system = _build_system(args.arm, args.model, key, args.k, stamp)

    text_by_id = _text_by_id(questions)
    gold_by_id = _gold_by_id(questions)
    partial_path = args.out / f"{stamp}.partial.jsonl"
    if skipped["total"]:
        print(f"skipped {skipped['total']} unscoreable qa rows: {skipped['by_reason']}", flush=True)

    outcomes: list[Outcome] = []
    for position, conv in enumerate(convs):
        sample_id = sample_id_of(conv)
        conv_questions = [q for q in questions if q["sample_id"] == sample_id]
        # Ingest THEN score, one conversation at a time: both adapters scope retrieval to the last
        # conversation ingested, so ingesting all of them up front would answer every question out
        # of the final conversation's memory.
        system.ingest(conv)
        conv_outcomes, _ = run_arm(system, completer, conv_questions)
        outcomes.extend(conv_outcomes)
        # Persist BEFORE the next conversation is touched: everything scored above has already
        # been paid for in generator and judge calls, and this is the write that keeps it.
        _append_records(
            partial_path, [_outcome_record(o, text_by_id, gold_by_id) for o in conv_outcomes]
        )
        print(
            f"  [{position + 1}/{len(convs)}] {sample_id}: {len(conv_outcomes)} questions scored",
            flush=True,
        )

    agg = aggregate(outcomes)
    payload = _results_payload(
        args.arm,
        args.model,
        convs,
        text_by_id,
        gold_by_id,
        outcomes,
        agg,
        _run_config(args.arm, args.model, args.k, llm, system),
        skipped,
    )
    path = args.out / f"{stamp}.json"
    # `aggregate` already sanitises its empty rate blocks to None, so this never emits the bare
    # `NaN` token that no non-Python JSON parser accepts.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    print(f"full results -> {path}")
    print(f"incremental  -> {partial_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
