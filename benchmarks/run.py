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

Exit status: 0 published, 2 a usage error from argparse, and 3 means the run finished but its
artifact was REFUSED publication (an unauditable cost claim, or a failed write) and quarantined
somewhere OUTSIDE the publishable glob, with the exact location printed on stderr: normally
``<out>/unpublished/<stamp>.json``, else a ``.json.txt`` sibling, and if no file could be written
at all, printed to stdout. 1 means it reached none of those, a genuine loss rather than a
salvageable one, and is also what the interpreter itself exits with on an uncaught exception. 3 is
distinct from 1 deliberately, so a wrapper can tell "salvageable, do not re-run" from "crashed".

The artifact also carries the run's full `config` block — retrieval budget, per-arm embedder,
`mem0ai` version, Mem0's vector-store settings, both system prompts verbatim, temperature and the
model string — plus the count of `qa` rows the loader could not score. Between them, a reader can
tell which settings produced the numbers and why n is what it is, from the file alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.artifact_contract import reject_unauditable_cost_claims
from benchmarks.usage import install_openai_meter
from benchmarks.usage import reset as reset_usage
from benchmarks.usage import snapshot as usage_snapshot
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
from recall.eval.arm_check import DEFAULT_SAMPLE
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


#: OpenRouter issues keys with this prefix. Checking it is free and turns a placeholder or a
#: mistyped key into an immediate failure instead of one discovered mid-run.
OPENROUTER_KEY_PREFIX = "sk-or-"

#: Exit status for "the run finished, its artifact was refused publication and quarantined".
#: Deliberately not 1: that is the interpreter's own uncaught-exception status, and a wrapper
#: must be able to tell a salvageable run from a crashed one before deciding to re-run and
#: re-spend. 2 is taken by argparse's usage error.
QUARANTINE_EXIT = 3


def validate_openrouter_key(key: str | None) -> str:
    """Return `key` if it can plausibly be an OpenRouter key, else raise ValueError explaining why.

    An is-it-set check is not enough. `export OPENROUTER_API_KEY=...` pasted verbatim leaves the
    three characters "..." in the environment: truthy, so it passes, and the run then dies on a 401
    at the first generator call — or, on the mem0 arm, only AFTER per-session LLM extraction during
    ingest has already been paid for. Validating the shape up front makes that failure free.

    The message never contains the key itself, only its length: this text goes to stderr and into
    logs, and a key with the wrong prefix is still a secret.
    """
    if key is None or not key.strip():
        raise ValueError("OPENROUTER_API_KEY is not set - the generator and judge both need it")
    if any(char.isspace() for char in key):
        raise ValueError(
            f"OPENROUTER_API_KEY contains whitespace (length {len(key)}) - a stray quote or "
            "newline usually means the shell captured more than the key itself"
        )
    if not key.startswith(OPENROUTER_KEY_PREFIX):
        raise ValueError(
            f"OPENROUTER_API_KEY does not look like an OpenRouter key: expected it to start with "
            f"{OPENROUTER_KEY_PREFIX!r}, got a {len(key)}-character value that does not. A "
            "placeholder such as '...' passes a mere is-it-set check and then fails only after "
            "the run has started - on the mem0 arm, after ingest has already spent money."
        )
    return key


def _build_system(
    arm: str,
    model: str,
    openrouter_key: str,
    k: int,
    run_id: str,
    embedder: str = "fastembed",
    reranker: str = "none",
) -> MemorySystem:
    """Construct the arm under test. `model` and `k` are shared so only the memory system differs.

    `run_id` is the run's unique stamp; the Mem0 arms carry it into their vector-store path and
    collection name so a rerun cannot reopen a previous run's accumulated store.
    """
    if arm == "recall":
        # RECALL_TEST_DSN first: it is the DSN the repo's own integration tests already point at,
        # so a machine set up to run them can run the benchmark with no extra configuration.
        dsn = os.environ.get("RECALL_TEST_DSN") or os.environ.get("RECALL_DSN") or DEFAULT_DSN
        return RecallSystem(dsn, embedder_name=embedder, k=k, reranker_name=reranker)
    if arm == "mem0":
        # `--embedder fastembed:MODEL` puts BOTH arms on the same local model: RE-call via
        # fastembed, Mem0 via its huggingface provider on the same MODEL. Bare `fastembed` (the
        # default) leaves Mem0 on its bge-small default.
        # `--embedder router:MODEL` puts BOTH arms on the same OpenRouter-served cloud embedder
        # (e.g. `router:openai/text-embedding-3-small`): RE-call via OpenAICompatEmbedder, Mem0 via
        # its `openai` provider pointed at OpenRouter with the SAME key the generator/judge use — so
        # Mem0's documented default embedder is measurable without an OpenAI account.
        if embedder.startswith("router:"):
            return Mem0System(
                openrouter_key,
                model,
                embedder="openai",
                openai_key=openrouter_key,
                openai_model=embedder[len("router:"):],
                openai_base_url="https://openrouter.ai/api/v1",
                k=k,
                run_id=run_id,
            )
        hf_model = "BAAI/bge-small-en-v1.5"
        if embedder.startswith("fastembed:"):
            hf_model = embedder[len("fastembed:"):]
        return Mem0System(openrouter_key, model, k=k, run_id=run_id, hf_model=hf_model)
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
    usage: dict[str, Any],
    provider_metadata: list[dict[str, object]] | None = None,
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
        "usage": usage,
        "provider_metadata": provider_metadata or [],
        "cost_claims": [],
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
    p.add_argument(
        "--embedder",
        default="fastembed",
        help=(
            "RE-call arm embedder (ignored by the mem0 arms). 'fastembed' = the default bge-small; "
            "'fastembed:BAAI/bge-large-en-v1.5' = a stronger 1024-dim local model, for the "
            "retrieval-quality ablation. Recorded in the results config."
        ),
    )
    p.add_argument(
        "--reranker",
        default="none",
        help=(
            "RE-call arm second-stage reranker (ignored by the mem0 arms). 'none' = default; "
            "'voyage:rerank-2.5' = Voyage cross-encoder over the fused candidate pool. Recorded in "
            "the results config."
        ),
    )
    p.add_argument(
        "--ablation-sample",
        type=_positive_int,
        default=DEFAULT_SAMPLE,
        help=(
            f"questions sampled by the inert-arm preflight (default {DEFAULT_SAMPLE}). Taken "
            "deterministically from the head of the question list, so the verdict is "
            "reproducible for a slice."
        ),
    )
    p.add_argument(
        "--allow-inert-arm",
        action="store_true",
        help=(
            "record an inert mechanism instead of refusing the run. The override AND every verdict "
            "are stamped into the artifact, so a run let through cannot read as clean afterwards."
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
    # Checked BEFORE ingestion: ingesting the Mem0 arm burns LLM calls, and discovering a bad
    # generator key afterwards would waste them. The shape is validated, not just the presence -
    # see validate_openrouter_key for the placeholder case that motivated it.
    # ASCII only: argparse writes this to stderr, and a Windows cp1252 console mangles anything
    # else.
    try:
        key = validate_openrouter_key(os.environ.get("OPENROUTER_API_KEY"))
    except ValueError as exc:
        p.error(str(exc))

    # Meter every LLM call in this process BEFORE any client is built: the harness generator/judge
    # AND the memory layer's own calls (Mem0's extraction) both route through the OpenAI SDK. The
    # memory layer's cost is then total - harness; for RE-call it is ~0, which is the whole point.
    reset_usage()
    install_openai_meter()
    llm = OpenRouterLLM(model=args.model, api_key=key)
    completer: Completer = llm.complete

    convs, questions, skipped = _load(args.data, args.conversations)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = _run_stamp(args.arm, args.model, len(convs), now or datetime.now(timezone.utc))
    # The stamp is built BEFORE the system so the Mem0 arms can name their vector-store path and
    # collection after this run, and never reopen an earlier run's accumulated store.
    system = _build_system(
        args.arm, args.model, key, args.k, stamp, embedder=args.embedder, reranker=args.reranker
    )

    text_by_id = _text_by_id(questions)
    gold_by_id = _gold_by_id(questions)
    partial_path = args.out / f"{stamp}.partial.jsonl"
    if skipped["total"]:
        print(f"skipped {skipped['total']} unscoreable qa rows: {skipped['by_reason']}", flush=True)

    outcomes: list[Outcome] = []
    ablation: list[dict[str, Any]] = []
    # Distinguishes "the preflight ran and found nothing configured" from "the preflight never
    # ran" — both stamp `verdicts: []` otherwise, and the second case would read as clean.
    ablation_ran = False
    for position, conv in enumerate(convs):
        sample_id = sample_id_of(conv)
        conv_questions = [q for q in questions if q["sample_id"] == sample_id]
        # Ingest THEN score, one conversation at a time: both adapters scope retrieval to the last
        # conversation ingested, so ingesting all of them up front would answer every question out
        # of the final conversation's memory.
        system.ingest(conv)
        if position == 0 and args.arm == "recall":
            # `_build_system` always returns a `RecallSystem` for `arm == "recall"` today, so this
            # is unreachable — but if that ever changed silently, falling through here would skip
            # the preflight with zero trace: the artifact would stamp `verdicts: []`, indistinguishable
            # from a preflight that ran and found nothing configured, and the run would read as
            # clean. Asserting keeps mypy able to see `ablation_preflight` (it is on the
            # `RecallSystem` adapter, not the shared `MemorySystem` protocol — it is retrieval-only
            # and has no meaning on the Mem0 arms) while making that failure mode loud instead of
            # silent.
            assert isinstance(system, RecallSystem), (
                f"arm='recall' but _build_system returned {type(system).__name__}; the ablation "
                f"preflight cannot run and would silently stamp verdicts: []"
            )
            # After index build, before the FIRST generator call: retrieval-only, so an inert arm
            # is caught before a single token is spent. BEAM best-config ran out of credits at
            # 5/60; a post-hoc check would have spent them first.
            ablation = system.ablation_preflight(
                [str(q["question"]) for q in conv_questions],
                sample=args.ablation_sample,
                metric_class="set",
                allow_inert=args.allow_inert_arm,
            )
            ablation_ran = True
            print(f"ablation preflight: {ablation}", flush=True)
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
    total_usage = usage_snapshot()
    harness_usage = llm.usage()
    # memory_layer = everything the process sent to an LLM, minus the harness's own generator+judge.
    # RE-call sends nothing from its retrieval path, so this is ~0; Mem0's is its extraction cost.
    memory_usage = {
        field: total_usage.get(field, 0) - harness_usage.get(field, 0)
        for field in ("calls", "prompt_tokens", "completion_tokens")
    }
    usage_block = {
        "total": total_usage,
        "harness_generator_judge": harness_usage,
        "memory_layer": memory_usage,
    }
    provider_metadata = [llm.provider_metadata().to_dict()]
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
        usage_block,
        provider_metadata,
    )
    payload["ablation_preflight"] = {
        "verdicts": ablation,
        "allow_inert_arm": bool(args.allow_inert_arm),
        "sample": args.ablation_sample,
        "ran": ablation_ran,
    }
    path = args.out / f"{stamp}.json"
    # Everything below runs AFTER the last billed call, so nothing here may throw the artifact
    # away. What is at stake is narrower than it looks: every scored answer is already in the
    # incremental sidecar, and `python -m benchmarks.salvage --merge-only` rebuilds a runnable
    # artifact from it without re-spending a cent. What a dropped artifact really costs is the
    # envelope the sidecar does NOT carry — the usage block, `provider_metadata`, the config and
    # the ablation preflight — plus a manual salvage step. Cheap to keep, so keep it.
    try:
        # Serialise BEFORE the contract runs, so the bytes are in hand whichever way it goes.
        # `aggregate` already sanitises its empty rate blocks to None, so this never emits the
        # bare `NaN` token that no non-Python JSON parser accepts.
        #
        # A payload json cannot encode loses to `TypeError` here, before the contract runs, and
        # that is the right winner: an unencodable payload cannot be quarantined either, so the
        # serializer's error is the actionable one rather than a contract message chained behind
        # it. It also means `json.dumps` now detects circular references first.
        encoded = json.dumps(payload, indent=2)
        try:
            reject_unauditable_cost_claims(payload)
        except ValueError as exc:
            preserved = _quarantine(
                args.out, stamp, payload, encoded, partial_path, reason=str(exc)
            )
            return QUARANTINE_EXIT if preserved else 1
        try:
            _write_atomic(path, encoded)
        except OSError as exc:
            # Likelier than the contract firing: a full disk, a >260-char path, an antivirus or
            # indexer lock. The bytes are already in hand, so losing them here is gratuitous.
            #
            # No cleanup needed: `_write_atomic` never touches `path` except by rename, so a
            # failure here cannot have left a truncated artifact in the publishable glob.
            preserved = _quarantine(
                args.out, stamp, payload, encoded, partial_path,
                reason=f"the published artifact could not be written to {path}: {exc}",
            )
            return QUARANTINE_EXIT if preserved else 1
        print(json.dumps(agg, indent=2))
        print(f"full results -> {path}")
        print(f"incremental  -> {partial_path}")
        return 0
    finally:
        # Release the arm's handles while the interpreter is still healthy. A `Mem0System` holds
        # exclusive Qdrant locks for as long as it lives (see `Mem0System.close`), including one
        # on a machine-global telemetry path that no `run_id` disambiguates. Process exit would
        # drop them anyway; the point is to do it here rather than during finalisation, where
        # qdrant-client's own finaliser dies with `ModuleNotFoundError: import of msvcrt halted`.
        #
        # SCOPE: this `finally` covers the write site only, not the whole run. An exception
        # during ingest or scoring still skips it and falls back to process exit. Widening it is
        # `with _build_system(...) as system:` around the body, not a second close bolted on
        # elsewhere. Duck-typed because `MemorySystem` is a three-member protocol and
        # `RecallSystem` has nothing to release: it holds a DSN string, not a connection.
        #
        # Guarded, and doubly so now that this runs on failure paths too. Teardown must never
        # change the outcome, whether that outcome is a written artifact or an in-flight
        # exception. Unguarded on success, a complete run would exit non-zero and a wrapper
        # reading that status would re-run it, re-spending credit on work that already
        # succeeded; unguarded on failure, a raise in here would MASK the exception that got us
        # here, which is a possibility that did not exist while this was success-path-only.
        close = getattr(system, "close", None)
        if callable(close):
            try:
                close()
            # `Exception`, not `BaseException`, and that asymmetry with the guarded report
            # below is deliberate: a Ctrl+C landing during `Mem0System.close` (releasing Qdrant
            # locks is not instantaneous) SHOULD still interrupt the process. It costs at most a
            # crash status on a run whose artifact is already safely on disk.
            except Exception as exc:  # noqa: BLE001 - a wedged handle is not a failed run
                # The REPORT is guarded as carefully as the call, for the two reasons
                # `Mem0System.close` documents: `warnings.warn` raises under `-W error`, and
                # `repr()` of an arbitrary exception can raise. The outer `except BaseException`
                # is belt and braces for being inside a `finally`: the helper is written never
                # to raise, and if that ever stops being true it must still not replace the
                # return value, nor mask an exception already on its way out.
                try:
                    _warn_teardown_failed(system, exc)
                except BaseException:  # noqa: BLE001 - nothing in teardown may escape a finally
                    pass


def _quarantine(
    out: Path,
    stamp: str,
    payload: dict[str, Any],
    encoded: str,
    partial_path: Path,
    *,
    reason: str,
) -> bool:
    """Preserve an artifact that must not be published. Returns whether it was preserved.

    A rescue that can itself throw the artifact away is not a rescue, so nothing in here raises:
    every filesystem call is guarded, every report goes through `_say`, and there is a last
    resort that needs no filesystem at all.

    WRITE FIRST, REPORT AFTER, and that order is the whole point. Writing touches a directory
    this process was given; reporting touches a stream it does not own and cannot vouch for. An
    earlier version announced the refusal before saving anything, so a closed stderr raised
    straight out of the rescue and destroyed the artifact with a perfectly writable output
    directory sitting there — and `main` then never returned QUARANTINE_EXIT, so a wrapper read
    the crash status and re-ran the arm.

    Neither destination may sit in the `results/*.json` glob that feeds `analyze`: not the
    preferred subdirectory, and not the `.json.txt` fallback. The payload is ALSO marked in band,
    because the subdirectory only protects the DOCUMENTED glob and a reader handed the file
    directly would otherwise find it byte identical to a publishable artifact. `analyze` refuses
    a payload carrying that mark, so it is a contract rather than a decoration. Republishing
    means fixing the cause and dropping the two marker keys.
    """

    marked = dict(payload, unpublished=True, unpublished_reason=reason)
    try:
        body = json.dumps(marked, indent=2)
    except (TypeError, ValueError):  # pragma: no cover - `encoded` already proved it serialises
        body = encoded

    notes = [f"artifact NOT published: {reason}"]
    written: Path | None = None
    for candidate in (
        out / "unpublished" / f"{stamp}.json",
        out / f"{stamp}.unpublished.json.txt",
    ):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(candidate, body)
        except OSError as exc:
            # `exist_ok=True` does NOT tolerate the path existing as a FILE, so the preferred
            # subdirectory is one `touch` away from being unusable. Fall back, do not give up.
            notes.append(f"  {candidate} could not be written: {exc}")
            continue
        written = candidate
        break

    preserved = written is not None
    if written is not None:
        notes.append(f"unpublished  -> {written}")
    notes.append(f"incremental  -> {partial_path}")
    notes.append(
        "the scored answers are safe either way: they are in that sidecar, and "
        "`python -m benchmarks.salvage --merge-only` rebuilds an artifact from it without "
        "re-spending. Fix the cause and republish rather than re-running the arm."
    )
    for note in notes:
        try:
            _say(note)
        except BaseException:  # noqa: BLE001 - the artifact is already on disk by this point
            # `_say` deliberately lets a KeyboardInterrupt through, because the last-resort
            # dump below is a long write to a pipe and an interrupt is the operator talking.
            # These notes print AFTER the artifact is safe, so an interrupt here
            # would throw away the verdict for a run that WAS preserved and hand a wrapper a
            # crash status — the harm the narrowing was meant to prevent, inverted. Per note, so
            # a single Ctrl+C costs one line and not the `unpublished -> ...` path with it.
            pass

    if written is None:
        # Nowhere on disk would take it. stdout might still be a file or a pipe.
        preserved = _say(body, stream=sys.stdout)
        _say(
            "the artifact reached no file; it was printed to stdout above"
            if preserved
            else "the artifact could not be preserved anywhere"
        )
    return preserved


#: Sentinel for "no stream argument", so `None` can keep its own meaning: THIS STREAM IS
#: UNUSABLE. Defaulting `stream=None` conflated the two, and the bug that produced was ugly —
#: `_say(body, stream=sys.stdout)` with a None stdout took the default branch, wrote the artifact
#: body to STDERR, reported "printed to stdout above", and counted it as preserved.
_NO_STREAM: Any = object()


def _write_atomic(path: Path, body: str) -> None:
    """Write via a temp file and `os.replace`, so a failed write can neither truncate the target
    nor destroy what was already there.

    `write_text` opens with mode "w", which has two distinct failure shapes and a naive cleanup
    gets one of them badly wrong. Fail AFTER open (ENOSPC mid-write) and the target is a
    half-written corpse that has to go. Fail AT open (an antivirus or indexer lock, EACCES, a
    read-only mount) and the file sitting there is INTACT and somebody else's, so unlinking it
    destroys a good artifact this function is not going to replace. Writing to a temp and
    renaming makes the distinction moot: the target is only ever touched by an atomic rename.
    """

    # PID in the name: `_run_stamp` resolves to the second, so two replicates of the same arm
    # launched in the same second would otherwise share one temp, and the loser's `os.replace`
    # would fail and quarantine a complete, good artifact as refused.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _say(message: str, stream: TextIO | None = _NO_STREAM) -> bool:
    """Print without raising on a broken stream. Returns whether the message went anywhere.

    `print(file=None)` writes to STDOUT rather than raising, and `sys.stderr` really is None
    under pythonw, so a bare `file=sys.stderr` would silently interleave diagnostics into the
    artifact body. Refusing a None stream is what keeps the last resort's return value honest.

    `Exception`, NOT `BaseException`. A broken pipe is an expected failure of reporting and is
    swallowed. A `KeyboardInterrupt` is the operator talking, and the last resort dumps the whole
    payload — every per-question record — to a possibly slow pipe, so an interrupt there is
    likely rather than exotic. Swallowing it would report "the artifact could not be preserved
    anywhere" and hand back the exit code that tells a wrapper to re-run and re-spend.
    """

    target = sys.stderr if stream is _NO_STREAM else stream
    if target is None:
        return False
    try:
        print(message, file=target, flush=True)
    except Exception:  # noqa: BLE001 - reporting a failure may not become one
        return False
    return True


def _warn_teardown_failed(subject: object, exc: BaseException) -> None:
    """Report a teardown failure without ever becoming one.

    Teardown of an already-written artifact must not change a run's verdict, so nothing in here
    may raise. But it must not be silent either: the whole cost of the incident this traces back
    to was diagnostic, and under `-W error` the warning IS the raise, so a bare `pass` would drop
    the breadcrumb precisely when someone has asked to be strict. stderr is the last resort.

    Every value that reaches the message is derived behind its own guard, including
    `type(x).__name__` (a metaclass can make that raise) — which is why `subject` is passed in
    whole rather than pre-formatted by the caller. By the time `message` is built both parts are
    plain strings, so the f-string itself cannot fail.

    `BaseException`, not `Exception`, in all three handlers. That does swallow a `KeyboardInterrupt`
    arriving in the microseconds this runs, and that is the intended trade: the results are already
    written and their paths already printed, so a Ctrl+C landing inside a diagnostic must not be
    what turns a finished run into a failed one.
    """
    try:
        name = type(subject).__name__
    except BaseException:  # noqa: BLE001 - see the docstring: nothing here may raise
        name = "<unnameable>"
    try:
        detail = repr(exc)
    except BaseException:  # noqa: BLE001 - an exception whose repr raises is still evidence
        try:
            detail = f"<unprintable {type(exc).__name__}>"
        except BaseException:  # noqa: BLE001 - a hostile metaclass, one level further down
            detail = "<unprintable>"
    message = f"benchmarks.run: closing {name} failed: {detail}"
    try:
        # stacklevel=2 points at the close site in `main`, which is where the pre-helper version
        # pointed. 3 would credit whoever called `main`, i.e. `raise SystemExit(main())`.
        warnings.warn(message, stacklevel=2)
    except BaseException:  # noqa: BLE001 - `-W error` turns the report into a raise
        # `sys.stderr` is None under pythonw, and `print(file=None)` writes to STDOUT rather than
        # raising — which would interleave this into the artifact paths and the aggregate JSON.
        stream = sys.stderr
        if stream is not None:
            try:
                print(message, file=stream)
            except BaseException:  # noqa: BLE001 - a closed stderr is not a failed benchmark run
                pass


if __name__ == "__main__":
    raise SystemExit(main())
