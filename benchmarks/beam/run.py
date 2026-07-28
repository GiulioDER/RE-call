"""Run the BEAM arm: RE-call retrieval through Mem0's own answerer, judge and rubric.

::

    # $0 — retrieval only, no LLM call is made. Run this first, always.
    python -m benchmarks.beam.run --dry-run --conversations 0

    # paid: RE-call over the full 1M bucket (35 conversations, 700 questions)
    python -m benchmarks.beam.run --data <beam_1M.parquet> --model openai/gpt-5

    # paid: re-score Mem0's PUBLISHED answers with this same judge instance
    python -m benchmarks.beam.run --rejudge-mem0 <beam_1m_results.json> --model openai/gpt-5

Design, and why each part is the way it is
------------------------------------------
**One judge, both arms.** Mem0's published artifact stores its generated answer per question id.
Scoring that stored answer with the SAME judge instance that scores RE-call's answer removes the
judge from the comparison entirely. Comparing our judge's verdict on our answer against their
judge's verdict on their answer would fold judge drift into the headline and call it retrieval.

**Answerer and judge prompts are not ours.** They are vendored verbatim (`prompts.py`,
`UPSTREAM.md`). Do not "improve" them here.

**Abstention is measured in two directions.** BEAM's `abstention` category rewards withholding.
Every other category punishes it. Reporting only the first is how a trust layer gets to look good
by refusing to answer: the run therefore reports, per system, the abstention rate on abstention
questions AND the FALSE-abstain rate on the other nine categories, and neither number is derived
from the other.

**Results are written incrementally, and a restart resumes.** One JSONL line per scored question,
flushed as it lands; `--resume <sidecars…>` reloads those lines and skips their question ids. A rate
limit on question 600 of 700 costs the question in flight, not the 599 already paid for — and the
retry costs nothing for the 599 either.

**Questions are scored concurrently.** Sequentially, one question is ~3.5 judge calls at reasoning
latency: measured at ~2 minutes per question, i.e. ~24 hours for 700. The questions are independent
and the judge is stateless, so they run on a thread pool. Only the sidecar write and the usage
counters are shared, and both are lock-guarded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import TextIO
from typing import Any

from benchmarks.beam.dataset import Question, count_conversations, iter_conversations
from benchmarks.beam.prompts import (
    BEAM_JUDGE_SYSTEM_PROMPT,
    get_beam_answer_generation_prompt,
    get_beam_nugget_judge_prompt,
)
from benchmarks.beam.systems import BEAM_TABLE, DEFAULT_TOP_K, BeamRecallSystem
from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.usage import install_openai_meter
from benchmarks.usage import reset as reset_usage
from benchmarks.usage import snapshot as usage_snapshot
from recall.eval.locomo import DEFAULT_DSN

#: The exact string the vendored answerer prompt instructs the model to emit when the retrieved
#: memories do not support an answer (rule 4). Detecting it is how an abstention is counted; it is
#: matched loosely (case- and punctuation-insensitive, prefix) because models add trailing prose.
REFUSAL = "i don't have enough information to answer this question"

#: A nugget score is one of three values. Upstream clamps the judge's raw number to this ladder,
#: and the published cells are means over it — so the same clamp is applied here rather than a
#: continuous score that would be a different metric wearing the same name.
def _clamp(raw: float) -> float:
    if raw >= 0.75:
        return 1.0
    if raw >= 0.25:
        return 0.5
    return 0.0


def _looks_like_refusal(answer: str) -> bool:
    """True when the answer is the harness's own refusal string.

    Normalises apostrophes and whitespace before matching: a model that emits a typographic
    apostrophe is refusing just as much as one that emits an ASCII one, and counting the first as
    an answer would understate abstention for exactly the systems that abstain most.
    """
    normalised = re.sub(r"\s+", " ", answer.replace("’", "'")).strip().lower()
    return normalised.startswith(REFUSAL)


def _parse_judge(raw: str) -> tuple[float, str]:
    """Pull ``{"score", "reason"}`` out of the judge's reply.

    A judge reply that cannot be parsed is NOT silently scored 0.0 — that would convert a harness
    failure into evidence against whichever system happened to be judged. It returns a NaN score
    and the raw text, and the caller counts it as an error that appears in the artifact.
    """
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return _clamp(float(obj.get("score", 0.0))), str(obj.get("reason", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return float("nan"), text[:300]


def judge_answer(
    question: Question, answer: str, judge: Completer
) -> tuple[float, list[dict[str, Any]], int]:
    """Score one answer against every rubric nugget. Returns (mean score, per-nugget, errors).

    Upstream judges each nugget in its own call and averages, so this does too: batching the
    rubric into a single call is cheaper and measurably more lenient, which would silently move
    every cell relative to the published ones.
    """
    scores: list[dict[str, Any]] = []
    errors = 0
    for nugget in question.rubric:
        raw = judge(BEAM_JUDGE_SYSTEM_PROMPT, get_beam_nugget_judge_prompt(question.question, nugget, answer))
        score, reason = _parse_judge(raw)
        if score != score:  # NaN
            errors += 1
        scores.append({"nugget": nugget, "score": score, "reason": reason})
    usable = [s["score"] for s in scores if s["score"] == s["score"]]
    return (statistics.mean(usable) if usable else float("nan")), scores, errors


def score_question(
    question: Question,
    memories: list[dict[str, str]],
    answerer: Completer,
    judge: Completer,
    cutoff: int,
) -> dict[str, Any]:
    """Retrieve-free scoring of one question: generate an answer, then judge it."""
    answer = answerer("", get_beam_answer_generation_prompt(question.question, memories, top_k=cutoff))
    if "ANSWER:" in answer:
        answer = answer.rsplit("ANSWER:", 1)[-1].strip()
    mean, nuggets, errors = judge_answer(question, answer, judge)
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "question": question.question,
        "rubric": question.rubric,
        "memories_retrieved": len(memories),
        "memories_evaluated": min(len(memories), cutoff),
        "generated_answer": answer,
        "abstained": _looks_like_refusal(answer),
        "retrieval_empty": not memories,
        "score": mean,
        "judgment": ("PASS" if mean >= 0.5 else "FAIL") if mean == mean else "ERROR",
        "nugget_scores": nuggets,
        "judge_errors": errors,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """BEAM's published metrics, plus the two abstention rates our methodology requires.

    `avg_score` is the headline Mem0 publishes (their 64.1 at 1M is `avg_score` 0.6409, NOT the
    70.14 % pass rate); both are reported so neither can be mistaken for the other.
    """
    scored = [r for r in rows if r["score"] == r["score"]]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_type[row["question_type"]].append(row)

    abstention_rows = by_type.get("abstention", [])
    answerable = [r for r in scored if r["question_type"] != "abstention"]

    return {
        "overall": {
            "n": len(scored),
            "errors": len(rows) - len(scored),
            "avg_score": round(statistics.mean(r["score"] for r in scored), 4) if scored else None,
            "accuracy_pct": round(
                100 * sum(1 for r in scored if r["score"] >= 0.5) / len(scored), 2
            )
            if scored
            else None,
        },
        "abstention": {
            "n": len(abstention_rows),
            # Did the system withhold when withholding was correct?
            "abstain_rate": round(
                sum(1 for r in abstention_rows if r["abstained"]) / len(abstention_rows), 4
            )
            if abstention_rows
            else None,
            "avg_score": round(statistics.mean(r["score"] for r in abstention_rows), 4)
            if abstention_rows
            else None,
        },
        "false_abstain": {
            # Did it withhold when an answer was available? Reported over the OTHER nine
            # categories, never folded into the abstention rate above.
            "n": len(answerable),
            "rate": round(sum(1 for r in answerable if r["abstained"]) / len(answerable), 4)
            if answerable
            else None,
            "retrieval_empty_rate": round(
                sum(1 for r in answerable if r["retrieval_empty"]) / len(answerable), 4
            )
            if answerable
            else None,
        },
        "by_question_type": {
            qt: {
                "n": len(items),
                "avg_score": round(statistics.mean(i["score"] for i in items), 4),
                "accuracy_pct": round(
                    100 * sum(1 for i in items if i["score"] >= 0.5) / len(items), 2
                ),
            }
            for qt, items in sorted(by_type.items())
        },
    }


def _writer(path: Path) -> tuple[Callable[[dict[str, Any]], None], TextIO]:
    """Append-and-flush sidecar writer, safe to call from several worker threads.

    The lock guards the write, not the scoring: two threads finishing at once must not interleave
    half-lines into the file that is the crash-recovery record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    lock = threading.Lock()

    def write(row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with lock:
            handle.write(line + "\n")
            handle.flush()

    return write, handle


def _coverage(expected: set[str], scored: set[str]) -> dict[str, Any]:
    """Expected vs scored question ids, with the shortfall spelled out.

    `complete` is the field to assert on: a paired comparison against a run that silently dropped
    questions is not the comparison it claims to be, and `benchmarks.beam.pair` refuses to align
    mismatched sets for the same reason.
    """
    missing = sorted(expected - scored)
    return {
        "expected": len(expected),
        "scored": len(scored & expected),
        "missing": len(missing),
        "complete": not missing,
        # Bounded: enough to resume from and to see the shape of the loss, not a second copy of
        # the question set inside every artifact.
        "missing_ids": missing[:50],
        "missing_ids_truncated": max(0, len(missing) - 50),
    }


def _already_done(paths: list[Path] | None) -> tuple[list[dict[str, Any]], set[str]]:
    """Rows from previous runs' sidecars, plus the question ids they cover.

    Takes a LIST because a run that was interrupted and resumed leaves one sidecar per attempt,
    and the questions already paid for are spread across all of them — resuming from only the
    newest would re-pay for the earlier ones. Duplicate ids across sidecars keep the first
    occurrence; they are the same scored question either way.

    Resume is by question id rather than by position: the pool finishes out of order, so a
    positional resume would skip questions that were never scored and re-pay for ones that were.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths or []:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid in seen:
                continue
            seen.add(qid)
            rows.append(row)
    return rows, seen


class RunAborted(RuntimeError):
    """A condition that will fail every remaining question identically, so the run stops."""


#: Substrings identifying a failure that is about the ACCOUNT, not about the question. Retrying
#: these burns the task list producing nothing, and dropping them one-by-one produces a run that
#: exits 0 with a clean-looking summary over whatever happened to succeed first.
_TERMINAL_MARKERS = (
    "402",           # OpenRouter: out of credits / max_tokens unaffordable
    "401",           # bad or revoked key
    "insufficient_quota",
    "requires more credits",
)


def _is_terminal(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker.lower() in text for marker in _TERMINAL_MARKERS)


def _run_pool(tasks: list[Any], work: Any, workers: int) -> list[Any]:
    """Map `work` over `tasks` on a thread pool, returning the successful results.

    A task that raises is logged and dropped rather than killing the run: one question failing
    after retries must not discard the hundreds already paid for. The dropped ids stay absent from
    the artifact, and the run reports `coverage`, so a short run is visible rather than silent.

    EXCEPT when the failure is terminal. Running out of credits mid-run is not a flaky question —
    it fails every remaining one identically. Dropping those one at a time is how this harness
    turned an empty OpenRouter balance into a tidy `n: 599` summary and exit 0: 101 questions
    "failed", the artifact looked publishable, and nothing said the account was empty. That is
    infrastructure wearing a verdict's clothes. A terminal error now aborts the run and says why,
    while the sidecar keeps every question already scored so `--resume` picks up where it stopped.
    """
    results: list[Any] = []
    if workers <= 1:
        for task in tasks:
            try:
                results.append(work(task))
            except Exception as exc:  # noqa: BLE001 - classified, then re-raised or reported
                if _is_terminal(exc):
                    raise RunAborted(str(exc)) from exc
                print(f"  ! question failed, dropped: {exc}")
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        aborted: BaseException | None = None
        for future in futures:
            if aborted is not None:
                future.cancel()
                continue
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - classified, then re-raised or reported
                if _is_terminal(exc):
                    aborted = exc
                    continue
                print(f"  ! question failed, dropped: {exc}")
        if aborted is not None:
            raise RunAborted(str(aborted)) from aborted
    return results


def _load_published(path: Path) -> dict[str, dict[str, Any]]:
    """Mem0's published per-question evaluations, keyed by question id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("evaluations", []):
        cutoffs = row.get("cutoff_results", {})
        label = next(iter(cutoffs), None)
        if label is None:
            continue
        out[row["question_id"]] = {
            "question": row.get("question", ""),
            "question_type": row.get("question_type", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "rubric": row.get("rubric", []),
            "published_score": cutoffs[label].get("score"),
            "generated_answer": cutoffs[label].get("generated_answer", ""),
            "memories_evaluated": cutoffs[label].get("memories_evaluated"),
            "cutoff_label": label,
        }
    return out


def main() -> None:
    try:
        _main()
    except RunAborted as exc:
        raise SystemExit(
            "RUN ABORTED — this fails every remaining question, so nothing further was "
            f"attempted.\n  {exc}\n"
            "Every question scored so far is in the .partial.jsonl sidecar; pass it (and any "
            "earlier ones) to --resume once the account is topped up, and only the unscored "
            "questions will be paid for."
        ) from exc


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="BEAM split parquet (or converted JSON)")
    parser.add_argument("--chat-size", default="1M", choices=["100K", "500K", "1M", "10M"])
    parser.add_argument(
        "--conversations",
        default=None,
        help="Conversation indices, e.g. '0-9' or '0,3,7'. Default: all in the split.",
    )
    parser.add_argument("--model", default="openai/gpt-5", help="Answerer AND judge model")
    parser.add_argument("--judge-model", default=None, help="Override the judge model")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Retrieval budget")
    parser.add_argument("--cutoff", type=int, default=DEFAULT_TOP_K, help="Memories shown to the answerer")
    parser.add_argument("--embedder", default="fastembed")
    parser.add_argument("--reranker", default="none")
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN", DEFAULT_DSN))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index and retrieve only; make NO LLM call and spend nothing. Prints retrieval "
        "coverage per question so the arm can be validated before any money is committed.",
    )
    parser.add_argument(
        "--rejudge-mem0",
        type=Path,
        default=None,
        help="Score Mem0's PUBLISHED answers (their beam_*_results.json) with this judge "
        "instance instead of running RE-call. Produces the paired Mem0 arm.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Questions scored concurrently. 1 = sequential (~2 min/question, ~24 h for 700).",
    )
    parser.add_argument(
        "--table",
        default=BEAM_TABLE,
        help="Chunk table to index into. Isolate a side experiment here: the arms clear and "
        "re-index their tenants as they walk the corpus, so two runs sharing a table can delete "
        "each other's chunks mid-question.",
    )
    parser.add_argument(
        "--question-types",
        default=None,
        help="Comma-separated BEAM categories to score (default: all ten). Exists so a defect fix "
        "can be VALIDATED on the categories it should move, for a few dollars, before committing "
        "the cost of a full re-run — and so the validation subset is recorded in the artifact "
        "rather than implied by a smaller n.",
    )
    parser.add_argument(
        "--entailment",
        type=int,
        default=0,
        metavar="N",
        help="Enable RE-call's near-miss entailment guard over the top N trusted hits (0 = off, "
        "the shipped default). The guard is a local QNLI cross-encoder, so it costs CPU and no "
        "API money; N caps that CPU. Judging is what a cosine threshold cannot do on BEAM, whose "
        "unanswerable questions score HIGHER than its answerable ones.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        nargs="*",
        default=None,
        help="One or more previous .partial.jsonl sidecars. Their question ids are skipped and "
        "their rows are carried into this run's artifact, so a restart never re-pays for scored "
        "questions. Pass every sidecar from every attempt — an interrupted run leaves one each.",
    )
    args = parser.parse_args()

    indices: list[int] | None = None
    if args.conversations:
        indices = []
        for part in args.conversations.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                indices.extend(range(int(lo), int(hi) + 1))
            else:
                indices.append(int(part))
        indices = sorted(set(indices))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arm = "mem0-rejudged" if args.rejudge_mem0 else "recall"
    out_base = args.out_dir / f"beam{args.chat_size}_{arm}_{stamp}"

    # ---- The Mem0 arm: re-score published answers, no retrieval, no dataset needed. ----
    if args.rejudge_mem0:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set")
        install_openai_meter()
        reset_usage()
        judge_llm = OpenRouterLLM(model=args.judge_model or args.model, api_key=key)
        published = _load_published(args.rejudge_mem0)
        if indices is not None:
            published = {
                qid: row
                for qid, row in published.items()
                if int(qid.split("_")[1]) in indices
            }
        rows, done = _already_done(args.resume)
        pending_published = [(qid, row) for qid, row in published.items() if qid not in done]
        print(
            f"mem0 re-judge: {len(published)} questions, {len(done)} already scored, "
            f"{len(pending_published)} to run on {args.workers} workers"
        )
        write, handle = _writer(out_base.with_suffix(".partial.jsonl"))

        def _score_published(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
            qid, row = item
            question = Question(
                question_id=qid,
                chat_size=args.chat_size,
                conversation_idx=int(qid.split("_")[1]),
                question_type=row["question_type"],
                question=row["question"],
                rubric=row["rubric"],
                difficulty=row["difficulty"],
            )
            mean, nuggets, errors = judge_answer(
                question, row["generated_answer"], judge_llm.complete
            )
            scored = {
                "question_id": qid,
                "question_type": question.question_type,
                "difficulty": question.difficulty,
                "question": question.question,
                "rubric": question.rubric,
                "memories_retrieved": row["memories_evaluated"],
                "memories_evaluated": row["memories_evaluated"],
                "generated_answer": row["generated_answer"],
                "abstained": _looks_like_refusal(row["generated_answer"]),
                "retrieval_empty": (row["memories_evaluated"] or 0) == 0,
                "score": mean,
                "published_score": row["published_score"],
                "judgment": ("PASS" if mean >= 0.5 else "FAIL") if mean == mean else "ERROR",
                "nugget_scores": nuggets,
                "judge_errors": errors,
            }
            write(scored)
            return scored

        try:
            rows.extend(_run_pool(pending_published, _score_published, args.workers))
        finally:
            handle.close()
        rows.sort(key=lambda r: r["question_id"])
        published_scores = [r["published_score"] for r in rows if r["published_score"] is not None]
        summary = {
            "arm": "mem0-platform-rejudged",
            "source_artifact": str(args.rejudge_mem0),
            "judge_model": args.judge_model or args.model,
            "chat_size": args.chat_size,
            "metrics": aggregate(rows),
            "published_avg_score": round(statistics.mean(published_scores), 4)
            if published_scores
            else None,
            "usage": {"total": usage_snapshot(), "harness_generator_judge": judge_llm.usage()},
            "n": len(rows),
            # Coverage is stated, not left to be inferred from `n`. A sustained provider outage
            # exhausts the retry budget on whatever is in flight and those questions are dropped
            # so the rest of the run survives — which is right, but it means a run can finish
            # cleanly, exit 0, and be short. It happened: 101 of 700 were lost to one outage
            # window and `n: 599` was the only trace. An explicit missing-id list makes a short
            # run impossible to mistake for a complete one, and feeds straight back into --resume.
            "coverage": _coverage(set(published), {str(r["question_id"]) for r in rows}),
        }
        out_base.with_suffix(".json").write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=1))
        return

    # ---- The RE-call arm. ----
    if not args.data:
        raise SystemExit("--data is required (path to the BEAM split parquet or converted JSON)")

    # Streamed, not materialised: converting all 35 conversations of the 1M bucket up front
    # cost >3 GB and 20+ minutes before the first turn was indexed. One at a time, the first
    # conversation is ready in ~3 s.
    n_conversations = count_conversations(args.data, indices)
    conversations = iter_conversations(args.data, args.chat_size, indices)
    system = BeamRecallSystem(
        args.dsn,
        embedder_name=args.embedder,
        k=args.k,
        table=args.table,
        entailment_top_n=args.entailment,
        reranker_name=args.reranker
    )

    if args.dry_run:
        report: list[dict[str, Any]] = []
        for conv in conversations:
            turns = system.ingest(conv)
            for question in conv.questions:
                memories = system.retrieve(question.question)
                report.append(
                    {
                        "question_id": question.question_id,
                        "question_type": question.question_type,
                        "turns_indexed": turns,
                        "memories": len(memories),
                        "abstained": not memories,
                        "rubric_nuggets": len(question.rubric),
                        "first_memory": memories[0]["memory"][:120] if memories else "",
                        "first_memory_date": memories[0]["created_at"] if memories else "",
                    }
                )
        empty = sum(1 for r in report if r["abstained"])
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "cost": 0.0,
                    "conversations": n_conversations,
                    "questions": len(report),
                    "questions_with_zero_memories": empty,
                    "mean_memories": round(
                        statistics.mean(r["memories"] for r in report), 2
                    )
                    if report
                    else 0,
                    "nuggets_missing": sum(1 for r in report if r["rubric_nuggets"] == 0),
                    "system": system.describe(),
                    "sample": report[:3],
                },
                indent=1,
            )
        )
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    install_openai_meter()
    reset_usage()
    answerer = OpenRouterLLM(model=args.model, api_key=key)
    judge_llm = OpenRouterLLM(model=args.judge_model or args.model, api_key=key)

    rows, done = _already_done(args.resume)
    print(
        f"RE-call arm: {n_conversations} conversations, {len(done)} questions already scored, "
        f"{args.workers} workers"
    )
    write, handle = _writer(out_base.with_suffix(".partial.jsonl"))

    def _score(question: Question) -> dict[str, Any]:
        # Retrieval happens INSIDE the worker: it opens its own store and connection, so the
        # threads share no cursor. Ingest stays on the main thread, one conversation at a time —
        # the tenant is per-conversation state on the system object, and indexing two
        # conversations at once would let one's turns answer the other's questions.
        memories = system.retrieve(question.question)
        scored = score_question(
            question, memories, answerer.complete, judge_llm.complete, args.cutoff
        )
        scored["conversation_idx"] = question.conversation_idx
        write(scored)
        return scored

    try:
        wanted_types = (
            {s.strip() for s in args.question_types.split(",")} if args.question_types else None
        )
        for conv in conversations:
            pending = [q for q in conv.questions if q.question_id not in done]
            if wanted_types is not None:
                pending = [q for q in pending if q.question_type in wanted_types]
            if not pending:
                print(f"  conversation {conv.index}: all {len(conv.questions)} scored, skipping")
                continue
            n_turns = system.ingest(conv)
            print(f"  conversation {conv.index}: {n_turns} turns indexed, {len(pending)} questions")
            rows.extend(_run_pool(pending, _score, args.workers))
    finally:
        handle.close()
    rows.sort(key=lambda r: r["question_id"])

    harness = {
        k: answerer.usage()[k] + judge_llm.usage()[k] for k in ("calls", "prompt_tokens", "completion_tokens")
    }
    total = usage_snapshot()
    summary = {
        "arm": "recall",
        "question_types": args.question_types or "all",
        "chat_size": args.chat_size,
        "model": args.model,
        "judge_model": args.judge_model or args.model,
        "k": args.k,
        "cutoff": args.cutoff,
        "system": system.describe(),
        "metrics": aggregate(rows),
        "usage": {
            "total": total,
            "harness_generator_judge": harness,
            # RE-call runs no LLM in its retrieval path, so this should be ~0. Publishing it
            # measured rather than asserted is the point.
            "memory_layer": {
                k: total[k] - harness[k] for k in ("calls", "prompt_tokens", "completion_tokens")
            },
        },
        "n": len(rows),
    }
    out_base.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
