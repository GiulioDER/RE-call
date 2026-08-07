"""Run the BEAM arm: RE-call retrieval through Mem0's own answerer, judge and rubric.

::

    # $0 — retrieval only, no LLM call is made. Run this first, always.
    # --data is required here too: a dry run still indexes and retrieves, it just never pays.
    python -m benchmarks.beam.run --data <beam_1M.parquet> --dry-run --conversations 0

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
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import TextIO
from typing import Any

from benchmarks.beam.dataset import Question, count_conversations, iter_conversations, parse_conversation_indices
from benchmarks.beam.prompts import (
    BEAM_JUDGE_SYSTEM_PROMPT,
    get_beam_answer_generation_prompt,
    get_beam_nugget_judge_prompt,
)
from benchmarks.beam.systems import BEAM_TABLE, DEFAULT_TOP_K, BeamRecallSystem
from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.run import _positive_int
from benchmarks.usage import install_openai_meter
from benchmarks.usage import reset as reset_usage
from benchmarks.usage import snapshot as usage_snapshot
from recall.eval.arm_check import DEFAULT_SAMPLE
from recall.eval.locomo import DEFAULT_DSN
from recall.eval.resume import read_ledger

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
    judge: Completer | None,
    cutoff: int,
) -> dict[str, Any]:
    """Retrieve-free scoring of one question: generate an answer, then judge it.

    `judge=None` generates WITHOUT judging. BEAM's judge reads only
    ``(question, nugget, answer)`` — never the retrieved context — so a row produced here carries
    everything `judge_answer` will later need, and a generate-only run can be scored afterwards
    with no regeneration and no second retrieval. That is the whole point of the mode: the answers
    are the expensive half, and which scoring protocol runs over them (an LLM judge, a human
    labelling pass, or both) is a decision worth keeping open after the money is spent.

    `context` and `memories` are persisted even though no judge consumes them. They are what makes
    grounding and faithfulness analysis possible later, and what lets a DIFFERENT answerer be run
    over the same retrieval without paying for retrieval again. Storing the prompt actually sent,
    rather than reconstructing it later from a config, also means a change to
    `get_beam_answer_generation_prompt` cannot silently reinterpret an old artifact.
    """
    context = get_beam_answer_generation_prompt(question.question, memories, top_k=cutoff)
    answer = answerer("", context)
    if "ANSWER:" in answer:
        answer = answer.rsplit("ANSWER:", 1)[-1].strip()
    judged = judge is not None
    if judge is not None:
        mean, nuggets, errors = judge_answer(question, answer, judge)
    else:
        # None, not NaN. `json.dumps` emits a bare `NaN` token, which is not valid JSON: jq,
        # JSON.parse and Postgres jsonb all reject it, and with --no-judge that is EVERY row of a
        # deliverable whose whole purpose is to be scored later by something else. `judged: false`
        # and `judgment: "UNJUDGED"` already carry the state. Verified: allow_nan=False rejects it.
        mean, nuggets, errors = None, [], 0
    mean_number: float | None = mean if isinstance(mean, float) and mean == mean else None
    if not judged:
        # NOT "ERROR". An unjudged row and a row whose judge failed are different states, and
        # `aggregate` counts every non-numeric score as an error — so reusing ERROR here would
        # report a clean generate-only run as 300 judge failures.
        judgment = "UNJUDGED"
    else:
        judgment = (
            ("PASS" if (mean_number or 0.0) >= 0.5 else "FAIL")
            if mean_number is not None
            else "ERROR"
        )
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "question": question.question,
        "rubric": question.rubric,
        "memories_retrieved": len(memories),
        "memories_evaluated": min(len(memories), cutoff),
        # The evidence itself, not just its cardinality. `memories_evaluated` counts what the
        # answerer saw; this is what it saw.
        "memories": list(memories[:cutoff]),
        "context": context,
        "generated_answer": answer,
        "abstained": _looks_like_refusal(answer),
        "retrieval_empty": not memories,
        "judged": judged,
        "score": mean,
        "judgment": judgment,
        "nugget_scores": nuggets,
        "judge_errors": errors,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """BEAM's published metrics, plus the two abstention rates our methodology requires.

    `avg_score` is the headline Mem0 publishes (their 64.1 at 1M is `avg_score` 0.6409, NOT the
    70.14 % pass rate); both are reported so neither can be mistaken for the other.

    A row from a generate-only run carries ``judged: False`` and a NaN score. Those are counted
    SEPARATELY from judge errors: both are non-numeric scores, but "never asked" and "asked and
    failed" are different facts, and folding them together would report an intentional
    generate-only run as a total judge outage. Rows written before this field existed have no
    ``judged`` key and default to True, so older artifacts aggregate exactly as they did.

    The judge-independent signals — abstention, false-abstention, empty retrieval — are ALSO
    reported over every row in `retrieval_all_rows`, because they are measurable without spending
    anything and a generate-only run would otherwise report `None` for facts it actually holds.
    They are a separate block rather than a widened denominator: the existing rates are computed
    over judged rows and published that way, and silently changing what they average over would
    move numbers that are already in an article.
    """
    judged_rows = [r for r in rows if r.get("judged", True)]
    unjudged_rows = [r for r in rows if not r.get("judged", True)]
    scored = [r for r in judged_rows if isinstance(r["score"], (int, float)) and r["score"] == r["score"]]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_type[row["question_type"]].append(row)

    abstention_rows = by_type.get("abstention", [])
    answerable = [r for r in scored if r["question_type"] != "abstention"]

    all_abstention = [r for r in rows if r["question_type"] == "abstention"]
    all_answerable = [r for r in rows if r["question_type"] != "abstention"]

    return {
        "overall": {
            "n": len(scored),
            "errors": len(judged_rows) - len(scored),
            "unjudged": len(unjudged_rows),
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
        # Judge-independent, over EVERY row including unjudged ones. `abstained` is derived from
        # the answer text and `retrieval_empty` from the retrieval itself, so neither needs a
        # judge; a generate-only run holds both facts and should not report None for them.
        "retrieval_all_rows": {
            "n": len(rows),
            "abstention": {
                "n": len(all_abstention),
                "abstain_rate": round(
                    sum(1 for r in all_abstention if r["abstained"]) / len(all_abstention), 4
                )
                if all_abstention
                else None,
            },
            "false_abstain": {
                "n": len(all_answerable),
                "rate": round(
                    sum(1 for r in all_answerable if r["abstained"]) / len(all_answerable), 4
                )
                if all_answerable
                else None,
                "retrieval_empty_rate": round(
                    sum(1 for r in all_answerable if r["retrieval_empty"]) / len(all_answerable), 4
                )
                if all_answerable
                else None,
            },
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


def _redacted_database(dsn: str) -> str:
    """`host:port/dbname` from a DSN, with every credential component discarded.

    The raw DSN must never reach the sidecar or the artifact — `recall/store.py` keeps a
    `redacted_dsn` helper for the same reason — but the database identity has to be compared, or
    "same table name, different host" passes the resume guard.
    """
    try:
        parsed = urllib.parse.urlsplit(dsn)
        return f"{parsed.hostname or ''}:{parsed.port or ''}/{(parsed.path or '').lstrip('/')}"
    except ValueError:
        return ""


def _run_config(args: argparse.Namespace, system: Any) -> dict[str, Any]:
    """The configuration that identifies a run's rows, for the resume guard.

    A DELIBERATE subset of `system.describe()`, not the whole thing. `describe()` also carries
    per-conversation state (`tenant`) and running counters (the entailment guard's
    `candidates_judged` / `candidates_skipped_by_cap`), which change WITHIN a single run. Comparing
    those would make every resume look like a configuration change, and a guard that always fires
    gets disabled — so it would end up protecting nothing.

    What is here is what alters the answers: which corpus, which model, which retrieval knobs,
    which index, and whether a judge ran at all.
    """
    described = system.describe() if hasattr(system, "describe") else {}
    embedder = described.get("embedder") or {}
    entailment = described.get("entailment") or {}
    return {
        "chat_size": args.chat_size,
        "model": args.model,
        "judge_model": args.judge_model or args.model,
        "no_judge": bool(args.no_judge),
        "k": args.k,
        "cutoff": args.cutoff,
        "candidate_k": described.get("candidate_k"),
        "embedder_name": embedder.get("name"),
        "embedder_model": embedder.get("model"),
        "reranker": described.get("reranker"),
        "entailment_guard": entailment.get("guard"),
        "entailment_top_n": entailment.get("top_n"),
        "table": described.get("table"),
        "question_types": args.question_types or "all",
        # Added after six auditors independently found them missing. The calibration changes which
        # memories reach the answerer; `--data` is the corpus the docstring already claimed to
        # cover; and the database is what makes "same table name, different host" — the exact
        # scenario transfer_index.py exists to create — distinguishable.
        "calibration_threshold": (described.get("calibration") or {}).get("threshold"),
        "calibration_certified": (described.get("calibration") or {}).get("certified"),
        "data": str(getattr(args, "data", None) or ""),
        "database": _redacted_database(getattr(args, "dsn", "")),
    }


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    """Write the run's configuration beside its sidecar, before any paid call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False) + "\n", encoding="utf-8")


def _config_sidecar_for(sidecar: Path) -> Path:
    """`x.partial.jsonl` -> `x.partial.config.jsonl`.

    `.jsonl` and NOT `.json`, which matters more than it looks. The results directory has a
    convention this file has to stay out of the way of: `<base>.json` is the finished artifact and
    several callers pick it up with `glob("*.json")` expecting exactly one match, while
    `glob("*.partial.jsonl")` collects the crash-recovery sidecars and is counted. A config named
    `.config.json` silently joins the first set — a real collision, caught by
    `[artifact] = out_dir.glob("*.json")` raising "too many values to unpack".

    `*.json` does not match a `.jsonl` suffix, and `*.partial.jsonl` does not match
    `.partial.config.jsonl`, so this name is in neither set. A single JSON object on one line is
    valid JSONL, so the extension is honest about the contents rather than a dodge.
    """
    return sidecar.with_suffix(".config.jsonl")


def _check_resume_config(
    resume: list[Path] | None, current: dict[str, Any], *, allow: bool
) -> None:
    """Refuse to resume across a configuration change, or where it cannot be verified.

    An unverifiable resume is treated as a FAILING one, not a passing one. A sidecar with no
    recorded configuration might have been produced by this exact setup or by a different
    embedder entirely, and nothing downstream can tell the two apart afterwards — the resulting
    artifact records only the CURRENT config while containing rows from both. "Cannot check" and
    "checked and fine" have to land differently or the guard is decorative.

    `--allow-config-change-on-resume` is the deliberate override, and the caller stamps it.
    """
    if not resume:
        return
    problems: list[str] = []
    for sidecar in resume:
        config_path = _config_sidecar_for(Path(sidecar))
        if not config_path.exists():
            problems.append(
                f"{sidecar}: no {config_path.name} beside it, so the configuration that produced "
                f"those rows is unknown and cannot be compared"
            )
            continue
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{config_path}: unreadable ({exc})")
            continue
        differing = sorted(
            key
            for key in set(current) | set(previous)
            if current.get(key) != previous.get(key)
        )
        if differing:
            detail = ", ".join(
                f"{key}: {previous.get(key)!r} -> {current.get(key)!r}" for key in differing
            )
            problems.append(f"{sidecar}: configuration differs ({detail})")
    if problems and not allow:
        joined = "\n  ".join(problems)
        raise SystemExit(
            "refusing to resume: the rows being carried in were not produced by this "
            "configuration, and the artifact would record only the current one.\n  "
            f"{joined}\n"
            "Pass --allow-config-change-on-resume to override, or start a clean run."
        )
    for problem in problems:
        print(f"WARNING (--allow-config-change-on-resume): {problem}")


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

    Delegates to `recall.eval.resume`, the one factored-out mechanism. That changes one behaviour
    deliberately: a truncated FINAL line used to raise `JSONDecodeError` here and now counts as
    "not yet scored". An interrupted run is exactly the state these sidecars are read in, so a
    torn tail was the one malformed line this function was guaranteed to meet, and refusing to
    resume from it meant re-paying for every question in the file. Corruption in the MIDDLE of a
    sidecar is still loud. The sidecars are NOT repaired in place: this function reads other
    attempts' files, and a reader must not rewrite them.
    """
    read = read_ledger(paths, id_field="question_id", repair_truncated_tail=False)
    return list(read.rows), set(read.ids)


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


def _load_published(path: Path, cutoff: int = DEFAULT_TOP_K) -> dict[str, dict[str, Any]]:
    """Mem0's published per-question evaluations, keyed by question id.

    The cutoff cell is selected BY NAME. Mem0's file nests each answer under a retrieval-budget
    key and ships more than one — `UPSTREAM.md` records `top_200` as the headline with a `top_50`
    variant alongside. This previously took `next(iter(cutoffs))`, i.e. whichever key upstream
    happened to serialise first, and then paired the result against our own k=200 arm: a silent
    four-fold retrieval-budget mismatch on the one axis the suite exists to hold equal.

    The tell was already in the tree — `benchmarks/labelling/build_beam_labelling.py` pins
    `THEIR_CUTOFF = "top_200"`. Two readers of the same third-party file disagreed about which
    cell was the comparator, and only one of them could be right.

    Missing is an error, not a fallback. A quiet substitution here is exactly what made the
    original defect invisible.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    wanted = f"top_{cutoff}"
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("evaluations", []):
        cutoffs = row.get("cutoff_results", {})
        if not cutoffs:
            continue
        if wanted not in cutoffs:
            raise SystemExit(
                f"{path}: question {row.get('question_id')!r} has no {wanted!r} cell "
                f"(available: {sorted(cutoffs)}). Pass --cutoff to name the retrieval budget "
                f"to compare against; picking one implicitly is how the arms drift apart."
            )
        label = wanted
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


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as its own function so the flags can be asserted without running a benchmark.

    `--candidate-k` was accepted by `BeamRecallSystem` and unreachable from here for the whole
    BEAM arm, which made every published cell a no-rerank cell by construction. A flag that
    exists in the adapter and not in the parser fails silently — the run works, it just measures
    a configuration nobody chose.
    """
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
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Fused candidate pool the reranker ranks within, before the top --k are kept. "
        "Defaults to --k, which makes a reranker INERT: pool, returned set and answerer context "
        "are then the same memories, so reranking reorders a prompt instead of selecting what "
        "goes into it. Set it ABOVE --k (e.g. --k 45 --candidate-k 250) for reranking to change "
        "WHICH memories the answerer sees. Past ~250 the HNSW scan stops widening proportionally "
        "(pgvector caps hnsw.ef_search at 1000, derived here as candidate_k x 4): the pool is "
        "still honoured and a RuntimeWarning says so, but the over-fetch margin shrinks.",
    )
    parser.add_argument(
        "--ablation-sample",
        type=_positive_int,
        default=DEFAULT_SAMPLE,
        help=(
            f"questions sampled by the inert-arm preflight (default {DEFAULT_SAMPLE}). Taken "
            "deterministically from the head of the question list, so the verdict is "
            "reproducible for a slice."
        ),
    )
    parser.add_argument(
        "--allow-inert-arm",
        action="store_true",
        help=(
            "record an inert mechanism instead of refusing the run. The override AND every verdict "
            "are stamped into the artifact, so a run let through cannot read as clean afterwards."
        ),
    )
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
        "--calibration",
        type=Path,
        default=None,
        help="A calibration file (from `benchmarks.beam.calibrate`) supplying the abstention "
        "threshold for --embedder. Without it the library's UNTUNED 0.50 cosine floor applies, "
        "which is not comparable across embedders: on voyage-4-large it starves 23.3%% of "
        "questions to empty retrieval (7%% on text-embedding-3-small, 0%% on bge-small), and an "
        "empty context makes the vendored answerer emit its refusal string — so a quarter of the "
        "run would score as false abstentions caused by the harness. The file is keyed by the "
        "embedding PROFILE ID (not the --embedder string) and the run REFUSES if it holds no "
        "entry for this one.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Generate answers and DO NOT judge them. BEAM's judge reads only (question, nugget, "
        "answer) and never the retrieved context, so every row written here can be scored later "
        "by the same judge with no regeneration and no second retrieval. Use when the scoring "
        "protocol is undecided or human — the answers are the expensive half, and this keeps the "
        "choice open after the money is spent. Rows are stamped judged=false and scored UNJUDGED, "
        "counted apart from judge errors.",
    )
    parser.add_argument(
        "--allow-config-change-on-resume",
        action="store_true",
        help="Permit --resume across a configuration change. Off by default: the sidecar's "
        "sibling .partial.config.jsonl is compared against this run's configuration and a "
        "mismatch aborts. Without that check a resume silently folds rows produced under one "
        "embedder/k/reranker into an artifact that records only the new one, and nothing "
        "downstream can tell afterwards.",
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
    return parser


def _main() -> None:
    args = build_parser().parse_args()

    indices: list[int] | None = None
    if args.conversations:
        indices = parse_conversation_indices(args.conversations)

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
        published = _load_published(args.rejudge_mem0, args.cutoff)
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
            # The retrieval budget this arm's answers were taken at. Recorded so `pair.compare`
            # can REFUSE to align two arms measured at different depths — selecting the right
            # cell is not enough on its own if nothing downstream checks that both sides agree.
            "cutoff": args.cutoff,
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
    # The PATH is passed down, not a pre-resolved object: only BeamRecallSystem knows the
    # embedding profile id, and that is the key the file is written under. Resolving it here
    # against `args.embedder` never matched.
    system = BeamRecallSystem(
        args.dsn,
        embedder_name=args.embedder,
        k=args.k,
        table=args.table,
        entailment_top_n=args.entailment,
        reranker_name=args.reranker,
        candidate_k=args.candidate_k,
        calibration_path=args.calibration,
    )
    if args.calibration:
        described = system.describe()["calibration"]
        print(f"calibration: threshold={described['threshold']} "
              f"certified={described['certified']}", flush=True)

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

    run_config = _run_config(args, system)
    _check_resume_config(args.resume, run_config, allow=args.allow_config_change_on_resume)

    rows, done = _already_done(args.resume)
    print(
        f"RE-call arm: {n_conversations} conversations, {len(done)} questions already scored, "
        f"{args.workers} workers"
    )
    write, handle = _writer(out_base.with_suffix(".partial.jsonl"))
    # Written BEFORE the first paid call, so a run that dies on its first question still leaves a
    # sidecar that says what produced it. The blocked 2026-07-28 best-config run is the reason
    # this exists: its runner lived in /tmp, /tmp was cleared by a reboot, and the 5 rows it had
    # already paid for became unusable because nothing recorded their configuration.
    _write_run_config(out_base.with_suffix(".partial.config.jsonl"), run_config)

    def _score(question: Question) -> dict[str, Any]:
        # Retrieval happens INSIDE the worker: it opens its own store and connection, so the
        # threads share no cursor. Ingest stays on the main thread, one conversation at a time —
        # the tenant is per-conversation state on the system object, and indexing two
        # conversations at once would let one's turns answer the other's questions.
        memories = system.retrieve(question.question)
        scored = score_question(
            question,
            memories,
            answerer.complete,
            None if args.no_judge else judge_llm.complete,
            args.cutoff,
        )
        scored["conversation_idx"] = question.conversation_idx
        write(scored)
        return scored

    # Every question this run was supposed to score, accumulated as the conversations are walked
    # and narrowed by exactly the same filters as `pending`. This is what `coverage` is measured
    # against below; deriving it here is the only place it is knowable.
    expected: set[str] = set()
    ablation: list[dict[str, Any]] = []
    ablation_ran = False

    try:
        wanted_types = (
            {s.strip() for s in args.question_types.split(",")} if args.question_types else None
        )
        for conv in conversations:
            in_scope = conv.questions
            if wanted_types is not None:
                in_scope = [q for q in in_scope if q.question_type in wanted_types]
            expected.update(str(q.question_id) for q in in_scope)
            pending = [q for q in in_scope if q.question_id not in done]
            if not pending:
                print(f"  conversation {conv.index}: all {len(conv.questions)} scored, skipping")
                continue
            n_turns = system.ingest(conv)
            print(f"  conversation {conv.index}: {n_turns} turns indexed, {len(pending)} questions")
            if not ablation_ran:
                # After index build, before the FIRST generator call: retrieval-only, so an inert
                # arm is caught before a single token is spent. Fires on whichever conversation is
                # ingested FIRST in this run's loop — not necessarily conv.index 0 — because
                # --resume can skip an already-fully-scored earlier conversation without ever
                # calling ingest() on it, and the preflight needs an index that has actually been
                # built to query.
                ablation = system.ablation_preflight(
                    [q.question for q in in_scope],
                    sample=args.ablation_sample,
                    metric_class="set",
                    allow_inert=args.allow_inert_arm,
                )
                ablation_ran = True
                print(f"ablation preflight: {ablation}", flush=True)
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
        # The same dict the sidecar guard compares on, so the final artifact and the crash-recovery
        # record describe the run in identical terms rather than two hand-maintained copies.
        "run_config": run_config,
        "no_judge": bool(args.no_judge),
        # An override that leaves no trace lets a run that crossed configurations read as clean.
        # `allow_inert_arm` two blocks below is stamped for exactly this reason, and the guard's
        # own docstring claimed "the caller stamps it" while nothing did.
        "resume": {
            "sidecars": [str(x) for x in (args.resume or [])],
            "allow_config_change": bool(args.allow_config_change_on_resume),
        },
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
        # The same guard the re-judge arm carries, on the arm that publishes OUR numbers.
        # `_run_pool` drops a question it could not score and keeps going, which is right — but
        # it means a short run exits 0 and looks complete, with `n` as its only trace. That is
        # the failure this field was written for (101 of 700 lost to one outage window), and it
        # was emitted for the comparator arm and not for this one.
        "coverage": _coverage(expected, {str(r["question_id"]) for r in rows}),
        # The override AND every verdict, stamped here so a run let through with
        # `--allow-inert-arm` cannot read as clean afterwards — this is the only artifact payload
        # the RE-call arm produces, so it is where the preflight's outcome has to land.
        "ablation_preflight": {
            "verdicts": ablation,
            "allow_inert_arm": bool(args.allow_inert_arm),
            "sample": args.ablation_sample,
            "ran": ablation_ran,
        },
    }
    out_base.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
