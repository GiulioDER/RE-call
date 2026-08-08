"""MTRAG Tasks B and C: generate answers, in the official submission format.

Prior work (searched before writing, per CLAUDE.md):
`docs_search(source_type='memory', "MTRAG Task B Task C generation RAG answer quality GPT-4o judge
RL_F RB_llm RB_alg harmonic mean")` returned substantial prior art, and three findings from it bind
this module rather than merely informing it:

  [[project-recall-mtrag-rbalg-probe-2026-08-05]] (merged, PR #212)
  1. ⛔ **`mtragun-human/generation_tasks/reference.jsonl` ships the per-task `answerability`
     label**, inside each turn's `enrichments`, even though MTRAGEval says that metadata was
     withheld from participants. **It must never reach an inference path.** It is the exact thing
     the abstention decision is supposed to INFER. `build_messages` below therefore reads only
     `speaker` and `text` from each turn, never the turn dict itself.
  2. 🔑 **A correct "I don't know" on an UNANSWERABLE task scores exactly 1.0 on RL_F, RB_llm AND
     RB_alg simultaneously** (72/72 model cells), so abstention moves the harmonic mean further per
     unit of effort than anything else. Published baselines manage it 0.0% to 32.7% of the time and
     the rate is close to INVERSELY ranked with overall score. That is why `SYSTEM_PROMPT` tells the
     model to say it does not know rather than guess: not prompt engineering for advantage, but
     refusing to handicap the arm on the benchmark's dominant lever.
  3. ⚠️ **`RAG.json` licenses a PAIRED WITHIN-FILE comparison and no parity claim.** Recomputing the
     published table from it runs +0.018 to +0.043 HIGH on every model. The aggregation formula (a
     harmonic mean of the three per-metric means) is right and reproduces 5ting's self-reported
     SemEval triple to four decimals, but the instance set is not the published one. So a number
     from this module is comparable as an ANCHORED LIFT against baselines recomputed from
     `RAG.json` the same way, and NOT against the published leaderboard directly.

  [[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]]
  4. RE-call's MTRAG bottleneck is retrieval COVERAGE: R@100 0.687 with a full-pool ceiling of
     0.7365 against a ~0.95 saturation threshold. A reranker was measured on MTRAG at **+0.0864,
     CI [+0.0671, +0.1061]**.
  5. 🔑 **"A downstream component measured over a starved retriever measures the retriever."** An
     abstention pilot that looked like three LLMs over-refusing was confounded: the judge was asked
     about the retrieved top-5 while the label describes the CORPUS. That is the trap this module
     walks into if Task C's numbers are read as a statement about generation.

The published baselines' own responses and per-metric scores already exist in
`mtrag-human/evaluations/RAG.json` (842 tasks x 10 models), analysed by `benchmarks/mtrag/probe/`
with no generation and no API call. This module does not re-derive them; it adds OUR row.


Task A asks which passages a system retrieves. Tasks B and C ask what it SAYS, and they are what
the MTRAG paper's leaderboard ranks, by the harmonic mean of RL_F, RB_llm and RB_alg.

  Task B (Reference) generates over the task's OWN gold contexts, so it measures the generator
  with retrieval held perfect. Task C (RAG) generates over RETRIEVED contexts, so it measures the
  whole system. Running both is what separates "our retrieval is behind" from "our generation is".

⚠️ Two limits come from the official `format_checker.py`, not from us, and both are enforced here
rather than discovered at submission time: at most **10** contexts per task (`MAX_CONTEXTS`), and a
prediction file of at most **20 MB** (`check_file_size`). RE-call's own Task A predictions carry
100 contexts each and run to 127 MB, so trimming is not optional.

Required fields, also from `format_checker.py`, and they differ per mode:

    retrieval_taska   task_id, Collection, contexts
    generation_taskb  task_id, input, contexts, predictions
    rag_taskc         task_id, Collection, input, contexts, predictions

Each context needs `document_id` (str) and `score` (numeric); each prediction needs `text` (str).

RESUMABLE BY DESIGN, and that is the whole point. Generation is the expensive step: 842 tasks
against a hosted model, twice. `benchmarks/mtrag/run.py` learned this the hard way, where a
transient failure discarded an arm's already-billed work because predictions were held in memory
until the end. Here every answer is appended to the output as it arrives, and a re-run skips the
task ids already present. A crash costs the current call, not the run.

The judge is deliberately NOT here. Generation writes predictions; scoring reads them. Keeping
them apart means the judge can be re-run, or swapped for a different judge, without paying to
generate again.

Usage:
    python -m benchmarks.mtrag.generation \\
        --mtrag-root /path/to/mt-rag-benchmark \\
        --task b \\
        --out results/taskb.predictions.jsonl \\
        --model openai/gpt-4o

    python -m benchmarks.mtrag.generation \\
        --mtrag-root /path/to/mt-rag-benchmark \\
        --task c --contexts-from benchmark \\
        --out results/taskc_benchmark.predictions.jsonl

    python -m benchmarks.mtrag.generation \\
        --mtrag-root /path/to/mt-rag-benchmark \\
        --task c --contexts-from /path/to/hybrid_splade.predictions.jsonl \\
        --out results/taskc_recall.predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

#: From the official `format_checker.py`. Named here so a violation fails in OUR run, with a
#: message naming the limit, rather than at submission.
MAX_CONTEXTS = 10
MAX_SUBMISSION_MB = 20

#: OpenRouter speaks the OpenAI wire format, so the official evaluator's `--provider openai` path
#: works against it unchanged by pointing the base URL here.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o"

#: The instruction is deliberately plain and identical across both tasks. Anything cleverer would
#: be prompt engineering that the leaderboard rows we compare against did not get, and the
#: comparison is about the retrieval and the model, not about our prompt.
SYSTEM_PROMPT = (
    "You are answering the last user turn in a conversation, using only the passages provided. "
    "If the passages do not contain the answer, say that you do not know rather than guessing. "
    "Answer directly and concisely."
)


def generation_tasks_path(root: Path, task: str) -> Path:
    """Where the 842 generation tasks live for this task letter.

    Both files carry the SAME 842 task ids. `reference.jsonl` supplies gold contexts and
    `RAG.jsonl` supplies the benchmark's own retrieved contexts, so which file is read decides
    what the generator is given, not which questions are asked.
    """
    if task == "b":
        return root / "mtrag-human" / "generation_tasks" / "reference.jsonl"
    return root / "mtrag-human" / "generation_tasks" / "RAG.jsonl"


def load_generation_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "task_id" not in item:
                raise RuntimeError(f"{path}:{line_no} has no task_id")
            tasks.append(item)
    return tasks


def load_recall_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Contexts keyed by task_id, from a `benchmarks.mtrag.run` predictions file.

    That file carries 100 contexts per task, in RE-call's fused and reranked order. The order is
    what matters and it is preserved; the trim to `MAX_CONTEXTS` happens at write time so the
    generator and the submission see the same passages.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_task[row["task_id"]] = row.get("contexts", [])
    return by_task


def normalise_context(ctx: dict[str, Any], rank: int, total: int) -> dict[str, Any]:
    """One context in the official shape, keeping the text the evaluator needs.

    `score` must be numeric and is what the evaluator reconstructs a ranking from, so a strictly
    descending value preserves whatever order the source produced. The benchmark's own contexts do
    not always carry one; deriving it from the rank keeps a single rule for every source.
    """
    return {
        "document_id": str(ctx.get("document_id") or ctx.get("id") or ""),
        "score": float(ctx["score"]) if isinstance(ctx.get("score"), (int, float)) else float(total - rank),
        "text": str(ctx.get("text") or ""),
        "title": ctx.get("title"),
    }


def contexts_for(
    task: dict[str, Any], recall_contexts: dict[str, list[dict[str, Any]]] | None
) -> list[dict[str, Any]]:
    """The passages this task's answer is generated from, trimmed to the official limit.

    When `recall_contexts` is given, a task ABSENT from it gets an empty list rather than falling
    back to the benchmark's own contexts. Silently substituting would mean the run reports RE-call
    numbers for turns RE-call never retrieved, which is the flattering direction and would be
    invisible in the output.
    """
    raw = task.get("contexts") or [] if recall_contexts is None else recall_contexts.get(task["task_id"], [])
    total = len(raw)
    trimmed = raw[:MAX_CONTEXTS]
    return [normalise_context(ctx, rank, total) for rank, ctx in enumerate(trimmed)]


def conversation_text(task: dict[str, Any]) -> str:
    turns = task.get("input") or []
    lines = []
    for turn in turns:
        speaker = str(turn.get("speaker", "user"))
        lines.append(f"{speaker}: {turn.get('text', '')}")
    return "\n".join(lines)


def build_messages(task: dict[str, Any], contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[{i + 1}] {ctx['text']}" for i, ctx in enumerate(contexts) if ctx["text"]
    ) or "(no passages were retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Passages:\n{passages}\n\nConversation:\n{conversation_text(task)}\n\n"
                       f"Answer the last user turn.",
        },
    ]


def already_done(path: Path) -> set[str]:
    """Task ids already in the output, so a re-run does not pay for them twice.

    A malformed trailing line, which is the shape an interrupted write leaves, is ignored rather
    than fatal: that task is simply regenerated. Re-doing one answer is the cheap error here;
    treating a truncated file as complete is the expensive one. Same rule as
    `scripts/encode_sparse.py --resume`, for the same reason.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "task_id" in row:
                done.add(str(row["task_id"]))
    return done


def openrouter_client(api_key: str | None = None) -> Any:
    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "set OPENROUTER_API_KEY (or pass --api-key-file). Resolved eagerly so a missing key "
            "fails before the first task rather than after the run has started."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


GENERATION_ATTEMPTS = 4
GENERATION_BACKOFF_S = 2.0
PERMANENT_ERROR_NAMES = frozenset(
    {"AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"}
)


def generate_one(client: Any, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    """One answer, retried on transient failures.

    ⚠️ Every attempt is a billed call. Raising `GENERATION_ATTEMPTS` raises the worst-case bill by
    the same factor. An authentication error is not retried: it is not a flaky network, and
    retrying it burns the backoff and reads like one.
    """
    last: Exception | None = None
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0.0
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - re-raised below once attempts are exhausted
            last = exc
            if type(exc).__name__ in PERMANENT_ERROR_NAMES or attempt == GENERATION_ATTEMPTS:
                break
            time.sleep(GENERATION_BACKOFF_S * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"generation gave up after {GENERATION_ATTEMPTS} attempts "
        f"({type(last).__name__}: {last}). Answers already written are kept; re-run to resume."
    ) from last


def submission_row(task: dict[str, Any], contexts: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    """One line in the official shape, carrying every field any of the three modes requires."""
    return {
        "task_id": task["task_id"],
        "conversation_id": task.get("conversation_id"),
        "Collection": task.get("Collection"),
        "input": task.get("input"),
        "contexts": contexts,
        "predictions": [{"text": answer}],
    }


def check_submission_size(path: Path) -> float:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_SUBMISSION_MB:
        raise RuntimeError(
            f"{path} is {size_mb:.1f} MB, above the official {MAX_SUBMISSION_MB} MB limit that "
            f"`format_checker.py:check_file_size` enforces. Contexts are already trimmed to "
            f"{MAX_CONTEXTS}; the remaining size is passage text."
        )
    return size_mb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mtrag.generation")
    ap.add_argument("--mtrag-root", type=Path, required=True)
    ap.add_argument("--task", choices=("b", "c"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--contexts-from", default="benchmark",
        help="task c only: 'benchmark' for the release's own RAG contexts, or a path to a "
             "benchmarks.mtrag.run predictions file to use RE-call's",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="first N tasks; for a cost probe")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="build every prompt and report the work and rough token count, WITHOUT calling the "
             "model or spending anything",
    )
    args = ap.parse_args(argv)

    tasks = load_generation_tasks(generation_tasks_path(args.mtrag_root, args.task))
    recall_contexts = None
    if args.task == "c" and args.contexts_from != "benchmark":
        recall_contexts = load_recall_contexts(Path(args.contexts_from))

    if args.limit is not None:
        tasks = tasks[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    pending = [t for t in tasks if str(t["task_id"]) not in done]
    print(
        json.dumps({
            "event": "start", "task": args.task, "model": args.model,
            "contexts_from": args.contexts_from if args.task == "c" else "reference",
            "tasks": len(tasks), "already_done": len(tasks) - len(pending), "pending": len(pending),
        }),
        flush=True,
    )

    if args.dry_run:
        chars = missing = 0
        for task in pending:
            contexts = contexts_for(task, recall_contexts)
            if not contexts:
                missing += 1
            chars += sum(len(m["content"]) for m in build_messages(task, contexts))
        print(
            json.dumps({
                "event": "dry_run", "prompt_chars": chars,
                "approx_input_tokens": chars // 4, "tasks_without_contexts": missing,
                "note": "no model was called and nothing was spent",
            }),
            flush=True,
        )
        return 0

    client = openrouter_client()
    written = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for position, task in enumerate(pending, 1):
            contexts = contexts_for(task, recall_contexts)
            answer = generate_one(client, args.model, build_messages(task, contexts), args.max_tokens)
            handle.write(json.dumps(submission_row(task, contexts, answer), ensure_ascii=False) + "\n")
            # Flushed per answer, not per batch. The file IS the checkpoint: a crash costs the
            # call in flight, never the ones already paid for.
            handle.flush()
            written += 1
            if position % 25 == 0:
                print(
                    json.dumps({"event": "progress", "written": written, "pending": len(pending)}),
                    flush=True,
                )

    size_mb = check_submission_size(args.out)
    print(
        json.dumps({"event": "done", "written": written, "output": str(args.out),
                    "size_mb": round(size_mb, 2)}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
