"""MTRAG Tasks B and C: generate answers, in the official submission format.

Prior work (searched before writing, per CLAUDE.md):
`docs_search(source_type='memory', "MTRAG Task B Task C generation RAG answer quality GPT-4o judge
RL_F RB_llm RB_alg harmonic mean")` returned substantial prior art, and three findings from it bind
this module rather than merely informing it:

  [[project-recall-mtrag-rbalg-probe-2026-08-05]] (merged, PR #212)
  1. ⛔ **`mtrag-human/generation_tasks/reference.jsonl` ships the per-task answerability label**,
     even though MTRAGEval says that metadata was withheld from participants. **It must never
     reach an inference path.** It is the exact thing the abstention decision is supposed to
     INFER. `build_messages` below therefore reads only `speaker` and `text` from each turn, never
     the turn dict and never the task dict.

     ⚠️ WHERE IT ACTUALLY LIVES, counted 2026-08-08 over all 842 rows of BOTH `reference.jsonl`
     and `RAG.jsonl` in the release this run reads:

         top-level `Answerability`  842/842      turns carrying `enrichments`  0/6684
         top-level `targets`        842/842      turn keys: speaker, text, metadata

     Two withheld fields, both TOP-LEVEL siblings of `input`, and `targets` is the more dangerous
     of the pair: `Answerability` reveals whether to abstain, `targets` IS the gold answer.
     (`Question Type` and `Multi-Turn` are top-level too.)

     An earlier draft of this docstring said the label sits "inside each turn's `enrichments`".
     The memo says no such thing, it says "per-task", which is right; the `enrichments` detail was
     invented here and no such key exists anywhere in either file. It was not harmless: the leak
     test's fixture was built to match it, so the guard covered a path the release never emits
     while leaving both real ones unguarded, and it would have stayed green through a genuine
     leak. A guard built on an unverified shape is a guard that cannot fire.
  2. 🔑 **A correct "I don't know" on an UNANSWERABLE task scores exactly 1.0 on RL_F, RB_llm AND
     RB_alg simultaneously** (72/72 model cells), so abstention moves the harmonic mean further per
     unit of effort than anything else. Published baselines manage it 0.0% to 32.7% of the time and
     the rate is close to INVERSELY ranked with overall score.

     ⛔ THE INFERENCE I DREW FROM THAT WAS WRONG, and it is measured wrong, not merely arguable.
     This docstring used to end: "That is why SYSTEM_PROMPT tells the model to say it does not know
     rather than guess: not prompt engineering for advantage, but refusing to handicap the arm on
     the benchmark's dominant lever." Task B over gold contexts, 842 tasks, says otherwise:

         false abstentions on ANSWERABLE      83 / 709  (11.7%)
         RL_F when it abstained vs answered   0.0726 vs 0.8901
         harmonic mean, ours vs THEIR gpt-4o  0.5508 vs 0.6208   (same task, same contexts)

     The prize is capped at 55 unanswerable tasks; the bill was 83 answerable ones at ~0.8 each. A
     lever being dominant does not mean pulling it harder is free, and I never priced the other
     side of the trade. See `PROMPTS`, and pass `--prompt neutral`.
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

from benchmarks.llm import CompletionTruncated

# Private on purpose, and imported rather than reimplemented: there is one definition in this
# repository of how long a provider asked us to wait, and a second copy here would drift from it
# exactly as the two retry policies this file just finished collapsing into one did.
from recall.embeddings import _retry_after_seconds

#: From the official `format_checker.py`. Named here so a violation fails in OUR run, with a
#: message naming the limit, rather than at submission.
MAX_CONTEXTS = 10
MAX_SUBMISSION_MB = 20

#: OpenRouter speaks the OpenAI wire format, so the official evaluator's `--provider openai` path
#: works against it unchanged by pointing the base URL here.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o"

#: ⚠️ MTRAG does NOT publish the prompt its baselines used — the repo ships retrieval conversion
#: and evaluation, no generation script. So an exact prompt match is impossible and any comparison
#: carries an irreducible prompt difference. What IS controllable is not handicapping ourselves,
#: which the first version did.
#:
#: MEASURED, Task B over gold contexts, 842 tasks:
#:   `abstain` produced 83 false abstentions on 709 ANSWERABLE tasks (11.7%), and a false
#:   abstention scores near zero — RL_F 0.0726 against 0.8901 when it answered, RB_llm 0.1542
#:   against 0.7334. Harmonic mean 0.5508 against the benchmark's own GPT-4o at 0.6208 on the same
#:   task, same contexts, same model.
#:
#: 🔑 The reasoning that produced `abstain` was that a correct "I don't know" scores 1.0 on all
#: three metrics at once, so abstention is the dominant lever. That is true and it is only half the
#: trade: 55 unanswerable tasks can win at most 55 points, while 83 false abstentions lose ~0.8
#: each. The prompt bought the smaller prize with the larger one.
#:
#: `neutral` states the task and stops. It does NOT instruct the model to answer regardless, which
#: would be the same error with the sign flipped; what to do when the passages fall short is left
#: to the model, which is the behaviour being measured.
#: ✅ `official` is the BASELINES' OWN PROMPT, quoted from the MTRAG paper (arXiv 2501.03468),
#: Appendix D.2 "Model invocation": "To evaluate LLMs on MTRAG, for each task we sent to the LLM
#: the following information: question, preceding turns, passages, and instruction."
#:
#:     Given one or more documents and a user query, generate a response to the query using less
#:     than 150 words that is grounded in the provided documents. If no answer can be found in the
#:     documents, say, "I do not have specific information"
#:     PASSAGE 1 ... PASSAGE M
#:     User turn 1 / Agent Turn 1 / ... / User Turn N
#:
#: ⛔ Do NOT substitute the prompt in `mtrag-human/conversations/conversations.json`
#: (`generator.prompt.system_instruction`). That one built the conversation DATASET with
#: mixtral-8x7b and has NO length limit; D.2 is the evaluation prompt and does. Reaching for the
#: first prompt found in the repo would be a quieter version of the same mistake documented below.
#:
#: My `abstain` differs from D.2 in THREE ways, not the one I first diagnosed:
#:   1. no 150-word limit (our mean Length 323; RB_alg is built from BertKPrec and RougeL against
#:      the target, both of which a verbose answer depresses);
#:   2. "say that you do not know rather than guessing" instead of the exact string "I do not have
#:      specific information" — far easier to trigger, and worth 83 false abstentions on 709
#:      ANSWERABLE tasks;
#:   3. different ordering and passage framing.
PROMPTS = {
    "official": (
        "Given one or more documents and a user query, generate a response to the query using "
        "less than 150 words that is grounded in the provided documents. If no answer can be "
        'found in the documents, say, "I do not have specific information"'
    ),
    "abstain": (
        "You are answering the last user turn in a conversation, using only the passages provided. "
        "If the passages do not contain the answer, say that you do not know rather than guessing. "
        "Answer directly and concisely."
    ),
    "neutral": (
        "You are answering the last user turn in a conversation, using the passages provided. "
        "Answer directly and concisely."
    ),
}

#: Prompts whose USER-message layout follows D.2 (instruction, PASSAGE 1..M, then the turns)
#: rather than this module's original "Passages:/Conversation:" framing.
OFFICIAL_LAYOUT = {"official"}

#: Default stays `abstain` so every artifact produced before this flag existed is still
#: reproducible from the committed code. New runs pass `--prompt` explicitly.
DEFAULT_PROMPT = "abstain"
SYSTEM_PROMPT = PROMPTS[DEFAULT_PROMPT]


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
    """Every task, validated against what the official checker requires BEFORE anything is spent.

    `format_checker.py` rejects a `rag_taskc` row whose `Collection` is not a string or whose
    `input` is missing. Those are the fields `submission_row` copies straight through, and it
    defaults both to `None`, so without this check a malformed release would be discovered by the
    checker only after ~842 paid calls had already gone out. Validating at load time costs one
    pass over a file we are reading anyway.
    """
    tasks = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "task_id" not in item:
                raise RuntimeError(f"{path}:{line_no} has no task_id")
            if not isinstance(item.get("Collection"), str):
                raise RuntimeError(
                    f"{path}:{line_no} task {item['task_id']}: `Collection` must be a string, got "
                    f"{type(item.get('Collection')).__name__}. The official checker rejects this."
                )
            if not item.get("input"):
                raise RuntimeError(
                    f"{path}:{line_no} task {item['task_id']}: no `input` turns, so there is no "
                    f"question to answer and the submission row would be rejected."
                )
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
    doc_id = ctx.get("document_id")
    if doc_id is None:
        doc_id = ctx.get("id")
    # `is None`, not truthiness: an id of `0` or `""` is a real id the source chose, and silently
    # swapping it for a fallback would mis-attribute the passage.
    score = ctx.get("score")
    # `bool` is a subclass of `int`, so a `True` here would otherwise become the score `1.0`
    # instead of falling through to the rank-derived value.
    #
    # Narrowed inline rather than through a separate `numeric` flag: mypy cannot carry a type
    # narrowing across a boolean variable, so `float(score)` read as `float(Any | None)` and the
    # typecheck job failed on it. Runtime behaviour is identical.
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        resolved_score = float(score)
    else:
        resolved_score = float(total - rank)
    return {
        "document_id": str(doc_id if doc_id is not None else ""),
        "score": resolved_score,
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


def build_messages(
    task: dict[str, Any],
    contexts: list[dict[str, Any]],
    system: str = SYSTEM_PROMPT,
    layout: str = "recall",
) -> list[dict[str, str]]:
    """The prompt for one task.

    `layout="official"` reproduces the paper's Appendix D.2 ordering — instruction, then
    `PASSAGE 1..M`, then the turns — because that is how the baselines saw it and the comparison is
    against them. D.2 says the pieces were "composed into the prompt below, adapted to different
    models using HuggingFace's ChatTemplate", so the split between system and user message is an
    adaptation choice rather than something the paper fixes; the instruction goes in the system
    role, which is the ordinary chat rendering, and the CONTENT and ORDER match D.2 exactly.
    """
    if layout == "official":
        # Filter FIRST, number second. Enumerating the unfiltered list and dropping empties inside
        # the comprehension leaves gaps — "PASSAGE 1, PASSAGE 3" — which D.2 does not have and
        # which this docstring claims not to produce. `normalise_context` tolerates an empty
        # `text`, so the case is reachable rather than theoretical.
        texts = [ctx["text"] for ctx in contexts if ctx["text"]]
        blocks = [f"PASSAGE {i + 1}\n{text}" for i, text in enumerate(texts)]
        body = "\n".join(blocks)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{body}\n{conversation_text(task)}" if body
             else conversation_text(task)},
        ]
    passages = "\n\n".join(
        f"[{i + 1}] {ctx['text']}" for i, ctx in enumerate(contexts) if ctx["text"]
    ) or "(no passages were retrieved)"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Passages:\n{passages}\n\nConversation:\n{conversation_text(task)}\n\n"
                       f"Answer the last user turn.",
        },
    ]


def run_manifest_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".manifest.json")


def write_run_manifest(out: Path, **fields: Any) -> None:
    """Persist what produced this artifact, NEXT TO the artifact.

    The prompt id changes the number — `abstain` cost 0.07 harmonic mean — and printing it to
    stdout only means an artifact whose console output was not captured cannot say which prompt
    made it. `benchmarks/mtrag/run.py` already learned this for git revisions and fixed it the same
    way, with a file rather than a print. It goes in a SIBLING file, not into the submission rows,
    because the official `format_checker.py` validates a fixed per-mode field set and an extra key
    is not worth the risk.
    """
    run_manifest_path(out).write_text(json.dumps(fields, indent=2), encoding="utf-8")


def previous_prompt(out: Path) -> str | None:
    path = run_manifest_path(out)
    if not path.exists():
        return None
    try:
        prompt = json.loads(path.read_text(encoding="utf-8")).get("prompt")
    except (json.JSONDecodeError, OSError):
        return None
    # Validate the type rather than trusting the manifest. This feeds the resume guard, which
    # refuses to continue a run whose prompt differs; a non-string here would compare unequal to
    # every known prompt name and read as UNKNOWN provenance, which is the safe direction but for
    # the wrong reason. Returning None says "no recorded prompt", which is what a bad value means.
    return prompt if isinstance(prompt, str) else None


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
    # `max_retries=0` because `generate_one` owns the retry policy. The SDK default is 2 retries,
    # which does not replace that loop but multiplies with it: at `GENERATION_ATTEMPTS = 4` one
    # 429 costs 12 billed requests rather than 4, which is 3x the worst-case bill the warning
    # above `generate_one` quotes. `PERMANENT_ERROR_NAMES` is unaffected either way: the statuses
    # the SDK retries (408, 409, 429, 5xx, plus ANY status on which the provider sends
    # `x-should-retry: true`) are all but disjoint from the 400/401/403/404 that set names,
    # so what is removed here is the multiplication on transient failures and nothing else.
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=0)


GENERATION_ATTEMPTS = 4
GENERATION_BACKOFF_S = 2.0
#: Consecutive per-task failures that stop the run. One task can fail on its own merits (a
#: content-policy refusal); five in a row is the whole run being broken, and the difference is
#: worth ~837 unnecessary billed attempts.
CONSECUTIVE_FAILURE_LIMIT = 5
PERMANENT_ERROR_NAMES = frozenset(
    {"AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"}
)


def generate_one(client: Any, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    """One answer, retried on transient failures.

    ⚠️ Every attempt is a billed call. Raising `GENERATION_ATTEMPTS` raises the worst-case bill by
    the same factor. An authentication error is not retried: it is not a flaky network, and
    retrying it burns the backoff and reads like one.

    That arithmetic holds only because `client` comes from `openrouter_client`, which builds the
    SDK with `max_retries=0`. A stock `openai` client retries twice inside its own transport, so
    the worst case would be `GENERATION_ATTEMPTS` x 3 — 12 requests, not 4 — and the warning
    above would understate the bill it exists to give by 3x. Pass a client built anywhere else
    and this loop no longer owns the retry policy.

    A completion that stopped for `length` raises `CompletionTruncated` rather than returning.
    `--max-tokens` defaults to 512, so this is an everyday outcome, and the returned string is the
    worst shape a failure can take: a plausible half-answer that the judge scores as if the system
    had produced it, which is a measurement error of OUR configuration wearing the costume of a
    model failure. The caller's per-task quarantine then keeps it out of the submission entirely,
    which is the only safe place for it. It is neither retried nor wrapped: the ceiling that cut
    this attempt cuts the next three identically, and the operator needs to be told which flag to
    raise, not "gave up after 4 attempts, re-run to resume", which is the wrong advice here.
    """
    last: Exception | None = None
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0.0
            )
            # Absence is not evidence: a provider that omits the field is not reporting truncation,
            # and refusing on a missing attribute would fail every task on such a provider.
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raise CompletionTruncated(
                    f"completion hit max_tokens={max_tokens} and was cut off. Raise --max-tokens; "
                    f"do NOT score this answer."
                )
            return (response.choices[0].message.content or "").strip()
        except CompletionTruncated:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised below once attempts are exhausted
            last = exc
            if type(exc).__name__ in PERMANENT_ERROR_NAMES or attempt == GENERATION_ATTEMPTS:
                break
            # `+ _retry_after_seconds` for the same reason `retry_with_backoff` does it, and it
            # is this loop's problem for the same reason: `openrouter_client` passes
            # `max_retries=0`, so the transport layer that used to obey `Retry-After` is gone and
            # nothing else here reads it. The fixed ladder spends all four attempts inside 14s,
            # so a per-minute rate limit the stock client would have waited out fails the task,
            # and `CONSECUTIVE_FAILURE_LIMIT` turns five such tasks into an aborted run.
            paced = _retry_after_seconds(exc) or 0.0
            time.sleep(GENERATION_BACKOFF_S * (2 ** (attempt - 1)) + paced)
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
    ap.add_argument(
        "--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT,
        help=(
            "which system prompt. `abstain` is what every pre-2026-08-08 artifact used and stays "
            "the default so those remain reproducible; see PROMPTS for the measurement that "
            "motivated `neutral`."
        ),
    )
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
    layout = "official" if args.prompt in OFFICIAL_LAYOUT else "recall"

    # ⛔ Resume dedupes on task_id alone, so pointing a DIFFERENT prompt at a partially written
    # file would append new-prompt rows beside old-prompt ones and score the mixture as one arm.
    # Nothing in the submission rows could tell them apart afterwards.
    #
    # 🔑 UNKNOWN provenance is refused too, not just known-different. The first version only fired
    # when a manifest recorded a DIFFERENT prompt, so it could not fire for any file written before
    # manifests existed — which is every artifact generated so far, i.e. exactly the population the
    # guard was added to protect. A guard that is disarmed precisely where it is needed is not a
    # guard. A corrupt manifest reads as unknown and lands here as well, which is the safe side.
    prior = previous_prompt(args.out)
    if done and prior != args.prompt:
        detail = (
            f"generated with --prompt {prior!r}" if prior is not None
            else "of UNKNOWN prompt (no readable manifest beside it)"
        )
        raise RuntimeError(
            f"{args.out} already holds {len(done)} rows {detail}, and this run asks for "
            f"{args.prompt!r}. Resuming would mix two prompts in one file. Write to a new --out, "
            f"or delete the existing one. If you know the existing rows used {args.prompt!r}, "
            f"write that into {run_manifest_path(args.out).name} to record it."
        )
    pending = [t for t in tasks if str(t["task_id"]) not in done]
    print(
        json.dumps({
            "event": "start", "task": args.task, "model": args.model,
            # Recorded because it CHANGES THE NUMBER: `abstain` cost 0.07 harmonic mean on
            # Task B. An artifact that does not name its prompt cannot be compared to one
            # that used a different one.
            "prompt": args.prompt,
            "contexts_from": args.contexts_from if args.task == "c" else "reference",
            "tasks": len(tasks), "already_done": len(tasks) - len(pending), "pending": len(pending),
        }),
        flush=True,
    )

    # To DISK, not only to stdout. A console line does not survive a lost terminal, and the prompt
    # is the difference between two artifacts that are otherwise indistinguishable.
    # NOT on a dry run. A dry run writes no rows, and `tasks` here is post-`--limit`, so a
    # limited preview against an in-progress output would overwrite a real run's manifest with a
    # partial count and a description of a run that produced nothing.
    if not args.dry_run:
        write_run_manifest(
            args.out, prompt=args.prompt, layout=layout, model=args.model, task=args.task,
            contexts_from=args.contexts_from if args.task == "c" else "reference",
            max_tokens=args.max_tokens, tasks=len(tasks), limit=args.limit,
        )

    if args.dry_run:
        chars = missing = 0
        for task in pending:
            contexts = contexts_for(task, recall_contexts)
            if not contexts:
                missing += 1
            chars += sum(len(m["content"]) for m in build_messages(
                task, contexts, PROMPTS[args.prompt], layout))
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
    failed: list[str] = []
    consecutive = 0
    failures_path = args.out.with_suffix(args.out.suffix + ".failed.jsonl")
    # Rewritten, not appended across runs: this file describes the run that just happened. Appending
    # would leave a task that failed once and later succeeded sitting in the log forever, so the
    # file would over-report while the "done" event under-reported, and the two would disagree.
    failures_path.unlink(missing_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for position, task in enumerate(pending, 1):
            try:
                # ⚠️ `contexts_for` and `build_messages` belong INSIDE the guard, not above it.
                # The property being defended is "no single task can end the run", and an earlier
                # version of this block guarded only `generate_one`, which implements the narrower
                # "no single API failure can end the run". A malformed context row raises
                # `AttributeError` out of `normalise_context`, which is not a `RuntimeError`, so it
                # escaped, killed the run, and logged nothing: the exact failure this quarantine
                # exists to prevent, reached by the one path it did not cover. The release's own
                # contexts are well-formed, but `--contexts-from` reads a file we do not control.
                contexts = contexts_for(task, recall_contexts)
                answer = generate_one(
                    client, args.model,
                    build_messages(task, contexts, PROMPTS[args.prompt], layout),
                    args.max_tokens
                )
            except MemoryError:
                # Not quarantinable. `MemoryError` is an `Exception`, so the handler below would
                # otherwise catch it and loop straight into the next allocation-heavy iteration,
                # with the interpreter in a state that is not safe to keep working in. Unlikely
                # with payloads this small, which is why it is a carve-out and not a redesign.
                raise
            except Exception as exc:
                # Deliberately `Exception`, not a list of the failures imagined so far. Enumerating
                # types is what produced the bug above: any type not on the list becomes fatal, and
                # the list is guesswork about data we do not control. `BaseException` still
                # propagates, so Ctrl-C and SystemExit stop the run as they should. The type is
                # recorded so a systematic fault is legible rather than hidden as "some failures".
                failed.append(str(task["task_id"]))
                consecutive += 1
                with failures_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"task_id": task["task_id"],
                                         "error_type": type(exc).__name__,
                                         "error": str(exc)}) + "\n")
                print(
                    json.dumps({"event": "task_failed", "task_id": task["task_id"],
                                "error_type": type(exc).__name__,
                                "consecutive": consecutive, "error": str(exc)[:200]}),
                    flush=True,
                )
                # But a run where nothing succeeds is a broken key or a dead model, not a run of
                # unlucky tasks, and burning through 842 of those is the expensive way to find out.
                if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                    raise RuntimeError(
                        f"{consecutive} tasks failed in a row, the last being {task['task_id']}: "
                        f"{exc}. Stopping rather than attempting the remaining "
                        f"{len(pending) - position}, because a whole-run fault (bad key, unknown "
                        f"model, exhausted credit) looks exactly like this."
                    ) from exc
                continue
            consecutive = 0
            handle.write(json.dumps(submission_row(task, contexts, answer), ensure_ascii=False) + "\n")
            # Flushed per answer, not per batch. The file IS the checkpoint: a crash costs the
            # call in flight, never the ones already paid for.
            handle.flush()
            written += 1
            if position % 25 == 0:
                print(
                    json.dumps({"event": "progress", "written": written,
                                "failed": len(failed), "pending": len(pending)}),
                    flush=True,
                )

    size_mb = check_submission_size(args.out)
    print(
        json.dumps({"event": "done", "written": written, "output": str(args.out),
                    "failed": len(failed), "failed_task_ids": failed[:20],
                    "failures_log": str(failures_path) if failed else None,
                    "size_mb": round(size_mb, 2)}),
        flush=True,
    )
    # A partial submission is not a submission, and saying so ONLY on stdout is a false green: a
    # caller that gates on the exit code, which is the normal thing to do, reads success.
    #
    # The consecutive-failure breaker cannot cover this. It trips on 5 IN A ROW, so a fault that
    # hits most tasks in a scattered pattern, or one confined to a tail shorter than the limit,
    # fails almost everything and still finishes "cleanly": measured at 30 of 40 tasks failed with
    # rc=0. `check_submission_size` counts bytes and never completeness, so nothing downstream
    # catches it either. Exit code 0 has to mean the submission is whole.
    if failed:
        print(
            json.dumps({"event": "incomplete", "note":
                        f"{len(failed)} of {len(pending)} task(s) have no answer, so this file is "
                        f"NOT submittable. Re-run the same command to retry only those; the "
                        f"completed ones are skipped."}),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
