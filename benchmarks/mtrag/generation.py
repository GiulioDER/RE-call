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

from benchmarks.llm import (
    CompletionTruncated,
    EmptyCompletion,
    NoCompletionChoices,
    _safe_reason,
    _shape_note,
)
from recall._chat_content import assistant_text

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


class CheckpointChanged(RuntimeError):
    """The output file moved under a repair that was about to rewrite it.

    Distinct from an `OSError` because the cause is another RUN rather than the filesystem, and the
    operator's fix is a different `--out` rather than a permission or a disk.
    """


def submission_answer(row: object) -> str:
    """The prediction text a written row carries, stripped, or "" if it carries none.

    Every shape that is not a non-blank string collapses to "", because they are the same thing to
    a judge: no `predictions` key, an empty list, an entry that is not an object, a `text` that is
    null or not a string, and a `text` of whitespace all score as a failure by the system under
    test rather than as an answer.

    ⚠️ EVERY entry is considered, not just the first. `submission_row` writes exactly one, so a
    row with more is a file this module did not write, which `--contexts-from` already reads and
    hand-merged submissions produce. `drop_answerless_rows` deletes the whole ROW on the strength
    of this answer, so reading only `predictions[0]` would destroy a real answer sitting behind a
    blank one. Deleting is irreversible and regenerating a task is not, so the cheap error has to
    be the second.
    """
    if not isinstance(row, dict):
        return ""
    predictions = row.get("predictions")
    if not isinstance(predictions, list):
        return ""
    for entry in predictions:
        text = entry.get("text") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def already_done(path: Path) -> set[str]:
    """Task ids already in the output CARRYING AN ANSWER, so a re-run does not pay for them twice.

    A malformed trailing line, which is the shape an interrupted write leaves, is ignored rather
    than fatal: that task is simply regenerated. Re-doing one answer is the cheap error here;
    treating a truncated file as complete is the expensive one. Same rule as
    `scripts/encode_sparse.py --resume`, for the same reason.

    ⚠️ Carrying an answer is checked, not just carrying a `task_id`. Counting the id alone was
    right while every written row necessarily held an answer, and stopped being right the moment
    this module could write `predictions: [{"text": ""}]`, which it did for every `content_filter`
    and `tool_calls` completion before `generate_one` grew its guards. Those guards fix new runs
    and leave OLD artifacts stranded: re-running the identical command over one skipped the empty
    rows, wrote nothing, and exited 0 over a file that cannot be submitted. The `incomplete` gate
    cannot catch that either, since it reports tasks that failed IN THIS RUN and these are never
    attempted.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    # `errors="surrogateescape"` so a checkpoint torn mid-character is malformed rather than
    # FATAL. It has to be the OPEN rather than an `except UnicodeDecodeError` below: strict
    # decoding raises from the file ITERATOR, outside the `try`, so no handler here could catch
    # it. `main` writes with `ensure_ascii=False`, so every non-ASCII answer is raw multi-byte
    # UTF-8 and a kill during the per-row flush can split one. Decoding strictly raised
    # `UnicodeDecodeError`, a `ValueError`, which no caller catches.
    with path.open(encoding="utf-8", errors="surrogateescape") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "task_id" in row and submission_answer(row):
                done.add(str(row["task_id"]))
    return done


def unparsable_rows(path: Path) -> int:
    """How many lines of `path` are not readable JSON.

    A fragment left by a kill mid-write is KEPT by `drop_answerless_rows`, deliberately, so it
    survives every later resume. Nothing else in this module parses what it wrote back:
    `check_submission_size` counts bytes, and the `incomplete` gate reports tasks that failed IN
    THIS RUN. So a run could hand back a file the official checker rejects and still exit 0, which
    is the contract `main` closes with: exit code 0 has to mean the submission is whole.

    Counted rather than deleted. Removing the fragment would silently drop whatever partial row it
    holds, and the operator is the one who should decide that; saying it is there costs nothing and
    is the part that was missing.
    """
    if not path.exists():
        return 0
    bad = 0
    # ⚠️ `newline="\n"` so this splits the file the way the CHECKER does. Universal-newline mode
    # treats a bare CR as a terminator, so a pair of rows glued by one read back as two clean rows
    # and this counted nothing, while the official `format_checker.py`, `jq` and
    # `pandas.read_json(lines=True)` all see a single unparseable line. `ensure_final_newline`
    # refuses a bare CR for the same reason; a mid-file one is permanent, since that guard only
    # fixes the end of the file and the repair keeps lines verbatim.
    with path.open(encoding="utf-8", newline="\n", errors="surrogateescape") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    return bad


def ensure_final_newline(path: Path) -> bool:
    """Terminate the last line if it is not, so the next appended row starts its own. Returns
    whether a byte was added.

    ⚠️ On EVERY resume, not only the ones that repair something. A run killed mid-flush leaves a
    last line without its terminator, and `main` then appends onto the end of it: two rows become
    one unparseable line, losing the answer already paid for and the one just paid for.
    `check_submission_size` counts bytes rather than rows, so the run still exits 0. It does not
    self-heal either, because the glued line reads as malformed on every later resume.

    A pure append of one byte, which is why it needs no staging, no atomicity window and no
    concurrency check: it cannot truncate, and a second run appending a whole row concurrently
    still lands on its own line.

    A bare CR does not count. It terminates a line for Python's own readers, which is why this went
    unnoticed, but not for anything that splits on LF: the official `format_checker.py`, `jq`,
    `pandas.read_json(lines=True)`.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False
    with path.open("ab") as handle:
        handle.write(b"\n")
    return True


def drop_answerless_rows(path: Path, task_ids: set[str]) -> int:
    """Remove rows for `task_ids` that carry no answer. Returns how many went.

    ⚠️ Required by `already_done`'s emptiness check rather than merely tidy. The output is opened
    in APPEND mode and IS the run's checkpoint, so regenerating a task whose empty row is still on
    disk would leave TWO rows under one `task_id`: one unscorable and one real, in a format whose
    checker expects one row per task, read by a judge that would score whichever it met first.
    Repairing the artifact and corrupting it differ by exactly this call.

    `task_ids` is the set this run is about to regenerate, and nothing outside it is touched. Under
    `--limit`, or against a task list that no longer carries an id, dropping a row that nothing
    will replace shrinks the artifact without repairing it, which is a worse failure than the one
    being fixed here.

    An id already carrying an answer elsewhere in the file is in scope too, whatever `task_ids`
    says. Such a row is a duplicate that already exists, so `already_done` reports the task as
    finished and it never becomes pending: without this the twin could never be removed, and the
    run would exit 0 over an artifact the official checker rejects, which is the false green this
    whole guard exists to kill. Dropping it leaves exactly the good row, so nothing is unreplaced.

    Lines that do not parse are KEPT verbatim. They carry no `task_id`, so they cannot become a
    duplicate, and `already_done` already treats them as absent; rewriting them away would be a
    second, unrelated change to what an interrupted write leaves behind.

    ⚠️ This rewrites the run's ONLY checkpoint for work that cost real money, so it is held to the
    standard `benchmarks/mtrag/multiquery.py:write_manifest` already sets for a far cheaper file:
    staged in a PID-suffixed sibling, flushed and fsynced before the swap, swapped with
    `os.replace`, and the scratch removed in a `finally`. Covers process death; NOT full power-loss
    safety, since the containing directory is not fsynced and cannot portably be on Windows.

    It only happens when there is something to drop, because a checkpoint is not worth replacing to
    change nothing: the window would be pure risk paid on every ordinary resume.

    Read-modify-write with no lock, so it refuses rather than races. Nothing under `benchmarks/`
    takes a lock on an output, and a row appended by a second run between the read and the swap
    would be destroyed by the swap after being paid for. A size that moved is enough to detect it
    and is cheaper than a lock protocol this module is the only user of.
    """
    if not path.exists():
        return 0
    size_before = path.stat().st_size
    answered = already_done(path)
    kept: list[str] = []
    dropped = 0
    # `newline=""` so the lines arrive with their own terminators intact. Universal-newline mode
    # strips the CR before the line is kept, which on Windows silently rewrote every CRLF row that
    # `main` had appended into LF, and split any torn line containing a bare CR into two.
    with path.open(encoding="utf-8", newline="", errors="surrogateescape") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            answerless = (
                isinstance(row, dict)
                # `"task_id" in row` before reading it: `str(row.get("task_id"))` renders a missing
                # key as "None", and a task whose id is null puts that same string in `task_ids`,
                # which made every untagged row in the file a match.
                and "task_id" in row
                and (str(row["task_id"]) in task_ids or str(row["task_id"]) in answered)
                and not submission_answer(row)
            )
            if answerless:
                dropped += 1
            else:
                kept.append(line)
    if not dropped:
        return 0
    # The prune owns every byte it writes back, so it is where the terminator is guaranteed. A last
    # line without one glues the next appended row onto it: two rows become one unparseable line,
    # losing both the kept answer and the one just paid for, and it does not self-heal, because the
    # glued line reads as malformed on every later resume.
    # An LF specifically. A bare CR terminates a line for Python's own readers, which is why this
    # went unnoticed, but not for anything that splits on LF: the official `format_checker.py`,
    # `jq`, `pandas.read_json(lines=True)`. A line already ending "\r\n" passes untouched.
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += "\n"
    if path.stat().st_size != size_before:
        raise CheckpointChanged(
            f"{path} grew from {size_before} to {path.stat().st_size} bytes while it was being "
            "read, which means another run is appending to it. Rewriting it now would destroy "
            "rows that run has already paid for. Give each run its own --out."
        )
    tmp = path.with_name(f"{path.name}.{os.getpid()}.pruned")
    try:
        # `surrogateescape` on BOTH sides or the round trip is not one: it is the only error
        # handler that writes the undecodable bytes back exactly as they were read.
        with tmp.open("w", encoding="utf-8", newline="", errors="surrogateescape") as handle:
            handle.write("".join(kept))
            handle.flush()
            os.fsync(handle.fileno())
        # Re-checked immediately before the swap. The first check closed the READ window and left
        # this one open, and this is the slow half: it spans the whole write and the fsync. This
        # narrows the race to the `os.replace` call itself; only a lock would close it, and nothing
        # under `benchmarks/` has one.
        if path.stat().st_size != size_before:
            raise CheckpointChanged(
                f"{path} changed size while the repair was being written, which means another run "
                "is appending to it. Give each run its own --out."
            )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return dropped


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


#: Distinguishes "the field is not there" from "the field is there and null". `getattr(..., None)`
#: cannot: a missing `message` is a MALFORMED BODY (the provider's fault, transient, worth another
#: route) while a `content` that arrived null is an EMPTY ANSWER (permanent, one billed call).
#: Collapsing the two would price a malformed body at one attempt, or an empty answer at four.
_NO_CONTENT_FIELD = object()

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
    #: What the failure actually COST, which is not `GENERATION_ATTEMPTS` whenever the loop broke
    #: early. Tracked separately rather than read off `attempt` after the loop, because that name
    #: is undefined if the range is empty and the failure would then be a `NameError` raised from
    #: the error path, losing the exception it was reporting.
    spent = 0
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        spent = attempt
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=0.0
            )
            # Read the list ONCE, and defensively. OpenRouter answers 200 with an empty `choices`
            # array when the upstream it routed to faults, and indexing it blind raised
            # `IndexError: list index out of range` — a message naming a list operation rather
            # than the provider, which is what an operator found in `.failed.jsonl`.
            #
            # ⚠️ `NoCompletionChoices` is TRANSIENT, unlike the two refusals below it, and that is
            # why it is absent from both `PERMANENT_ERROR_NAMES` and the
            # `except (CompletionTruncated, EmptyCompletion): raise` tuple. This loop retries
            # anything not named permanent, which is the behaviour wanted: the request is well
            # formed and a re-route can serve it. `benchmarks/llm.py` classifies the same type the
            # same way in `TRANSIENT_ERRORS`, so there is one answer to this question repo-wide.
            choices = response.choices
            if not choices:
                raise NoCompletionChoices(
                    # "empty OR ABSENT": `not choices` also takes this branch for `None`, which is
                    # what the SDK surfaces when the body omits `choices` entirely — OpenRouter's
                    # documented failure shape, so the realistic case rather than the exotic one.
                    # Naming only "empty" asserted a shape that was never observed.
                    "the provider returned a 200 with an empty or absent `choices` list, so there "
                    "is no completion to read. This is an upstream fault on the provider's side, "
                    "not a property of the request."
                )
            choice = choices[0]
            # Absence is not evidence: a provider that omits the field is not reporting truncation,
            # and refusing on a missing attribute would fail every task on such a provider.
            reason = getattr(choice, "finish_reason", None)
            if reason == "length":
                raise CompletionTruncated(
                    f"completion hit max_tokens={max_tokens} and was cut off. Raise --max-tokens; "
                    f"do NOT score this answer."
                )
            # `assistant_text`, not `content or ""`. A list of text blocks is TRUTHY, so `or ""`
            # passed it straight through and `.strip()` raised `AttributeError` — a type that is
            # NOT in `PERMANENT_ERROR_NAMES`, so every affected task paid four BILLED attempts
            # before failing, and five such tasks in a row aborted the run. Shared with the other
            # two OpenAI-compatible clients in this repo so the rule cannot drift between them.
            # ⛔ ONE FIELD FURTHER than the `choices` guard above, because stopping there left the
            # next dereference blind: a 200 whose first choice carries `message: null` raised
            # `AttributeError: 'NoneType' object has no attribute 'content'` at four billed calls,
            # which is the failure class this guard exists to abolish, reached through the field
            # along. `assistant_text` is total, so `message` was the only crash surface a
            # WELL FORMED SDK response object could still present here — narrower than the
            # first wording, which claimed it was the only one at all. `getattr(..., default)`
            # suppresses only `AttributeError`, so a descriptor raising anything else still
            # escapes, and `not choices`, `choices[0]` and `reason == "length"` all run
            # provider-supplied code inside this same `try`. The real SDK exposes these as
            # plain pydantic attributes, and the enclosing retry bounds the cost either way.
            #
            # A SENTINEL, not `getattr(..., None)`, because the two cases must not merge. A missing
            # `message` is a malformed body: the provider's fault, transient, worth another route.
            # A `content` that arrived and is null is an empty ANSWER: permanent, and it has to
            # keep falling through to the `EmptyCompletion` below at one billed call, not four.
            content = getattr(getattr(choice, "message", None), "content", _NO_CONTENT_FIELD)
            if content is _NO_CONTENT_FIELD:
                raise NoCompletionChoices(
                    "the provider returned a 200 whose first choice carries no `message.content` "
                    "field at all, so there is no completion to read. This is an upstream fault "
                    "on the provider's side, not a property of the request."
                )
            raw = assistant_text(content)
            answer = raw.strip()
            if not answer:
                # ⛔ An empty reading is REFUSED, not returned. `main` writes whatever this
                # returns straight into `predictions` and counts the task as done, so returning
                # "" put an unscorable empty answer in the submission with rc=0 — the exact
                # defect PR #307 fixed in `benchmarks/llm.py`, which nothing here guarded.
                #
                # Reading block lists (rather than crashing on them) is what made this urgent:
                # `(content or "").strip()` raised `AttributeError` on a block list, so the task
                # was quarantined loudly. Reading it correctly routed every UNREADABLE block list
                # into the silent hole that `content=None` was already in.
                # ⚠️ Through `_safe_reason`, and the raw value on the ATTRIBUTE, exactly as
                # `EmptyCompletion`'s own docstring mandates. Rendering the provider's chosen
                # string with `!r` put untrusted text into a message, which is the hazard that
                # docstring exists to close — and a `finish_reason` whose `__repr__` raises turned
                # this single-billed-call typed refusal into four attempts and an untyped wrapper,
                # because the f-string is evaluated inside the `try`.
                #
                # `_shape_note` distinguishes the three causes this message used to conflate: a
                # block list this reader could not read, blocks holding only whitespace, and a
                # shape that is neither. `.failed.jsonl` is the only record of why a task was
                # quarantined, and on 842 tasks those want different responses.
                raise EmptyCompletion(
                    f"the provider returned a completion with no usable text "
                    f"(finish_reason={_safe_reason(reason)}).{_shape_note(content, raw)} "
                    f"Writing it would put an unscorable empty prediction in the submission.",
                    finish_reason=reason,
                )
            return answer
        except (CompletionTruncated, EmptyCompletion):
            # Re-raised directly, for the reason truncation already is: wrapping these in
            # "gave up after 4 attempts, re-run to resume" gives the operator advice that cannot
            # work. Neither a ceiling that cut this attempt nor a body this reader cannot read
            # changes on the next three, and the caller's per-task quarantine keeps both out of
            # the submission, which is the only safe place for them.
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
    # The count is what was SPENT, not the budget. `main` copies this string verbatim into
    # `.failed.jsonl` and into the `task_failed` event, where `error_type` is `RuntimeError` for
    # every wrapped failure, so this sentence carries the only attempt information an operator
    # sees. Interpolating the budget priced every permanent failure at 4x what it cost, which is
    # backwards for errors that are classified permanent BECAUSE they cost one request.
    #
    # Pluralised, because "1 attempts" is the tell that a number is being interpolated into fixed
    # prose rather than described.
    raise RuntimeError(
        # `spent`, not `GENERATION_ATTEMPTS`. A permanent cause BREAKS at the first attempt, so
        # interpolating the budget priced every early break at 4x what it cost — and with
        # `cause_type` now recorded beside it, the row read `cause_type=BadRequestError` next to
        # "after 4 attempts" for a failure that billed once: two accurate fields and one wrong.
        f"generation gave up after {spent} attempt{'' if spent == 1 else 's'} "
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
    # Before the file is opened for APPEND, and only for the tasks about to be regenerated. An
    # artifact written before `generate_one` grew its guards can hold rows with an empty
    # `predictions` text; `already_done` no longer counts those as finished, so without this the
    # regenerated answer would land BESIDE the empty row rather than replace it.
    #
    # AHEAD of the failure log being cleared, so a run that refuses to proceed has not already
    # destroyed the previous run's record of what failed. Under the concurrent-run case this guard
    # exists for, that log belongs to the other run.
    try:
        repaired = drop_answerless_rows(args.out, {str(t["task_id"]) for t in pending})
    except (OSError, CheckpointChanged) as exc:
        # `os.replace` raises on Windows whenever another process holds the destination open, which
        # a tail, an editor or a virus scanner will do. Letting it escape killed an 842-task resume
        # with a raw traceback before a single task was attempted. The file is untouched at this
        # point, so refusing is the correct outcome: appending beside rows that should have been
        # replaced would produce the duplicate the repair exists to prevent.
        print(
            json.dumps({"event": "repair_failed", "output": str(args.out),
                        "error_type": type(exc).__name__, "error": str(exc), "note":
                        "the existing rows carrying no answer could not be removed, so nothing was "
                        "appended; the predictions file is unchanged and no task was attempted"}),
            flush=True,
        )
        return 1
    if repaired:
        print(
            json.dumps({"event": "repaired", "answerless_rows_dropped": repaired, "note":
                        "rows carrying no answer, from a run that predates the empty-completion "
                        "guards; they are regenerated below rather than left beside the new ones"}),
            flush=True,
        )
    # Independent of the repair: an unterminated last line glues the first appended row onto it,
    # and the overwhelmingly common resume repairs nothing at all.
    ensure_final_newline(args.out)
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
                # ⚠️ The ENCODE is inside the guard and the WRITE is not, and the line between them
                # is which failures belong to one task. The handle encodes strictly, so a lone
                # surrogate raises `UnicodeEncodeError` on the way out; `json.loads` accepts an
                # escaped one from any third-party file `load_generation_tasks` or
                # `load_recall_contexts` reads, and `submission_row` copies those fields straight
                # through, so one such row would otherwise raise out of `main` and abandon every
                # remaining task. That is a property of THIS task, so it fails this task.
                #
                # An `OSError` from the write or the flush is not. A full disk is a run-level
                # fault, and quarantining it would file a task as having no answer while its row
                # sits complete in the submission: `BufferedWriter` keeps the unwritten remainder
                # and emits it in order on the next successful flush, so the row lands anyway and
                # `.failed.jsonl`, `written` and the `incomplete` note would all be false about it.
                row = json.dumps(submission_row(task, contexts, answer), ensure_ascii=False)
                row.encode("utf-8")
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
                # ⚠️ `cause_type` beside `error_type`, because `generate_one` WRAPS every retried
                # failure in a `RuntimeError`. That makes `error_type` the word "RuntimeError" for
                # essentially every generation failure, which defeats the comment above ("the type
                # is recorded so a systematic fault is legible") for the whole class. The typed
                # cause is what tells an empty `choices` array apart from a dead key, and it was
                # otherwise present only inside the free-text `error` string, where nothing can
                # count it.
                cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else None
                with failures_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"task_id": task["task_id"],
                                         "error_type": type(exc).__name__,
                                         "cause_type": cause,
                                         "error": str(exc)}) + "\n")
                print(
                    json.dumps({"event": "task_failed", "task_id": task["task_id"],
                                "error_type": type(exc).__name__, "cause_type": cause,
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
            handle.write(row + "\n")
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
    fragments = unparsable_rows(args.out)
    if fragments:
        print(
            json.dumps({"event": "unparsable_rows", "count": fragments, "output": str(args.out),
                        "note": "line(s) that are not readable JSON, which an interrupted write "
                                "leaves behind. The official format checker rejects the file. The "
                                "answers around them are intact: delete those line(s) and re-run, "
                                "which regenerates only what they were holding"}),
            flush=True,
        )
    if failed:
        print(
            json.dumps({"event": "incomplete", "note":
                        f"{len(failed)} of {len(pending)} task(s) have no answer, so this file is "
                        f"NOT submittable. Re-run the same command to retry only those; the "
                        f"completed ones are skipped."}),
            flush=True,
        )
    return 1 if (failed or fragments) else 0


if __name__ == "__main__":
    sys.exit(main())
