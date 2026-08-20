"""Run the RE call ATM retrieve, rerank, answer pipeline.

This driver is deliberately separate from ``atm_bench.py``.  That file measures retrieval only.
This file adds bounded answer generation and writes checkpointed artifacts which the official ATM
evaluator can consume without transforming the predictions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
import re
import sys
import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

import requests

from benchmarks.atm_bench import build_memory_items, git_revision, load_questions, sha256
from recall.embeddings import (
    VoyageEmbedder,
    embed_passages,
    embedding_profile_id,
    resolve_embedder,
)
from recall.rerank import reranker_from_name
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import Chunk


DEFAULT_DSN = "postgresql://recall:recall@localhost:5432/recall"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDER = "voyage:voyage-4-large"
DEFAULT_RERANKER = "voyage:rerank-2.5"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_EVIDENCE_CHARS = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CANDIDATE_K = 25
DEFAULT_RETRIEVAL_K = 10
DEFAULT_EVIDENCE_FLOOR = 100
DEFAULT_ANSWER_POLICY = "baseline"
EVIDENCE_PACKERS = ("greedy", "allocated")
DEFAULT_EVIDENCE_PACKER = "allocated"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_ANSWER_WORKERS = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"invalid JSONL row at {path}:{line_number}")
            rows[str(row["id"])] = row
    return rows


def _compact_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _truncate_on_boundary(part: str, limit: int) -> str:
    """Trim to `limit`, preferring the last word boundary so a fact is not cut mid-token.

    Tail-first is deliberate and is not a detail: `format_media` emits ID, Type, Timestamp and
    Location as its first four fields, so trimming from the end costs Tags and OCR and preserves
    the header that identifies WHICH memory this is. Cutting from the front would destroy exactly
    the fields that let a reader tell two similar photos apart.
    """
    if limit <= 0:
        return ""
    if len(part) <= limit:
        return part
    cut = part[:limit]
    boundary = cut.rfind(" ")
    # Only honour the boundary when it keeps most of the allowance. A very early space would trade
    # a mid-token cut for throwing away the field the allowance was bought for.
    if boundary >= limit // 2:
        return cut[:boundary]
    return cut


def _allocate(lengths: list[int], budget: int, *, floor: int) -> list[int]:
    """Spend `budget` greedily in rank order, but never below a reserve for the items still to come.

    The greedy packer this replaces spent in rank order and stopped. Measured on the 1,013 saved
    retrievals of the Voyage run, that dropped a retrieved GOLD item entirely on 61 questions,
    overran the stated budget on 757, and left 3 blocks whose identifier was cut mid-string.

    ⚠️ The first design here shared the budget by weighted water filling, and its own replay
    falsified it: whole items fell from 6.13 to 4.09 of 10 and the top ranked item lost text on
    252 of 1,013 questions. Spreading a fixed budget evenly buys visibility for rank 10 with text
    from rank 1, and rank 1 is the likeliest to hold the answer. That is why this is rank-greedy
    with a lookahead reserve rather than a fair share: the floor buys each later item enough to be
    IDENTIFIED, since `format_media` leads with ID, Type, Timestamp and Location, and everything
    above the floor still goes to the top of the ranking.

    The floor is 100 characters because that is where a rendering property has its knee, not where
    an outcome metric peaked: measured over 4,000 rendered blocks from the saved retrieval, a usable
    `Timestamp` survives in 0% of blocks at 60 characters and in 100% at 100. The timestamp is what
    tells two near identical photos apart, and telling them apart is the largest measured reader
    failure on this benchmark. Raising the floor past 100 guarantees no further FIELD, it only buys
    prose for the tail with text taken from the top: at 260 the number of gold items downgraded from
    whole to trimmed rises from 21 to 61.
    """
    n = len(lengths)
    if n == 0:
        return []
    # A floor that cannot fit every item is not a floor. Shrink it rather than dropping the tail,
    # which is the behaviour being removed.
    floor = max(0, min(floor, budget // n))
    allowance = [0] * n
    remaining = budget
    for i, length in enumerate(lengths):
        # What the items after this one must be left in order to stay identifiable.
        reserve = sum(min(floor, later) for later in lengths[i + 1 :])
        take = min(length, max(0, remaining - reserve))
        if take < min(floor, length):
            take = min(floor, length, remaining)
        allowance[i] = take
        remaining -= take
    # Anything left over goes back up the ranking, to the highest ranked item still trimmed.
    for i, length in enumerate(lengths):
        if remaining <= 0:
            break
        give = min(remaining, length - allowance[i])
        allowance[i] += give
        remaining -= give
    return allowance


def _greedy_evidence_text(hits: list[dict[str, Any]], max_chars: int) -> str:
    """The packer that produced the 68.92 answer file, kept runnable so it stays a real control.

    ⛔ Do not fix its defects. They are the measurement: it hides a retrieved gold item on 61 of
    1,013 questions, overruns `max_chars` by up to 2*(n-1) because it never counts the separators,
    and can cut the final block mid-identifier. Repairing them here would leave the experiment with
    no baseline to compare against, and every earlier record in this repository was produced by
    exactly this code.
    """
    parts: list[str] = []
    used = 0
    for hit in hits:
        part = f"[{str(hit['id'])}] {_compact_text(hit.get('text'))}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(part) > remaining:
            part = part[:remaining]
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


def _evidence_text(
    hits: list[dict[str, Any]],
    max_chars: int,
    *,
    floor: int = DEFAULT_EVIDENCE_FLOOR,
    packer: str = DEFAULT_EVIDENCE_PACKER,
) -> str:
    if packer not in EVIDENCE_PACKERS:
        raise ValueError(f"unknown evidence packer: {packer}")
    if packer == "greedy":
        return _greedy_evidence_text(hits, max_chars)
    rendered = [f"[{str(hit['id'])}] {_compact_text(hit.get('text'))}" for hit in hits]
    if not rendered:
        return ""
    # The separators are part of what the reader receives, so they come out of the budget rather
    # than being spent on top of it. The greedy packer ignored them and overran by 2*(n-1).
    budget = max(0, max_chars - 2 * (len(rendered) - 1))
    if sum(len(part) for part in rendered) <= budget:
        return "\n\n".join(rendered)
    allowance = _allocate([len(part) for part in rendered], budget, floor=floor)
    trimmed = [
        _truncate_on_boundary(part, size) for part, size in zip(rendered, allowance)
    ]
    # An item allocated nothing must not appear as an empty block. Beyond being noise, joining
    # empties still pays for their separators, and at a budget smaller than the separators
    # themselves that alone overran `max_chars` with nothing but blank lines.
    return "\n\n".join(part for part in trimmed if part)


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces).strip()
    return ""


class CompletionTruncated(RuntimeError):
    """A completion stopped because it hit `max_tokens` rather than finishing.

    Its own type, and excluded from the retry loop, because the request is byte identical on every
    attempt: retrying buys the same refusal at four times the price. `benchmarks/llm.py` records
    the same conclusion for the other benchmark harness.

    This matters more under `--answer-policy selection`, whose envelope adds a few hundred output
    tokens on top of the reasoning tokens OpenRouter already counts inside the ceiling. The default
    ceiling here is 1,024 while the last completed full run needed 8,192, so the selection arm will
    meet this error immediately unless the ceiling is raised with it. The fix is a bigger ceiling,
    not another attempt, and the message says so.
    """


def _retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


#: Mirrors `ABSTENTION_PHRASES` in the official evaluator's `config.py`, plus the wording this
#: runner's own earlier prompt used, which matches none of the official phrases.
#:
#: ⛔ This list is for MEASUREMENT only. It exists so a run reports its own refusal rate, and it
#: must never reach the prompt or gate an answer: feeding the scorer's vocabulary back into the
#: system under test is tuning to the metric, which is a different thing from reporting against it.
_REFUSAL_MARKERS = (
    "unknown",
    "abstention",
    "no information",
    "not available",
    "no evidence",
    "insufficient information",
    # This runner's earlier prompt asked for exactly this wording, and it matches none of the
    # official phrases. A bare "does not contain" is deliberately NOT here: it would mark
    # "the email does not contain a price, but the total is £50" as a refusal, and over-counting
    # refusals corrupts the one number these arms must be judged against.
    "not contain enough",
)


def is_refusal(answer: str) -> bool:
    """Whether an answer declines to answer, for the refusal rate every arm must report.

    Measured on the 1,013 answer baseline, 58 answers refused while scoring below 0.5 with the
    complete gold evidence retrieved, and 35 of those had the answer fully on screen. An arm that
    trades those away for something else is invisible in the total, so the rate is carried beside
    it rather than derived later.
    """
    lowered = " ".join(answer.lower().split())
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


#: The prompt from the official ATM oracle baseline, `memqa/qa_agent_baselines/oracle/config.py`.
#: Kept verbatim so `baseline` is a real control rather than our paraphrase of one.
BASELINE_SYSTEM = (
    "You are a QA assistant. Use ONLY the provided evidence to answer. "
    "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
    "If the question asks to recall or list items (photos/emails/videos), respond with the "
    "corresponding evidence IDs only, comma-separated, with no extra text."
)

#: H17. Measured on the 1,013 answer baseline: 58 answers refused while scoring below 0.5 with the
#: complete gold evidence retrieved, and on 35 of them the answer was fully inside the evidence
#: block the model had been given. Refusing is not free here, it scores zero exactly like a wrong
#: answer, so the disposition is deliberately asymmetric.
#:
#: ⚠️ It is bounded on the other side by 23 gold-abstention questions currently scoring 0.7391,
#: which is 1.68 QS this sentence can destroy. Never report an arm carrying it without the refusal
#: rate and the gold-abstention score beside the total.
DISPOSITION_SYSTEM = (
    " The evidence block is the entire memory available to you, so there is nothing further to "
    "retrieve. Answer whenever the evidence supports an answer, including when it supports only "
    "part of what was asked, and give the part it supports. Reserve 'Unknown' for when no item in "
    "the block bears on the question at all."
)

#: H21. Evidence first, answer second, in ONE call. Measured target: 47 questions where the reader
#: answered from a real but wrong item that was on screen beside the right one, including a
#: question that named a city by name AND country, answered with the same-named city in the other
#: country, while items for both were present.
#:
#: The per-item marks are diagnostics, never a gate. Nothing in the runner drops an item the model
#: marked `no`, because a filter that drops the gold item converts a wrong answer into a refusal,
#: and refusals were measured to cost more than wrong answers on this benchmark.
SELECTION_SYSTEM = (
    " Work in two steps and return a single JSON object, with no prose outside it.\n"
    'First, before reading the evidence, list the question\'s qualifiers in "qualifiers": every '
    "constraint an item must satisfy to be the one asked about, such as the place, the date, the "
    "person, the event, or the object.\n"
    'Second, in "items", give one entry per evidence item in the order shown, each with its "id", '
    'a "matches" value of "yes", "no" or "partial", and "failing_qualifier" naming the first '
    "qualifier it fails, or null.\n"
    'Then answer in "answer", using only the items marked "yes", or the "partial" ones when no '
    'item is "yes". The "answer" value must follow the answer rules above exactly: the answer and '
    "nothing else, 'Unknown' when the evidence is insufficient, and bare comma separated evidence "
    "IDs for a recall or list question.\n"
    'Return exactly: {"qualifiers": [...], "items": [...], "answer": "..."}'
)

#: H-coverage. The two halves of ATM are scored by opposite rules and one prompt has been applying
#: one of them to both. `number` is exact multiset equality over extracted values, so every extra
#: value fails an otherwise correct answer, and brevity is right. `open_end` goes to a judge whose
#: published rubric marks accuracy true when the ground truth is COVERED and permits additional
#: information, so brevity costs points for every element of the answer it drops.
#:
#: Measured on the 300 question subset, arm B, 118 `open_end` questions: answers whose gold content
#: was fully covered scored 98.04, partial ones scored 43.28, and 17 of the 67 partial answers were
#: missing exactly ONE content token. The official prompt cut these answers from 53 to 17 median
#: characters and coverage from 0.701 to 0.640.
#:
#: ⚠️ This wording names only the QUESTION and the evidence. It does not mention the rubric, quote
#: its criterion, or carry any phrase list, because the line between "answer well" and "answer the
#: way this judge scores" is thinner here than anywhere else in this study.
COVERAGE_SYSTEM = (
    " The brevity rule above is for a question with a single value answer, and for a recall or list "
    "question. When the question instead asks for a description, an explanation, a reason, or "
    "several attributes of one thing, give the most complete answer the evidence supports rather "
    "than the shortest phrase that identifies it: include every detail the question asks about that "
    "the evidence provides, and never drop a qualifier, a place or a time the question asked for."
)

ANSWER_POLICIES = ("baseline", "disposition", "selection", "both", "coverage")


def system_prompt(answer_policy: str) -> str:
    """Compose the system prompt for one preregistered arm.

    The arms are additive on purpose: `both` is `disposition` plus `selection` over the same
    baseline text, so a difference between arms is attributable to the sentences that differ.
    """
    if answer_policy not in ANSWER_POLICIES:
        raise ValueError(f"unknown answer policy: {answer_policy}")
    system = BASELINE_SYSTEM
    if answer_policy in ("disposition", "both"):
        system += DISPOSITION_SYSTEM
    if answer_policy in ("selection", "both"):
        system += SELECTION_SYSTEM
    if answer_policy == "coverage":
        system += COVERAGE_SYSTEM
    return system


def parse_selection(text: str) -> tuple[str, dict[str, Any]]:
    """Read the H21 envelope, and fail CLOSED to the raw completion when it is not there.

    Failing closed rather than raising is the whole safety property of this arm: a malformed
    envelope costs the diagnostics for one question, while raising would drop an answer the model
    had already been paid for. `parse_failed` is recorded so the rate can be reported, because an
    arm whose envelope fails often is not the arm that was preregistered.
    """
    def _give_up(reason: str) -> tuple[str, dict[str, Any]]:
        """Rescue the answer field by pattern before surrendering the whole completion.

        ⛔ Surrendering is worse here than it looks, and worse for this arm than for any other.
        The raw completion of a failed envelope is a JSON blob listing every evidence id in
        `items`, and the official scorer harvests ids out of free text for a `list_recall`
        question, so submitting the blob hands it a ten item prediction against a gold set that is
        a singleton in 100 of 139 cases. A truncated closing brace would turn a good answer into
        the worst possible one, which is why the cheap rescue runs first.
        """
        rescued = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if rescued:
            value = rescued.group(1).strip()
            if value:
                return value, {"parse_failed": True, "reason": reason, "rescued_answer": True}
        return text, {"parse_failed": True, "reason": reason, "rescued_answer": False}

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return _give_up("no JSON object in the completion")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return _give_up(f"invalid JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return _give_up("JSON was not an object")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return _give_up("no usable answer field")
    items = payload.get("items")
    marks: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                marks.append(
                    {
                        "id": str(item.get("id", "")),
                        "matches": str(item.get("matches", "")),
                        "failing_qualifier": item.get("failing_qualifier"),
                    }
                )
    qualifiers = payload.get("qualifiers")
    return answer.strip(), {
        "parse_failed": False,
        "qualifiers": [str(q) for q in qualifiers] if isinstance(qualifiers, list) else [],
        "items": marks,
        "matched": sum(1 for mark in marks if mark["matches"] == "yes"),
        "rejected": sum(1 for mark in marks if mark["matches"] == "no"),
    }


def generate_answer(
    *,
    question: str,
    qtype: str | None,
    evidence: str,
    model: str,
    base_url: str,
    api_key: str,
    reasoning_effort: str,
    max_output_tokens: int,
    max_attempts: int,
    answer_policy: str = "baseline",
) -> tuple[str, dict[str, int], str | None, dict[str, Any]]:
    system = system_prompt(answer_policy)
    user = (
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Provide the answer based solely on the evidence."
    )
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            if _retryable_status(response.status_code):
                last_error = f"provider status {response.status_code}"
                if attempt + 1 < max_attempts:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("provider returned no choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise RuntimeError("provider returned an invalid choice")
            if choice.get("finish_reason") == "length":
                raise CompletionTruncated(
                    f"the answer reached the {max_output_tokens} token output ceiling. "
                    "Raise --max-output-tokens; retrying sends the identical request."
                )
            message = choice.get("message", {})
            answer = _message_text(message.get("content") if isinstance(message, dict) else None)
            if not answer:
                raise RuntimeError("provider returned an empty answer")
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            usage_row = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
            returned_model = body.get("model") if isinstance(body, dict) else None
            if answer_policy in ("selection", "both"):
                answer, selection = parse_selection(answer)
            else:
                selection = {"parse_failed": False}
            selection["answer_policy"] = answer_policy
            return (
                answer,
                usage_row,
                str(returned_model) if returned_model else None,
                selection,
            )
        except CompletionTruncated:
            # Deliberately not retried: see the class docstring. Re-raised as-is so the operator
            # reads the ceiling and the remedy rather than an attempt count.
            raise
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = str(exc).splitlines()[0][:240]
            if attempt + 1 < max_attempts:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise RuntimeError(f"answer generation failed after {max_attempts} attempts: {last_error}") from exc
    raise RuntimeError(f"answer generation failed: {last_error or 'unknown provider error'}")


@contextmanager
def _no_index() -> Iterator[tuple[Any, Any, list[Chunk], int]]:
    """Stand in for the index when every question already has a checkpointed retrieval row.

    Yields `None` for the retriever deliberately rather than a stub that would answer a query: if
    the run ever does reach for it, the failure should be an immediate AttributeError naming the
    line, not a silent empty result set that looks like a corpus with nothing in it.
    """
    yield None, None, [], 0


@contextmanager
def _build_retriever(args: argparse.Namespace) -> Iterator[tuple[Any, Any, list[Chunk], int]]:
    memory = build_memory_items(args.image_file, args.video_file, args.email_file)
    if args.embedder == "voyage" or args.embedder.startswith("voyage:"):
        voyage_model = args.embedder[len("voyage:") :] if args.embedder.startswith("voyage:") else "voyage-3"
        embedder = VoyageEmbedder(model=voyage_model, batch_size=args.embedding_batch_size)
    else:
        embedder = resolve_embedder(args.embedder)
    chunks = [
        Chunk(
            id=evidence_id,
            source=evidence_id,
            text=text,
            metadata={**metadata, "evidence_id": evidence_id, "modality": modality},
        )
        for evidence_id, modality, text, metadata in memory
    ]
    dsn = args.dsn or os.environ.get("RECALL_DSN") or DEFAULT_DSN
    reranker = reranker_from_name(args.reranker)
    index_started = time.perf_counter()
    embeddings: list[list[float]] = []
    store = PgVectorStore(dsn, dim=embedder.dim, table=args.table, tenant=args.tenant)
    try:
        store.ensure_schema()
        if args.reuse_index:
            facts = store.readiness_facts()
            if int(facts["rows"]) != len(chunks):
                raise ValueError(
                    f"reused ATM index has {facts['rows']} rows, expected {len(chunks)}"
                )
        else:
            embeddings = embed_passages(embedder, [chunk.text for chunk in chunks])
            store.upsert(chunks, embeddings)
            store.analyze()
        index_ms = int((time.perf_counter() - index_started) * 1000)
        retriever = HybridRetriever(
            store,
            embedder,
            reranker=reranker,
            candidate_k=args.candidate_k,
            gap_threshold=args.gap_threshold,
            use_dense=True,
            use_sparse=True,
            sparse_backend="lexical",
            retrieval_profile="atm_voyage4_lexical_hybrid",
            index_generation="atm_2026_08_19_voyage4",
        )
        yield retriever, embedder, chunks, index_ms
    finally:
        store.close()


def run(args: argparse.Namespace) -> int:
    questions = load_questions(args.qa_file)
    memory = build_memory_items(args.image_file, args.video_file, args.email_file)
    memory_ids = {item[0] for item in memory}
    if args.dry_run:
        print(json.dumps({
            "questions": len(questions),
            "memory_items": len(memory),
            "embedder": args.embedder,
            "reranker": args.reranker,
            "candidate_k": args.candidate_k,
            "embedding_batch_size": args.embedding_batch_size,
            "retrieval_k": args.retrieval_k,
            "answer_model": args.answer_model,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "answer_workers": args.answer_workers,
            "answer_policy": args.answer_policy,
            "evidence_packer": args.evidence_packer,
            "evidence_chars": args.evidence_chars,
            "official_judge": "gpt-5-mini",
        }, indent=2))
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for answer generation")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.out_dir / "answers.jsonl"
    retrieval_path = args.out_dir / "retrieval.jsonl"
    # Kept out of `answers.jsonl`, which the official evaluator reads and which must stay exactly
    # the `{"id", "answer"}` rows it expects.
    diagnostics_path = args.out_dir / "diagnostics.jsonl"
    answers = _load_jsonl(answers_path)
    retrieval_rows = _load_jsonl(retrieval_path)
    previous_manifest: dict[str, Any] = {}
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded_manifest, dict):
            previous_manifest = loaded_manifest
    # Only stand up the index when a question actually needs retrieving. An answer-arm comparison
    # seeds every arm's directory with one completed run's checkpoints so all arms read
    # byte-identical evidence, and in that shape the store, the Voyage embedder and the reranker
    # are all built, connected to, and never used.
    #
    # This is not only waste. It couples a replay to the schema version of a SHARED database: the
    # first attempt at this comparison died on `SchemaTooNew` for migration 0014, applied to that
    # table by a newer branch, on a run that was never going to issue a query.
    needs_index = any(
        question["id"] not in answers and question["id"] not in retrieval_rows
        for question in questions
    )
    index_context = _build_retriever(args) if needs_index else _no_index()
    with index_context as (retriever, embedder, chunks, index_ms):
        previous_usage = previous_manifest.get("usage", {})
        usage_total = {
            key: int(previous_usage.get(key, 0) or 0)
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        returned_models: set[str] = set(previous_manifest.get("answer_models_returned", []))
        errors: list[dict[str, str]] = []
        # Counted over the questions THIS invocation answers, and the manifest says so. A resumed
        # run answers a subset, so a rate over `len(questions)` would silently shrink each resume.
        refusals = 0
        parse_failures = 0
        truncated = 0
        pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for position, question in enumerate(questions, start=1):
            question_id = question["id"]
            if question_id in answers:
                continue
            row = retrieval_rows.get(question_id)
            if row is None:
                result = retriever.search(question["question"], k=args.retrieval_k)
                hits = [
                    {
                        "id": hit.chunk.id,
                        "text": hit.chunk.text,
                        "score": float(hit.score),
                    }
                    for hit in result.hits
                ]
                row = {
                    "id": question_id,
                    "question": question["question"],
                    "qtype": question.get("qtype"),
                    "retrieval_ids": [hit["id"] for hit in hits],
                    "hits": hits,
                    "gap_warning": bool(result.gap_warning),
                    "reranking_ran": bool(result.diagnostics.reranking_ran),
                }
                if any(hit_id not in memory_ids for hit_id in row["retrieval_ids"]):
                    raise RuntimeError(f"retrieval returned an unknown memory id for {question_id}")
                _append_jsonl(retrieval_path, row)
                retrieval_rows[question_id] = row
            pending.append((position, question, row))

        def answer_one(
            item: tuple[int, dict[str, Any], dict[str, Any]],
        ) -> tuple[int, str, dict[str, Any], dict[str, int], str | None, dict[str, Any]]:
            position, question, row = item
            evidence = _evidence_text(
                row["hits"], args.evidence_chars, packer=args.evidence_packer
            )
            try:
                answer, usage, returned_model, selection = generate_answer(
                    question=question["question"],
                    qtype=question.get("qtype"),
                    evidence=evidence,
                    model=args.answer_model,
                    base_url=args.answer_base_url,
                    api_key=api_key,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                    max_attempts=args.max_attempts,
                    answer_policy=args.answer_policy,
                )
            except CompletionTruncated as exc:
                # One question that will not terminate must not cost the other 299, and it must
                # not cost the two arms queued behind this one either: arm C died on its LAST
                # question and took arm D with it.
                #
                # ⛔ Raising the ceiling again is the move this project has already made three
                # times, and the pre-registration registered a falsifier against making it a
                # fourth. The measured distribution says why: median 349 completion tokens, p99
                # 3,534, and a question that wants more than 16,384. A 47x gap between the middle
                # and the tail is a reasoning loop that does not terminate, not a long answer, and
                # no ceiling catches that.
                #
                # The question is recorded as unanswered WITH ITS REASON rather than skipped. No
                # truncated text enters answers.jsonl, so nothing half-written is ever scored, and
                # the paired comparison drops the question from every arm symmetrically.
                return (
                    position,
                    question["id"],
                    None,
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    None,
                    {
                        "id": question["id"],
                        "qtype": question.get("qtype"),
                        "truncated": True,
                        "reason": str(exc),
                        "evidence_chars": len(evidence),
                        "answer_policy": args.answer_policy,
                    },
                )
            # Everything needed to score an arm WITHOUT reparsing the answers, and in particular
            # the refusal rate, which no arm here may be reported without.
            diagnostics = {
                "id": question["id"],
                "qtype": question.get("qtype"),
                "refused": is_refusal(answer),
                "answer_chars": len(answer),
                # Per question, not just in the manifest total, because the output ceiling has now
                # bound four times in this project (1,024, 2,048, 4,096, 8,192) and the only way to
                # know whether a ceiling DISTORTED an arm rather than merely killing it is to see
                # how many completions came close to it. Arm A averaged 466 completion tokens and
                # still had a sibling arm die at 8,192, so the mean says nothing about the tail.
                "completion_tokens": usage.get("completion_tokens"),
                "evidence_chars": len(evidence),
                "items_presented": evidence.count("\n\n") + 1 if evidence else 0,
                "items_retrieved": len(row["hits"]),
                **selection,
            }
            return (
                position,
                question["id"],
                {"id": question["id"], "answer": answer},
                usage,
                returned_model,
                diagnostics,
            )

        completed = len(answers)
        with ThreadPoolExecutor(max_workers=args.answer_workers) as executor:
            futures = [executor.submit(answer_one, item) for item in pending]
            for future in as_completed(futures):
                (
                    position,
                    question_id,
                    answer_row,
                    usage,
                    returned_model,
                    diagnostics,
                ) = future.result()
                _append_jsonl(diagnostics_path, diagnostics)
                if answer_row is None:
                    # Recorded in diagnostics, deliberately absent from answers.jsonl. A resume
                    # will retry it, which is right: it is a question this configuration could not
                    # answer, not a question that was answered badly.
                    truncated += 1
                    print(
                        f"TRUNCATED question_position={position}, recorded unanswered "
                        f"({truncated} so far)",
                        flush=True,
                    )
                    continue
                _append_jsonl(answers_path, answer_row)
                answers[question_id] = answer_row
                usage_total["calls"] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage_total[key] += usage[key]
                if returned_model:
                    returned_models.add(returned_model)
                refusals += int(bool(diagnostics["refused"]))
                parse_failures += int(bool(diagnostics.get("parse_failed")))
                completed += 1
                print(f"answered {completed}/{len(questions)} question_position={position}", flush=True)

        manifest = {
        "benchmark": "ATM-Bench",
        # ⛔ Provenance, not decoration. "replayed" means retrieval was NOT rerun: every hit came
        # from a checkpoint written by an earlier run, so this artifact says nothing about the
        # retriever and must never be read as a retrieval measurement.
        "measurement": "retrieve_rerank_answer" if needs_index else "answer_only_replayed",
        "retrieval_source": "live" if needs_index else "seeded_checkpoints",
        "question_count": len(questions),
        "answer_count": len(answers),
        "corpus_items": len(chunks) if needs_index else None,
        "embedder": args.embedder if needs_index else None,
        "embedding_profile": embedding_profile_id(embedder) if needs_index else None,
        "reranker": args.reranker if needs_index else None,
        "sparse_backend": "lexical" if needs_index else None,
        "candidate_k": args.candidate_k,
        "embedding_batch_size": args.embedding_batch_size,
        "retrieval_k": args.retrieval_k,
        "evidence_chars": args.evidence_chars,
        "answer_model_requested": args.answer_model,
        "answer_models_returned": sorted(returned_models),
        "answer_base_url": args.answer_base_url,
        "answer_workers": args.answer_workers,
        "answer_policy": args.answer_policy,
        "evidence_packer": args.evidence_packer,
        # ⛔ Read this beside the score, never after it. 58 baseline refusals scored zero with the
        # complete gold retrieved and 35 had the answer on screen, while 23 gold-abstention
        # questions score 0.7391, so an arm can buy points on one side and pay them on the other
        # with no sign of it in the total.
        "answers_this_invocation": len(pending),
        "refusals_this_invocation": refusals,
        "refusal_rate_this_invocation": (refusals / len(pending)) if pending else None,
        "selection_parse_failures": parse_failures,
        # Questions this configuration could not finish. They are absent from answers.jsonl on
        # purpose, so a reader who counts rows sees the shortfall instead of a silent 300.
        "truncated_questions": truncated,
        "reasoning": {
            "enabled": True,
            "requested_effort": args.reasoning_effort,
            "effective_deepseek_effort": "high" if args.reasoning_effort == "medium" else None,
        },
        "max_output_tokens": args.max_output_tokens,
        "usage": usage_total,
        "errors": errors,
        "index_ms": index_ms,
        "table": args.table,
        "tenant": args.tenant,
        "git_revision": git_revision(),
        "data_sha256": {
            str(path): sha256(path)
            for path in (args.qa_file, args.image_file, args.video_file, args.email_file)
        },
        }
        _write_json(args.out_dir / "manifest.json", manifest)
    print(f"wrote {answers_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", type=Path, required=True)
    ap.add_argument("--image-file", type=Path, required=True)
    ap.add_argument("--video-file", type=Path, required=True)
    ap.add_argument("--email-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dsn")
    ap.add_argument("--table", default="bench_atm_voyage4_chunks")
    ap.add_argument("--tenant", default="atm-bench-voyage4-20260819")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    ap.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    ap.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    ap.add_argument("--gap-threshold", type=float, default=0.50)
    ap.add_argument("--answer-base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--answer-model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="medium")
    ap.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    ap.add_argument("--evidence-chars", type=int, default=DEFAULT_EVIDENCE_CHARS)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--answer-workers", type=_positive_int, default=DEFAULT_ANSWER_WORKERS)
    ap.add_argument(
        "--evidence-packer",
        choices=EVIDENCE_PACKERS,
        default=DEFAULT_EVIDENCE_PACKER,
        help=(
            "greedy is the packer that produced the 68.92 answer file, kept runnable as the H7 "
            "control; allocated reserves every retrieved item enough to stay identifiable"
        ),
    )
    ap.add_argument(
        "--answer-policy",
        choices=ANSWER_POLICIES,
        default=DEFAULT_ANSWER_POLICY,
        help=(
            "which preregistered answer arm to run. baseline is the official oracle prompt "
            "verbatim and is the control; disposition adds H17; selection adds H21; both adds "
            "each. See docs/preregistrations/2026-08-20-atm-evidence-allocation-and-selection.md"
        ),
    )
    ap.add_argument("--reuse-index", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
