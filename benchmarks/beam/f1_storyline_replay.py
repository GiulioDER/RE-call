"""Replay C9's stored BEAM retrieval with an Add-time gpt-4o-mini storyline for summary questions.

Pre-registration: ``docs/preregistrations/2026-09-25-f1-storyline-replay.md``.

Why: on public BEAM 100K through the served C9, summarization scored 0.304 while its rubric
evidence was in the returned 100 items for 0.950 of the points (``2026-09-24-c9-beam-ability-
probe.md``). AML's reader answers "direct and concise" in 512 tokens and leans on the top of the
context, so a relevance-ordered heap of fragments yields a third of a multi-phase summary. AML
forbids Search from generating answers ("Search must not generate final answers or disguise
answers as memory records"), so any LLM synthesis has to happen at Add, where the rules expect
gpt-4o-mini anyway. This harness asks whether such an Add-time synthesis, placed on top of the
SAME stored retrieval only for questions that ask for a summary, moves the rubric score.

Nothing here touches C9. The Add stream is simulated offline from the BEAM input, cut the way
AML's Textual adapter cuts it (a chunk closes at 20 messages or 2,000 words, whichever comes
first, and never crosses a session), and each chunk is one gpt-4o-mini call that returns a digest
of the chunk and the user's updated storyline. Answering and judging are the probe's own
functions, so everything AML-side is byte-identical to the probe.

Phases, each resumable, all writing under ``--out``:

    build -> gate -> answer --arm A -> judge --arm A -> storycover -> report

Arms: ``r0`` re-answers the stored items unchanged (a concurrent replicate of the probe's
``returned``); ``story``, ``digests`` and ``story_digests`` put the synthesis on top for gated
questions and reuse ``r0``'s answer for every other question (their context is byte-identical);
``story_all`` puts the storyline on top of every question, ungated, as an exploratory check of
what the gate protects. Every arm returns at most 100 items, the platform's ``top_k``.

The model key comes from ``OPENROUTER_API_KEY``. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any

from benchmarks.beam.aml_c9_probe import (
    ANSWER_PROMPT,
    Appender,
    Spend,
    TOP_K,
    complete,
    context_of,
    http_json,
    judge as probe_judge,
    pool,
    read_jsonl,
    COVERAGE_PROMPT,
    _judged,
)

BUILDER_MODEL = "openai/gpt-4o-mini"
#: USD per token for BUILDER_MODEL on OpenRouter (OpenAI list price, same on Azure).
BUILDER_PRICE_IN = 0.15e-6
BUILDER_PRICE_OUT = 0.60e-6
#: AML's Textual adapter freezes a chunk at the first of these (API guide, read 2026-09-18).
CHUNK_MESSAGES = 20
CHUNK_WORDS = 2_000
STORYLINE_WORDS = 600
DIGEST_WORDS = 80
MAX_DIGESTS = 24
#: Amendment 1 (2026-09-25, before any outcome was read): the output cap was 1,400 tokens and cut
#: the JSON of long storylines; the length bound is now enforced by a compression call.
BUILDER_MAX_TOKENS = 2_600
COMPRESS_ABOVE_WORDS = 700
COMPRESS_TARGET_WORDS = 450
#: Amendment 3: 1,400 truncated compressions that did not compress (253 of 338 stayed over 700).
COMPRESS_MAX_TOKENS = 2_600
GATED_ARMS = ("story", "digests", "story_digests")
ARMS = ("r0", *GATED_ARMS, "story_all")

#: Frozen with the pre-registration. English summary intent plus the Chinese words for summarise,
#: summary, review and sort out, because AML sends the question in its original language.
GATE = re.compile(
    r"(?i)\b(?:summar(?:y|ies|ize|ise|izing|ising|ized|ised)|overview|recap|sum(?:ming)? up|"
    r"comprehensive (?:account|review|rundown)|walk me through (?:the )?(?:whole|entire))\b"
    r"|总结|概括|摘要|回顾|梳理"
)

BUILDER_SYSTEM = f"""You maintain the long-term memory of one user's conversations with an assistant.
You receive the CURRENT STORYLINE (possibly empty) and one NEW EXCERPT of the conversation, with
its date. Return a JSON object with exactly two string fields.

"digest": at most {DIGEST_WORDS} words about the new excerpt only: what the user was working on or
dealing with, what was decided or recommended and why, the named specifics (tools, versions,
numbers, names, places, dates), and any outcome or open issue.

"storyline": the updated storyline of the whole history so far, at most {STORYLINE_WORDS} words.
It is a chronological account in dated phases, oldest first, one short paragraph per phase, each
beginning with its date in square brackets. Each phase states the goal, the key decisions and the
reasons for them, named specifics, problems met and how they were solved, and results. Merge the
new excerpt into the phase it continues, or add a new phase. When the storyline would exceed
{STORYLINE_WORDS} words, compress the oldest phases first: keep decisions, named specifics and
outcomes, drop small talk and generic advice, and never drop a phase entirely. If the user has
several unrelated projects or topics, keep each as its own thread of phases, labelled by topic.

Use only information in the storyline and the excerpt. Do not invent anything. Write in the
language of the conversation."""

COMPRESS_SYSTEM = f"""You compress the long-term storyline of one user's conversations with an assistant.
Rewrite the storyline you receive to at most {COMPRESS_TARGET_WORDS} words. Keep it chronological in
dated phases, oldest first, each beginning with its date in square brackets. Keep every phase (at
least one sentence each), and within each phase keep the decisions, the reasons for them, named
specifics (tools, versions, numbers, names, places, dates) and outcomes; drop small talk, generic
advice and repetition. Keep topic labels if the storyline has several threads. Use only the
information in the storyline. Return a JSON object with one string field, "storyline"."""


# ------------------------------------------------------------------------------------------
# The simulated Add stream
# ------------------------------------------------------------------------------------------


def aml_chunks(conversation: dict) -> list[dict[str, Any]]:
    """Cut each session the way AML's Textual adapter does: 20 messages or 2,000 words.

    A chunk closes BEFORE the message that would take it past either limit, so a single message
    longer than 2,000 words is a chunk of its own rather than being split. Chunks never cross a
    session (a BEAM batch), because AML adds one source session at a time.
    """
    chunks: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(conversation["batches"]):
        current: list[dict] = []
        words = 0
        for message in batch["messages"]:
            size = len(str(message.get("content", "")).split())
            if current and (len(current) >= CHUNK_MESSAGES or words + size > CHUNK_WORDS):
                chunks.append({"batch": batch_index, "date": batch["date"], "messages": current})
                current, words = [], 0
            current.append(message)
            words += size
        if current:
            chunks.append({"batch": batch_index, "date": batch["date"], "messages": current})
    return chunks


def render_excerpt(chunk: dict) -> str:
    lines = [f"{m.get('role', 'user')}: {' '.join(str(m.get('content', '')).split())}"
             for m in chunk["messages"]]
    return "\n".join(lines)


def _json_call(spend: Spend, system: str, user: str, max_tokens: int,
               fields: tuple[str, ...]) -> tuple[dict[str, str], dict[str, Any]]:
    """One gpt-4o-mini JSON call returning the named string fields, retried on transport faults."""
    spend.check()
    payload = {"model": BUILDER_MODEL, "temperature": 0, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    key = os.environ["OPENROUTER_API_KEY"].strip()
    status, body = 599, {}
    for attempt in range(6):
        started = time.perf_counter()
        status, body, _ = http_json("https://openrouter.ai/api/v1/chat/completions", payload,
                                    {"Authorization": f"Bearer {key}"}, 180)
        seconds = time.perf_counter() - started
        if status == 200 and body.get("choices"):
            usage = body.get("usage", {})
            with spend.lock:
                spend.calls += 1
                spend.usd += (usage.get("prompt_tokens", 0) * BUILDER_PRICE_IN
                              + usage.get("completion_tokens", 0) * BUILDER_PRICE_OUT)
            try:
                parsed = json.loads(body["choices"][0]["message"].get("content") or "")
                result = {name: str(parsed[name]).strip() for name in fields}
            except (json.JSONDecodeError, KeyError, TypeError):
                time.sleep(2)
                continue
            if not result[fields[-1]]:
                continue
            return result, {"usage": usage, "seconds": round(seconds, 3),
                            "provider": body.get("provider"), "attempts": attempt + 1,
                            "finish": body["choices"][0].get("finish_reason")}
        time.sleep(min(60, 3 * 2**attempt))
    raise RuntimeError(f"builder call failed with status {status}: {str(body)[:200]}")


def needs_compression(storyline: str) -> bool:
    """gpt-4o-mini ignores the word limit in the prompt (measured 2026-09-25: storylines reached
    1,068 words, p90 933, and truncated their own JSON), so the bound is enforced here."""
    return len(storyline.split()) > COMPRESS_ABOVE_WORDS


def builder_call(spend: Spend, storyline: str, chunk: dict) -> tuple[dict[str, str], dict[str, Any]]:
    """One gpt-4o-mini call per Add, plus a compression call only when the storyline outgrows
    its bound: the chunk's digest and the user's updated storyline."""
    user = (f"CURRENT STORYLINE:\n{storyline or '(empty)'}\n\n"
            f"NEW EXCERPT (date: {chunk['date'] or 'unknown'}):\n{render_excerpt(chunk)}")
    result, meta = _json_call(spend, BUILDER_SYSTEM, user, BUILDER_MAX_TOKENS, ("digest", "storyline"))
    if needs_compression(result["storyline"]):
        try:
            compressed, extra = _json_call(spend, COMPRESS_SYSTEM, result["storyline"],
                                           COMPRESS_MAX_TOKENS, ("storyline",))
        except RuntimeError as exc:
            if "status 402" in str(exc):
                raise
            # Amendment 3: fail forward. The uncompressed storyline already holds this Add's
            # content; keeping the previous one instead froze storylines that stayed long.
            meta["compress_failed"] = str(exc)[:200]
            return result, meta
        meta["compress"] = {**extra, "words_before": len(result["storyline"].split()),
                            "words_after": len(compressed["storyline"].split())}
        result["storyline"] = compressed["storyline"]
    return result, meta


def build(data: list[dict], out: Path, workers: int, spend: Spend) -> None:
    log = Appender(out / "memory.jsonl")
    previous = read_jsonl(out / "memory.jsonl")

    def one(conversation: dict) -> None:
        index = conversation["conversation"]
        done = sorted((r for r in previous if r["conversation"] == index), key=lambda r: r["chunk"])
        storyline = done[-1]["storyline"] if done else ""
        for position, chunk in enumerate(aml_chunks(conversation)):
            if position < len(done):
                continue
            try:
                result, meta = builder_call(spend, storyline, chunk)
            except RuntimeError as exc:
                if "status 402" in str(exc):
                    # An account out of credit is the apparatus failing, not the service: stop
                    # rather than record it as a fail-closed Add (happened 2026-09-25).
                    raise SystemExit(f"conversation {index} chunk {position}: {exc}") from exc
                # Fail closed, as C9 would: the Add keeps the previous storyline.
                log.write({"conversation": index, "chunk": position, "batch": chunk["batch"],
                           "date": chunk["date"], "digest": "", "storyline": storyline,
                           "failed": str(exc)[:200]})
                continue
            storyline = result["storyline"]
            log.write({"conversation": index, "chunk": position, "batch": chunk["batch"],
                       "date": chunk["date"], **result, **meta,
                       "words": sum(len(str(m.get("content", "")).split()) for m in chunk["messages"]),
                       "messages": len(chunk["messages"])})

    pool(workers, [(lambda c=c: one(c)) for c in data])


# ------------------------------------------------------------------------------------------
# Search-side placement
# ------------------------------------------------------------------------------------------


def gated(question: str) -> bool:
    return GATE.search(question) is not None


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def bm25_top(query: str, documents: list[str], k: int) -> list[int]:
    """Indices of the k best documents for the query (BM25, k1 1.2, b 0.75), ties by position."""
    tokenised = [_tokens(d) for d in documents]
    average = statistics.fmean(len(t) for t in tokenised) if tokenised else 0.0
    frequency: dict[str, int] = {}
    for tokens in tokenised:
        for token in set(tokens):
            frequency[token] = frequency.get(token, 0) + 1
    count = len(documents)

    def score(tokens: list[str]) -> float:
        total = 0.0
        for term in set(_tokens(query)):
            tf = tokens.count(term)
            if not tf:
                continue
            idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
            total += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(tokens) / (average or 1)))
        return total

    ranked = sorted(range(count), key=lambda i: (-score(tokenised[i]), i))
    return ranked[:k]


def memory_of(out: Path) -> dict[int, list[dict]]:
    by_conversation: dict[int, list[dict]] = {}
    for record in read_jsonl(out / "memory.jsonl"):
        by_conversation.setdefault(record["conversation"], []).append(record)
    return {k: sorted(v, key=lambda r: r["chunk"]) for k, v in by_conversation.items()}


def storyline_item(records: list[dict]) -> dict[str, str] | None:
    if not records or not records[-1]["storyline"]:
        return None
    last = records[-1]
    return {"id": f"storyline-{last['conversation']}-{last['chunk']}",
            "content": ("Summary of the user's conversation history, oldest to newest "
                        f"(last updated {last['date'] or 'unknown'}):\n{last['storyline']}")}


def digest_items(records: list[dict], query: str) -> list[dict[str, str]]:
    usable = [r for r in records if r.get("digest")]
    if len(usable) > MAX_DIGESTS:
        keep = set(bm25_top(query, [r["digest"] for r in usable], MAX_DIGESTS))
        usable = [r for i, r in enumerate(usable) if i in keep]
    return [{"id": f"digest-{r['conversation']}-{r['chunk']}",
             "content": f"[{r['date'] or 'unknown'}] {r['digest']}"} for r in usable]


def arm_items(arm: str, items: list[dict], records: list[dict], question: str) -> list[dict]:
    """The items an arm answers from, never more than the platform's top_k."""
    if arm == "r0" or (arm in GATED_ARMS and not gated(question)):
        return items[:TOP_K]
    story = storyline_item(records)
    top: list[dict] = []
    if arm in ("story", "story_digests", "story_all") and story:
        top.append(story)
    if arm in ("digests", "story_digests"):
        top.extend(digest_items(records, question))
    return (top + items)[:TOP_K]


# ------------------------------------------------------------------------------------------
# Answer, judge, mechanism
# ------------------------------------------------------------------------------------------


def answer(data: list[dict], out: Path, retrieval: Path, arm: str, workers: int,
           spend: Spend) -> None:
    questions = {q["id"]: q for c in data for q in c["questions"]}
    memory = memory_of(out)
    log = Appender(out / f"answers-{arm}.jsonl")
    done = {r["id"] for r in read_jsonl(out / f"answers-{arm}.jsonl")}
    r0 = {r["id"]: r for r in read_jsonl(out / "answers-r0.jsonl")}

    def one(record: dict) -> None:
        question = questions[record["id"]]
        if arm in GATED_ARMS and not gated(question["question"]):
            # Byte-identical context to r0, so r0's answer is this arm's answer.
            log.write({**r0[record["id"]], "arm": arm, "reused_from": "r0"})
            return
        items = arm_items(arm, record["items"], memory.get(record["conversation"], []),
                          question["question"])
        prompt = (ANSWER_PROMPT.replace("<context>", context_of(items))
                  .replace("<question>", question["question"]))
        text = complete(spend, [{"role": "user", "content": prompt}], 512)
        log.write({"id": record["id"], "type": record["type"], "arm": arm, "answer": text,
                   "items": len(items), "gated": gated(question["question"])})

    records = [r for r in read_jsonl(retrieval) if r["status"] == 200 and r["id"] not in done
               and r["id"] in questions]
    if arm in GATED_ARMS:
        missing = [r["id"] for r in records if not gated(questions[r["id"]]["question"])
                   and r["id"] not in r0]
        if missing:
            raise SystemExit(f"{len(missing)} ungated questions have no r0 answer; answer r0 first")
    pool(workers, [(lambda r=r: one(r)) for r in records])


def judge(data: list[dict], out: Path, arm: str, workers: int, spend: Spend) -> None:
    """The probe's judge, except that an answer reused from r0 keeps r0's judgement.

    Its context and answer are r0's byte for byte, so re-judging it would only add the judge's
    own noise to a delta that is zero by construction.
    """
    if arm in GATED_ARMS:
        reused = [r for r in read_jsonl(out / f"answers-{arm}.jsonl") if r.get("reused_from") == "r0"]
        r0 = {r["id"]: r for r in read_jsonl(out / "judged-r0.jsonl")}
        missing = [r["id"] for r in reused if r["id"] not in r0]
        if missing:
            raise SystemExit(f"{len(missing)} reused answers have no r0 judgement; judge r0 first")
        done = {r["id"] for r in read_jsonl(out / f"judged-{arm}.jsonl")}
        log = Appender(out / f"judged-{arm}.jsonl")
        for record in reused:
            if record["id"] not in done:
                log.write({**r0[record["id"]], "arm": arm, "reused_from": "r0"})
    probe_judge(data, out, arm, workers, spend)


def storycover(data: list[dict], out: Path, types: set[str], workers: int, spend: Spend) -> None:
    """Mechanism: does the storyline ALONE hold what each rubric point needs?"""
    memory = memory_of(out)
    log = Appender(out / "storycover.jsonl")
    done = {r["id"] for r in read_jsonl(out / "storycover.jsonl")}

    def one(conversation: dict, question: dict) -> None:
        story = storyline_item(memory.get(conversation["conversation"], []))
        rubric = question["rubric"]
        criteria = "\n".join(f"[{i}] {r}" for i, r in enumerate(rubric))
        scores = None
        if story and rubric:
            prompt = (COVERAGE_PROMPT.replace("<question>", question["question"])
                      .replace("<rubric_item>", criteria).replace("<context>", story["content"]))
            scores = _judged(spend, prompt, len(rubric))
        log.write({"id": question["id"], "type": question["type"], "scores": scores,
                   "score": sum(scores) / len(scores) if scores else None})

    pool(workers, [(lambda c=c, q=q: one(c, q)) for c in data for q in c["questions"]
                   if q["type"] in types and q["id"] not in done])


def gate_report(data: list[dict], locomo: Path | None) -> dict[str, Any]:
    by_type: dict[str, list[int]] = {}
    for conversation in data:
        for question in conversation["questions"]:
            by_type.setdefault(question["type"], []).append(int(gated(question["question"])))
    result: dict[str, Any] = {"beam": {k: {"fired": sum(v), "n": len(v)} for k, v in sorted(by_type.items())}}
    if locomo is not None:
        samples = json.loads(locomo.read_text(encoding="utf-8"))
        fired = [qa["question"] for s in samples for qa in s["qa"] if gated(str(qa.get("question", "")))]
        result["locomo"] = {"fired": len(fired), "n": sum(len(s["qa"]) for s in samples),
                            "examples": fired[:10]}
    return result


def paired_bootstrap(deltas: list[float], seed: int = 20260925, resamples: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(statistics.fmean(rng.choices(deltas, k=n)) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)]


def report(data: list[dict], out: Path, locomo: Path | None) -> dict[str, Any]:
    judged = {arm: {r["id"]: r for r in read_jsonl(out / f"judged-{arm}.jsonl")} for arm in ARMS}
    types = sorted({q["type"] for c in data for q in c["questions"]})

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 4) if values else None

    table: dict[str, dict[str, Any]] = {}
    for kind in types:
        row: dict[str, Any] = {}
        for arm in ARMS:
            values = [r["score"] for r in judged[arm].values() if r["type"] == kind and r["score"] is not None]
            if values:
                row[arm] = {"mean": mean(values), "n": len(values)}
        table[kind] = row
    deltas: dict[str, Any] = {}
    for arm in ARMS[1:]:
        for kind in types:
            common = [i for i, r in judged[arm].items() if r["type"] == kind and r["score"] is not None
                      and judged["r0"].get(i, {}).get("score") is not None]
            if not common:
                continue
            diff = [judged[arm][i]["score"] - judged["r0"][i]["score"] for i in common]
            deltas.setdefault(arm, {})[kind] = {"delta": round(statistics.fmean(diff), 4),
                                                "ci95": paired_bootstrap(diff), "n": len(diff)}
        everything = [i for i, r in judged[arm].items() if r["score"] is not None
                      and judged["r0"].get(i, {}).get("score") is not None]
        if everything:
            diff = [judged[arm][i]["score"] - judged["r0"][i]["score"] for i in everything]
            deltas.setdefault(arm, {})["ALL"] = {"delta": round(statistics.fmean(diff), 4),
                                                 "ci95": paired_bootstrap(diff), "n": len(diff)}
    memory = read_jsonl(out / "memory.jsonl")
    built = [r for r in memory if "usage" in r]
    finals = memory_of(out)
    story_words = [len(v[-1]["storyline"].split()) for v in finals.values() if v]
    cover = [r["score"] for r in read_jsonl(out / "storycover.jsonl") if r["score"] is not None]
    summary = {
        "by_type": table,
        "delta_vs_r0": deltas,
        "gate": gate_report(data, locomo),
        "build": {
            "chunks": len(memory), "failed": sum(1 for r in memory if r.get("failed")),
            "seconds_p50": None if not built else round(statistics.median(r["seconds"] for r in built), 2),
            "seconds_p90": None if len(built) < 10 else round(statistics.quantiles([r["seconds"] for r in built], n=10)[-1], 2),
            "prompt_tokens_p50": None if not built else statistics.median(r["usage"].get("prompt_tokens", 0) for r in built),
            "completion_tokens_p50": None if not built else statistics.median(r["usage"].get("completion_tokens", 0) for r in built),
            "usd": round(sum(r["usage"].get("prompt_tokens", 0) * BUILDER_PRICE_IN
                             + r["usage"].get("completion_tokens", 0) * BUILDER_PRICE_OUT for r in built), 4),
            "providers": sorted({str(r.get("provider")) for r in built}),
            "final_storyline_words": {"min": min(story_words, default=0), "max": max(story_words, default=0),
                                      "median": statistics.median(story_words) if story_words else 0},
        },
        "storyline_coverage_summarization": {"mean": mean(cover), "n": len(cover)},
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["build", "gate", "answer", "judge", "storycover", "report"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--retrieval", type=Path, help="the probe's stored retrieval.jsonl")
    parser.add_argument("--locomo", type=Path, help="locomo10.json, for the gate's false fires")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arm", choices=ARMS, default="r0")
    parser.add_argument("--conversations", help="comma separated indices (default: all)")
    parser.add_argument("--cap-usd", type=float, default=3.0, help="model spend cap per process")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(args.data)
    if args.conversations:
        wanted = {int(v) for v in args.conversations.split(",")}
        data = [c for c in data if c["conversation"] in wanted]
    spend = Spend(args.cap_usd)
    if args.phase == "build":
        build(data, args.out, args.workers, spend)
    elif args.phase == "gate":
        print(json.dumps(gate_report(data, args.locomo), indent=2, ensure_ascii=False))
        return
    elif args.phase == "answer":
        if args.retrieval is None:
            raise SystemExit("--retrieval is required for answer")
        answer(data, args.out, args.retrieval, args.arm, args.workers, spend)
    elif args.phase == "judge":
        judge(data, args.out, args.arm, args.workers, spend)
    elif args.phase == "storycover":
        storycover(data, args.out, {"summarization"}, args.workers, spend)
    print(json.dumps(report(data, args.out, args.locomo), indent=2, ensure_ascii=False))
    print(f"model spend this process: ${spend.usd:.4f} over {spend.calls} calls")


if __name__ == "__main__":
    main()
