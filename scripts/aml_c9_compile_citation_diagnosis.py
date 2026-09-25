"""Which anchor ids does C9's compiler cite that it was never sent? A replay on public data.

Pre-registration: docs/preregistrations/2026-09-25-c9-compile-citation-diagnosis.md

On the 2026-09-25 AML Textual smoke, 25 of 134 C9 Adds fell back, mostly because every record
gpt-4o-mini proposed cited only unknown anchor ids. This replays C9's own compiler
(``OpenAICompiler.compile_anchored_v3``, the served code) on public LoCoMo and BEAM 100K sessions
cut the way AML's Textual adapter cuts them, keeps the raw model output of every call, and sorts
every cited id against the ids that call actually sent.

It never touches the official C9 service or its database: it builds the compiler in process and
talks only to OpenRouter.

Usage (VPS3):
    python scripts/aml_c9_compile_citation_diagnosis.py run --locomo locomo10.json \
        --beam beam100k.jsonl --out DIR
    python scripts/aml_c9_compile_citation_diagnosis.py summarize --out DIR
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.beam.f1_storyline_replay import aml_chunks  # noqa: E402
from recall_aml.__main__ import build_openrouter_client  # noqa: E402
from recall_aml.compiler import OpenAICompiler, StoredCodingRecord  # noqa: E402
from recall_aml.models import Message  # noqa: E402
from scripts.aml_locomo_route_compare import (  # noqa: E402
    SESSION_KEY,
    session_timestamp_ms,
    turn_content,
)

#: gpt-4o-mini list price in USD per million tokens (input, output), for the spend cap.
PRICE_IN, PRICE_OUT = 0.15, 0.60
SPEND_CAP_USD = 0.60
LOCOMO_CONVERSATIONS = 10
LOCOMO_SESSIONS_EACH = 2
BEAM_CONVERSATIONS = 20
BEAM_CHUNKS_EACH = 2

_V3 = re.compile(r"a(\d+)_([0-9A-Za-z]*)")
_BARE = re.compile(r"a(\d+)")
#: Citations the served compiler resolves (``resolve_bare_anchor_ids`` handles a bare index).
RESOLVED = frozenset({"known", "bare_index_known"})
_STORED = re.compile(r"<stored_data>(.*?)</stored_data>", re.DOTALL)


# ------------------------------------------------------------------------------------------------
# The simulated Add stream
# ------------------------------------------------------------------------------------------------


def locomo_conversations(path: Path) -> list[dict[str, Any]]:
    """LoCoMo as BEAM-shaped conversations: one batch per session, AML millisecond timestamps."""
    conversations = []
    for sample in json.loads(path.read_bytes())[:LOCOMO_CONVERSATIONS]:
        conversation = sample["conversation"]
        keys = sorted(
            (k for k in conversation if SESSION_KEY.match(k) and conversation[k]),
            key=lambda k: int(SESSION_KEY.match(k).group(1)),  # type: ignore[union-attr]
        )[:LOCOMO_SESSIONS_EACH]
        batches = []
        for key in keys:
            stamp = session_timestamp_ms(conversation[f"{key}_date_time"])
            batches.append(
                {
                    "date": conversation[f"{key}_date_time"],
                    "messages": [
                        {"role": "user", "content": turn_content(turn), "timestamp": stamp}
                        for turn in conversation[key]
                    ],
                }
            )
        conversations.append({"id": f"locomo:{sample['sample_id']}", "batches": batches})
    return conversations


def beam_conversations(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        {"id": f"beam:{row['conversation']}", "batches": row["batches"]}
        for row in rows[:BEAM_CONVERSATIONS]
    ]


def planned_calls(locomo: Path, beam: Path) -> list[dict[str, Any]]:
    """Every compile this run makes, in order. Chunks of one session stay in their order."""
    calls: list[dict[str, Any]] = []
    for conversation in locomo_conversations(locomo):
        for chunk in aml_chunks(conversation):
            calls.append({"source": "locomo", "conversation": conversation["id"], **chunk})
    for conversation in beam_conversations(beam):
        for chunk in aml_chunks(conversation)[:BEAM_CHUNKS_EACH]:
            calls.append({"source": "beam", "conversation": conversation["id"], **chunk})
    return calls


# ------------------------------------------------------------------------------------------------
# Recording the served compiler
# ------------------------------------------------------------------------------------------------


class RecordingClient:
    """Wraps the OpenRouter client C9 builds, keeping every request and raw answer."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.exchanges: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        exchange: dict[str, Any] = {"request_messages": kwargs.get("messages")}
        try:
            response = self._inner.chat.completions.create(**kwargs)
        except Exception as exc:  # BROAD-CATCH: recorded, then re-raised into the compiler
            exchange.update(error_class=type(exc).__name__, seconds=time.perf_counter() - started)
            self.exchanges.append(exchange)
            raise
        usage = getattr(response, "usage", None)
        exchange.update(
            content=response.choices[0].message.content,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            seconds=time.perf_counter() - started,
        )
        self.exchanges.append(exchange)
        return response


class CompileLines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("compiler_anchor_compile_complete "):
            self.lines.append(json.loads(message.removeprefix("compiler_anchor_compile_complete ")))


def sent_anchor_ids(request_messages: list[dict[str, Any]]) -> list[str]:
    """The anchor ids one request actually carried, read back from its ``<stored_data>``."""
    for message in request_messages:
        match = _STORED.search(str(message.get("content", "")))
        if match:
            return [str(anchor["id"]) for anchor in json.loads(match.group(1))["anchors"]]
    raise ValueError("request carried no <stored_data>")


def classify(cited: str, sent: list[str]) -> str:
    """Sort one cited id against the ids its call sent. The first matching rule wins."""
    if cited in sent:
        return "known"
    by_index = {}
    for anchor_id in sent:
        match = _V3.fullmatch(anchor_id)
        if match:
            by_index[int(match.group(1))] = match.group(2)
    hashes = {digest: index for index, digest in by_index.items()}
    if _BARE.fullmatch(cited):
        return "bare_index_known" if int(cited[1:]) in by_index else "bare_index_out_of_range"
    if any(anchor_id.startswith(cited) for anchor_id in sent):
        return "truncated_id"
    match = _V3.fullmatch(cited)
    if match:
        index, digest = int(match.group(1)), match.group(2)
        if index not in by_index:
            return "hash_of_sent_anchor" if digest in hashes else "index_out_of_range"
        true = by_index[index]
        if digest and true.startswith(digest):
            return "truncated_id"
        if digest in hashes:
            return "hash_of_other_anchor"
        return "right_index_wrong_hash"
    if cited.startswith("anchor_"):
        return "v2_anchor_form"
    if re.search(r"\s", cited) or len(cited) > 40:
        return "quoted_text"
    return "other"


def run(args: argparse.Namespace) -> None:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "calls.jsonl"
    done = {
        json.loads(line)["call"] for line in rows_path.read_text().splitlines()
    } if rows_path.exists() else set()
    handler = CompileLines()
    logging.getLogger("recall_aml").addHandler(handler)
    logging.getLogger("recall_aml").setLevel(logging.INFO)
    client = RecordingClient(build_openrouter_client(os.environ["OPENROUTER_API_KEY"].strip()))
    compiler = OpenAICompiler(client)
    calls = planned_calls(args.locomo, args.beam)
    print(f"planned calls {len(calls)}, already done {len(done)}", flush=True)
    spent = 0.0
    prior: dict[str, list[StoredCodingRecord]] = {}
    with rows_path.open("a", encoding="utf-8") as sink:
        for number, call in enumerate(calls):
            session_id = f"{call['conversation']}:b{call['batch']}"
            if number in done:
                continue
            if spent >= SPEND_CAP_USD:
                print(f"spend cap reached at call {number}", flush=True)
                break
            messages = [
                Message(role=m.get("role", "user"), content=str(m.get("content", "")),
                        timestamp=m.get("timestamp"))
                for m in call["messages"]
            ]
            first_exchange, first_line = len(client.exchanges), len(handler.lines)
            error = None
            records: list[Any] = []
            try:
                records = compiler.compile_anchored_v3(messages, session_id, prior.get(session_id, []))
            except Exception as exc:  # BROAD-CATCH: a fallback is the thing being measured
                error = type(exc).__name__
            exchanges = client.exchanges[first_exchange:]
            line = handler.lines[first_line] if len(handler.lines) > first_line else None
            for exchange in exchanges:
                spent += (exchange.get("prompt_tokens", 0) * PRICE_IN
                          + exchange.get("completion_tokens", 0) * PRICE_OUT) / 1e6
            final = next((e for e in reversed(exchanges) if "content" in e), None)
            sent = sent_anchor_ids(final["request_messages"]) if final else []
            cited: list[dict[str, str]] = []
            proposed = 0
            if final is not None:
                try:
                    parsed = json.loads(final["content"])
                    proposals = parsed.get("records", []) if isinstance(parsed, dict) else []
                except json.JSONDecodeError:
                    proposals = []
                for proposal in proposals if isinstance(proposals, list) else []:
                    if not isinstance(proposal, dict):
                        continue
                    proposed += 1
                    for anchor_id in proposal.get("evidence_anchor_ids") or []:
                        cited.append({"id": str(anchor_id), "class": classify(str(anchor_id), sent)})
            prior.setdefault(session_id, []).extend(
                StoredCodingRecord(id=f"rec{number}_{i}", record=record)
                for i, record in enumerate(records)
            )
            row = {
                "call": number,
                "source": call["source"],
                "session_id": session_id,
                "message_count": len(messages),
                "word_count": sum(len(m.content.split()) for m in messages if isinstance(m.content, str)),
                "sent_anchor_count": len(sent),
                "attempts": len(exchanges),
                "error": error,
                "accepted_records": len(records),
                "fallback": error is not None or not records,
                "proposed_records": proposed,
                "cited": cited,
                "diagnostics": line,
                "raw_content": final["content"] if final else None,
                "exchange_errors": [e.get("error_class") for e in exchanges if "error_class" in e],
                "spent_usd_so_far": round(spent, 4),
            }
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"call {number} {call['source']} anchors={len(sent)} accepted={len(records)} "
                  f"error={error} spent={spent:.3f}", flush=True)


def plan(args: argparse.Namespace) -> None:
    calls = planned_calls(args.locomo, args.beam)
    for source in ("locomo", "beam"):
        part = [c for c in calls if c["source"] == source]
        words = sorted(sum(len(str(m.get("content", "")).split()) for m in c["messages"]) for c in part)
        print(source, "calls", len(part), "median words", words[len(words) // 2] if words else None)


def summarize(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in (args.out / "calls.jsonl").read_text().splitlines()]
    summary: dict[str, Any] = {"calls": len(rows)}
    for source in ("all", "locomo", "beam"):
        part = [r for r in rows if source == "all" or r["source"] == source]
        classes = Counter(c["class"] for r in part for c in r["cited"])
        unknown = {k: v for k, v in classes.items() if k not in RESOLVED}
        fallback = [r for r in part if r["fallback"]]
        summary[source] = {
            "calls": len(part),
            "fallbacks": len(fallback),
            "fallback_rate": round(len(fallback) / len(part), 3) if part else None,
            "fallback_errors": Counter(str(r["error"]) for r in fallback),
            "cited_ids": sum(classes.values()),
            "unknown_ids": sum(unknown.values()),
            "unknown_by_class": dict(Counter(unknown).most_common()),
            "unknown_by_class_in_fallbacks": dict(
                Counter(c["class"] for r in fallback for c in r["cited"] if c["class"] not in RESOLVED)
            ),
            "median_anchors_fallback": _median([r["sent_anchor_count"] for r in fallback]),
            "median_anchors_ok": _median([r["sent_anchor_count"] for r in part if not r["fallback"]]),
        }
    summary["spent_usd"] = rows[-1]["spent_usd_so_far"] if rows else 0.0
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=dict))
    print(json.dumps(summary, indent=1, default=dict))


def _median(values: list[int]) -> float | None:
    values = sorted(values)
    return float(values[len(values) // 2]) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--locomo", type=Path, required=True)
    run_parser.add_argument("--beam", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--locomo", type=Path, required=True)
    plan_parser.add_argument("--beam", type=Path, required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"run": run, "plan": plan, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
