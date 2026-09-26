"""Does a shorter compile answer (``lean`` or ``select``) make C9's Add compile faster and cheaper?

Pre-registration: docs/preregistrations/2026-09-26-c9-compile-output.md

Compiles the 200 BEAM Adds built by ``aml_c9_compile_cost_replay.py build`` (the prompt-size mix of
the official Textual Full) through the served v3 anchored compiler with #775's bounds, once per
output mode, and records each compile's wall time, tokens, accepted records and the compiler's own
diagnostics. Only OpenRouter is called, with the key in ``OPENROUTER_API_KEY``: it must not be the
official C9 key while an official run is live.

Usage (VPS3):
    python aml_c9_compile_output_replay.py run --mode full --code <checkout> --adds adds.jsonl --out full.jsonl
    python aml_c9_compile_output_replay.py run --mode select --code <checkout> --adds adds.jsonl --out select.jsonl
    python aml_c9_compile_output_replay.py compare --runs full.jsonl lean.jsonl select.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

#: gpt-4o-mini list price, USD per million tokens (input, output).
PRICE_IN, PRICE_OUT = 0.15, 0.60
LIMIT_CHARS = 150_000
SPEND_CAP_USD = 1.5
WORKERS = 6


class Recording:
    """Wraps the OpenRouter client, recording each call's tokens and finish reason per thread."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._local = threading.local()
        self.chat = self
        self.completions = self

    def begin(self) -> None:
        self._local.calls = []

    def calls(self) -> list[dict[str, Any]]:
        return list(getattr(self._local, "calls", []))

    def create(self, **kwargs: Any) -> Any:
        tick = time.perf_counter()
        try:
            response = self._inner.chat.completions.create(**kwargs)
        except Exception as exc:  # BROAD-CATCH: recorded, then re-raised into the compiler
            self._local.calls.append(
                {"error": type(exc).__name__, "seconds": time.perf_counter() - tick}
            )
            raise
        usage = getattr(response, "usage", None)
        self._local.calls.append({
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "seconds": time.perf_counter() - tick,
        })
        return response


class Diagnostics(logging.Handler):
    """Keeps the last ``compiler_anchor_compile_complete`` counters of the emitting thread."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def begin(self) -> None:
        self._local.last = None

    def last(self) -> dict[str, Any] | None:
        return getattr(self._local, "last", None)

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("compiler_anchor_compile_complete "):
            self._local.last = json.loads(message.split(" ", 1)[1])


def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.code.resolve()))
    from recall_aml.__main__ import build_openrouter_client
    from recall_aml.compiler import OpenAICompiler
    from recall_aml.models import Message

    adds = [json.loads(line) for line in args.adds.read_text(encoding="utf-8").splitlines() if line]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line}
    client = Recording(build_openrouter_client(os.environ["OPENROUTER_API_KEY"].strip()))
    diagnostics = Diagnostics()
    logger = logging.getLogger("recall_aml")
    logger.addHandler(diagnostics)
    logger.setLevel(logging.INFO)
    compiler = OpenAICompiler(
        client,
        prior_record_mode="without-ids",
        max_anchor_payload_chars=LIMIT_CHARS,
        anchor_output_mode=args.mode,
    )
    spent = [0.0]
    lock = threading.Lock()

    def one(add: dict[str, Any]) -> dict[str, Any] | None:
        with lock:
            if spent[0] >= SPEND_CAP_USD:
                return None
        client.begin()
        diagnostics.begin()
        messages = [Message(role=m["role"], content=m["content"], timestamp=m.get("timestamp"))
                    for m in add["messages"]]
        error, records = None, []
        tick = time.perf_counter()
        try:
            records = compiler.compile_anchored_v3(messages, add["id"], [])
        except Exception as exc:  # BROAD-CATCH: a fallback is one of the things measured
            error = type(exc).__name__
        seconds = time.perf_counter() - tick
        calls = client.calls()
        cost = sum((c.get("prompt_tokens", 0) * PRICE_IN + c.get("completion_tokens", 0) * PRICE_OUT) / 1e6
                   for c in calls)
        with lock:
            spent[0] += cost
        return {
            "id": add["id"], "stratum": add["stratum"], "mode": args.mode, "calls": calls,
            "seconds": seconds, "cost_usd": cost, "accepted_records": len(records),
            "records_with_event_time": sum(r.event_time is not None for r in records),
            "error": error, "compiled": bool(records) and error is None,
            "diagnostics": diagnostics.last(),
        }

    todo = [a for a in adds if a["id"] not in done]
    with args.out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(WORKERS) as pool:
        for row in pool.map(one, todo):
            if row is None:
                continue
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    print(f"mode {args.mode}: spent USD {spent[0]:.3f}")


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def compare(args: argparse.Namespace) -> None:
    runs = {}
    for path in args.runs:
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        runs[loaded[0]["mode"]] = {row["id"]: row for row in loaded}
    ids = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    out: dict[str, Any] = {"adds_in_every_run": len(ids)}
    for mode, rows in runs.items():
        part = [rows[i] for i in ids]
        answered = [r for r in part if r["calls"] and "completion_tokens" in r["calls"][-1]]
        diag = [r["diagnostics"] for r in part if r["diagnostics"]]
        accepted = sum(d["accepted_records"] for d in diag)
        out[mode] = {
            "cost_usd": round(sum(r["cost_usd"] for r in part), 4),
            "calls": sum(len(r["calls"]) for r in part),
            "compiled_adds": sum(r["compiled"] for r in part),
            "fallback_adds": sum(not r["compiled"] for r in part),
            "errors": {e: sum(1 for r in part if r["error"] == e)
                       for e in sorted({str(r["error"]) for r in part})},
            "completion_tokens_p50": _quantile(
                [float(r["calls"][-1]["completion_tokens"]) for r in answered], 0.5),
            "completion_tokens_total": sum(c.get("completion_tokens", 0) for r in part for c in r["calls"]),
            "prompt_tokens_total": sum(c.get("prompt_tokens", 0) for r in part for c in r["calls"]),
            "compile_seconds_p50": _quantile([r["seconds"] for r in part if r["compiled"]], 0.5),
            "compile_seconds_p90": _quantile([r["seconds"] for r in part if r["compiled"]], 0.9),
            "accepted_records_per_compiled_add": (
                round(accepted / len(diag), 3) if diag else None),
            "backfilled_share": (
                round(sum(d["evidence_backfilled_records"] for d in diag) / accepted, 3)
                if accepted else None),
            "records_with_event_time": sum(r["records_with_event_time"] for r in part),
        }
    if "full" in runs:
        for mode in runs:
            if mode == "full":
                continue
            pairs = [(runs["full"][i]["seconds"], runs[mode][i]["seconds"]) for i in ids
                     if runs["full"][i]["compiled"] and runs[mode][i]["compiled"]]
            ratios = sorted(b / a for a, b in pairs if a > 0)
            out[f"{mode}_over_full"] = {
                "paired_compiles": len(pairs),
                "median_seconds_ratio": _quantile(ratios, 0.5),
                "cost_ratio": (round(out[mode]["cost_usd"] / out["full"]["cost_usd"], 3)
                               if out["full"]["cost_usd"] else None),
                "completion_tokens_ratio": (
                    round(out[mode]["completion_tokens_total"] / out["full"]["completion_tokens_total"], 3)
                    if out["full"]["completion_tokens_total"] else None),
            }
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mode", choices=("full", "lean", "select"), required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--adds", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
