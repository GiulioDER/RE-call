"""Does C9's compile cost bound (#775) save what the official Full's journal says it would?

Pre-registration: docs/preregistrations/2026-09-26-c9-compile-cost-replay.md

Builds public BEAM 100K Adds whose sizes follow the prompt-size mix of the official Textual Full
(2026-09-25, 10:49 to 02:19 UTC, from C9's journal), then compiles every Add twice through the
served compiler code: arm A as served (a cut-off answer is resent, no size limit), arm B with
#775 (a cut-off answer is not resent, payloads over 150,000 encoded characters are skipped). Both
arms use ``prior_record_mode="without-ids"`` and no prior records. Each arm imports
``recall_aml`` from its own checkout, given by ``--code``. Only OpenRouter is called.

Usage (VPS3):
    python aml_c9_compile_cost_replay.py build --beam beam100k.jsonl --out adds.jsonl
    python aml_c9_compile_cost_replay.py run --arm A --code <master checkout> --adds adds.jsonl --out A.jsonl
    python aml_c9_compile_cost_replay.py run --arm B --code <PR checkout> --adds adds.jsonl --out B.jsonl
    python aml_c9_compile_cost_replay.py compare --a A.jsonl --b B.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import sys
import threading
from typing import Any

#: gpt-4o-mini list price, USD per million tokens (input, output).
PRICE_IN, PRICE_OUT = 0.15, 0.60
#: Encoded characters per prompt token, measured with o200k_base on English payloads (3.56 to
#: 3.71). Used only to aim each Add at its target size; the result is read back from the
#: provider's own prompt_tokens.
CHARS_PER_TOKEN = 3.7
LIMIT_CHARS = 150_000
SPEND_CAP = {"A": 2.0, "B": 1.0}
WORKERS = 6
SEED = 20260926
#: Prompt-token bins of the Full's first compile calls and how many of its 12,833 compiled-period
#: Adds fell in each, scaled to 200 Adds (largest remainder).
STRATA = [
    ((1_000, 5_000), 90),
    ((5_000, 10_000), 46),
    ((10_000, 15_000), 2),
    ((15_000, 20_000), 2),
    ((20_000, 25_000), 9),
    ((25_000, 30_000), 12),
    ((30_000, 40_000), 13),
    ((40_000, 60_000), 19),
    ((60_000, 90_000), 7),
]
BINS = [0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 60_000]


def build(args: argparse.Namespace) -> None:
    """One Add per stratum draw: consecutive BEAM messages until the target size is reached."""
    rng = random.Random(SEED)
    conversations = [
        [m for batch in json.loads(line)["batches"] for m in batch["messages"]]
        for line in args.beam.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adds = []
    for (low, high), count in STRATA:
        for _ in range(count):
            target_chars = int(rng.uniform(low, high) * CHARS_PER_TOKEN)
            conversation = rng.choice(conversations)
            start = rng.randrange(len(conversation))
            messages, size, index = [], 0, start
            while size < target_chars and len(messages) < len(conversation):
                message = conversation[index % len(conversation)]
                content = str(message.get("content", ""))
                if size + len(content) > target_chars and messages:
                    content = content[: max(200, target_chars - size)]
                messages.append(
                    {"role": message.get("role", "user"), "content": content,
                     "timestamp": message.get("timestamp")}
                )
                size += len(content)
                index += 1
            adds.append({"id": f"add{len(adds):03d}", "stratum": [low, high],
                         "target_tokens": target_chars / CHARS_PER_TOKEN, "messages": messages})
    args.out.write_text("\n".join(json.dumps(a) for a in adds) + "\n", encoding="utf-8")
    print(f"built {len(adds)} adds into {args.out}")


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
        try:
            response = self._inner.chat.completions.create(**kwargs)
        except Exception as exc:  # BROAD-CATCH: recorded, then re-raised into the compiler
            self._local.calls.append({"error": type(exc).__name__})
            raise
        usage = getattr(response, "usage", None)
        self._local.calls.append({
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        })
        return response


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
    kwargs: dict[str, Any] = {"prior_record_mode": "without-ids"}
    if args.arm == "B":
        kwargs["max_anchor_payload_chars"] = LIMIT_CHARS
    compiler = OpenAICompiler(client, **kwargs)
    spent = [0.0]
    lock = threading.Lock()

    def one(add: dict[str, Any]) -> dict[str, Any] | None:
        with lock:
            if spent[0] >= SPEND_CAP[args.arm]:
                return None
        client.begin()
        messages = [Message(role=m["role"], content=m["content"], timestamp=m.get("timestamp"))
                    for m in add["messages"]]
        error, records = None, []
        try:
            records = compiler.compile_anchored_v3(messages, add["id"], [])
        except Exception as exc:  # BROAD-CATCH: a fallback is the thing being measured
            error = type(exc).__name__
        calls = client.calls()
        cost = sum((c.get("prompt_tokens", 0) * PRICE_IN + c.get("completion_tokens", 0) * PRICE_OUT) / 1e6
                   for c in calls)
        with lock:
            spent[0] += cost
        return {"id": add["id"], "stratum": add["stratum"], "arm": args.arm, "calls": calls,
                "cost_usd": cost, "accepted_records": len(records), "error": error,
                "compiled": bool(records) and error is None}

    todo = [a for a in adds if a["id"] not in done]
    with args.out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(WORKERS) as pool:
        for row in pool.map(one, todo):
            if row is None:
                continue
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    print(f"arm {args.arm}: spent USD {spent[0]:.3f}")


def _bin(tokens: int) -> int:
    return max(b for b in BINS if b <= tokens)


def compare(args: argparse.Namespace) -> None:
    a = {r["id"]: r for r in map(json.loads, args.a.read_text().splitlines()) if r}
    b = {r["id"]: r for r in map(json.loads, args.b.read_text().splitlines()) if r}
    ids = sorted(set(a) & set(b))
    first_tokens = {i: next((c["prompt_tokens"] for c in a[i]["calls"] if "prompt_tokens" in c), 0) for i in ids}
    out: dict[str, Any] = {"adds_both": len(ids)}
    for name, rows in (("A", a), ("B", b)):
        part = [rows[i] for i in ids]
        out[name] = {
            "cost_usd": round(sum(r["cost_usd"] for r in part), 4),
            "calls": sum(len(r["calls"]) for r in part),
            "compiled_adds": sum(r["compiled"] for r in part),
            "accepted_records": sum(r["accepted_records"] for r in part),
            "errors": dict(sorted({e: sum(1 for r in part if r["error"] == e)
                                   for e in {r["error"] for r in part}}.items(), key=str)),
        }
    out["cost_ratio_B_over_A"] = round(out["B"]["cost_usd"] / out["A"]["cost_usd"], 3) if out["A"]["cost_usd"] else None
    out["compiled_kept_B_over_A"] = (round(out["B"]["compiled_adds"] / out["A"]["compiled_adds"], 3)
                                     if out["A"]["compiled_adds"] else None)
    truncated_first = [i for i in ids if a[i]["calls"] and a[i]["calls"][0].get("finish_reason") == "length"]
    out["A_truncated_first_answers"] = len(truncated_first)
    out["A_truncated_then_compiled"] = sum(a[i]["compiled"] for i in truncated_first)
    by_bin: dict[int, dict[str, Any]] = {}
    for i in ids:
        row = by_bin.setdefault(_bin(first_tokens[i]), {"adds": 0, "A_compiled": 0, "B_compiled": 0,
                                                        "A_usd": 0.0, "B_usd": 0.0, "B_skipped": 0})
        row["adds"] += 1
        row["A_compiled"] += a[i]["compiled"]
        row["B_compiled"] += b[i]["compiled"]
        row["A_usd"] += a[i]["cost_usd"]
        row["B_usd"] += b[i]["cost_usd"]
        row["B_skipped"] += b[i]["error"] == "CompilerInputTooLarge"
    out["by_first_call_prompt_tokens"] = {
        str(k): {**v, "A_usd": round(v["A_usd"], 4), "B_usd": round(v["B_usd"], 4)}
        for k, v in sorted(by_bin.items())
    }
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--beam", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=("A", "B"), required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--adds", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()
    {"build": build, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
