"""Time C9's own Add-time compiler through each OpenRouter route for gpt-4o-mini.

Pre-registration: ``docs/preregistrations/2026-09-24-aml-c9-add-speed.md``.

C9 lets OpenRouter choose between the providers that serve ``openai/gpt-4o-mini``. This compiles
the same LoCoMo sessions through three routes, one call at a time: ``default`` (unpinned, as
served), ``openai`` and ``azure`` (each pinned with fallbacks off). Every session goes through all
three, in a rotating order, so time drift and warm-up fall on every route alike. The compiler is
``OpenAICompiler.compile_anchored_v3`` exactly as an Add calls it, including its own retries; the
only change is the ``provider`` field added to each request. A call whose served provider is not
the pinned one makes the run invalid, which is the apparatus check.

    python scripts/aml_compiler_provider_latency.py --data locomo10.json --out providers.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_add_throughput import provider_sessions, size  # noqa: E402

from recall_aml.__main__ import build_openrouter_client  # noqa: E402
from recall_aml.compiler import OpenAICompiler  # noqa: E402
from recall_aml.models import AddRequest  # noqa: E402

ROUTES = ("default", "openai", "azure")
ORDERS = list(itertools.permutations(ROUTES))


def served_provider(response: Any) -> str | None:
    value = getattr(response, "provider", None)
    if value is None:
        value = (getattr(response, "model_extra", None) or {}).get("provider")
    return value


class RoutedClient:
    """The OpenRouter client C9 builds, with one provider route added to every request."""

    def __init__(self, inner: Any, route: str, calls: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._route = route
        self._calls = calls
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        if self._route != "default":
            kwargs["extra_body"] = {"provider": {"order": [self._route], "allow_fallbacks": False}}
        started = time.perf_counter()
        try:
            response = self._inner.chat.completions.create(**kwargs)
        except Exception as exc:
            self._calls.append(
                {"seconds": round(time.perf_counter() - started, 2), "error": type(exc).__name__}
            )
            raise
        usage = getattr(response, "usage", None)
        self._calls.append(
            {
                "seconds": round(time.perf_counter() - started, 2),
                "served_provider": served_provider(response),
                "model": getattr(response, "model", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        )
        return response


def compile_once(inner: Any, route: str, messages: list[Any], session_id: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    compiler = OpenAICompiler(RoutedClient(inner, route, calls))
    started = time.perf_counter()
    try:
        records = compiler.compile_anchored_v3(messages, session_id, [])
        error = None
    except Exception as exc:  # BROAD-CATCH: a failed compile is a measured outcome here
        records, error = [], type(exc).__name__
    return {
        "route": route,
        "seconds": round(time.perf_counter() - started, 2),
        "accepted_records": len(records),
        "error": error,
        "calls": calls,
    }


def summary(rows: list[dict[str, Any]], route: str) -> dict[str, Any]:
    mine = [r for r in rows if r["route"] == route]
    seconds = sorted(r["seconds"] for r in mine if r["error"] is None)
    rates = [
        c["completion_tokens"] / c["seconds"]
        for r in mine
        for c in r["calls"]
        if c.get("completion_tokens") and c["seconds"] > 0
    ]
    served: dict[str, int] = {}
    for r in mine:
        for c in r["calls"]:
            key = str(c.get("served_provider", c.get("error")))
            served[key] = served.get(key, 0) + 1
    return {
        "compiles": len(mine),
        "failed_compiles": sum(1 for r in mine if r["error"] is not None),
        "seconds_p50": statistics.median(seconds) if seconds else None,
        "seconds_p90": seconds[int(len(seconds) * 0.9)] if seconds else None,
        "completion_tokens_per_second_p50": statistics.median(rates) if rates else None,
        "accepted_records_mean": statistics.mean(r["accepted_records"] for r in mine),
        "served_provider_counts": served,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    args = parser.parse_args()
    data = json.loads(args.data.read_bytes())
    inner = build_openrouter_client(os.environ["OPENROUTER_API_KEY"])
    rows: list[dict[str, Any]] = []
    for index, session in enumerate(provider_sessions(data, args.sessions)):
        session_id = f"provider-{args.run_id}-s{index}"
        messages = AddRequest.model_validate(
            {"request_id": session_id, "user_id": "provider-latency", "session_id": session_id,
             "messages": session}
        ).messages
        for route in ORDERS[index % len(ORDERS)]:
            row = compile_once(inner, route, messages, session_id)
            row.update({"session": index, "characters": size(session)})
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("session", "route", "seconds", "accepted_records",
                                                  "error")}), flush=True)
    by_session: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["error"] is None:
            by_session.setdefault(r["session"], {})[r["route"]] = r["seconds"]
    paired = {}
    for a, b in (("openai", "azure"), ("openai", "default"), ("azure", "default")):
        ratios = [s[a] / s[b] for s in by_session.values() if a in s and b in s and s[b] > 0]
        paired[f"{a}_over_{b}"] = {
            "pairs": len(ratios),
            "median_ratio": statistics.median(ratios) if ratios else None,
            "share_first_faster": sum(1 for x in ratios if x < 1) / len(ratios) if ratios else None,
        }
    pinned_mismatch = sum(
        1
        for r in rows
        if r["route"] != "default"
        for c in r["calls"]
        if c.get("served_provider") is not None
        and c["served_provider"].split("/")[0].lower() != r["route"]
    )
    result = {
        "preregistration": "docs/preregistrations/2026-09-24-aml-c9-add-speed.md",
        "measurement": "compiler_provider_latency",
        "run_id": args.run_id,
        "sessions": args.sessions,
        "routes": {route: summary(rows, route) for route in ROUTES},
        "paired": paired,
        "pinned_provider_mismatches": pinned_mismatch,
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
