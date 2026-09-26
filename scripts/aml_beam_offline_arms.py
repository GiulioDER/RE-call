"""Answer registered offline arms on the stored BEAM probe retrieval, with the registered reader.

T-1 amendment 4 (docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md):
arms H (the production ``dated_items`` applied to the stored items, which predate the date
header), H2 (H answered separately) and T1 (``resolve_relative_times`` on H), on the 400 stored
BEAM 100K questions of 2026-09-24.

Everything except the model call is ``benchmarks/beam/aml_c9_probe.py`` unchanged: AML's BEAM answer
prompt, its batch rubric judge and the event-ordering score. The model call is replaced by
``deepseek/deepseek-v4.1-flash`` pinned to one provider, reasoning off, temperature 0, for answer
and judge alike. Arms run concurrently, so their calls interleave in time.

    python scripts/aml_beam_offline_arms.py run --data beam100k.jsonl --out OUT --arms H,H2,T1
    python scripts/aml_beam_offline_arms.py score --out OUT --a T1 --b H --type temporal_reasoning

``OUT`` must hold the probe's ``retrieval.jsonl`` and ``state.json``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmarks.beam.aml_c9_probe as probe  # noqa: E402
from recall_aml.models import SearchItem  # noqa: E402
from recall_aml.temporal_render import resolve_relative_times  # noqa: E402
from recall_aml.window_format import dated_items  # noqa: E402

READER = "deepseek/deepseek-v4.1-flash"
PROVIDER = "DeepInfra"
#: OpenRouter list price, read 2026-09-26 from /api/v1/models.
probe.PRICES[READER] = (0.30e-6, 1.20e-6)
RETRYABLE = {408, 429, 500, 502, 503, 504, 520, 522, 524}
SEED = 20260925


def complete(spend: probe.Spend, messages: list[dict[str, str]], max_tokens: int,
             json_mode: bool = False, model: str = READER) -> str:
    """``probe.complete`` with the registered reader: one pinned provider, reasoning off."""
    spend.check()
    payload: dict[str, Any] = {
        "model": READER, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        "provider": {"order": [PROVIDER], "allow_fallbacks": False},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    key = os.environ["OPENROUTER_API_KEY"].strip()
    status, body = 0, {}
    for attempt in range(6):
        status, body, _ = probe.http_json("https://openrouter.ai/api/v1/chat/completions", payload,
                                          {"Authorization": f"Bearer {key}"}, 180)
        if status == 200 and body.get("choices"):
            spend.add(body.get("usage", {}), READER)
            content = body["choices"][0]["message"].get("content") or ""
            return probe._THINK.sub("", content).strip()
        if status and status not in RETRYABLE:
            break
        time.sleep(min(60, 3 * 2**attempt))
    raise RuntimeError(f"model call failed with status {status}: {str(body)[:200]}")


def _models(items: list[dict]) -> list[SearchItem]:
    return [
        SearchItem(
            id=str(item["id"]), content=item["content"],
            created_at=datetime.fromisoformat(str(item["created_at"]).replace("Z", "+00:00"))
            if item.get("created_at") else None,
            source=str(item.get("source") or ""), session_id=str(item.get("session_id") or ""),
            kind=str(item.get("kind") or "raw"), score=float(item.get("score") or 0.0),
        )
        for item in items
    ]


def offline_arm(arm: str, items: list[dict]) -> list[dict]:
    """The stored items as the registered arm renders them; ids and order never change."""
    dated = dated_items(_models(items))
    if arm == "T1":
        dated = resolve_relative_times(dated)
    return [{**item, "content": model.content} for item, model in zip(items, dated, strict=True)]


OFFLINE_ARMS = ("H", "H2", "T1")
_probe_arm_items = probe.arm_items


def arm_items(arm: str, items: list[dict], conversation: dict, user: str) -> list[dict]:
    if arm in OFFLINE_ARMS:
        return offline_arm(arm, items)
    return _probe_arm_items(arm, items, conversation, user)


probe.complete = complete
probe.arm_items = arm_items


def run(args: argparse.Namespace) -> None:
    data = probe.read_jsonl(args.data)
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in OFFLINE_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}")
    spend = probe.Spend(args.cap_usd)
    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        list(pool.map(lambda arm: probe.answer(data, args.out, arm, None, args.workers, spend), arms))
    print(json.dumps({"answered": {a: len(probe.read_jsonl(args.out / f"answers-{a}.jsonl")) for a in arms},
                      "spend_usd": round(spend.usd, 4)}), flush=True)
    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        list(pool.map(lambda arm: probe.judge(data, args.out, arm, args.workers, spend), arms))
    print(json.dumps({"judged": {a: len(probe.read_jsonl(args.out / f"judged-{a}.jsonl")) for a in arms},
                      "spend_usd": round(spend.usd, 4)}), flush=True)


def _per_question(out: Path, arm: str, field: str) -> dict[str, tuple[str, float | None]]:
    return {r["id"]: (r["type"], r.get(field)) for r in probe.read_jsonl(out / f"judged-{arm}.jsonl")}


def score(args: argparse.Namespace) -> None:
    field = "event_ordering" if args.type == "event_ordering" and args.kendall else "score"
    a = _per_question(args.out, args.a, field)
    b = _per_question(args.out, args.b, field)
    keys = sorted(k for k in set(a) & set(b)
                  if (args.type is None or a[k][0] == args.type) and a[k][1] is not None and b[k][1] is not None)
    unscored = sum(1 for k in set(a) & set(b) if (args.type is None or a[k][0] == args.type)
                   and (a[k][1] is None or b[k][1] is None))
    diffs = [a[k][1] - b[k][1] for k in keys]
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10_000)) if diffs else []
    result = {
        "contrast": f"{args.a} - {args.b}", "type": args.type or "all", "field": field, "n": len(keys),
        "unscored_excluded": unscored,
        "a_mean": sum(a[k][1] for k in keys) / len(keys) if keys else None,
        "b_mean": sum(b[k][1] for k in keys) / len(keys) if keys else None,
        "diff": sum(diffs) / len(diffs) if diffs else None,
        "ci95": [boots[249], boots[9_749]] if boots else None,
        "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--arms", default="H,H2,T1")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--cap-usd", type=float, default=7.0)
    r.set_defaults(handler=run)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--type", default=None)
    s.add_argument("--kendall", action="store_true", help="event_ordering: use the alignment score")
    s.set_defaults(handler=score)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
