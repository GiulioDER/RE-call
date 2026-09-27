"""Does C9 with timestamped windows keep its Coding retrieval?

Pre-registration: docs/preregistrations/2026-09-24-aml-c9-coding-window-check.md

The frozen Agent Memory Bench coding corpus (manifest ``58055df1…``, 196 sessions, 34 task prompts)
is added to C9 built in process, one Add per session and one message per transcript event with
the event's role and timestamp, as an AML Coding trajectory arrives. Each event's text is exactly
what ``scripts/aml_c7_qualification.py`` flattens, so with the served renderer the stored text is
byte for byte the historical screen's; with ``--timestamped-windows`` every event also carries its
timestamp and role inside the window text. Then one Search per task prompt at top_k 100, scored at
session level as the C6 and C7 qualifications scored it.

    python scripts/aml_c9_coding_window_check.py collect --amb-root <agent-memory-bench> \\
        --arm K0 --out K0.json.gz --dated-search-content [--timestamped-windows] \\
        [--expect-search-content created-at-header-v1+relative-times-resolved-v1]
    python scripts/aml_c9_coding_window_check.py report --arms K0.json.gz K0b.json.gz K1.json.gz

Served C9 has reported ``created-at-header-v1`` since #761, so ``--dated-search-content`` is
needed to match it, and with T-1 on it reports ``created-at-header-v1+relative-times-resolved-v1``,
which ``--expect-search-content`` names; the collect refuses any other profile.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aml_c7_qualification import (  # noqa: E402
    _TEXT_FIELDS,
    _normalise,
    load_frozen_corpus,
)
from aml_locomo_loss_diagnosis import timestamped_windows  # noqa: E402

PREREGISTRATION = "docs/preregistrations/2026-09-24-aml-c9-coding-window-check.md"
_DATE_HEADER = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\] ")
EXPECTED_VARIANT = "C9_routed_specialists_grounded_graph_atomic"


def unix_ms(value: object) -> int | None:
    """An ISO 8601 event time as AML's Unix milliseconds, or None when it cannot be read."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)


def event_messages(path: Path) -> list[dict[str, Any]]:
    """One AML message per transcript event: its role, its timestamp, and its flattened text.

    The text of each event is what ``_render_transcript`` takes from it, normalised the same way,
    so joining the contents with single spaces gives the historical flattened session back
    (checked for every session before a collect starts).
    """
    messages: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event, parts = {}, [line]
        else:
            parts = []
            for field in _TEXT_FIELDS:
                value = event.get(field)
                if isinstance(value, str):
                    parts.append(value)
                elif value is not None:
                    parts.append(json.dumps(value, ensure_ascii=False))
        content = _normalise(" ".join(parts)).strip()
        if not content:
            continue
        message: dict[str, Any] = {"role": str(event.get("role") or "user"), "content": content}
        timestamp = unix_ms(event.get("ts"))
        if timestamp is not None:
            message["timestamp"] = timestamp
        messages.append(message)
    return messages


def condition_sessions(base: dict[str, str], condition_manifest: dict[str, Any]) -> list[str]:
    """The base corpus sessions a condition corpus keeps, refused unless each is byte-identical.

    agent-memory-bench's condition corpora (``corpus/conditions/<name>/seed-<n>/manifest.json``)
    drop or replace sessions of the base corpus; this collect ingests only a drop, so every
    session the condition lists must be a base session with the same sha256.
    """
    listed = condition_manifest.get("sessions")
    if not isinstance(listed, dict) or not listed:
        raise SystemExit("condition manifest lists no sessions")
    unknown = sorted(relative for relative in listed if relative not in base)
    changed = sorted(relative for relative in listed if relative in base and listed[relative] != base[relative])
    if unknown or changed:
        raise SystemExit(f"condition corpus is not a subset of the base corpus: unknown={unknown[:3]}, changed={changed[:3]}")
    return sorted(listed)


def returned_cosines(ids: list[str], dense: list[tuple[str, float]]) -> list[float | None]:
    """Each returned item's exact dense cosine to the query, or None when the dense scan lacks it."""
    by_id = dict(dense)
    return [by_id.get(item_id) for item_id in ids]


def first_relevant_rank(sessions: list[str], gold: frozenset[str]) -> int | None:
    """1-based rank of the first returned item whose session is relevant, as C6 and C7 scored."""
    return next((rank for rank, session in enumerate(sessions, start=1) if session in gold), None)


def clear_user(client, user_id: str, headers: dict[str, str]) -> None:
    """Delete everything stored under ``user_id`` before a collect, and stop if that fails."""
    response = client.post("/v1/delete", json={"user_id": user_id}, headers=headers)
    if response.status_code != 200:
        raise SystemExit(f"could not clear {user_id!r} before collecting: HTTP {response.status_code}")


def collect(args: argparse.Namespace) -> None:
    from starlette.testclient import TestClient

    import recall_aml.__main__ as hosted_main

    corpus = load_frozen_corpus(args.amb_root)
    kept = list(corpus.sessions)
    if args.condition_manifest:
        kept = condition_sessions(corpus.sessions, json.loads(args.condition_manifest.read_text(encoding="utf-8")))
    sessions = {relative: event_messages(corpus.corpus_root / relative) for relative in kept}
    mismatched = [
        relative
        for relative, messages in sessions.items()
        if " ".join(m["content"] for m in messages) != corpus.rendered[relative]
    ]
    if mismatched:
        raise SystemExit(f"event text does not rejoin to the flattened session: {mismatched[:3]}")

    expected_renderer = "message-content-only-v1"
    if args.timestamped_windows:
        served_variant = hosted_main.variant
        hosted_main.variant = lambda name: timestamped_windows(served_variant(name))  # type: ignore[assignment]
        expected_renderer = "timestamp-role-content-v1"
    expected_content = "content-v1"
    if args.dated_search_content:
        import dataclasses

        undated_variant = hosted_main.variant
        hosted_main.variant = lambda name: dataclasses.replace(  # type: ignore[assignment]
            undated_variant(name), dated_search_content=True
        )
        expected_content = "created-at-header-v1"
    if args.compile_output != "full":
        import dataclasses

        served_output_variant = hosted_main.variant
        hosted_main.variant = lambda name: dataclasses.replace(  # type: ignore[assignment]
            served_output_variant(name), anchor_compile_output=args.compile_output
        )
    if args.expect_search_content:
        # A candidate whose render adds more than the date header names its own profile (TS-1:
        # T-1 appends ``+relative-times-resolved-v1``); the check stays exact.
        expected_content = args.expect_search_content

    headers = {"Authorization": f"Bearer {os.environ['RECALL_AML_API_KEY']}"}
    dense_probe = None
    dense_store = None
    if args.dense_cosines:
        # CD-1: the served score mixes dense cosines with BM25 scores, so a relevance cut needs the
        # exact dense cosine of every returned item, read from the same tenant before it is deleted.
        from recall.embeddings import embed_query
        from recall.store import PgVectorStore
        from recall_aml.config import HostedSettings
        from recall_aml.identity import tenant_for

        settings = HostedSettings.from_env()
        embedder, _ = hosted_main._resolve_hosted_embedders(settings, hosted_main.variant(settings.variant_name))
        dense_store = PgVectorStore(settings.database_url, embedder.dim, table=settings.table,
                                    tenant=tenant_for(f"coding-window-check-{args.arm}"),
                                    generation_id=settings.generation_id)

        def dense_probe(prompt: str) -> list[tuple[str, float]]:
            hits = dense_store.query_dense_exact(embed_query(embedder, prompt), k=args.dense_k)
            return [(hit.chunk.id, float(hit.score)) for hit in hits]
    user_id = f"coding-window-check-{args.arm}"
    started = time.perf_counter()
    with TestClient(hosted_main.build_app()) as client:
        version = client.get("/version", headers=headers).json()
        if version.get("variant") != EXPECTED_VARIANT:
            raise SystemExit(f"served variant {version.get('variant')!r}")
        if version.get("window_renderer_profile") != expected_renderer:
            raise SystemExit(f"served renderer {version.get('window_renderer_profile')!r}")
        if version.get("search_content_profile", "content-v1") != expected_content:
            raise SystemExit(f"search content {version.get('search_content_profile')!r}")
        if version.get("anchor_compile_output", "full") != args.compile_output:
            raise SystemExit(f"compile output {version.get('anchor_compile_output')!r}")
        # A rerun after an abort would otherwise add under request ids the old run already
        # used, get 409s counted only as add failures, and leave the old corpus searchable.
        clear_user(client, user_id, headers)
        try:
            raw_windows = 0
            add_failures = 0
            add_latency: list[float] = []
            fallbacks = 0
            for position, (relative, messages) in enumerate(sorted(sessions.items()), start=1):
                body = {
                    "request_id": f"{args.arm}-{position:04d}",
                    "messages": messages,
                    "user_id": user_id,
                    "session_id": relative,
                }
                for attempt in range(4):
                    tick = time.perf_counter()
                    response = client.post("/v1/add", json=body, headers=headers)
                    if response.status_code < 500 or attempt == 3:
                        break
                    time.sleep(2**attempt)
                add_latency.append(1_000 * (time.perf_counter() - tick))
                if response.status_code != 200:
                    add_failures += 1
                    continue
                payload = response.json()
                raw_windows += int(payload.get("raw_count", 0))
                fallbacks += int(bool(payload.get("compiler_fallback")))
                if position % 25 == 0:
                    print(f"added {position}/{len(sessions)}", file=sys.stderr, flush=True)
            rows = []
            for task_id, prompt, gold in corpus.tasks:
                tick = time.perf_counter()
                response = client.post(
                    "/v1/search",
                    json={"query": prompt, "user_id": user_id, "top_k": 100},
                    headers=headers,
                )
                latency = 1_000 * (time.perf_counter() - tick)
                data = response.json().get("data", []) if response.status_code == 200 else []
                returned = [str(item.get("session_id", "")) for item in data]
                rows.append(
                    {
                        "task_id": task_id,
                        "status": response.status_code,
                        "route": response.headers.get("X-Recall-Specialist-Route"),
                        "latency_ms": round(latency, 1),
                        "first_relevant_rank": first_relevant_rank(returned, gold),
                        "ids": [str(item.get("id", "")) for item in data],
                        "sessions": returned,
                        "kinds": [str(item.get("kind", "")) for item in data],
                        "scores": [item.get("score") for item in data],
                        **(dense_fields(dense_probe(prompt), data) if dense_probe is not None and data else {}),
                        # The served items themselves, for a Task Solve replay of this arm (TS-1).
                        **({"top_items": [
                            {key: item.get(key) for key in ("id", "kind", "session_id", "created_at", "content")}
                            for item in data[: args.keep_items]
                        ]} if args.keep_items else {}),
                        "dated_share": (
                            sum(bool(_DATE_HEADER.match(str(item.get("content", "")))) for item in data)
                            / len(data)
                            if data
                            else 0.0
                        ),
                    }
                )
        finally:
            client.post("/v1/delete", json={"user_id": user_id}, headers=headers)
            if dense_store is not None:
                dense_store.close()
    result = {
        "preregistration": PREREGISTRATION,
        "arm": args.arm,
        "timestamped_windows": bool(args.timestamped_windows),
        "dated_search_content": bool(args.dated_search_content),
        "compile_output": args.compile_output,
        "condition_manifest": str(args.condition_manifest) if args.condition_manifest else None,
        "version": version,
        "sessions": len(sessions),
        "messages": sum(len(m) for m in sessions.values()),
        "raw_windows": raw_windows,
        "add_failures": add_failures,
        "compiler_fallbacks": fallbacks,
        "add_p50_ms": round(statistics.median(add_latency), 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    args.out.write_bytes(gzip.compress(json.dumps(result).encode("utf-8")))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, default=str))


def dense_fields(dense: list[tuple[str, float]], data: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dense_cosines": returned_cosines([str(item.get("id", "")) for item in data], dense),
        "dense_top1": dense[0][1] if dense else None,
        "dense_scanned": len(dense),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["first_relevant_rank"] for row in rows]
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "queries": len(rows),
        "failures": sum(row["status"] != 200 for row in rows),
        "mrr": statistics.fmean(0.0 if r is None else 1.0 / r for r in ranks),
        "recall_at_10": sum(r is not None and r <= 10 for r in ranks),
        "recall_at_100": sum(r is not None for r in ranks),
        "search_p95_ms": latencies[max(0, round(0.95 * len(latencies)) - 1)],
        "routes": sorted({str(row["route"]) for row in rows}),
        "mean_dated_share": statistics.fmean(float(row.get("dated_share", 0.0)) for row in rows),
    }


def paired_mrr(control: list[dict[str, Any]], treatment: list[dict[str, Any]]) -> dict[str, Any]:
    """Treatment minus control reciprocal rank, paired by task, 10,000 resamples, seed 0."""
    by_task = {row["task_id"]: row for row in control}

    def rr(rank: int | None) -> float:
        return 0.0 if rank is None else 1.0 / rank

    diffs = [
        rr(row["first_relevant_rank"]) - rr(by_task[row["task_id"]]["first_relevant_rank"])
        for row in treatment
    ]
    rng = random.Random(0)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(10_000))
    overlap = [
        len(set(row["ids"]) & set(by_task[row["task_id"]]["ids"])) / max(1, len(row["ids"]))
        for row in treatment
    ]
    return {
        "delta_mrr": sum(diffs) / n,
        "ci95": [means[250], means[9_749]],
        "tasks_better": sum(d > 0 for d in diffs),
        "tasks_worse": sum(d < 0 for d in diffs),
        "mean_top100_id_overlap": statistics.fmean(overlap),
    }


def report(args: argparse.Namespace) -> None:
    arms = {}
    for path in args.arms:
        data = json.loads(gzip.decompress(path.read_bytes()))
        arms[data["arm"]] = data
    out: dict[str, Any] = {
        arm: {
            **{k: data[k] for k in ("messages", "raw_windows", "add_failures", "compiler_fallbacks",
                                    "add_p50_ms", "total_seconds", "timestamped_windows")},
            "renderer": data["version"].get("window_renderer_profile"),
            **summary(data["rows"]),
        }
        for arm, data in arms.items()
    }
    for treatment, control in (
        ("K0b", "K0"),
        ("K1", "K0"),
        ("K1", "K0b"),
        ("K3", "K0"),
        ("K3", "K0b"),
    ):
        if treatment in arms and control in arms:
            out[f"{treatment}-vs-{control}"] = paired_mrr(
                arms[control]["rows"], arms[treatment]["rows"]
            )
    print(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("collect")
    stage.add_argument("--amb-root", type=Path, required=True)
    stage.add_argument("--arm", required=True)
    stage.add_argument("--out", type=Path, required=True)
    stage.add_argument("--timestamped-windows", action="store_true")
    stage.add_argument("--expect-search-content", default="",
                       help="the exact search_content_profile the served build must report")
    stage.add_argument("--keep-items", type=int, default=0,
                       help="store the top N served items (id, kind, session, date, content) per task")
    stage.add_argument("--dated-search-content", action="store_true")
    stage.add_argument("--compile-output", choices=("full", "lean", "select"), default="full")
    stage.add_argument("--condition-manifest", type=Path,
                       help="ingest only the sessions an agent-memory-bench condition corpus keeps (CD-1)")
    stage.add_argument("--dense-cosines", action="store_true",
                       help="store each returned item's exact dense cosine to the query (CD-1)")
    stage.add_argument("--dense-k", type=int, default=5000,
                       help="how many chunks the exact dense scan ranks per task (default 5000)")
    stage.set_defaults(run=collect)
    stage = commands.add_parser("report")
    stage.add_argument("--arms", type=Path, nargs="+", required=True)
    stage.set_defaults(run=report)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
