"""Run the deterministic graph-first retrieval probe against the remote MCP server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from mcp import ClientSession

from recall.graph_first import MAX_GRAPH_FIRST_CANDIDATES
from scripts.run_query_construction_batch import (
    _assert_generation,
    _server_command,
    _ssh_stdio_client,
    _tool_payload,
)


def _item_key(mode: str, index: int) -> str:
    return f"{mode}:{index}"


def _item_digest(item: dict[str, object]) -> str:
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _settings(args: argparse.Namespace, input_sha256: str) -> dict[str, object]:
    return {
        "input_sha256": input_sha256,
        "mode": args.mode,
        "limit": args.limit,
        "gold_class": args.gold_class,
        "k": args.k,
        "max_candidates": args.max_candidates,
        "tenant": args.tenant,
        "embedder": args.embedder,
        "index_root": args.index_root,
        "profile": args.profile,
        "pinned_generation_id": args.pinned_generation_id,
    }


def _load_checkpoint(
    path: Path, *, expected_settings: dict[str, object], resume: bool
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    if not resume:
        raise SystemExit(f"checkpoint exists; pass --resume to continue it: {path}")
    completed: dict[str, dict[str, object]] = {}
    meta_seen = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid checkpoint JSON at line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"invalid checkpoint record at line {line_number}: {path}")
            if record.get("type") == "meta":
                meta_seen = True
                if record.get("settings") != expected_settings:
                    raise SystemExit("checkpoint settings do not match this benchmark invocation")
            elif record.get("type") == "row":
                key = record.get("key")
                digest = record.get("item_sha256")
                row = record.get("row")
                if isinstance(key, str) and isinstance(digest, str) and isinstance(row, dict):
                    completed[key] = {"item_sha256": digest, "row": row}
            else:
                raise SystemExit(f"unknown checkpoint record at line {line_number}: {path}")
    if not meta_seen:
        raise SystemExit(f"checkpoint has no metadata record: {path}")
    return completed


async def main_async(args: argparse.Namespace) -> None:
    items = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("input must contain a JSON list")
    items = [dict(item) for item in items[: args.limit or None]]
    if args.gold_class:
        items = [item for item in items if item.get("gold_class") == args.gold_class]
    if not items:
        raise SystemExit("population filter selected no input items")
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.output.with_name(args.output.name + ".checkpoint.jsonl")
    checkpoint_exists = checkpoint_path.exists()
    settings = _settings(args, input_sha256)
    completed = _load_checkpoint(
        checkpoint_path, expected_settings=settings, resume=args.resume
    )
    ssh, command = _server_command(
        args.tenant,
        args.embedder,
        args.index_root,
        args.profile,
        args.pinned_generation_id,
    )
    diagnostics_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    diagnostics_path = Path(diagnostics_file.name)
    diagnostics_file.close()
    rows_by_key: dict[str, dict[str, object]] = {}
    item_digests = {
        _item_key(args.mode, index): _item_digest(item)
        for index, item in enumerate(items, start=1)
    }
    for key, saved in completed.items():
        if key in item_digests and saved["item_sha256"] == item_digests[key]:
            rows_by_key[key] = saved["row"]
    checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
    if not checkpoint_exists:
        checkpoint_handle.write(
            json.dumps({"type": "meta", "settings": settings}, ensure_ascii=False) + "\n"
        )
        checkpoint_handle.flush()
        os.fsync(checkpoint_handle.fileno())
    semaphore = asyncio.Semaphore(args.workers)

    async def run_one(index: int, item: dict[str, object]) -> tuple[str, dict[str, object]]:
        key = _item_key(args.mode, index)
        async with semaphore:
            print(f"{args.mode} {index}/{len(items)} {item.get('task_id', index)}", flush=True)
            result = await session.call_tool(
                "recall_graph_first_retrieval",
                {
                    "query": str(item["query"]),
                    "mode": args.mode,
                    "k": args.k,
                    "max_candidates": args.max_candidates,
                    **(
                        {"expected_generation_id": args.pinned_generation_id}
                        if args.pinned_generation_id
                        else {}
                    ),
                },
            )
            payload = await _tool_payload(result)
            _assert_generation(payload, args.pinned_generation_id)
            row = {
                "task_id": item.get("task_id"),
                "arm": args.mode,
                "original_prompt": item.get("original_prompt"),
                "query": item.get("query"),
                "gold": {
                    key: value
                    for key, value in item.items()
                    if key.startswith("gold") or key.endswith("_ids")
                },
                "final": payload,
                "tool_calls": [payload],
                "model_calls": [],
                "apparatus": {
                    "expected_generation_id": args.pinned_generation_id,
                    "mode": args.mode,
                },
            }
            return key, row

    try:
        diagnostics_handle = diagnostics_path.open("w", encoding="utf-8")
        try:
            async with _ssh_stdio_client(ssh, command, errlog=diagnostics_handle) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=60)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=30)
                    names = {tool.name for tool in listed.tools}
                    if "recall_graph_first_retrieval" not in names:
                        raise RuntimeError("VPS2 MCP does not expose recall_graph_first_retrieval")
                    tasks = [
                        asyncio.create_task(run_one(index, item))
                        for index, item in enumerate(items, start=1)
                        if _item_key(args.mode, index) not in rows_by_key
                    ]
                    for task in asyncio.as_completed(tasks):
                        key, row = await task
                        rows_by_key[key] = row
                        checkpoint_handle.write(
                            json.dumps(
                                {
                                    "type": "row",
                                    "key": key,
                                    "item_sha256": item_digests[key],
                                    "row": row,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        checkpoint_handle.flush()
                        os.fsync(checkpoint_handle.fileno())
        finally:
            diagnostics_handle.close()
    finally:
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        diagnostics_path.unlink(missing_ok=True)
        if diagnostics.strip():
            print(f"[server diagnostics] {diagnostics.strip()}", flush=True)
        checkpoint_handle.close()
    ordered_keys = [_item_key(args.mode, index) for index in range(1, len(items) + 1)]
    if len(rows_by_key) != len(ordered_keys):
        raise SystemExit("benchmark ended before all rows completed; resume from the checkpoint")
    artifact = {
        "artifact": "RE-call graph-first retrieval benchmark",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(args.input),
        "input_sha256": input_sha256,
        "mode": args.mode,
        "k": args.k,
        "max_candidates": args.max_candidates,
        "workers": args.workers,
        "tenant": args.tenant,
        "embedder": args.embedder,
        "profile": args.profile,
        "pinned_generation_id": args.pinned_generation_id,
        "checkpoint": str(checkpoint_path),
        "rows": [rows_by_key[key] for key in ordered_keys],
    }
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(artifact["rows"]), "output": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("entity", "relation", "hybrid"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gold-class", choices=("miss", "control"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=MAX_GRAPH_FIRST_CANDIDATES)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--pinned-generation-id")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.k < 1 or not 1 <= args.max_candidates <= MAX_GRAPH_FIRST_CANDIDATES:
        raise SystemExit("limit, k, and max-candidates must be valid")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
