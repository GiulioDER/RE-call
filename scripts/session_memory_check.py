#!/usr/bin/env python
"""Prove this checkout's memory server actually answers, before a session starts relying on it.

WHY THIS EXISTS
===============
`session-open.sh` used to report `.mcp.json present (2 servers)` and stop there. That line was
printed, truthfully, at the start of every session for days while BOTH servers were dead: the file
existed, the JSON parsed, the count was right, and not one of the servers could start. Counting
entries in a config is not a health check, and it reads exactly like one.

The cost of that gap is not a missing tool. It is that a session with no memory looks, from the
inside, identical to a session whose memory had nothing to say. Both produce silence. So the
session works from whatever it happens to remember, re-derives decisions that were already made,
and the person on the other end has to explain the same thing again -- with no signal anywhere that
a lookup was even attempted, let alone that it failed.

That ambiguity is the exact thing recall's own trust envelope exists to remove. Serving it from a
config that could rot silently was the one place the idea was not applied.

WHAT IT DISTINGUISHES
=====================
Four outcomes, and they are NOT the same:

  DEAD      the server did not start, or did not answer.        Nothing this session says about
                                                                what is or is not in memory means
                                                                anything.
  DEGRADED  it answered, but the corpus is not trusted --       Answers are coming back, but
            uncalibrated, uncertified, or bound to no           un-gated: a ranked list with no
            generation.                                         threshold behind it.
  QUIET     trusted and certified, but the canary query         Probably fine. Worth seeing,
            returned nothing.                                   because it is what a broken
                                                                embedder choice also looks like.
  OK        trusted, certified, and answering.

A wrong embedder is the reason QUIET is reported rather than ignored. Three tenants here use three
DIFFERENT 1024-dimension models, and pgvector will compute a cosine over any of them without
complaint -- so the failure is not an error, it is a confidently ranked list of the wrong things,
or nothing at all.

Usage:
  python scripts/session_memory_check.py            # check the memory server
  python scripts/session_memory_check.py --all      # every recall server in .mcp.json
  python scripts/session_memory_check.py --quiet    # one line per server, for session-open
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

# A query whose answer is in this project's own memory store by construction. It is deliberately
# about the store's own shape rather than about any one memo, so a normal week of edits cannot
# retire it.
CANARY = "how should a session pick which worktree to work in"

DEAD, DEGRADED, QUIET, OK = "DEAD", "DEGRADED", "QUIET", "OK"


def _call(entry: dict, calls: list[tuple[int, str, dict]], timeout: float) -> dict[int, dict]:
    """Run one stdio MCP session and return {id: reply} for the given tool calls.

    Two details here are load-bearing, and both were learned by getting them wrong:

    The handshake is THREE messages. A server that has answered `initialize` ignores everything
    afterwards until it receives the `notifications/initialized` notification, so a probe that
    sends only two gets a valid initialize result and then silence -- indistinguishable from a
    server that never worked.

    stdin must stay OPEN until the replies arrive. `subprocess.run(input=...)` closes it after
    writing, and a stdio server reads EOF as shutdown, so it exits with the request still in
    flight. `tools/list` survives that because it answers from memory; a search does not, because
    it waits on an embedding round trip. That failure also presents as silence, with exit code 0.
    """
    proc = subprocess.Popen(
        [entry["command"], *entry["args"]],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    out: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: [out.put(x) for x in proc.stdout], daemon=True).start()
    errs: list[str] = []
    threading.Thread(target=lambda: [errs.append(x) for x in proc.stderr], daemon=True).start()

    messages: list[dict] = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "session-memory-check", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    for call_id, tool, arguments in calls:
        messages.append({"jsonrpc": "2.0", "id": call_id, "method": "tools/call",
                         "params": {"name": tool, "arguments": arguments}})

    replies: dict[int, dict] = {}
    try:
        for message in messages:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
    except OSError:
        proc.kill()
        raise RuntimeError("".join(errs[-3:]).strip() or "server closed stdin") from None

    wanted = {call_id for call_id, _, _ in calls}
    deadline = time.monotonic() + timeout
    try:
        while wanted and time.monotonic() < deadline:
            try:
                line = out.get(timeout=1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if reply.get("id") in wanted:
                replies[reply["id"]] = reply
                wanted.discard(reply["id"])
    finally:
        proc.kill()

    if wanted:
        raise RuntimeError("".join(errs[-3:]).strip() or f"no reply to {sorted(wanted)}")
    return replies


def _body(reply: dict) -> dict:
    """The tool result, unwrapped from MCP's content envelope."""
    if "error" in reply:
        raise RuntimeError(str(reply["error"])[:200])
    blocks = reply.get("result", {}).get("content") or []
    text = blocks[0].get("text", "") if blocks else ""
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(f"non-JSON body: {text[:160]}") from None


def _one_line(exc: object) -> str:
    """Collapse a failure to one line.

    A psycopg connection failure is a multi-line ASCII-art block, and pasting it into the banner
    destroys the shape that makes the banner legible at a glance, which is the banner's only job.
    """
    parts = [ln.strip(" |+-") for ln in str(exc).splitlines()]
    return " / ".join(p for p in parts if p)[:200] or type(exc).__name__


def check(name: str, entry: dict, canary: str, timeout: float) -> dict:
    """Probe one server and classify it. Never raises: a failure IS the result."""
    started = time.monotonic()
    try:
        replies = _call(entry, [(2, "recall_stats", {}),
                                (3, "recall_search", {"query": canary})], timeout)
        stats = _body(replies[2])
        search = _body(replies[3])
    except Exception as exc:  # noqa: BLE001 - the reason is the payload, whatever its type
        return {"name": name, "verdict": DEAD, "detail": _one_line(exc),
                "seconds": time.monotonic() - started}

    hits = len(search.get("hits") or [])
    result = {
        "name": name,
        "seconds": time.monotonic() - started,
        "chunks": stats.get("chunks"),
        "newest": stats.get("newest_indexed_at"),
        "trust_state": search.get("trust_state"),
        "calibration_status": search.get("calibration_status"),
        "abstained": search.get("abstained"),
        "hits": hits,
        "detail": "",
    }

    if not stats.get("chunks"):
        result["verdict"] = DEAD
        result["detail"] = "the corpus is empty"
    elif search.get("trust_state") != "trusted" or not search.get("calibrated"):
        result["verdict"] = DEGRADED
        result["detail"] = (
            f"trust_state={search.get('trust_state')} "
            f"calibrated={search.get('calibrated')} "
            f"failure_code={search.get('failure_code')}"
        )
    elif hits == 0:
        result["verdict"] = QUIET
        result["detail"] = "certified and trusted, but the canary matched nothing"
    else:
        result["verdict"] = OK
    return result


def servers(config_path: Path, every: bool) -> dict[str, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8")).get("mcpServers") or {}
    recall = {k: v for k, v in config.items() if k.startswith("recall")}
    if every:
        return recall
    # Default to the memory server alone. The others are useful; this one is the difference
    # between a session that knows what has already been decided and one that does not, and a
    # preflight nobody waits for is a preflight nobody runs.
    return {k: v for k, v in recall.items() if k in ("recall-memory", "recall")} or recall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="check every recall server")
    parser.add_argument("--quiet", action="store_true", help="one line per server")
    parser.add_argument("--canary", default=CANARY)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    root = Path(args.config).parent if args.config else Path(
        os.environ.get("RECALL_ROOT") or Path(__file__).resolve().parents[1]
    )
    config_path = Path(args.config) if args.config else root / ".mcp.json"
    if not config_path.exists():
        print(f"  memory: no .mcp.json at {config_path}; run scripts/session-mcp.sh")
        return 1

    found = servers(config_path, args.all)
    if not found:
        print("  memory: .mcp.json has no recall server. This session has NO memory layer.")
        print("          run scripts/session-mcp.sh; if it refuses, read what it says.")
        return 1

    results = [check(name, entry, args.canary, args.timeout) for name, entry in sorted(found.items())]

    for r in results:
        if args.quiet:
            extra = f"{r['hits']} hit(s), {r['chunks']} chunks" if r["verdict"] in (OK, QUIET) else r["detail"]
            print(f"  {r['name']}: {r['verdict']} ({r['seconds']:.0f}s) {extra}")
            continue
        print(f"  {r['name']}: {r['verdict']} in {r['seconds']:.1f}s")
        if r["verdict"] == DEAD:
            print(f"    {r['detail']}")
        else:
            print(f"    chunks {r['chunks']}, newest {r['newest']}")
            print(f"    trust {r['trust_state']}/{r['calibration_status']}, canary {r['hits']} hit(s)")
            if r["detail"]:
                print(f"    {r['detail']}")

    worst = [r for r in results if r["verdict"] in (DEAD, DEGRADED)]
    if worst:
        # Loud, and shaped so it cannot be read as ordinary progress output. A degraded memory
        # layer that announces itself in an indented line is what produced this script.
        print()
        print("  " + "=" * 68)
        for r in worst:
            print(f"  MEMORY {r['verdict']}: {r['name']}: {r['detail']}")
        print("  Nothing this session concludes about what is or is not in memory is reliable.")
        print("  Do not treat an empty recall result as evidence of absence until this is fixed.")
        print("  " + "=" * 68)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
