"""Drive RE-call the way a user does, on a throwaway lite store, and keep the evidence.

Every command prints one JSON object on stdout and exits 0 when the command ran, 1 when what it
drove failed (the JSON says how), and 2 when it was used wrongly (the JSON's ``fix`` says what to do
instead). Every drive also writes its argv, stdout, stderr and exit code to the run's
``evidence/`` directory, which ``cleanup`` never removes.

A run lives under ``<temp>/recall-verify/<run id>/``: ``scratch/`` holds the store and the memo
folder, ``evidence/`` holds the proof, ``run.json`` names the checkout that created it. Commands
refuse a run another checkout created. The code driven is THIS checkout's (``python -m`` with it as
the working directory), every inherited ``RECALL_*`` variable is dropped, and the embedding cache
is off, so a run cannot touch real memory, client configuration or another run.

    python .claude/skills/verify-recall/control_recall.py launch
    python .claude/skills/verify-recall/control_recall.py doctor --run RUN
    python .claude/skills/verify-recall/control_recall.py search --run RUN "a question"
    python .claude/skills/verify-recall/control_recall.py cleanup --run RUN
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[3]
ROOT = Path(tempfile.gettempdir()) / "recall-verify"
SEED = CHECKOUT / "recall" / "eval" / "corpus"
ROW = re.compile(r"^  (?P<state>\w+)\s+conf=(?P<conf>[\d.]+) cos=(?P<cos>[\d.]+)\s+(?P<source>\S+)(?: -> use (?P<use>\S+))?")
CODE = re.compile(r"^(?:\[DEGRADED:)?(?P<code>[A-Z][A-Z_]+)[\]:]")


class Usage(Exception):
    def __init__(self, error: str, fix: str) -> None:
        super().__init__(error)
        self.fix = fix


def _emit(payload: dict[str, Any], status: int) -> int:
    print(json.dumps(payload, indent=1, default=str))
    return status


def _run_dir(run_id: str) -> Path:
    path = (ROOT / run_id).resolve()
    if path.parent != ROOT.resolve() or not (path / "run.json").is_file():
        raise Usage(f"no run {run_id!r} under {ROOT}", "run `launch`, or `list` to see existing runs")
    owner = json.loads((path / "run.json").read_text(encoding="utf-8"))["checkout"]
    if Path(owner) != CHECKOUT:
        raise Usage(f"run {run_id} belongs to checkout {owner}", "drive only runs this checkout launched; `launch` a new one")
    return path


def _env(run: Path | None, *, development: bool = False) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RECALL_")}
    env.update(RECALL_EMBED_CACHE="off", PYTHONPATH=str(CHECKOUT), PYTHONUTF8="1")
    if run is not None:
        env["RECALL_SERVING_DSN"] = f"sqlite:///{(run / 'scratch' / 'memory.db').as_posix()}"
        env["RECALL_INDEX_ROOT"] = str(run / "scratch" / "notes")
    if development:
        env["RECALL_TRUST_MODE"] = "development"
    return env


def _record(run: Path, name: str, payload: dict[str, Any]) -> Path:
    evidence = run / "evidence"
    evidence.mkdir(exist_ok=True)
    path = evidence / f"{len(list(evidence.iterdir())) + 1:03d}-{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def _recall(run: Path, name: str, args: list[str], *, development: bool = False, embedder: str | None = None) -> dict[str, Any]:
    argv = [sys.executable, "-X", "utf8", "-m", "recall.cli", "--serving-dsn", _env(run)["RECALL_SERVING_DSN"]]
    argv += ["--embedder", embedder] if embedder else []
    started = time.monotonic()
    done = subprocess.run(argv + args, cwd=CHECKOUT, env=_env(run, development=development),
                          capture_output=True, text=True, encoding="utf-8")
    result = {"argv": argv + args, "exit": done.returncode, "stdout": done.stdout, "stderr": done.stderr,
              "seconds": round(time.monotonic() - started, 2), "development": development}
    result["evidence"] = str(_record(run, name, result))
    return result


def launch(args: argparse.Namespace) -> int:
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    run = ROOT / run_id
    (run / "scratch").mkdir(parents=True)
    shutil.copytree(SEED, run / "scratch" / "notes")
    (run / "run.json").write_text(json.dumps({"checkout": str(CHECKOUT), "created": time.time(), "embedder": args.embedder}), encoding="utf-8")
    done = _recall(run, "launch-index", ["index", str(run / "scratch" / "notes")], embedder=args.embedder)
    calibration = next((line.split(":", 1)[1].split(":")[0].strip() for line in done["stdout"].splitlines() if line.startswith("calibration:")), None)
    return _emit({"run": run_id, "store": str(run / "scratch" / "memory.db"), "notes": str(run / "scratch" / "notes"),
                  "index_exit": done["exit"], "calibration": calibration, "evidence": done["evidence"]}, 0 if done["exit"] == 0 else 1)


def doctor(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    # Doctor checks the embedder it is told, else the default fastembed: name the run's own.
    embedder = json.loads((run / "run.json").read_text(encoding="utf-8"))["embedder"]
    done = _recall(run, "doctor", ["doctor", "--json", "--project-root", str(run / "scratch")], embedder=embedder)
    checks = {c["name"]: c for c in json.loads(done["stdout"] or '{"checks": []}')["checks"]}
    package = checks.get("recall package", {}).get("detail", "")
    problems = [] if str(CHECKOUT / "recall") in package else [f"recall loaded from elsewhere: {package}"]
    problems += [f"{n}: {c['detail']}" for n, c in checks.items() if c["status"] == "fail"]
    problems += [] if (run / "scratch" / "memory.db").is_file() else ["the run's store is missing; `launch` again"]
    return _emit({"run": args.run, "healthy": not problems, "problems": problems,
                  "checks": {n: c["status"] for n, c in checks.items()}, "evidence": done["evidence"]}, 0 if not problems else 1)


def search(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    done = _recall(run, "search", ["search", args.query], development=args.development)
    lines = done["stdout"].splitlines()
    # A strict refusal is written to stderr and a degraded answer to stdout; either names the code.
    code = next((m["code"] for line in lines + done["stderr"].splitlines() if (m := CODE.match(line))), None)
    rows = [{k: v for k, v in m.groupdict().items() if v is not None} for line in lines if (m := ROW.match(line))]
    return _emit({"run": args.run, "query": args.query, "exit": done["exit"], "code": code, "rows": rows,
                  "development": args.development, "evidence": done["evidence"]}, 0)


def refresh(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    flags = ["--check"] if args.check else []
    done = _recall(run, "refresh", ["lite", "refresh", "--store", str(run / "scratch" / "memory.db"),
                                    "--memory", str(run / "scratch" / "notes"), "--json", *flags])
    # `--check` exits 1 for "not current" with a full report, so the report decides, not the exit.
    report = json.loads(done["stdout"]) if done["stdout"].strip().startswith("{") else None
    return _emit({"run": args.run, "exit": done["exit"], "report": report, "stderr": done["stderr"][-800:],
                  "evidence": done["evidence"]}, 0 if report is not None else 1)


def write_memo(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    notes = run / "scratch" / "notes"
    target = (notes / args.name).resolve()
    if target.parent != notes.resolve() or target.suffix != ".md":
        raise Usage(f"{args.name!r} is not a .md file directly in the run's memo folder", "pass a bare name such as new_memo.md")
    existed = target.exists()
    target.write_text(args.text, encoding="utf-8")
    path = _record(run, "write-memo", {"name": args.name, "replaced": existed, "text": args.text})
    return _emit({"run": args.run, "memo": str(target), "replaced": existed, "evidence": str(path)}, 0)


def forget(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    # The store names a source by its native absolute path; a bare memo name means one in the run.
    sources = [str(run / "scratch" / "notes" / s) if Path(s).name == s else s for s in args.sources]
    done = _recall(run, "forget", ["forget", *sources, *(["--yes"] if args.yes else [])])
    return _emit({"run": args.run, "exit": done["exit"], "applied": args.yes, "stdout": done["stdout"][-1500:],
                  "stderr": done["stderr"][-800:], "evidence": done["evidence"]}, 0 if done["exit"] == 0 else 1)


def mcp(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as error:
        raise Usage(f"the MCP client is not installed: {error}", 'pip install -e ".[mcp]" in this checkout') from error
    tool_args = json.loads(args.args)

    async def drive() -> dict[str, Any]:
        params = StdioServerParameters(command=sys.executable, args=["-X", "utf8", "-m", "recall_mcp.server"],
                                       env=_env(run, development=args.development) | {"RECALL_EMBEDDER": json.loads((run / "run.json").read_text(encoding="utf-8"))["embedder"]},
                                       cwd=str(CHECKOUT))
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            if args.tool not in tools:
                return {"tools": tools, "error": f"tool {args.tool!r} is not served"}
            result = await session.call_tool(args.tool, tool_args)
            structured = result.structured_content
            # The server wraps its JSON answer as a string under `result`; hand back the object.
            if isinstance(structured, dict) and isinstance(structured.get("result"), str) and structured["result"].lstrip().startswith("{"):
                structured = json.loads(structured["result"])
            return {"tools": tools, "is_error": result.is_error, "structured": structured,
                    "text": [getattr(c, "text", "") for c in result.content]}

    started = time.monotonic()
    out = asyncio.run(drive())
    out |= {"tool": args.tool, "arguments": tool_args, "seconds": round(time.monotonic() - started, 2)}
    out["evidence"] = str(_record(run, f"mcp-{args.tool}", out))
    return _emit({"run": args.run, **out}, 1 if "error" in out else 0)


def list_runs(args: argparse.Namespace) -> int:
    runs = []
    for marker in sorted(ROOT.glob("*/run.json")) if ROOT.is_dir() else []:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        runs.append({"run": marker.parent.name, "mine": Path(meta["checkout"]) == CHECKOUT,
                     "scratch": (marker.parent / "scratch").exists(), "evidence": len(list((marker.parent / "evidence").glob("*.json")))})
    return _emit({"root": str(ROOT), "runs": runs}, 0)


def cleanup(args: argparse.Namespace) -> int:
    run = _run_dir(args.run)
    scratch = run / "scratch"
    if args.dry_run:
        return _emit({"run": args.run, "would_remove": str(scratch), "keeps": str(run / "evidence")}, 0)
    if scratch.exists():
        shutil.rmtree(scratch)
    kept = sorted(p.name for p in (run / "evidence").glob("*.json"))
    return _emit({"run": args.run, "removed": str(scratch), "evidence_dir": str(run / "evidence"), "evidence_kept": kept}, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__.split("\n\n", 1)[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("launch", help="new run: seed the memo folder from recall/eval/corpus and index it")
    p.add_argument("--embedder", default="hashing", help="default hashing (no model, never certifies); a real model only where embedding is allowed")
    p.set_defaults(handler=launch)
    p = sub.add_parser("doctor", help="read-only: is this run worth driving? (`recall doctor --json` plus checkout and store checks)")
    p.add_argument("--run", required=True)
    p.set_defaults(handler=doctor)
    p = sub.add_parser("search", help="`recall search`; returns the refusal code and the parsed result rows")
    p.add_argument("--run", required=True)
    p.add_argument("--development", action="store_true", help="RECALL_TRUST_MODE=development: results come back DEGRADED, for ranking checks only")
    p.add_argument("query")
    p.set_defaults(handler=search)
    p = sub.add_parser("refresh", help="`recall lite refresh --json` over the run's memo folder")
    p.add_argument("--run", required=True)
    p.add_argument("--check", action="store_true", help="report what a refresh would find; change nothing")
    p.set_defaults(handler=refresh)
    p = sub.add_parser("write-memo", help="create or replace one memo in the run's memo folder")
    p.add_argument("--run", required=True)
    p.add_argument("name")
    p.add_argument("text")
    p.set_defaults(handler=write_memo)
    p = sub.add_parser("forget", help="`recall forget`; previews unless --yes")
    p.add_argument("--run", required=True)
    p.add_argument("--yes", action="store_true", help="actually delete; without it this is a dry run")
    p.add_argument("sources", nargs="+")
    p.set_defaults(handler=forget)
    p = sub.add_parser("mcp", help="start the MCP server over stdio on the run's store, list tools, call one")
    p.add_argument("--run", required=True)
    p.add_argument("--development", action="store_true")
    p.add_argument("tool")
    p.add_argument("--args", default="{}", help="the tool's arguments as a JSON object")
    p.set_defaults(handler=mcp)
    p = sub.add_parser("list", help="every run under the verify root, and which are this checkout's")
    p.set_defaults(handler=list_runs)
    p = sub.add_parser("cleanup", help="remove a run's scratch (store and memos); its evidence stays")
    p.add_argument("--run", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(handler=cleanup)
    args = parser.parse_args(argv)
    try:
        status: int = args.handler(args)
        return status
    except Usage as error:
        return _emit({"error": str(error), "fix": error.fix}, 2)


if __name__ == "__main__":
    sys.exit(main())
