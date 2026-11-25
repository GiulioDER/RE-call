"""Walk every recipe in references/features/ on a fresh run and check each stated end state.

The mechanical half of keeping this skill honest: if RE-call's behavior drifts from what a feature
file says, a check here fails and names the feature. It drives the app only through
`control_recall.py`, as an agent would, removes its run's scratch when done (pass or fail), keeps
the evidence, and exits 1 when any check failed. The nightly `verify-recall` workflow runs it on
master and files an issue with the report; a person or `/maintain-verification-skill` then fixes
the feature file or reports the regression.

    python .claude/skills/verify-recall/walk.py [--report walk-report.md]

When you change a recipe's end state in a feature file, change its check here in the same commit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CONTROL = [sys.executable, "-X", "utf8", str(Path(__file__).with_name("control_recall.py"))]
QUESTION = "how long do pricing snapshot cache entries live"
NEW_MEMO = "# Zephyr rollout\n\nThe zephyr rollout ships behind the zephyr flag in three waves, ten percent first.\n"


class Walk:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []
        self.named: list[str] = []

    def ctl(self, *args: str) -> tuple[dict[str, Any], int]:
        done = subprocess.run(CONTROL + list(args), capture_output=True, text=True, encoding="utf-8")
        try:
            out: dict[str, Any] = json.loads(done.stdout)
        except json.JSONDecodeError:
            out = {"error": "control_recall printed no JSON", "stderr": done.stderr[-2000:]}
        if out.get("evidence"):
            self.named.append(str(out["evidence"]))
        return out, done.returncode

    def check(self, feature: str, name: str, ok: object, detail: object = "") -> None:
        passed = bool(ok)
        self.results.append((feature, name, passed, "" if passed else json.dumps(detail, default=str)[:600]))
        print(f"{'PASS' if passed else 'FAIL'} {feature}: {name}" + ("" if passed else f"  [{self.results[-1][3]}]"))


def run(walk: Walk, run_id: str, launched: dict[str, Any]) -> None:
    w = walk
    w.check("indexing", "launch indexes the seed", launched.get("index_exit") == 0, launched)
    w.check("indexing", "hashing leaves calibration uncertified", launched.get("calibration") == "uncertified", launched)
    first = json.loads(Path(launched["evidence"]).read_text(encoding="utf-8"))["stdout"]
    w.check("indexing", "22 chunks from 22 files", "indexed 22 chunks from 22 files" in first, first[-300:])
    d, _ = w.ctl("refresh", "--run", run_id, "--check")
    report = d.get("report") or {}
    w.check("indexing", "first check is not current, 22 of 22", report.get("current") is False and report.get("chunks") == 22 and report.get("files_seen") == 22, report)

    d, rc = w.ctl("doctor", "--run", run_id)
    checks = d.get("checks", {})
    w.check("doctor", "healthy", d.get("healthy") and rc == 0, d.get("problems", d))
    w.check("doctor", "database and corpus ok, docker skipped", (checks.get("database"), checks.get("corpus"), checks.get("docker")) == ("ok", "ok", "skip"), checks)
    warned = {k for k, v in checks.items() if v == "warn"}
    # `claude code` warns where a Claude Code configuration exists and is skipped where none does.
    w.check("doctor", "calibration warns, and nothing else but claude code", "calibration" in warned and warned <= {"calibration", "claude code"}, checks)
    w.check("doctor", "the run's own embedder is checked", checks.get("embedder") == "ok", checks)

    d, _ = w.ctl("search", "--run", run_id, QUESTION)
    w.check("search", "strict search refuses with CALIBRATION_UNCERTIFIED", (d.get("exit"), d.get("code"), d.get("rows")) == (1, "CALIBRATION_UNCERTIFIED", []), d)
    d, _ = w.ctl("search", "--run", run_id, "--development", QUESTION)
    rows = {r["source"]: r for r in d.get("rows", [])}
    w.check("search", "development search returns up to five rows", d.get("exit") == 0 and 1 <= len(rows) <= 5, d)
    v1, v2 = rows.get("cache_ttl_v1.md", {}), rows.get("cache_ttl_v2.md", {})
    w.check("supersession", "v2 ok, v1 superseded and pointing at v2", (v2.get("state"), v1.get("state"), v1.get("use")) == ("ok", "superseded", "cache_ttl_v2.md"), rows)
    w.check("supersession", "v1 is the nearer match", v1 and v2 and float(v1["cos"]) > float(v2["cos"]), rows)

    d, _ = w.ctl("mcp", "--run", run_id, "recall_search", "--args", json.dumps({"query": QUESTION}))
    text = (d.get("text") or [""])[0]
    w.check("mcp-server", "recall_search is served", "recall_search" in d.get("tools", []), d.get("tools", d))
    w.check("mcp-server", "strict call is a trust_refusal", d.get("is_error") and '"error": "trust_refusal"' in text and "CALIBRATION_" in text, text[:400])
    d, _ = w.ctl("mcp", "--run", run_id, "--development", "recall_search", "--args", json.dumps({"query": QUESTION}))
    structured = d.get("structured") or {}
    hits = structured.get("hits") or []
    w.check("mcp-server", "development call is degraded with hits", not d.get("is_error") and structured.get("trust_state") == "degraded" and hits, structured.get("trust_state"))
    w.check("supersession", "over MCP, v2 first and v1 superseded_by v2",
            hits and hits[0].get("source") == "cache_ttl_v2.md" and any(h.get("source") == "cache_ttl_v1.md" and h.get("superseded_by") == "cache_ttl_v2.md" for h in hits),
            [(h.get("source"), h.get("superseded_by")) for h in hits])

    w.ctl("refresh", "--run", run_id)
    d, _ = w.ctl("refresh", "--run", run_id, "--check")
    w.check("lite", "current after a refresh", (d.get("report") or {}).get("current") is True, d.get("report"))
    w.ctl("write-memo", "--run", run_id, "zephyr_rollout.md", NEW_MEMO)
    d, _ = w.ctl("refresh", "--run", run_id, "--check")
    report = d.get("report") or {}
    w.check("lite", "a new memo makes it not current, 23 files", report.get("current") is False and report.get("files_seen") == 23, report)
    d, _ = w.ctl("refresh", "--run", run_id)
    report = d.get("report") or {}
    w.check("lite", "refresh indexes 1 file, 1 chunk, 23 total", (report.get("indexed_files"), report.get("chunks_written"), report.get("chunks_total")) == (1, 1, 23), report)
    d, _ = w.ctl("search", "--run", run_id, "--development", "zephyr rollout waves")
    w.check("lite", "the new memo ranks first", (d.get("rows") or [{}])[0].get("source") == "zephyr_rollout.md", (d.get("rows") or [d])[:1])
    d, _ = w.ctl("search", "--run", run_id, QUESTION)
    w.check("search", "after a corpus change the code is CALIBRATION_STALE", d.get("code") == "CALIBRATION_STALE", d.get("code"))

    d, _ = w.ctl("forget", "--run", run_id, "zephyr_rollout.md")
    w.check("forget", "preview deletes nothing", not d.get("applied") and "would forget 1 source(s)" in d.get("stdout", ""), d.get("stdout", d))
    d, _ = w.ctl("forget", "--run", run_id, "zephyr_rollout.md", "--yes")
    w.check("forget", "--yes forgets 1 chunk", "forgot 1 chunk(s) from 1 source(s)." in d.get("stdout", ""), d.get("stdout", d))
    d, _ = w.ctl("search", "--run", run_id, "--development", "zephyr rollout waves")
    w.check("forget", "the memo is gone from search", all(r.get("source") != "zephyr_rollout.md" for r in d.get("rows", [])), d.get("rows", d))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, help="also write a Markdown report here")
    args = parser.parse_args(argv)
    walk = Walk()
    launched, _ = walk.ctl("launch")
    run_id = str(launched.get("run", ""))
    try:
        if run_id:
            run(walk, run_id, launched)
        else:
            walk.check("indexing", "launch printed a run id", False, launched)
    except Exception as error:  # BROAD-CATCH: error-translation  # a crash is a failed walk, reported like one
        walk.check("walk", "the walk completed", False, f"{type(error).__name__}: {error}")
    finally:
        if run_id:
            cleaned, _ = walk.ctl("cleanup", "--run", run_id)
            kept = cleaned.get("evidence_dir", "")
            walk.check("cleanup", "every evidence file the walk named survives",
                       walk.named and all(Path(e).is_file() for e in walk.named), (len(walk.named), kept))
    failed = [r for r in walk.results if not r[2]]
    summary = f"run {run_id}: {len(walk.results) - len(failed)} of {len(walk.results)} checks passed"
    print("\n" + summary)
    if args.report:
        lines = [f"# verify-recall walk: {'FAILED' if failed else 'passed'}", "", summary, ""]
        lines += ["| Feature | Check | Detail |", "|---|---|---|"]
        lines += [f"| {f} | {n} | `{d}` |" for f, n, _, d in failed] or ["| | all checks passed | |"]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
