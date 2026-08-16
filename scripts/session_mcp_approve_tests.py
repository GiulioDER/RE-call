"""Regression tests for the MCP approval step.

Every test names the defect it pins. The refusals have controls, because a script that approved
nothing at all would otherwise pass half this file, and "it did not break anything" is exactly
how a guard that does nothing looks.

The client config under test is always a temp file, selected through
RECALL_MCP_CLIENT_CONFIG. Nothing here reads or writes the real `~/.claude.json`.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent / "session_mcp_approve.py")
SCRATCH = Path(tempfile.gettempdir()) / "recall-mcp-approve-tests"

results = []


def load():
    spec = importlib.util.spec_from_file_location("approve_mod", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def fresh(config: dict) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=SCRATCH, suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps(config), encoding="utf-8")
    return Path(path)


def run(cfg: Path, root: str, *servers: str, check_only=False):
    env = dict(os.environ)
    env["RECALL_MCP_CLIENT_CONFIG"] = str(cfg)
    argv = [sys.executable, SCRIPT, "--root", root, "--servers", *servers]
    if check_only:
        argv.append("--check")
    return subprocess.run(argv, capture_output=True, text=True, env=env)


def read(cfg: Path) -> dict:
    return json.loads(cfg.read_text(encoding="utf-8"))


def test_a_pending_server_becomes_approved():
    """The point of the whole script: a checkout with no approval gets one."""
    root = str(SCRATCH / "wt1")
    cfg = fresh({"projects": {}})
    r = run(cfg, root, "recall", "recall-memory")
    entry = read(cfg)["projects"].get(os.path.normpath(root), {})
    got = entry.get("enabledMcpjsonServers")
    check("a pending server becomes approved",
          r.returncode == 0 and got == ["recall", "recall-memory"],
          f"rc={r.returncode} enabled={got} err={r.stderr.strip()[:200]}")


def test_a_forward_slash_root_updates_the_clients_own_key():
    """Pins the silent-no-op: git hands back C:/x, the client stores C:\\x.

    Keyed on the raw git form this writes a SECOND project entry that the client never reads.
    Everything reports success and no server ever loads, which is indistinguishable from the
    bug this script was written to fix.
    """
    native = os.path.normpath(str(SCRATCH / "wt2"))
    slashed = native.replace(os.sep, "/")
    cfg = fresh({"projects": {native: {"enabledMcpjsonServers": [], "allowedTools": ["keep-me"]}}})
    run(cfg, slashed, "recall")
    projects = read(cfg)["projects"]
    entry = projects.get(native, {})
    check("a forward-slash root updates the client's own key",
          len(projects) == 1 and entry.get("enabledMcpjsonServers") == ["recall"],
          f"keys={list(projects)} entry={entry}")


def test_it_is_idempotent():
    """A session opens many times. The second run must not duplicate the names."""
    root = str(SCRATCH / "wt3")
    cfg = fresh({"projects": {}})
    run(cfg, root, "recall", "recall-memory")
    r2 = run(cfg, root, "recall", "recall-memory")
    got = read(cfg)["projects"][os.path.normpath(root)]["enabledMcpjsonServers"]
    check("it is idempotent",
          got == ["recall", "recall-memory"] and "already in place" in r2.stdout,
          f"enabled={got} stdout={r2.stdout.strip()[:160]}")


def test_an_explicit_disable_is_not_reversed():
    """An entry in disabledMcpjsonServers is a decision, not a gap to be filled."""
    root = str(SCRATCH / "wt4")
    cfg = fresh({"projects": {os.path.normpath(root): {
        "enabledMcpjsonServers": [], "disabledMcpjsonServers": ["recall"]}}})
    r = run(cfg, root, "recall")
    entry = read(cfg)["projects"][os.path.normpath(root)]
    check("an explicit disable is not reversed",
          entry["enabledMcpjsonServers"] == [] and entry["disabledMcpjsonServers"] == ["recall"]
          and "disabledMcpjsonServers" in r.stderr,
          f"entry={entry} err={r.stderr.strip()[:160]}")


def test_control_a_sibling_of_a_disabled_server_is_still_approved():
    """CONTROL for the test above: refusing everything must not read as respecting a decision."""
    root = str(SCRATCH / "wt5")
    cfg = fresh({"projects": {os.path.normpath(root): {
        "enabledMcpjsonServers": [], "disabledMcpjsonServers": ["recall"]}}})
    run(cfg, root, "recall", "recall-memory")
    entry = read(cfg)["projects"][os.path.normpath(root)]
    check("CONTROL a sibling of a disabled server is still approved",
          entry["enabledMcpjsonServers"] == ["recall-memory"],
          f"entry={entry}")


def test_unrelated_projects_survive():
    """This file holds 306 projects and the client's whole account state. Touch one key."""
    root = str(SCRATCH / "wt6")
    cfg = fresh({"projects": {"C:\\other": {"allowedTools": ["a"], "enabledMcpjsonServers": ["x"]}},
                 "userID": "sentinel", "oauthAccount": {"k": "v"}})
    run(cfg, root, "recall")
    after = read(cfg)
    check("unrelated projects survive",
          after["userID"] == "sentinel" and after["oauthAccount"] == {"k": "v"}
          and after["projects"]["C:\\other"] == {"allowedTools": ["a"],
                                                "enabledMcpjsonServers": ["x"]},
          f"after={json.dumps(after)[:240]}")


def test_no_server_definition_crosses_the_boundary():
    """Only names. A URL or a token in the client config is the disclosure this repo avoids."""
    root = str(SCRATCH / "wt7")
    cfg = fresh({"projects": {}})
    run(cfg, root, "recall", "recall-memory")
    blob = cfg.read_text(encoding="utf-8")
    leaked = [s for s in ("http://", "https://", "Bearer", "postgresql://", "RECALL_DSN")
              if s in blob]
    check("no server definition crosses the boundary", not leaked, f"leaked={leaked}")


def test_a_corrupt_config_is_left_intact():
    """An unreadable client config is the client's problem. Never truncate it, never fail open
    into a rewrite: that would cost the user their account state to fix a convenience."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=SCRATCH, suffix=".json")
    os.close(fd)
    Path(path).write_text('{"projects": {"a": ', encoding="utf-8")  # truncated JSON
    before = Path(path).read_text(encoding="utf-8")
    r = run(Path(path), str(SCRATCH / "wt8"), "recall")
    check("a corrupt config is left intact",
          r.returncode == 0 and Path(path).read_text(encoding="utf-8") == before,
          f"rc={r.returncode} after={Path(path).read_text(encoding='utf-8')[:80]}")


def test_a_missing_config_is_not_an_error():
    """A machine where the client has never run must not fail a session open."""
    missing = SCRATCH / "definitely-not-here.json"
    if missing.exists():
        missing.unlink()
    r = run(missing, str(SCRATCH / "wt9"), "recall")
    check("a missing config is not an error",
          r.returncode == 0 and not missing.exists(),
          f"rc={r.returncode} created={missing.exists()}")


def test_check_mode_writes_nothing():
    root = str(SCRATCH / "wt10")
    cfg = fresh({"projects": {}})
    before = cfg.read_text(encoding="utf-8")
    r = run(cfg, root, "recall", check_only=True)
    check("check mode writes nothing",
          r.returncode == 0 and cfg.read_text(encoding="utf-8") == before
          and "would approve" in r.stdout,
          f"rc={r.returncode} stdout={r.stdout.strip()[:160]}")


def test_names_are_read_back_out_of_the_generated_file():
    """The caller must not carry its own copy of the server list: a rename would then leave a
    server that is never approved, with nothing reporting a problem."""
    root = str(SCRATCH / "wt12")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    mcp = SCRATCH / "generated.mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"recall": {}, "recall-memory": {}}}),
                   encoding="utf-8")
    cfg = fresh({"projects": {}})
    env = dict(os.environ)
    env["RECALL_MCP_CLIENT_CONFIG"] = str(cfg)
    r = subprocess.run([sys.executable, SCRIPT, "--root", root, "--from-mcp-json", str(mcp)],
                       capture_output=True, text=True, env=env)
    got = read(cfg)["projects"].get(os.path.normpath(root), {}).get("enabledMcpjsonServers")
    check("names are read back out of the generated file",
          r.returncode == 0 and got == ["recall", "recall-memory"],
          f"rc={r.returncode} enabled={got} err={r.stderr.strip()[:200]}")


def test_a_backup_is_written_before_the_edit():
    root = str(SCRATCH / "wt11")
    cfg = fresh({"projects": {}, "userID": "sentinel"})
    before = cfg.read_text(encoding="utf-8")
    run(cfg, root, "recall")
    bak = Path(str(cfg) + ".session-mcp.bak")
    check("a backup is written before the edit",
          bak.exists() and bak.read_text(encoding="utf-8") == before,
          f"bak_exists={bak.exists()}")


if __name__ == "__main__":
    for fn in (test_a_pending_server_becomes_approved,
               test_a_forward_slash_root_updates_the_clients_own_key,
               test_it_is_idempotent,
               test_an_explicit_disable_is_not_reversed,
               test_control_a_sibling_of_a_disabled_server_is_still_approved,
               test_unrelated_projects_survive,
               test_no_server_definition_crosses_the_boundary,
               test_a_corrupt_config_is_left_intact,
               test_a_missing_config_is_not_an_error,
               test_check_mode_writes_nothing,
               test_names_are_read_back_out_of_the_generated_file,
               test_a_backup_is_written_before_the_edit):
        try:
            fn()
        except Exception as exc:
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
