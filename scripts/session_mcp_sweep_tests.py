#!/usr/bin/env python3
"""Tests for the MCP sweep, which kills processes on a shared host.

Every test here is about ONE question: which servers does it select? The process tables are
fixtures and the killer is a log, so nothing needs VPS2, ssh, or a running client.

The fixtures are shaped from the real fleet measured 2026-08-26, because the hazard is real there:
of 18 live servers, 16 were launched by a DIFFERENT agent on this same workstation with a command
line that differs from ours only in the checkout it enters. Test 2 is the one that pins those, and
it is the reason the sweep keys on a mark rather than on a pattern.

Mutation-tested 2026-08-26, six ways, each watched to go red. These are the MEASURED reds, which
are wider than the ones I predicted, because a mis-selection shows up in the CLI tests as well as
in the classifier:

    the host check `s.mark.startswith(host + "-")` -> False    1, 3, 8, 9 red
    the held check `s.mark in marks` -> False                  1, 4b, 8, 9 red
    the unmarked gate ignores `unmarked_ours_local`            5 red
    `SERVER_RE` unanchored (the double-count defect)           7 red
    the "ours"/"other" shape test inverted                     2, 4, 5b red
    `parse_remote` accepts a short row                         the parse test raises, and is
                                                               reported as a harness failure rather
                                                               than a clean red: unpacking five
                                                               fields into six names is a
                                                               ValueError, which is still a
                                                               failure, and saying so beats
                                                               claiming a tidy assertion
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SWEEP = str(Path(__file__).resolve().parent / "session_mcp_sweep.py")
SCRATCH = Path(tempfile.gettempdir()) / "recall-sweeptests"
results = []


def load():
    spec = importlib.util.spec_from_file_location("sweep", SWEEP)
    m = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec_module, which is not a formality here: with
    # `from __future__ import annotations` the @dataclass decorator resolves its field types by
    # looking the module up in sys.modules, and an unregistered module makes that lookup return
    # None. The whole file then failed with "'NoneType' object has no attribute '__dict__'" while
    # the code under test was perfectly fine.
    sys.modules["sweep"] = m
    spec.loader.exec_module(m)
    return m


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))


def write(name: str, body: str) -> str:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    p = SCRATCH / name
    p.write_text(body, encoding="utf-8")
    return str(p)


# `pid ppid rss etimes mark shape`, tab separated, as the remote probe emits it.
def remote_row(pid, wpid, rss, age, mark, shape):
    return "\t".join([pid, wpid, str(rss), str(age), mark, shape])


FLEET = "\n".join([
    remote_row("101", "100", 834000, 50000, "Bot-6f131daf", "ours"),    # ours, no transport left
    remote_row("111", "110", 830000, 40000, "Bot-aaaaaaaa", "ours"),    # ours, transport alive
    remote_row("121", "120", 820000, 60000, "Laptop-bbbb", "ours"),     # another machine's
    remote_row("131", "130", 815000, 49000, "-", "ours"),               # ours, pre-marker
    remote_row("141", "140", 812000, 48000, "-", "other"),              # the other agent's
]) + "\n"

MCP = "recall_mcp.server"
LOCAL_WITH_LIVE_MARK = (
    f"900 1 claude.exe\n"
    f"901 900 ssh vps2 cd ~/recall-repos/serving && RECALL_MCP_CLIENT=Bot-aaaaaaaa exec python -m {MCP}\n"
    f"801 800 ssh vps2 cd ~/recall-repos && exec python -m {MCP}\n"   # other agent's, unmarked
)
LOCAL_WITH_UNMARKED_OURS = (
    f"901 900 ssh vps2 cd ~/recall-repos/serving && exec python -m {MCP}\n"  # ours, pre-marker
)


def test_classification():
    m = load()
    servers = m.parse_remote(FLEET)
    check("0  every fixture row parses", len(servers) == 5, f"{len(servers)} parsed")
    marks, unmarked_ours = m.local_transports(LOCAL_WITH_LIVE_MARK)
    check("0b live marks are read from the local table, and the other agent's is not counted",
          marks == {"Bot-aaaaaaaa"} and unmarked_ours == 0, f"{marks} {unmarked_ours}")
    b = m.classify(servers, marks, "Bot", unmarked_ours)
    check("1  a mark with no live transport is an orphan",
          [s.pid for s in b["orphan"]] == ["101"], str([s.pid for s in b["orphan"]]))
    check("2  another agent's server is never in the orphan bucket",
          "141" not in [s.pid for s in b["orphan"]]
          and [s.pid for s in b["unmarked_other"]] == ["141"], str(b["unmarked_other"]))
    check("3  a mark naming another machine is left to that machine",
          [s.pid for s in b["other_host"]] == ["121"], str([s.pid for s in b["other_host"]]))
    check("4  a pre-marker server of ours is held back for --unmarked",
          [s.pid for s in b["unmarked_ours"]] == ["131"], str(b["unmarked_ours"]))
    check("4b a mark held by a live transport is left alone",
          [s.pid for s in b["held"]] == ["111"], str([s.pid for s in b["held"]]))


def test_unmarked_gate():
    """The gate that makes --unmarked safe rather than a guess."""
    m = load()
    servers = m.parse_remote(FLEET)
    marks, unmarked_ours = m.local_transports(LOCAL_WITH_UNMARKED_OURS)
    check("5a an unmarked transport of our shape is counted", unmarked_ours == 1, str(unmarked_ours))
    b = m.classify(servers, marks, "Bot", unmarked_ours)
    check("5  while an unmarked transport of ours is live, nothing pre-marker is sweepable",
          b["unmarked_ours_sweepable"] == [], str(b["unmarked_ours_sweepable"]))
    b2 = m.classify(servers, marks, "Bot", 0)
    check("5b with none live, the pre-marker servers become sweepable",
          [s.pid for s in b2["unmarked_ours_sweepable"]] == ["131"],
          str(b2["unmarked_ours_sweepable"]))


def test_parse_rejects_a_short_row():
    """A truncated line must be dropped, not padded into a Server with the wrong fields."""
    m = load()
    bad = "101\t100\t834000\t50000\tBot-6f131daf\n"     # five fields, no shape
    check("6  a malformed remote row is dropped rather than half-read",
          m.parse_remote(bad) == [], str(m.parse_remote(bad)))


def test_a_wrapper_is_not_a_server():
    """The double-count defect, pinned where it can be seen.

    Both the server and its ssh wrapper carry the string `python -m recall_mcp.server`: the server
    as the command it runs, the wrapper inside `--cmd=...`. An unanchored match counts each server
    twice, which is how "89 servers" was published when there were about 44.
    """
    m = load()
    server = ".venv/bin/python -m recall_mcp.server"
    wrapper = ("/usr/sbin/tailscaled be-child ssh --cmd=cd ~/recall-repos/serving && "
               "exec .venv/bin/python -m recall_mcp.server")
    check("7  the server command matches and its wrapper does not",
          bool(m.SERVER_RE.match(server)) and not m.SERVER_RE.match(wrapper),
          f"server={bool(m.SERVER_RE.match(server))} wrapper={bool(m.SERVER_RE.match(wrapper))}")


def run_cli(*args, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, SWEEP, *args], capture_output=True, text=True, env=env)
    return p


def test_cli_reports_without_killing():
    kills = SCRATCH / "kills.txt"
    if kills.exists():
        kills.unlink()
    p = run_cli("--client-host", "Bot", env_extra={
        "RECALL_MCP_REMOTE_FILE": write("fleet.tsv", FLEET),
        "RECALL_MCP_PS_FILE": write("local.txt", LOCAL_WITH_LIVE_MARK),
        "RECALL_MCP_KILL_FILE": str(kills),
    })
    check("8  a report names one closeable server and kills nothing",
          "1 server(s) closeable" in p.stdout and not kills.exists(), p.stdout.strip()[-200:])


def test_cli_kills_only_the_orphan():
    kills = SCRATCH / "kills2.txt"
    if kills.exists():
        kills.unlink()
    p = run_cli("--kill", "--client-host", "Bot", env_extra={
        "RECALL_MCP_REMOTE_FILE": write("fleet2.tsv", FLEET),
        "RECALL_MCP_PS_FILE": write("local2.txt", LOCAL_WITH_LIVE_MARK),
        "RECALL_MCP_KILL_FILE": str(kills),
    })
    killed = kills.read_text(encoding="utf-8").split() if kills.exists() else []
    check("9  --kill closes the orphan and its wrapper, and nothing else",
          sorted(killed) == ["100", "101"], f"{killed} :: {p.stdout.strip()[-160:]}")


def test_cli_refuses_unmarked_while_one_could_be_live():
    kills = SCRATCH / "kills3.txt"
    if kills.exists():
        kills.unlink()
    p = run_cli("--kill", "--unmarked", "--client-host", "Bot", env_extra={
        "RECALL_MCP_REMOTE_FILE": write("fleet3.tsv", FLEET),
        "RECALL_MCP_PS_FILE": write("local3.txt", LOCAL_WITH_UNMARKED_OURS),
        "RECALL_MCP_KILL_FILE": str(kills),
    })
    killed = kills.read_text(encoding="utf-8").split() if kills.exists() else []
    check("10 --unmarked is refused while an unmarked transport of ours is live",
          "REFUSED" in p.stdout and "131" not in killed, f"{killed} :: {p.stdout.strip()[-200:]}")


def test_unreachable_host_is_not_an_empty_fleet():
    p = run_cli("--client-host", "Bot", env_extra={
        "RECALL_MCP_REMOTE_FILE": str(SCRATCH / "does-not-exist.tsv"),
        "RECALL_MCP_PS_FILE": write("local4.txt", LOCAL_WITH_LIVE_MARK),
    })
    check("11 an unreadable host is UNREACHABLE, not a clean fleet",
          p.returncode == 2 and "UNREACHABLE" in p.stdout, p.stdout.strip()[-120:])


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for fn in (test_classification, test_unmarked_gate, test_parse_rejects_a_short_row,
               test_a_wrapper_is_not_a_server, test_cli_reports_without_killing,
               test_cli_kills_only_the_orphan, test_cli_refuses_unmarked_while_one_could_be_live,
               test_unreachable_host_is_not_an_empty_fleet):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(f"{fn.__name__} (harness)", False, f"raised {type(exc).__name__}: {exc}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
