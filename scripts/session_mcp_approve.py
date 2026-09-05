"""Pre-approve this checkout's own `.mcp.json` servers in the MCP client's config.

Why this exists
---------------
Writing `.mcp.json` is not enough to get a server. A project-scoped server is **pending approval**
until the client records the approval, and it records it per project directory in
`~/.claude.json` under `projects[<dir>].enabledMcpjsonServers`. A non-interactive session can
never answer that prompt, so a checkout can have a perfectly good `.mcp.json` on disk and still
start with zero servers, every time, forever.

Measured 2026-08-17 on this machine: 306 tracked projects, **zero** with any approved
`.mcp.json` server, while `claude mcp list` reported both recall servers as
`⏸ Pending approval`. That is the real reason the recall servers never load, and it is not
the write-ordering hypothesis that was pre-registered for it.

What this deliberately does NOT do
----------------------------------
- It does not set `enableAllProjectMcpServers`. That flag approves the `.mcp.json` of *every*
  repository on the machine, including one that arrives with a hostile server definition in it.
  The whole value of the approval gate is that it is per project.
- It does not write a URL, a token, or any other part of the server definition into the client
  config. **Only server names cross this boundary.** The definitions stay in the gitignored
  `.mcp.json`, and the secrets stay in `~/.claude/recall-mcp-secrets.json`.
- It does not re-enable a server the operator has explicitly disabled. An entry in
  `disabledMcpjsonServers` is a decision, and silently reversing it is how a tool loses trust.
- It does not accept the folder-trust dialog on anyone's behalf.

Usage:
    python scripts/session_mcp_approve.py --root <checkout> --servers recall recall-memory
    python scripts/session_mcp_approve.py --root <checkout> --from-mcp-json <path>
    python scripts/session_mcp_approve.py --root <checkout> --servers ... --check
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

CONFIG_ENV = "RECALL_MCP_CLIENT_CONFIG"


def client_config_path() -> str:
    """The MCP client's own config. Overridable so the tests never touch the real one."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".claude.json")


def project_key(root: str) -> str:
    """Match the key format the client uses for a project directory.

    The client stores native paths (backslashes on Windows), while `git rev-parse
    --show-toplevel` hands back forward slashes even there. Keying on the git form creates a
    *second*, parallel entry that the client never reads, so the approval silently does nothing
    and the servers stay pending. Normalising here is the whole reason this is a function.
    """
    return os.path.normpath(os.path.abspath(root))


def approve(config: dict, root: str, servers: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (newly_approved, already_approved, refused_because_disabled).

    Mutates `config` in place. Pure enough to test without a filesystem.
    """
    key = project_key(root)
    projects = config.setdefault("projects", {})
    entry = projects.setdefault(key, {})

    enabled = list(entry.get("enabledMcpjsonServers") or [])
    disabled = list(entry.get("disabledMcpjsonServers") or [])

    newly, already, refused = [], [], []
    for name in servers:
        if name in disabled:
            refused.append(name)
        elif name in enabled:
            already.append(name)
        else:
            enabled.append(name)
            newly.append(name)

    entry["enabledMcpjsonServers"] = enabled
    entry.setdefault("disabledMcpjsonServers", disabled)
    return newly, already, refused


def _dump(config: dict, path: str, indented: bool) -> None:
    """Write atomically, into the same directory so os.replace cannot cross a filesystem."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".claude.json.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            if indented:
                json.dump(config, fh, indent=2)
            else:
                json.dump(config, fh)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="the checkout whose .mcp.json to approve")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--servers", nargs="+", help="server names from that .mcp.json")
    src.add_argument("--from-mcp-json", help="read the names out of this .mcp.json instead")
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args(argv)

    servers = args.servers
    if servers is None:
        # Reading the names back out of the file that was just written keeps one source of
        # truth. A hand-maintained list in the caller drifts the moment a server is renamed,
        # and the symptom of that drift is a server that is silently never approved.
        try:
            with open(args.from_mcp_json, encoding="utf-8") as fh:
                servers = list(json.load(fh).get("mcpServers", {}))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"session-mcp: cannot read {args.from_mcp_json} ({exc})", file=sys.stderr)
            return 0
    if not servers:
        print("session-mcp: no servers to approve")
        return 0

    path = client_config_path()
    if not os.path.exists(path):
        # No client config means no client has ever run here. Nothing to approve, and inventing
        # the file would be guessing at a schema this script does not own.
        print(f"session-mcp: no client config at {path}, nothing to approve")
        return 0

    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        # Never fail a session open over this: an unreadable client config is the client's
        # problem, and the servers being pending is a degradation, not a stoppage.
        print(f"session-mcp: cannot read {path} ({exc}); servers stay pending", file=sys.stderr)
        return 0

    with open(path, encoding="utf-8") as fh:
        indented = "\n" in fh.read(400)

    newly, already, refused = approve(config, args.root, list(servers))

    for name in refused:
        print(
            f"session-mcp: {name!r} is in disabledMcpjsonServers for this checkout, leaving it "
            f"alone (remove it there if you want the server back)",
            file=sys.stderr,
        )

    if not newly:
        print(f"session-mcp: MCP approval already in place ({', '.join(already) or 'none'})")
        return 0

    if args.check:
        print(f"session-mcp: would approve {', '.join(newly)} for {project_key(args.root)}")
        return 0

    # One rolling backup. The client owns this file and rewrites it on its own schedule, so a
    # copy from just before the edit is the difference between "restore it" and "recreate it".
    try:
        shutil.copyfile(path, path + ".session-mcp.bak")
    except OSError as exc:
        print(f"session-mcp: could not back up {path} ({exc})", file=sys.stderr)

    _dump(config, path, indented)
    print(f"session-mcp: approved {', '.join(newly)} for {project_key(args.root)}")
    print("session-mcp: approval is read at client start, so it takes effect NEXT session here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
