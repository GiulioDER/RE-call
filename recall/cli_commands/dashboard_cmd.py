"""`recall dashboard`: a local page to review supersession proposals and agent reports.

The memo pages are filesystem only, like `recall rewrite`: they read and write the memo files under
`--root` and their `.recall` sidecar, and never call a model (arbiter proposals are read from its
cache). The overview and retrieval pages read the corpus database when `--db-dsn-file` names a DSN,
over a read-only session; `--tunnel HOST` opens an SSH forward to it for the life of the command.
It serves on 127.0.0.1 only, and the session starts from the one link it prints.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recall.dashboard.db import DashboardDB

DEFAULT_PORT = 8765
DEFAULT_TUNNEL_PORTS = "55433:5432"
TUNNEL_WAIT_SECONDS = 15.0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "dashboard",
        help="open a local page to review supersession proposals and agent stale reports",
        description=(
            "Serve a review page on this machine only. Accepting a claim writes `supersedes:` "
            "into the newer memo, exactly as `recall rewrite apply` does; rejecting records it "
            "so it is not proposed again. Nothing is sent anywhere and no model is called."
        ),
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("RECALL_INDEX_ROOT", "."),
        help="the memo directory to review (default: RECALL_INDEX_ROOT, else the current directory)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"local port (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="print the link without opening a browser")
    parser.add_argument(
        "--db-dsn-file",
        default=os.environ.get("RECALL_DASHBOARD_DSN_FILE", ""),
        help="a file holding the corpus database DSN, ideally for a SELECT-only role "
        "(default: RECALL_DASHBOARD_DSN_FILE). Without it the database pages say so.",
    )
    parser.add_argument(
        "--tunnel",
        default=os.environ.get("RECALL_DASHBOARD_TUNNEL", ""),
        help="an ssh host to forward the database port from, for the life of this command "
        "(default: RECALL_DASHBOARD_TUNNEL)",
    )
    parser.add_argument(
        "--tunnel-ports",
        default=os.environ.get("RECALL_DASHBOARD_TUNNEL_PORTS", DEFAULT_TUNNEL_PORTS),
        help="LOCAL:REMOTE ports for --tunnel, both on the loopback "
        f"(default: RECALL_DASHBOARD_TUNNEL_PORTS, else {DEFAULT_TUNNEL_PORTS})",
    )
    parser.add_argument(
        "--config-host",
        default=os.environ.get("RECALL_DASHBOARD_CONFIG_HOST", ""),
        help="an ssh host whose RE-call server configuration the Configuration page also shows, "
        "redacted on that host before it is sent (default: RECALL_DASHBOARD_CONFIG_HOST)",
    )
    parser.add_argument(
        "--config-command",
        default=os.environ.get("RECALL_DASHBOARD_CONFIG_COMMAND", ""),
        help="the shell command run on --config-host to print the report, e.g. one that loads the "
        "server's environment first (default: RECALL_DASHBOARD_CONFIG_COMMAND, else "
        "'python3 -m recall_mcp.config_report --json')",
    )
    parser.add_argument(
        "--reports-tenant",
        action="append",
        default=None,
        metavar="TENANT",
        help="a tenant whose agent stale reports belong to this folder's review queue; repeat for "
        "more (default: memory). Reports filed in any other tenant are not shown.",
    )
    parser.set_defaults(func=_cmd_dashboard)


def _cmd_dashboard(args: argparse.Namespace) -> None:
    from recall.dashboard.server import DEFAULT_TENANT, DashboardApp, serve

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"recall dashboard: {root} is not a directory", file=sys.stderr)
        raise SystemExit(2)
    if not 1 <= args.port <= 65535:
        print(f"recall dashboard: --port {args.port} is not a port", file=sys.stderr)
        raise SystemExit(2)
    reports_tenants = tuple(name.strip() for name in args.reports_tenant or (DEFAULT_TENANT,))
    for tenant in reports_tenants:
        if not tenant:  # a bound parameter, so any name is safe to ask for; an empty one is a mistake
            print("recall dashboard: --reports-tenant needs a tenant name", file=sys.stderr)
            raise SystemExit(2)
    db = _database(args.db_dsn_file)
    tunnel = _open_tunnel(args.tunnel, args.tunnel_ports) if args.tunnel else None
    if args.config_host.startswith("-") or any(c.isspace() for c in args.config_host):
        print(f"recall dashboard: --config-host {args.config_host!r} is not a host", file=sys.stderr)
        raise SystemExit(2)
    app = DashboardApp(
        root, port=args.port, db=db, reports_tenants=reports_tenants, project_root=Path.cwd(),
        config_host=args.config_host, config_command=args.config_command,
    )
    try:
        server = serve(app)
    except OSError as exc:
        print(
            f"recall dashboard: cannot listen on 127.0.0.1:{args.port}: {exc}\n"
            f"Another dashboard may already be running on that port: stop it (Ctrl+C in its window), "
            f"or start this one with --port {args.port + 1}.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    url = f"http://127.0.0.1:{args.port}/?token={app.token}"
    print(f"Reviewing {root}", flush=True)
    print(f"Database: {'read-only, from ' + args.db_dsn_file if db else 'not configured (memo pages only)'}", flush=True)
    print(f"Open: {url}", flush=True)
    print("This link is the session key for this run. Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()


def _database(dsn_file: str) -> DashboardDB | None:
    """The read-only database handle, or None. The DSN is read from a file so it never sits in argv."""
    if not dsn_file:
        return None
    from recall.dashboard.db import DashboardDB

    path = Path(dsn_file).expanduser()
    try:
        dsn = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"recall dashboard: cannot read --db-dsn-file {path}: {exc.strerror}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not dsn:
        print(f"recall dashboard: --db-dsn-file {path} is empty", file=sys.stderr)
        raise SystemExit(2)
    return DashboardDB(dsn)


def _parse_ports(spec: str) -> tuple[int, int]:
    local, sep, remote = spec.partition(":")
    if not sep or not local.isdigit() or not remote.isdigit():
        print(f"recall dashboard: --tunnel-ports {spec!r} is not LOCAL:REMOTE", file=sys.stderr)
        raise SystemExit(2)
    pair = int(local), int(remote)
    if not all(1 <= port <= 65535 for port in pair):
        print(f"recall dashboard: --tunnel-ports {spec!r} names a port out of range", file=sys.stderr)
        raise SystemExit(2)
    return pair


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _open_tunnel(host: str, ports: str) -> subprocess.Popen[bytes] | None:
    """Forward 127.0.0.1:LOCAL to the host's 127.0.0.1:REMOTE; reuse a forward already listening."""
    local, remote = _parse_ports(ports)
    if host.startswith("-"):
        print(f"recall dashboard: --tunnel {host!r} is not a host", file=sys.stderr)
        raise SystemExit(2)
    if _port_open(local):
        print(f"Tunnel: 127.0.0.1:{local} already answers; using it as is", flush=True)
        return None
    command = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
               "-L", f"127.0.0.1:{local}:127.0.0.1:{remote}", host]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + TUNNEL_WAIT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"recall dashboard: ssh to {host} exited with {process.returncode}; the tunnel did not open", file=sys.stderr)
            raise SystemExit(2)
        if _port_open(local):
            print(f"Tunnel: 127.0.0.1:{local} -> {host}:{remote}", flush=True)
            return process
        time.sleep(0.2)
    process.terminate()
    print(f"recall dashboard: the tunnel to {host} did not open within {TUNNEL_WAIT_SECONDS:.0f}s", file=sys.stderr)
    raise SystemExit(2)
