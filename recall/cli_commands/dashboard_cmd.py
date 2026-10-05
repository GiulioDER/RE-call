"""`recall dashboard`: a local page to review supersession proposals and agent reports.

Filesystem only, like `recall rewrite`: it reads and writes the memo files under `--root` and their
`.recall` sidecar, opens no database, and never calls a model (arbiter proposals are read from its
cache). It serves on 127.0.0.1 only, and the session starts from the one link it prints.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8765


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
    parser.set_defaults(func=_cmd_dashboard)


def _cmd_dashboard(args: argparse.Namespace) -> None:
    from recall.dashboard.server import DashboardApp, serve

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"recall dashboard: {root} is not a directory", file=sys.stderr)
        raise SystemExit(2)
    if not 1 <= args.port <= 65535:
        print(f"recall dashboard: --port {args.port} is not a port", file=sys.stderr)
        raise SystemExit(2)
    app = DashboardApp(root, port=args.port)
    try:
        server = serve(app)
    except OSError as exc:
        print(f"recall dashboard: cannot listen on 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    url = f"http://127.0.0.1:{args.port}/?token={app.token}"
    print(f"Reviewing {root}")
    print(f"Open: {url}")
    print("This link is the session key for this run. Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
