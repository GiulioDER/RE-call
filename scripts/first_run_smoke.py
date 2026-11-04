"""Walk the path a new user takes after `pip install`, against an installed RE-call.

The release workflow's `smoke` job proves the wheel imports and its console scripts start. That
says nothing about the first ten minutes a new user spends with it, which is where a broken release
is first seen. This script does those ten minutes, in a fresh project directory far from any source
tree, and fails on the first step that would fail for them:

1. `recall setup --lite --memory notes` indexes a folder of memos into `.recall/memory.db` and
   registers `recall-memory` with Claude Code. Claude Code's config is pointed at a temporary
   directory with `CLAUDE_CONFIG_DIR`, so nothing real is touched.
2. The registration is read back from that `.claude.json`: the server entry must exist for this
   project, launch an interpreter that exists, and name this store.
3. `recall lite refresh --check` must report the store current straight after setup, and
   `recall lite refresh` must then succeed.
4. The server is started EXACTLY as Claude Code would start it (that entry's command, args and env)
   and spoken to over stdio with the MCP SDK: it must list `recall_search`, and a search for one
   memo's subject must return that memo among its hits.

Run it with the interpreter of the environment the wheel is installed in, and NOT from inside a
checkout, so an import cannot resolve to the source tree:

    python scripts/first_run_smoke.py                    # fastembed, as a new user gets it
    python scripts/first_run_smoke.py --embedder hashing # no model download, for a quick local run

Exit 0 when every step passed, 1 with the failing step named otherwise. `--keep` leaves the
temporary project in place for inspection.

With a small corpus the store can stay uncertified, and strict trust then refuses every search.
Step 4 reads the calibration status the store reports and checks the matching contract: a
certified store must return the expected memo, an uncertified one must refuse with the structured
`CALIBRATION_UNCERTIFIED` error.

Red proof, recorded 2026-11-04: local runs with `--embedder hashing` (which leaves this corpus
uncertified) against a wheel built from `3eb9c71f4`. Each line below is one edit to the installed
copy, the step that then failed, and its message; restored, the same run passed.

- `recall/lite/setup.py`, the registered `RECALL_INDEX_ROOT` set to the notes folder's parent:
  step 2, "registered RECALL_INDEX_ROOT ... is not the notes folder".
- `recall/lite/refresh.py`, `check()` never reporting current: step 3, `--check` exited 1.
- `recall_mcp/server.py`, `recall_search` registered under another name: step 4, "recall_search is
  not among the tools".
- `recall/trust_policy.py`, the uncertified refusal code changed: step 4, "an uncertified store did
  not refuse as documented".

NOT red-proved: the certified branch of step 4 (the expected memo among the hits). It needs a
real embedder, and embedding does not run on the workstation this was written on; it is exercised
where the CI job runs fastembed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The server name `recall setup --lite` registers (`recall.lite.setup.SERVER_NAME`). Written out
#: rather than imported: the point is to check what an installed copy does, not what it says.
SERVER_NAME = "recall-memory"

#: Distinct subjects, so a search for one has a single right answer. Each memo is long enough to
#: be chunked as a real note would be.
MEMOS = {
    "postgres-backups.md": (
        "Postgres backups",
        "We take a nightly pg_dump of the production database at 02:00 UTC and keep fourteen days. "
        "A restore was rehearsed on 2026-09-12 and took eleven minutes for a 4 GB dump.",
    ),
    "frontend-build.md": (
        "Frontend build",
        "The frontend is built with Vite. The production bundle must stay under 300 kB gzipped; "
        "the size check runs in CI and fails the build when the limit is crossed.",
    ),
    "release-process.md": (
        "Release process",
        "A release is cut from master by pushing a signed tag. The release workflow builds the "
        "wheel, runs the smoke job, attests the artifacts and publishes to PyPI.",
    ),
    "oncall-rotation.md": (
        "On-call rotation",
        "On-call rotates weekly on Monday at 09:00. The person on call owns the pager, triages "
        "alerts within fifteen minutes and writes the incident note before handing over.",
    ),
    "api-rate-limits.md": (
        "API rate limits",
        "The public API allows 600 requests per minute per key. Bursts above that receive HTTP 429 "
        "with a Retry-After header; internal services are exempt through the gateway allowlist.",
    ),
    "coffee-machine.md": (
        "Office coffee machine",
        "The espresso machine on the third floor is descaled every second Friday. Use the filtered "
        "water tap; the descaling tablets are in the cupboard under the sink.",
    ),
    "dns-provider.md": (
        "DNS provider",
        "Our domains are registered with Gandi and the zones are served by Cloudflare. Record TTLs "
        "are 300 seconds except the apex MX records, which stay at one hour.",
    ),
    "laptop-policy.md": (
        "Laptop policy",
        "Company laptops are replaced every three years. Disk encryption must be on, and a lost "
        "laptop is reported to the security channel within one hour of noticing.",
    ),
    "error-budget.md": (
        "Checkout error budget",
        "The checkout service has a 99.9 percent monthly availability target, about 43 minutes of "
        "downtime. When the budget is spent, feature deploys to checkout pause until it recovers.",
    ),
    "feature-flags.md": (
        "Feature flags",
        "Feature flags live in LaunchDarkly. A flag older than ninety days must be removed or "
        "justified in the weekly review; the default for a new flag is off in production.",
    ),
    "translation-workflow.md": (
        "Translation workflow",
        "User interface strings are extracted to Crowdin every Tuesday. German and Japanese are "
        "reviewed by contracted translators before a string ships in those locales.",
    ),
    "invoice-runs.md": (
        "Invoice runs",
        "Customer invoices are generated on the first business day of each month. Failed card "
        "charges are retried after three, five and seven days before the account is suspended.",
    ),
    "search-index.md": (
        "Product search index",
        "Product search runs on OpenSearch with two replicas. The catalog is reindexed from scratch "
        "every Sunday night; price changes are pushed incrementally within a minute.",
    ),
    "mobile-releases.md": (
        "Mobile releases",
        "The iOS and Android apps ship every second Wednesday. A release goes to five percent of "
        "users first and is halted automatically if the crash rate rises above 0.5 percent.",
    ),
    "data-retention.md": (
        "Data retention",
        "Application logs are kept for thirty days and audit logs for seven years. Personal data of "
        "a closed account is deleted within forty-five days of the closure request.",
    ),
    "interview-loop.md": (
        "Engineering interview loop",
        "The engineering interview loop is four sessions: a pairing exercise, a system design talk, "
        "a code review of a prepared pull request, and a conversation with the hiring manager.",
    ),
    "office-wifi.md": (
        "Office wifi",
        "Guests join the Visitors network, which is isolated from the corporate VLAN. The guest "
        "password rotates on the first of every month and is posted at reception.",
    ),
    "load-testing.md": (
        "Load testing",
        "Load tests run with k6 against the staging cluster every Thursday. The target is 2,000 "
        "requests per second at a p95 latency below 250 milliseconds for the catalog API.",
    ),
    "secrets-management.md": (
        "Secrets management",
        "Application secrets are stored in Vault and injected at deploy time. Developers never "
        "copy production secrets to laptops; a leaked key is rotated within one hour.",
    ),
    "design-reviews.md": (
        "Design reviews",
        "Any change touching more than two services needs a written design review. The document "
        "is circulated for three working days before the review meeting decides.",
    ),
    "customer-support-tiers.md": (
        "Customer support tiers",
        "Enterprise customers get a one-hour first response around the clock. Standard plans get "
        "a reply within one business day, through email only.",
    ),
    "kubernetes-upgrades.md": (
        "Kubernetes upgrades",
        "Clusters are upgraded one minor Kubernetes version per quarter, staging first. Node pools "
        "are drained one at a time with a pod disruption budget of one.",
    ),
    "analytics-events.md": (
        "Analytics events",
        "Product analytics events are named object_action in snake case, such as cart_checkout. "
        "A new event needs a schema entry before the client sends it.",
    ),
    "parking.md": (
        "Office parking",
        "The garage has twelve bicycle spaces and four electric car chargers. Charger slots are "
        "booked through the facilities form for at most four hours at a time.",
    ),
}

#: A question whose answer is one memo, and that memo.
QUERY = "how long do we keep the nightly database dumps"
EXPECTED = "postgres-backups.md"


class StepFailed(Exception):
    """A first-run step did not do what a new user needs it to do."""


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], step: str) -> str:
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise StepFailed(
            f"{step}: `{' '.join(argv)}` exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


def _write_memos(notes: Path) -> None:
    notes.mkdir()
    for name, (title, body) in MEMOS.items():
        (notes / name).write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _registered_server(config: Path, project: Path) -> dict:
    """The server entry setup wrote for `project`, as Claude Code would look it up."""
    document = json.loads(config.read_text(encoding="utf-8"))
    projects = document.get("projects") or {}
    # Claude Code keys projects by absolute path; setup may write more than one spelling of it.
    for key, entry in projects.items():
        if Path(key).resolve() == project.resolve():
            server = (entry.get("mcpServers") or {}).get(SERVER_NAME)
            if server is not None:
                return server
    raise StepFailed(
        f"step 2: no `{SERVER_NAME}` server registered for {project} in {config}; "
        f"project keys present: {sorted(projects)}"
    )


def _check_registration(server: dict, store: Path, notes: Path) -> None:
    command = server.get("command")
    if not command or shutil.which(command) is None and not Path(command).is_file():
        raise StepFailed(f"step 2: the registered command {command!r} does not exist")
    if server.get("args") != ["-m", "recall_mcp.server"]:
        raise StepFailed(f"step 2: unexpected registered args {server.get('args')!r}")
    env = server.get("env") or {}
    dsn = env.get("RECALL_SERVING_DSN", "")
    if not dsn.startswith("sqlite:///") or Path(dsn[len("sqlite:///"):]).resolve() != store.resolve():
        raise StepFailed(f"step 2: registered RECALL_SERVING_DSN {dsn!r} does not name {store}")
    root = env.get("RECALL_INDEX_ROOT", "")
    if not root or Path(root).resolve() != notes.resolve():
        raise StepFailed(f"step 2: registered RECALL_INDEX_ROOT {root!r} is not the notes folder {notes}")


async def _talk_to_server(server: dict, cwd: Path, *, certified: bool) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=server["command"], args=server["args"], env=server.get("env") or {}, cwd=str(cwd)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await asyncio.wait_for(session.initialize(), timeout=300)
        tools = {tool.name for tool in (await session.list_tools()).tools}
        if "recall_search" not in tools:
            raise StepFailed(f"step 4: recall_search is not among the tools: {sorted(tools)}")
        result = await asyncio.wait_for(
            session.call_tool("recall_search", {"query": QUERY, "k": 3}), timeout=300
        )
        text = "".join(getattr(part, "text", "") for part in result.content)
        # The SDK renamed the field; the wheel admits versions on both sides of the rename.
        is_error = getattr(result, "is_error", getattr(result, "isError", False))
        try:
            payload = json.loads(text[text.index("{"):] if is_error and "{" in text else text)
        except ValueError as exc:
            raise StepFailed(f"step 4: recall_search did not return JSON: {text[:500]}") from exc
        if not certified:
            # Strict trust with no certified calibration must refuse, and say why in a form a
            # client can act on; anything else (a crash, an answer) is the failure.
            if not is_error or payload.get("code") != "CALIBRATION_UNCERTIFIED":
                raise StepFailed(f"step 4: an uncertified store did not refuse as documented: {text[:800]}")
            print(f"  tools: {len(tools)}; search refused as documented: {payload['code']}")
            return
        if is_error:
            raise StepFailed(f"step 4: recall_search on a certified store returned an error: {text[:800]}")
        hits = payload.get("hits")
        if not isinstance(hits, list) or not hits:
            raise StepFailed(f"step 4: recall_search returned no hits: {text[:800]}")
        sources = [str(hit.get("source", "")) for hit in hits]
        if not any(source.endswith(EXPECTED) for source in sources):
            raise StepFailed(f"step 4: {EXPECTED} is not among the hits for {QUERY!r}: {sources}")
        print(f"  tools: {len(tools)}; hits: {sources}; abstained: {payload.get('abstained')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedder", default=None, help="passed as `recall --embedder`; default: what a new user gets")
    parser.add_argument("--keep", action="store_true", help="leave the temporary project in place")
    args = parser.parse_args()

    # The console script beside this interpreter, not the first one on PATH: on a machine with
    # more than one Python, PATH can name another installation's `recall` (measured here: Git
    # Bash put a system Python 3.12 Scripts directory ahead of the venv's).
    recall = shutil.which("recall", path=str(Path(sys.executable).parent))
    if recall is None:
        print(f"first-run smoke: no `recall` console script beside {sys.executable}", file=sys.stderr)
        return 1

    base = Path(tempfile.mkdtemp(prefix="recall-first-run-")).resolve()
    project, claude_dir = base / "project", base / "claude-config"
    project.mkdir()
    claude_dir.mkdir()
    notes, store, config = project / "notes", project / ".recall" / "memory.db", claude_dir / ".claude.json"
    _write_memos(notes)
    # An existing, empty Claude Code config: what a user who has opened Claude Code once has.
    config.write_text("{}", encoding="utf-8")

    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("RECALL_")  # nothing from the caller's own RE-call setup
    }
    env["CLAUDE_CONFIG_DIR"] = str(claude_dir)
    env["RECALL_EMBED_CACHE"] = str(base / "embed-cache.sqlite")
    env["PYTHONSAFEPATH"] = "1"
    embedder = ["--embedder", args.embedder] if args.embedder else []

    try:
        print("step 1: recall setup --lite")
        out = _run([recall, *embedder, "setup", "--lite", "--memory", str(notes)], cwd=project, env=env, step="step 1")
        print("  " + out.strip().replace("\n", "\n  "))
        if not store.is_file():
            raise StepFailed(f"step 1: setup exited 0 but {store} does not exist")

        print("step 2: the Claude Code registration")
        server = _registered_server(config, project)
        _check_registration(server, store, notes)

        # Setup is the store's first refresh, so the folder it just indexed must already be
        # current; until setup indexed through the refresh, this said "never refreshed" and exited 1.
        print("step 3: recall lite refresh --check straight after setup, then a refresh")
        refresh = [recall, *embedder, "lite", "refresh", "--store", str(store), "--memory", str(notes)]
        status = _run([*refresh, "--check"], cwd=project, env=env, step="step 3 (--check after setup)")
        _run(refresh, cwd=project, env=env, step="step 3 (refresh)")
        print("  " + status.strip().replace("\n", "\n  "))
        calibration = next((line.split(None, 1)[1] for line in status.splitlines() if line.startswith("calibration ")), "")
        certified = calibration.startswith("certified")

        print(f"step 4: start the server as Claude Code would, and search (calibration {calibration.split(':')[0] or 'unknown'})")
        asyncio.run(_talk_to_server(server, project, certified=certified))
    except (StepFailed, BaseExceptionGroup) as raised:
        # The MCP client's task groups wrap a step 4 failure in (nested) exception groups.
        exc = raised
        while isinstance(exc, BaseExceptionGroup):
            leaves = exc.subgroup(lambda e: isinstance(e, StepFailed))
            if leaves is None:
                raise
            exc = leaves.exceptions[0]
        print(f"first-run smoke FAILED at {exc}", file=sys.stderr)
        print(f"(temporary project kept at {base})", file=sys.stderr)
        return 1
    if args.keep:
        print(f"first-run smoke passed; project kept at {base}")
    else:
        shutil.rmtree(base, ignore_errors=True)
        print("first-run smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
