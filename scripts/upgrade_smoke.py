"""Prove that a PostgreSQL install written by an earlier RE-call release survives the upgrade.

CI's `schema-migrations` job migrates an empty database, which says nothing about the rows a real
user already has. This script is run twice against one database, with two interpreters:

    seed    run with the PREVIOUS release installed (0.14.0 from PyPI). It does what an operator
            following that release's docs/FIRST_CALIBRATION.md does: schema apply, a file://
            manifest, generation build, validate, promote, a published calibration. Then it starts
            that release's MCP server exactly as a client would and records, through it, what a
            user sees: the top hits for a few queries and their trust state, and one structured
            fact applied through `recall_evidence` and `recall_apply_fact`.

    verify  run after the wheel built from this commit is installed over the same environment.
            It runs the upgrade sequence docs/COMPATIBILITY.md gives (`recall schema status`,
            `recall schema apply`, then readiness), and then asks the NEW server the same
            questions: the schema reports compatible, the old memos return the same top hits and
            are still trusted, the fact 0.14.0 wrote still reads, `recall doctor` passes, and
            forgetting an old memo removes it from search.

No paid API is called: the embedder is `hashing`, which is deterministic and identical in both
releases. Every step prints its name first, and a failure exits 1 naming the step that broke.

    python -I upgrade_smoke.py seed   --workdir DIR     # with the old release installed
    python -I upgrade_smoke.py verify --workdir DIR     # with the new wheel installed

The database is read from RECALL_SERVING_DSN and RECALL_MIGRATION_DSN. Run it with the
interpreter of the environment under test, from outside any checkout, so no import can resolve to
a source tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TENANT = "upgrade-smoke"
EMBEDDER = "hashing"

# 24 memos: certification needs at least 20 answerable and 20 unanswerable queries, and each memo
# answers exactly one. The words are disjoint across memos so the hashing embedder, which is a
# bag of words, separates them; the unanswerable queries use words no memo contains.
TOPICS = [
    ("caching", "redis eviction ttl warmup stampede"),
    ("billing", "invoice proration refund ledger currency"),
    ("deploys", "canary rollback bluegreen pipeline artifact"),
    ("search", "tokenizer stemming synonyms facets relevance"),
    ("auth", "oauth refresh scopes rotation session"),
    ("logging", "structured retention sampling verbosity shipping"),
    ("queues", "backpressure deadletter visibility consumer partition"),
    ("storage", "bucket lifecycle versioning encryption replication"),
    ("metrics", "histogram cardinality scrape exemplar quantile"),
    ("alerts", "pager escalation runbook silence threshold"),
    ("payments", "chargeback settlement acquirer tokenization dispute"),
    ("mobile", "push badge offline sync deeplink"),
    ("email", "bounce suppression dkim template unsubscribe"),
    ("backups", "snapshot restore pitr archive offsite"),
    ("dns", "resolver ttl cname failover propagation"),
    ("frontend", "hydration bundle treeshaking prefetch skeleton"),
    ("database", "vacuum bloat index planner autovacuum"),
    ("security", "pentest cve patching hardening firewall"),
    ("onboarding", "checklist mentor laptop badge orientation"),
    ("pricing", "tier discount annual seat trial"),
    ("support", "ticket macro triage satisfaction backlog"),
    ("analytics", "funnel cohort retention attribution dashboard"),
    ("licensing", "copyleft notice attribution dependency audit"),
    ("hiring", "loop rubric referral sourcing offer"),
]
ABSENT = [
    "volcano glacier tundra lagoon", "violin cello oboe timpani", "saffron cumin fennel clove",
    "falcon heron osprey kestrel", "granite basalt quartz obsidian", "sonnet haiku limerick ode",
    "comet nebula quasar pulsar", "maple birch willow sequoia", "trumpet tuba bassoon harp",
    "otter beaver badger weasel", "amber jade topaz garnet", "canoe kayak dinghy schooner",
    "lentil chickpea barley quinoa", "tango waltz polka foxtrot", "cobalt nickel zinc tungsten",
    "glacier fjord delta estuary", "pigeon sparrow robin wren", "cedar spruce larch hemlock",
    "mango papaya guava lychee", "sitar banjo ukulele lute", "ferret lynx ocelot puma",
    "opal onyx beryl zircon", "rumba samba mambo salsa", "nutmeg anise cardamom mace",
]
# The memo that carries a structured fact, and the memo that is forgotten after the upgrade.
FACT_TOPIC = "caching"
FORGET_TOPIC = "hiring"
FACT = {
    "namespace": "memory",
    "subject": "service:cache",
    "predicate": "owner",
    "object": "team:platform",
}
PROBES = ["caching", "billing", "deploys", "search", FORGET_TOPIC]


class StepFailed(Exception):
    """One upgrade step did not do what a user upgrading needs it to do."""


def _memo_text(topic: str, words: str) -> str:
    return f"{topic} decision: {words}"


def _query(topic: str) -> str:
    return _memo_text(topic, dict(TOPICS)[topic])


def _recall(*args: str) -> list[str]:
    scripts = Path(sys.executable).parent
    exe = scripts / ("recall.exe" if os.name == "nt" else "recall")
    return [str(exe), "--tenant", TENANT, "--embedder", EMBEDDER, *args]


def _env(workdir: Path, *, production: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["RECALL_EMBEDDER"] = EMBEDDER
    env["RECALL_TENANT"] = TENANT
    env["RECALL_LOCAL_ALLOWLIST"] = str(workdir / "notes")
    env["RECALL_EMBED_CACHE"] = "off"
    env.pop("RECALL_TRUST_MODE", None)
    if production:
        env["RECALL_ENV"] = "production"
    else:
        env.pop("RECALL_ENV", None)
    return env


def _run(step: str, argv: list[str], *, env: dict[str, str], cwd: Path, ok_codes=(0,)) -> str:
    print(f"  $ {' '.join([Path(argv[0]).name, *argv[1:]])}", flush=True)
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    out = proc.stdout + (("\n" + proc.stderr) if proc.stderr.strip() else "")
    for line in out.strip().splitlines()[-40:]:
        print(f"    {line}")
    if proc.returncode not in ok_codes:
        raise StepFailed(f"{step}: exited {proc.returncode}")
    return proc.stdout


async def _server_calls(env: dict[str, str], cwd: Path, calls):
    """Start this environment's MCP server over stdio, as a client would, and run `calls`."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-I", "-m", "recall_mcp.server"], env=env, cwd=str(cwd)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await asyncio.wait_for(session.initialize(), timeout=300)

        async def call(tool: str, arguments: dict) -> dict:
            result = await asyncio.wait_for(session.call_tool(tool, arguments), timeout=300)
            text = "".join(getattr(part, "text", "") for part in result.content)
            is_error = getattr(result, "is_error", getattr(result, "isError", False))
            if is_error:
                raise StepFailed(f"{tool} returned an error: {text[:1500]}")
            try:
                return json.loads(text)
            except ValueError as exc:
                raise StepFailed(f"{tool} did not return JSON: {text[:1500]}") from exc

        return await calls(call)


def _top(payload: dict) -> list[str]:
    return [str(hit.get("source", "")) for hit in payload.get("hits") or []][:3]


def _require_trusted(step: str, query: str, payload: dict) -> None:
    state = payload.get("trust_state")
    if state != "trusted" or payload.get("calibrated") is not True:
        raise StepFailed(
            f"{step}: {query!r} came back trust_state={state!r} calibrated={payload.get('calibrated')!r} "
            f"failure_code={payload.get('failure_code')!r} reason={payload.get('reason')!r}"
        )


# ---------------------------------------------------------------------------------------- seed


def _packaged_migrations() -> list[str]:
    from importlib.resources import files

    return sorted(p.name for p in files("recall.migrations").joinpath("sql").iterdir() if p.name.endswith(".sql"))


def seed(workdir: Path) -> None:
    import recall

    version = getattr(recall, "__version__", "unknown")
    state: dict = {"seeded_by": version, "migrations": _packaged_migrations()}
    dev = _env(workdir, production=False)
    prod = _env(workdir, production=True)

    print(f"seed 1: apply the schema with recall {version}", flush=True)
    _run("seed 1", _recall("schema", "apply"), env=dev, cwd=workdir)

    print("seed 2: write the memo folder", flush=True)
    notes = workdir / "notes"
    notes.mkdir(parents=True)
    for topic, words in TOPICS:
        front = ""
        if topic == FACT_TOPIC:
            front = "---\nrecall_graph: " + json.dumps({"facts": [FACT]}) + "\n---\n"
        (notes / f"{topic}.md").write_text(front + _memo_text(topic, words) + "\n", encoding="utf-8")
    queries = [{"query": _query(t), "answerable": True, "relevant_ids": []} for t, _ in TOPICS]
    queries += [{"query": q, "answerable": False, "relevant_ids": []} for q in ABSENT]
    (workdir / "queries.json").write_text(json.dumps(queries), encoding="utf-8")

    print("seed 3: build a file:// manifest", flush=True)
    _run("seed 3", _recall("manifest", "inventory", str(notes), "--output", str(workdir / "inventory.json")),
         env=dev, cwd=workdir)
    _run("seed 3", _recall("manifest", "create", "--corpus-version", "2026-09-17",
                           "--objects", str(workdir / "inventory.json"),
                           "--output", str(workdir / "manifest.json")), env=dev, cwd=workdir)

    print("seed 4: build, validate and promote a generation", flush=True)
    out = _run("seed 4", _recall("generation", "build", str(workdir / "manifest.json"),
                                 "--unverified-development", "--project", "upgrade-smoke",
                                 "--no-commit-stamp"), env=dev, cwd=workdir)
    match = re.search(r"^built (\S+?):", out, re.MULTILINE)
    if not match:
        raise StepFailed("seed 4: `generation build` printed no generation id")
    generation = match.group(1)
    state["generation"] = generation
    _run("seed 4", _recall("generation", "validate", generation), env=dev, cwd=workdir)
    _run("seed 4", _recall("generation", "promote", generation, "--unsafe-development-promotion"),
         env=dev, cwd=workdir)

    print("seed 5: calibrate the generation and publish it", flush=True)
    _run("seed 5", _recall("calibration", "calibrate", "--generation", generation,
                           "--queries", str(workdir / "queries.json"), "--publish"), env=dev, cwd=workdir)

    print("seed 6: search and apply a fact through the 0.14.0 MCP server", flush=True)

    async def calls(call):
        probes = {}
        for topic in PROBES:
            payload = await call("recall_search", {"query": _query(topic), "k": 3})
            _require_trusted("seed 6", topic, payload)
            top = _top(payload)
            if not top or not top[0].endswith(f"{topic}.md"):
                raise StepFailed(f"seed 6: {topic}.md is not the top hit for its own query: {top}")
            probes[topic] = top
            print(f"    {topic}: {top} ({payload.get('trust_state')})")
        evidence = await call("recall_evidence", {"query": _query(FACT_TOPIC), "k": 3})
        cards = [c for c in evidence.get("cards") or [] if c.get("structured_facts")]
        if not cards:
            raise StepFailed(f"seed 6: no evidence card carries the memo's fact: {json.dumps(evidence)[:1500]}")
        return probes, cards[:1]

    probes, cards = asyncio.run(_server_calls(prod, workdir, calls))

    # Applied with the CLI, not `recall_apply_fact`: 0.14.0's generation-mode server reports its
    # store's generation as `legacy`, so the controller refuses every card the same server minted
    # (GENERATION_MISMATCH, then FRESH_SEARCH_INSUFFICIENT). `recall provenance apply
    # --generation` binds the generation explicitly and is the route a 0.14.0 user had.
    print("seed 7: apply a fact with `recall provenance apply`, from a card the server minted", flush=True)
    (workdir / "claim.json").write_text(json.dumps(FACT), encoding="utf-8")
    (workdir / "cards.json").write_text(json.dumps({"cards": cards}), encoding="utf-8")
    out = _run("seed 7", _recall("provenance", "apply", "--claim", str(workdir / "claim.json"),
                                 "--cards", str(workdir / "cards.json"), "--request-id", "upgrade-smoke-1",
                                 "--generation", generation), env=prod, cwd=workdir)
    if json.loads(out).get("decision_code") != "APPLIED":
        raise StepFailed("seed 7: the fact was not applied")

    async def read_facts(call):
        return await call("recall_current_facts", {})

    current = asyncio.run(_server_calls(prod, workdir, read_facts))
    facts = _fact_objects(current)
    if FACT["object"] not in facts:
        raise StepFailed(f"seed 7: the applied fact is not current: {json.dumps(current)[:1500]}")
    state["probes"] = probes
    state["facts"] = facts
    (workdir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"seeded: {json.dumps(state)}")


def _fact_objects(current: dict) -> list[str]:
    return sorted(
        str((item.get("fact") or {}).get("object")) for item in current.get("facts") or []
        if (item.get("fact") or {}).get("subject") == FACT["subject"]
    )


# -------------------------------------------------------------------------------------- verify


def verify(workdir: Path) -> None:
    import recall

    version = getattr(recall, "__version__", "unknown")
    state = json.loads((workdir / "state.json").read_text(encoding="utf-8"))
    # Compared by the migrations the package ships, not by version: until the release bump the
    # new wheel carries the old version number.
    added = sorted(set(_packaged_migrations()) - set(state["migrations"]))
    if not added:
        raise StepFailed("verify 0: the installed package ships no migration the seeding release lacked")
    prod = _env(workdir, production=True)
    print(f"verify 0: upgrading a database recall {state['seeded_by']} wrote, with recall {version}; "
          f"new migrations: {', '.join(added)}")

    print("verify 1: `recall schema status` before migrating (expected to report pending)", flush=True)
    _run("verify 1", _recall("schema", "status"), env=prod, cwd=workdir, ok_codes=(0, 1))

    print("verify 2: `recall schema apply`", flush=True)
    _run("verify 2", _recall("schema", "apply"), env=prod, cwd=workdir)

    print("verify 3: `recall schema status` reports compatible", flush=True)
    out = _run("verify 3", _recall("schema", "status"), env=prod, cwd=workdir)
    if "compatible: yes" not in out:
        raise StepFailed("verify 3: schema status does not say `compatible: yes`")

    print("verify 4: `recall doctor` passes", flush=True)
    _run("verify 4", [*_recall("doctor")], env=prod, cwd=workdir)

    print("verify 5: the old memos search the same, still trusted, and the old fact reads", flush=True)

    async def read_calls(call):
        for topic, before in state["probes"].items():
            payload = await call("recall_search", {"query": _query(topic), "k": 3})
            _require_trusted("verify 5", topic, payload)
            after = _top(payload)
            print(f"    {topic}: {after} ({payload.get('trust_state')})")
            if after != before:
                raise StepFailed(f"verify 5: top hits for {topic!r} changed: before {before}, after {after}")
        return await call("recall_current_facts", {})

    current = asyncio.run(_server_calls(prod, workdir, read_calls))
    if _fact_objects(current) != state["facts"]:
        raise StepFailed(
            f"verify 5: facts written by {state['seeded_by']} do not read back: "
            f"before {state['facts']}, after {json.dumps(current)[:1500]}"
        )

    print(f"verify 6: forget {FORGET_TOPIC}.md, a memo {state['seeded_by']} indexed", flush=True)
    source = next(s for s in state["probes"][FORGET_TOPIC] if s.endswith(f"{FORGET_TOPIC}.md"))
    _run("verify 6", _recall("forget", source, "--yes"), env=prod, cwd=workdir)

    async def after_forget(call):
        return await call("recall_search", {"query": _query(FORGET_TOPIC), "k": 3})

    payload = asyncio.run(_server_calls(prod, workdir, after_forget))
    remaining = _top(payload)
    print(f"    {FORGET_TOPIC} after forget: {remaining} ({payload.get('trust_state')})")
    if any(s.endswith(f"{FORGET_TOPIC}.md") for s in remaining):
        raise StepFailed(f"verify 6: {FORGET_TOPIC}.md still searchable after forget: {remaining}")
    print("upgrade verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("seed", "verify"))
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        (seed if args.phase == "seed" else verify)(workdir)
    except StepFailed as exc:
        print(f"\nUPGRADE FAILED at {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
