"""Prove that a PostgreSQL install written by an earlier RE-call release survives the upgrade.

CI's `schema-migrations` job migrates an empty database, which says nothing about the rows a real
user already has. This script is run twice against one database, with two interpreters:

    seed    run with the PREVIOUS release installed (0.14.0 from PyPI). It does what an operator
            following that release's docs/FIRST_CALIBRATION.md does: schema apply, `recall
            index`, a file:// manifest, generation build, validate, promote, a published
            calibration. Then it starts that release's MCP server exactly as a client would and
            records, through it, what a user sees: the hits for a few queries with their cosines,
            verdicts and trust state. One structured fact is applied with a card that server
            minted through `recall_evidence` (see seed 5 for why not through `recall_apply_fact`).

    verify  run after the wheel built from this commit is installed over the same environment.
            It runs the upgrade sequence the docs give: `recall schema status`, `recall schema
            apply` (docs/COMPATIBILITY.md), `recall provenance rebuild` (CHANGELOG.md and
            docs/MIGRATIONS.md, for migration 0030), `recall doctor`. Then it asks the NEW server
            the same questions: the old memos come back with the same trusted hits in the same
            order and the same cosines, still trusted, matching a fresh install of this release
            into a second tenant; the fact 0.14.0 wrote still reads; and forgetting an old memo
            removes it (strict search refuses as stale until the labelled refit, then answers
            without it).

Red proof (2026-11-04, CI, PostgreSQL 18): see the pull request that added this script. Each
mutation was applied to the installed new wheel before `verify`, and the job failed at the named
step.

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
# The control: the same notes installed fresh by the new release, in a tenant of their own.
FRESH_TENANT = "upgrade-smoke-fresh"
# Set for the two-role leg: RECALL_SERVING_DSN then logs in as this unprivileged role (no
# superuser, no BYPASSRLS, no schema CREATE), and RECALL_MIGRATION_DSN stays the owner. Each phase
# grants it what that release's `recall schema grants --role` prints, after its `schema apply`, as
# docs/MIGRATIONS.md ("Role split") tells an operator to.
SERVING_ROLE = os.environ.get("UPGRADE_SMOKE_SERVING_ROLE")
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
# Near-duplicates of two probed memos: each shares six of its seven tokens, so it scores about
# 6/7 against that probe, above the fitted threshold and below the original. Those two probes
# therefore have two trusted hits in a fixed order, which is what makes "the same top hits" a
# statement about ranking and not only about rank 1. Everything below the threshold in this
# fixture is an exact tie (hashing-64 cosines come in steps of 1/7, and every memo shares one
# token with every query), ordered by chunk id, which a new generation changes; so only the
# trusted hits are compared in order.
FOLLOWUPS = {
    "caching-followup": "caching decision: redis eviction ttl warmup jitter",
    "deploys-followup": "deploys decision: canary rollback bluegreen pipeline freeze",
}


class StepFailed(Exception):
    """One upgrade step did not do what a user upgrading needs it to do."""


def _memo_text(topic: str, words: str) -> str:
    return f"{topic} decision: {words}"


def _query(topic: str) -> str:
    return _memo_text(topic, dict(TOPICS)[topic])


def _recall(*args: str, tenant: str = TENANT) -> list[str]:
    scripts = Path(sys.executable).parent
    exe = scripts / ("recall.exe" if os.name == "nt" else "recall")
    return [str(exe), "--tenant", tenant, "--embedder", EMBEDDER, *args]


def _env(workdir: Path, *, production: bool, tenant: str = TENANT) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["RECALL_EMBEDDER"] = EMBEDDER
    env["RECALL_TENANT"] = tenant
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

        async def call(tool: str, arguments: dict, *, refusal_ok: bool = False) -> dict:
            result = await asyncio.wait_for(session.call_tool(tool, arguments), timeout=300)
            text = "".join(getattr(part, "text", "") for part in result.content)
            is_error = getattr(result, "is_error", getattr(result, "isError", False))
            if is_error and refusal_ok and "{" in text:
                return {"refused": True, **json.loads(text[text.index("{"):text.rindex("}") + 1])}
            if is_error:
                raise StepFailed(f"{tool} returned an error: {text[:1500]}")
            try:
                return json.loads(text)
            except ValueError as exc:
                raise StepFailed(f"{tool} did not return JSON: {text[:1500]}") from exc

        return await calls(call)


def _top(payload: dict) -> list[str]:
    return [hit[0] for hit in _scored(payload)]


def _scored(payload: dict) -> list[list]:
    """The top three hits as [source, dense cosine, verdict], the cosine rounded past float noise."""
    return [
        [str(hit.get("source", "")), round(float(hit.get("score") or 0.0), 6), str(hit.get("verdict"))]
        for hit in payload.get("hits") or []
    ][:3]


def _trusted_hits(hits: list[list]) -> list[list]:
    """The hits the trust layer admitted (verdict ok), in order: what a user is told to use."""
    return [hit for hit in hits if hit[2] == "ok"]


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


def _grant(step: str, workdir: Path, env: dict[str, str]) -> None:
    """Apply, as the owner, the serving grants this release generates; a no-op with one role."""
    if not SERVING_ROLE:
        return
    import psycopg

    sql = _run(step, _recall("schema", "grants", "--role", SERVING_ROLE), env=env, cwd=workdir)
    statements = [line for line in sql.splitlines() if line.strip()]
    with psycopg.connect(env["RECALL_MIGRATION_DSN"], autocommit=True) as conn:
        for statement in statements:
            conn.execute(statement)
    print(f"    applied {len(statements)} grant statement(s) to {SERVING_ROLE}")


def _install(step: str, workdir: Path, tenant: str, *, legacy_index: bool = True) -> str:
    """0.14.0's docs/FIRST_CALIBRATION.md, steps 1 to 6, for the notes in `workdir`.

    Run by whichever release is installed: by the old one to seed, and by the new one for the
    fresh-install control. `legacy_index=False` skips step 1, leaving a tenant whose corpus is
    only its promoted generation, as the wizard builds its docs and code tenants. Returns the
    promoted generation.
    """
    dev = _env(workdir, production=False, tenant=tenant)
    notes = workdir / "notes"
    files = workdir / tenant
    files.mkdir()

    def recall(*args: str) -> str:
        return _run(step, _recall(*args, tenant=tenant), env=dev, cwd=workdir)

    if legacy_index:
        recall("index", str(notes), "--project", "upgrade-smoke")
    recall("manifest", "inventory", str(notes), "--output", str(files / "inventory.json"))
    recall("manifest", "create", "--corpus-version", "2026-09-17",
           "--objects", str(files / "inventory.json"), "--output", str(files / "manifest.json"))
    out = recall("generation", "build", str(files / "manifest.json"), "--unverified-development",
                 "--project", "upgrade-smoke", "--no-commit-stamp")
    match = re.search(r"^built (\S+?):", out, re.MULTILINE)
    if not match:
        raise StepFailed(f"{step}: `generation build` printed no generation id")
    generation = match.group(1)
    recall("generation", "validate", generation)
    recall("generation", "promote", generation, "--unsafe-development-promotion")
    recall("calibration", "calibrate", "--generation", generation,
           "--queries", str(workdir / "queries.json"), "--publish")
    return generation


async def _probe(step: str, call) -> dict[str, list[list]]:
    """Each probe's top hits through the server, which must answer trusted."""
    probes = {}
    for topic in PROBES:
        payload = await call("recall_search", {"query": _query(topic), "k": 3})
        _require_trusted(step, topic, payload)
        probes[topic] = _scored(payload)
        print(f"    {topic}: {probes[topic]} ({payload.get('trust_state')})")
    return probes


def seed(workdir: Path) -> None:
    import recall

    version = getattr(recall, "__version__", "unknown")
    state: dict = {"seeded_by": version, "migrations": _packaged_migrations()}
    dev = _env(workdir, production=False)
    prod = _env(workdir, production=True)

    print(f"seed 1: apply the schema with recall {version}", flush=True)
    _run("seed 1", _recall("schema", "apply"), env=dev, cwd=workdir)
    _grant("seed 1", workdir, dev)

    print("seed 2: write the memo folder and its labelled queries", flush=True)
    notes = workdir / "notes"
    notes.mkdir(parents=True)
    for topic, words in TOPICS:
        front = ""
        if topic == FACT_TOPIC:
            front = "---\nrecall_graph: " + json.dumps({"facts": [FACT]}) + "\n---\n"
        (notes / f"{topic}.md").write_text(front + _memo_text(topic, words) + "\n", encoding="utf-8")
    for name, text in FOLLOWUPS.items():
        (notes / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    queries = [{"query": _query(t), "answerable": True, "relevant_ids": []} for t, _ in TOPICS]
    queries += [{"query": text, "answerable": True, "relevant_ids": []} for text in FOLLOWUPS.values()]
    queries += [{"query": q, "answerable": False, "relevant_ids": []} for q in ABSENT]
    (workdir / "queries.json").write_text(json.dumps(queries), encoding="utf-8")
    print("seed 3: index, build, validate, promote and calibrate, as FIRST_CALIBRATION.md says", flush=True)
    generation = _install("seed 3", workdir, TENANT)
    state["generation"] = generation

    print("seed 4: search, and mint an evidence card, through the 0.14.0 MCP server", flush=True)

    async def calls(call):
        probes = await _probe("seed 4", call)
        for topic, hits in probes.items():
            if not hits or not hits[0][0].endswith(f"{topic}.md"):
                raise StepFailed(f"seed 4: {topic}.md is not the top hit for its own query: {hits}")
            followup = f"{topic}-followup.md"
            if f"{topic}-followup" in FOLLOWUPS and [h[0] for h in _trusted_hits(hits)][1:2] != [followup]:
                raise StepFailed(f"seed 4: {followup} is not the second trusted hit for {topic!r}: {hits}")
        evidence = await call("recall_evidence", {"query": _query(FACT_TOPIC), "k": 3})
        cards = [c for c in evidence.get("cards") or [] if c.get("structured_facts")]
        if not cards:
            raise StepFailed(f"seed 4: no evidence card carries the memo's fact: {json.dumps(evidence)[:1500]}")
        return probes, cards[:1]

    probes, cards = asyncio.run(_server_calls(prod, workdir, calls))

    # Neither public surface of 0.14.0 can apply a fact on a generation-backed PostgreSQL tenant,
    # measured on this job's first runs: `recall_apply_fact` refuses because that release's
    # generation-mode server reports its store's generation as `legacy` (GENERATION_MISMATCH,
    # then FRESH_SEARCH_INSUFFICIENT; fixed on master by #852 and #1033), and
    # `recall provenance apply --generation` revalidates the card against the legacy table and
    # then refuses its fresh search for want of a calibration (FRESH_SEARCH_UNAVAILABLE). So the
    # fact goes through 0.14.0's own controller and ledger, bound to the active generation, with
    # the card the server minted: the rows are the ones 0.14.0 writes, which is what the
    # fact-ledger migrations since have to carry.
    print("seed 5: apply a fact with 0.14.0's provenance controller, from a card the server minted", flush=True)
    from recall.fact_ledger import PostgresFactLedger
    from recall.provenance_cards import PostgresEvidenceCardStore
    from recall.provenance_controller import FactApplicationRequest, ProvenanceController
    from recall.types import AtomicFact

    dsn = prod["RECALL_SERVING_DSN"]
    decision = ProvenanceController(
        tenant_id=TENANT,
        generation_id=generation,
        cards=PostgresEvidenceCardStore(dsn, tenant_id=TENANT),
        ledger=PostgresFactLedger(dsn, tenant_id=TENANT),
        writer="upgrade-smoke",
    ).apply_fact(FactApplicationRequest(
        AtomicFact.from_payload(FACT), (cards[0]["card_id"],), "upgrade-smoke-1"
    ))
    print(f"    {decision.code} fact_id={decision.fact_id}")
    if str(decision.code) != "APPLIED":
        raise StepFailed(f"seed 5: the fact was not applied: {decision.code} {decision.detail}")

    async def read_facts(call):
        return await call("recall_current_facts", {})

    current = asyncio.run(_server_calls(prod, workdir, read_facts))
    facts = _fact_objects(current)
    if FACT["object"] not in facts:
        raise StepFailed(f"seed 5: the applied fact is not current: {json.dumps(current)[:1500]}")
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
    if os.environ.get("UPGRADE_SMOKE_SKIP_REGRANT") != "1":  # set only to red-prove the two-role leg
        _grant("verify 2", workdir, prod)

    print("verify 3: `recall schema status` reports compatible", flush=True)
    out = _run("verify 3", _recall("schema", "status"), env=prod, cwd=workdir)
    if "compatible: yes" not in out:
        raise StepFailed("verify 3: schema status does not say `compatible: yes`")

    # CHANGELOG.md and docs/MIGRATIONS.md: migration 0030 creates the current-facts tables empty,
    # and a ledger that already holds events needs one rebuild per tenant, or every read replays.
    print("verify 4: `recall provenance rebuild`, which 0030 requires for a ledger 0.14.0 wrote", flush=True)
    before = json.loads(_run("verify 4", _recall("provenance", "rebuild", "--check"), env=prod, cwd=workdir))
    print(f"    before the rebuild: current={before.get('current')!r}")
    _run("verify 4", _recall("provenance", "rebuild"), env=prod, cwd=workdir)
    after = json.loads(_run("verify 4", _recall("provenance", "rebuild", "--check", "--verify"),
                            env=prod, cwd=workdir))
    if after.get("current") is not True or (after.get("verify") or {}).get("matches") is not True:
        raise StepFailed("verify 4: the current-facts tables are not current and verified after the rebuild")

    print("verify 5: `recall doctor` passes", flush=True)
    _run("verify 5", _recall("doctor"), env=prod, cwd=workdir)

    print("verify 6: the old memos search the same, still trusted, and the old fact reads", flush=True)

    # Then a fact through the NEW server's `recall_apply_fact`, which 0.14.0's server refused on
    # a generation tenant (seed 5). The same fact again, so APPLIED or DUPLICATE both pass.
    async def read_calls(call):
        probes = await _probe("verify 6", call)
        current = await call("recall_current_facts", {})
        evidence = await call("recall_evidence", {"query": _query(FACT_TOPIC), "k": 3})
        cards = [c["card_id"] for c in evidence.get("cards") or [] if c.get("structured_facts")]
        applied = await call("recall_apply_fact", {
            **FACT, "evidence_card_ids": cards[:1], "request_id": "upgrade-smoke-2",
        })
        print(f"    recall_apply_fact: {applied.get('decision_code')}")
        if applied.get("allowed") is not True:
            raise StepFailed(f"verify 6: recall_apply_fact refused on the upgraded tenant: {applied}")
        return probes, current

    upgraded, current = asyncio.run(_server_calls(prod, workdir, read_calls))
    if _fact_objects(current) != state["facts"]:
        raise StepFailed(
            f"verify 6: facts written by {state['seeded_by']} do not read back: "
            f"before {state['facts']}, after {json.dumps(current)[:1500]}"
        )
    for topic, before in state["probes"].items():
        after = upgraded[topic]
        if _trusted_hits(after) != _trusted_hits(before):
            raise StepFailed(
                f"verify 6: the trusted hits for {topic!r} changed: before {before}, after {after}"
            )
        # A memo returned both before and after must score the same dense cosine, which only an
        # unchanged stored vector gives; this covers the below-threshold hits too.
        old = {hit[0]: hit[1] for hit in before}
        moved = {hit[0]: (old[hit[0]], hit[1]) for hit in after if hit[0] in old and old[hit[0]] != hit[1]}
        if moved:
            raise StepFailed(f"verify 6: cosines for {topic!r} changed (before, after): {moved}")

    # The control. A release may change retrieval policy, so matching 0.14.0 is not enough on
    # its own; what an upgrade must not do is answer differently from a fresh install of the
    # same release. So the same notes are installed fresh, by this release, into a tenant of
    # their own, and the upgraded tenant's trusted hits must match it exactly.
    print("verify 7: the same notes installed fresh by this release, as the control", flush=True)
    _install("verify 7", workdir, FRESH_TENANT, legacy_index=False)
    fresh_env = _env(workdir, production=True, tenant=FRESH_TENANT)
    fresh = asyncio.run(_server_calls(fresh_env, workdir, lambda call: _probe("verify 7", call)))
    for topic in PROBES:
        if _trusted_hits(upgraded[topic]) != _trusted_hits(fresh[topic]):
            raise StepFailed(
                f"verify 7: the upgraded tenant's trusted hits for {topic!r} differ from a fresh "
                f"install of this release: upgraded {upgraded[topic]}, fresh {fresh[topic]}"
            )
    # The control tenant has no legacy rows: doctor must judge it by its active generation and
    # that generation's calibration, which is what the server serves by.
    out = _run("verify 7", _recall("doctor", tenant=FRESH_TENANT), env=fresh_env, cwd=workdir)
    if "certified for active generation" not in out:
        raise StepFailed("verify 7: doctor does not report the generation's certified calibration")

    # By both canonical names. The tenant holds the memo twice, as 0.14.0's FIRST_CALIBRATION.md
    # leaves it: `recall index` stored the path, the generation stored the file:// URI. Since
    # #1149 a forget refuses the short name `recall_search` prints (`hiring.md`) as ambiguous
    # and lists these two; 0.14.0 has no such refusal.
    print(f"verify 8: forget {FORGET_TOPIC}.md, a memo {state['seeded_by']} indexed", flush=True)
    memo = workdir / "notes" / f"{FORGET_TOPIC}.md"
    out = _run("verify 8", _recall("forget", str(memo), memo.as_uri(), "--yes"), env=prod, cwd=workdir)
    if "outbox failed" in out:
        raise StepFailed("verify 8: forget warned that scrubbing the migration outbox failed")

    # The erasure changes the generation's corpus fingerprint, so strict search must refuse
    # (CALIBRATION_STALE) until a recalibration: refusing serves nothing, which is correct.
    async def search_forgotten(call):
        return await call("recall_search", {"query": _query(FORGET_TOPIC), "k": 3}, refusal_ok=True)

    def require_forgotten(step: str, payload: dict) -> None:
        remaining = _top(payload)
        print(f"    {FORGET_TOPIC}: {remaining} (trust_state={payload.get('trust_state')!r}, "
              f"code={payload.get('code')!r})")
        if any(s.endswith(f"{FORGET_TOPIC}.md") for s in remaining):
            raise StepFailed(f"{step}: {FORGET_TOPIC}.md still searchable after forget: {remaining}")

    payload = asyncio.run(_server_calls(prod, workdir, search_forgotten))
    require_forgotten("verify 8", payload)
    if payload.get("refused") and payload.get("code") != "CALIBRATION_STALE":
        raise StepFailed(f"verify 8: search after forget refused for an unexpected reason: {payload.get('code')!r}")

    # `recall calibration auto` does not apply: a forget supersedes the generation's calibration
    # in place, and auto only carries one forward to a new generation. It must say so and name
    # the labelled refit, which follows, with the forgotten memo's question taken out of the set.
    print("verify 9: recalibrate as the refusal advises, then search again", flush=True)
    out = _run("verify 9", _recall("calibration", "auto", "--generation", state["generation"]),
               env=prod, cwd=workdir, ok_codes=(0, 1))
    if f"calibration calibrate --generation {state['generation']}" not in out:
        raise StepFailed("verify 9: `calibration auto` after a forget does not name the labelled refit")
    labels = [q for q in json.loads((workdir / "queries.json").read_text(encoding="utf-8"))
              if q["query"] != _query(FORGET_TOPIC)]
    (workdir / "queries-after-forget.json").write_text(json.dumps(labels), encoding="utf-8")
    _run("verify 9", _recall("calibration", "calibrate", "--generation", state["generation"],
                             "--queries", str(workdir / "queries-after-forget.json"), "--publish"),
         env=prod, cwd=workdir)
    payload = asyncio.run(_server_calls(prod, workdir, search_forgotten))
    if payload.get("refused"):
        raise StepFailed(f"verify 9: search still refused after recalibrating: {payload.get('code')!r}")
    _require_trusted("verify 9", FORGET_TOPIC, payload)
    require_forgotten("verify 9", payload)
    print("upgrade verified")


def floor(workdir: Path) -> None:
    """The pgvector floor: on pgvector older than 0.8.0 the upgrade refuses before migrating.

    Run with the new wheel against a database whose `vector` extension is 0.7.x and whose schema
    an earlier release already applied. `recall schema apply` must refuse and name the floor,
    the schema must stay exactly where the earlier release left it, and `recall doctor` must
    report the old pgvector as a blocking problem.
    """
    env = _env(workdir, production=True)
    print("floor 1: `recall schema status` before (the earlier release's schema)", flush=True)
    before = _run("floor 1", _recall("schema", "status"), env=env, cwd=workdir, ok_codes=(0, 1))
    current = re.search(r"^current: (\S+)", before, re.MULTILINE)
    if not current or current.group(1) == "none":
        raise StepFailed("floor 1: no earlier schema to protect; the job did not seed one")

    print("floor 2: `recall schema apply` refuses, naming the floor", flush=True)
    proc = subprocess.run(_recall("schema", "apply"), cwd=workdir, env=env, capture_output=True,
                          text=True, timeout=600)
    print("    " + (proc.stdout + proc.stderr).strip().replace("\n", "\n    "))
    if proc.returncode == 0:
        raise StepFailed("floor 2: `schema apply` succeeded on pgvector older than 0.8.0")
    if "older than 0.8.0" not in proc.stdout + proc.stderr:
        raise StepFailed("floor 2: `schema apply` failed without naming the pgvector floor")

    print("floor 3: nothing was migrated", flush=True)
    after = _run("floor 3", _recall("schema", "status"), env=env, cwd=workdir, ok_codes=(0, 1))
    still = re.search(r"^current: (\S+)", after, re.MULTILINE)
    if not still or still.group(1) != current.group(1):
        raise StepFailed(f"floor 3: the schema moved from {current.group(1)} despite the refusal")

    print("floor 4: `recall doctor` reports the old pgvector as blocking", flush=True)
    out = _run("floor 4", _recall("doctor"), env=env, cwd=workdir, ok_codes=(1,))
    if not re.search(r"FAIL\s+pgvector", out):
        raise StepFailed("floor 4: doctor does not fail the pgvector check")
    print("floor verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("seed", "verify", "floor"))
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        {"seed": seed, "verify": verify, "floor": floor}[args.phase](workdir)
    except StepFailed as exc:
        print(f"\nUPGRADE FAILED at {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
