"""Turn a finished install into the three MCP servers that serve it, and refuse to lie about them.

`.mcp.json` is the artifact that decides whether any of the preceding work is reachable. Every
variable below is set in the server's OWN `env` block rather than left to the operator's shell,
because a stdio server launched with an explicit `env` inherits nothing: a corpus that searches
correctly from the terminal answered `INDEX_NOT_READY` through the client for exactly that reason.
That per-block isolation is also what lets one machine serve `docs` in production mode and `memory`
in development mode at the same time.

**The rule this module exists to enforce: a server block is only written for a tenant that can
actually answer.** Three cases, and they are not interchangeable:

* **Certified and promoted** → `RECALL_ENV=production` with strict trust. `RECALL_ENV` is what
  selects `GenerationStore` (`recall_mcp/server.py:627`), so a calibrated generation is only reached
  through it, and the published calibration is what makes strict trust answerable.

* **Degraded with a predecessor still serving** → production, but development trust, and the block
  carries a note saying the generation just built is NOT the one serving. Relaxing trust here is a
  judgement about a calibration that no longer matches the corpus on disk, not about the older
  generation being broken.

* **Degraded with nothing serving** → **no block at all.** This is the case worth stating, because
  it is the one a wizard gets wrong. The generation was deliberately not promoted, so under
  `RECALL_ENV=production` the tenant has no active generation and `GenerationStore.snapshot` raises
  `NoActiveGeneration` from OUTSIDE `trusted_search`'s try block: the caller gets a raw exception
  with no failure code and no advice. Writing `RECALL_TRUST_MODE=development` does not fix it, and
  it is tempting to think it would, because the failure is upstream of the trust gate entirely.
  A configured server that raises on every query is worse than an absent one, which at least says
  what is missing.

An uncalibrated corpus (`memory`) gets development trust and NO `RECALL_ENV`, because its rows live
in the legacy `chunks` table that only `PgVectorStore` reads, and production mode routes past it.

⚠️ **Docker autostart is deliberately not done here.** Making a container start on login is a change
to the machine, not to the project: on Windows it is a Docker Desktop setting, and a driver that
silently edits it would be modifying system configuration the operator did not ask about. The report
names the container instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from recall.wizard.corpora import CorpusPlan, CorpusSpec

__all__ = [
    "ServerBlock",
    "SmokeResult",
    "UnservableTenant",
    "mcp_config",
    "server_blocks",
    "write_mcp_config",
    "write_project_files",
    "write_runtime_profile",
]


@dataclass(frozen=True)
class ServerBlock:
    """One MCP server, its tenant, and the reason it is configured the way it is."""

    name: str
    tenant: str
    env: dict[str, str]
    #: Why this block looks like this, carried so the report can say it and a reader of the JSON is
    #: not left guessing why one tenant is strict and another is not.
    rationale: str


@dataclass(frozen=True)
class SmokeResult:
    """One query put through one configured server, and what came back.

    **A raise is the failure; an abstention is not.** That distinction is the whole design. Abstaining
    is a trust decision the gate is entitled to make, and a smoke test that treated it as broken
    would fail on a server behaving correctly. What must never happen is an exception, because that
    means the block cannot serve at all: a wrong `RECALL_ENV` reads a table the build never wrote to,
    and a tenant with no active generation raises `NoActiveGeneration` from outside the trust gate.

    So this exists to check the reasoning in `server_blocks` against the database rather than trust
    it. If that reasoning is ever wrong about which tenants are servable, `error` is where it shows.
    """

    tenant: str
    #: The query actually used, drawn from the tenant's own indexed text so a hit is possible at all.
    query: str
    hits: int
    abstained: bool
    trust_state: str
    failure_code: str | None
    #: Set only when the search RAISED. Non-None means this server cannot answer.
    error: str | None = None
    #: True when the tenant has no rows to draw a query from. Expected for a fresh `memory`.
    empty: bool = False

    @property
    def answered(self) -> bool:
        """Reached the trust gate without raising. Abstaining still counts as answering."""
        return self.error is None and not self.empty


@dataclass(frozen=True)
class UnservableTenant:
    """A tenant deliberately given no server, and why. Reported, never silently omitted."""

    tenant: str
    reason: str


def server_blocks(
    plan: CorpusPlan,
    *,
    dsn: str,
    promoted: frozenset[str],
    serving: frozenset[str],
) -> tuple[tuple[ServerBlock, ...], tuple[UnservableTenant, ...]]:
    """The blocks to write, and the tenants deliberately left out.

    `promoted` is the tenants whose freshly built generation went live. `serving` is the tenants that
    have SOME active generation, which includes a predecessor left in place by a degraded run. The
    two are different questions and collapsing them is what produces a server that raises.
    """
    blocks: list[ServerBlock] = []
    unservable: list[UnservableTenant] = []

    for spec in plan.corpora:
        if not spec.calibrated:
            blocks.append(_legacy_block(spec, dsn=dsn, embedder=plan.embedder))
            continue
        if spec.tenant in promoted:
            blocks.append(
                _generation_block(
                    spec,
                    dsn=dsn,
                    embedder=plan.embedder,
                    trust="strict",
                    rationale=(
                        "certified and promoted, so the published calibration backs strict trust"
                    ),
                )
            )
            continue
        if spec.tenant in serving:
            blocks.append(
                _generation_block(
                    spec,
                    dsn=dsn,
                    embedder=plan.embedder,
                    trust="development",
                    rationale=(
                        "certification fell short, so an EARLIER generation is what serves this "
                        "tenant. Trust is relaxed for this server only, because the calibration "
                        "that is published no longer describes the corpus on disk. Re-run once the "
                        "corpus can certify, then tighten this back to strict."
                    ),
                )
            )
            continue
        unservable.append(
            UnservableTenant(
                tenant=spec.tenant,
                reason=(
                    "certification fell short and this tenant has no earlier generation, so nothing "
                    "is promoted and nothing can answer. A production server would raise "
                    "NoActiveGeneration from outside the trust gate, with no failure code and no "
                    "advice, so no server is configured for it. Add content and re-run, or promote "
                    "the built generation deliberately to serve it uncertified."
                ),
            )
        )

    return tuple(blocks), tuple(unservable)


def _base_env(spec: CorpusSpec, *, dsn: str, embedder: str) -> dict[str, str]:
    return {"RECALL_DSN": dsn, "RECALL_EMBEDDER": embedder, "RECALL_TENANT": spec.tenant}


def _generation_block(
    spec: CorpusSpec,
    *,
    dsn: str,
    embedder: str,
    trust: str,
    rationale: str,
) -> ServerBlock:
    env = _base_env(spec, dsn=dsn, embedder=embedder)
    # `RECALL_ENV=production` is what selects `GenerationStore`, so a calibrated corpus is only
    # reachable through it. Without this the server would read the legacy `chunks` table, which a
    # generation build never wrote to, and answer nothing while looking configured.
    env["RECALL_ENV"] = "production"
    env["RECALL_TRUST_MODE"] = trust
    return ServerBlock(
        name=spec.tenant, tenant=spec.tenant, env=env, rationale=rationale
    )


def _legacy_block(spec: CorpusSpec, *, dsn: str, embedder: str) -> ServerBlock:
    env = _base_env(spec, dsn=dsn, embedder=embedder)
    # No RECALL_ENV. This corpus lives in the legacy `chunks` table, which only `PgVectorStore`
    # reads; production mode would route past it and find nothing.
    env["RECALL_TRUST_MODE"] = "development"
    return ServerBlock(
        name=spec.tenant,
        tenant=spec.tenant,
        env=env,
        rationale=(
            "writable and never calibrated, so it has no generation and no calibration to be "
            "strict about. A fresh memory directory cannot meet the certification floor, and this "
            "is the one corpus that must accept writes after install, which production mode refuses."
        ),
    )


def mcp_config(blocks: tuple[ServerBlock, ...], *, project_root: Path) -> dict[str, object]:
    """The `.mcp.json` document, in the shape a working configuration already uses."""
    return {
        "mcpServers": {
            block.name: {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "recall_mcp.server"],
                "cwd": str(project_root),
                "env": dict(block.env),
            }
            for block in blocks
        }
    }


def write_project_files(
    *,
    project_root: Path,
    dsn: str,
    embedder: str,
    memory_dir: Path,
) -> tuple[Path, ...]:
    """Write `.env`, the `CLAUDE.md` block and `memory/MEMORY.md`, returning what was touched.

    Every one of these reuses `recall.setup`'s writers rather than reimplementing them, and that
    matters for a specific reason: all three are BLOCK-scoped, delimited by begin/end markers, so
    they update an operator's existing file instead of replacing it. A wizard that overwrote a
    project's `CLAUDE.md` would destroy work no installer has any business touching.

    `.env` carries the serving DSN and embedder for the CLI, not for the MCP servers. The servers get
    their variables in their own `env` blocks, because a stdio server launched with an explicit `env`
    inherits nothing, so `.env` alone would leave them on defaults.
    """
    from recall.setup import _update_env_block, scaffold_claude_md, scaffold_memory_index

    written: list[Path] = []

    env_path = project_root / ".env"
    _update_env_block(env_path, {"RECALL_DSN": dsn, "RECALL_EMBEDDER": embedder})
    written.append(env_path)

    claude_md = project_root / "CLAUDE.md"
    scaffold_claude_md(claude_md)
    written.append(claude_md)

    if scaffold_memory_index(memory_dir):
        written.append(memory_dir / "MEMORY.md")

    return tuple(written)


def write_runtime_profile(
    *,
    compose_path: Path,
    project: str,
    compose_project: str,
    shared_profile: str = "user",
    path: Path | None = None,
) -> Path:
    """Write `runtime.json`, the handoff from the wizard to the desktop UI.

    **`recall.desktop.profiles.save_profile` existed with ZERO callers.** `main.py` reads the file
    and, finding none, falls back to `RuntimeProfile(mode=DOCKER, compose_file=
    "docker-compose.desktop.yml")` — a RELATIVE path resolved against the process working directory,
    so Docker mode worked only when the app happened to be launched from the repository root. The
    wizard is the missing writer, and it writes an ABSOLUTE path.

    This is also what decides which projects the UI offers. Verified by constructing the real
    window offscreen: it makes no runtime calls at all on startup, and its scope selector is built
    from the PROFILE rather than from `list_tenants`. So what is written here is what the user sees.

    `default_tenant` is the project SCOPE (`myapp`), not a full tenant (`myapp-docs`). The UI
    appends the corpus kind itself in `SourceSelection.physical_tenant`, and handing it a complete
    tenant would produce `myapp-docs-docs`.

    Imported lazily, and reachable without PySide6: the desktop extra may not be installed at the
    moment the wizard runs, and refusing to record the configuration because the GUI is absent
    would make the install order matter for no reason.
    """
    from recall.desktop.models import RuntimeMode, RuntimeProfile
    from recall.desktop.profiles import save_profile

    profile = RuntimeProfile(
        mode=RuntimeMode.DOCKER,
        compose_file=str(compose_path),
        compose_project=compose_project,
        default_tenant=project,
        shared_profile=shared_profile,
    )
    return save_profile(profile, path)


@dataclass(frozen=True)
class ClientApproval:
    """What was done about the client's approval gate, so the report can say rather than assume."""

    #: Absolute path of the client config, or None when there is none to write.
    config_path: Path | None
    #: Server names now recorded as approved for this project root.
    approved: tuple[str, ...]
    #: Why nothing was written, when nothing was. Empty on success.
    skipped_reason: str = ""

    @property
    def recorded(self) -> bool:
        return bool(self.approved)


def claude_config_path() -> Path:
    """Claude Code's own configuration. Not recall's, which is why it is touched so carefully."""
    return Path.home() / ".claude.json"


def approve_mcp_servers(
    project_root: Path, names: tuple[str, ...], *, config_path: Path | None = None
) -> ClientApproval:
    """Record the wizard's servers as approved for `project_root`, so they load without a prompt.

    ⚠️ **A `.mcp.json` the client has not been told to trust yields NO TOOLS, and says nothing.**
    Project-scoped servers are gated: the client asks for approval in an interactive session, and
    until it is given, `mcp__recall__*` simply is not there. A first-run user sees an assistant that
    cannot search their corpus and no error explaining why — the silent-nothing failure this
    project keeps meeting. Verified against Claude Code's documented scope table: `local` and
    `project` scope load only in the current project and `project` scope prompts for approval;
    `user` scope loads everywhere without one.

    Recording it here is defensible for one specific reason: the gate exists so that **a cloned
    repository cannot approve its own servers**, and this is the opposite situation — the person at
    the keyboard just ran an installer against their own machine, naming their own database. The
    consent the gate is asking for is the consent they already gave. It is still reported rather
    than done silently, and it writes exactly one key.

    Written with a backup and atomically, because `~/.claude.json` is CLAUDE CODE's file, not ours:
    it holds every project the user has, and the client owns its schema. Nothing else in it is
    touched, an unreadable or absent file is a refusal rather than an overwrite, and the backup is
    what makes a bad write recoverable.
    """
    target = config_path or claude_config_path()
    if not names:
        return ClientApproval(config_path=target, approved=(), skipped_reason="no servers to approve")

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        # No client config means Claude Code has not run here. Creating one would be inventing a
        # file for another application; the client writes it on first run and reads `.mcp.json`
        # then, prompting normally.
        return ClientApproval(
            config_path=target,
            approved=(),
            skipped_reason=f"no Claude Code config at {target} ({exc.strerror or exc})",
        )

    try:
        document = json.loads(raw)
    except ValueError as exc:
        return ClientApproval(
            config_path=target,
            approved=(),
            skipped_reason=f"{target} is not readable JSON ({exc}); left untouched",
        )
    if not isinstance(document, dict):
        return ClientApproval(
            config_path=target, approved=(), skipped_reason=f"{target} is not a JSON object"
        )

    projects = document.get("projects")
    if not isinstance(projects, dict):
        projects = {}
        document["projects"] = projects

    # The client keys projects by the absolute path it was launched from.
    key = str(project_root.resolve())
    entry = projects.get(key)
    if not isinstance(entry, dict):
        entry = {}
        projects[key] = entry

    enabled = entry.get("enabledMcpjsonServers")
    merged = list(enabled) if isinstance(enabled, list) else []
    for name in names:
        if name not in merged:
            merged.append(name)
    entry["enabledMcpjsonServers"] = merged

    backup = target.with_name(target.name + ".recall-backup")
    backup.write_text(raw, encoding="utf-8", newline="\n")
    temporary = target.with_name(target.name + ".recall-tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)
    return ClientApproval(config_path=target, approved=tuple(names))


def write_mcp_config(path: Path, config: dict[str, object]) -> None:
    """Write `.mcp.json`, merging into any existing `mcpServers` rather than replacing the file.

    An operator's `.mcp.json` is very likely to hold servers this wizard knows nothing about, and
    replacing the file would silently delete them. Only the keys this install owns are overwritten.

    `newline="\\n"` because Python would otherwise write CRLF on Windows, and a JSON file that
    changes every line on every platform is a diff nobody can read.
    """
    existing: dict[str, object] = {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    except (OSError, json.JSONDecodeError):
        existing = {}

    servers = existing.get("mcpServers")
    merged = dict(servers) if isinstance(servers, dict) else {}
    incoming = config.get("mcpServers")
    if isinstance(incoming, dict):
        merged.update(incoming)
    existing["mcpServers"] = merged

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(existing, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)
