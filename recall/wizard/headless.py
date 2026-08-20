"""One config file in, one report out: the entry point CI and the installer both drive.

Headless because the interesting question about the wizard is whether the pipeline works, and a GUI
answers none of it. A CI job can run this against a throwaway container and get the same answer a
user gets, which is the only way the install path stays honest between releases.

Four decisions worth stating, because each is a place a driver usually gets it wrong:

**A missing or malformed config is refused by name.** The reader is a first-run installer or a CI
log, and neither can act on a `KeyError` raised inside a module they have never heard of. That
applies to a wrong VALUE as much as an absent key: an unknown embedder, a root that is a file and an
unreachable database are the three most likely first-run conditions, and all three used to arrive as
tracebacks while this docstring claimed otherwise.

**One corpus failing does not abort the others, and the report is never lost.** They are independent,
and a driver that dies on the second of three leaves the user with less than one that names which
corpus failed and why. Crucially this holds for a crash and not only for a refusal: `promote` retires
whatever the tenant was serving, so a driver that throws away the report after promoting `docs`
leaves nobody able to say which generation is now active.

**A REFUSED corpus, a FAILED corpus and a DEGRADED corpus are three outcomes, not two.** A refusal is
a configuration problem the user must act on, before anything was built. A failure is a crash partway
through, which may have left irreversible work behind. A degraded corpus built, validated, did not
certify, and was deliberately not promoted.

**"Degraded" is only a working install if something is still serving.** On an upgrade the previous
generation keeps answering, so a named limitation is the whole story. On a FIRST install there is no
previous generation, the tenant answers nothing, and under production trust a query raises
`NoActiveGeneration` from outside `trusted_search`'s try block, so the caller gets a raw exception
with no failure code and no advice. Those two states rendered identically, under the heading "install
complete", with a reason that said "whatever this tenant was serving still serves" about an empty
tenant. `ok` now turns on whether the tenant can answer, not merely on whether a build finished.

The schema is applied at the embedder's own dimension, because the vector dimension is welded to the
table: a hardcoded one produces a schema no chosen embedder fits, discovered as an insert error
minutes into the first build.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from recall.calibration_v2 import CalibrationRepository
from recall.embeddings import Embedder, resolve_embedder
from recall.generations import GenerationManager
from recall.store import redacted_dsn, scrub_dsn_secrets
from recall.wizard.corpora import (
    DEFAULT_PROJECT,
    RELATIVE_ROOT,
    CorpusPlan,
    CorpusSpec,
    default_plan,
)
from recall.wizard.pipeline import CorpusOutcome, PipelineRefusal, run_corpus
from recall.wizard.stack import (
    COMPOSE_NAME as _COMPOSE_NAME,
    StackSpec,
    bring_up,
    choose_port,
    compose_document,
    existing_port,
    host_dsn,
    wait_for_database,
    write_compose,
)
from recall.wizard.state import WizardState, config_digest, load_state, save_state
from recall.wizard.wiring import (
    ServerBlock,
    SmokeResult,
    UnservableTenant,
    LocalScopeRegistration,
    register_local_scope,
    server_blocks,
    write_project_files,
    write_runtime_profile,
)

__all__ = [
    "ConfigRefusal",
    "Failure",
    "HeadlessConfig",
    "HeadlessReport",
    "LegacyIndex",
    "PipelineRefusal",
    "Refusal",
    "load_config",
    "run_headless",
]

#: Width of the tenant column, and the continuation indent derived from it. Both were unnamed
#: literals repeated six times, which meant a tenant longer than the number silently broke every
#: column and a reason containing a newline lost its indent entirely.
_TENANT_COLUMN = 8
_GUTTER = 2


@dataclass(frozen=True)
class HeadlessConfig:
    """Everything the driver needs. No REQUIRED field has a default it could guess wrong.

    **`dsn` and `data_root` are alternatives, and exactly one must be present.** `dsn` points at a
    database somebody else provisioned, which is how CI drives this. `data_root` says "make me one,
    here", which is what a person installing on Windows wants and what the desktop UI then shares.
    Accepting both would mean ignoring one, and a setting that is accepted and discarded is the
    defect this branch has already fixed twice.
    """

    embedder: str
    corpus_version: str
    docs_root: Path
    code_root: Path
    memory_root: Path
    #: An existing database. Absent when `data_root` is set and the wizard provisions one.
    dsn: str | None = None
    #: The DDL owner. Defaults to `dsn` when the two are the same principal, which is the ordinary
    #: single-role install.
    migration_dsn: str | None = None
    #: Where the user wants their index kept. Provisions the stack the desktop UI also uses, so
    #: both surfaces reach ONE store. Absolute, because a relative location resolves against
    #: whatever directory the installer happened to run from.
    data_root: Path | None = None
    #: The project scope. It names the tenants (`{project}-docs` and so on) AND is stamped on every
    #: chunk as provenance, because for the person installing this they are one idea: "my project".
    project: str = DEFAULT_PROJECT
    #: Optional: the role the SERVING dsn authenticates as, when it differs from the DDL owner.
    #: No migration emits a GRANT, because the role name is a deployment decision the packaged SQL
    #: cannot know, so a two-role install has to be told the name or it will not work.
    serving_role: str | None = None
    #: Optional: the project this install belongs to. It is the `cwd` every registered server is
    #: launched from, and where `.env`, the `CLAUDE.md` block and `MEMORY.md` are written. Optional
    #: rather than derived from the config's own directory, because editing an operator's project
    #: files in a guessed location is a side effect they cannot predict, and a config may well live
    #: somewhere temporary. When absent the report says the wiring was skipped and names the key.
    project_root: Path | None = None

    @property
    def resolved_dsn(self) -> str:
        """The serving DSN, after the database has been resolved.

        `dsn` is genuinely optional on the way IN, because `data_root` is the alternative. It is
        never optional by the time anything connects, and this property is where that invariant is
        stated once instead of being narrowed at every call site. Reaching it unresolved is a
        programming error in the driver, not a user's configuration mistake, so it raises rather
        than refusing.
        """
        if not self.dsn:
            raise RuntimeError(
                "the database has not been resolved yet; `run_headless` provisions it from "
                "`data_root` before anything connects"
            )
        return self.dsn

    @property
    def resolved_migration_dsn(self) -> str:
        """The DDL owner, which defaults to the serving DSN on a single-role install."""
        return self.migration_dsn or self.resolved_dsn


#: Every key the config must carry, DERIVED from the fields that have no default rather than
#: retyped. The hand-written copy was one of five listings of the same vocabulary, and a key added
#: to `HeadlessConfig` without a matching entry became a silently optional key.
_REQUIRED = tuple(f.name for f in fields(HeadlessConfig) if f.default is MISSING)
_OPTIONAL = tuple(f.name for f in fields(HeadlessConfig) if f.default is not MISSING)
_ROOT_KEYS = tuple(name for name in _REQUIRED if name.endswith("_root"))


class ConfigRefusal(PipelineRefusal):
    """A refusal about the config file, carrying the offending keys as data.

    The keys are an attribute and not only prose because the prose lists EVERY required key, so a
    test matching the message against one key name matched no matter which key the message actually
    blamed. Eight parametrised assertions passed against a mutation that named `dsn` for every
    absent key. Assert on `.keys`.
    """

    def __init__(self, message: str, keys: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.keys = keys


@dataclass(frozen=True)
class Refusal:
    """A corpus the driver could not even attempt, and the reason. Nothing was built."""

    tenant: str
    reason: str


@dataclass(frozen=True)
class LegacyIndex:
    """An uncalibrated corpus written into the legacy `chunks` table, and what it wrote.

    Reported with its counts rather than as a skip, because `memory` is a third of the install and
    the report used to assert it had been "indexed into the legacy chunks table" while the driver
    did nothing at all with it. Counts are the difference between a claim and a result.
    """

    tenant: str
    files: int
    chunks: int


@dataclass(frozen=True)
class Failure:
    """A corpus that crashed partway through, and may have left irreversible work behind.

    Distinct from `Refusal` because the two demand different things of the reader: a refusal is a
    configuration problem with nothing to clean up, while a failure means the run got far enough to
    build, and possibly far enough to promote an earlier corpus and retire its predecessor.
    """

    tenant: str
    error: str


@dataclass(frozen=True)
class HeadlessReport:
    """What happened to every corpus in the plan."""

    outcomes: tuple[CorpusOutcome, ...] = ()
    refused: tuple[Refusal, ...] = ()
    failures: tuple[Failure, ...] = ()
    #: Uncalibrated corpora, indexed into the legacy `chunks` table. `memory` has no generation, so
    #: handing it to `run_corpus` would produce a promoted generation nothing can calibrate, but it
    #: still has to be INDEXED or its server answers nothing.
    indexed: tuple[LegacyIndex, ...] = ()
    #: Tenants taken from a previous run's recorded state rather than rebuilt. Reported, because
    #: "this took four seconds" and "this took eleven minutes" should not look identical, and an
    #: operator who expected a rebuild needs to see that they did not get one.
    reused: tuple[str, ...] = ()
    #: What was done about registering these servers with Claude Code. None when no wiring ran at
    #: all. Carried on the report because "the servers are configured" and "the servers will load"
    #: are different claims, and only the second one is what the user wanted.
    registration: LocalScopeRegistration | None = None
    #: MCP servers the wizard registered, and the tenants deliberately left without one.
    servers: tuple[ServerBlock, ...] = ()
    unservable: tuple[UnservableTenant, ...] = ()
    #: One query put through each configured server. This is the difference between "a config was
    #: written" and "the install answers", and it checks `server_blocks`'s reasoning against the
    #: database rather than trusting it.
    smoke: tuple[SmokeResult, ...] = ()
    #: Every file the wiring step created or edited. Named because these are edits to files the
    #: operator owns, made block-scoped so they merge rather than replace, and a block-scoped edit
    #: is invisible in a directory listing.
    files_written: tuple[Path, ...] = ()

    @property
    def unserved(self) -> tuple[str, ...]:
        """Tenants left unable to answer: not promoted, and with nothing serving before this run."""
        return tuple(
            o.tenant for o in self.outcomes if not o.promoted and not o.previously_serving
        )

    @property
    def ok(self) -> bool:
        """True when every corpus in the plan can answer a query.

        A DEGRADED corpus counts as ok only when the tenant has a previous generation still
        serving. A degraded FIRST install is not a working install with a limitation, it is a
        tenant that answers nothing, and exit 0 is the only thing CI reads.

        A server whose smoke query RAISED is also not ok, whatever the corpora did. That is the
        whole point of running one: the corpora can all be built and promoted correctly and the
        configuration still be unable to reach them.

        **And an install nothing was registered with is not ok either.** The servers go into Claude
        Code's own config at local scope, and when that write is skipped there is no other place a
        client will find them: not now, and not after a restart. The corpora would be perfect and
        the user would meet an assistant with no recall tools and nothing naming the cause, which is
        this project's recurring failure and the reason the exit code has to carry it.
        """
        raised = any(s.error for s in self.smoke)
        unregistered = self.registration is not None and not self.registration.recorded
        return not (self.refused or self.failures or self.unserved or raised or unregistered)

    @property
    def degraded(self) -> tuple[str, ...]:
        return tuple(o.tenant for o in self.outcomes if not o.certified)

    def render(self) -> str:
        """The text a headless caller reads. Every corpus in the plan appears, including skips."""
        tenants = [
            *(o.tenant for o in self.outcomes),
            *(r.tenant for r in self.refused),
            *(f.tenant for f in self.failures),
            *(i.tenant for i in self.indexed),
        ]
        width = max([_TENANT_COLUMN, *(len(t) for t in tenants)]) if tenants else _TENANT_COLUMN
        indent = " " * (_GUTTER + width + 1)

        def detail(text: str) -> str:
            """Wrapped and re-indented, so a multi-sentence reason keeps the column it started in."""
            return textwrap.indent(textwrap.fill(text, 96), indent)

        lines: list[str] = []
        for outcome in self.outcomes:
            head = f"{' ' * _GUTTER}{outcome.tenant:<{width}} "
            note = "  [reused from a previous run]" if outcome.tenant in self.reused else ""
            if outcome.certified:
                lines.append(
                    f"{head}certified and promoted  "
                    f"({outcome.answerable} answerable / {outcome.unanswerable} unanswerable, "
                    f"{outcome.generation_id}){note}"
                )
            else:
                lines.append(f"{head}DEGRADED, not promoted  {outcome.generation_id}")
                # `degraded_reason` is guaranteed non-empty by `CorpusOutcome.__post_init__`; the
                # fallback is here so a future producer that bypasses it cannot print "None" as
                # the operator's advice, which is what shipped before that invariant existed.
                lines.append(
                    detail(outcome.degraded_reason or "no reason recorded (this is a defect)")
                )
            if outcome.unverified_embedder:
                lines.append(
                    detail("note: no verifiable embedder identity, generation is unverified")
                )
        for refusal in self.refused:
            lines.append(f"{' ' * _GUTTER}{refusal.tenant:<{width}} REFUSED")
            lines.append(detail(refusal.reason))
        for failure in self.failures:
            lines.append(f"{' ' * _GUTTER}{failure.tenant:<{width}} FAILED")
            lines.append(detail(failure.error))
        for entry in self.indexed:
            # Counts, because this line used to assert the corpus was "indexed into the legacy
            # chunks table" while the driver did nothing with it at all. A count cannot be claimed
            # without doing the work.
            lines.append(
                f"{' ' * _GUTTER}{entry.tenant:<{width}} indexed  "
                f"({entry.chunks} chunks from {entry.files} files, legacy chunks table)"
            )
            if not entry.chunks:
                # Says what is true of the CORPUS, and stops there.
                #
                # 🔁 Corrected twice, in opposite directions, which is why it now says less. The
                # first version claimed this tenant's server "carries RECALL_TRUST_MODE=development"
                # when the wizard wrote no `.mcp.json` at all. The replacement said the wizard "does
                # not write that configuration" — true when written, and false by the time Phase 4
                # landed: the first real install printed that sentence three lines above a server
                # block showing exactly that setting, so the report contradicted itself.
                #
                # The server lines below report themselves, and they report what actually happened
                # rather than what this line predicts. A sentence about a neighbouring section is a
                # sentence that goes stale when the neighbour changes.
                lines.append(
                    detail(
                        "the directory was empty, which is normal on a first install: this tenant "
                        "is writable and fills up as you use it, and it is never calibrated."
                    )
                )

        for block in self.servers:
            # RECALL_DSN is omitted deliberately: it carries the password, and this report is what
            # a CI job prints into a log. The variables that are shown are the ones an operator
            # needs to sanity-check, and the DSN is the one they already know.
            shown = ", ".join(f"{k}={v}" for k, v in block.env.items() if k != "RECALL_DSN")
            lines.append(f"{' ' * _GUTTER}{block.name:<{width}} server   ({shown})")
            lines.append(detail(block.rationale))
        for missing in self.unservable:
            lines.append(f"{' ' * _GUTTER}{missing.tenant:<{width}} NO SERVER")
            lines.append(detail(missing.reason))
        for result in self.smoke:
            head = f"{' ' * _GUTTER}{result.tenant:<{width}} "
            if result.error:
                lines.append(f"{head}SMOKE FAILED")
                lines.append(
                    detail(
                        f"a query through this server's own configuration raised: {result.error}. "
                        "The corpus may be fine; this server cannot reach it."
                    )
                )
            elif result.empty:
                lines.append(f"{head}smoke skipped")
                lines.append(
                    detail("no indexed rows to draw a query from, which is normal for a fresh tenant")
                )
            else:
                verdict = "abstained" if result.abstained else f"{result.hits} hits"
                # An abstention is NOT a failure: it is a trust decision the gate is entitled to
                # make. What the smoke test proves is that the query reached the gate at all.
                lines.append(
                    f"{head}smoke ok  ({verdict}, trust={result.trust_state}"
                    f"{', ' + result.failure_code if result.failure_code else ''})"
                )
        for written in self.files_written:
            lines.append(f"{' ' * _GUTTER}wrote {written}")
        # ⚠️ Reported ALWAYS, both ways. Servers the client has not been told about load no tools
        # and explain nothing, so the "registered" line is the one that says the install is actually
        # usable, and the skip line is the one that stops a user concluding recall is broken when
        # the client simply never heard of it.
        if self.registration is not None:
            if self.registration.recorded:
                lines.append(
                    f"{' ' * _GUTTER}registered {', '.join(self.registration.registered)} at local "
                    f"scope in {self.registration.config_path}"
                )
                lines.append(
                    f"{' ' * _GUTTER}they load in this project only and need no approval; restart "
                    f"Claude Code to pick them up"
                )
                # ⚠️ The path is NAMED because the entry is keyed by it. Moving or renaming the
                # project orphans the registration silently, which is the same shape as the memory
                # store this project has already lost to a directory rename.
                for key in self.registration.project_keys:
                    lines.append(f"{' ' * _GUTTER}  keyed to {key}")
            else:
                lines.append(
                    detail(
                        f"the MCP servers were NOT registered: {self.registration.skipped_reason}. "
                        "The corpora are built and reachable, but no client is configured to "
                        "reach them yet."
                    )
                )
            # A refused name is the one thing here a user must act on, so it is never folded into
            # the success line: their first install keeps working and their second one has no
            # servers, which looks like the second install silently failing.
            for name, where in self.registration.conflicts:
                lines.append(
                    detail(
                        f"{name} is already registered by something else ({where}) and was left "
                        f"alone. Server names are `{{project}}-{{kind}}`, so re-run with a "
                        f"different `project` in the config to give this install its own."
                    )
                )
        if self.registration is None and (self.outcomes or self.indexed):
            lines.append(
                detail(
                    "no MCP configuration was written: the config has no `project_root`, so nothing "
                    "serves these corpora yet. This command will not guess a location to put one "
                    "in. Set `project_root` and re-run; the corpora are recorded and will be reused."
                )
            )

        if self.ok:
            head = "install complete"
        elif (
            self.registration is not None
            and not self.registration.recorded
            and not (self.refused or self.failures or self.unserved)
        ):
            # Named separately because every corpus succeeded: blaming one of them would send the
            # operator to rebuild an index that is already correct.
            head = (
                "install incomplete: the corpora are built, but no client was registered, so no "
                "recall tools will appear"
            )
        elif self.unserved and not (self.refused or self.failures):
            head = (
                "install incomplete: "
                f"{', '.join(self.unserved)} has no generation serving and will answer nothing"
            )
        else:
            head = "install incomplete: a corpus was refused or failed"
        return "\n".join([head, *lines])


class _Services(Protocol):
    """The side-effecting operations, injected so the driver is testable without a database."""

    def dim(self) -> int: ...
    def apply_schema(self, dsn: str, *, dim: int) -> None: ...
    def grant(self, dsn: str, *, role: str) -> None: ...
    def run(
        self, spec: CorpusSpec, *, progress: Callable[[str], None] | None = None
    ) -> CorpusOutcome: ...
    def index_legacy(self, spec: CorpusSpec) -> LegacyIndex: ...
    def smoke(self, block: ServerBlock) -> SmokeResult: ...


@dataclass
class _RealServices:
    """The production wiring. Constructed lazily so importing this module opens no connection."""

    config: HeadlessConfig
    #: `init=False` so it is not a public constructor keyword, and `compare=False` so two services
    #: built from one config do not start equal and become unequal the moment one is used.
    _embedder: Embedder | None = field(default=None, repr=False, compare=False, init=False)

    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = resolve_embedder(self.config.embedder)
        return self._embedder

    def dim(self) -> int:
        """Asked of the wiring, so the ONE embedder it holds answers it.

        `run_headless` used to call `resolve_embedder` itself for the dimension and throw the
        instance away, so a real install built two embedders — two eagerly-loaded ONNX sessions for
        the default local model — and an injected fake still triggered a real model load, which is
        the opposite of what injecting services is for.
        """
        return self.embedder().dim

    def apply_schema(self, dsn: str, *, dim: int) -> None:
        from recall.schema import apply_migrations

        apply_migrations(dsn, dim=dim)

    def grant(self, dsn: str, *, role: str) -> None:
        """Grant the serving role its privileges, over the DDL-owner connection that can.

        No migration emits a GRANT (`recall/schema.py:148` says so and says why), and
        `serving_grants` had exactly one caller in the package: a CLI subcommand that only PRINTS
        the SQL. So a two-role install — the deployment the two DSN keys exist for — completed
        "successfully" and then failed at the first query with `permission denied`. It cannot fail
        locally or in CI, where both DSNs are the same superuser.
        """
        import psycopg

        from recall.schema import serving_grants

        with psycopg.connect(dsn) as conn, conn.transaction():
            for statement in serving_grants(role):
                conn.execute(statement)

    def index_legacy(self, spec: CorpusSpec) -> LegacyIndex:
        """Index an uncalibrated corpus into the legacy `chunks` table, under ITS OWN tenant.

        `recall.setup.index_memory_directory` exists and is deliberately not used: it hardcodes
        `DEFAULT_TENANT`, so it would write `memory`'s content into `default`, and it swallows
        every exception to print a suggestion, which is right for an interactive scaffolder and
        wrong for a driver whose report is the only thing a CI job reads.

        The directory is created when absent, because `memory_corpus`'s docstring says the wizard
        creates it and `CorpusSpec` deliberately permits an absent root on that basis. A fresh
        install therefore gets an empty, working memory tenant rather than a refusal.
        """
        from recall.index import Indexer, chunk_text
        from recall.store import DEFAULT_TABLE, PgVectorStore

        spec.root.mkdir(parents=True, exist_ok=True)
        embedder = self.embedder()
        with PgVectorStore(
            self.config.resolved_dsn, dim=embedder.dim, table=DEFAULT_TABLE, tenant=spec.tenant
        ) as store:
            store.check_schema()
            stats = Indexer(store, embedder, chunker=chunk_text).index_path(
                spec.root, glob=spec.glob
            )
        return LegacyIndex(tenant=spec.tenant, files=stats.files, chunks=stats.chunks)

    def smoke(self, block: ServerBlock) -> SmokeResult:
        """Put one real query through one configured server, exactly as that server would.

        The store is chosen from the BLOCK's own `RECALL_ENV`, mirroring `recall_mcp/server.py:629`,
        because the point is to exercise the configuration that was just written rather than a
        second guess at it. If `server_blocks` is ever wrong about which tenants can be served, this
        is where it surfaces: as an exception, from the same code path the client will take.

        The query is drawn from the tenant's OWN indexed text. A fixed question would abstain on
        every corpus and prove only that nothing raised; a phrase the corpus actually contains can
        produce a hit, so an empty result is informative instead of expected.
        """
        import psycopg

        from recall.generation_store import GenerationStore
        from recall.store import DEFAULT_TABLE, PgVectorStore
        from recall.trust import trusted_search
        from recall.trust_policy import TrustPolicy

        production = block.env.get("RECALL_ENV") == "production"
        table = "recall_chunks_v1" if production else DEFAULT_TABLE
        # Written out per branch rather than interpolated: a query built by f-string is the shape
        # that goes wrong later, even when today's value is a constant.
        statement = (
            "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s LIMIT 1"
            if production
            else "SELECT text FROM chunks WHERE tenant_id = %s LIMIT 1"
        )
        try:
            with psycopg.connect(self.config.resolved_dsn) as conn:
                conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (block.tenant,))
                row = conn.execute(statement, (block.tenant,)).fetchone()
        except Exception as exc:  # noqa: BLE001 - a failure to look is a failure, and it is reported
            return SmokeResult(
                tenant=block.tenant,
                query="",
                hits=0,
                abstained=True,
                trust_state="unknown",
                failure_code=None,
                error=scrub_dsn_secrets(f"{type(exc).__name__}: {exc}", self.config.resolved_dsn),
            )

        if not row or not str(row[0]).strip():
            return SmokeResult(
                tenant=block.tenant,
                query="",
                hits=0,
                abstained=True,
                trust_state="unknown",
                failure_code=None,
                empty=True,
            )

        # A short phrase, not the whole chunk: a chunk-length query is a different retrieval problem
        # and would not resemble anything a user types.
        query = " ".join(str(row[0]).split()[:8])
        embedder = self.embedder()
        try:
            # Annotated as the base, matching `recall_mcp/server.py`'s own `store: PgVectorStore`.
            # `GenerationStore` subclasses it, and `trusted_search` takes the base.
            store: PgVectorStore
            if production:
                store = GenerationStore(self.config.resolved_dsn, dim=embedder.dim, tenant=block.tenant)
            else:
                store = PgVectorStore(
                    self.config.resolved_dsn, dim=embedder.dim, table=table, tenant=block.tenant
                )
            with store:
                # ⚠️ **The BLOCK's trust mode, not this process's.** `trusted_search` resolves its
                # policy from `TrustPolicy.from_env()` when none is passed, which reads the WIZARD's
                # environment — and the wizard sets no `RECALL_TRUST_MODE`, so every smoke ran
                # strict regardless of what the server being smoke-tested was configured with.
                #
                # Measured on a clean install: `default-memory` is written with
                # `RECALL_TRUST_MODE=development` and its smoke reported
                # `TrustRefusal: INDEX_NOT_READY: refused in strict trust mode`, then the run
                # concluded "install incomplete". The corpus was fine and the server as configured
                # would have answered it; the check was testing a configuration nobody runs, while
                # its own message says "a query through this server's own configuration".
                result = trusted_search(
                    store, embedder, query, k=5, policy=TrustPolicy.from_env(block.env)
                )
        except Exception as exc:  # noqa: BLE001 - THIS is the failure the smoke test exists for
            return SmokeResult(
                tenant=block.tenant,
                query=query,
                hits=0,
                abstained=True,
                trust_state="unknown",
                failure_code=None,
                error=scrub_dsn_secrets(f"{type(exc).__name__}: {exc}", self.config.resolved_dsn),
            )

        return SmokeResult(
            tenant=block.tenant,
            query=query,
            hits=len(result.hits),
            abstained=result.abstained,
            trust_state=result.trust_state,
            failure_code=result.failure_code,
        )

    def run(
        self, spec: CorpusSpec, *, progress: Callable[[str], None] | None = None
    ) -> CorpusOutcome:
        return run_corpus(
            spec,
            # `development`, not the corpus's SERVING environment. A production build requires a
            # verifiable embedder identity, which the wizard's embedders do not have, and `docs`
            # is served in production; the two coexist because each MCP server gets its own env
            # block. (This comment used to blame a file:// manifest. Nothing gates file:// on the
            # environment — see `_BUILD_ENVIRONMENTS` in pipeline.py for the correction.)
            manager=GenerationManager(
                self.config.resolved_dsn, spec.tenant, actor="recall-wizard", environment="development"
            ),
            calibrations=CalibrationRepository(self.config.resolved_dsn, spec.tenant),
            embedder=self.embedder(),
            corpus_version=self.config.corpus_version,
            project=self.config.project,
            progress=progress,
        )


def _database_identity(dsn: str) -> tuple[str, int | None, str]:
    """Host, port and database name — what decides whether two DSNs reach the same database."""
    parts = urlsplit(dsn)
    return (parts.hostname or "", parts.port, parts.path)


def load_config(path: str | Path) -> HeadlessConfig:
    """Read and validate the config, naming the key at fault rather than the line number."""
    location = Path(path)
    try:
        raw = json.loads(location.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        hint = ""
        if "Invalid \\escape" in str(exc):
            # The commonest way a Windows operator writes this file, and the message names neither
            # backslashes nor paths.
            hint = (
                " On Windows write paths with doubled backslashes (C:\\\\corpus\\\\docs) or with "
                "forward slashes (C:/corpus/docs)."
            )
        raise ConfigRefusal(f"cannot read the wizard config {location}: {exc}.{hint}") from exc
    except OSError as exc:
        raise ConfigRefusal(f"cannot read the wizard config {location}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigRefusal(
            f"the wizard config {location} must be a JSON object, not {type(raw).__name__}"
        )

    # A typo in a key name was silently discarded, so `"tabel"` looked like it had been applied.
    unknown = tuple(sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL)))
    if unknown:
        raise ConfigRefusal(
            f"the wizard config {location} has unknown keys: {', '.join(unknown)}. "
            f"Known keys are {', '.join((*_REQUIRED, *_OPTIONAL))}.",
            unknown,
        )

    # Type-checked, not stringified. `str(raw.get(key, ""))` read `null` as "None", `0` as "0" and
    # `false` as "False", all non-empty, so every non-string JSON value passed this check and was
    # then coerced into that garbage — or reached `Path()` and raised a bare `TypeError` out of the
    # very function whose docstring promises to name the key at fault.
    missing = tuple(
        key for key in _REQUIRED if not isinstance(raw.get(key), str) or not raw[key].strip()
    )
    if missing:
        raise ConfigRefusal(
            f"the wizard config {location} is missing, or has a non-string value for, "
            f"{', '.join(missing)}. Every one of {', '.join(_REQUIRED)} is required as a "
            "non-empty string; none has a default the wizard could guess without being wrong "
            "somewhere.",
            missing,
        )

    for key in _ROOT_KEYS:
        # Checked here as well as in `CorpusSpec`, because that refusal names the PATH and this
        # reader needs to be told which config KEY holds it.
        if not Path(raw[key]).is_absolute():
            raise ConfigRefusal(
                f"{key} must be an absolute path on this platform, not {raw[key]!r}: {RELATIVE_ROOT}",
                (key,),
            )

    for key in _OPTIONAL:
        if key in raw and raw[key] is not None and not isinstance(raw[key], str):
            raise ConfigRefusal(
                f"{key} must be a string or absent, not {type(raw[key]).__name__}", (key,)
            )

    # Checked with the same rule as the required roots, and for the same reason: a relative path
    # here resolves against the wizard's own working directory, so the file would land somewhere
    # that depends on where the command happened to be run from.
    for optional_root, consequence in (
        ("project_root", "the MCP configuration would land somewhere unpredictable"),
        ("data_root", "the user's index would land somewhere they did not choose"),
    ):
        if raw.get(optional_root) and not Path(raw[optional_root]).is_absolute():
            raise ConfigRefusal(
                f"{optional_root} must be an absolute path on this platform, not "
                f"{raw[optional_root]!r}: {consequence}.",
                (optional_root,),
            )

    # `project_root` is the one root the wizard WRITES into, and it is checked HERE because the
    # writing happens last. Every corpus is built, calibrated, promoted and registered before
    # `write_project_files` is reached, so a bad value discovered there costs the whole install:
    # measured in CI, thirty minutes of work followed by `could not write .../project/.env: No such
    # file or directory` and `install incomplete`. The cheap check has to run before the expensive
    # work, not after it.
    #
    # Two conditions, deliberately answered differently, because they are different mistakes:
    #
    # * **The path exists and is not a directory.** Refused by name, like every other root, and the
    #   module docstring already calls this out as one of the three likeliest first-run conditions.
    #
    # * **The path is absent and so is its PARENT.** That is a typo, not an intent. Somebody who
    #   means `C:/Users/me/myapp` does not have `C:/Users/me` missing, so creating the whole tree
    #   would manufacture directories nobody named. A missing LEAF is the ordinary first install
    #   and is created by `write_project_files`, which is why only the parent is required here.
    project_root_raw = raw.get("project_root")
    if project_root_raw:
        candidate = Path(project_root_raw)
        if candidate.exists():
            if not candidate.is_dir():
                raise ConfigRefusal(
                    f"project_root {project_root_raw!r} exists and is not a directory. The wizard "
                    "writes .env, a CLAUDE.md block and memory/MEMORY.md inside it, and none of "
                    "those can be written underneath a file.",
                    ("project_root",),
                )
        elif not candidate.parent.is_dir():
            raise ConfigRefusal(
                f"project_root {project_root_raw!r} does not exist and neither does its parent "
                f"{str(candidate.parent)!r}. The wizard creates the last directory of a project "
                "root, because that is an ordinary first install; it does not create the tree "
                "above it, because a missing parent is a mistyped path rather than a place you "
                "meant to put a project. Create the parent, or correct the path.",
                ("project_root",),
            )

    # `dsn` and `data_root` are ALTERNATIVES, and the refusal names both keys.
    #
    # Accepting both would mean silently ignoring one, which is the exact defect already fixed
    # twice on this branch: a setting taken and discarded looks applied and is not. Accepting
    # neither would leave the driver with nowhere to write, discovered as a connection error.
    has_dsn = bool(raw.get("dsn"))
    has_data_root = bool(raw.get("data_root"))
    if has_dsn and has_data_root:
        raise ConfigRefusal(
            "dsn and data_root cannot both be set. `dsn` uses a database somebody else "
            "provisioned, which is how CI drives this; `data_root` provisions one at a location "
            "the user chose, which is what an install does. Setting both means one is ignored, and "
            "you would not be able to tell which.",
            ("dsn", "data_root"),
        )
    if not has_dsn and not has_data_root:
        raise ConfigRefusal(
            "one of dsn or data_root is required: either point at an existing database, or say "
            "where the wizard should create one.",
            ("dsn", "data_root"),
        )
    if has_data_root and raw.get("migration_dsn"):
        raise ConfigRefusal(
            "migration_dsn cannot be set alongside data_root: the wizard provisions the database "
            "and owns both credentials, so a DDL address supplied here would be ignored.",
            ("migration_dsn", "data_root"),
        )

    return HeadlessConfig(
        embedder=raw["embedder"],
        corpus_version=raw["corpus_version"],
        docs_root=Path(raw["docs_root"]),
        code_root=Path(raw["code_root"]),
        memory_root=Path(raw["memory_root"]),
        dsn=raw.get("dsn") or None,
        # Defaulted to `dsn`, not left None. The ordinary install has one role, and requiring the
        # same string twice is a way to get them different by typo.
        migration_dsn=raw.get("migration_dsn") or raw.get("dsn") or None,
        data_root=Path(raw["data_root"]) if raw.get("data_root") else None,
        project=raw.get("project") or DEFAULT_PROJECT,
        serving_role=raw.get("serving_role") or None,
        project_root=Path(raw["project_root"]) if raw.get("project_root") else None,
    )


def _prepare(config: HeadlessConfig, wiring: _Services) -> None:
    """Everything that must succeed before the first corpus is built, refusing by name.

    Split out so every failure between reading the config and starting work becomes a
    `PipelineRefusal` rather than a traceback from a module the operator has never heard of. The
    DSN is redacted in every message: a connection failure is exactly when an operator wants the
    DSN in the log, and exactly when printing it verbatim writes the password to disk. Measured,
    a password containing `%` reached the terminal verbatim inside `invalid percent-encoded token`.
    """
    if _database_identity(config.resolved_dsn) != _database_identity(config.resolved_migration_dsn):
        raise ConfigRefusal(
            f"dsn and migration_dsn name different databases ({redacted_dsn(config.resolved_dsn)} and "
            f"{redacted_dsn(config.resolved_migration_dsn)}). The schema would be applied to one and every "
            "chunk written to the other, whose vector width was never checked — discovered after "
            "a full build, at calibration.",
            ("dsn", "migration_dsn"),
        )

    two_role = urlsplit(config.resolved_dsn).username != urlsplit(config.resolved_migration_dsn).username
    if two_role and not config.serving_role:
        raise ConfigRefusal(
            "dsn and migration_dsn authenticate as different roles, which is the two-role "
            "deployment, but serving_role is absent. No migration emits a GRANT, so the serving "
            "role would own no privileges on the tables just created and every query would fail "
            "with permission denied. Set serving_role to the role dsn authenticates as, or run "
            "`recall schema grants --role <role>` yourself.",
            ("serving_role",),
        )

    try:
        dim = wiring.dim()
    except PipelineRefusal:
        raise
    except Exception as exc:
        raise ConfigRefusal(
            f"embedder {config.embedder!r} could not be resolved: {type(exc).__name__}: {exc}",
            ("embedder",),
        ) from exc

    try:
        wiring.apply_schema(config.resolved_migration_dsn, dim=dim)
    except PipelineRefusal:
        raise
    except Exception as exc:
        raise ConfigRefusal(
            scrub_dsn_secrets(
                f"cannot prepare the schema at {redacted_dsn(config.resolved_migration_dsn)} for embedder "
                f"{config.embedder!r} (vector width {dim}): {type(exc).__name__}: {exc}",
                config.resolved_migration_dsn,
                config.resolved_dsn,
            ),
            ("migration_dsn",),
        ) from exc

    if config.serving_role:
        try:
            wiring.grant(config.resolved_migration_dsn, role=config.serving_role)
        except PipelineRefusal:
            raise
        except Exception as exc:
            raise ConfigRefusal(
                scrub_dsn_secrets(
                    f"cannot grant the serving role {config.serving_role!r} at "
                    f"{redacted_dsn(config.resolved_migration_dsn)}: {type(exc).__name__}: {exc}",
                    config.resolved_migration_dsn,
                    config.resolved_dsn,
                ),
                ("serving_role",),
            ) from exc


#: Re-exported from `stack`, which owns it now. Two definitions of this name is what made
#: "+ Add project" refuse on every real install; see `recall.wizard.stack.COMPOSE_NAME`.
COMPOSE_NAME = _COMPOSE_NAME


def compose_project_for(data_root: Path, project: str) -> str:
    """A Compose project name unique to THIS install.

    ⚠️ A fixed name does not work, and the first real end-to-end run proved it. Compose namespaces
    containers by project, so a constant `recall-desktop` collides with anything else using that
    name: the run failed with `Found orphan containers (recall-desktop-recall-docs-1, ...)` against
    services left by the SHIPPED `docker-compose.desktop.yml`, and it would collide the same way
    with a second install on the same machine. Two installs would then fight over one namespace and
    each would see the other's containers as orphans to remove.

    Keyed on the LOCATION as well as the project, because "myproject" installed in two places is two
    installs, and the location is what the user actually chose.
    """
    import hashlib

    digest = hashlib.sha256(data_root.as_posix().encode("utf-8")).hexdigest()[:8]
    # Compose restricts project names to lowercase alphanumerics, hyphen and underscore.
    scope = "".join(char for char in project.lower() if char.isalnum() or char in "-_") or "recall"
    return f"recall-{scope}-{digest}"


def _provisional_env(tenants: tuple[str, ...], *, dsn: str, embedder: str) -> dict[str, dict[str, str]]:
    """Enough environment to WRITE the stack before the corpora have been driven.

    The real trust posture per tenant is not knowable yet: it depends on what certified and what was
    promoted, which is the whole output of the run. So the first compose is written with development
    trust and rewritten at the wiring stage from `server_blocks`, which is the single source for it.
    Only `db` is started from this first version, so nothing ever runs against the provisional value.
    """
    return {
        tenant: {
            "RECALL_DSN": dsn,
            "RECALL_EMBEDDER": embedder,
            "RECALL_TENANT": tenant,
            "RECALL_TRUST_MODE": "development",
        }
        for tenant in tenants
    }


def _provision(
    config: HeadlessConfig,
    plan: CorpusPlan,
    progress: Callable[[str], None] | None,
) -> HeadlessConfig:
    """Create the database at the user's chosen location and return a config that can reach it.

    Only when `data_root` is set. With a `dsn` the caller already has a database and this does
    nothing, which is how CI drives the rest of the pipeline without Docker.

    The port is READ BACK from an existing compose file when there is one. It must be stable across
    runs: `runtime.json` names this file and the UI connects through whatever it publishes, so
    re-choosing would repoint the store out from under the UI and the agent at once.
    """
    if config.data_root is None:
        return config

    compose_path = config.data_root / COMPOSE_NAME
    compose_project = compose_project_for(config.data_root, config.project)
    port = existing_port(compose_path) or choose_port()
    dsn = host_dsn(port)
    tenants = tuple(spec.tenant for spec in plan.corpora)

    if progress:
        # Says where the stack is CONFIGURED, not where the database file sits, because those are
        # no longer the same place: the database lives on a Docker named volume rather than under
        # `data_root`. See `compose_document` for the measured reason. Naming the data root here as
        # if it held the database would be a claim the code stopped creating.
        progress(f"stack: {config.data_root}, database on volume, port {port}")
    write_compose(
        compose_path,
        compose_document(
            StackSpec(
                data_root=config.data_root,
                port=port,
                tenants=tenants,
                env=_provisional_env(tenants, dsn=dsn, embedder=config.embedder),
                project_name=compose_project,
            )
        ),
    )
    # `db` only. The recall image is built later and the store must not wait for it.
    try:
        bring_up(compose_path, project_name=compose_project, services=("db",))
    except RuntimeError as exc:
        raise ConfigRefusal(
            f"could not start the database for {config.data_root}: {exc}. Check that Docker "
            "Desktop is running.",
            ("data_root",),
        ) from exc
    # ⚠️ `--wait` is NOT enough: the healthcheck is `pg_isready` inside the container, which the
    # temporary server the postgres entrypoint runs during initdb also answers. Poll the address
    # the caller will actually use.
    try:
        wait_for_database(dsn)
    except RuntimeError as exc:
        raise ConfigRefusal(f"the database started but never accepted a connection: {exc}",
                            ("data_root",)) from exc
    if progress:
        progress("database: ready")

    return replace(config, dsn=dsn, migration_dsn=dsn)


def _step_reporter(
    tenant: str, progress: Callable[[str], None] | None
) -> Callable[[str], None] | None:
    """Prefix each build step with the corpus it belongs to.

    A named factory rather than a lambda inside the loop, because binding the loop variable in a
    closure would report every corpus's steps under the last tenant's name, and that is a bug the
    output would not obviously reveal: the steps are identical for every corpus.
    """
    if progress is None:
        return None
    return lambda message: progress(f"{tenant}: {message}")


def run_headless(
    config: HeadlessConfig,
    *,
    services: _Services | None = None,
    progress: Callable[[str], None] | None = None,
    state_path: Path | None = None,
    fresh: bool = False,
    profile_path: Path | None = None,
    #: Claude Code's own config, where the local-scope servers are recorded.
    #: ⚠️ Threaded rather than defaulted deep inside, for the same reason `profile_path` is: the
    #: default is a USER-GLOBAL file, and the first test run that reached it wrote five junk
    #: entries into the real one. A caller that does not want to touch the user's client — every
    #: test — passes a path of its own.
    claude_config_path: Path | None = None,
) -> HeadlessReport:
    """Apply the schema, then drive every calibrated corpus, reporting rather than aborting.

    `state_path` makes the run resumable: a corpus a previous run promoted under the same
    configuration is reused instead of rebuilt, and the state is written after EVERY corpus so a run
    that dies part-way keeps what it finished. `fresh=True` ignores any recorded state. With
    `state_path` unset the behaviour is exactly as before, which keeps every existing caller and
    test honest about what they are exercising.
    """
    try:
        plan = default_plan(
            embedder=config.embedder,
            docs_root=config.docs_root,
            code_root=config.code_root,
            memory_root=config.memory_root,
            project=config.project,
        )
    except PipelineRefusal:
        raise
    except (ValueError, TypeError) as exc:
        # `CorpusSpec.__post_init__` and `default_plan` raise plain value errors that name the
        # PATH, and the operator needs to be told which config key holds it.
        raise ConfigRefusal(f"the corpus plan is not buildable from this config: {exc}") from exc

    # The database, at the user's chosen location, before anything tries to reach one.
    config = _provision(config, plan, progress)

    wiring = services if services is not None else _RealServices(config)
    _prepare(config, wiring)

    # Resumable state, keyed on a digest of the config fields that would invalidate finished work.
    # `WizardState(digest=...)` with no recorded corpora is the no-state case, so the loops below do
    # not branch on whether resuming is enabled.
    digest = config_digest(config)
    state = (
        WizardState(digest=digest)
        if fresh or state_path is None
        else load_state(state_path, digest=digest)
    )

    def _remember() -> None:
        """Written after EVERY corpus, not at the end. The end is what a crashed run never reaches.

        A write failure is reported and swallowed. The state file is an OPTIMISATION: losing it
        costs a rebuild, while raising here would throw away a completed corpus because a directory
        was read-only, which is the more expensive of the two by a wide margin. The default path
        sits beside the operator's config, which they may well not own.
        """
        if state_path is None:
            return
        try:
            save_state(state_path, state)
        except OSError as exc:
            if progress:
                progress(
                    f"note: could not write {state_path} ({exc.strerror or exc}); this run cannot "
                    "be resumed, but nothing already built is lost"
                )

    outcomes: list[CorpusOutcome] = []
    refused: list[Refusal] = []
    failures: list[Failure] = []
    reused: list[str] = []
    for spec in plan.calibrated:
        if (recorded := state.outcome_for(spec.tenant)) is not None:
            # Only a PROMOTED outcome is recorded, so this cannot silently reuse a degraded corpus:
            # that one was left unpromoted precisely so re-running would retry it.
            outcomes.append(recorded)
            reused.append(spec.tenant)
            if progress:
                progress(f"{spec.tenant}: reused from a previous run")
            continue
        try:
            outcome = wiring.run(spec, progress=_step_reporter(spec.tenant, progress))
            outcomes.append(outcome)
            state = state.with_outcome(outcome)
            _remember()
        except PipelineRefusal as exc:
            # Reported, not raised. The remaining corpora are independent and a user with one
            # working corpus is better off than a user with a traceback.
            refused.append(
                Refusal(tenant=spec.tenant, reason=scrub_dsn_secrets(str(exc), config.resolved_dsn, config.resolved_migration_dsn))
            )
        except Exception as exc:  # noqa: BLE001 - see below
            # Everything else, for the same reason and one more: `promote` is irreversible and
            # retires whatever the tenant was serving, so losing the report loses the only record
            # of which generations are now active. `Exception`, not `BaseException`, so
            # KeyboardInterrupt and SystemExit still stop the run.
            #
            # Scrubbed: this is the path a psycopg error takes, and a psycopg error can quote the
            # password back at you.
            failures.append(
                Failure(
                    tenant=spec.tenant,
                    error=scrub_dsn_secrets(
                        f"{type(exc).__name__}: {exc}", config.resolved_dsn, config.resolved_migration_dsn
                    ),
                )
            )

    # The uncalibrated corpora. Driven, not skipped: `memory` is a third of the install, and a
    # report that says "skipped" for it describes an install nobody finished.
    indexed: list[LegacyIndex] = []
    for spec in plan.corpora:
        if spec.calibrated:
            continue
        if (previous := state.indexed_for(spec.tenant)) is not None:
            indexed.append(LegacyIndex(tenant=previous[0], files=previous[1], chunks=previous[2]))
            reused.append(spec.tenant)
            if progress:
                progress(f"{spec.tenant}: reused from a previous run")
            continue
        if progress:
            progress(f"{spec.tenant}: index")
        try:
            entry = wiring.index_legacy(spec)
            indexed.append(entry)
            state = state.with_indexed(entry)
            _remember()
        except PipelineRefusal as exc:
            refused.append(
                Refusal(
                    tenant=spec.tenant,
                    reason=scrub_dsn_secrets(str(exc), config.resolved_dsn, config.resolved_migration_dsn),
                )
            )
        except Exception as exc:  # noqa: BLE001 - same reasoning as the calibrated loop above
            failures.append(
                Failure(
                    tenant=spec.tenant,
                    error=scrub_dsn_secrets(
                        f"{type(exc).__name__}: {exc}", config.resolved_dsn, config.resolved_migration_dsn
                    ),
                )
            )

    # The wiring. Written from what ACTUALLY happened, not from the plan: a tenant with nothing
    # promoted and nothing serving gets no server, because a production server over it would raise
    # `NoActiveGeneration` on every query rather than refuse with a failure code.
    promoted = frozenset(o.tenant for o in outcomes if o.promoted)
    serving = promoted | frozenset(
        o.tenant for o in outcomes if o.previously_serving
    ) | frozenset(i.tenant for i in indexed)
    blocks, unservable = server_blocks(
        plan, dsn=config.resolved_dsn, promoted=promoted, serving=serving
    )
    files: tuple[Path, ...] = ()

    # Rewrite the stack with the REAL trust posture, now that it is known, and hand the desktop UI
    # its profile. The first compose was written with development trust because certification had
    # not happened yet; only `db` was ever started from it, so nothing ran against the provisional
    # value. `server_blocks` stays the single source, so the agent's server and the UI's service
    # cannot end up on different rules for one corpus.
    if config.data_root is not None and blocks:
        compose_path = config.data_root / COMPOSE_NAME
        compose_project = compose_project_for(config.data_root, config.project)
        try:
            port = existing_port(compose_path) or choose_port()
            write_compose(
                compose_path,
                compose_document(
                    StackSpec(
                        data_root=config.data_root,
                        port=port,
                        tenants=tuple(block.tenant for block in blocks),
                        env={block.tenant: dict(block.env) for block in blocks},
                        project_name=compose_project,
                    )
                ),
            )
            profile_written = write_runtime_profile(
                compose_path=compose_path,
                project=config.project,
                compose_project=compose_project,
                path=profile_path,
            )
            files = (*files, compose_path, profile_written)
            if progress:
                progress(f"desktop: {profile_written}")
        except OSError as exc:
            # Reported, not raised. The corpora are built and promoted by now; losing that because
            # a profile could not be written would be the expensive half of the trade. The install
            # works from the CLI and the agent either way; only the UI handoff is missing.
            failures.append(
                Failure(
                    tenant="desktop",
                    error=f"could not write {exc.filename or 'the desktop profile'}: "
                    f"{exc.strerror or exc}",
                )
            )

    # Bound before the branch, not inside the `try`: the failure path and the no-`project_root`
    # path both reach the report, and an unbound name there would turn a written install into a
    # NameError at the moment it reports success.
    registration: LocalScopeRegistration | None = None
    if config.project_root is not None:
        if progress:
            progress("wiring: MCP servers (local scope)")
        # ⚠️ **LOCAL scope, and deliberately no `.mcp.json`.** Project-scoped servers are gated: the
        # client asks for approval in an interactive session and the tools are silently absent
        # until it is answered — measured, 2 of 310 tracked projects had any approval recorded.
        # This installer exists for people who are not Claude Code experts, so an invisible gate
        # between "the installer said it worked" and "the tools are there" is the product failing.
        #
        # Writing BOTH would be worse than either. Precedence is local, then project, then user,
        # ordered HIGHEST FIRST, and entries are not merged, so the local entry written here already
        # beats a `.mcp.json` of the same name: that file would be inert weight that still has to be
        # trusted, explained and kept in step, and whoever later found the tools missing would delete
        # it and see nothing change.
        #
        # 🔁 This comment said the opposite until the local-scope switch was audited. It was written
        # for USER scope, where a project file genuinely does win (2 beats 3), and stayed behind when
        # the code moved to local (1). Correct sentence, wrong scope, and nothing failed.
        try:
            registration = register_local_scope(
                blocks,
                project_root=config.project_root,
                config_path=claude_config_path,
            )
            wrote = write_project_files(
                project_root=config.project_root,
                dsn=config.resolved_dsn,
                embedder=config.embedder,
                memory_dir=config.memory_root,
            )
            files = tuple(wrote)
        except OSError as exc:
            # Reported, not raised. Every corpus is already built and promoted by now; throwing that
            # away because a directory is read-only would be the expensive half of the trade.
            #
            # The FAILING FILE is named from the exception rather than assumed. This block covers
            # `~/.claude.json`, `.env`, `CLAUDE.md` and `MEMORY.md`, and an earlier version of
            # this message blamed one of them for all four, which would have sent an operator to
            # check a file that was written correctly.
            blamed = exc.filename or "the project files"
            failures.append(
                Failure(
                    tenant="wiring",
                    error=f"could not write {blamed}: {exc.strerror or exc}",
                )
            )
    else:
        blocks = ()

    # The verification half. Run for every block the wiring produced, and deliberately NOT gated on
    # the registration: whether a client has been told about a server is a different question from
    # whether the corpus behind it answers, and the moment registration is refused is exactly when
    # the operator most needs to know which half failed.
    smoke: tuple[SmokeResult, ...] = ()
    if blocks:
        results: list[SmokeResult] = []
        for block in blocks:
            if progress:
                progress(f"{block.tenant}: smoke")
            try:
                results.append(wiring.smoke(block))
            except Exception as exc:  # noqa: BLE001 - a smoke test must never be the thing that
                # aborts an install. Its whole job is to report, and an unexpected failure inside
                # it is still a report about this server rather than a reason to lose the run.
                results.append(
                    SmokeResult(
                        tenant=block.tenant,
                        query="",
                        hits=0,
                        abstained=True,
                        trust_state="unknown",
                        failure_code=None,
                        error=scrub_dsn_secrets(
                            f"{type(exc).__name__}: {exc}", config.resolved_dsn, config.resolved_migration_dsn
                        ),
                    )
                )
        smoke = tuple(results)

    return HeadlessReport(
        outcomes=tuple(outcomes),
        refused=tuple(refused),
        failures=tuple(failures),
        indexed=tuple(indexed),
        reused=tuple(reused),
        servers=blocks,
        registration=registration,
        unservable=unservable,
        files_written=files,
        smoke=smoke,
    )
