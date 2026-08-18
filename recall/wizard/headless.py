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
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from recall.calibration_v2 import CalibrationRepository
from recall.embeddings import Embedder, resolve_embedder
from recall.generations import GenerationManager
from recall.store import redacted_dsn
from recall.wizard.corpora import DEFAULT_PROJECT, RELATIVE_ROOT, CorpusSpec, default_plan
from recall.wizard.pipeline import CorpusOutcome, PipelineRefusal, run_corpus
from recall.wizard.state import WizardState, config_digest, load_state, save_state
from recall.wizard.wiring import (
    ServerBlock,
    SmokeResult,
    UnservableTenant,
    mcp_config,
    server_blocks,
    write_mcp_config,
    write_project_files,
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
    #: Optional: where `.mcp.json` is written, and the `cwd` each server is launched from.
    #: Optional rather than derived from the config's own directory, because writing an MCP
    #: configuration into a guessed location is a side effect an operator cannot predict, and a
    #: config may well live somewhere temporary. When absent the report says the wiring was skipped
    #: and names the key, which is better than putting files where nobody will look for them.
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
    #: MCP servers written to `.mcp.json`, and the tenants deliberately left without one.
    servers: tuple[ServerBlock, ...] = ()
    unservable: tuple[UnservableTenant, ...] = ()
    #: One query put through each configured server. This is the difference between "a config was
    #: written" and "the install answers", and it checks `server_blocks`'s reasoning against the
    #: database rather than trusting it.
    smoke: tuple[SmokeResult, ...] = ()
    #: Where `.mcp.json` was written, or None when `project_root` was absent from the config.
    mcp_path: Path | None = None
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
        """
        raised = any(s.error for s in self.smoke)
        return not (self.refused or self.failures or self.unserved or raised)

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
                # Says what is true of the DATABASE, and stops there. An earlier version added that
                # this tenant's server "carries RECALL_TRUST_MODE=development", which the wizard
                # does not do: writing `.mcp.json` is Phase 4 and does not exist yet. The corpus
                # spec requires that setting; nothing here has applied it.
                lines.append(
                    detail(
                        "the directory was empty, which is normal on a first install: this tenant "
                        "is writable and fills up as you use it. It is not calibrated, so it needs "
                        "a server configured for development trust; this command does not write "
                        "that configuration."
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
        if self.mcp_path is None and (self.outcomes or self.indexed):
            lines.append(
                detail(
                    "no MCP configuration was written: the config has no `project_root`, so nothing "
                    "serves these corpora yet. This command will not guess a location to put one "
                    "in. Set `project_root` and re-run; the corpora are recorded and will be reused."
                )
            )

        if self.ok:
            head = "install complete"
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

        The store is chosen from the BLOCK's own `RECALL_ENV`, mirroring `recall_mcp/server.py:627`,
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
                error=_scrub(f"{type(exc).__name__}: {exc}", self.config.resolved_dsn),
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
                result = trusted_search(store, embedder, query, k=5)
        except Exception as exc:  # noqa: BLE001 - THIS is the failure the smoke test exists for
            return SmokeResult(
                tenant=block.tenant,
                query=query,
                hits=0,
                abstained=True,
                trust_state="unknown",
                failure_code=None,
                error=_scrub(f"{type(exc).__name__}: {exc}", self.config.resolved_dsn),
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


def _scrub(text: str, *dsns: str) -> str:
    """Remove any of these DSNs' passwords from `text`.

    `redacted_dsn` is not enough on its own, and assuming it was left the leak open. The password
    can be inside the EXCEPTION rather than inside a DSN we format: measured, a password containing
    `%` produced `psycopg.ProgrammingError: invalid percent-encoded token: "S3cr%tPw"`, so wrapping
    the error with a redacted DSN beside it still printed the secret verbatim. Anything derived
    from a connection attempt goes through here.
    """
    for dsn in dsns:
        password = urlsplit(dsn).password
        if password:
            text = text.replace(password, "***")
    return text


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
            _scrub(
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
                _scrub(
                    f"cannot grant the serving role {config.serving_role!r} at "
                    f"{redacted_dsn(config.resolved_migration_dsn)}: {type(exc).__name__}: {exc}",
                    config.resolved_migration_dsn,
                    config.resolved_dsn,
                ),
                ("serving_role",),
            ) from exc


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
                Refusal(tenant=spec.tenant, reason=_scrub(str(exc), config.resolved_dsn, config.resolved_migration_dsn))
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
                    error=_scrub(
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
                    reason=_scrub(str(exc), config.resolved_dsn, config.resolved_migration_dsn),
                )
            )
        except Exception as exc:  # noqa: BLE001 - same reasoning as the calibrated loop above
            failures.append(
                Failure(
                    tenant=spec.tenant,
                    error=_scrub(
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
    mcp_path: Path | None = None
    files: tuple[Path, ...] = ()
    if config.project_root is not None:
        if progress:
            progress("wiring: .mcp.json")
        mcp_path = config.project_root / ".mcp.json"
        try:
            write_mcp_config(mcp_path, mcp_config(blocks, project_root=config.project_root))
            wrote = write_project_files(
                project_root=config.project_root,
                dsn=config.resolved_dsn,
                embedder=config.embedder,
                memory_dir=config.memory_root,
            )
            files = (mcp_path, *wrote)
        except OSError as exc:
            # Reported, not raised. Every corpus is already built and promoted by now; throwing that
            # away because a directory is read-only would be the expensive half of the trade.
            #
            # The FAILING FILE is named from the exception rather than assumed. This block covers
            # `.mcp.json`, `.env`, `CLAUDE.md` and `MEMORY.md`, and an earlier version of this
            # message blamed `.mcp.json` for all four, which would have sent an operator to check a
            # file that was written correctly.
            mcp_path = None
            blamed = exc.filename or "the project files"
            failures.append(
                Failure(
                    tenant="wiring",
                    error=f"could not write {blamed}: {exc.strerror or exc}",
                )
            )
    else:
        blocks = ()

    # The verification half. Run against the blocks that were WRITTEN, so a config nobody wrote is
    # not credited with answering, and after the wiring so it exercises the file on disk's semantics.
    smoke: tuple[SmokeResult, ...] = ()
    if blocks and mcp_path is not None:
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
                        error=_scrub(
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
        unservable=unservable,
        mcp_path=mcp_path,
        files_written=files,
        smoke=smoke,
    )
