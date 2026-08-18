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
from recall.wizard.corpora import RELATIVE_ROOT, CorpusSpec, default_plan
from recall.wizard.pipeline import CorpusOutcome, PipelineRefusal, run_corpus

__all__ = [
    "ConfigRefusal",
    "Failure",
    "HeadlessConfig",
    "HeadlessReport",
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
    """Everything the driver needs. No REQUIRED field has a default it could guess wrong."""

    dsn: str
    migration_dsn: str
    embedder: str
    corpus_version: str
    docs_root: Path
    code_root: Path
    memory_root: Path
    #: Optional: a `BuildRequest` label stamped on every chunk, not something the pipeline needs.
    project: str | None = None
    #: Optional: the role the SERVING dsn authenticates as, when it differs from the DDL owner.
    #: No migration emits a GRANT, because the role name is a deployment decision the packaged SQL
    #: cannot know, so a two-role install has to be told the name or it will not work.
    serving_role: str | None = None


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
    #: Corpora the driver deliberately did not drive. `memory` has no generation, so handing it to
    #: `run_corpus` would produce a promoted generation nothing can calibrate.
    skipped: tuple[str, ...] = ()

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
        """
        return not (self.refused or self.failures or self.unserved)

    @property
    def degraded(self) -> tuple[str, ...]:
        return tuple(o.tenant for o in self.outcomes if not o.certified)

    def render(self) -> str:
        """The text a headless caller reads. Every corpus in the plan appears, including skips."""
        tenants = [
            *(o.tenant for o in self.outcomes),
            *(r.tenant for r in self.refused),
            *(f.tenant for f in self.failures),
            *self.skipped,
        ]
        width = max([_TENANT_COLUMN, *(len(t) for t in tenants)]) if tenants else _TENANT_COLUMN
        indent = " " * (_GUTTER + width + 1)

        def detail(text: str) -> str:
            """Wrapped and re-indented, so a multi-sentence reason keeps the column it started in."""
            return textwrap.indent(textwrap.fill(text, 96), indent)

        lines: list[str] = []
        for outcome in self.outcomes:
            head = f"{' ' * _GUTTER}{outcome.tenant:<{width}} "
            if outcome.certified:
                lines.append(
                    f"{head}certified and promoted  "
                    f"({outcome.answerable} answerable / {outcome.unanswerable} unanswerable, "
                    f"{outcome.generation_id})"
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
        for tenant in self.skipped:
            lines.append(f"{' ' * _GUTTER}{tenant:<{width}} skipped")
            # What is TRUE, not what a reader might assume. This line used to say the corpus was
            # "indexed into the legacy chunks table", asserting a database state the wizard never
            # creates: nothing indexes it and nothing creates its directory.
            lines.append(
                detail(
                    "not indexed by this command and not calibrated. Index it yourself with "
                    f"`recall index --tenant {tenant} <root>`."
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
                self.config.dsn, spec.tenant, actor="recall-wizard", environment="development"
            ),
            calibrations=CalibrationRepository(self.config.dsn, spec.tenant),
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

    return HeadlessConfig(
        dsn=raw["dsn"],
        migration_dsn=raw["migration_dsn"],
        embedder=raw["embedder"],
        corpus_version=raw["corpus_version"],
        docs_root=Path(raw["docs_root"]),
        code_root=Path(raw["code_root"]),
        memory_root=Path(raw["memory_root"]),
        project=raw.get("project") or None,
        serving_role=raw.get("serving_role") or None,
    )


def _prepare(config: HeadlessConfig, wiring: _Services) -> None:
    """Everything that must succeed before the first corpus is built, refusing by name.

    Split out so every failure between reading the config and starting work becomes a
    `PipelineRefusal` rather than a traceback from a module the operator has never heard of. The
    DSN is redacted in every message: a connection failure is exactly when an operator wants the
    DSN in the log, and exactly when printing it verbatim writes the password to disk. Measured,
    a password containing `%` reached the terminal verbatim inside `invalid percent-encoded token`.
    """
    if _database_identity(config.dsn) != _database_identity(config.migration_dsn):
        raise ConfigRefusal(
            f"dsn and migration_dsn name different databases ({redacted_dsn(config.dsn)} and "
            f"{redacted_dsn(config.migration_dsn)}). The schema would be applied to one and every "
            "chunk written to the other, whose vector width was never checked — discovered after "
            "a full build, at calibration.",
            ("dsn", "migration_dsn"),
        )

    two_role = urlsplit(config.dsn).username != urlsplit(config.migration_dsn).username
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
        wiring.apply_schema(config.migration_dsn, dim=dim)
    except PipelineRefusal:
        raise
    except Exception as exc:
        raise ConfigRefusal(
            _scrub(
                f"cannot prepare the schema at {redacted_dsn(config.migration_dsn)} for embedder "
                f"{config.embedder!r} (vector width {dim}): {type(exc).__name__}: {exc}",
                config.migration_dsn,
                config.dsn,
            ),
            ("migration_dsn",),
        ) from exc

    if config.serving_role:
        try:
            wiring.grant(config.migration_dsn, role=config.serving_role)
        except PipelineRefusal:
            raise
        except Exception as exc:
            raise ConfigRefusal(
                _scrub(
                    f"cannot grant the serving role {config.serving_role!r} at "
                    f"{redacted_dsn(config.migration_dsn)}: {type(exc).__name__}: {exc}",
                    config.migration_dsn,
                    config.dsn,
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
) -> HeadlessReport:
    """Apply the schema, then drive every calibrated corpus, reporting rather than aborting."""
    try:
        plan = default_plan(
            embedder=config.embedder,
            docs_root=config.docs_root,
            code_root=config.code_root,
            memory_root=config.memory_root,
        )
    except PipelineRefusal:
        raise
    except (ValueError, TypeError) as exc:
        # `CorpusSpec.__post_init__` and `default_plan` raise plain value errors that name the
        # PATH, and the operator needs to be told which config key holds it.
        raise ConfigRefusal(f"the corpus plan is not buildable from this config: {exc}") from exc

    wiring = services if services is not None else _RealServices(config)
    _prepare(config, wiring)

    outcomes: list[CorpusOutcome] = []
    refused: list[Refusal] = []
    failures: list[Failure] = []
    for spec in plan.calibrated:
        try:
            outcomes.append(wiring.run(spec, progress=_step_reporter(spec.tenant, progress)))
        except PipelineRefusal as exc:
            # Reported, not raised. The remaining corpora are independent and a user with one
            # working corpus is better off than a user with a traceback.
            refused.append(
                Refusal(tenant=spec.tenant, reason=_scrub(str(exc), config.dsn, config.migration_dsn))
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
                        f"{type(exc).__name__}: {exc}", config.dsn, config.migration_dsn
                    ),
                )
            )

    skipped = tuple(c.tenant for c in plan.corpora if not c.calibrated)
    return HeadlessReport(
        outcomes=tuple(outcomes),
        refused=tuple(refused),
        failures=tuple(failures),
        skipped=skipped,
    )
