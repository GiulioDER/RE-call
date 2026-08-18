"""One config file in, one report out: the entry point CI and the installer both drive.

Headless because the interesting question about the wizard is whether the pipeline works, and a GUI
answers none of it. A CI job can run this against a throwaway container and get the same answer a
user gets, which is the only way the install path stays honest between releases.

Three decisions worth stating, because each is a place a driver usually gets it wrong:

**A missing or malformed config is refused by name.** The reader is a first-run installer or a CI
log, and neither can act on a `KeyError` raised inside a module they have never heard of.

**One corpus refusing does not abort the others.** They are independent, and a driver that dies on
the second of three leaves the user with less than one that names which corpus failed and why. Same
reasoning as the degraded path inside `run_corpus`, one level up.

**A REFUSED corpus and a DEGRADED corpus are different outcomes.** A refusal is a configuration
problem the user must act on. A degraded corpus is a working install with a named limitation: it
built, it validated, it did not certify, and it was deliberately not promoted. The exit code turns
on that distinction, so the two must not collapse into "failed".

The schema is applied at the embedder's own dimension, because the vector dimension is welded to the
table: a hardcoded one produces a schema no chosen embedder fits, discovered as an insert error
minutes into the first build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from recall.wizard.corpora import default_plan
from recall.wizard.pipeline import CorpusOutcome, PipelineRefusal, run_corpus

__all__ = [
    "HeadlessConfig",
    "HeadlessReport",
    "PipelineRefusal",
    "Refusal",
    "load_config",
    "run_headless",
]

#: Every key the config must carry. Iterated to build the error messages, so a field added to
#: `HeadlessConfig` without a corresponding entry here is a missing-message bug rather than a
#: silently optional key.
_REQUIRED = (
    "dsn",
    "migration_dsn",
    "embedder",
    "corpus_version",
    "docs_root",
    "code_root",
    "memory_root",
)

_ROOT_KEYS = ("docs_root", "code_root", "memory_root")


@dataclass(frozen=True)
class HeadlessConfig:
    """Everything the driver needs, with no defaults it could guess wrong."""

    dsn: str
    migration_dsn: str
    embedder: str
    corpus_version: str
    docs_root: Path
    code_root: Path
    memory_root: Path
    project: str | None = None


@dataclass(frozen=True)
class Refusal:
    """A corpus the driver could not even attempt, and the reason."""

    tenant: str
    reason: str


@dataclass(frozen=True)
class HeadlessReport:
    """What happened to every corpus in the plan."""

    outcomes: tuple[CorpusOutcome, ...] = ()
    refused: tuple[Refusal, ...] = ()
    #: Corpora the driver deliberately did not drive. `memory` is indexed into the legacy chunks
    #: table and has no generation, so handing it to `run_corpus` would produce a promoted
    #: generation nothing can calibrate.
    skipped: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every corpus was at least built. A DEGRADED corpus is still ok."""
        return not self.refused

    @property
    def degraded(self) -> tuple[str, ...]:
        return tuple(o.tenant for o in self.outcomes if not o.certified)

    def render(self) -> str:
        """The text a headless caller reads. Every corpus in the plan appears, including skips."""
        lines: list[str] = []
        for outcome in self.outcomes:
            if outcome.certified:
                lines.append(
                    f"  {outcome.tenant:<8} certified and promoted  "
                    f"({outcome.answerable} answerable / {outcome.unanswerable} unanswerable, "
                    f"{outcome.generation_id})"
                )
            else:
                lines.append(
                    f"  {outcome.tenant:<8} DEGRADED, not promoted  {outcome.generation_id}\n"
                    f"           {outcome.degraded_reason}"
                )
            if outcome.unverified_embedder:
                lines.append(
                    "           note: no verifiable embedder identity, generation is unverified"
                )
        for refusal in self.refused:
            lines.append(f"  {refusal.tenant:<8} REFUSED  {refusal.reason}")
        for tenant in self.skipped:
            lines.append(
                f"  {tenant:<8} skipped   indexed into the legacy chunks table, not calibrated"
            )
        head = "install complete" if self.ok else "install incomplete: a corpus was refused"
        return "\n".join([head, *lines])


class _Services(Protocol):
    """The two side-effecting operations, injected so the driver is testable without a database."""

    def apply_schema(self, dsn: str, *, dim: int) -> None: ...
    def run(self, spec: Any, **kwargs: Any) -> CorpusOutcome: ...


@dataclass
class _RealServices:
    """The production wiring. Constructed lazily so importing this module opens no connection."""

    config: HeadlessConfig
    _embedder: Any = field(default=None, repr=False)

    def embedder(self) -> Any:
        from recall.embeddings import resolve_embedder

        if self._embedder is None:
            self._embedder = resolve_embedder(self.config.embedder)
        return self._embedder

    def apply_schema(self, dsn: str, *, dim: int) -> None:
        from recall.schema import apply_migrations

        apply_migrations(dsn, dim=dim)

    def run(self, spec: Any, **kwargs: Any) -> CorpusOutcome:
        from recall.calibration_v2 import CalibrationRepository
        from recall.generations import GenerationManager

        return run_corpus(
            spec,
            # `development`, not the corpus's SERVING environment. A file:// manifest is refused in
            # production, and `docs` is served there; the two coexist because each MCP server gets
            # its own env block.
            manager=GenerationManager(
                self.config.dsn, spec.tenant, actor="recall-wizard", environment="development"
            ),
            calibrations=CalibrationRepository(self.config.dsn, spec.tenant),
            embedder=self.embedder(),
            corpus_version=self.config.corpus_version,
            project=self.config.project,
            **kwargs,
        )


def load_config(path: str | Path) -> HeadlessConfig:
    """Read and validate the config, naming the key at fault rather than the line number."""
    location = Path(path)
    try:
        raw = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineRefusal(f"cannot read the wizard config {location}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineRefusal(
            f"the wizard config {location} must be a JSON object, not {type(raw).__name__}"
        )

    missing = [key for key in _REQUIRED if not str(raw.get(key, "")).strip()]
    if missing:
        raise PipelineRefusal(
            f"the wizard config {location} is missing {', '.join(missing)}. "
            f"Every one of {', '.join(_REQUIRED)} is required; none has a default the wizard "
            "could guess without being wrong somewhere."
        )

    for key in _ROOT_KEYS:
        # Checked here as well as in `CorpusSpec`, because that refusal names the PATH and this
        # reader needs to be told which config KEY holds it.
        if not Path(raw[key]).is_absolute():
            raise PipelineRefusal(
                f"{key} must be an absolute path, not {raw[key]!r}: a relative root resolves "
                "against the wizard's own working directory, so the commit stamped on every chunk "
                "would be the wizard's rather than the corpus's."
            )

    return HeadlessConfig(
        dsn=str(raw["dsn"]),
        migration_dsn=str(raw["migration_dsn"]),
        embedder=str(raw["embedder"]),
        corpus_version=str(raw["corpus_version"]),
        docs_root=Path(raw["docs_root"]),
        code_root=Path(raw["code_root"]),
        memory_root=Path(raw["memory_root"]),
        project=(str(raw["project"]) if raw.get("project") else None),
    )


def run_headless(
    config: HeadlessConfig, *, services: _Services | None = None
) -> HeadlessReport:
    """Apply the schema, then drive every calibrated corpus, reporting rather than aborting."""
    plan = default_plan(
        embedder=config.embedder,
        docs_root=config.docs_root,
        code_root=config.code_root,
        memory_root=config.memory_root,
    )
    wiring = services if services is not None else _RealServices(config)

    # The embedder's own dimension. `_RealServices` resolves it; a caller injecting services is
    # responsible for its own, which is why the dimension is asked for rather than assumed.
    from recall.embeddings import resolve_embedder

    dim = resolve_embedder(config.embedder).dim
    wiring.apply_schema(config.migration_dsn, dim=dim)

    outcomes: list[CorpusOutcome] = []
    refused: list[Refusal] = []
    for spec in plan.calibrated:
        try:
            outcomes.append(wiring.run(spec))
        except PipelineRefusal as exc:
            # Reported, not raised. The remaining corpora are independent and a user with one
            # working corpus is better off than a user with a traceback.
            refused.append(Refusal(tenant=spec.tenant, reason=str(exc)))

    skipped = tuple(c.tenant for c in plan.corpora if not c.calibrated)
    return HeadlessReport(
        outcomes=tuple(outcomes), refused=tuple(refused), skipped=skipped
    )
