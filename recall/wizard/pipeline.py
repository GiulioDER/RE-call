"""Drive one corpus from a directory to a promoted, calibrated generation.

The sequence is build, validate, calibrate, publish, promote, and its order is the reason this is a
module rather than a script. **Promotion gives a generation a fresh corpus fingerprint**, so a
calibration measured after it binds to a fingerprint that no longer matches and the tenant answers
`CALIBRATION_STALE`. An inverted run reaches the same end state and is broken, which is why the
tests assert the sequence rather than the result.

**The certification floor is checked BEFORE the build**, which is a departure from the design's
ordering and the one change here worth arguing for. The design generates the query set after
`generation validate`. It does not have to: `chunks_from_directory` reads the corpus off disk rather
than out of the database, so the set can be generated and counted first. That matters because a
corpus too small to certify is the only failure in this sequence that cannot be recovered from
afterwards — it builds, validates and promotes without complaint, and then refuses every query
forever with nothing naming the size as the cause. Measured previously at about seven minutes for a
1793-chunk build, which is what checking first saves on a corpus that was never going to work.

**Certification failing is a measurement, not a crash.** The rejected artifact is kept as evidence,
the corpus is marked degraded with its measured separability, and the install continues. A wizard
that dies on the third of three corpora leaves the user with less than one that says which corpus is
degraded and why.

Every collaborator is injected. Nothing here reads the environment or constructs a connection, for
the same reason `generation_build` does not: the wizard drives three corpora in one process, and
`RECALL_ENV` and `RECALL_LOCAL_ALLOWLIST` are process-global. The reader in particular is
constructed from the corpus root rather than from `RECALL_LOCAL_ALLOWLIST`, which is what makes the
variable unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from recall.wizard.corpora import CorpusSpec
from recall.wizard.queryset import (
    MIN_PER_CLASS,
    QuerySetError,
    canonicalize,
    chunks_from_directory,
    generate_offline,
)

__all__ = ["CorpusOutcome", "PipelineRefusal", "run_corpus"]

#: Generated per class before canonicalisation. Above the floor of 20 on purpose: canonicalisation
#: drops duplicates and the vocabulary filter drops off-topic questions that reuse corpus words, so
#: generating exactly the floor lands under it. Measured previously: the off-topic subject pool caps
#: a generated set at 65 questions on this repository's own docs.
DEFAULT_PER_CLASS = 40


class PipelineRefusal(RuntimeError):
    """A refusal raised BEFORE anything expensive or irreversible has happened.

    Distinct from a failure during the run: everything raised as a `PipelineRefusal` is checked
    while the corpus is still just a directory, so the caller knows nothing was built, nothing was
    promoted and there is no artifact to clean up.
    """


@dataclass(frozen=True)
class CorpusOutcome:
    """What happened to one corpus, including the ways it fell short."""

    tenant: str
    generation_id: str
    calibration_id: str | None = None
    certified: bool = False
    #: Populated when the corpus completed but is not fully trustworthy. `None` means it certified.
    degraded_reason: str | None = None
    answerable: int = 0
    unanswerable: int = 0
    steps: tuple[str, ...] = field(default_factory=tuple)


class _Manager(Protocol):
    """The slice of `GenerationManager` this module uses."""

    environment: str

    def create(self, manifest: Any, pipeline: Any, *, allow_unverified: bool = ...) -> Any: ...
    def build(
        self, generation_id: str, reader: Any, embedder: Any, chunker: Any, *, provenance: Any = ...
    ) -> Any: ...
    def validate(self, generation_id: str) -> Any: ...
    def promote(self, generation_id: str, *, unsafe_development: bool = ...) -> None: ...


class _Calibrations(Protocol):
    """The slice of `CalibrationRepository` this module uses."""

    def calibrate(self, generation_id: str, queries: Any, embedder: Any) -> Any: ...
    def publish(self, calibration_id: str) -> Any: ...


def _labelled_set(spec: CorpusSpec, *, per_class: int, seed: int) -> list[dict[str, Any]]:
    """Generate and canonicalise the query set, off disk, before anything is built.

    `canonicalize` rather than `prepare_for_calibration`, because the balance check is applied here
    with a message naming both counts and the floor. `prepare_for_calibration` raises a
    `QuerySetError` whose text is about a query set; at this point the user needs to be told about
    their CORPUS, which is the thing they can change.
    """
    chunks = chunks_from_directory(spec.root, spec.glob)
    try:
        entries = canonicalize(generate_offline(chunks, per_class=per_class, seed=seed))
    except QuerySetError as exc:
        raise PipelineRefusal(
            f"corpus {spec.tenant!r} at {spec.root} is too small to certify: {exc}"
        ) from exc
    return entries


def _refuse_if_below_the_floor(spec: CorpusSpec, entries: list[dict[str, Any]]) -> tuple[int, int]:
    answerable = sum(1 for e in entries if e.get("answerable"))
    unanswerable = len(entries) - answerable
    if answerable < MIN_PER_CLASS or unanswerable < MIN_PER_CLASS:
        # Both counts and the floor, because which side is thin decides what the user changes: a
        # thin answerable side means the corpus is too small, a thin unanswerable side means the
        # off-topic pool ran out. "Calibration failed" tells them neither.
        raise PipelineRefusal(
            f"corpus {spec.tenant!r} at {spec.root} is too small to certify: it yields "
            f"{answerable} answerable and {unanswerable} unanswerable queries, and certification "
            f"needs at least {MIN_PER_CLASS} of each. Building it would take minutes and then "
            f"refuse every query. Add more content under {spec.root}, or serve this corpus "
            "uncalibrated under development trust."
        )
    return answerable, unanswerable


def _identified_request(spec: CorpusSpec, embedder: Any, project: str | None) -> Any:
    """Attach the strongest embedder identity available, or say plainly that there is none.

    This is where `wizard.identity` finally gets wired up. It was written to give a locally
    downloaded embedder a verifiable identity, and until now nothing called it: a generation built
    without one is refused outright by `EmbedderIdentity`, which insists on a revision, a digest, or
    an explicit statement that this is a development build.

    `lineage_revision`, not `revision`, and the difference is not cosmetic. fastembed serves
    `BAAI/bge-small-en-v1.5` out of `qdrant/bge-small-en-v1.5-onnx-q`, so the snapshot SHA is a
    commit in the qdrant repository and does not exist in the BAAI one. `lineage_revision` hands out
    the revision only when the two agree, which is the difference between immutable provenance and a
    revision nobody can resolve.

    `None` is an ordinary answer, not a failure: `hashing` has no weights, and the cloud embedders
    keep theirs on somebody else's machine. Measured previously, and worth not re-deriving:
    `verified: false` costs nothing at serving time, because the serving path never consults it.
    """
    from dataclasses import replace

    from recall.wizard.identity import artifact_identity_for

    request = spec.build_request(project=project)
    identity = artifact_identity_for(embedder)
    if identity is None:
        return replace(request, unverified=True)
    return replace(
        request,
        provider=identity.provider,
        revision=identity.lineage_revision,
        artifact_digest=identity.artifact_digest,
    )


def run_corpus(
    spec: CorpusSpec,
    *,
    manager: _Manager,
    calibrations: _Calibrations,
    embedder: Any,
    corpus_version: str,
    reader: Any | None = None,
    project: str | None = None,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 0,
) -> CorpusOutcome:
    """Take one calibrated corpus from a directory to a promoted, calibrated generation."""
    if not spec.calibrated:
        raise PipelineRefusal(
            f"corpus {spec.tenant!r} is not calibrated, so it has no generation to build: it is "
            "indexed into the legacy chunks table. Driving it through here would produce a "
            "promoted generation nothing can calibrate."
        )
    if manager.environment != "development":
        # A `file://` manifest is refused in production, and `docs` is SERVED in production, so the
        # two environments genuinely coexist on one machine. Getting this wrong fails inside the
        # build with "production generation builds require a versioned S3 manifest", which names
        # neither the wizard nor the corpus.
        raise PipelineRefusal(
            f"the build must run against a development manager, not {manager.environment!r}: a "
            "file:// manifest is refused in production. Serving this corpus under production is "
            "correct and separate; only the BUILD is a development operation."
        )

    # Everything above this line is free. Everything below costs minutes, so the floor is checked
    # here rather than after `validate` where the design sequences it.
    entries = _labelled_set(spec, per_class=per_class, seed=seed)
    answerable, unanswerable = _refuse_if_below_the_floor(spec, entries)

    from recall.generation_build import build_generation
    from recall.lineage import IndexManifestV1
    from recall.manifest import LocalObjectReader
    from recall.wizard.inventory import build_inventory

    # Through `from_dict`, not the positional constructor: `build_inventory` returns plain dicts and
    # `IndexManifestV1.__post_init__` sorts by `item.uri`, so passing them straight in dies with
    # `AttributeError: 'dict' object has no attribute 'uri'`. This is the same conversion
    # `manifest.load_inventory` performs, done without a temporary file.
    manifest = IndexManifestV1.from_dict(
        {
            "schema_version": 1,
            "tenant_id": spec.tenant,
            "corpus_version": corpus_version,
            "objects": build_inventory(spec.root, spec.glob),
        }
    )
    # Constructed from the corpus root, NOT from `RECALL_LOCAL_ALLOWLIST`. Passing the reader is
    # what makes that process-global variable unnecessary, which a wizard driving three corpora in
    # one process cannot do without.
    reader = reader if reader is not None else LocalObjectReader([spec.root])

    steps: list[str] = []
    stats = build_generation(
        manager, manifest, reader, embedder, _identified_request(spec, embedder, project)
    )
    steps.append("build")
    manager.validate(stats.generation_id)
    steps.append("validate")

    artifact = calibrations.calibrate(stats.generation_id, entries, embedder)
    steps.append("calibrate")
    published = calibrations.publish(artifact.calibration_id)
    steps.append("publish")

    # LAST. Promotion gives the generation a fresh corpus fingerprint, so a calibration measured
    # after this point binds to a fingerprint that no longer matches.
    manager.promote(stats.generation_id, unsafe_development=True)
    steps.append("promote")

    certified = bool(getattr(published, "certified", False))
    return CorpusOutcome(
        tenant=spec.tenant,
        generation_id=stats.generation_id,
        calibration_id=artifact.calibration_id,
        certified=certified,
        # The rejected artifact is kept rather than discarded: it is the evidence for why this
        # corpus is degraded, and the measured number is what tells the user whether more content
        # would fix it.
        degraded_reason=(
            None
            if certified
            else (
                f"certification failed: separability {getattr(artifact, 'separability', 'unknown')}"
                f" ({getattr(artifact, 'certification_reason', '')}). The generation is promoted "
                f"and serving; this tenant runs with development trust until it certifies."
            )
        ),
        answerable=answerable,
        unanswerable=unanswerable,
        steps=tuple(steps),
    )
