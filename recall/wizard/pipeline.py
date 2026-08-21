"""Drive one corpus from a directory to a calibrated, promoted generation.

The sequence is build, validate, calibrate, publish, promote.

**Promote is last because it is the irreversible step**, and because it retires whatever the tenant
was serving. Between promotion and a published calibration the tenant is live with no calibration
bound to it, which resolves `CalibrationStatus.MISSING` and refuses under strict trust. 🔁 An
earlier version of this docstring justified the order by claiming promotion gives the generation a
fresh corpus fingerprint, so a calibration measured afterwards would be `CALIBRATION_STALE`. That is
false: `corpus_fingerprint` is written only by `GenerationManager.create` and by `forget`, and
`promote` touches neither, so `CalibrationRepository` explicitly accepts an already-active
generation. The order is right, the reason was not, and the same false sentence is still at
`recall/generation_build.py` awaiting the same correction.

**An uncertified generation is NOT promoted.** This is the correction that most changed the module.
Promoting anyway looked like the friendly choice, and it is destructive: a calibrated `CorpusSpec` is
forced to `TrustMode.STRICT`, a strict tenant whose calibration resolves uncertified refuses every
query before retrieval, and `promote` has by then retired the generation that was working. So on a
re-run over a healthy tenant the wizard would replace something that answers with something that
answers nothing. Degrade therefore means *leave it unpromoted and say so*, naming the generation so
an operator can promote deliberately.

**The certification floor is checked before the build.** `chunks_from_directory` reads the corpus off
disk rather than out of the database, so the query set can be generated and counted first, and a
corpus that cannot certify is refused in seconds rather than after a build measured in minutes. The
floor is the generator's own, not a second copy: asking for `DEFAULT_PER_CLASS` and falling back to
`MIN_PER_CLASS` before refusing, so a corpus that can certify is never turned away for lacking the
headroom.

**The query set is chunked with the SAME callable the build uses.** Not a detail: `code_corpus`
builds with `chunk_code` while `chunks_from_directory` defaults to `chunk_text`, and measured on this
module the two share no chunk, so the calibration would have been measured against text that is not
in the index.

Every collaborator is injected and nothing here reads the environment. The reader is constructed from
the corpus root rather than from `RECALL_LOCAL_ALLOWLIST`, which is what makes that process-global
variable unnecessary for a wizard driving three corpora in one process.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from recall.calibration_v2 import CalibrationUncertified
from recall.embeddings import Embedder
from recall.generation_build import BuildRequest, build_generation, chunker_for
from recall.generations import (
    GenerationManager,
    InvalidGenerationTransition,
    NoActiveGeneration,
)
from recall.lineage import MANIFEST_SCHEMA_VERSION, IndexManifestV1
from recall.manifest import LocalObjectReader, ObjectReader
from recall.wizard.corpora import CorpusSpec
from recall.wizard.identity import artifact_identity_for
from recall.wizard.inventory import build_inventory
from recall.wizard.queryset import (
    DEFAULT_PER_CLASS,
    MIN_PER_CLASS,
    QuerySetError,
    canonicalize,
    chunks_from_directory,
    generate_offline,
    require_balance,
)

__all__ = ["CorpusOutcome", "PipelineRefusal", "run_corpus"]

#: Environments this build may run under. `test` is included because the suite uses it and the
#: underlying guards key on production specifically.
#:
#: 🔁 Corrected. This used to say `production` is excluded because `GenerationManager.create`
#: refuses a non-S3 manifest there. It does not, and nothing in the tree gates `file://` on the
#: environment: `lineage.py` accepts `file://` alongside `s3://` unconditionally, and `manifest.py`
#: gates it on `RECALL_LOCAL_ALLOWLIST` being set. What `create` actually calls in production is
#: `pipeline.require_production_identity()` (`lineage.py:184`), which refuses an embedder with no
#: immutable revision or artifact digest. That is the real reason the wizard cannot build there:
#: its embedders are unverified, which is the same fact `CorpusOutcome.unverified_embedder`
#: already reports. The old wording sent an operator with a properly pinned embedder looking for a
#: manifest problem they did not have.
_BUILD_ENVIRONMENTS = ("development", "test")


class PipelineRefusal(ValueError):
    """A refusal raised BEFORE anything expensive or irreversible has happened.

    Nothing was built, nothing was promoted, and there is no artifact to clean up. A `ValueError`
    so one `except ValueError` catches every refusal in `recall.wizard`; `QuerySetError` and the
    `CorpusSpec` guards are already value errors, and a `RuntimeError` here made this the one
    outlier a caller had to know about by name.
    """


@dataclass(frozen=True)
class CorpusOutcome:
    """What happened to one corpus, including the ways it fell short."""

    tenant: str
    generation_id: str
    calibration_id: str | None = None
    certified: bool = False
    #: Whether the generation was actually promoted. False for an uncertified corpus, which is left
    #: READY on purpose so the previously serving generation keeps serving.
    promoted: bool = False
    #: Populated when the corpus completed but is not fully trustworthy. `None` means it certified.
    degraded_reason: str | None = None
    #: True when no verifiable embedder identity could be resolved, so the generation records
    #: `verified: false`. Surfaced rather than swallowed: the wizard should say it, not discover it.
    unverified_embedder: bool = False
    #: The generation this tenant was serving BEFORE this run, or `None` on a first install. It is
    #: the difference between two outcomes that otherwise render identically: an uncertified build
    #: on an existing install is a working tenant with a named limitation, while an uncertified
    #: build on an empty tenant leaves nothing serving at all. Under production trust that second
    #: case raises `NoActiveGeneration` from outside `trusted_search`'s try block, so the caller
    #: gets a raw exception with no failure code and no advice.
    previously_serving: str | None = None
    answerable: int = 0
    unanswerable: int = 0
    steps: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Coerced for the same reason `CorpusPlan.corpora` is: this is written into resumable state
        # and read back, and a JSON round trip returns a list, which compares unequal to the tuple
        # that produced it and makes the frozen record unhashable.
        object.__setattr__(self, "steps", tuple(self.steps))
        # An uncertified outcome without a reason is a defect, not a state to render. The renderer
        # used to interpolate this field straight into the operator's explanation line, so the word
        # `None` shipped as the advice. Refusing it here means the renderer does not have to guard
        # a case that should never exist.
        if not self.certified and not (self.degraded_reason or "").strip():
            raise ValueError(
                f"an uncertified outcome for {self.tenant!r} must carry a degraded_reason: it is "
                "the only thing that tells the operator what fell short"
            )


def _labelled_set(
    spec: CorpusSpec, chunker: Callable[[str], list[str]], *, per_class: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    """Generate and canonicalise the query set, off disk, before anything is built.

    Returns the entries and the per-class size actually used. Asking for `per_class` and retrying at
    `MIN_PER_CLASS` matters: `generate_offline` refuses outright when the corpus has fewer distinct
    chunks than requested, so requesting the headroom value made every corpus between the floor and
    the headroom "too small to certify" while the message quoted the floor. Measured before the fix:
    39 chunks refused, 40 accepted, against a floor of 20.

    The whole read is inside the try. `chunks_from_directory` raises `QuerySetError` for a missing
    root, a glob matching nothing and unreadable files, and those were escaping uncaught, which
    defeated the contract `PipelineRefusal`'s docstring states.
    """
    attempts = [per_class] if per_class <= MIN_PER_CLASS else [per_class, MIN_PER_CLASS]
    last: QuerySetError | None = None
    for attempt in attempts:
        try:
            chunks = chunks_from_directory(spec.root, spec.glob, chunker)
            return canonicalize(generate_offline(chunks, per_class=attempt, seed=seed)), attempt
        except QuerySetError as exc:
            last = exc
    # Neutral prefix. "too small to certify" was applied to every refusal, including the one whose
    # own text says the corpus OVERLAPS the off-topic pool, whose fix is a disjoint subject list
    # rather than more content.
    raise PipelineRefusal(
        f"corpus {spec.tenant!r} at {spec.root} cannot produce a certifiable query set: {last}"
    ) from last


def _counts(spec: CorpusSpec, entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Both class counts, refusing through the generator's own predicate rather than a copy of it.

    `require_balance` is delegated to rather than reimplemented: an identical comparison lived here
    and would have drifted silently from the one the query set is actually judged by. Only the
    message is this module's, because the user needs to be told about their CORPUS.
    """
    answerable = sum(1 for entry in entries if entry.get("answerable"))
    unanswerable = len(entries) - answerable
    try:
        require_balance(entries)
    except QuerySetError as exc:
        raise PipelineRefusal(
            f"corpus {spec.tenant!r} at {spec.root} yields {answerable} answerable and "
            f"{unanswerable} unanswerable queries, and certification needs at least "
            f"{MIN_PER_CLASS} of each. Add more content under {spec.root}, or serve this corpus "
            "uncalibrated under development trust."
        ) from exc
    return answerable, unanswerable


def _identified_request(spec: CorpusSpec, embedder: Any, project: str | None) -> BuildRequest:
    """Attach the strongest embedder identity available, or say plainly that there is none.

    `lineage_revision`, not `revision`: fastembed serves `BAAI/bge-small-en-v1.5` out of
    `qdrant/bge-small-en-v1.5-onnx-q`, so the snapshot SHA is a commit in the qdrant repository and
    does not exist in the one it would be recorded against. `lineage_revision` hands it out only
    when the two agree.

    `None` is an ordinary answer: `hashing` has no weights and the cloud embedders keep theirs
    elsewhere. The caller is told, via `CorpusOutcome.unverified_embedder`, rather than the
    downgrade happening silently.
    """
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


def _degraded_reason(
    spec: CorpusSpec,
    generation_id: str,
    certification_reason: str,
    previously_serving: str | None,
) -> str:
    """What an uncertified corpus means for this tenant, which depends on what it was serving.

    Two states rendered identically for one release, and only one of them is an install: on an
    upgrade the previous generation keeps answering, and on a first install nothing answers at all.
    Saying "whatever this tenant was serving still serves" in the second case is not a hedge, it is
    a false statement about an empty tenant, and it was printed under the heading "install
    complete".

    Both cases also leave the generation READY, holding a full copy of the corpus's chunk rows.
    That is deliberate (it is the evidence, and it can still be promoted), but it is not free and
    the operator is the only one who can decide to reclaim it, so the command is named here rather
    than left to be discovered when the disk fills.
    """
    lead = (
        f"certification failed ({certification_reason}). The generation {generation_id} is built "
        "and validated but NOT promoted"
    )
    if previously_serving:
        serving = (
            f", so generation {previously_serving} keeps serving this tenant. Promote it "
            "deliberately once a certified calibration is published, or add content and re-run."
        )
    else:
        serving = (
            f", and {spec.tenant!r} has NO active generation, so this tenant answers nothing. "
            f"It is served as {spec.serving_environment}/{spec.trust_mode}, under which a query "
            "raises NoActiveGeneration rather than returning a refusal with a failure code. Add "
            "content and re-run, or promote this generation deliberately to serve it uncertified."
        )
    return (
        f"{lead}{serving} It still holds a full copy of the corpus, which `recall generation "
        f"abandon {generation_id}` releases for `recall generation gc`."
    )


#: Reserved for `recall_mcp/service.py`'s desktop upload path, which abandons superseded READY
#: generations whose `corpus_version` starts with it. Defined here rather than imported to keep the
#: wizard free of a dependency on the MCP service; the two are pinned equal by a test.
_RESERVED_CORPUS_PREFIX = "desktop-"


def run_corpus(
    spec: CorpusSpec,
    *,
    manager: GenerationManager,
    calibrations: Any,
    embedder: Embedder | Any,
    corpus_version: str,
    reader: ObjectReader | None = None,
    project: str | None = None,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> CorpusOutcome:
    """Take one calibrated corpus from a directory to a calibrated generation.

    `progress` is called with each step name as it starts. Without it the wizard is silent through a
    build measured in minutes, three times over, and a user cannot tell a running install from a
    hung one.
    """
    announce = progress or (lambda _step: None)

    if not spec.calibrated:
        raise PipelineRefusal(
            f"corpus {spec.tenant!r} is not calibrated, so it has no generation to build: it is "
            "indexed into the legacy chunks table. Driving it through here would produce a "
            "promoted generation nothing can calibrate."
        )
    if manager.environment not in _BUILD_ENVIRONMENTS:
        raise PipelineRefusal(
            f"the build must run under one of {_BUILD_ENVIRONMENTS}, not "
            f"{manager.environment!r}: a production build requires an embedder with an immutable "
            "revision or artifact digest, and the wizard's embedders have neither. Serving this "
            "corpus under production is correct and separate; only the BUILD is not."
        )
    if manager.tenant_id != spec.tenant:
        # Refused here rather than inside `create`, which would already have cost the walk, the
        # query generation and the inventory.
        raise PipelineRefusal(
            f"manager is bound to tenant {manager.tenant_id!r}, not {spec.tenant!r}"
        )
    if not corpus_version.strip():
        raise PipelineRefusal("corpus_version must be non-empty; the convention is an ISO date")
    # ⛔ **`desktop-` is RESERVED, and the reservation is load-bearing for a destructive operation.**
    # `recall_mcp/service.py::_release_superseded` selects generations to ABANDON by this prefix,
    # precisely so a desktop upload never reclaims a corpus a wizard install built and deliberately
    # left READY for an operator to promote. `corpus_version` reaches here straight from user config
    # with no other validation, so a copy-pasted value starting with the prefix would opt a wizard
    # corpus into that reclaim — abandon, then gc, then the chunk rows cascade away. Refusing at the
    # only place it enters the system is cheaper than making the reclaim smarter.
    if corpus_version.strip().startswith(_RESERVED_CORPUS_PREFIX):
        raise PipelineRefusal(
            f"corpus_version must not begin with {_RESERVED_CORPUS_PREFIX!r}: that prefix is "
            "reserved for uploads made through the desktop app, which reclaims its own superseded "
            "builds by matching on it. Use an ISO date, which is the convention."
        )

    request = _identified_request(spec, embedder, project)
    # The SAME callable the build will bind, so the queries describe chunks that exist in the index.
    chunker = chunker_for(request)[0]

    announce("queries")
    entries, _used = _labelled_set(spec, chunker, per_class=per_class, seed=seed)
    answerable, unanswerable = _counts(spec, entries)

    announce("manifest")
    manifest = IndexManifestV1.from_dict(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tenant_id": spec.tenant,
            "corpus_version": corpus_version,
            "objects": build_inventory(spec.root, spec.glob),
        }
    )
    # Constructed from the corpus root, NOT from `RECALL_LOCAL_ALLOWLIST`.
    reader = reader if reader is not None else LocalObjectReader([spec.root])

    steps: list[str] = []
    announce("build")
    stats = build_generation(manager, manifest, reader, embedder, request)
    steps.append("build")
    generation_id = stats.generation_id

    def _fail(step: str, exc: BaseException) -> None:
        """Mark the generation failed so `gc` can reclaim it, then let the caller re-raise.

        Without this an abort after `validate` leaves the generation READY, and `gc` collects only
        `retired` and `failed`, so every failed attempt leaks a full copy of the corpus's chunk rows
        that nothing will ever reclaim.

        ⚠️ For most of this module's life the paragraph above was false, and it was false on the
        path it was written for. `validate` sets the state to READY, and `GenerationManager.fail`
        refuses READY outright; the `InvalidGenerationTransition` landed in a bare `except
        Exception: pass` directly below, so the leak this guard exists to prevent happened every
        time, silently. The test that pinned it asserted against a fake whose `fail` succeeds from
        any state, so the guard could not be observed failing. Hence `abandon`, which is the
        READY-only reclaim route, and hence the real-manager test alongside the fake one.
        """
        reason = f"wizard pipeline failed at {step}: {exc}"
        try:
            manager.fail(generation_id, reason)
            return
        except InvalidGenerationTransition:
            pass
        except Exception:  # noqa: BLE001 - the original failure is what matters
            return
        try:
            manager.abandon(generation_id, reason)
        except Exception:  # noqa: BLE001 - the original failure is what matters
            pass

    try:
        announce("validate")
        manager.validate(generation_id)
        steps.append("validate")

        announce("calibrate")
        artifact = calibrations.calibrate(generation_id, entries, embedder)
        steps.append("calibrate")
    except BaseException as exc:
        _fail("validate/calibrate", exc)
        raise

    announce("publish")
    try:
        calibrations.publish(artifact.calibration_id)
        certified = True
        steps.append("publish")
    except CalibrationUncertified:
        # NOT an error. The artifact is kept as evidence and the generation is left unpromoted.
        # `publish` raises rather than returning an uncertified artifact, which is why the earlier
        # version of this module never reached its own degraded branch against the real repository.
        certified = False

    # Read BEFORE promoting, because promoting is what replaces it. Its absence is the whole
    # difference between a degraded upgrade and a degraded first install, and only one of those is
    # an install that answers anything.
    try:
        previously_serving: str | None = manager.active_generation_id()
    except NoActiveGeneration:
        previously_serving = None

    if certified:
        announce("promote")
        try:
            # ⚠️ The flag is a DEVELOPMENT requirement and a refusal wherever certification is
            # required, so it cannot be passed unconditionally. `certification_required` is the one
            # switch both sides read, so this cannot disagree with what `promote` will do.
            #
            # It reads the SERVING environment, which for every wizard corpus is production while
            # the build runs under development. Before that distinction existed this expression was
            # `manager.environment != "production"` and was therefore ALWAYS true here, so the gate
            # never ran on an install and the `certified` check below was the only thing enforcing
            # it. Two implementations of one rule, one of them untested.
            manager.promote(
                generation_id,
                unsafe_development=not manager.certification_required,
            )
        except BaseException as exc:
            _fail("promote", exc)
            raise
        steps.append("promote")

    return CorpusOutcome(
        tenant=spec.tenant,
        generation_id=generation_id,
        calibration_id=artifact.calibration_id,
        certified=certified,
        promoted=certified,
        degraded_reason=(
            None
            if certified
            else _degraded_reason(
                spec, generation_id, artifact.certification_reason, previously_serving
            )
        ),
        unverified_embedder=request.unverified,
        previously_serving=previously_serving,
        answerable=answerable,
        unanswerable=unanswerable,
        steps=tuple(steps),
    )
