"""Automatic calibration for a `LiteStore`: certified trust with no labelled query file.

The hosted upload path already proved the chain (`recall_mcp.desktop_ingest._certify_upload`,
memo `the-hosted-ingest-path-auto-calibrates`): generate a labelled query set from the corpus
itself (`recall.wizard.queryset.generate_offline`, no model, no network), measure each query's
exact best cosine (`recall.eval.calibrate.measure_top_cosines` over `LiteStore.top_cosine`), and
fit with the shared `recall.calibration.from_samples`. Nothing here is a second implementation of
a rule: the query generator, the measurement and the fit are the ones the Postgres path uses, and
the result is the same `CalibrationArtifactV2`, checksummed, so `trusted_search` consumes it
unchanged through `LiteStore.resolve_calibration`.

What differs is only where it lives: one SQLite table instead of `recall_calibrations`, and the
corpus fingerprint is a digest of the stored chunks (ids, text and vectors) rather than a generation's.

**A changed corpus re-verifies its threshold instead of refitting it** (`carry_forward`). A fit
stores what it was measured on: its labelled questions, their vectors and the corpus by source.
When the corpus changes, those vectors are re-scored against it (no embedding, so it costs
milliseconds rather than the seconds a fit costs) and the FIXED threshold is judged on the fresh
scores by the rule the Postgres carry-forward applies
(`recall.calibration_v2.judge_inherited_threshold`): it must still certify AND decide each class
within `DEFAULT_MAX_CARRY_FORWARD_ERROR`. The exact semantics, which every caller and the trust
gate see through `resolve`:

| corpus delta from the fit | re-verification | resolves | refit |
|---|---|---|---|
| any | fails, or impossible | `stale`: strict search refuses | needed now |
| up to `DEFAULT_MAX_CORPUS_DELTA` (0.25) | passes | `certified`, carried | none |
| above that, up to `MAX_BRIDGE_CORPUS_DELTA` (0.50) | passes | `certified`, carried | due |
| above `MAX_BRIDGE_CORPUS_DELTA` | not tried | `stale`: strict search refuses | needed now |

So a calibration that failed re-verification is never served, whatever the delta, and the delta
decides only how far the labelled questions are trusted to speak for the questions nobody
labelled. `ensure_calibrated` performs a due or needed refit inline by default, or in a background
thread with `background=True` (the MCP write path), so the write returns at once; meanwhile the
carried calibration serves only while it still re-verifies, which `resolve` re-checks on every
corpus change. Carried calibrations are never stored as calibrations: each is recomputed from the
fit and its evidence, and `ensure_calibrated` records the lineage as a `calibration_carried_forward`
audit event, so the newest stored calibration is always a FIT.

Why the delta is not the alarm: measured over 57 snapshots of three real corpora, a frozen
threshold first exceeded the 0.10 error bound at a corpus delta of 0.945, and a delta-only rule at
0.25 would have demanded 56 recalibrations of which 5 were needed. What moves is false confirm,
tracking growth, which the re-verification measures directly.

A corpus too small to produce a query set (about 21 chunks, memo `the-certifying-floor-is-21-chunks`)
has no calibration at all and resolves `missing`, with the reason in plain words.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from recall.calibration import from_samples
from recall.calibration_v2 import (
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    DEFAULT_MAX_CORPUS_DELTA,
    CalibrationArtifactV2,
    CalibrationError,
    CalibrationResolution,
    CalibrationStatus,
    canonical_query_set,
    corpus_delta,
    judge_inherited_threshold,
)
from recall.embedding_core import embed_query
from recall.eval.calibrate import measure_top_cosines
from recall.lineage import canonical_json, canonical_sha256
from recall.lite.store import LiteStoreError
from recall.near_miss import describe as describe_near_miss
from recall.near_miss import near_miss_coverage, near_miss_probes
from recall.observability import get_logger

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.store import LiteStore

CREATED_BY = "recall-lite"
_PLACEHOLDER_CHECKSUM = "0" * 64
_log = get_logger("lite.calibration")
_COVERAGE_UNAVAILABLE = "near-miss coverage unavailable; provisional for in-vocabulary near-misses"

#: Past this corpus delta from the fit, a carried threshold is no longer served at all, however
#: well it re-verifies. Between `DEFAULT_MAX_CORPUS_DELTA` (where a refit becomes due) and this, the
#: carried calibration bridges the refit, so a write that crosses the refit line does not turn
#: strict search off while the refit runs.
#:
#: 0.50 is a ceiling, not a measured safe distance: about half the smallest delta at which a frozen
#: threshold was measured to exceed its error bound (0.945, `docs` corpus, 2026-08-21), on one
#: embedder and generated queries. It bounds only how long a store with no refit running (a writer
#: that never calls `ensure_calibrated`) keeps answering from a threshold it did not fit.
MAX_BRIDGE_CORPUS_DELTA = 0.5

#: Fits a background refit may run in a row when writes keep moving the corpus under it.
_MAX_REFIT_ATTEMPTS = 3


@dataclass(frozen=True)
class CarryForward:
    """What `carry_forward` found: a certified carried artifact, or why there is none."""

    #: The fit's threshold bound to the corpus as it is now; None when it may not be served.
    artifact: CalibrationArtifactV2 | None
    reason: str
    #: True when the fit should be replaced: the corpus moved past `DEFAULT_MAX_CORPUS_DELTA`, or
    #: the threshold may not be served at all.
    refit_due: bool
    #: The fit the threshold comes from, when there is one.
    origin: CalibrationArtifactV2 | None = None
    #: Delta counts and per-class error rates, as a carried artifact's provenance records them.
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationOutcome:
    """What `auto_calibrate` did: the status the store now resolves to, and why in plain words."""

    status: CalibrationStatus
    reason: str
    artifact: CalibrationArtifactV2 | None = None


def pipeline_fingerprint(model: str, dimension: int) -> str:
    """The lite pipeline's identity: the store kind and the embedder that fills it."""
    return canonical_sha256({"store": "recall-lite", "embedder": {"model": model, "dimension": dimension}})


class _RecordingEmbedder:
    """Passes queries through to `embed_query(inner, ...)` and keeps each vector, in order.

    Lets a fit measure with `measure_top_cosines`, the one sampling rule every calibration uses,
    and keep the vectors it embedded for `carry_forward` without embedding anything twice.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.vectors: list[list[float]] = []

    def embed_query(self, text: str) -> list[float]:
        vector = embed_query(self._inner, text)
        self.vectors.append(vector)
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:  # not reached: embed_query is preferred
        return [self.embed_query(text) for text in texts]


class _StoredQueryEmbedder:
    """Answers each labelled question with the vector stored at fit time: re-scoring embeds nothing."""

    def __init__(self, name: str, dim: int, vectors: Mapping[str, list[float]]) -> None:
        self.name = name
        self.dim = dim
        self._vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]

    def embed(self, texts: list[str]) -> list[list[float]]:  # not reached: embed_query is preferred
        return [self._vectors[text] for text in texts]


def source_digests(rows: Sequence[tuple[str, str, str]]) -> dict[str, str]:
    """`{source: corpus_digest of its (chunk id, text)}` from `(source, id, text)` rows."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for source, cid, text in rows:
        grouped.setdefault(source, []).append((cid, text))
    return {source: corpus_digest(chunks) for source, chunks in grouped.items()}


def _query_set(texts: list[str]) -> tuple[list[dict[str, Any]] | None, str]:
    """`_query_set_for` from the hosted path: the default size, then the certification floor."""
    from recall.wizard.queryset import DEFAULT_PER_CLASS, MIN_PER_CLASS, QuerySetError, canonicalize, generate_offline

    last = ""
    for per_class in (DEFAULT_PER_CLASS, MIN_PER_CLASS):
        try:
            return canonicalize(generate_offline(texts, per_class=per_class)), ""
        except QuerySetError as exc:
            last = str(exc)
    return None, last


def artifact_from_json(raw: str) -> CalibrationArtifactV2:
    """Rebuild a stored artifact and verify its checksum, as `import_bundle` does."""
    values = json.loads(raw)
    values["separability_ci"] = tuple(values["separability_ci"])
    artifact = CalibrationArtifactV2(**values)
    artifact.verify_checksum()
    return artifact


def artifact_to_json(artifact: CalibrationArtifactV2) -> str:
    payload = artifact.immutable_payload()
    payload["lifecycle_state"] = artifact.lifecycle_state
    payload["checksum"] = artifact.checksum
    return canonical_json(payload).decode("utf-8")


def _manifest_objects(sources: Mapping[str, str]) -> list[dict[str, str]]:
    return [{"uri": uri, "sha256": digest} for uri, digest in sources.items()]


def _identity(model: object, dimension: object) -> tuple[str, int]:
    return str(model), int(cast("Any", dimension))


def carry_forward(
    store: LiteStore,
    fit: CalibrationArtifactV2 | None = None,
    *,
    embedder: Embedder | None = None,
    now: datetime | None = None,
    max_corpus_delta: float = DEFAULT_MAX_CORPUS_DELTA,
    max_error: float = DEFAULT_MAX_CARRY_FORWARD_ERROR,
) -> CarryForward:
    """Re-verify the newest fit's FIXED threshold on the corpus as it is now, without refitting.

    The fit's own labelled questions are re-scored against the current corpus from the vectors
    stored with it, and the threshold is judged on those fresh scores by
    `judge_inherited_threshold`, the rule the Postgres carry-forward applies. The delta is measured
    from the FIT's corpus, by source, not summed over steps: the fit's source manifest is kept with
    it, so the chain is never longer than one step and its drift is exact. See the module docstring
    for what each outcome resolves to.

    `embedder`, when given, must be the one the fit used (a threshold binds to one model). Read
    paths pass none: the trust gate compares the runtime embedder with the stored one itself.
    """
    if fit is None:
        raw = store.latest_calibration_json()
        if raw is None:
            return CarryForward(None, "no calibration has been fitted yet", refit_due=True)
        fit = artifact_from_json(raw)
    origin = fit

    def refuse(reason: str, provenance: Mapping[str, Any] | None = None) -> CarryForward:
        return CarryForward(None, reason, refit_due=True, origin=origin, provenance=dict(provenance or {}))

    if fit.carry_forward is not None:
        return refuse("a carried calibration is not carried again; it is refitted")
    if not fit.certified:
        return refuse("the last fit did not certify, so it has no threshold to carry")
    fit_identity = _identity(fit.embedder_identity.get("model"), fit.embedder_identity.get("dimension", -1))
    if embedder is not None and _identity(embedder.name, embedder.dim) != fit_identity:
        return refuse("the embedder is not the one the fit used, and a threshold binds to one model")
    stored = store.calibration_evidence(fit.calibration_id)
    if stored is None:
        return refuse("this calibration was fitted before its test questions were kept, so it is refitted")
    import numpy as np

    try:
        labels, digest = canonical_query_set(json.loads(stored[0]))
        if digest != fit.query_set_digest:
            return refuse("the stored test questions no longer match the calibration's digest")
        vectors = np.frombuffer(stored[1], dtype=np.float32)
        if vectors.size != len(labels) * fit_identity[1]:
            return refuse("the stored test-question vectors do not match the questions")
        fitted_sources = json.loads(stored[2])
        if not isinstance(fitted_sources, dict):
            return refuse("the stored source manifest does not read")
    except (CalibrationError, ValueError, TypeError) as exc:
        return refuse(f"the stored evidence does not read: {exc}")
    fingerprint = store.corpus_fingerprint()
    delta = corpus_delta(_manifest_objects(fitted_sources), _manifest_objects(store.source_manifest()))
    moved = float(delta["corpus_delta"])
    changes = (
        f"{delta['sources_added']} added, {delta['sources_removed']} removed, "
        f"{delta['sources_modified']} modified of {delta['sources_union']} sources"
    )
    if moved > MAX_BRIDGE_CORPUS_DELTA:
        return refuse(
            f"the memory changed by {moved:.0%} of its sources since the fit ({changes}), past the "
            f"{MAX_BRIDGE_CORPUS_DELTA:.0%} a carried threshold may span",
            delta,
        )
    by_text = {
        str(entry["query"]): row.tolist()
        for entry, row in zip(labels, vectors.reshape(len(labels), fit_identity[1]), strict=True)
    }
    scorer = _StoredQueryEmbedder(fit_identity[0], fit_identity[1], by_text)
    answerable, unanswerable = measure_top_cosines(cast("Any", store), cast("Any", scorer), list(labels))
    if store.corpus_fingerprint() != fingerprint:
        return refuse("the memory changed while its calibration was being re-verified", delta)
    try:
        verdict = judge_inherited_threshold(fit, answerable, unanswerable, max_error=max_error)
    except CalibrationError as exc:
        return refuse(f"the fit's threshold cannot be judged on the memory as it is now: {exc}", delta)
    provenance: dict[str, Any] = {
        "parent_calibration_id": fit.calibration_id,
        "parent_generation_id": fit.generation_id,
        "parent_corpus_fingerprint": fit.corpus_fingerprint,
        "parent_separability": fit.separability,
        "inherited_threshold": fit.threshold,
        "inherited_scale": fit.scale,
        "refit_threshold": verdict.refit_threshold,
        "max_corpus_delta": max_corpus_delta,
        "max_bridge_corpus_delta": MAX_BRIDGE_CORPUS_DELTA,
        "max_carry_forward_error": max_error,
        **verdict.errors,
        **delta,
        # Measured from the fit itself, not summed over steps: there is only ever one step.
        "cumulative_corpus_delta": moved,
        "refit_due": moved > max_corpus_delta,
    }
    if not verdict.certified:
        return refuse(f"the fit's threshold does not pass on the memory as it is now: {verdict.reason}", provenance)
    runtime = verdict.runtime
    if runtime.separability is None or runtime.separability_ci is None:  # narrowed for mypy
        return refuse("both kinds of test question are needed to re-verify", provenance)
    draft = CalibrationArtifactV2(
        # One id per (fit, corpus), so every process serving this corpus names the same calibration.
        calibration_id="cal_" + canonical_sha256({"carried_from": fit.calibration_id, "corpus": fingerprint})[:32],
        tenant_id=store.tenant,
        generation_id=store.lite_generation_id(),
        embedder_identity=dict(fit.embedder_identity),
        pipeline_fingerprint=fit.pipeline_fingerprint,
        corpus_fingerprint=fingerprint,
        query_set_digest=fit.query_set_digest,
        threshold=runtime.threshold,
        scale=runtime.scale,
        separability=runtime.separability,
        separability_ci=runtime.separability_ci,
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        certified=True,
        certification_reason=runtime.certification_reason,
        lifecycle_state="published",
        created_at=(now or datetime.now(UTC)).isoformat(),
        created_by=CREATED_BY,
        scores={"answerable": answerable, "unanswerable": unanswerable},
        checksum=_PLACEHOLDER_CHECKSUM,
        carry_forward=provenance,
    )
    artifact = dataclasses.replace(draft, checksum=canonical_sha256(draft.immutable_payload()))
    errors = verdict.errors
    return CarryForward(
        artifact,
        f"carried forward: the memory changed by {moved:.1%} of its sources since the fit ({changes}) and "
        f"the fit's threshold {fit.threshold:.4f} still decides its test questions (false abstain "
        f"{errors['false_abstain_rate']:.1%}, false confirm {errors['false_confirm_rate']:.1%})",
        refit_due=moved > max_corpus_delta,
        origin=fit,
        provenance=provenance,
    )


def resolve(store: LiteStore) -> CalibrationResolution:
    """The newest calibration, judged against the corpus as it is now.

    `missing` when none was ever stored; `uncertified` when it was fitted to this corpus and did
    not separate; `certified` when it was fitted to this corpus, or when the corpus changed and its
    threshold re-verifies on it (`carry_forward`, with the carried artifact); else `stale`.
    Only a certified resolution carries the artifact, so no other state can lend its threshold.
    """
    raw = store.latest_calibration_json()
    if raw is None:
        return CalibrationResolution(CalibrationStatus.MISSING)
    artifact = artifact_from_json(raw)
    if artifact.corpus_fingerprint != store.corpus_fingerprint():
        carried = carry_forward(store, artifact)
        if carried.artifact is None:
            return CalibrationResolution(CalibrationStatus.STALE)
        return CalibrationResolution(CalibrationStatus.CERTIFIED, carried.artifact)
    if not artifact.certified:
        return CalibrationResolution(CalibrationStatus.UNCERTIFIED)
    return CalibrationResolution(CalibrationStatus.CERTIFIED, artifact)


def auto_calibrate(
    store: LiteStore,
    embedder: Embedder,
    *,
    now: datetime | None = None,
    queries: Sequence[Mapping[str, Any]] | None = None,
) -> CalibrationOutcome:
    """Fit a calibration to the corpus as it is now and store it, certified or not, refusing
    publication if the corpus changes during the fit.

    `queries` is a labelled set (`{"query", "answerable"}` entries) to fit on instead of the
    questions generated from the corpus; it is checked and canonicalised like any stored set.
    """
    revision = store.corpus_revision()
    fingerprint = store.corpus_fingerprint()
    generation_id = store.lite_generation_id()
    fitted_on = store.corpus_fingerprint()
    sources = store.source_manifest()
    texts = [chunk.text for chunk in store.iter_chunks() if chunk.text.strip()]
    entries, why = ([dict(q) for q in queries], "") if queries is not None else _query_set(texts)
    if entries is None:
        return CalibrationOutcome(
            CalibrationStatus.MISSING,
            f"Trusted answers turn on once memory holds enough text to test itself: {len(texts)} "
            f"chunk(s) now ({why}). Add a few more memos.",
        )
    labels, query_digest = canonical_query_set(entries)
    recorder = _RecordingEmbedder(embedder)
    # Duck-typed as everywhere else: the scorer needs only `top_cosine`, which LiteStore has.
    answerable, unanswerable = measure_top_cosines(cast("Any", store), cast("Any", recorder), list(labels))
    fingerprint = store.corpus_fingerprint()
    if fingerprint != fitted_on:
        # A write landed while this fit read the corpus (a background refit beside a server that
        # keeps indexing), so its questions and its scores may describe two different corpora.
        # Nothing is stored; the caller fits again on the corpus as it is now.
        return CalibrationOutcome(
            CalibrationStatus.STALE, "the memory changed while it was being calibrated; the fit was discarded"
        )
    runtime = from_samples(str(embedder.name), answerable, unanswerable)
    if runtime.separability is None or runtime.separability_ci is None:
        return CalibrationOutcome(CalibrationStatus.UNCERTIFIED, "both kinds of test question are needed to calibrate")
    certified = runtime.certified is True
    model, dimension = str(embedder.name), int(embedder.dim)
    draft = CalibrationArtifactV2(
        calibration_id="cal_" + uuid.uuid4().hex,
        tenant_id=store.tenant,
        generation_id=generation_id,
        embedder_identity={"model": model, "dimension": dimension},
        pipeline_fingerprint=pipeline_fingerprint(model, dimension),
        corpus_fingerprint=fingerprint,
        query_set_digest=query_digest,
        threshold=runtime.threshold,
        scale=runtime.scale,
        separability=runtime.separability,
        separability_ci=runtime.separability_ci,
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        certified=certified,
        certification_reason=runtime.certification_reason,
        lifecycle_state="published" if certified else "rejected",
        created_at=(now or datetime.now(UTC)).isoformat(),
        created_by=CREATED_BY,
        scores={"answerable": answerable, "unanswerable": unanswerable},
        checksum=_PLACEHOLDER_CHECKSUM,
    )
    artifact = dataclasses.replace(draft, checksum=canonical_sha256(draft.immutable_payload()))
    import numpy as np

    evidence = (
        artifact.calibration_id,
        canonical_json(list(labels)).decode("utf-8"),
        np.asarray(recorder.vectors, dtype=np.float32).tobytes(),
        canonical_json(sources).decode("utf-8"),
    )
    store.save_calibration(
        artifact_to_json(artifact), created_at=artifact.created_at, model=model, dimension=dimension,
        expected_revision=revision, evidence=evidence
    )
    coverage = ""
    if runtime.threshold is not None:
        # Report only: measured with the threshold just fitted, never fed back into the fit or the
        # certified decision. See `recall.near_miss` for why the two are kept apart.
        try:
            asked = [str(e["query"]) for e in labels if e.get("answerable") is True]
            probes = near_miss_probes(texts, asked, per_class=len(asked))
            report = near_miss_coverage(store, embedder, float(runtime.threshold), probes)
            store.save_near_miss_coverage(artifact.calibration_id, report)
            coverage = "; " + describe_near_miss(report)
        except Exception as exc:  # BROAD-CATCH: fail-open
            # The fit is already saved. Optional diagnostics cannot withdraw it, and
            # provider errors may carry private input or credentials, so log only type.
            _log.warning("Optional near-miss coverage unavailable (%s)", type(exc).__name__)
            coverage = "; " + _COVERAGE_UNAVAILABLE
    if certified:
        return CalibrationOutcome(
            CalibrationStatus.CERTIFIED,
            f"certified on {len(answerable)} answerable and {len(unanswerable)} unanswerable test questions "
            f"(separability {runtime.separability:.3f}){coverage}",
            artifact,
        )
    return CalibrationOutcome(
        CalibrationStatus.UNCERTIFIED, f"calibration did not certify: {runtime.certification_reason}{coverage}"
    )


def _record_carry(store: LiteStore, carried: CarryForward) -> None:
    """The carry's lineage in the audit ledger: carried calibrations are never stored themselves."""
    origin, keys = carried.origin, ("corpus_delta", "false_abstain_rate", "false_confirm_rate", "refit_due")
    if origin is None or not origin.certified:
        return
    payload: dict[str, Any] = {"parent_calibration_id": origin.calibration_id}
    payload.update({key: carried.provenance[key] for key in keys if key in carried.provenance})
    if carried.artifact is not None:
        store.append_audit_event(
            "calibration_carried_forward",
            {"calibration_id": carried.artifact.calibration_id, **payload},
            generation_id=carried.artifact.generation_id,
            actor=CREATED_BY,
            event_id="evt_carry_" + carried.artifact.calibration_id,
        )
    else:
        store.append_audit_event(
            "calibration_carry_forward_refused",
            {**payload, "reason": carried.reason},
            generation_id=store.lite_generation_id(),
            actor=CREATED_BY,
        )


def ensure_calibrated(store: LiteStore, embedder: Embedder, *, background: bool = False) -> CalibrationOutcome:
    """Keep a calibration in force after a write: carry it forward when it re-verifies, else refit.

    Never fitted: fit now, inline, whatever `background` says (the first fit, below or at the
    certifying floor, behaves as it always has). Unchanged since the fit: keep it; a fit that did
    not certify is not retried on the same corpus, because the query set and the measurement are
    deterministic. Changed: `carry_forward`. A carry within `DEFAULT_MAX_CORPUS_DELTA` is the whole
    answer and costs no embedding. A due refit (past that delta), or a needed one (the carry
    failed), runs inline by default; with `background=True`, and once a fit has certified, it runs
    in a background thread (`schedule_refit`) and this returns at once, with the carried
    calibration in force when it re-verified and `stale` (strict search refuses) when it did not.
    """
    raw = store.latest_calibration_json()
    if raw is None:
        return auto_calibrate(store, embedder)
    fit = artifact_from_json(raw)
    if fit.corpus_fingerprint == store.corpus_fingerprint():
        if fit.certified:
            return CalibrationOutcome(CalibrationStatus.CERTIFIED, "already certified for this corpus", fit)
        return CalibrationOutcome(
            CalibrationStatus.UNCERTIFIED, "this corpus was calibrated and did not certify; it is refitted when it changes"
        )
    carried = carry_forward(store, fit, embedder=embedder)
    _record_carry(store, carried)
    if carried.artifact is not None and not carried.refit_due:
        return CalibrationOutcome(CalibrationStatus.CERTIFIED, carried.reason, carried.artifact)
    if background and fit.certified:
        schedule_refit(store, embedder)
        if carried.artifact is not None:
            return CalibrationOutcome(
                CalibrationStatus.CERTIFIED, carried.reason + "; a refit is due and is running in the background", carried.artifact
            )
        return CalibrationOutcome(
            CalibrationStatus.STALE,
            f"{carried.reason}; strict search refuses until the refit running in the background lands",
        )
    return auto_calibrate(store, embedder)


@dataclass
class _RefitJob:
    thread: threading.Thread | None = None
    #: Set by `schedule_refit` while this job runs: a write landed and may want another fit.
    again: bool = False
    finished: bool = False
    outcome: CalibrationOutcome | None = None
    error: str | None = None


_REFIT_LOCK = threading.Lock()
_REFITS: dict[str, _RefitJob] = {}


def _refit_key(store: LiteStore) -> str:
    return str(store.path.resolve())


def _refit_wanted(store: LiteStore) -> bool:
    """True while the newest fit no longer covers the corpus without a refit."""
    raw = store.latest_calibration_json()
    if raw is None:
        return True
    fit = artifact_from_json(raw)
    if fit.corpus_fingerprint == store.corpus_fingerprint():
        return False
    carried = carry_forward(store, fit)
    return carried.artifact is None or carried.refit_due


def _run_refit(store: LiteStore, embedder: Embedder, job: _RefitJob) -> None:
    try:
        for _attempt in range(_MAX_REFIT_ATTEMPTS):
            job.outcome = auto_calibrate(store, embedder)
            with _REFIT_LOCK:
                job.again = False
            if not _refit_wanted(store):
                # Checked again under the lock, so a write that asks between the two checks is
                # never lost: either it sees this job unfinished and sets `again`, or it starts a
                # new job.
                with _REFIT_LOCK:
                    if not job.again:
                        job.finished = True
                        return
    except Exception as exc:  # BROAD-CATCH: fail-closed  # a failed refit leaves the store stale, never wrong
        job.error = f"{type(exc).__name__}: {exc}"
        _log.warning("background calibration refit of %s failed: %s", store.path, job.error)
    finally:
        with _REFIT_LOCK:
            job.finished = True


def schedule_refit(store: LiteStore, embedder: Embedder) -> bool:
    """Refit `store` in a background thread; False when one is already running for its file.

    A running refit is not doubled: it re-checks the corpus after each fit and fits again (up to
    `_MAX_REFIT_ATTEMPTS` in a row) while the newest fit no longer covers it. The thread is a
    daemon, so a process that exits mid-fit stores nothing, and the next write schedules again.
    """
    key = _refit_key(store)
    with _REFIT_LOCK:
        job = _REFITS.get(key)
        if job is not None and not job.finished:
            job.again = True
            return False
        job = _RefitJob()
        job.thread = threading.Thread(
            target=_run_refit, args=(store, embedder, job), name="recall-lite-refit", daemon=True
        )
        _REFITS[key] = job
        job.thread.start()
    return True


def wait_for_refit(store: LiteStore, timeout: float | None = None) -> CalibrationOutcome | None:
    """Wait for `store`'s background refit, if any, and return its last outcome (None if none ran)."""
    with _REFIT_LOCK:
        job = _REFITS.get(_refit_key(store))
    if job is None or job.thread is None:
        return None
    job.thread.join(timeout)
    return job.outcome


def status_report(store: LiteStore) -> dict[str, object]:
    """`recall_calibration_status` for a lite store: the newest calibration and what it means now."""
    resolution = resolve(store)
    raw = store.latest_calibration_json()
    latest = artifact_from_json(raw) if raw is not None else None
    report: dict[str, object] = {
        "tenant": store.tenant,
        "store": "lite",
        "generation_id": store.lite_generation_id(),
        "status": resolution.status.value,
    }
    if latest is None:
        report["message"] = "No calibration yet: indexing fits one once memory holds enough text to test itself."
        return report
    carried = carry_forward(store, latest) if latest.corpus_fingerprint != store.corpus_fingerprint() else None
    try:
        coverage_report = store.near_miss_coverage(latest.calibration_id)
        coverage_message = describe_near_miss(coverage_report) if coverage_report else _COVERAGE_UNAVAILABLE
    except Exception as exc:  # BROAD-CATCH: fail-open  # optional report, never a reason to fail status
        _log.warning("Optional near-miss coverage unavailable (%s)", type(exc).__name__)
        coverage_report = None
        coverage_message = _COVERAGE_UNAVAILABLE
    if resolution.status is CalibrationStatus.STALE:
        message = (
            "The memory changed since this calibration was fitted and its threshold is not carried forward "
            f"({carried.reason if carried else 'no carry was possible'}); recall_index or "
            "recall_calibration_run refits it."
        )
    elif carried is not None and carried.artifact is not None:
        message = f"The memory changed since this calibration was fitted; {carried.reason}."
    else:
        message = str(latest.certification_reason or "") + "; " + coverage_message
    report.update(
        {
            "calibration_id": latest.calibration_id,
            "threshold": latest.threshold,
            "separability": latest.separability,
            "separability_ci": list(latest.separability_ci),
            "n_answerable": latest.n_answerable,
            "n_unanswerable": latest.n_unanswerable,
            "certified": latest.certified,
            "created_at": latest.created_at,
            "fitted_generation_id": latest.generation_id,
            "near_miss_coverage": coverage_report,
            "message": message,
        }
    )
    if carried is not None and carried.artifact is not None:
        keys = (
            "corpus_delta",
            "sources_added",
            "sources_removed",
            "sources_modified",
            "false_abstain_rate",
            "false_confirm_rate",
            "refit_due",
            "refit_threshold",
        )
        report["carried_forward"] = {
            "calibration_id": carried.artifact.calibration_id,
            **{key: carried.provenance.get(key) for key in keys},
        }
    return report


def run_report(
    store: LiteStore,
    embedder: Embedder,
    generation_id: str | None = None,
    queries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """`recall_calibration_run` for a lite store: fit now, store it, and report.

    A lite calibration that certifies is in force at once; there is no draft to publish, because
    there is one corpus and nothing else it could be bound to.
    """
    if generation_id is not None and generation_id != store.lite_generation_id():
        raise LiteStoreError(
            f"generation {generation_id!r} is not this lite store's corpus ({store.lite_generation_id()}); "
            "a lite store calibrates the corpus it holds now"
        )
    outcome = auto_calibrate(store, embedder, queries=queries)
    return {**status_report(store), "outcome": outcome.status.value, "message": outcome.reason}


def corpus_digest(rows: list[tuple[str, str, bytes]]) -> str:
    """sha256 over ids, text hashes and stored vector hashes, sorted by id."""
    outer = hashlib.sha256()
    for cid, text, embedding in sorted(rows):
        outer.update(cid.encode("utf-8"))
        outer.update(b"\x00")
        outer.update(hashlib.sha256(text.encode("utf-8")).digest())
        outer.update(hashlib.sha256(embedding).digest())
    return outer.hexdigest()
