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
corpus fingerprint is a digest of the stored chunks (ids and text) rather than a generation's.
A calibration fitted to one corpus reads `stale` as soon as the corpus changes; `ensure_calibrated`
then fits a new one.

A corpus too small to produce a query set (about 21 chunks, memo `the-certifying-floor-is-21-chunks`)
has no calibration at all and resolves `missing`, with the reason in plain words.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from recall.calibration import from_samples
from recall.calibration_v2 import (
    CalibrationArtifactV2,
    CalibrationResolution,
    CalibrationStatus,
    canonical_query_set,
)
from recall.eval.calibrate import measure_top_cosines
from recall.lineage import canonical_json, canonical_sha256
from recall.lite.store import LiteStoreError
from recall.near_miss import describe as describe_near_miss
from recall.near_miss import near_miss_coverage, near_miss_probes

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.store import LiteStore

CREATED_BY = "recall-lite"
_PLACEHOLDER_CHECKSUM = "0" * 64


@dataclass(frozen=True)
class CalibrationOutcome:
    """What `auto_calibrate` did: the status the store now resolves to, and why in plain words."""

    status: CalibrationStatus
    reason: str
    artifact: CalibrationArtifactV2 | None = None


def pipeline_fingerprint(model: str, dimension: int) -> str:
    """The lite pipeline's identity: the store kind and the embedder that fills it."""
    return canonical_sha256({"store": "recall-lite", "embedder": {"model": model, "dimension": dimension}})


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


def resolve(store: LiteStore) -> CalibrationResolution:
    """The newest calibration, judged against the corpus as it is now.

    `missing` when none was ever stored; `stale` when the corpus changed since it was fitted;
    `uncertified` when it was fitted to this corpus and did not separate; else `certified`.
    Only a certified resolution carries the artifact, so no other state can lend its threshold.
    """
    raw = store.latest_calibration_json()
    if raw is None:
        return CalibrationResolution(CalibrationStatus.MISSING)
    artifact = artifact_from_json(raw)
    if artifact.corpus_fingerprint != store.corpus_fingerprint():
        return CalibrationResolution(CalibrationStatus.STALE)
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
    """Fit a calibration to the corpus as it is now and store it, certified or not.

    `queries` is a labelled set (`{"query", "answerable"}` entries) to fit on instead of the
    questions generated from the corpus; it is checked and canonicalised like any stored set.
    """
    texts = [chunk.text for chunk in store.iter_chunks() if chunk.text.strip()]
    entries, why = ([dict(q) for q in queries], "") if queries else _query_set(texts)
    if entries is None:
        return CalibrationOutcome(
            CalibrationStatus.MISSING,
            f"Trusted answers turn on once memory holds enough text to test itself: {len(texts)} "
            f"chunk(s) now ({why}). Add a few more memos.",
        )
    labels, query_digest = canonical_query_set(entries)
    # Duck-typed as everywhere else: the scorer needs only `top_cosine`, which LiteStore has.
    answerable, unanswerable = measure_top_cosines(cast("Any", store), embedder, list(labels))
    runtime = from_samples(str(embedder.name), answerable, unanswerable)
    if runtime.separability is None or runtime.separability_ci is None:
        return CalibrationOutcome(CalibrationStatus.UNCERTIFIED, "both kinds of test question are needed to calibrate")
    certified = runtime.certified is True
    fingerprint = store.corpus_fingerprint()
    model, dimension = str(embedder.name), int(embedder.dim)
    draft = CalibrationArtifactV2(
        calibration_id="cal_" + uuid.uuid4().hex,
        tenant_id=store.tenant,
        generation_id=store.lite_generation_id(),
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
    store.save_calibration(artifact_to_json(artifact), created_at=artifact.created_at, model=model, dimension=dimension)
    coverage = ""
    if runtime.threshold is not None:
        # Report only: measured with the threshold just fitted, never fed back into the fit or the
        # certified decision. See `recall.near_miss` for why the two are kept apart.
        asked = [str(e["query"]) for e in labels if e.get("answerable") is True]
        probes = near_miss_probes(texts, asked, per_class=len(asked))
        report = near_miss_coverage(store, embedder, float(runtime.threshold), probes)
        store.save_near_miss_coverage(artifact.calibration_id, report)
        coverage = "; " + describe_near_miss(report)
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


def ensure_calibrated(store: LiteStore, embedder: Embedder) -> CalibrationOutcome:
    """Calibrate when the corpus changed since the last fit, or was never fitted; else keep it.

    A fit that did not certify is not retried on the same corpus: the query set and the
    measurement are deterministic, so a second run would give the same answer.
    """
    current = resolve(store)
    if current.status is CalibrationStatus.CERTIFIED:
        return CalibrationOutcome(current.status, "already certified for this corpus", current.artifact)
    if current.status is CalibrationStatus.UNCERTIFIED:
        return CalibrationOutcome(current.status, "this corpus was calibrated and did not certify; it is refitted when it changes")
    return auto_calibrate(store, embedder)


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
            "near_miss_coverage": store.near_miss_coverage(latest.calibration_id),
            "message": (
                "The memory changed since this calibration was fitted; recall_index or "
                "recall_calibration_run refits it."
                if resolution.status is CalibrationStatus.STALE
                else str(latest.certification_reason or "")
                + (
                    "; " + describe_near_miss(coverage_report)
                    if (coverage_report := store.near_miss_coverage(latest.calibration_id))
                    else ""
                )
            ),
        }
    )
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


def corpus_digest(rows: list[tuple[str, str]]) -> str:
    """sha256 over (chunk id, sha256 of text), sorted by id: the corpus as search sees it."""
    outer = hashlib.sha256()
    for cid, text in sorted(rows):
        outer.update(cid.encode("utf-8"))
        outer.update(b"\x00")
        outer.update(hashlib.sha256(text.encode("utf-8")).digest())
    return outer.hexdigest()
