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
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from recall.calibration import from_samples
from recall.calibration_v2 import (
    CalibrationArtifactV2,
    CalibrationResolution,
    CalibrationStatus,
    canonical_query_set,
    write_calibration_bundle,
)
from recall.eval.calibrate import measure_top_cosines
from recall.lineage import canonical_json, canonical_sha256
from recall.lite.store import LiteStoreError

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
    """Fit a calibration to the corpus as it is now and store it, certified or not, refusing
    publication if the corpus changes during the fit.

    `queries` is a labelled set (`{"query", "answerable"}` entries) to fit on instead of the
    questions generated from the corpus; it is checked and canonicalised like any stored set.
    """
    revision = store.corpus_revision()
    fingerprint = store.corpus_fingerprint()
    generation_id = store.lite_generation_id()
    texts = [chunk.text for chunk in store.iter_chunks() if chunk.text.strip()]
    entries, why = ([dict(q) for q in queries], "") if queries is not None else _query_set(texts)
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
    store.save_calibration(
        artifact_to_json(artifact), created_at=artifact.created_at, model=model, dimension=dimension,
        expected_revision=revision,
    )
    if certified:
        return CalibrationOutcome(
            CalibrationStatus.CERTIFIED,
            f"certified on {len(answerable)} answerable and {len(unanswerable)} unanswerable test questions "
            f"(separability {runtime.separability:.3f})",
            artifact,
        )
    return CalibrationOutcome(CalibrationStatus.UNCERTIFIED, f"calibration did not certify: {runtime.certification_reason}")


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
            "message": (
                "The memory changed since this calibration was fitted; recall_index or "
                "recall_calibration_run refits it."
                if resolution.status is CalibrationStatus.STALE
                else str(latest.certification_reason or "")
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


def corpus_digest(rows: list[tuple[str, str, bytes]]) -> str:
    """sha256 over ids, text hashes and stored vector hashes, sorted by id."""
    outer = hashlib.sha256()
    for cid, text, embedding in sorted(rows):
        outer.update(cid.encode("utf-8"))
        outer.update(b"\x00")
        outer.update(hashlib.sha256(text.encode("utf-8")).digest())
        outer.update(hashlib.sha256(embedding).digest())
    return outer.hexdigest()


# ---------------------------------------------------------------------- the `recall calibration` CLI


def _stored_artifacts(
    store: LiteStore, problems: list[str] | None = None
) -> tuple[list[CalibrationArtifactV2], bool]:
    """Every stored calibration that verifies, in the order the store serves them, and whether the
    row the store serves is among them.

    The order is `LiteStore.calibration_rows`', so the first calibration listed is the one search
    uses, even when two share a `created_at`. A row that does not parse or verify is left out and
    described in `problems`, so one bad old row cannot hide the others. When the bad row is the one
    the store serves, search refuses until the store is recalibrated, and the caller must not
    report any other calibration as in force.
    """
    from recall.calibration_v2 import CalibrationError

    rows = store.calibration_rows()
    served_rowid = rows[0][0] if rows else None
    served_verifies = True
    artifacts: list[CalibrationArtifactV2] = []
    for rowid, payload in rows:
        try:
            artifacts.append(artifact_from_json(str(payload)))
        except (CalibrationError, ValueError, KeyError, TypeError) as exc:
            if rowid == served_rowid:
                served_verifies = False
            if problems is not None:
                what = "the calibration this store serves" if rowid == served_rowid else "an older calibration"
                problems.append(
                    f"calibration row {rowid} ({what}) does not verify and was skipped: {type(exc).__name__}: {exc}"
                    + ("; searches refuse until the store is recalibrated (recall index)" if rowid == served_rowid else "")
                )
    return artifacts, served_verifies


def _one_artifact(store: LiteStore, calibration_id: str) -> tuple[CalibrationArtifactV2, str]:
    """The stored calibration `calibration_id` and its state now, without parsing the others.

    It is in force when it is the row the store serves; a `published` one that is not has been
    replaced, so it reads `superseded`, as in the listing.
    """
    from recall.calibration_v2 import CalibrationError

    served = store.calibration_rows(limit=1)
    for rowid, payload in store.calibration_rows(calibration_id=calibration_id):
        try:
            artifact = artifact_from_json(payload)
        except (CalibrationError, ValueError, KeyError, TypeError):
            continue
        if artifact.calibration_id != calibration_id:
            continue
        in_force = bool(served) and served[0][0] == rowid
        state = artifact.lifecycle_state
        return artifact, "superseded" if state == "published" and not in_force else state
    problems: list[str] = []
    _stored_artifacts(store, problems)  # only to say which rows did not verify
    raise _not_found(calibration_id, problems)


def _lifecycle_states(artifacts: Sequence[CalibrationArtifactV2], served_verifies: bool = True) -> dict[str, str]:
    """The state each calibration is in now: a lite store serves only its newest one.

    The stored artifact says `published` for every certified calibration a lite store saved (and
    `rejected` for one that did not certify), because it was in force when it was saved. A later
    one replaced it, so it is reported `superseded`, which is what PostgreSQL's
    `recall_calibrations.lifecycle_state` says of a replaced calibration. When the row the store
    serves does not verify, none of these is in force, so none is reported `published`.
    """
    states: dict[str, str] = {}
    for index, artifact in enumerate(artifacts):
        state = artifact.lifecycle_state
        replaced = index > 0 or not served_verifies
        states[artifact.calibration_id] = "superseded" if replaced and state == "published" else state
    return states


def list_records(store: LiteStore, problems: list[str] | None = None) -> list[dict[str, Any]]:
    """`recall calibration list` on a lite store, with `CalibrationRepository.list_records`' keys."""
    artifacts, served_verifies = _stored_artifacts(store, problems)
    states = _lifecycle_states(artifacts, served_verifies)
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        carry = artifact.carry_forward if isinstance(artifact.carry_forward, Mapping) else None
        records.append(
            {
                "calibration_id": artifact.calibration_id,
                "generation_id": artifact.generation_id,
                "lifecycle_state": states[artifact.calibration_id],
                "certified": bool(artifact.certified),
                "pipeline_fingerprint": artifact.pipeline_fingerprint,
                "corpus_fingerprint": artifact.corpus_fingerprint,
                "query_set_digest": artifact.query_set_digest,
                "created_at": artifact.created_at,
                "threshold_was_measured_here": carry is None,
                "carried_forward_from": carry.get("parent_calibration_id") if carry else None,
                "corpus_delta": carry.get("corpus_delta") if carry else None,
            }
        )
    return records


def show_record(store: LiteStore, calibration_id: str) -> dict[str, Any]:
    """`recall calibration show` on a lite store: the stored artifact, with its state now.

    PostgreSQL prints its `recall_calibrations` row; a lite store has no such table, so this is the
    artifact itself (the same facts, in the artifact's own field names).
    """
    artifact, state = _one_artifact(store, calibration_id)
    return {**artifact.to_dict(), "lifecycle_state": state}


def export_artifact(store: LiteStore, calibration_id: str, path: str | Path) -> Path:
    """`recall calibration export` on a lite store: the artifact alone, flagged as such.

    A PostgreSQL bundle carries the labelled query set; a lite store never kept one (it fits on
    questions it generates and keeps only their digest), so the bundle says `artifact_only` rather
    than shipping a `query_set` it does not have. The artifact carries its state now, as
    PostgreSQL's export does: a replaced calibration exports `superseded`, not the `published` it
    was saved with (the state is outside the artifact's checksum).
    """
    artifact, state = _one_artifact(store, calibration_id)
    return write_calibration_bundle(
        {"artifact": {**artifact.to_dict(), "lifecycle_state": state}, "artifact_only": True}, path
    )


def _not_found(calibration_id: str, problems: Sequence[str]) -> Exception:
    """`CalibrationNotFound`, saying when stored rows that do not verify may hold the id."""
    from recall.calibration_v2 import CalibrationNotFound

    if not problems:
        return CalibrationNotFound(calibration_id)
    return CalibrationNotFound(
        f"calibration {calibration_id} is not among the calibrations that verify; "
        f"{len(problems)} stored row(s) do not verify: " + "; ".join(problems)
    )
