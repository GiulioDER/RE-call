"""Tenant and generation bound calibration artifacts for the v1 index path."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from recall.calibration import Calibration, from_samples
from recall.embeddings import Embedder
from recall.lineage import PipelineIdentity, canonical_json, canonical_sha256

ARTIFACT_VERSION = 2


class CalibrationStatus(StrEnum):
    CERTIFIED = "certified"
    MISSING = "missing"
    STALE = "stale"
    UNCERTIFIED = "uncertified"
    DRAFT = "draft"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    LEGACY_UNBOUND = "legacy_unbound"


class CalibrationError(RuntimeError):
    pass


class CalibrationNotFound(CalibrationError):
    pass


class CalibrationUncertified(CalibrationError):
    pass


class CalibrationBindingError(CalibrationError):
    pass


def _frozen_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _frozen_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json(item) for item in value)
    return value


def _utc_isoformat(value: Any) -> str:
    """Render a stored timestamp as the UTC ISO string the checksum was computed over.

    `created_at` is part of `immutable_payload()`, so its *string* form is inside the
    artifact checksum, but the column is `timestamptz` and psycopg decodes it in the
    connection's TimeZone. Rendering it as-is makes the recomputed digest depend on the
    session's `TimeZone`, so an artifact written by one session fails `verify_checksum()`
    in another. `calibrate()` always writes `datetime.now(UTC).isoformat()`, so
    normalising back to UTC on read restores the exact bytes that were hashed.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def _require_utc_isoformat(value: str, field_name: str) -> None:
    """Refuse a timestamp the `timestamptz` round trip would not return unchanged.

    The column keeps the instant and discards the rendering, while the *string* is what the
    artifact checksum covers. Any other valid ISO-8601 spelling of the same instant (a non-UTC
    offset, or the `Z`-plus-milliseconds form a JavaScript producer emits) would therefore
    store cleanly and then fail `verify_checksum()` on every later read. Enforcing the
    canonical form at construction keeps that failure at the boundary, where it names the
    cause, instead of committing a row nothing can read back.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise CalibrationBindingError(f"{field_name} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        # `replace(tzinfo=UTC)`, not `astimezone()`: astimezone would read a naive value as
        # machine-local, naming a different instant on every host. Not string concatenation
        # either: `fromisoformat` accepts spellings the canonical form does not (a space
        # separator, minute precision, milliseconds, a bare date), and appending an offset to
        # those produced advice this very guard then rejected. `str()` on a naive datetime
        # yields the space-separated form, so that was the likely shape to hit it.
        raise CalibrationBindingError(
            f"{field_name} must carry a UTC offset "
            f"(e.g. {parsed.replace(tzinfo=UTC).isoformat()!r}); re-export the bundle"
        )
    try:
        canonical = parsed.astimezone(UTC).isoformat()
    except (OverflowError, OSError):
        # astimezone() is not total: it overflows near datetime.min/max. Rendering a value
        # already judged unusable would turn this boundary check into an unnamed crash that
        # escapes the CLI's `except CalibrationError`.
        raise CalibrationBindingError(
            f"{field_name} is not a representable UTC instant"
        ) from None
    if canonical != value:
        raise CalibrationBindingError(
            f"{field_name} must be UTC in the form written by calibrate() "
            f"(e.g. {canonical!r}); re-export the bundle"
        )


def _require_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CalibrationBindingError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CalibrationArtifactV2:
    calibration_id: str
    tenant_id: str
    generation_id: str
    embedder_identity: Mapping[str, Any]
    pipeline_fingerprint: str
    corpus_fingerprint: str
    query_set_digest: str
    threshold: float
    scale: float
    separability: float
    separability_ci: tuple[float, float]
    n_answerable: int
    n_unanswerable: int
    certified: bool
    certification_reason: str
    lifecycle_state: str
    created_at: str
    created_by: str
    scores: Mapping[str, Any]
    checksum: str
    artifact_version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.artifact_version != ARTIFACT_VERSION:
            raise CalibrationBindingError(
                f"unsupported calibration artifact version {self.artifact_version}"
            )
        for field_name in (
            "calibration_id",
            "tenant_id",
            "generation_id",
            "created_at",
            "created_by",
            "certification_reason",
        ):
            if not str(getattr(self, field_name)).strip():
                raise CalibrationBindingError(f"{field_name} must be non-empty")
        for field_name in (
            "pipeline_fingerprint",
            "corpus_fingerprint",
            "query_set_digest",
            "checksum",
        ):
            _require_digest(str(getattr(self, field_name)), field_name)
        _require_utc_isoformat(str(self.created_at), "created_at")
        numeric = (self.threshold, self.scale, self.separability, *self.separability_ci)
        if not all(math.isfinite(value) for value in numeric):
            raise CalibrationBindingError("calibration statistics must be finite")
        if not -1.0 <= self.threshold <= 1.0 or self.scale <= 0:
            raise CalibrationBindingError("calibration threshold or scale is invalid")
        low, high = self.separability_ci
        if not 0.0 <= self.separability <= 1.0 or not 0.0 <= low <= high <= 1.0:
            raise CalibrationBindingError("calibration separability interval is invalid")
        if self.n_answerable < 0 or self.n_unanswerable < 0:
            raise CalibrationBindingError("calibration sample counts cannot be negative")
        allowed_states = {"draft", "published", "superseded", "rejected"}
        if self.lifecycle_state not in allowed_states:
            raise CalibrationBindingError(
                f"invalid bound calibration lifecycle state {self.lifecycle_state!r}"
            )
        if self.certified != (self.lifecycle_state in {"draft", "published", "superseded"}):
            raise CalibrationBindingError(
                "certified flag is inconsistent with calibration lifecycle state"
            )
        canonical_json(self.embedder_identity)
        canonical_json(self.scores)
        object.__setattr__(self, "embedder_identity", _frozen_json(self.embedder_identity))
        object.__setattr__(self, "scores", _frozen_json(self.scores))

    @property
    def status(self) -> CalibrationStatus:
        if self.lifecycle_state == "published" and self.certified:
            return CalibrationStatus.CERTIFIED
        if self.lifecycle_state == "rejected" or not self.certified:
            return CalibrationStatus.REJECTED
        if self.lifecycle_state == "superseded":
            return CalibrationStatus.SUPERSEDED
        return CalibrationStatus.DRAFT

    @property
    def runtime(self) -> Calibration:
        model = str(self.embedder_identity.get("model", "bound-v2"))
        return Calibration(
            embedder=model,
            threshold=self.threshold,
            scale=self.scale,
            separability=self.separability,
            n_answerable=self.n_answerable,
            n_unanswerable=self.n_unanswerable,
        )

    def immutable_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "calibration_id": self.calibration_id,
            "tenant_id": self.tenant_id,
            "generation_id": self.generation_id,
            "embedder_identity": dict(self.embedder_identity),
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "query_set_digest": self.query_set_digest,
            "threshold": self.threshold,
            "scale": self.scale,
            "separability": self.separability,
            "separability_ci": list(self.separability_ci),
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "certified": self.certified,
            "certification_reason": self.certification_reason,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "scores": dict(self.scores),
        }

    def verify_checksum(self) -> None:
        actual = canonical_sha256(self.immutable_payload())
        if actual != self.checksum:
            raise CalibrationBindingError(
                f"calibration checksum mismatch: expected {self.checksum}, computed {actual}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.immutable_payload(),
            "lifecycle_state": self.lifecycle_state,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class CalibrationResolution:
    status: CalibrationStatus
    artifact: CalibrationArtifactV2 | None = None


def canonical_query_set(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    normalised: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        query = entry.get("query")
        answerable = entry.get("answerable")
        if not isinstance(query, str) or not query.strip() or not isinstance(answerable, bool):
            raise CalibrationBindingError(
                f"query set entry {index} requires a non-empty query and boolean answerable"
            )
        item: dict[str, Any] = {"query": query, "answerable": answerable}
        if "relevant_ids" in entry:
            relevant = entry["relevant_ids"]
            if not isinstance(relevant, list) or not all(isinstance(v, str) for v in relevant):
                raise CalibrationBindingError(f"query set entry {index} has invalid relevant_ids")
            item["relevant_ids"] = sorted(set(relevant))
        normalised.append(item)
    if not normalised:
        raise CalibrationBindingError("query set must not be empty")
    normalised.sort(key=canonical_json)
    keys = [canonical_json(item) for item in normalised]
    if len(keys) != len(set(keys)):
        raise CalibrationBindingError("query set contains a duplicate labelled query")
    value = tuple(normalised)
    return value, canonical_sha256(list(value))


def load_query_set(path: str | Path) -> tuple[tuple[dict[str, Any], ...], str]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationBindingError(f"cannot read labelled query set {path}: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise CalibrationBindingError("labelled query set must be a JSON array of objects")
    return canonical_query_set(raw)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class CalibrationRepository:
    def __init__(self, dsn: str, tenant_id: str, *, actor: str = "recall-cli") -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        self.dsn = dsn
        self.tenant_id = tenant_id
        self.actor = actor

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.dsn, autocommit=True, connect_timeout=10) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
            yield conn

    def _audit(
        self,
        conn: psycopg.Connection,
        event_type: str,
        calibration_id: str,
        generation_id: str | None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO recall_audit_events "
            "(tenant_id, event_id, event_type, actor, generation_id, payload) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                self.tenant_id,
                _id("evt"),
                event_type,
                self.actor,
                generation_id,
                Jsonb({"calibration_id": calibration_id, **dict(payload or {})}),
            ),
        )

    def _generation(
        self, conn: psycopg.Connection, generation_id: str
    ) -> tuple[str, dict[str, Any], str, str]:
        row = conn.execute(
            "SELECT state, pipeline_identity, pipeline_fingerprint, corpus_fingerprint "
            "FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None:
            raise CalibrationBindingError(f"generation {generation_id!r} does not exist")
        if str(row[0]) not in {"ready", "active", "retired"}:
            raise CalibrationBindingError(
                f"generation {generation_id!r} is {row[0]!r}, expected ready, active, or retired"
            )
        if not isinstance(row[1], Mapping):
            raise CalibrationBindingError("stored pipeline identity is malformed")
        return str(row[0]), dict(row[1]), str(row[2]), str(row[3])

    @staticmethod
    def _artifact(row: tuple[Any, ...]) -> CalibrationArtifactV2:
        identity = row[3] if isinstance(row[3], Mapping) else {}
        scores = row[17] if isinstance(row[17], Mapping) else {}
        artifact = CalibrationArtifactV2(
            calibration_id=str(row[0]),
            tenant_id=str(row[1]),
            generation_id=str(row[2]),
            embedder_identity=dict(identity),
            pipeline_fingerprint=str(row[4]),
            corpus_fingerprint=str(row[5]),
            query_set_digest=str(row[6]),
            threshold=float(row[7]),
            scale=float(row[8]),
            separability=float(row[9]),
            separability_ci=(float(row[10]), float(row[11])),
            n_answerable=int(row[12]),
            n_unanswerable=int(row[13]),
            certified=bool(row[14]),
            certification_reason=str(row[15]),
            lifecycle_state=str(row[16]),
            scores=dict(scores),
            created_at=_utc_isoformat(row[18]),
            created_by=str(row[19]),
            checksum=str(row[20]),
        )
        artifact.verify_checksum()
        return artifact

    _COLUMNS = (
        "calibration_id, tenant_id, generation_id, embedder_identity, pipeline_fingerprint, "
        "corpus_fingerprint, query_set_digest, threshold, scale, separability, ci_low, ci_high, "
        "n_answerable, n_unanswerable, certified, certification_reason, lifecycle_state, scores, "
        "created_at, created_by, artifact_checksum"
    )

    def calibrate(
        self,
        generation_id: str,
        queries: Sequence[Mapping[str, Any]],
        embedder: Embedder,
    ) -> CalibrationArtifactV2:
        labels, query_digest = canonical_query_set(queries)
        with self._connect() as conn:
            _state, pipeline_raw, pipeline_fingerprint, corpus_fingerprint = self._generation(
                conn, generation_id
            )
        pipeline = PipelineIdentity.from_dict(pipeline_raw)
        if embedder.name != pipeline.embedder.model or embedder.dim != pipeline.embedder.dimension:
            raise CalibrationBindingError(
                "embedder implementation does not match the generation pipeline identity"
            )

        from recall.eval.calibrate import measure_top_cosines
        from recall.generation_store import GenerationStore

        store = GenerationStore(self.dsn, embedder.dim, tenant=self.tenant_id)
        try:
            store.check_schema()
            with store.pin_generation(generation_id):
                answerable, unanswerable = measure_top_cosines(store, embedder, list(labels))
        finally:
            store.close()
        runtime = from_samples(embedder.name, answerable, unanswerable)
        if runtime.separability is None or runtime.separability_ci is None:
            raise CalibrationBindingError("both labelled classes are required for calibration")
        certified = runtime.certified is True
        calibration_id = _id("cal")
        created_at = datetime.now(UTC).isoformat()
        scores = {"answerable": answerable, "unanswerable": unanswerable}
        immutable = {
            "artifact_version": ARTIFACT_VERSION,
            "calibration_id": calibration_id,
            "tenant_id": self.tenant_id,
            "generation_id": generation_id,
            "embedder_identity": pipeline.embedder.to_dict(),
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "query_set_digest": query_digest,
            "threshold": runtime.threshold,
            "scale": runtime.scale,
            "separability": runtime.separability,
            "separability_ci": list(runtime.separability_ci),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "certified": certified,
            "certification_reason": runtime.certification_reason,
            "created_at": created_at,
            "created_by": self.actor,
            "scores": scores,
        }
        checksum = canonical_sha256(immutable)
        lifecycle = "draft" if certified else "rejected"
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibration_query_sets "
                "(tenant_id, query_set_digest, queries, sample_count, created_by) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.tenant_id, query_digest, Jsonb(list(labels)), len(labels), self.actor),
            )
            conn.execute(
                "INSERT INTO recall_calibrations "
                "(tenant_id, calibration_id, generation_id, embedder_identity, "
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, threshold, scale, "
                "separability, ci_low, ci_high, n_answerable, n_unanswerable, certified, "
                "certification_reason, lifecycle_state, scores, created_at, created_by, "
                "artifact_checksum) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.tenant_id,
                    calibration_id,
                    generation_id,
                    Jsonb(pipeline.embedder.to_dict()),
                    pipeline_fingerprint,
                    corpus_fingerprint,
                    query_digest,
                    runtime.threshold,
                    runtime.scale,
                    runtime.separability,
                    runtime.separability_ci[0],
                    runtime.separability_ci[1],
                    len(answerable),
                    len(unanswerable),
                    certified,
                    runtime.certification_reason,
                    lifecycle,
                    Jsonb(scores),
                    created_at,
                    self.actor,
                    checksum,
                ),
            )
            self._audit(conn, "calibration_created", calibration_id, generation_id)
            if not certified:
                self._audit(
                    conn,
                    "calibration_rejected",
                    calibration_id,
                    generation_id,
                    {"reason": runtime.certification_reason},
                )
        return self.get(calibration_id)

    def get(self, calibration_id: str) -> CalibrationArtifactV2:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM recall_calibrations "
                "WHERE tenant_id = %s AND calibration_id = %s AND generation_id IS NOT NULL",
                (self.tenant_id, calibration_id),
            ).fetchone()
        if row is None:
            raise CalibrationNotFound(calibration_id)
        return self._artifact(row)

    def publish(self, calibration_id: str) -> CalibrationArtifactV2:
        artifact = self.get(calibration_id)
        if not artifact.certified:
            raise CalibrationUncertified(artifact.certification_reason)
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"calibration\x1f{self.tenant_id}\x1f{artifact.generation_id}",),
            )
            _state, _identity, pipeline, corpus = self._generation(conn, artifact.generation_id)
            if pipeline != artifact.pipeline_fingerprint or corpus != artifact.corpus_fingerprint:
                raise CalibrationBindingError("generation lineage changed after calibration")
            old = conn.execute(
                "SELECT calibration_id FROM recall_calibrations WHERE tenant_id = %s "
                "AND generation_id = %s AND lifecycle_state = 'published' FOR UPDATE",
                (self.tenant_id, artifact.generation_id),
            ).fetchall()
            for row in old:
                old_id = str(row[0])
                if old_id == calibration_id:
                    continue
                conn.execute(
                    "UPDATE recall_calibrations SET lifecycle_state = 'superseded', "
                    "superseded_at = clock_timestamp() WHERE tenant_id = %s "
                    "AND calibration_id = %s",
                    (self.tenant_id, old_id),
                )
                self._audit(
                    conn,
                    "calibration_superseded",
                    old_id,
                    artifact.generation_id,
                    {"superseded_by": calibration_id},
                )
            conn.execute(
                "UPDATE recall_calibrations SET lifecycle_state = 'published', "
                "published_at = clock_timestamp(), superseded_at = NULL WHERE tenant_id = %s "
                "AND calibration_id = %s "
                "AND lifecycle_state IN ('draft', 'published', 'superseded')",
                (self.tenant_id, calibration_id),
            )
            self._audit(conn, "calibration_published", calibration_id, artifact.generation_id)
        return self.get(calibration_id)

    def resolve(self, generation_id: str) -> CalibrationResolution:
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            _state, _identity, pipeline, corpus = self._generation(conn, generation_id)
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM recall_calibrations WHERE tenant_id = %s "
                "AND generation_id = %s AND lifecycle_state = 'published'",
                (self.tenant_id, generation_id),
            ).fetchone()
            if row is None:
                local = conn.execute(
                    "SELECT certified, lifecycle_state FROM recall_calibrations "
                    "WHERE tenant_id = %s AND generation_id = %s "
                    "ORDER BY created_at DESC, calibration_id DESC LIMIT 1",
                    (self.tenant_id, generation_id),
                ).fetchone()
                if local is not None and not bool(local[0]):
                    return CalibrationResolution(CalibrationStatus.UNCERTIFIED)
                if local is not None and str(local[1]) == "draft":
                    return CalibrationResolution(CalibrationStatus.DRAFT)
                any_published = conn.execute(
                    "SELECT 1 FROM recall_calibrations WHERE tenant_id = %s "
                    "AND lifecycle_state = 'published' LIMIT 1",
                    (self.tenant_id,),
                ).fetchone()
                return CalibrationResolution(
                    CalibrationStatus.STALE if any_published else CalibrationStatus.MISSING
                )
            artifact = self._artifact(row)
            query_row = conn.execute(
                "SELECT queries FROM recall_calibration_query_sets WHERE tenant_id = %s "
                "AND query_set_digest = %s",
                (self.tenant_id, artifact.query_set_digest),
            ).fetchone()
        if artifact.pipeline_fingerprint != pipeline or artifact.corpus_fingerprint != corpus:
            return CalibrationResolution(CalibrationStatus.STALE)
        if query_row is None or not isinstance(query_row[0], list):
            return CalibrationResolution(CalibrationStatus.STALE)
        _queries, digest = canonical_query_set(query_row[0])
        if digest != artifact.query_set_digest:
            return CalibrationResolution(CalibrationStatus.STALE)
        if not artifact.certified:
            return CalibrationResolution(CalibrationStatus.UNCERTIFIED)
        return CalibrationResolution(CalibrationStatus.CERTIFIED, artifact)

    def list_records(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT calibration_id, generation_id, lifecycle_state, certified, "
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, created_at "
                "FROM recall_calibrations WHERE tenant_id = %s "
                "ORDER BY created_at DESC, calibration_id",
                (self.tenant_id,),
            ).fetchall()
        return [
            {
                "calibration_id": str(row[0]),
                "generation_id": row[1],
                "lifecycle_state": str(row[2]),
                "certified": bool(row[3]),
                "pipeline_fingerprint": row[4],
                "corpus_fingerprint": row[5],
                "query_set_digest": row[6],
                "created_at": _utc_isoformat(row[7]),
            }
            for row in rows
        ]

    def show_record(self, calibration_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT to_jsonb(c) FROM recall_calibrations c "
                "WHERE tenant_id = %s AND calibration_id = %s",
                (self.tenant_id, calibration_id),
            ).fetchone()
        if row is None or not isinstance(row[0], Mapping):
            raise CalibrationNotFound(calibration_id)
        value = dict(row[0])
        for key, item in tuple(value.items()):
            if isinstance(item, datetime):
                value[key] = _utc_isoformat(item)
        # `to_jsonb` renders a timestamptz as a STRING already formatted in the session
        # TimeZone, so the loop above never sees it and `show` would otherwise disagree with
        # `list` about the same field. Re-render it as the checksummed form.
        created_at = value.get("created_at")
        if isinstance(created_at, str):
            value["created_at"] = _utc_isoformat(datetime.fromisoformat(created_at))
        return value

    def export_bundle(self, calibration_id: str, path: str | Path) -> Path:
        artifact = self.get(calibration_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT queries FROM recall_calibration_query_sets WHERE tenant_id = %s "
                "AND query_set_digest = %s",
                (self.tenant_id, artifact.query_set_digest),
            ).fetchone()
        if row is None:
            raise CalibrationBindingError("calibration query set is missing")
        payload = {"artifact": artifact.to_dict(), "query_set": row[0]}
        payload["bundle_checksum"] = canonical_sha256(payload)
        target = Path(path)
        target.write_bytes(canonical_json(payload) + b"\n")
        return target

    def import_bundle(self, path: str | Path) -> str:
        raw = Path(path).read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CalibrationBindingError(f"invalid calibration JSON: {exc}") from exc
        if not isinstance(data, Mapping) or "artifact" not in data:
            return self._import_legacy(data, raw)
        supplied = data.get("bundle_checksum")
        unsigned = {key: value for key, value in data.items() if key != "bundle_checksum"}
        if supplied != canonical_sha256(unsigned):
            raise CalibrationBindingError("calibration bundle checksum mismatch")
        artifact_raw = data.get("artifact")
        query_raw = data.get("query_set")
        if not isinstance(artifact_raw, Mapping) or not isinstance(query_raw, list):
            raise CalibrationBindingError("calibration bundle is malformed")
        if int(artifact_raw.get("artifact_version", 0)) != ARTIFACT_VERSION:
            raise CalibrationBindingError("unsupported calibration artifact version")
        labels, digest = canonical_query_set(query_raw)
        if digest != artifact_raw.get("query_set_digest"):
            raise CalibrationBindingError("imported query set digest does not match artifact")
        if artifact_raw.get("tenant_id") != self.tenant_id:
            raise CalibrationBindingError("imported calibration belongs to another tenant")
        generation_id = str(artifact_raw.get("generation_id", ""))
        with self._connect() as conn:
            _state, identity, pipeline, corpus = self._generation(conn, generation_id)
        if (
            artifact_raw.get("pipeline_fingerprint") != pipeline
            or artifact_raw.get("corpus_fingerprint") != corpus
        ):
            raise CalibrationBindingError("imported calibration does not match generation lineage")
        immutable_keys = CalibrationArtifactV2.__dataclass_fields__.keys()
        values = {key: artifact_raw[key] for key in immutable_keys if key in artifact_raw}
        values["separability_ci"] = tuple(values["separability_ci"])
        values["lifecycle_state"] = "draft" if values.get("certified") else "rejected"
        artifact = CalibrationArtifactV2(**values)
        artifact.verify_checksum()
        generation_embedder = identity.get("embedder")
        if not isinstance(generation_embedder, Mapping) or canonical_sha256(
            artifact.embedder_identity
        ) != canonical_sha256(generation_embedder):
            raise CalibrationBindingError("imported calibration embedder identity does not match")
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibration_query_sets "
                "(tenant_id, query_set_digest, queries, sample_count, created_by) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.tenant_id, digest, Jsonb(list(labels)), len(labels), self.actor),
            )
            conn.execute(
                "INSERT INTO recall_calibrations "
                "(tenant_id, calibration_id, generation_id, embedder_identity, "
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, threshold, scale, "
                "separability, ci_low, ci_high, n_answerable, n_unanswerable, certified, "
                "certification_reason, lifecycle_state, scores, created_at, created_by, "
                "artifact_checksum) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    self.tenant_id,
                    artifact.calibration_id,
                    generation_id,
                    Jsonb(dict(identity["embedder"])),
                    artifact.pipeline_fingerprint,
                    artifact.corpus_fingerprint,
                    digest,
                    artifact.threshold,
                    artifact.scale,
                    artifact.separability,
                    artifact.separability_ci[0],
                    artifact.separability_ci[1],
                    artifact.n_answerable,
                    artifact.n_unanswerable,
                    artifact.certified,
                    artifact.certification_reason,
                    artifact.lifecycle_state,
                    Jsonb(dict(artifact.scores)),
                    artifact.created_at,
                    artifact.created_by,
                    artifact.checksum,
                ),
            )
            self._audit(conn, "calibration_imported", artifact.calibration_id, generation_id)
        return artifact.calibration_id

    def _import_legacy(self, data: Any, raw: bytes) -> str:
        if not isinstance(data, Mapping):
            raise CalibrationBindingError("legacy calibration JSON must be an object")
        try:
            threshold = float(data["threshold"])
            scale = float(data["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationBindingError("legacy calibration is malformed") from exc
        if not math.isfinite(threshold) or not math.isfinite(scale) or scale <= 0:
            raise CalibrationBindingError("legacy calibration contains invalid numeric values")
        calibration_id = _id("legacy_cal")
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibrations "
                "(tenant_id, calibration_id, embedder_identity, threshold, scale, certified, "
                "certification_reason, lifecycle_state, scores, created_by, artifact_checksum) "
                "VALUES (%s,%s,%s,%s,%s,false,%s,'legacy_unbound','{}'::jsonb,%s,%s)",
                (
                    self.tenant_id,
                    calibration_id,
                    Jsonb({"model": str(data.get("embedder", "unknown"))}),
                    threshold,
                    scale,
                    "legacy artifact has no tenant, generation, pipeline, corpus, or query binding",
                    self.actor,
                    hashlib.sha256(raw).hexdigest(),
                ),
            )
            self._audit(conn, "calibration_imported_legacy_unbound", calibration_id, None)
        return calibration_id
