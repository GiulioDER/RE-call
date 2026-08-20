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

from recall.calibration import Calibration, from_samples, separability
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


#: Largest corpus delta a threshold may be carried across without being refitted, as a fraction of
#: the union of parent and child source URIs.
#:
#: 0.25 is a CEILING on the mechanism, not a claim that a threshold survives a 25% delta. Nothing
#: is carried on the strength of this number alone: the stored labelled query set is re-scored
#: against the child generation and must still certify, so the bound only decides how far the
#: mechanism will bother trying before demanding a fresh fit. It exists because re-scoring says
#: nothing about the queries nobody labelled, and a query set fitted on a corpus half of which has
#: since been replaced is measuring a different corpus while wearing the same digest.
#:
#: ⚠️ Deliberately NOT tuned. The only delta this has been measured at is recorded in
#: `docs/preregistrations/2026-08-20-calibration-carry-forward.md`; treat anything above it as
#: untested, and lower the bound per tenant rather than reading this default as evidence.
DEFAULT_MAX_CORPUS_DELTA = 0.25

#: Largest per-class error the INHERITED threshold may make on the child generation's fresh
#: scores, as a fraction of that class.
#:
#: ⛔ This check is not redundant with certification, and the difference is the entire reason
#: carry-forward needs a rule `calibrate` does not. Separability is threshold-free: it asks
#: whether the two classes are still ORDERED. Adding documents that lift every unanswerable score
#: by the same amount leaves the ordering perfect — AUC 1.00, certified — while sliding the whole
#: unanswerable class above a threshold that is no longer allowed to move. A refit cannot be
#: fooled this way because it puts the cut back between the classes; carry-forward holds the cut
#: still, so it has to check the cut.
#:
#: Found by `test_carry_forward_rejects_when_the_threshold_stops_separating`, which reused 20 of
#: 22 sources and moved only the 2 new ones: separability stayed 1.00 and the inherited threshold
#: admitted 100% of the unanswerable queries.
#:
#: 0.10 is a ceiling, not a measured safe distance. The reference point is the 2026-08-17 fit on
#: this corpus, which measured leave-one-out false-confirm 3.6% and false-abstain 4.5%, and the
#: same session measured the answerable and unanswerable distributions OVERLAPPING in 4 of 4
#: corpora, so a bar at zero is not reachable and would only mean nothing ever carries.
DEFAULT_MAX_CARRY_FORWARD_ERROR = 0.10


def threshold_error_rates(
    answerable: Sequence[float], unanswerable: Sequence[float], threshold: float
) -> dict[str, float]:
    """Per-class error of a FIXED threshold, each named by its own denominator.

    `>=` is a confirm, matching `recall.trust`, which promotes a hit whose cosine reaches the
    threshold. A `>` here would disagree with serving on exactly the boundary cases the threshold
    was placed to decide.
    """
    false_abstain = sum(1 for score in answerable if score < threshold)
    false_confirm = sum(1 for score in unanswerable if score >= threshold)
    return {
        "false_abstain_rate": (false_abstain / len(answerable)) if answerable else 0.0,
        "false_confirm_rate": (false_confirm / len(unanswerable)) if unanswerable else 0.0,
        "false_abstain_count": false_abstain,
        "false_confirm_count": false_confirm,
    }


def _require_carry_forward(value: Mapping[str, Any], threshold: float, scale: float) -> None:
    """Validate carry-forward provenance, including that it names the numbers it actually carried.

    The last two checks are the ones worth having. Provenance saying "inherited from cal_X" beside
    a threshold that is not cal_X's threshold is not provenance, it is a decoration, and it would
    survive every other check in this class because nothing else compares the two.
    """
    for key in ("parent_calibration_id", "parent_generation_id", "corpus_delta"):
        if key not in value:
            raise CalibrationBindingError(f"carry-forward provenance is missing {key!r}")
    for key in ("parent_calibration_id", "parent_generation_id"):
        if not str(value[key]).strip():
            raise CalibrationBindingError(f"carry-forward {key} must be non-empty")
    delta = value["corpus_delta"]
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(delta):
        raise CalibrationBindingError("carry-forward corpus_delta must be a finite number")
    if not 0.0 <= float(delta) <= 1.0:
        raise CalibrationBindingError("carry-forward corpus_delta must be a fraction in [0, 1]")
    inherited_threshold = value.get("inherited_threshold")
    inherited_scale = value.get("inherited_scale")
    if inherited_threshold is not None and float(inherited_threshold) != threshold:
        raise CalibrationBindingError(
            "carry-forward provenance names an inherited threshold this artifact does not carry"
        )
    if inherited_scale is not None and float(inherited_scale) != scale:
        raise CalibrationBindingError(
            "carry-forward provenance names an inherited scale this artifact does not carry"
        )


def corpus_delta(
    parent_objects: Sequence[Mapping[str, Any]],
    child_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Source-level difference between two generation manifests.

    The denominator is the UNION of source URIs, not the child's count, and that choice is load
    bearing: over the child alone, deleting nine tenths of a corpus scores a delta near zero,
    because the survivors all still match. A deletion is a change to what the index can answer and
    must count as one.

    Compares `(uri, sha256)`, so a file edited in place counts as changed even though its URI is
    unmoved. That is the same identity `_reuse_source` uses to decide whether a chunk may be
    copied forward, so the delta reported here and the work the rebuild actually does cannot
    disagree.
    """
    parent = {str(obj["uri"]): str(obj.get("sha256", "")) for obj in parent_objects}
    child = {str(obj["uri"]): str(obj.get("sha256", "")) for obj in child_objects}
    added = sorted(set(child) - set(parent))
    removed = sorted(set(parent) - set(child))
    modified = sorted(uri for uri in set(parent) & set(child) if parent[uri] != child[uri])
    union = len(set(parent) | set(child))
    changed = len(added) + len(removed) + len(modified)
    return {
        "sources_parent": len(parent),
        "sources_child": len(child),
        "sources_added": len(added),
        "sources_removed": len(removed),
        "sources_modified": len(modified),
        "sources_union": union,
        "corpus_delta": (changed / union) if union else 0.0,
    }


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
    #: Provenance when this threshold was INHERITED from an earlier generation rather than fitted
    #: on this one. None means fitted here, which is the true statement about every artifact
    #: written before `carry_forward` existed — hence None rather than an empty mapping, and hence
    #: no backfill. See `CalibrationRepository.carry_forward`.
    carry_forward: Mapping[str, Any] | None = None

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
        if self.carry_forward is not None:
            _require_carry_forward(self.carry_forward, self.threshold, self.scale)
            object.__setattr__(self, "carry_forward", _frozen_json(self.carry_forward))

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
        payload: dict[str, Any] = {
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
        # Added ONLY when present, so every artifact written before carry-forward existed hashes
        # to exactly the bytes it hashed to then and still verifies. An unconditional key (even
        # `None`) would invalidate every stored checksum on upgrade, and a checksum that fails
        # after a version bump teaches the operator to ignore checksum failures.
        if self.carry_forward is not None:
            payload["carry_forward"] = dict(self.carry_forward)
        return payload

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

    @property
    def threshold_was_measured_here(self) -> bool:
        """False when this threshold was inherited from an earlier generation.

        Exposed as a property rather than left to callers testing `carry_forward is not None`,
        because that test reads as "has provenance" and the question every caller actually means
        is the opposite one.
        """
        return self.carry_forward is None


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
            carry_forward=dict(row[21]) if isinstance(row[21], Mapping) else None,
        )
        artifact.verify_checksum()
        return artifact

    _COLUMNS = (
        "calibration_id, tenant_id, generation_id, embedder_identity, pipeline_fingerprint, "
        "corpus_fingerprint, query_set_digest, threshold, scale, separability, ci_low, ci_high, "
        "n_answerable, n_unanswerable, certified, certification_reason, lifecycle_state, scores, "
        "created_at, created_by, artifact_checksum, carry_forward"
    )

    def _manifest_objects(
        self, conn: psycopg.Connection, generation_id: str
    ) -> list[Mapping[str, Any]]:
        row = conn.execute(
            "SELECT manifest FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
            (self.tenant_id, generation_id),
        ).fetchone()
        if row is None or not isinstance(row[0], Mapping):
            raise CalibrationBindingError(f"generation {generation_id!r} has no stored manifest")
        objects = row[0].get("objects")
        if not isinstance(objects, list) or not objects:
            raise CalibrationBindingError(
                f"generation {generation_id!r} manifest lists no source objects"
            )
        return [obj for obj in objects if isinstance(obj, Mapping)]

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

    def carry_forward(
        self,
        generation_id: str,
        embedder: Embedder,
        *,
        parent_calibration_id: str | None = None,
        max_corpus_delta: float = DEFAULT_MAX_CORPUS_DELTA,
        max_error: float = DEFAULT_MAX_CARRY_FORWARD_ERROR,
    ) -> CalibrationArtifactV2:
        """Re-verify a published threshold against a NEW generation, without refitting it.

        This exists because a corpus that changes at all currently costs a full recalibration.
        `resolve` binds a calibration to one `generation_id`, so any rebuild — adding a handful of
        files to a thousand — leaves the new generation with no artifact, resolves `STALE`, and
        strict policy refuses every query. The operational effect is that a live index cannot
        absorb an increment without either a manual recalibration or a period of serving
        uncalibrated, and both of those are worse than the problem.

        **This is not a tolerance and it does not loosen the gate.** Nothing is carried on the
        strength of the delta being small. The parent's own stored labelled query set is re-scored
        against the child generation, and the inherited threshold must clear the SAME
        certification bar on those fresh scores that a new fit would have to clear. What is
        inherited is the threshold, not the certification. If the corpus moved enough to break the
        threshold, this produces a `rejected` artifact and the operator still has to recalibrate,
        which is the correct outcome and the one the delta bound cannot deliver on its own.

        Three refusals, in the order an operator can act on them:

        1. **A different pipeline is refused outright**, never bounded. A threshold is a property
           of an embedder's cosine regime, and 2026-08-17 measured `voyage-4` at 0.269 to 0.413 on
           one corpus where `voyage-code-3` returned 0.480 to 0.834. Carrying a number across that
           is not a small error, and no delta is small enough to make it one.
        2. **A delta above `max_corpus_delta` is refused before any embedding work**, because
           re-scoring a query set says nothing about the queries nobody labelled, and past some
           point the labelled set is describing a corpus that no longer exists.
        3. **A query set that no longer canonicalises to its stored digest is refused**, the same
           check `resolve` makes, because otherwise the evidence could be edited between the fit
           and the re-verification.

        `refit_threshold` is recorded in the provenance and **changes nothing**. It is what a
        fresh fit on these scores would have chosen, so an operator can see the inherited number
        drifting away from the data before it drifts far enough to fail. A diagnosis that silently
        moved the boundary would be a different feature wearing this one's name.
        """
        if not 0.0 <= max_corpus_delta <= 1.0:
            raise CalibrationBindingError("max_corpus_delta must be a fraction in [0, 1]")
        if not 0.0 <= max_error <= 1.0:
            raise CalibrationBindingError("max_error must be a fraction in [0, 1]")
        with self._connect() as conn:
            if parent_calibration_id is None:
                row = conn.execute(
                    "SELECT calibration_id FROM recall_calibrations WHERE tenant_id = %s "
                    "AND lifecycle_state = 'published' AND generation_id IS NOT NULL "
                    "AND generation_id <> %s ORDER BY published_at DESC, created_at DESC LIMIT 1",
                    (self.tenant_id, generation_id),
                ).fetchone()
                if row is None:
                    raise CalibrationNotFound(
                        f"tenant {self.tenant_id!r} has no published calibration on another "
                        f"generation to carry forward; calibrate {generation_id!r} directly"
                    )
                parent_calibration_id = str(row[0])
        parent = self.get(parent_calibration_id)
        if parent.generation_id == generation_id:
            raise CalibrationBindingError(
                f"calibration {parent.calibration_id!r} is already bound to generation "
                f"{generation_id!r}; there is nothing to carry forward"
            )
        if parent.status is not CalibrationStatus.CERTIFIED:
            # A draft or rejected parent has no certified threshold to carry, and inheriting one
            # would launder an uncertified number into a certified-looking artifact.
            raise CalibrationUncertified(
                f"parent calibration {parent.calibration_id!r} is {parent.status.value}, so it "
                f"has no certified threshold to carry forward"
            )

        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            _state, pipeline_raw, pipeline_fingerprint, corpus_fingerprint = self._generation(
                conn, generation_id
            )
            if pipeline_fingerprint != parent.pipeline_fingerprint:
                raise CalibrationBindingError(
                    f"generation {generation_id!r} has a different pipeline fingerprint from "
                    f"calibration {parent.calibration_id!r}; a threshold cannot be carried across "
                    f"a pipeline change, recalibrate instead"
                )
            if corpus_fingerprint == parent.corpus_fingerprint:
                raise CalibrationBindingError(
                    f"generation {generation_id!r} has the same corpus fingerprint as "
                    f"calibration {parent.calibration_id!r}; carry-forward is for a CHANGED "
                    f"corpus, and an identical one means the wrong generation was named"
                )
            delta = corpus_delta(
                self._manifest_objects(conn, parent.generation_id),
                self._manifest_objects(conn, generation_id),
            )
            query_row = conn.execute(
                "SELECT queries FROM recall_calibration_query_sets WHERE tenant_id = %s "
                "AND query_set_digest = %s",
                (self.tenant_id, parent.query_set_digest),
            ).fetchone()
        if delta["corpus_delta"] > max_corpus_delta:
            raise CalibrationBindingError(
                f"corpus delta {delta['corpus_delta']:.3f} exceeds the carry-forward bound "
                f"{max_corpus_delta:.3f} ({delta['sources_added']} added, "
                f"{delta['sources_removed']} removed, {delta['sources_modified']} modified over "
                f"{delta['sources_union']} sources); recalibrate against a labelled query set"
            )
        if query_row is None or not isinstance(query_row[0], list):
            raise CalibrationBindingError(
                f"the labelled query set {parent.query_set_digest} behind calibration "
                f"{parent.calibration_id!r} is missing, so its threshold cannot be re-verified"
            )
        labels, query_digest = canonical_query_set(query_row[0])
        if query_digest != parent.query_set_digest:
            raise CalibrationBindingError(
                "stored labelled query set no longer matches its digest"
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

        # The inherited threshold and scale, judged on the CHILD's scores. `from_samples` is used
        # only for `refit_threshold`, which is diagnostic and reaches nothing.
        runtime = Calibration(
            embedder=parent.runtime.embedder,
            threshold=parent.threshold,
            scale=parent.scale,
            separability=separability(answerable, unanswerable),
            n_answerable=len(answerable),
            n_unanswerable=len(unanswerable),
        )
        if runtime.separability is None or runtime.separability_ci is None:
            raise CalibrationBindingError("both labelled classes are required for carry-forward")
        errors = threshold_error_rates(answerable, unanswerable, parent.threshold)
        # BOTH conditions, and the second is the one that catches a shifted class. See
        # DEFAULT_MAX_CARRY_FORWARD_ERROR: certification alone would pass a threshold that has
        # stopped deciding anything, because separability cannot see a cut it is not asked about.
        within_error = (
            errors["false_abstain_rate"] <= max_error and errors["false_confirm_rate"] <= max_error
        )
        certified = runtime.certified is True and within_error
        reason = runtime.certification_reason
        if runtime.certified is True and not within_error:
            reason = (
                f"the inherited threshold {parent.threshold:.4f} no longer decides this corpus: "
                f"false abstain {errors['false_abstain_rate']:.1%} of {len(answerable)} "
                f"answerable, false confirm {errors['false_confirm_rate']:.1%} of "
                f"{len(unanswerable)} unanswerable, against a bound of {max_error:.1%}. "
                f"Separability is {runtime.separability:.4f}, so the classes are still ordered "
                f"and only the boundary has moved; recalibrate to place it again."
            )
        refit = from_samples(parent.runtime.embedder, answerable, unanswerable)
        provenance = {
            "parent_calibration_id": parent.calibration_id,
            "parent_generation_id": parent.generation_id,
            "parent_corpus_fingerprint": parent.corpus_fingerprint,
            "parent_separability": parent.separability,
            "inherited_threshold": parent.threshold,
            "inherited_scale": parent.scale,
            "refit_threshold": refit.threshold,
            "max_corpus_delta": max_corpus_delta,
            "max_carry_forward_error": max_error,
            **errors,
            **delta,
        }
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
            "certification_reason": reason,
            "created_at": created_at,
            "created_by": self.actor,
            "scores": scores,
            "carry_forward": provenance,
        }
        checksum = canonical_sha256(immutable)
        lifecycle = "draft" if certified else "rejected"
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibrations "
                "(tenant_id, calibration_id, generation_id, embedder_identity, "
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, threshold, scale, "
                "separability, ci_low, ci_high, n_answerable, n_unanswerable, certified, "
                "certification_reason, lifecycle_state, scores, created_at, created_by, "
                "artifact_checksum, carry_forward) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
                    # `reason`, not `runtime.certification_reason`: the column has to hold the
                    # same string the checksum was taken over, or every later read of this row
                    # fails verification and reports corruption instead of a rejection.
                    reason,
                    lifecycle,
                    Jsonb(scores),
                    created_at,
                    self.actor,
                    checksum,
                    Jsonb(provenance),
                ),
            )
            self._audit(
                conn,
                "calibration_carried_forward",
                calibration_id,
                generation_id,
                {
                    "parent_calibration_id": parent.calibration_id,
                    "parent_generation_id": parent.generation_id,
                    "corpus_delta": delta["corpus_delta"],
                    "certified": certified,
                },
            )
            if not certified:
                self._audit(
                    conn,
                    "calibration_rejected",
                    calibration_id,
                    generation_id,
                    {"reason": reason},
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
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, created_at, "
                "carry_forward FROM recall_calibrations WHERE tenant_id = %s "
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
                # Listed, not left to `show`. After a chain of rebuilds the question an operator
                # has is which of these thresholds anyone actually measured, and a listing that
                # renders an inherited threshold identically to a fitted one cannot answer it.
                "threshold_was_measured_here": row[8] is None,
                "carried_forward_from": (
                    row[8].get("parent_calibration_id") if isinstance(row[8], Mapping) else None
                ),
                "corpus_delta": (
                    row[8].get("corpus_delta") if isinstance(row[8], Mapping) else None
                ),
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
                # `carry_forward` is written here for the same reason it is checksummed: it is
                # part of the immutable payload, so an import that dropped it would store a row
                # whose checksum can never verify again, and the artifact would come back from
                # `get` as a corruption rather than as an import.
                "certification_reason, lifecycle_state, scores, created_at, created_by, "
                "artifact_checksum, carry_forward) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
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
                    Jsonb(dict(artifact.carry_forward))
                    if artifact.carry_forward is not None
                    else None,
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
