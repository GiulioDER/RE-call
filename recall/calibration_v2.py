"""Tenant and generation bound calibration artifacts for the v1 index path."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import psycopg
from psycopg.types.json import Jsonb

from recall.calibration import Calibration, from_samples
from recall.embedding_core import profile_lineage_mismatch
from recall.embeddings import Embedder
from recall.lineage import PipelineIdentity, canonical_json, canonical_sha256
from recall.observability import get_logger

# Every name of the two split-out modules is re-exported, the underscore ones included, because
# this module's own code and its tests reach several of those private helpers through here.
from recall.calibration_v2_artifact import (  # noqa: F401  # re-exported, see above
    ARTIFACT_VERSION,
    CalibrationStatus,
    CalibrationError,
    CalibrationNotFound,
    CalibrationUncertified,
    CalibrationBindingError,
    LifecycleState,
    _ARTIFACT_STATES,
    _CERTIFIED_STATES,
    _FALLBACK_EMBEDDER,
    _REDERIVE_TOLERANCE,
    _RESCORE_TOLERANCE,
    _CUMULATIVE_DELTA_TOLERANCE,
    _frozen_json,
    _utc_isoformat,
    _require_utc_isoformat,
    _require_digest,
    _require_carry_forward,
    CalibrationArtifactV2,
    CalibrationResolution,
)
from recall.calibration_v2_bounds import (  # noqa: F401  # re-exported, see above
    DEFAULT_MAX_CORPUS_DELTA,
    DEFAULT_MAX_CARRY_FORWARD_ERROR,
    ThresholdErrors,
    threshold_error_rates,
    within_error_bound,
    InheritedThresholdVerdict,
    judge_inherited_threshold,
    _cumulative_corpus_delta,
    _exceeds_cumulative_bound,
    _carried_cumulative_delta,
    _BUNDLE_FLOAT_FIELDS,
    _BUNDLE_INT_FIELDS,
    _BUNDLE_MAPPING_FIELDS,
    _require_bundle_field_types,
    _bundle_scores,
    _import_error_bound,
    _imported_verdict,
    _require_scores_reproduce,
    _require_fresh_certification,
    _rederived_matches,
    _require_scores_support_certification,
    corpus_delta,
)


_log = get_logger("calibration")


def canonical_query_set(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Normalise a labelled query set and return it with its content digest.

    The canonical form keeps only `query`, `answerable`, and (when present) a sorted,
    deduplicated `relevant_ids`, sorted by each entry's canonical JSON, so the same labels in
    any order and with any extra keys canonicalise identically. A duplicate labelled query is
    refused rather than deduplicated, because two copies of one label would silently weight it.
    Only an identical ENTRY is refused here; a fit also applies `require_distinct_scored_queries`,
    which refuses the same query repeated with different `relevant_ids` or labels.
    The returned digest is `canonical_sha256` of the canonical entries and is what is stored as
    an artifact's `query_set_digest`, so `resolve` and `carry_forward` can detect a stored set
    that no longer matches the evidence a threshold was fitted on.
    """
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


def require_distinct_scored_queries(labels: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a labelled set that repeats a query the scorer would score twice.

    `canonical_query_set` refuses an identical ENTRY, but `relevant_ids` is part of an entry and
    the scorer (`measure_top_cosines`) never reads it: two entries that differ only there, or in
    their answerable label, are the same cosine counted twice. Twenty such copies of one query
    per class would clear the certification sample floor on two queries' evidence, and one query
    labelled both ways is evidence nobody should fit on.

    Applied in `calibrate` and in the lite fit (`recall.lite.calibration.auto_calibrate`),
    whatever the set's origin: a new one, or a stored one when drift's unattended refit passes
    the parent's set back in, which then ends `failed` with this reason.
    It is not applied inside `canonical_query_set`, which also re-canonicalises stored sets on
    the serving path (`resolve_within`), where refusing would be a resolution error instead of
    a refused fit.
    """
    seen: dict[str, bool] = {}
    for item in labels:
        query = str(item["query"])
        if query in seen:
            label = (
                "with contradictory answerable labels"
                if seen[query] != bool(item["answerable"])
                else "with different relevant_ids"
            )
            raise CalibrationBindingError(
                f"query set repeats the query {query!r} {label}; the scorer would count one "
                f"cosine twice, so keep one entry per query"
            )
        seen[query] = bool(item["answerable"])


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


#: Passed as `checksum` when an artifact is constructed purely to COMPUTE its checksum from
#: `immutable_payload()`. The checksum field is not part of that payload, so the placeholder never
#: reaches a digest; it exists because the dataclass requires a well-formed value at construction.
#: The row is always read back through `_artifact`, which verifies the REAL checksum.
_UNCOMPUTED_CHECKSUM = "0" * 64


def _require_runtime_embedder(pipeline: PipelineIdentity, embedder: Embedder) -> None:
    """Refuse to score a generation's vectors with an embedder that did not write them.

    Model and dimension first, then the recorded profile, which is the only field that separates
    two profiles of one model (Voyage with and without `input_type`). A threshold fitted on those
    cosines would certify numbers that compare two encoders. Shared by `calibrate` and
    `carry_forward`, the two paths that score labels with the runtime embedder.
    """
    if embedder.name != pipeline.embedder.model or embedder.dim != pipeline.embedder.dimension:
        raise CalibrationBindingError(
            "embedder implementation does not match the generation pipeline identity"
        )
    mismatch = profile_lineage_mismatch(
        embedder, pipeline.embedder.profile_id, pipeline.embedder.profile_fingerprint
    )
    if mismatch is not None:
        raise CalibrationBindingError(f"{mismatch} in the generation pipeline identity")


class _GenerationLineage(NamedTuple):
    """What `CalibrationRepository._generation` reads: the identity and the two fingerprints."""

    pipeline_identity: dict[str, Any]
    pipeline_fingerprint: str
    corpus_fingerprint: str


def write_calibration_bundle(payload: dict[str, Any], path: str | Path) -> Path:
    """Seal `payload` with its `bundle_checksum` and write it as canonical JSON.

    The one bundle format: `CalibrationRepository.export_bundle` and the lite store's export use it.
    """
    payload["bundle_checksum"] = canonical_sha256(payload)
    target = Path(path)
    target.write_bytes(canonical_json(payload) + b"\n")
    return target


class CalibrationRepository:
    """Tenant-scoped store of v2 calibration artifacts and their labelled query sets.

    Lifecycle: `calibrate` and `carry_forward` write a `draft` when certification passes and a
    `rejected` row when it does not, deliberately recording the failure rather than raising, so
    an operator can read WHY a fit failed. `publish` promotes a certified draft under a
    per-generation advisory lock, marking any previously published artifact for the same
    generation `superseded`; `superseded` rows are history, never deleted. `legacy_unbound`
    rows come from `import_bundle`'s legacy fallback (a pre-v2 JSON file holding only a
    threshold and scale) and can never be published or resolved, because they bind to no
    tenant, generation, pipeline, corpus, or query set.

    Every artifact's immutable payload is checksummed at write and re-verified on every read,
    and rows are guarded by database triggers that refuse payload updates, so evidence cannot
    be edited between a fit and a later decision that relies on it.
    """

    def __init__(
        self,
        dsn: str,
        tenant_id: str,
        *,
        actor: str = "recall-cli",
        connect_timeout_s: int = 10,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        self.dsn = dsn
        self.tenant_id = tenant_id
        self.actor = actor
        self._connect_timeout_s = connect_timeout_s

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        """A tenant-scoped autocommit connection.

        A missing table or column is re-raised as `SchemaTooOld`. Only the fitting paths check the
        schema up front (through `score_query_set`), so `list`, `show`, `publish`, `export` and
        `import` against an unmigrated database otherwise ended in a raw psycopg traceback naming a
        column, instead of the message that says to run `recall schema apply`.
        """
        from recall.schema import SchemaTooOld

        with psycopg.connect(
            self.dsn, autocommit=True, connect_timeout=self._connect_timeout_s
        ) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
            try:
                yield conn
            except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as exc:
                raise SchemaTooOld(
                    f"the database schema is older than this package expects "
                    f"({exc.diag.message_primary or exc}); run `recall schema apply`"
                ) from exc

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
        self, conn: psycopg.Connection, generation_id: str, *, lock: bool = False
    ) -> _GenerationLineage:
        """A ready, active or retired generation's lineage, refusing any other state.

        `lock` reads the row `FOR SHARE`, for a caller that acts on the fingerprints it reads in
        the same transaction (`publish`). `forget()` rewrites `corpus_fingerprint` under `FOR
        UPDATE`, which a plain read does not wait for, so without the lock a concurrent erasure
        could commit between the check and the write.
        """
        row = conn.execute(
            "SELECT state, pipeline_identity, pipeline_fingerprint, corpus_fingerprint "
            "FROM recall_generations WHERE tenant_id = %s AND generation_id = %s"
            + (" FOR SHARE" if lock else ""),
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
        return _GenerationLineage(dict(row[1]), str(row[2]), str(row[3]))

    def _query_set_rows(self, conn: psycopg.Connection, digest: str) -> list[Any] | None:
        """The stored labelled queries behind `digest`, or None when absent or not a list."""
        row = conn.execute(
            "SELECT queries FROM recall_calibration_query_sets WHERE tenant_id = %s "
            "AND query_set_digest = %s",
            (self.tenant_id, digest),
        ).fetchone()
        if row is None or not isinstance(row[0], list):
            return None
        return row[0]

    #: The one source of truth for the SELECT column list AND for `_artifact`'s row decoding.
    #: Keeping them a single tuple is what makes it impossible for the projection and the
    #: positional decode to drift apart, which they previously could: the decode indexed
    #: `row[0]..row[21]` against a comma-separated string thirty lines away.
    _COLUMN_NAMES = (
        "calibration_id",
        "tenant_id",
        "generation_id",
        "embedder_identity",
        "pipeline_fingerprint",
        "corpus_fingerprint",
        "query_set_digest",
        "threshold",
        "scale",
        "separability",
        "ci_low",
        "ci_high",
        "n_answerable",
        "n_unanswerable",
        "certified",
        "certification_reason",
        "lifecycle_state",
        "scores",
        "created_at",
        "created_by",
        "artifact_checksum",
        "carry_forward",
    )
    _COLUMNS = ", ".join(_COLUMN_NAMES)
    #: The one INSERT of a v2 artifact row: every column `_COLUMN_NAMES` decodes, with `tenant_id`
    #: first. `_insert_payload_row` and `import_bundle` bind different VALUES (the import takes
    #: the rebound generation, the verified digest and the generation's embedder), but one column
    #: list, so a column added to one can no longer be missing from the other.
    _INSERT_ARTIFACT = (
        "INSERT INTO recall_calibrations (tenant_id, "
        + ", ".join(name for name in _COLUMN_NAMES if name != "tenant_id")
        + ") VALUES ("
        + ", ".join(["%s"] * len(_COLUMN_NAMES))
        + ")"
    )

    @classmethod
    def _artifact(cls, row: tuple[Any, ...]) -> CalibrationArtifactV2:
        record = dict(zip(cls._COLUMN_NAMES, row, strict=True))
        identity = record["embedder_identity"]
        scores = record["scores"]
        carry = record["carry_forward"]
        artifact = CalibrationArtifactV2(
            calibration_id=str(record["calibration_id"]),
            tenant_id=str(record["tenant_id"]),
            generation_id=str(record["generation_id"]),
            embedder_identity=dict(identity) if isinstance(identity, Mapping) else {},
            pipeline_fingerprint=str(record["pipeline_fingerprint"]),
            corpus_fingerprint=str(record["corpus_fingerprint"]),
            query_set_digest=str(record["query_set_digest"]),
            threshold=float(record["threshold"]),
            scale=float(record["scale"]),
            separability=float(record["separability"]),
            separability_ci=(float(record["ci_low"]), float(record["ci_high"])),
            n_answerable=int(record["n_answerable"]),
            n_unanswerable=int(record["n_unanswerable"]),
            certified=bool(record["certified"]),
            certification_reason=str(record["certification_reason"]),
            lifecycle_state=str(record["lifecycle_state"]),
            scores=dict(scores) if isinstance(scores, Mapping) else {},
            created_at=_utc_isoformat(record["created_at"]),
            created_by=str(record["created_by"]),
            checksum=str(record["artifact_checksum"]),
            carry_forward=dict(carry) if isinstance(carry, Mapping) else None,
        )
        artifact.verify_checksum()
        return artifact

    def score_query_set(
        self, generation_id: str, embedder: Embedder, labels: Sequence[Mapping[str, Any]]
    ) -> tuple[list[float], list[float]]:
        """Best dense cosine per labelled query against ONE pinned generation.

        Extracted because `calibrate`, `carry_forward` and drift monitoring all need exactly this
        and had two byte-identical copies of it between them. The sampling rule is the single one
        in `recall.eval.calibrate.measure_top_cosines`, so a fit, a re-verification and a drift
        probe cannot come to different answers about the same generation for the reason that they
        sampled it differently.

        `pin_generation` is what makes the scores belong to the generation named rather than to
        whatever is active at the moment the query runs.
        """
        from recall.eval.calibrate import measure_top_cosines
        from recall.generation_store import GenerationStore

        store = GenerationStore(self.dsn, embedder.dim, tenant=self.tenant_id)
        try:
            store.check_schema()
            with store.pin_generation(generation_id):
                # Real dicts, not the stored Mappings. `measure_top_cosines` is annotated
                # `list[dict]`, and `canonical_query_set` hands back frozen mappings.
                return measure_top_cosines(store, embedder, [dict(item) for item in labels])
        finally:
            store.close()

    def stored_query_set(
        self, query_set_digest: str
    ) -> tuple[tuple[Mapping[str, Any], ...], str]:
        """The labelled queries behind a digest, re-canonicalised and re-digested.

        Returned with the digest RECOMPUTED rather than echoed back, so a caller comparing it
        against the artifact's `query_set_digest` is checking the stored bytes and not its own
        argument. That is the same check `resolve` makes, and it exists because the evidence could
        otherwise be edited between the fit and any later re-verification.
        """
        with self._connect() as conn:
            rows = self._query_set_rows(conn, query_set_digest)
        if rows is None:
            raise CalibrationNotFound(
                f"labelled query set {query_set_digest} is not stored for tenant "
                f"{self.tenant_id!r}, so nothing can be re-scored against it"
            )
        return canonical_query_set(rows)

    def manifest_objects_for(self, generation_id: str) -> list[Mapping[str, Any]]:
        """The generation's manifest source objects, on a connection of this repository's own."""
        with self._connect() as conn:
            return self._manifest_objects(conn, generation_id)

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

    def _insert_payload_row(
        self,
        conn: psycopg.Connection,
        payload: Mapping[str, Any],
        lifecycle: str,
        checksum: str,
        *,
        carried: bool,
    ) -> None:
        """Insert one artifact row whose every checksummed value comes from ``payload``.

        ``payload`` is the mapping the checksum was computed over, so the row cannot hold a value
        the checksum does not; that is also why ``certification_reason`` is right by
        construction. ``carried`` names a carried-forward artifact, whose payload must hold
        ``carry_forward`` (read strictly, so a payload without it fails rather than storing
        NULL); a fresh calibration stores NULL there, as the column's default always did.
        `import_bundle` keeps its own insert: it writes the rebound generation, the verified
        query-set digest and the embedder from the bundle identity, not the payload's.
        """
        conn.execute(
            self._INSERT_ARTIFACT,
            (
                self.tenant_id,
                payload["calibration_id"],
                payload["generation_id"],
                Jsonb(payload["embedder_identity"]),
                payload["pipeline_fingerprint"],
                payload["corpus_fingerprint"],
                payload["query_set_digest"],
                payload["threshold"],
                payload["scale"],
                payload["separability"],
                payload["separability_ci"][0],
                payload["separability_ci"][1],
                payload["n_answerable"],
                payload["n_unanswerable"],
                payload["certified"],
                payload["certification_reason"],
                lifecycle,
                Jsonb(payload["scores"]),
                payload["created_at"],
                payload["created_by"],
                checksum,
                Jsonb(payload["carry_forward"]) if carried else None,
            ),
        )

    def _new_artifact(
        self,
        *,
        generation_id: str,
        pipeline: PipelineIdentity,
        lineage: _GenerationLineage,
        query_digest: str,
        runtime: Calibration,
        answerable: Sequence[float],
        unanswerable: Sequence[float],
        certified: bool,
        reason: str,
        carry_forward: Mapping[str, Any] | None = None,
    ) -> tuple[CalibrationArtifactV2, dict[str, Any], str]:
        """Build a new artifact and the payload and checksum its row is written from.

        Shared by `calibrate` and `carry_forward`. The artifact is constructed FIRST and the
        checksum computed over ITS payload, so the bytes hashed here and the bytes
        `verify_checksum` recomputes on every later read come from the same method. A hand-built
        dict previously made three independent copies of the payload shape, and a key added to one
        but not the others would fail every subsequent read as corruption.
        """
        if runtime.separability is None or runtime.separability_ci is None:
            raise CalibrationBindingError("both labelled classes are required for calibration")
        lifecycle = LifecycleState.DRAFT if certified else LifecycleState.REJECTED
        artifact = CalibrationArtifactV2(
            calibration_id=_id("cal"),
            tenant_id=self.tenant_id,
            generation_id=generation_id,
            embedder_identity=pipeline.embedder.to_dict(),
            pipeline_fingerprint=lineage.pipeline_fingerprint,
            corpus_fingerprint=lineage.corpus_fingerprint,
            query_set_digest=query_digest,
            threshold=runtime.threshold,
            scale=runtime.scale,
            separability=runtime.separability,
            separability_ci=runtime.separability_ci,
            n_answerable=len(answerable),
            n_unanswerable=len(unanswerable),
            certified=certified,
            certification_reason=reason,
            # `.value`, a plain str: the row is written from this field, and the enum member is
            # only for comparisons, so the driver never has to choose how to adapt it.
            lifecycle_state=lifecycle.value,
            created_at=datetime.now(UTC).isoformat(),
            created_by=self.actor,
            scores={"answerable": list(answerable), "unanswerable": list(unanswerable)},
            checksum=_UNCOMPUTED_CHECKSUM,
            carry_forward=carry_forward,
        )
        payload = artifact.immutable_payload()
        return artifact, payload, canonical_sha256(payload)

    def _audit_if_rejected(self, conn: psycopg.Connection, artifact: CalibrationArtifactV2) -> None:
        if not artifact.certified:
            self._audit(
                conn,
                "calibration_rejected",
                artifact.calibration_id,
                artifact.generation_id,
                {"reason": artifact.certification_reason},
            )

    def calibrate(
        self,
        generation_id: str,
        queries: Sequence[Mapping[str, Any]],
        embedder: Embedder,
    ) -> CalibrationArtifactV2:
        """Fit a threshold for one generation from a labelled query set, and store it.

        The queries are scored on the named generation with `embedder`, which must be the one
        that wrote its vectors. A certified fit is stored as a `draft` (run `publish` to serve
        it); a fit that fails certification is stored and RETURNED as `rejected`, not raised, so
        check `artifact.certified`. Raises `CalibrationBindingError` for a malformed or repeated
        query, a missing labelled class, or an embedder mismatch. After the artifact is saved,
        near-miss coverage is measured and audited, fail-open (`_record_near_miss_coverage`).
        """
        labels, query_digest = canonical_query_set(queries)
        require_distinct_scored_queries(labels)
        with self._connect() as conn:
            lineage = self._generation(conn, generation_id)
        pipeline = PipelineIdentity.from_dict(lineage.pipeline_identity)
        _require_runtime_embedder(pipeline, embedder)

        answerable, unanswerable = self.score_query_set(generation_id, embedder, labels)
        runtime = from_samples(embedder.name, answerable, unanswerable)
        artifact, payload, checksum = self._new_artifact(
            generation_id=generation_id,
            pipeline=pipeline,
            lineage=lineage,
            query_digest=query_digest,
            runtime=runtime,
            answerable=answerable,
            unanswerable=unanswerable,
            certified=runtime.certified is True,
            reason=runtime.certification_reason,
        )
        calibration_id = artifact.calibration_id
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibration_query_sets "
                "(tenant_id, query_set_digest, queries, sample_count, created_by) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.tenant_id, query_digest, Jsonb(list(labels)), len(labels), self.actor),
            )
            self._insert_payload_row(
                conn, payload, artifact.lifecycle_state, checksum, carried=False
            )
            self._audit(conn, "calibration_created", calibration_id, generation_id)
            self._audit_if_rejected(conn, artifact)
        stored = self.get(calibration_id)
        self._record_near_miss_coverage(stored, labels, embedder)
        return stored

    def _record_near_miss_coverage(
        self,
        artifact: CalibrationArtifactV2,
        labels: Sequence[Mapping[str, Any]],
        embedder: Embedder,
    ) -> None:
        """Measure near-miss coverage with a fresh fit and append it as an audit event. Fail-open.

        The PostgreSQL side of the lite fit (`recall.lite.calibration.auto_calibrate`): probes are
        built from this generation's own chunks and the fit's answerable questions
        (`recall.near_miss.near_miss_probes`), scored at the threshold just fitted with the
        unknown-term gate (`near_miss_coverage`), and stored by calibration id, counts only. It runs
        after the artifact is saved and never touches it: certification, lifecycle and threshold
        are decided before this line. A carried calibration is not measured, as on lite, so it reads
        as unmeasured and provisional until the next fit.
        """
        from recall.generation_store import GenerationStore
        from recall.near_miss import fit_coverage
        from recall.support import COVERAGE_EVENT

        try:
            with self._connect() as conn:
                texts = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s "
                        "AND generation_id = %s ORDER BY chunk_id",
                        (self.tenant_id, artifact.generation_id),
                    ).fetchall()
                ]
            store = GenerationStore(self.dsn, embedder.dim, tenant=self.tenant_id)
            try:
                with store.pin_generation(artifact.generation_id):
                    report = fit_coverage(
                        store, embedder, float(artifact.threshold), texts, labels
                    )
            finally:
                store.close()
            with self._connect() as conn:
                self._audit(
                    conn, COVERAGE_EVENT, artifact.calibration_id, artifact.generation_id, report
                )
        except Exception as exc:  # BROAD-CATCH: fail-open
            # The fit is already saved. Optional diagnostics cannot withdraw it, and provider
            # errors may carry private input or credentials, so log only the type.
            _log.warning("Optional near-miss coverage unavailable (%s)", type(exc).__name__)

    def near_miss_coverage(self, calibration_id: str) -> dict[str, Any] | None:
        """The near-miss coverage measured with this calibration's fit, validated, or None.

        Raises ValueError for a stored report that disagrees with the producer's count rules, so a
        reader can report it unavailable rather than print it.
        """
        from recall.support import COVERAGE_EVENT, latest_calibration_event, validated_coverage_report

        with self._connect() as conn:
            payload = latest_calibration_event(conn, self.tenant_id, COVERAGE_EVENT, calibration_id)
        return None if payload is None else validated_coverage_report(payload, calibration_id)

    def _carry_parent(
        self, generation_id: str, parent_calibration_id: str | None
    ) -> CalibrationArtifactV2:
        """The calibration `carry_forward` inherits from: named, or the newest published one.

        Refused when it is already on `generation_id`, or when its row is not published and
        certified.
        """
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
        return parent

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

        Before any of the bounds below, the parent itself is checked: it must be on a DIFFERENT
        generation from `generation_id`, and its row must be published and certified (a draft,
        rejected or superseded parent has no certified threshold to carry). Then four refusals:

        1. **A different pipeline is refused outright**, never bounded. A threshold is a property
           of an embedder's cosine regime, and 2026-08-17 measured `voyage-4` at 0.269 to 0.413 on
           one corpus where `voyage-code-3` returned 0.480 to 0.834. Carrying a number across that
           is not a small error, and no delta is small enough to make it one. An IDENTICAL corpus
           fingerprint is refused at the same point: carry-forward is for a changed corpus, and
           an unchanged one means the wrong generation was named.
        2. **A delta above `max_corpus_delta` is refused before any embedding work**, because
           re-scoring a query set says nothing about the queries nobody labelled, and past some
           point the labelled set is describing a corpus that no longer exists.
        3. **A CHAIN whose cumulative delta exceeds the same bound is refused**, even when every
           individual step passed it. Each step's delta is measured only against the immediate
           parent, so without this a corpus could be replaced 20% at a time indefinitely while
           the threshold still wears the digest of the original fit. The cumulative figure is
           carried in each artifact's provenance rather than recomputed against the origin
           generation, because generation gc can have pruned the origin's manifest. A chain at
           exactly the bound is allowed: the sum is compared within `_CUMULATIVE_DELTA_TOLERANCE`.
        4. **A query set that no longer canonicalises to its stored digest is refused**, the same
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
        parent = self._carry_parent(generation_id, parent_calibration_id)

        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            lineage = self._generation(conn, generation_id)
            if lineage.pipeline_fingerprint != parent.pipeline_fingerprint:
                raise CalibrationBindingError(
                    f"generation {generation_id!r} has a different pipeline fingerprint from "
                    f"calibration {parent.calibration_id!r}; a threshold cannot be carried across "
                    f"a pipeline change, recalibrate instead"
                )
            if lineage.corpus_fingerprint == parent.corpus_fingerprint:
                raise CalibrationBindingError(
                    f"generation {generation_id!r} has the same corpus fingerprint as "
                    f"calibration {parent.calibration_id!r}; carry-forward is for a CHANGED "
                    f"corpus, and an identical one means the wrong generation was named"
                )
            delta = corpus_delta(
                self._manifest_objects(conn, parent.generation_id),
                self._manifest_objects(conn, generation_id),
            )
            query_rows = self._query_set_rows(conn, parent.query_set_digest)
        cumulative = _carried_cumulative_delta(parent, delta, max_corpus_delta)
        if query_rows is None:
            raise CalibrationBindingError(
                f"the labelled query set {parent.query_set_digest} behind calibration "
                f"{parent.calibration_id!r} is missing, so its threshold cannot be re-verified"
            )
        labels, query_digest = canonical_query_set(query_rows)
        if query_digest != parent.query_set_digest:
            raise CalibrationBindingError(
                "stored labelled query set no longer matches its digest"
            )

        pipeline = PipelineIdentity.from_dict(lineage.pipeline_identity)
        _require_runtime_embedder(pipeline, embedder)

        answerable, unanswerable = self.score_query_set(generation_id, embedder, labels)

        # The inherited threshold and scale, judged on the CHILD's scores, by the rule the lite
        # store's carry-forward shares (`judge_inherited_threshold`).
        verdict = judge_inherited_threshold(parent, answerable, unanswerable, max_error=max_error)
        runtime, errors, certified, reason = (
            verdict.runtime,
            verdict.errors,
            verdict.certified,
            verdict.reason,
        )
        if runtime.separability is None or runtime.separability_ci is None:  # narrowed for mypy
            raise CalibrationBindingError("both labelled classes are required for carry-forward")
        provenance = {
            "parent_calibration_id": parent.calibration_id,
            "parent_generation_id": parent.generation_id,
            "parent_corpus_fingerprint": parent.corpus_fingerprint,
            "parent_separability": parent.separability,
            "inherited_threshold": parent.threshold,
            "inherited_scale": parent.scale,
            "refit_threshold": verdict.refit_threshold,
            "max_corpus_delta": max_corpus_delta,
            "max_carry_forward_error": max_error,
            **errors,
            **delta,
            # The chain total, not this step's delta: what the NEXT carry-forward reads to keep
            # cumulative drift bounded. New in this key only for NEW artifacts; older provenance
            # without it is read by `_cumulative_corpus_delta` as its recorded step delta.
            "cumulative_corpus_delta": cumulative,
        }
        artifact, payload, checksum = self._new_artifact(
            generation_id=generation_id,
            pipeline=pipeline,
            lineage=lineage,
            query_digest=query_digest,
            runtime=runtime,
            answerable=answerable,
            unanswerable=unanswerable,
            certified=certified,
            reason=reason,
            carry_forward=provenance,
        )
        calibration_id = artifact.calibration_id
        with self._connect() as conn, conn.transaction():
            self._insert_payload_row(
                conn, payload, artifact.lifecycle_state, checksum, carried=True
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
            self._audit_if_rejected(conn, artifact)
        return self.get(calibration_id)

    def get(self, calibration_id: str) -> CalibrationArtifactV2:
        """One v2 artifact by id, with its checksum verified.

        A `legacy_unbound` row (no generation) is not an artifact and raises
        `CalibrationNotFound` here, although `list_records` and `show_record` return it; use
        `show_record` to inspect one.
        """
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
        """Make a certified artifact the one its generation serves, and return it.

        Accepts a `draft`, or a `superseded` artifact, which is how a rollback to an earlier
        threshold is done; any other published artifact for the generation is marked
        `superseded`, under a per-generation advisory lock. Refuses an uncertified artifact, and
        one whose generation's pipeline or corpus fingerprint changed since the fit. The
        generation row is read `FOR SHARE`, so a concurrent `forget()` either commits first (and
        this refuses on the changed fingerprint) or waits, and then supersedes what this published.
        """
        artifact = self.get(calibration_id)
        if not artifact.certified:
            raise CalibrationUncertified(artifact.certification_reason)
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"calibration\x1f{self.tenant_id}\x1f{artifact.generation_id}",),
            )
            lineage = self._generation(conn, artifact.generation_id, lock=True)
            if (
                lineage.pipeline_fingerprint != artifact.pipeline_fingerprint
                or lineage.corpus_fingerprint != artifact.corpus_fingerprint
            ):
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
        """Resolve on a connection of this repository's own, in a read-only snapshot."""
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self.resolve_within(conn, generation_id)

    def resolve_within(
        self, conn: psycopg.Connection, generation_id: str
    ) -> CalibrationResolution:
        """The same resolution, on a connection and transaction the CALLER already holds.

        ⚠️ **This exists so a gate can decide and act in one transaction.** `promote` locks the
        tenant row and the generation row `FOR UPDATE`, then asks whether the generation is
        certified. Asking through a second connection made that two decisions instead of one:

        * **The verdict came from a different snapshot than the write.** Between `resolve`
          returning CERTIFIED and the promotion committing, a concurrent `forget()` can rewrite the
          generation's `corpus_fingerprint` and make the calibration STALE. The gate would have
          approved a state that no longer existed by the time it was acted on. Read on the
          promoting connection, the generation row is the one the caller holds `FOR UPDATE`, so
          the check and the activation are atomic with respect to it.
        * **A connection was acquired while a tenant lock was held.** Measured: +1 backend for the
          width of the lock window, against `max_connections=100`. Under exhaustion the acquisition
          blocks, and it blocks holding the lock, so every other promotion and rollback for that
          tenant queues behind it rather than behind the work.

        Measured on the same probe: this is **not** a deadlock, which is what it looks like and is
        the thing worth stating. A plain `SELECT` does not wait on `FOR UPDATE` under MVCC, so the
        nested read completed and returned. The failure it produces is a wrong verdict under
        concurrency and a stall under load, not a hang.

        The caller owns the transaction, so no isolation level is set here: `promote`'s transaction
        has already written and cannot be made `REPEATABLE READ READ ONLY`. Reading at the caller's
        isolation is the point rather than a compromise, because the question being asked is about
        the caller's own transaction.
        """
        lineage = self._generation(conn, generation_id)
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
            if local is not None and str(local[1]) == LifecycleState.DRAFT:
                return CalibrationResolution(CalibrationStatus.DRAFT)
            if local is not None and str(local[1]) == LifecycleState.SUPERSEDED:
                return CalibrationResolution(CalibrationStatus.STALE)
            any_published = conn.execute(
                "SELECT 1 FROM recall_calibrations WHERE tenant_id = %s "
                "AND lifecycle_state = 'published' LIMIT 1",
                (self.tenant_id,),
            ).fetchone()
            return CalibrationResolution(
                CalibrationStatus.STALE if any_published else CalibrationStatus.MISSING
            )
        artifact = self._artifact(row)
        query_rows = self._query_set_rows(conn, artifact.query_set_digest)
        if (
            artifact.pipeline_fingerprint != lineage.pipeline_fingerprint
            or artifact.corpus_fingerprint != lineage.corpus_fingerprint
        ):
            return CalibrationResolution(CalibrationStatus.STALE)
        if query_rows is None:
            return CalibrationResolution(CalibrationStatus.STALE)
        _queries, digest = canonical_query_set(query_rows)
        if digest != artifact.query_set_digest:
            return CalibrationResolution(CalibrationStatus.STALE)
        if not artifact.certified:
            return CalibrationResolution(CalibrationStatus.UNCERTIFIED)
        return CalibrationResolution(CalibrationStatus.CERTIFIED, artifact)

    def list_records(self, generation_id: str | None = None) -> list[dict[str, Any]]:
        """Every calibration row of the tenant, newest first, or only one generation's.

        Rows are never deleted, so the unfiltered list grows with every fit; a caller that wants
        one generation passes `generation_id` and reads only that generation's rows.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT calibration_id, generation_id, lifecycle_state, certified, "
                "pipeline_fingerprint, corpus_fingerprint, query_set_digest, created_at, "
                "carry_forward FROM recall_calibrations WHERE tenant_id = %s "
                "AND (%s::text IS NULL OR generation_id = %s) "
                "ORDER BY created_at DESC, calibration_id",
                (self.tenant_id, generation_id, generation_id),
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
        # `list` about the same field. Re-render as the canonical UTC form: `created_at` is the
        # checksummed string, and `published_at` / `superseded_at` are set by `clock_timestamp()`
        # in `publish`, so all three arrive in whatever TimeZone the session happens to run in.
        for key in ("created_at", "published_at", "superseded_at"):
            item = value.get(key)
            if isinstance(item, str):
                value[key] = _utc_isoformat(datetime.fromisoformat(item))
        return value

    def measure_near_miss(
        self,
        calibration_id: str,
        queries: Sequence[Mapping[str, Any]],
        embedder: Embedder,
        *,
        record: bool = True,
    ) -> dict[str, Any]:
        """Score held-out near-miss questions against one calibration; report, never a gate.

        See `recall.near_miss_certificate`. The questions are scored on the calibration's OWN
        generation by the calibration sampling rule and compared with the answerable scores the
        artifact stores, at its threshold. With `record`, the report is appended to the audit
        ledger as one event bound to the calibration id: counts, scores and the set's digest,
        never a question's text. Nothing that decides serving reads that event.
        """
        from recall.near_miss_certificate import EVENT_TYPE, certificate_report

        artifact = self.get(calibration_id)
        labels, digest = canonical_query_set(queries)
        if any(item["answerable"] is not False for item in labels):
            raise CalibrationBindingError(
                "a near-miss set holds unanswerable questions only (answerable: false); an "
                "answerable entry would be scored as a near-miss"
            )
        with self._connect() as conn:
            lineage = self._generation(conn, artifact.generation_id)
        _require_runtime_embedder(PipelineIdentity.from_dict(lineage.pipeline_identity), embedder)
        _answerable, near_miss = self.score_query_set(artifact.generation_id, embedder, labels)
        report = certificate_report(
            artifact.scores["answerable"], near_miss, float(artifact.threshold)
        )
        report.update(generation_id=artifact.generation_id, query_set_digest=digest)
        if record:
            with self._connect() as conn:
                self._audit(conn, EVENT_TYPE, calibration_id, artifact.generation_id, report)
        return {"calibration_id": calibration_id, **report}

    def near_miss_report(self, calibration_id: str) -> dict[str, Any] | None:
        """The newest near-miss certificate recorded for this calibration id, or None.

        Validated against its own stored scores; raises ValueError for a row that disagrees with
        itself, so a reader can report it unavailable rather than print it.
        """
        from recall.near_miss_certificate import EVENT_TYPE, validated_certificate
        from recall.support import latest_calibration_event

        with self._connect() as conn:
            payload = latest_calibration_event(conn, self.tenant_id, EVENT_TYPE, calibration_id)
        return None if payload is None else validated_certificate(payload, calibration_id)

    def served_gap(
        self,
        calibration_id: str,
        embedder: Embedder,
        search: Callable[[str], tuple[float | None, bool]],
        *,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Compare served and exact best cosines over this calibration's labelled queries.

        See `recall.served_gap`. `search(query)` runs the search path being measured and returns
        `(best dense cosine among the hits it served, or None; abstained)`; it must search the
        calibration's own generation. The exact side is the calibration sampling rule. Read only:
        nothing is stored, and nothing that decides serving reads the result.
        """
        from recall.served_gap import DEFAULT_TOLERANCE, ServedGapRow, served_gap_report

        artifact = self.get(calibration_id)
        labels, digest = self.stored_query_set(artifact.query_set_digest)
        if digest != artifact.query_set_digest:
            raise CalibrationBindingError("stored labelled query set does not match its digest")
        with self._connect() as conn:
            lineage = self._generation(conn, artifact.generation_id)
        _require_runtime_embedder(PipelineIdentity.from_dict(lineage.pipeline_identity), embedder)
        answerable, unanswerable = self.score_query_set(artifact.generation_id, embedder, labels)
        # `measure_top_cosines` keeps each class in label order, so walking the labels in the same
        # order pairs every query with its own exact score.
        exact = {True: iter(answerable), False: iter(unanswerable)}
        rows = []
        for item in labels:
            served, abstained = search(str(item["query"]))
            rows.append(ServedGapRow(
                bool(item["answerable"]), next(exact[bool(item["answerable"])]), served, abstained
            ))
        report = served_gap_report(
            rows, float(artifact.threshold),
            DEFAULT_TOLERANCE if tolerance is None else tolerance,
        )
        report.update(calibration_id=calibration_id, generation_id=artifact.generation_id,
                      query_set_digest=digest)
        return report

    def export_bundle(self, calibration_id: str, path: str | Path) -> Path:
        artifact = self.get(calibration_id)
        with self._connect() as conn:
            rows = self._query_set_rows(conn, artifact.query_set_digest)
        if rows is None:
            raise CalibrationBindingError("calibration query set is missing")
        return write_calibration_bundle({"artifact": artifact.to_dict(), "query_set": rows}, path)

    def import_bundle(
        self,
        path: str | Path,
        embedder: Embedder | None = None,
        *,
        labels: Sequence[Mapping[str, Any]] | None = None,
        make_embedder: Callable[[], Embedder] | None = None,
    ) -> str:
        """Store a calibration from a bundle file, and return its calibration id.

        Two input forms. A v2 bundle (`export_bundle`'s output: an `artifact` and its
        `query_set`, sealed by `bundle_checksum`) is verified and stored. Any other JSON object is
        read as a pre-v2 file holding a bare threshold and scale and stored `legacy_unbound`,
        which can never be published or resolved (`_import_legacy`).

        A v2 bundle is refused unless its bundle checksum and artifact checksum verify, its field
        types are the ones a stored row reads back, its statistics re-derive from its own scores,
        and its tenant, generation lineage, embedder identity and query-set digest match this
        database. Its lifecycle is NOT imported: it arrives as `draft` when certified and
        `rejected` otherwise, so it serves nothing until `publish`.

        **Authentication.** The checks above prove only that the bundle agrees with itself, and
        its checksums are unkeyed. A forger can choose the queries and which of them count as
        answerable, so the labels are the operator's, not the bundle's: `labels` is the
        operator's own labelled set, whose digest must equal the bundle's, and which may not
        repeat a query. It is then scored on the named generation with the embedder that wrote
        its vectors (`embedder`, or `make_embedder()`, built only once every cheaper check has
        passed), the recorded scores must agree query by query (`_require_scores_reproduce`),
        and a certified bundle must certify on the fresh scores too
        (`_require_fresh_certification`). That costs one embedding call per labelled query. A
        carried bundle is accepted on the same evidence a local `carry_forward` requires; its
        parent may live in the exporting database and is not looked up. A legacy file needs no
        labels and no embedder, because nothing it holds can reach serving.

        Every refusal of the bundle itself is a `CalibrationError`; a database that has not been
        migrated raises `recall.schema.SchemaTooOld` instead. The CLI names both.
        """
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise CalibrationBindingError(f"cannot read calibration bundle {path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            # ValueError, not only JSONDecodeError: bytes that are not UTF-8 raise
            # UnicodeDecodeError, which is a ValueError and not a JSONDecodeError.
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
        if not all(isinstance(item, Mapping) for item in query_raw):
            raise CalibrationBindingError("bundle is malformed: query_set must hold objects")
        version = artifact_raw.get("artifact_version", 0)
        if type(version) is not int or version != ARTIFACT_VERSION:
            raise CalibrationBindingError("unsupported calibration artifact version")
        _require_bundle_field_types(artifact_raw)
        _bundle_labels, digest = canonical_query_set(query_raw)
        if digest != artifact_raw.get("query_set_digest"):
            raise CalibrationBindingError("imported query set digest does not match artifact")
        # Before any connection or embedding call, so a bundle whose labels the operator did not
        # supply costs nothing.
        if labels is None:
            raise CalibrationBindingError(
                "importing a calibration needs your own labelled query set (`--labels FILE`): a "
                "bundle's labels are not evidence, because whoever wrote it chose which queries "
                "count as answerable"
            )
        operator_labels, operator_digest = canonical_query_set(labels)
        if operator_digest != digest:
            raise CalibrationBindingError(
                f"your labelled query set ({operator_digest}) is not the one this bundle was "
                f"fitted on ({digest}); import only a calibration fitted on labels you hold"
            )
        require_distinct_scored_queries(operator_labels)
        if artifact_raw.get("tenant_id") != self.tenant_id:
            raise CalibrationBindingError("imported calibration belongs to another tenant")
        generation_id = str(artifact_raw.get("generation_id", ""))
        with self._connect() as conn:
            lineage = self._generation(conn, generation_id)
        identity = lineage.pipeline_identity
        if (
            artifact_raw.get("pipeline_fingerprint") != lineage.pipeline_fingerprint
            or artifact_raw.get("corpus_fingerprint") != lineage.corpus_fingerprint
        ):
            raise CalibrationBindingError("imported calibration does not match generation lineage")
        immutable_keys = CalibrationArtifactV2.__dataclass_fields__.keys()
        # Validated by name BEFORE construction: a missing key would otherwise surface as a bare
        # KeyError (from the separability_ci coercion) or TypeError (from the constructor), both
        # of which escape the CLI's `except CalibrationError` as an unnamed crash.
        optional_keys = {"artifact_version", "carry_forward", "lifecycle_state"}
        for key in immutable_keys:
            if key not in optional_keys and key not in artifact_raw:
                raise CalibrationBindingError(f"bundle is malformed: missing {key!r}")
        values = {key: artifact_raw[key] for key in immutable_keys if key in artifact_raw}
        # A list of two floats by now: `_require_bundle_field_types` checked its shape and the
        # loop above its presence.
        values["separability_ci"] = tuple(values["separability_ci"])
        values["lifecycle_state"] = (
            LifecycleState.DRAFT.value if values.get("certified") else LifecycleState.REJECTED.value
        )
        try:
            artifact = CalibrationArtifactV2(**values)
        except (TypeError, ValueError) as exc:
            raise CalibrationBindingError(f"bundle is malformed: {exc}") from exc
        artifact.verify_checksum()
        # The checksum above only proves internal consistency; it is unkeyed, so a forger can
        # recompute it over any content. The recorded statistics and the certification verdict
        # must also FOLLOW from the scores the bundle carries.
        _require_scores_support_certification(artifact)
        generation_embedder = identity.get("embedder")
        if not isinstance(generation_embedder, Mapping) or canonical_sha256(
            artifact.embedder_identity
        ) != canonical_sha256(generation_embedder):
            raise CalibrationBindingError("imported calibration embedder identity does not match")
        # Last, because it is the one check that costs embedding calls: everything cheaper has
        # already refused what it can.
        if embedder is None and make_embedder is not None:
            embedder = make_embedder()
        if embedder is None:
            raise CalibrationBindingError(
                "importing a calibration re-scores its labelled queries on the named generation, "
                "so it needs that generation's embedder (`recall --embedder <spec> calibration "
                "import ...`)"
            )
        _require_runtime_embedder(PipelineIdentity.from_dict(identity), embedder)
        fresh_answerable, fresh_unanswerable = self.score_query_set(
            generation_id, embedder, operator_labels
        )
        _require_scores_reproduce(artifact, fresh_answerable, fresh_unanswerable)
        _require_fresh_certification(artifact, fresh_answerable, fresh_unanswerable)
        with self._connect() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO recall_calibration_query_sets "
                "(tenant_id, query_set_digest, queries, sample_count, created_by) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    self.tenant_id,
                    digest,
                    Jsonb(list(operator_labels)),
                    len(operator_labels),
                    self.actor,
                ),
            )
            try:
                conn.execute(
                    # `_INSERT_ARTIFACT` writes `carry_forward` for the same reason it is
                    # checksummed: it is part of the immutable payload, so an import that dropped
                    # it would store a row whose checksum can never verify again, and the
                    # artifact would come back from `get` as a corruption rather than an import.
                    self._INSERT_ARTIFACT,
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
            except psycopg.errors.UniqueViolation as exc:
                # A re-import of an id that is already stored. Named rather than allowed to
                # escape as a raw UniqueViolation, which the CLI cannot catch and which reads
                # as a crash instead of as the benign fact it is.
                raise CalibrationBindingError(
                    f"calibration already imported: {artifact.calibration_id}"
                ) from exc
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
        # The range check mirrors the table's CHECK (threshold BETWEEN -1.0 AND 1.0): without it
        # a legacy file with a cosine outside that range reaches the INSERT and surfaces as a raw
        # CheckViolation instead of a named refusal at the boundary.
        if (
            not math.isfinite(threshold)
            or not math.isfinite(scale)
            or scale <= 0
            or not -1.0 <= threshold <= 1.0
        ):
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
