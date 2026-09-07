"""Generation administration boundary for MCP and desktop clients.

Generation ingest and calibration administration are owned here. The service module retains
compatibility aliases for existing imports.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import psycopg
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from recall.calibration_v2 import CalibrationRepository
from recall.embeddings import Embedder, embedder_artifact_digest
from recall.generations import (
    GenerationManager,
    InvalidGenerationTransition,
    NoActiveGeneration,
    UnsafePromotion,
)
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import ExtractingLocalObjectReader
from recall_mcp.models import IndexResult

if TYPE_CHECKING:
    from recall.store import PgVectorStore
_DESKTOP_CORPUS_PREFIX = "desktop-"


def _local_path(uri: str) -> Path | None:
    from recall.manifest import ObjectNotAllowed, local_path_for

    try:
        return local_path_for(uri)
    except ObjectNotAllowed:
        return None


def _roots_of(objects: dict[str, ManifestObjectV1]) -> tuple[Path, ...]:
    roots: dict[str, Path] = {}
    for uri in objects:
        path = _local_path(uri)
        if path is not None:
            roots.setdefault(str(path.parent), path.parent)
    return tuple(roots.values())


def _carry_forward(
    objects: dict[str, ManifestObjectV1],
) -> tuple[dict[str, ManifestObjectV1], tuple[Path, ...], int, int]:
    """Keep reachable objects, count vanished files, and restamp changed local files."""
    kept: dict[str, ManifestObjectV1] = {}
    vanished = 0
    restamped = 0
    for uri, entry in objects.items():
        local = _local_path(uri)
        if local is None:
            kept[uri] = entry
            continue
        try:
            stat = local.stat()
        except FileNotFoundError:
            vanished += 1
            continue
        except OSError:
            kept[uri] = entry
            continue
        digest = _digest_of(local)
        if stat.st_size == entry.size and digest == entry.sha256:
            kept[uri] = entry
        elif digest is None:
            kept[uri] = entry
        else:
            kept[uri] = replace(entry, version_id=digest, size=stat.st_size, sha256=digest)
            restamped += 1
    return kept, _roots_of(kept), vanished, restamped


def _digest_of(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _vanished_note(vanished: int) -> str:
    if not vanished:
        return ""
    return (
        f" ({vanished} file(s) from an earlier upload could not be re-read and are NOT in this "
        "build; re-upload them if you still need them)"
    )


def _restamped_note(restamped: int) -> str:
    if not restamped:
        return ""
    return f" ({restamped} file(s) changed since they were indexed and were re-read)"


def _query_set_for(chunks: list[str]) -> tuple[list[dict[str, object]] | None, Exception | None]:
    from recall.wizard.queryset import (
        DEFAULT_PER_CLASS,
        MIN_PER_CLASS,
        QuerySetError,
        canonicalize,
        generate_offline,
    )

    last: Exception | None = None
    for per_class in (DEFAULT_PER_CLASS, MIN_PER_CLASS):
        try:
            return canonicalize(generate_offline(chunks, per_class=per_class)), None
        except QuerySetError as exc:
            last = exc
    return None, last


def _certify_upload(
    dsn: str,
    tenant: str,
    generation_id: str,
    embedder: Embedder,
) -> str | None:
    """Calibrate and publish a desktop generation, returning a bounded refusal reason."""
    from recall.calibration_v2 import CalibrationError, CalibrationUncertified

    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY chunk_id",
            (tenant, generation_id),
        ).fetchall()
    chunks = [str(row[0]) for row in rows if str(row[0]).strip()]
    entries, last = _query_set_for(chunks)
    if entries is None:
        return f"no certifiable query set could be generated from {len(chunks)} chunk(s): {last}"
    repository = CalibrationRepository(dsn, tenant, actor="recall-desktop")
    try:
        artifact = repository.calibrate(generation_id, entries, embedder)
        if not artifact.certified:
            return f"calibration was not certified: {artifact.certification_reason}"
        repository.publish(artifact.calibration_id)
    except CalibrationUncertified as exc:
        return f"calibration was not certified: {exc}"
    except CalibrationError as exc:
        return f"calibration could not be completed: {exc}"
    return None


def _reclaim_failed(manager: GenerationManager, generation_id: str, reason: str) -> None:
    try:
        manager.fail(generation_id, reason)
        return
    except InvalidGenerationTransition:
        pass
    except Exception:  # noqa: BLE001  # BROAD-CATCH: cleanup-only
        return
    with suppress(Exception):
        manager.abandon(generation_id, reason)


def _release_superseded(manager: GenerationManager, keep: str) -> int:
    reclaimed = 0
    try:
        stale = manager.superseded_ready_generations(
            keep, corpus_version_prefix=_DESKTOP_CORPUS_PREFIX
        )
    except Exception:  # noqa: BLE001  # BROAD-CATCH: cleanup-only
        return 0
    for generation_id in stale:
        try:
            manager.abandon(generation_id, "superseded by a later desktop upload")
        except Exception:  # noqa: BLE001  # BROAD-CATCH: cleanup-only
            continue
        reclaimed += 1
    return reclaimed


def generation_ingest(
    store: PgVectorStore,
    embedder: Embedder,
    staged_root: str,
    category: str,
) -> IndexResult:
    """Build, validate, and activate one local generation for a desktop upload."""
    job_root = Path(staged_root)
    tenant_root = job_root.parent
    job_files = sorted(path for path in job_root.rglob("*") if path.is_file())
    if not job_files:
        raise ValueError("the staged upload contains no files")

    manager = GenerationManager(
        store._dsn,
        store.tenant,
        actor="recall-desktop",
        serving_environment=os.environ.get("RECALL_SERVING_ENV", os.environ.get("RECALL_ENV")),
    )
    with manager.tenant_ingest_lock():
        try:
            base = manager.servable_manifest()
            active_objects, carried_roots, vanished, restamped = _carry_forward(
                {entry.uri: entry for entry in base.objects}
            )
        except NoActiveGeneration:
            active_objects, carried_roots, vanished, restamped = {}, (), 0, 0

        for path in job_files:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            media_type = (
                "text/x-code"
                if category == "code"
                else mimetypes.guess_type(path.name)[0] or "text/plain"
            )
            entry = ManifestObjectV1(
                uri=path.resolve().as_uri(),
                version_id=digest,
                media_type=media_type,
                size=len(data),
                sha256=digest,
            )
            active_objects[entry.uri] = entry

        from recall.generation_build import BuildRequest, pipeline_for

        artifact_digest = embedder_artifact_digest(embedder)
        chunker, pipeline = pipeline_for(
            embedder,
            BuildRequest(
                chunker="code" if category == "code" else "text",
                artifact_digest=artifact_digest,
                unverified=artifact_digest is None,
            ),
        )
        manifest = IndexManifestV1(
            tenant_id=store.tenant,
            corpus_version=f"{_DESKTOP_CORPUS_PREFIX}"
            f"{hashlib.sha256(job_root.name.encode()).hexdigest()[:12]}",
            objects=tuple(sorted(active_objects.values(), key=lambda entry: entry.uri)),
        )
        generation = manager.create(manifest, pipeline, allow_unverified=not pipeline.verified)
        try:
            stats = manager.build(
                generation.generation_id,
                ExtractingLocalObjectReader((tenant_root, *carried_roots)),
                embedder,
                chunker,
            )
            manager.validate(generation.generation_id)
            uncertified: str | None = None
            if manager.certification_required:
                uncertified = _certify_upload(
                    store._dsn, store.tenant, generation.generation_id, embedder
                )
            try:
                manager.promote(
                    generation.generation_id,
                    unsafe_development=not manager.certification_required,
                )
            except UnsafePromotion as exc:
                reclaimed = _release_superseded(manager, generation.generation_id)
                return IndexResult(
                    files=stats.objects,
                    chunks=stats.chunks,
                    message=(
                        f"Indexed {stats.chunks} chunk(s) from {stats.objects} file(s) into "
                        f"generation {generation.generation_id}, built and validated but not live. "
                        f"It carries forward everything previously uploaded"
                        + _vanished_note(vanished)
                        + _restamped_note(restamped)
                        + (f"; {reclaimed} superseded build(s) released" if reclaimed else "")
                        + f". {uncertified or exc}"
                    ),
                )
        except Exception as exc:  # BROAD-CATCH: fail-closed
            _reclaim_failed(manager, generation.generation_id, f"desktop upload failed: {exc}")
            raise

        _release_superseded(manager, generation.generation_id)
        return IndexResult(
            files=stats.objects,
            chunks=stats.chunks,
            message=(
                f"Built and activated generation {generation.generation_id} with "
                f"{stats.chunks} chunk(s) from {stats.objects} file(s)."
                + _vanished_note(vanished)
                + _restamped_note(restamped)
            ),
        )


def _generated_calibration_queries(store: PgVectorStore, generation_id: str) -> list[dict[str, object]]:
    """Build a deterministic draft query set from the active corpus.

    This is intentionally a prototype helper. The generated labels are useful for checking the
    complete workflow, but a production deployment should replace them with reviewed labels.
    """
    with psycopg.connect(store._dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,))
        rows = conn.execute(
            "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY chunk_id LIMIT 20",
            (store.tenant, generation_id),
        ).fetchall()
    answerable: list[str] = []
    for row in rows:
        value = str(row[0]).strip()
        if value and value not in answerable:
            answerable.append(value[:500])
    if len(answerable) < 2:
        raise ValueError("at least two distinct corpus chunks are required to generate calibration labels")
    return [
        *({"query": query, "answerable": True} for query in answerable),
        *(
            {
                "query": f"Prototype calibration negative sample {index}: {nonce}",
                "answerable": False,
            }
            for index, nonce in enumerate(
                (
                    "the unrecorded weather on Europa",
                    "the private password for a fictional account",
                    "the exact weight of an imaginary blue comet",
                    "the inventory of a library that does not exist",
                    "the recipe for a machine never described here",
                    "the birthplace of a person absent from this corpus",
                    "the result of a future election",
                    "the serial number of a nonexistent device",
                    "the internal schedule of an unrelated company",
                    "the answer to an invented mathematical riddle",
                    "the color of a silent radio signal",
                    "the number of doors in an imaginary building",
                    "the owner of a fictional island",
                    "the temperature inside an empty thought",
                    "the name of a removed document",
                    "the location of a lost moon",
                    "the version of an unreleased program",
                    "the price of an unnamed object",
                    "the title of a nonexistent chapter",
                    "the identity of an imaginary maintainer",
                ),
                start=1,
            )
        ),
    ]


def run_calibration(
    store: PgVectorStore,
    embedder: Embedder,
    generation_id: str | None = None,
    queries: Sequence[dict[str, object]] | None = None,
    *,
    _generated_calibration_queries_fn=_generated_calibration_queries,
) -> dict[str, object]:
    """Measure a draft artifact, generating prototype labels when none were supplied."""
    from recall.generation_store import GenerationStore

    generation_store = GenerationStore(store._dsn, embedder.dim, tenant=store.tenant)
    try:
        selected_generation = generation_id or generation_store.active_generation_id()
    finally:
        generation_store.close()
    labels = list(queries) if queries is not None else _generated_calibration_queries_fn(store, selected_generation)
    artifact = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp").calibrate(
        selected_generation,
        labels,
        embedder,
    )
    return artifact.to_dict()


def publish_calibration(store: PgVectorStore, calibration_id: str) -> dict[str, object]:
    """Publish a certified artifact after the user explicitly confirms the action."""
    artifact = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp").publish(calibration_id)
    return artifact.to_dict()


__all__ = ["generation_ingest", "publish_calibration", "run_calibration"]
