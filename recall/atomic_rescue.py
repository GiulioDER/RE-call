"""Generation-bound atomic fact rescue artifacts and exact candidate selection.

The module deliberately imports NumPy only while an explicitly configured artifact is loaded.
An ordinary RE-call process with atomic rescue disabled therefore keeps its original import,
allocation, and request cost surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from recall.embeddings import Embedder, embedding_profile, embedding_profile_id
from recall.errors import RecallError
from recall.types import ScoredChunk, TrustedResult


ATOMIC_RESCUE_SCHEMA_VERSION = 1


class AtomicRescueArtifactError(ValueError, RecallError):
    """The configured atomic rescue artifact is absent, malformed, or stale."""


class AtomicRescueLineageError(AtomicRescueArtifactError):
    """The artifact is valid but does not match the serving lineage."""


class AtomicRescueSelectionError(RuntimeError, RecallError):
    """A valid artifact could not produce a bounded rescue selection."""


@dataclass(frozen=True)
class AtomicRescueView:
    """The nontext identity of one embedded atomic view."""

    chunk_id: str
    source: str
    parent_ordinal: int
    view_ordinal: int


@dataclass(frozen=True)
class AtomicRescueSelection:
    """One exact nonexcluded parent selected from the atomic matrix."""

    chunk_id: str
    source: str
    parent_ordinal: int
    view_ordinal: int
    score: float


@dataclass(frozen=True)
class AtomicRescueArtifact:
    """A validated immutable matrix and its generation-bound parent identities."""

    manifest_path: Path
    generation_id: str
    calibration_id: str
    pipeline_fingerprint: str
    corpus_fingerprint: str
    embedding_profile: str
    embedding_fingerprint: str
    dimension: int
    ordinary_chunk_count: int
    views: tuple[AtomicRescueView, ...]
    matrix: Any
    parent_codes: Any
    code_by_chunk_id: Mapping[str, int]
    matrix_sha256: str
    metadata_sha256: str
    source_commit: str
    constructed_at: str
    load_ms: float
    resident_memory_delta_bytes: int

    @property
    def view_count(self) -> int:
        return len(self.views)

    @property
    def parent_count(self) -> int:
        return len(self.code_by_chunk_id)

    def assert_compatible(
        self,
        *,
        result: TrustedResult,
        embedder: Embedder,
    ) -> None:
        """Refuse an artifact that does not describe the current serving lineage."""

        self.assert_lineage(
            generation_id=result.generation_id,
            calibration_id=result.calibration_id,
            pipeline_fingerprint=result.pipeline_fingerprint,
            corpus_fingerprint=result.corpus_fingerprint,
            embedder=embedder,
        )

    def assert_lineage(
        self,
        *,
        generation_id: str | None,
        calibration_id: str | None,
        pipeline_fingerprint: str | None,
        corpus_fingerprint: str | None,
        embedder: Embedder,
    ) -> None:
        """Refuse active use unless every serving identity matches the artifact."""

        expected = {
            "generation_id": generation_id,
            "calibration_id": calibration_id,
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "embedding_profile": embedding_profile_id(embedder),
            "embedding_fingerprint": embedding_profile(embedder).fingerprint(),
            "dimension": int(embedder.dim),
        }
        actual = {
            "generation_id": self.generation_id,
            "calibration_id": self.calibration_id,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "embedding_profile": self.embedding_profile,
            "embedding_fingerprint": self.embedding_fingerprint,
            "dimension": self.dimension,
        }
        mismatches = [name for name, value in expected.items() if actual[name] != value]
        if mismatches:
            raise AtomicRescueLineageError(
                "atomic rescue artifact lineage mismatch: " + ", ".join(mismatches)
            )


def _atomic_artifact_path_component(value: str, name: str) -> str:
    """Return one registry component, rejecting traversal on either host platform."""

    if (
        not value
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise AtomicRescueArtifactError(
            f"atomic rescue {name} is not a path segment"
        )
    return value


def _atomic_artifact_corpus_fingerprint(value: str) -> str:
    """Return a canonical SHA256 fingerprint safe for one registry component."""

    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AtomicRescueArtifactError(
            "atomic rescue corpus fingerprint is not a lower case SHA256 digest"
        )
    return value


def resolve_atomic_rescue_manifest(
    root: str | Path,
    generation_id: str,
    *,
    scope_id: str | None = None,
    corpus_fingerprint: str | None = None,
) -> Path:
    """Resolve one generation manifest, optionally within one opaque corpus scope."""

    generation_id = _atomic_artifact_path_component(generation_id, "generation id")
    if (scope_id is None) != (corpus_fingerprint is None):
        raise AtomicRescueArtifactError(
            "atomic rescue scope and corpus fingerprint must be configured together"
        )
    resolved_root = Path(root).expanduser().resolve()
    scope_root = resolved_root
    if scope_id is not None and corpus_fingerprint is not None:
        scope_id = _atomic_artifact_path_component(scope_id, "scope id")
        scope_root = (resolved_root / scope_id).resolve()
        if scope_root.parent != resolved_root:
            raise AtomicRescueArtifactError("atomic rescue scope escapes artifact root")
        corpus_fingerprint = _atomic_artifact_corpus_fingerprint(corpus_fingerprint)
    generation_root = (scope_root / generation_id).resolve()
    if generation_root.parent != scope_root:
        raise AtomicRescueArtifactError("atomic rescue generation escapes artifact root")
    artifact_root = generation_root
    if corpus_fingerprint is not None:
        artifact_root = (generation_root / corpus_fingerprint).resolve()
        if artifact_root.parent != generation_root:
            raise AtomicRescueArtifactError("atomic rescue corpus fingerprint escapes generation root")
    manifest = (artifact_root / "manifest.json").resolve()
    if manifest.parent != artifact_root:
        raise AtomicRescueArtifactError("atomic rescue manifest escapes generation root")
    return manifest


_ARTIFACT_CACHE: dict[Path, AtomicRescueArtifact] = {}
_ARTIFACT_CACHE_LOCK = threading.Lock()
_SELECTION_LOCK = threading.Lock()
_EXPECTATION_CACHE: dict[Path, Mapping[str, tuple[str, int, float]]] = {}
_EXPECTATION_CACHE_LOCK = threading.Lock()


def clear_atomic_rescue_artifact_cache() -> None:
    """Clear the process cache for isolated tests."""

    with _ARTIFACT_CACHE_LOCK:
        _ARTIFACT_CACHE.clear()
    with _EXPECTATION_CACHE_LOCK:
        _EXPECTATION_CACHE.clear()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resident_bytes() -> int:
    """Return current Linux resident bytes when available, otherwise zero."""

    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        sysconf = getattr(os, "sysconf", None)
        if sysconf is None:
            return 0
        return resident_pages * int(sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def _artifact_member(root: Path, raw: object, field: str) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or "/" in raw
        or "\\" in raw
        or Path(raw).name != raw
    ):
        raise AtomicRescueArtifactError(f"atomic rescue {field} must be a local filename")
    path = (root / raw).resolve()
    if path.parent != root:
        raise AtomicRescueArtifactError(f"atomic rescue {field} escapes the artifact directory")
    return path


def _string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise AtomicRescueArtifactError(f"atomic rescue manifest lacks {name}")
    return value


def _integer(payload: Mapping[str, object], name: str, *, minimum: int = 1) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AtomicRescueArtifactError(f"atomic rescue manifest has invalid {name}")
    return value


def _load_atomic_rescue_artifact(path: Path) -> AtomicRescueArtifact:
    started = time.perf_counter()
    resident_before = _resident_bytes()
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover, exercised in a minimal package smoke test
        raise AtomicRescueArtifactError(
            "atomic rescue requires the recall-rag[atomic] optional dependency"
        ) from exc

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicRescueArtifactError("atomic rescue manifest is unreadable") from exc
    if not isinstance(decoded, dict):
        raise AtomicRescueArtifactError("atomic rescue manifest must be a JSON object")
    if decoded.get("schema_version") != ATOMIC_RESCUE_SCHEMA_VERSION:
        raise AtomicRescueArtifactError("atomic rescue manifest schema is unsupported")

    root = path.parent.resolve()
    matrix_path = _artifact_member(root, decoded.get("matrix_file"), "matrix_file")
    metadata_path = _artifact_member(root, decoded.get("metadata_file"), "metadata_file")
    matrix_digest = _string(decoded, "matrix_sha256")
    metadata_digest = _string(decoded, "metadata_sha256")
    try:
        if _sha256(matrix_path) != matrix_digest:
            raise AtomicRescueArtifactError("atomic rescue matrix digest mismatch")
        if _sha256(metadata_path) != metadata_digest:
            raise AtomicRescueArtifactError("atomic rescue metadata digest mismatch")
    except OSError as exc:
        raise AtomicRescueArtifactError("atomic rescue artifact member is unreadable") from exc

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicRescueArtifactError("atomic rescue metadata is unreadable") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise AtomicRescueArtifactError("atomic rescue metadata schema is unsupported")
    raw_views = metadata.get("views")
    if not isinstance(raw_views, list):
        raise AtomicRescueArtifactError("atomic rescue metadata lacks views")

    views: list[AtomicRescueView] = []
    code_by_chunk_id: dict[str, int] = {}
    identity_by_chunk_id: dict[str, tuple[str, int]] = {}
    parent_codes: Any = np.empty(len(raw_views), dtype=np.int32)
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, dict):
            raise AtomicRescueArtifactError("atomic rescue view must be an object")
        chunk_id = _string(raw_view, "chunk_id")
        source = _string(raw_view, "source")
        parent_ordinal = _integer(raw_view, "parent_ordinal", minimum=0)
        view_ordinal = _integer(raw_view, "view_ordinal", minimum=0)
        identity = (source, parent_ordinal)
        previous_identity = identity_by_chunk_id.setdefault(chunk_id, identity)
        if previous_identity != identity:
            raise AtomicRescueArtifactError(
                "atomic rescue parent identities disagree with chunk ids"
            )
        code = code_by_chunk_id.setdefault(chunk_id, len(code_by_chunk_id))
        parent_codes[index] = code
        views.append(
            AtomicRescueView(chunk_id, source, parent_ordinal, view_ordinal)
        )
    if not views:
        raise AtomicRescueArtifactError("atomic rescue artifact has no views")
    if len({(view.source, view.parent_ordinal) for view in views}) != len(code_by_chunk_id):
        raise AtomicRescueArtifactError("atomic rescue parent identities disagree with chunk ids")

    try:
        matrix = np.load(matrix_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AtomicRescueArtifactError("atomic rescue matrix is unreadable") from exc
    if matrix.dtype != np.dtype("float32") or matrix.ndim != 2:
        raise AtomicRescueArtifactError("atomic rescue matrix must be two dimensional float32")
    dimension = _integer(decoded, "dimension")
    if matrix.shape != (len(views), dimension):
        raise AtomicRescueArtifactError("atomic rescue matrix shape mismatch")
    if not np.all(np.isfinite(matrix)):
        raise AtomicRescueArtifactError("atomic rescue matrix contains nonfinite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
        raise AtomicRescueArtifactError("atomic rescue matrix rows are not normalized")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    view_count = _integer(decoded, "view_count")
    parent_count = _integer(decoded, "parent_count")
    if view_count != len(views) or parent_count != len(code_by_chunk_id):
        raise AtomicRescueArtifactError("atomic rescue manifest counts disagree with metadata")
    load_ms = (time.perf_counter() - started) * 1000.0
    resident_after = _resident_bytes()
    return AtomicRescueArtifact(
        manifest_path=path,
        generation_id=_string(decoded, "generation_id"),
        calibration_id=_string(decoded, "calibration_id"),
        pipeline_fingerprint=_string(decoded, "pipeline_fingerprint"),
        corpus_fingerprint=_string(decoded, "corpus_fingerprint"),
        embedding_profile=_string(decoded, "embedding_profile"),
        embedding_fingerprint=_string(decoded, "embedding_fingerprint"),
        dimension=dimension,
        ordinary_chunk_count=_integer(decoded, "ordinary_chunk_count"),
        views=tuple(views),
        matrix=matrix,
        parent_codes=parent_codes,
        code_by_chunk_id=code_by_chunk_id,
        matrix_sha256=matrix_digest,
        metadata_sha256=metadata_digest,
        source_commit=_string(decoded, "source_commit"),
        constructed_at=_string(decoded, "constructed_at"),
        load_ms=load_ms,
        resident_memory_delta_bytes=max(0, resident_after - resident_before),
    )


def load_atomic_rescue_artifact(path: str | Path) -> AtomicRescueArtifact:
    """Load and validate one artifact exactly once per process."""

    resolved = Path(path).expanduser().resolve()
    with _ARTIFACT_CACHE_LOCK:
        cached = _ARTIFACT_CACHE.get(resolved)
        if cached is None:
            cached = _load_atomic_rescue_artifact(resolved)
            _ARTIFACT_CACHE[resolved] = cached
        return cached


def select_atomic_rescue(
    artifact: AtomicRescueArtifact,
    query_vector: Sequence[float],
    dense: Sequence[ScoredChunk],
) -> AtomicRescueSelection:
    """Select one exact winner while bounding concurrent matrix CPU contention."""

    with _SELECTION_LOCK:
        return _select_atomic_rescue_unlocked(artifact, query_vector, dense)


def _select_atomic_rescue_unlocked(
    artifact: AtomicRescueArtifact,
    query_vector: Sequence[float],
    dense: Sequence[ScoredChunk],
) -> AtomicRescueSelection:
    """Select the exact best atomic parent outside the first five dense parents."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise AtomicRescueArtifactError(
            "atomic rescue requires the recall-rag[atomic] optional dependency"
        ) from exc
    if len(dense) < 5:
        raise AtomicRescueSelectionError("atomic rescue requires five dense candidates")
    dense_ids = [hit.chunk.id for hit in dense[:5]]
    if len(set(dense_ids)) != 5:
        raise AtomicRescueSelectionError("atomic rescue dense prefix repeats a parent")
    query: Any = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != artifact.dimension:
        raise AtomicRescueSelectionError("atomic rescue query dimension mismatch")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm == 0.0:
        raise AtomicRescueSelectionError("atomic rescue query has nonfinite or zero norm")
    query = np.ascontiguousarray(query / norm, dtype=np.float32)
    scores = artifact.matrix @ query
    if not np.all(np.isfinite(scores)):
        raise AtomicRescueSelectionError("atomic rescue produced nonfinite scores")

    excluded: Any = np.zeros(artifact.view_count, dtype=np.bool_)
    for chunk_id in dense_ids:
        code = artifact.code_by_chunk_id.get(chunk_id)
        if code is not None:
            excluded |= artifact.parent_codes == code
    valid = ~excluded
    if not np.any(valid):
        raise AtomicRescueSelectionError("atomic rescue has no parent outside dense top five")
    best_score = np.max(scores[valid])
    tied = np.flatnonzero(valid & (scores == best_score))
    winner = min(
        (int(index) for index in tied),
        key=lambda index: (
            artifact.views[index].source,
            artifact.views[index].parent_ordinal,
            artifact.views[index].view_ordinal,
        ),
    )
    view = artifact.views[winner]
    return AtomicRescueSelection(
        view.chunk_id,
        view.source,
        view.parent_ordinal,
        view.view_ordinal,
        float(scores[winner]),
    )


def atomic_rescue_reference_parity(
    artifact: AtomicRescueArtifact,
    query_vector: Sequence[float],
    dense: Sequence[ScoredChunk],
    selection: AtomicRescueSelection,
) -> tuple[bool, bool]:
    """Compare the fast selection with a deterministic same-vector full sort."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise AtomicRescueArtifactError(
            "atomic rescue requires the recall-rag[atomic] optional dependency"
        ) from exc
    if len(dense) < 5:
        raise AtomicRescueSelectionError("atomic rescue requires five dense candidates")
    dense_ids = [hit.chunk.id for hit in dense[:5]]
    if len(set(dense_ids)) != 5:
        raise AtomicRescueSelectionError("atomic rescue dense prefix repeats a parent")
    query: Any = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != artifact.dimension:
        raise AtomicRescueSelectionError("atomic rescue query dimension mismatch")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm == 0.0:
        raise AtomicRescueSelectionError("atomic rescue query has nonfinite or zero norm")
    query = np.ascontiguousarray(query / norm, dtype=np.float32)
    scores = artifact.matrix @ query
    if not np.all(np.isfinite(scores)):
        raise AtomicRescueSelectionError("atomic rescue produced nonfinite scores")

    excluded = set(dense_ids)
    order = sorted(
        range(artifact.view_count),
        key=lambda index: (
            -float(scores[index]),
            artifact.views[index].source,
            artifact.views[index].parent_ordinal,
            artifact.views[index].view_ordinal,
        ),
    )
    for index in order:
        view = artifact.views[index]
        if view.chunk_id in excluded:
            continue
        return (
            (selection.source, selection.parent_ordinal, selection.view_ordinal)
            == (view.source, view.parent_ordinal, view.view_ordinal),
            selection.score == float(scores[index]),
        )
    raise AtomicRescueSelectionError("atomic rescue has no parent outside dense top five")


def insert_atomic_rescue_dense(
    artifact: AtomicRescueArtifact,
    query_vector: Sequence[float],
    dense: Sequence[ScoredChunk],
    hit_loader: Callable[[str, float], ScoredChunk | None],
) -> list[ScoredChunk]:
    """Move the exact atomic winner to dense rank six and retain every other parent."""

    selection = select_atomic_rescue(artifact, query_vector, dense)
    rescue = hit_loader(selection.chunk_id, selection.score)
    if rescue is None or rescue.chunk.id != selection.chunk_id:
        raise AtomicRescueSelectionError("atomic rescue selected parent is unavailable")
    later = [hit for hit in dense[5:] if hit.chunk.id != selection.chunk_id]
    return [*dense[:5], rescue, *later]


@dataclass(frozen=True)
class GatedAtomicRescueSelection:
    """A rescue admitted by the dense-rank gate, with the evidence for admitting it."""

    selection: AtomicRescueSelection
    probe_index: int
    margin: float


def select_gated_atomic_rescue(
    artifact: AtomicRescueArtifact,
    probes: Sequence[tuple[Sequence[float], Sequence[ScoredChunk]]],
    protected: Sequence[ScoredChunk],
    *,
    gate_rank: int = 5,
) -> GatedAtomicRescueSelection | None:
    """Admit the best atomic parent only when it beats a whole window under the same vector.

    ``protected`` is the served query's dense ranking; its first five parents are never
    rescued. Each probe is one query vector (the query itself, then any decomposed
    sub-questions) with that vector's own dense ranking. For every probe the best view outside
    the protected parents is compared with the probe's ``gate_rank``-th best whole window, and
    the probe with the largest non-negative margin wins. Comparing a view only against windows
    scored by the same vector keeps the gate free of cross-query score comparisons.

    Returns None when no probe clears its gate, which callers must treat as "leave the dense
    ranking unchanged", not as a failure.
    """

    with _SELECTION_LOCK:
        return _select_gated_unlocked(artifact, probes, protected, gate_rank=gate_rank)


def _select_gated_unlocked(
    artifact: AtomicRescueArtifact,
    probes: Sequence[tuple[Sequence[float], Sequence[ScoredChunk]]],
    protected: Sequence[ScoredChunk],
    *,
    gate_rank: int,
) -> GatedAtomicRescueSelection | None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise AtomicRescueArtifactError(
            "atomic rescue requires the recall-rag[atomic] optional dependency"
        ) from exc
    if not probes:
        raise AtomicRescueSelectionError("gated atomic rescue requires at least one probe")
    if gate_rank < 1:
        raise AtomicRescueSelectionError("gated atomic rescue gate rank must be positive")
    if len(protected) < 5:
        raise AtomicRescueSelectionError("atomic rescue requires five dense candidates")
    protected_ids = [hit.chunk.id for hit in protected[:5]]
    if len(set(protected_ids)) != 5:
        raise AtomicRescueSelectionError("atomic rescue dense prefix repeats a parent")
    excluded: Any = np.zeros(artifact.view_count, dtype=np.bool_)
    for chunk_id in protected_ids:
        code = artifact.code_by_chunk_id.get(chunk_id)
        if code is not None:
            excluded |= artifact.parent_codes == code
    valid = ~excluded
    if not np.any(valid):
        raise AtomicRescueSelectionError("atomic rescue has no parent outside dense top five")

    best: tuple[float, int, int, float] | None = None  # margin, probe, view index, score
    for probe_index, (vector, dense) in enumerate(probes):
        if len(dense) < gate_rank:
            raise AtomicRescueSelectionError("gated atomic rescue probe lacks its gate window")
        query: Any = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != artifact.dimension:
            raise AtomicRescueSelectionError("atomic rescue query dimension mismatch")
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm == 0.0:
            raise AtomicRescueSelectionError("atomic rescue query has nonfinite or zero norm")
        scores = artifact.matrix @ np.ascontiguousarray(query / norm, dtype=np.float32)
        if not np.all(np.isfinite(scores)):
            raise AtomicRescueSelectionError("atomic rescue produced nonfinite scores")
        best_score = np.max(scores[valid])
        tied = np.flatnonzero(valid & (scores == best_score))
        winner = min(
            (int(index) for index in tied),
            key=lambda index: (
                artifact.views[index].source,
                artifact.views[index].parent_ordinal,
                artifact.views[index].view_ordinal,
            ),
        )
        margin = float(best_score) - float(dense[gate_rank - 1].score)
        if best is None or margin > best[0]:
            best = (margin, probe_index, winner, float(best_score))
    assert best is not None
    margin, probe_index, winner, score = best
    if margin < 0.0:
        return None
    view = artifact.views[winner]
    return GatedAtomicRescueSelection(
        AtomicRescueSelection(
            view.chunk_id, view.source, view.parent_ordinal, view.view_ordinal, score
        ),
        probe_index,
        margin,
    )


def insert_gated_atomic_rescue_dense(
    artifact: AtomicRescueArtifact,
    probes: Sequence[tuple[Sequence[float], Sequence[ScoredChunk]]],
    dense: Sequence[ScoredChunk],
    hit_loader: Callable[[str, float], ScoredChunk | None],
    *,
    gate_rank: int = 5,
) -> tuple[list[ScoredChunk], GatedAtomicRescueSelection | None]:
    """Insert an admitted rescue at dense rank six; return the ranking unchanged otherwise."""

    gated = select_gated_atomic_rescue(artifact, probes, dense, gate_rank=gate_rank)
    if gated is None:
        return list(dense), None
    rescue = hit_loader(gated.selection.chunk_id, gated.selection.score)
    if rescue is None or rescue.chunk.id != gated.selection.chunk_id:
        raise AtomicRescueSelectionError("atomic rescue selected parent is unavailable")
    later = [hit for hit in dense[5:] if hit.chunk.id != gated.selection.chunk_id]
    return [*dense[:5], rescue, *later], gated


def select_view_gated_atomic_rescue(
    artifact: AtomicRescueArtifact,
    query_vector: Sequence[float],
    dense: Sequence[ScoredChunk],
) -> GatedAtomicRescueSelection | None:
    """Admit the best outside view only if it matches like the protected parents' own views.

    The window-rank gate compares a short view with a 160-word window, and short texts score
    systematically higher under the same query, so it admitted about 99% of probes (measured
    2026-09-22). This gate compares views with views: the reference is the weakest of the
    protected parents' best view scores, so a rescue is admitted only when its best fact matches
    the query at least as well as the facts already in the protected top five do. Protected
    parents without views do not contribute; if none has a view there is no like-for-like
    reference and the rescue is refused. None means leave the dense ranking unchanged.
    """

    with _SELECTION_LOCK:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise AtomicRescueArtifactError(
                "atomic rescue requires the recall-rag[atomic] optional dependency"
            ) from exc
        if len(dense) < 5:
            raise AtomicRescueSelectionError("atomic rescue requires five dense candidates")
        protected_ids = [hit.chunk.id for hit in dense[:5]]
        if len(set(protected_ids)) != 5:
            raise AtomicRescueSelectionError("atomic rescue dense prefix repeats a parent")
        query: Any = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != artifact.dimension:
            raise AtomicRescueSelectionError("atomic rescue query dimension mismatch")
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm == 0.0:
            raise AtomicRescueSelectionError("atomic rescue query has nonfinite or zero norm")
        scores = artifact.matrix @ np.ascontiguousarray(query / norm, dtype=np.float32)
        if not np.all(np.isfinite(scores)):
            raise AtomicRescueSelectionError("atomic rescue produced nonfinite scores")
        excluded: Any = np.zeros(artifact.view_count, dtype=np.bool_)
        protected_best: list[float] = []
        for chunk_id in protected_ids:
            code = artifact.code_by_chunk_id.get(chunk_id)
            if code is None:
                continue
            mask = artifact.parent_codes == code
            excluded |= mask
            protected_best.append(float(np.max(scores[mask])))
        valid = ~excluded
        if not protected_best or not np.any(valid):
            return None
        reference = min(protected_best)
        best_score = np.max(scores[valid])
        tied = np.flatnonzero(valid & (scores == best_score))
        winner = min(
            (int(index) for index in tied),
            key=lambda index: (
                artifact.views[index].source,
                artifact.views[index].parent_ordinal,
                artifact.views[index].view_ordinal,
            ),
        )
        margin = float(best_score) - reference
        if margin < 0.0:
            return None
        view = artifact.views[winner]
        return GatedAtomicRescueSelection(
            AtomicRescueSelection(
                view.chunk_id, view.source, view.parent_ordinal, view.view_ordinal, float(best_score)
            ),
            0,
            margin,
        )


def insert_view_gated_atomic_rescue_dense(
    artifact: AtomicRescueArtifact,
    query_vector: Sequence[float],
    dense: Sequence[ScoredChunk],
    hit_loader: Callable[[str, float], ScoredChunk | None],
) -> tuple[list[ScoredChunk], GatedAtomicRescueSelection | None]:
    """Insert a view-gated rescue at dense rank six; return the ranking unchanged otherwise."""

    gated = select_view_gated_atomic_rescue(artifact, query_vector, dense)
    if gated is None:
        return list(dense), None
    rescue = hit_loader(gated.selection.chunk_id, gated.selection.score)
    if rescue is None or rescue.chunk.id != gated.selection.chunk_id:
        raise AtomicRescueSelectionError("atomic rescue selected parent is unavailable")
    later = [hit for hit in dense[5:] if hit.chunk.id != gated.selection.chunk_id]
    return [*dense[:5], rescue, *later], gated


def atomic_rescue_expectation_parity(
    path: str | Path,
    *,
    query: str,
    selection: AtomicRescueSelection,
) -> tuple[bool, bool]:
    """Compare a selection with a private benchmark receipt without exposing either value."""

    resolved = Path(path).expanduser().resolve()
    with _EXPECTATION_CACHE_LOCK:
        expectations = _EXPECTATION_CACHE.get(resolved)
        if expectations is None:
            try:
                decoded = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise AtomicRescueArtifactError(
                    "atomic rescue expectation artifact is unreadable"
                ) from exc
            rows = decoded.get("rows") if isinstance(decoded, dict) else None
            if (
                not isinstance(decoded, dict)
                or decoded.get("schema_version") != 1
                or not isinstance(rows, dict)
            ):
                raise AtomicRescueArtifactError(
                    "atomic rescue expectation artifact schema is unsupported"
                )
            parsed: dict[str, tuple[str, int, float]] = {}
            for query_digest, row in rows.items():
                if (
                    not isinstance(query_digest, str)
                    or len(query_digest) != 64
                    or not isinstance(row, dict)
                ):
                    raise AtomicRescueArtifactError(
                        "atomic rescue expectation row is malformed"
                    )
                source = row.get("source")
                ordinal = row.get("parent_ordinal")
                score = row.get("score")
                if (
                    not isinstance(source, str)
                    or not source
                    or isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or ordinal < 0
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise AtomicRescueArtifactError(
                        "atomic rescue expectation row is malformed"
                    )
                parsed[query_digest] = source, ordinal, float(score)
            expectations = parsed
            _EXPECTATION_CACHE[resolved] = expectations
    query_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    expected = expectations.get(query_digest)
    if expected is None:
        raise AtomicRescueArtifactError("atomic rescue expectation lacks the query")
    return (
        (selection.source, selection.parent_ordinal) == expected[:2],
        selection.score == expected[2],
    )


def write_atomic_rescue_artifact(
    directory: str | Path,
    *,
    matrix: Any,
    views: Sequence[Mapping[str, object]],
    generation_id: str,
    calibration_id: str,
    pipeline_fingerprint: str,
    corpus_fingerprint: str,
    embedding_profile: str,
    embedding_fingerprint: str,
    ordinary_chunk_count: int,
    source_commit: str,
) -> Path:
    """Write a new immutable artifact directory and return its manifest path."""

    import numpy as np

    root = Path(directory).resolve()
    if root.exists():
        raise FileExistsError(f"atomic rescue artifact directory already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    values = np.ascontiguousarray(matrix, dtype=np.float32)
    try:
        if (
            values.ndim != 2
            or values.shape[0] != len(views)
            or values.shape[1] < 1
            or not np.all(np.isfinite(values))
            or not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0.0, atol=1e-4)
        ):
            raise AtomicRescueArtifactError("atomic rescue build matrix violates loader invariants")
        matrix_path = staging / "matrix.npy"
        metadata_path = staging / "views.json"
        manifest_path = staging / "manifest.json"
        np.save(matrix_path, values, allow_pickle=False)
        metadata = {"schema_version": 1, "views": list(views)}
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        parent_count = len({str(view["chunk_id"]) for view in views})
        manifest = {
            "schema_version": ATOMIC_RESCUE_SCHEMA_VERSION,
            "generation_id": generation_id,
            "calibration_id": calibration_id,
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "embedding_profile": embedding_profile,
            "embedding_fingerprint": embedding_fingerprint,
            "dimension": int(values.shape[1]),
            "ordinary_chunk_count": ordinary_chunk_count,
            "view_count": len(views),
            "parent_count": parent_count,
            "matrix_file": matrix_path.name,
            "metadata_file": metadata_path.name,
            "matrix_sha256": _sha256(matrix_path),
            "metadata_sha256": _sha256(metadata_path),
            "source_commit": source_commit,
            "constructed_at": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, root)
        return root / "manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "ATOMIC_RESCUE_SCHEMA_VERSION",
    "AtomicRescueArtifact",
    "AtomicRescueArtifactError",
    "AtomicRescueLineageError",
    "AtomicRescueSelection",
    "AtomicRescueSelectionError",
    "AtomicRescueView",
    "GatedAtomicRescueSelection",
    "clear_atomic_rescue_artifact_cache",
    "atomic_rescue_expectation_parity",
    "atomic_rescue_reference_parity",
    "insert_atomic_rescue_dense",
    "insert_gated_atomic_rescue_dense",
    "insert_view_gated_atomic_rescue_dense",
    "load_atomic_rescue_artifact",
    "resolve_atomic_rescue_manifest",
    "select_atomic_rescue",
    "select_gated_atomic_rescue",
    "select_view_gated_atomic_rescue",
    "write_atomic_rescue_artifact",
]
