"""Atomic rescue on a lite store: build the artifact the trust gate serves, after each write.

`recall.trust` applies the rescue to any store once RECALL_ATOMIC_RESCUE_MODE is active: it loads
one artifact of micro-view vectors (`recall.atomic_rescue`), refuses it unless its lineage is the
serving lineage, and moves at most one parent, scored by its own cosine, to rank six. On
PostgreSQL the artifact is built by `scripts/build_atomic_micro_artifact.py` before a generation
is promoted. A lite store has no promotion, so `ensure_rescue_artifact` builds it after every
write that keeps a calibration in force: a refresh, an index, and a background refit landing.

Off by default. It builds only when RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is set and the store's
tenant resolves to the active mode exactly as the server's search resolves it
(`recall_mcp.settings.tenant_retrieval_environment`), so the builder and the search agree.

An artifact is keyed by the generation AND the calibration. A lite store's calibration can change
while its corpus does not (a background refit lands), and an artifact is immutable once written:
a directory named by the generation alone would have to be replaced while a search may have its
matrix mapped, which Windows refuses. `LiteStore.atomic_rescue_manifest` names the same key for
the trust gate. Nothing is built without a certified calibration: strict search refuses before it
reaches the rescue, and an artifact's lineage names the calibration it serves beside.

The views of each chunk text are embedded once and kept in `<root>/micro-views.sqlite`
(`recall.atomic_view_store`), keyed by embedder profile and text, so a build embeds only the
views of texts no earlier build saw. The first build is the expensive one: a chunk of `w` words
has about `w / 12` views, each embedded like a passage.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    load_atomic_rescue_artifact,
    resolve_atomic_rescue_manifest,
    write_atomic_rescue_artifact,
)

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.store import LiteStore

__all__ = ["BuildReport", "configured_root", "ensure_rescue_artifact", "manifest_path"]

ROOT_ENV = "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"
VIEW_STORE_NAME = "micro-views.sqlite"
#: The directories this module writes under the root, and so the only ones it removes.
_ARTIFACT_NAME = re.compile(r"lite-[0-9a-f]{16}\.cal_[0-9a-f]+")


@dataclass(frozen=True)
class BuildReport:
    """What one `ensure_rescue_artifact` did: the artifact in force, or why there is none."""

    manifest: Path | None
    reason: str
    counts: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def render(self) -> str:
        return f"atomic       {self.reason}"


def manifest_path(root: str | Path, generation_id: str, calibration_id: str | None) -> Path:
    """The manifest of the artifact serving `generation_id` beside `calibration_id`."""
    if calibration_id is None:
        raise AtomicRescueArtifactError("atomic rescue on a lite store requires a certified calibration")
    return resolve_atomic_rescue_manifest(root, f"{generation_id}.{calibration_id}")


def configured_root(tenant: str, env: Mapping[str, str] | None = None) -> Path | None:
    """The artifact root when the rescue is active for `tenant`, else None."""
    from recall_mcp.settings import tenant_retrieval_environment

    values = tenant_retrieval_environment(os.environ if env is None else env, tenant)
    root = values.get(ROOT_ENV, "").strip()
    if not root or values.get("RECALL_ATOMIC_RESCUE_MODE", "off").strip().lower() != "active":
        return None
    return Path(root).expanduser()


def ensure_rescue_artifact(
    store: LiteStore, embedder: Embedder, *, env: Mapping[str, str] | None = None
) -> BuildReport | None:
    """Build the artifact for the corpus and calibration in force now; None when the rescue is off.

    Raises when a build fails, so the write that called it reports the failure instead of leaving
    a store whose every active search would refuse on a missing artifact without saying why.
    """
    from recall import __version__
    from recall.atomic_view_store import AtomicViewStore, assemble
    from recall.calibration_v2 import CalibrationStatus
    from recall.embeddings import embed_document_groups, embedding_profile, embedding_profile_id

    root = configured_root(store.tenant, env)
    if root is None:
        return None
    started = time.perf_counter()
    resolution = store.resolve_calibration()
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        return BuildReport(None, f"not built: the calibration is {resolution.status.value}")
    calibration_id = resolution.artifact.calibration_id
    binding = store.generation_binding()
    generation_id, corpus_fingerprint = binding["generation_id"], binding["corpus_fingerprint"]
    manifest = manifest_path(root, generation_id, calibration_id)
    lineage = {
        "generation_id": generation_id,
        "calibration_id": calibration_id,
        "pipeline_fingerprint": binding.get("pipeline_fingerprint"),
        "corpus_fingerprint": corpus_fingerprint,
    }
    if manifest.parent.exists():
        # The same corpus and calibration again (a write undone): the artifact already describes
        # them, unless an interrupted removal left part of it, which the loader refuses.
        try:
            load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
            return BuildReport(manifest, f"in force for {generation_id}", elapsed_s=time.perf_counter() - started)
        except AtomicRescueArtifactError:
            shutil.rmtree(manifest.parent)
    parents: list[tuple[str, str, int, str]] = []
    for chunk in store.iter_chunks():
        ordinal = chunk.metadata.get("ord")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise AtomicRescueArtifactError(f"chunk {chunk.id} has no integer ord, so it cannot be a rescue parent")
        parents.append((chunk.id, chunk.source, ordinal, chunk.text))
    parents.sort(key=lambda item: (item[1], item[2]))
    profile = embedding_profile(embedder).fingerprint()
    with AtomicViewStore(root / VIEW_STORE_NAME) as views:
        matrix, metadata, counts = assemble(
            parents, views, profile, int(embedder.dim), lambda groups: embed_document_groups(embedder, groups)
        )
    if store.corpus_fingerprint() != corpus_fingerprint:
        # A write landed while the views were read; that write builds its own artifact.
        return BuildReport(None, "not built: the memory changed during the build", counts, time.perf_counter() - started)
    written = write_atomic_rescue_artifact(
        manifest.parent,
        matrix=matrix,
        views=metadata,
        generation_id=generation_id,
        calibration_id=calibration_id,
        pipeline_fingerprint=str(lineage["pipeline_fingerprint"]),
        corpus_fingerprint=corpus_fingerprint,
        embedding_profile=embedding_profile_id(embedder),
        embedding_fingerprint=profile,
        ordinary_chunk_count=len(parents),
        source_commit=f"recall-rag {__version__}",
    )
    load_atomic_rescue_artifact(written).assert_lineage(**lineage, embedder=embedder)
    _remove_others(root, keep=manifest.parent.name)
    return BuildReport(
        written,
        f"built for {generation_id}: {counts['views']} view(s) of {counts['parents']} chunk(s), "
        f"{counts['embedded_views']} newly embedded",
        counts,
        time.perf_counter() - started,
    )


def _remove_others(root: Path, *, keep: str) -> None:
    """Remove the artifacts this module wrote for other corpora or calibrations.

    Best effort, for a stated reason: Windows refuses to delete a matrix a serving process still
    has mapped, and that artifact is then removed by the next build. A part left behind is never
    served, since nothing resolves its key until a build of that key replaces it.
    """
    for entry in root.iterdir():
        if entry.name != keep and entry.is_dir() and _ARTIFACT_NAME.fullmatch(entry.name):
            shutil.rmtree(entry, ignore_errors=True)
