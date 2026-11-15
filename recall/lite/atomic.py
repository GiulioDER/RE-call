"""Atomic rescue on a lite store: build the artifact the trust gate serves, after each write.

`recall.trust` applies the rescue to any store once RECALL_ATOMIC_RESCUE_MODE is active: it loads
one artifact of micro-view vectors (`recall.atomic_rescue`), refuses it unless its lineage is the
serving lineage, and moves at most one parent, scored by its own cosine, to rank six. On
PostgreSQL the artifact is built by `scripts/build_atomic_micro_artifact.py` before a generation
is promoted. A lite store has no promotion, so `ensure_rescue_artifact` builds it after every
write that keeps a calibration in force: a refresh, an index, and a background refit landing.

Off by default. It builds only when RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT is set and the store's
tenant resolves to the active mode exactly as the server's and the CLI's search resolve it
(`recall._env.tenant_retrieval_environment`), so the builder and the search agree.

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
from collections.abc import Container, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    resolve_atomic_rescue_manifest,
    write_atomic_rescue_artifact,
)
from recall.state_dir import write_ignore

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.store import LiteStore

__all__ = ["BuildReport", "configured_root", "ensure_rescue_artifact", "manifest_path"]

ROOT_ENV = "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"
VIEW_STORE_NAME = "micro-views.sqlite"
#: The directories this module writes under the root, and so the only ones it removes.
_ARTIFACT_NAME = re.compile(r"lite-[0-9a-f]{16}\.cal_[0-9a-f]+")
#: The files `write_atomic_rescue_artifact` publishes in an artifact's directory.
_MEMBERS = ("manifest.json", "views.json", "matrix.npy")


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
    if not Path(root).expanduser().is_absolute():
        # Each process would resolve it against its own working directory, so a CLI build and an
        # MCP server started elsewhere would write and read different roots.
        raise AtomicRescueArtifactError(f"{ROOT_ENV} must be an absolute path, not {str(root)!r}")
    return resolve_atomic_rescue_manifest(root, f"{generation_id}.{calibration_id}")


def configured_root(tenant: str, env: Mapping[str, str] | None = None) -> Path | None:
    """The artifact root when the rescue is active for `tenant`, else None."""
    from recall._env import tenant_retrieval_environment

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
        # them, unless an interrupted removal left part of it. Checked on disk first, because the
        # loader answers a key it has cached without reading the directory again.
        if all((manifest.parent / name).is_file() for name in _MEMBERS):
            try:
                load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
                return BuildReport(manifest, f"in force for {generation_id}", elapsed_s=time.perf_counter() - started)
            except AtomicRescueArtifactError:
                pass
        # This process's own cache maps the matrix it last built or loaded, and Windows refuses to
        # delete a mapped file; searches in flight keep their own references.
        clear_atomic_rescue_artifact_cache()
        shutil.rmtree(manifest.parent, ignore_errors=True)
        if manifest.parent.exists():
            raise AtomicRescueArtifactError(
                f"{manifest.parent} is left from an earlier build and cannot be removed while a process "
                "(another one, or a search in flight here) has its matrix open; the next write builds "
                "it once that mapping is released"
            )
    parents: list[tuple[str, str, int, str]] = []
    for chunk in store.iter_chunks():
        ordinal = chunk.metadata.get("ord")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise AtomicRescueArtifactError(f"chunk {chunk.id} has no integer ord, so it cannot be a rescue parent")
        parents.append((chunk.id, chunk.source, ordinal, chunk.text))
    parents.sort(key=lambda item: (item[1], item[2]))
    profile = embedding_profile(embedder).fingerprint()
    # The root holds view vectors and the sources of memo chunks, like the `.recall` directory.
    root.mkdir(parents=True, exist_ok=True)
    write_ignore(root)
    with AtomicViewStore(root / VIEW_STORE_NAME) as views:
        matrix, metadata, counts = assemble(
            parents, views, profile, int(embedder.dim), lambda groups: embed_document_groups(embedder, groups)
        )
    if store.corpus_fingerprint() != corpus_fingerprint or _key_in_force(store) != manifest.parent.name:
        # A write, or a background refit under its own lane, landed while the views were read;
        # it builds its own artifact, which this build must neither replace nor remove.
        return BuildReport(
            None, "not built: the memory or its calibration changed during the build", counts, time.perf_counter() - started
        )
    try:
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
    except OSError:
        # A concurrent build of the same key (a refresh beside a background refit) published
        # first: refused before staging (FileExistsError) or at the rename onto its directory
        # (ENOTEMPTY, or access denied on Windows). Its artifact is verified below like this one
        # would have been; with no artifact there, the failure was this build's own.
        if not manifest.is_file():
            raise
        written = manifest
    load_atomic_rescue_artifact(written).assert_lineage(**lineage, embedder=embedder)
    _remove_others(root, keep={manifest.parent.name, _key_in_force(store)})
    return BuildReport(
        written,
        f"built for {generation_id}: {counts['views']} view(s) of {counts['parents']} chunk(s), "
        f"{counts['embedded_views']} newly embedded",
        counts,
        time.perf_counter() - started,
    )


def _key_in_force(store: LiteStore) -> str | None:
    """The artifact key the store resolves now (`manifest_path`); None without a certified calibration."""
    from recall.calibration_v2 import CalibrationStatus

    resolution = store.resolve_calibration()
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        return None
    return f"{store.generation_binding()['generation_id']}.{resolution.artifact.calibration_id}"


def _remove_others(root: Path, *, keep: Container[str | None]) -> None:
    """Remove the artifacts this module wrote for other corpora or calibrations.

    `keep` names this build's key and the key in force when it finished: a background refit
    builds under its own lane, so the other build may have published the one a search now needs.
    Best effort, for a stated reason: Windows refuses to delete a matrix a serving process still
    has mapped, and that artifact is then removed by the next build. A part left behind is never
    served: a build of its key checks every member on disk before reusing it.
    """
    for entry in root.iterdir():
        if entry.name not in keep and entry.is_dir() and _ARTIFACT_NAME.fullmatch(entry.name):
            shutil.rmtree(entry, ignore_errors=True)
