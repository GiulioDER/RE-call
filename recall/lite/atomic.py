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

Each store keeps its artifacts in a folder of its own under the root, `store-<digest>`, named by
its tenant and resolved file path, and a build removes old artifacts from that folder only. Two
stores sharing a root (two projects, or two MCP servers) therefore never remove each other's.
Artifacts built before this layout sit at the top of the root: a search still serves one whose key
is in force, after the same lineage check, and no build removes them, since their owner is unknown.
A refit that lands on an unchanged corpus copies the matrix and views of the artifact it
supersedes rather than assembling them again.

The views of each chunk text are embedded once and kept in `<root>/micro-views.sqlite`
(`recall.atomic_view_store`), keyed by embedder profile and text, so a build embeds only the
views of texts no earlier build saw. The first build is the expensive one: a chunk of `w` words
has about `w / 12` views, each embedded like a passage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from recall.atomic_rescue import (
    AtomicRescueArtifact,
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

__all__ = [
    "BuildReport", "artifact_root", "configured_root", "ensure_rescue_artifact", "erase_forgotten",
    "manifest_path", "rescue_active", "rescue_settings", "serving_manifest", "store_default_root",
]

ROOT_ENV = "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"
VIEW_STORE_NAME = "micro-views.sqlite"
#: The directory names of artifacts, in a store's folder and, from the flat layout, at the top of
#: the root: there a build only counts them, and only a forget naming their sources removes one.
_ARTIFACT_NAME = re.compile(r"lite-[0-9a-f]{16}\.cal_[0-9a-f]+")
#: The files `write_atomic_rescue_artifact` publishes in an artifact's directory.
_MEMBERS = ("manifest.json", "views.json", "matrix.npy")


@dataclass(frozen=True)
class BuildReport:
    """What one `ensure_rescue_artifact` did: the artifact in force, or why there is none."""

    manifest: Path | None
    reason: str
    counts: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        return f"atomic       {self.reason}"


def manifest_path(root: str | Path, store: LiteStore, generation_id: str, calibration_id: str | None) -> Path:
    """The manifest of `store`'s artifact serving `generation_id` beside `calibration_id`."""
    if calibration_id is None:
        raise AtomicRescueArtifactError("atomic rescue on a lite store requires a certified calibration")
    if not Path(root).expanduser().is_absolute():
        # Each process would resolve it against its own working directory, so a CLI build and an
        # MCP server started elsewhere would write and read different roots.
        raise AtomicRescueArtifactError(f"{ROOT_ENV} must be an absolute path, not {str(root)!r}")
    return resolve_atomic_rescue_manifest(_store_folder(root, store), f"{generation_id}.{calibration_id}")


def _store_folder(root: str | Path, store: LiteStore) -> Path:
    """`store`'s own folder under the root, named by its tenant and resolved file path."""
    identity = f"{store.tenant}\0{os.path.normcase(store.path.resolve())}"
    return Path(root).expanduser() / f"store-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def serving_manifest(root: str | Path, store: LiteStore, generation_id: str, calibration_id: str | None) -> Path:
    """The manifest a search loads: `store`'s own, else one of the same key left by the flat layout.

    So a root built before per-store folders keeps serving until the next build writes its own.
    The trust gate checks the lineage of either before serving it.
    """
    manifest = manifest_path(root, store, generation_id, calibration_id)
    if manifest.is_file():
        return manifest
    flat = resolve_atomic_rescue_manifest(root, manifest.parent.name)
    return flat if flat.is_file() else manifest


def configured_root(tenant: str, env: Mapping[str, str] | None = None) -> Path | None:
    """The artifact root when the rescue is active for `tenant`, else None."""
    from recall._env import tenant_retrieval_environment

    values = tenant_retrieval_environment(os.environ if env is None else env, tenant)
    root = values.get(ROOT_ENV, "").strip()
    if not root or not rescue_active(tenant, env):
        return None
    return Path(root).expanduser()


def rescue_active(tenant: str, env: Mapping[str, str] | None = None) -> bool:
    """Whether RECALL_ATOMIC_RESCUE_MODE resolves to `active` for `tenant`, with or without a root."""
    from recall._env import tenant_retrieval_environment

    values = tenant_retrieval_environment(os.environ if env is None else env, tenant)
    return values.get("RECALL_ATOMIC_RESCUE_MODE", "off").strip().lower() == "active"


def artifact_root(env: Mapping[str, str] | None = None) -> Path | None:
    """The configured root whatever the mode, when it is absolute; None otherwise.

    The one reading of the root for the paths that act on what a rescue built, which a rescue
    turned off keeps (a forget, gc, `recall generation rescue`). The builders refuse a relative
    root, so nothing was written under one.
    """
    value = (os.environ if env is None else env).get(ROOT_ENV, "").strip()
    root = Path(value).expanduser() if value else None
    return root if root is not None and root.is_absolute() else None


def store_default_root(store_path: Path) -> Path:
    """Where `recall setup --lite` puts a store's rescue artifacts: beside the store file."""
    return store_path.with_name(store_path.name + ".atomic")


def rescue_settings(
    default_root: Path | None, *, off_reason: str = "", env: Mapping[str, str] | None = None
) -> tuple[dict[str, str], str]:
    """The atomic rescue settings an installer registers, and a note when the rescue stays off.

    Shared by `recall setup --lite` and the PostgreSQL wizard. On by default (owner decision
    2026-11-15): active, with its artifacts under `default_root`. A mode or root the shell sets is
    used as it is, and a root is kept with `off` or `shadow` too, so a forget still erases what it
    holds. A mistyped mode or a relative root raises ValueError: the server refuses to start on the
    first, and under the second the build and the server would resolve different roots. With no
    mode set, the rescue stays off, with a note, without the `atomic` dependencies, when the
    caller gives an `off_reason`, or when there is no `default_root`.
    """
    import importlib.util

    from recall._env import _ATOMIC_RESCUE_MODES

    source = os.environ if env is None else env
    mode = source.get("RECALL_ATOMIC_RESCUE_MODE", "").strip()
    root = source.get(ROOT_ENV, "").strip()
    if mode and mode.lower() not in _ATOMIC_RESCUE_MODES:
        raise ValueError(f"RECALL_ATOMIC_RESCUE_MODE must be off, shadow, or active, not {mode!r}")
    if root and not Path(root).expanduser().is_absolute():
        raise ValueError(f"{ROOT_ENV} must be an absolute path, not {root!r}")
    if not mode:
        off = ""
        if any(importlib.util.find_spec(name) is None for name in ("numpy", "threadpoolctl")):
            off = 'atomic rescue is off: pip install "recall-rag[atomic]" and run it again to turn it on'
        elif off_reason:
            off = f"atomic rescue is off: {off_reason}"
        if off:
            # A root an earlier run built under stays registered, so a forget still erases what it
            # holds (CCA ENV-002).
            kept = root or (str(default_root) if default_root is not None and default_root.is_dir() else "")
            return ({"RECALL_ATOMIC_RESCUE_MODE": "off", ROOT_ENV: kept} if kept else {}), off
        mode = "active"
    if mode.lower() == "active" and not root:
        if default_root is None:
            return {}, f"atomic rescue is off: set {ROOT_ENV} to turn it on"
        root = str(default_root)
    return {"RECALL_ATOMIC_RESCUE_MODE": mode, **({ROOT_ENV: root} if root else {})}, ""


def ensure_rescue_artifact(
    store: LiteStore, embedder: Embedder, *, env: Mapping[str, str] | None = None
) -> BuildReport | None:
    """Build the artifact for the corpus and calibration in force now; None when the rescue is off.

    Raises when a build fails, so a write that called it reports the failure instead of leaving a
    store whose every active search would refuse on a missing artifact without saying why. A
    background refit has no caller to report to: it logs the failure, and the next write that
    changes the memory, or `recall lite refresh --force`, builds again.
    """
    from recall import __version__
    from recall.atomic_view_store import AtomicViewStore, assemble, chunk_view_digest
    from recall.calibration_v2 import CalibrationStatus
    from recall.embeddings import embed_document_groups, embedding_profile, embedding_profile_id

    root = configured_root(store.tenant, env)
    if root is None:
        return None
    try:
        import threadpoolctl  # noqa: F401
    except ImportError as exc:
        # The loader needs it too: refused here, before any view is embedded, and before an
        # artifact another environment built could be taken for a damaged one and removed.
        raise AtomicRescueArtifactError("atomic rescue requires the recall-rag[atomic] optional dependency") from exc
    resolution = store.resolve_calibration()
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        return BuildReport(None, f"not built: the calibration is {resolution.status.value}")
    calibration_id = resolution.artifact.calibration_id
    binding = store.generation_binding()
    generation_id, corpus_fingerprint = binding["generation_id"], binding["corpus_fingerprint"]
    manifest = manifest_path(root, store, generation_id, calibration_id)
    lineage = {
        "generation_id": generation_id,
        "calibration_id": calibration_id,
        "pipeline_fingerprint": binding.get("pipeline_fingerprint"),
        "corpus_fingerprint": corpus_fingerprint,
    }
    if manifest.parent.exists():
        # The same corpus and calibration again (a write undone): the artifact already describes
        # them, unless an interrupted removal left part of it. Checked on disk first, because the
        # loader answers a key it has cached without reading the directory again; then loaded in
        # full, since only the verified load can tell a damaged matrix, which must be rebuilt.
        if all((manifest.parent / name).is_file() for name in _MEMBERS):
            try:
                load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
                return BuildReport(manifest, f"in force for {generation_id}")
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
    profile = embedding_profile(embedder).fingerprint()
    sibling = _sibling(manifest.parent, generation_id, lineage, embedder)
    if sibling is not None:
        # The same corpus under the calibration a refit replaced: the matrix and views depend on
        # the corpus and the embedder only, so they are copied rather than assembled again.
        matrix, metadata = sibling.matrix, [dataclasses.asdict(view) for view in sibling.views]
        ordinary = sibling.ordinary_chunk_count
        counts = {"parents": sibling.parent_count, "views": sibling.view_count, "embedded_views": 0}
    else:
        parents: list[tuple[str, str, int, str]] = []
        for chunk in store.iter_chunks():
            ordinal = chunk.metadata.get("ord")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise AtomicRescueArtifactError(f"chunk {chunk.id} has no integer ord, so it cannot be a rescue parent")
            parents.append((chunk.id, chunk.source, ordinal, chunk.text))
        parents.sort(key=lambda item: (item[1], item[2]))
        ordinary = len(parents)
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
        if sibling is None and store.corpus_fingerprint() != corpus_fingerprint:
            # A forget that landed after the chunks were read erased their views before this
            # build stored them, so they are erased here, by the texts the store no longer holds.
            # A build that copied a sibling stored no views.
            current = {chunk.text for chunk in store.iter_chunks()}
            stale = {chunk_view_digest(text) for *_, text in parents if text not in current}
            if stale:
                with AtomicViewStore(root / VIEW_STORE_NAME) as views:
                    views.erase_digests(stale)
        return BuildReport(None, "not built: the memory or its calibration changed during the build", counts)
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
            ordinary_chunk_count=ordinary,
            source_commit=f"recall-rag {__version__}",
        )
    except OSError:
        # A concurrent build of the same key (a refresh beside a background refit) published
        # first: refused before staging (FileExistsError) or at the rename onto its directory
        # (ENOTEMPTY, or access denied on Windows). With no artifact there, the failure was this
        # build's own. The winner's is verified here; this build's own was, by the writer, which
        # loads the staged directory with the serving loader before it publishes it.
        if not manifest.is_file():
            raise
        load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
        written = manifest
    _remove_others(manifest.parent.parent, keep={manifest.parent.name, _key_in_force(store)})
    # Their owner is unknown, so they are reported rather than removed (docs/LITE.md).
    flat = sum(1 for entry in root.iterdir() if entry.is_dir() and _ARTIFACT_NAME.fullmatch(entry.name))
    left = f"; {flat} artifact folder(s) from before per-store folders remain in {root}" if flat else ""
    return BuildReport(
        written,
        f"built for {generation_id}: {counts['views']} view(s) of {counts['parents']} chunk(s), "
        f"{counts['embedded_views']} newly embedded{left}",
        counts,
    )


def _sibling(
    directory: Path, generation_id: str, lineage: Mapping[str, str | None], embedder: Embedder
) -> AtomicRescueArtifact | None:
    """A complete artifact of this corpus under another calibration, verified; None when there is none."""
    for entry in sorted(directory.parent.glob(f"{generation_id}.*")):
        if entry == directory or not all((entry / name).is_file() for name in _MEMBERS):
            continue
        try:
            artifact = load_atomic_rescue_artifact(entry / "manifest.json")
            artifact.assert_lineage(**{**lineage, "calibration_id": artifact.calibration_id}, embedder=embedder)
        except AtomicRescueArtifactError:
            continue
        return artifact
    return None


def erase_forgotten(
    store: LiteStore,
    texts: Sequence[str],
    sources: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Erase what the rescue keeps of forgotten chunks: ``(view rows erased, artifacts removed)``.

    Run by a forget after the chunks are deleted, with their texts read before. The views are
    erased by text digest from `<root>/micro-views.sqlite`, and every artifact in the store's own
    folder is removed except the one the store resolves now: each other one was built from a
    corpus that held the forgotten chunks, and keeps their view vectors and sources. An artifact
    left at the top of the root by the layout before per-store folders has no known owner, so it is
    removed only when its `views.json` names one of `sources`. The root is read whatever the rescue
    mode, since a rescue turned off keeps what it built. Raises when something is left.
    """
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest

    # Without the variable, the root `recall setup --lite` chose: it registers the root only on the
    # MCP server, so a CLI forget would otherwise erase nothing and say nothing (CCA ENV-001).
    root = artifact_root(env) or store_default_root(store.path)
    if not root.is_dir():
        return 0, 0
    erased = 0
    views_path = root / VIEW_STORE_NAME
    if texts and views_path.is_file():
        with AtomicViewStore(views_path) as views:
            erased = views.erase_digests({chunk_view_digest(text) for text in texts})
    keep = _key_in_force(store)
    own = _store_folder(root, store)
    doomed = [
        entry for entry in (own.iterdir() if own.is_dir() else ())
        if entry.name != keep and entry.is_dir() and _ARTIFACT_NAME.fullmatch(entry.name)
    ]
    doomed += [
        entry for entry in root.iterdir()
        if entry.name != keep and entry.is_dir() and _ARTIFACT_NAME.fullmatch(entry.name)
        and _names_any(entry, set(sources))
    ]
    if doomed:
        # This process's cache maps the matrix it last loaded, and Windows refuses to delete it.
        clear_atomic_rescue_artifact_cache()
    for entry in doomed:
        shutil.rmtree(entry, ignore_errors=True)
    left = [entry.name for entry in doomed if entry.exists()]
    if left:
        raise AtomicRescueArtifactError(
            f"atomic rescue artifact(s) {', '.join(left)} under {root} could not be removed while a "
            "process has their matrix open"
        )
    return erased, len(doomed)


def _names_any(artifact: Path, sources: set[str]) -> bool:
    """Whether the artifact's `views.json` names one of `sources`; False when it cannot be read."""
    try:
        views = json.loads((artifact / "views.json").read_text(encoding="utf-8"))["views"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return any(isinstance(view, dict) and view.get("source") in sources for view in views)


def _key_in_force(store: LiteStore) -> str | None:
    """The artifact key the store resolves now (`manifest_path`); None without a certified calibration."""
    from recall.calibration_v2 import CalibrationStatus

    resolution = store.resolve_calibration()
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        return None
    return f"{store.generation_binding()['generation_id']}.{resolution.artifact.calibration_id}"


def _remove_others(folder: Path, *, keep: Container[str | None]) -> None:
    """Remove the artifacts this module wrote in one store's `folder` for other corpora or calibrations.

    `keep` names this build's key and the key in force when it finished: a background refit
    builds under its own lane, so the other build may have published the one a search now needs.
    The folder is this module's alone, so every directory in it is an artifact, except a dot-named
    one, which is a build still staging and is left to it.
    Best effort, for a stated reason: Windows refuses to delete a matrix a serving process still
    has mapped, and that artifact is then removed by the next build. A part left behind is never
    served: a build of its key checks every member on disk before reusing it.
    """
    for entry in folder.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)
