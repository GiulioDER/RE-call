"""Atomic rescue on PostgreSQL: build a generation's artifact before it can serve.

`recall.trust` applies the rescue to an unscoped search once RECALL_ATOMIC_RESCUE_MODE resolves to
`active` for the tenant (`recall._env.tenant_retrieval_environment`): it loads
`<RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT>/<generation_id>/manifest.json`, refuses it unless its lineage
is the serving one, and refuses the search when it is missing. So in active mode a generation must
have its artifact before it serves. `ensure_for_promotion` is called before every certified
promotion in this package (the wizard, `recall generation promote` and `rollback`, a certified
desktop upload; a development-mode upload promotes without one) and raises when it cannot build,
which refuses the promotion; `recall generation rescue` builds one by hand, which a recalibrated
generation needs, since an artifact names the calibration it serves beside.

The artifact is assembled as a lite store's is (`recall.lite.atomic`): the micro views of each chunk
text are embedded once and kept in `<root>/micro-views.sqlite` (`recall.atomic_view_store`), keyed
by embedder profile and text, so a generation embeds only the views of texts no earlier one held.
The first build embeds every view, about `w / 12` for a chunk of `w` words, which is why it costs
more than the index build. Generation ids are unique across tenants, so one root may serve several.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    artifact_names_any,
    evict_atomic_rescue_artifacts,
    load_atomic_rescue_artifact,
    remove_artifact_directories,
    require_atomic_dependency,
    rescue_parents,
    resolve_atomic_rescue_manifest,
    write_atomic_rescue_artifact,
)
from recall.lite.atomic import ROOT_ENV, VIEW_STORE_NAME, artifact_root, configured_root, rescue_active
from recall.lite.writer_lock import try_lock, unlock
from recall.observability import get_logger
from recall.state_dir import write_ignore

if TYPE_CHECKING:
    from recall.embeddings import Embedder

_log = get_logger("generation_rescue")

__all__ = [
    "after_publish", "artifact_manifest", "artifact_serves", "build_artifact", "build_backoff", "build_lock",
    "ensure_for_promotion", "erase_forgotten", "forget_roots", "generation_artifact_dirs", "prune_superseded",
    "remove_artifacts", "set_build_backoff",
]


def _active_root(tenant: str, env: Mapping[str, str] | None) -> Path | None:
    """The artifact root when the rescue is active for `tenant`, else None; refuses a relative one,
    and an active rescue with no root, under which the promoted generation would refuse every
    unscoped search (CCA DOC-007)."""
    root = configured_root(tenant, env)
    if root is None and rescue_active(tenant, env):
        raise AtomicRescueArtifactError(
            f"the atomic rescue is active for tenant {tenant!r} but {ROOT_ENV} is not set, so its "
            "artifact cannot be built"
        )
    if root is not None and not root.is_absolute():
        # Each process resolves it against its own working directory, so the build and the server
        # would write and read different roots.
        raise AtomicRescueArtifactError(f"{ROOT_ENV} must be an absolute path, not {str(root)!r}")
    return root


def ensure_for_promotion(
    dsn: str,
    tenant: str,
    generation_id: str,
    make_embedder: Callable[[], Embedder],
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Build `generation_id`'s artifact when the rescue is active for `tenant`; None when it is not.

    Raises when it cannot, an active rescue with no root included, so the caller refuses the
    promotion: promoted without one, the
    generation would refuse every unscoped search. `env` is the environment the server resolves
    its mode from; the default is this process's. `make_embedder` is called only when a build runs, so
    a promotion with the rescue off loads no model.
    """
    root = _active_root(tenant, env)
    if root is None:
        return None
    return build_artifact(dsn, tenant, generation_id, make_embedder(), root)


def after_publish(
    dsn: str,
    tenant: str,
    generation_id: str,
    make_embedder: Callable[[], Embedder],
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Rebuild `generation_id`'s artifact for the calibration just published, and say what happened.

    An artifact names the calibration it serves beside, so a new one on a serving generation leaves
    every unscoped search refusing until it is rebuilt. The publication has happened by now and is
    not undone, so a failed build is reported, with the command that retries it, rather than raised.
    "" when the rescue is off for `tenant`.
    """
    try:
        built = ensure_for_promotion(dsn, tenant, generation_id, make_embedder, env=env)
    except Exception as exc:  # BROAD-CATCH: error-translation  # the publication stands; the operator must see this
        return (
            f"WARNING: the atomic rescue is active for tenant {tenant!r} and its artifact for "
            f"{generation_id} could not be rebuilt for the new calibration ({exc}), so its unscoped "
            f"searches refuse; retry with `recall --tenant {tenant} generation rescue {generation_id}`"
        )
    return f"atomic rescue: {built}" if built else ""


def artifact_manifest(root: str | Path, generation_id: str, calibration_id: str | None) -> Path:
    """The manifest serving `generation_id` beside `calibration_id` under `root`.

    An artifact is keyed by both, `<generation_id>.<calibration_id>`, so a new calibration is built
    beside the old one instead of replacing a directory another server may still have mapped, which
    Windows refuses (docs/DESKTOP_ATOMIC_RESCUE_DESIGN.md). One built before that, at the old
    `<generation_id>` key, is returned when it exists and the keyed one does not, so it keeps serving;
    the loader's lineage check decides whether it serves this calibration. Otherwise the keyed path,
    which is where a build writes.
    """
    if calibration_id is None:
        raise AtomicRescueArtifactError("an atomic rescue artifact on PostgreSQL requires a certified calibration")
    keyed = resolve_atomic_rescue_manifest(root, f"{generation_id}.{calibration_id}")
    if keyed.is_file():
        return keyed
    legacy = resolve_atomic_rescue_manifest(root, generation_id)
    return legacy if legacy.is_file() else keyed


def generation_artifact_dirs(root: str | Path, generation_id: str) -> list[Path]:
    """Every artifact directory of `generation_id` under `root` that exists: the old `<generation_id>`
    key and each `<generation_id>.<calibration_id>`, sorted."""
    legacy = resolve_atomic_rescue_manifest(root, generation_id).parent
    found = [legacy] if legacy.is_dir() else []
    found.extend(entry for entry in legacy.parent.glob(f"{generation_id}.*") if entry.is_dir())
    return sorted(found)


def build_artifact(dsn: str, tenant: str, generation_id: str, embedder: Embedder, root: Path) -> str:
    """Build the artifact `generation_id` serves beside its published calibration, and describe it.

    An artifact already in force is kept; one for another calibration is left in place, and this one
    is written beside it (`artifact_manifest`). The
    calibration must be certified and published, as the trust gate requires before it reaches the
    rescue, and the embedder must be the one the generation was built with.
    """
    from recall import __version__
    from recall.atomic_view_store import AtomicViewStore, assemble
    from recall.embedding_core import profile_lineage_mismatch
    from recall.embeddings import embed_document_groups, embedding_profile, embedding_profile_id
    from recall.generation_store import GenerationStore
    from recall.generations import GenerationManager

    require_atomic_dependency()
    calibration_id = _published_calibration(dsn, tenant, generation_id)
    manifest = resolve_atomic_rescue_manifest(root, f"{generation_id}.{calibration_id}")
    # Entered as a name, not bound by `as`: `__enter__` is typed on the base store.
    store = GenerationStore(dsn, embedder.dim, tenant=tenant)
    with store:
        store.set_fixed_generation(generation_id)
        binding = store.generation_binding()
        model, dimension = binding.get("embedder_model"), binding.get("embedder_dimension")
        if (
            (model is not None and str(embedder.name) != model)
            or (dimension is not None and str(embedder.dim) != dimension)
            or profile_lineage_mismatch(
                embedder, binding.get("embedder_profile_id"), binding.get("embedder_profile_fingerprint")
            )
        ):
            # The trust gate refuses such a server anyway; views from another model would rank nonsense.
            raise AtomicRescueArtifactError(
                f"generation {generation_id} was built with {model!r}, not with {embedder.name!r}; "
                "build its atomic rescue with the embedder that built it"
            )
        lineage = {
            "generation_id": generation_id,
            "calibration_id": calibration_id,
            "pipeline_fingerprint": binding.get("pipeline_fingerprint"),
            "corpus_fingerprint": binding.get("corpus_fingerprint"),
        }
        # Before anything is written under the root (the view store first), so nothing the rescue
        # keeps is ever there that a forget run without the root variable cannot learn of
        # (`forget_roots`); an artifact already in force is recorded too, since one built before
        # the record existed has none.
        GenerationManager(dsn, tenant, actor="recall-atomic-rescue").record_rescue_root(generation_id, str(root))
        # Before the scan, which an artifact already in force (a rollback, a repeated rescue) would
        # make for nothing (CCA PERF-002).
        if artifact_serves(artifact_manifest(root, generation_id, calibration_id), lineage, embedder):
            return f"in force for {generation_id}"
        # Before the scan: on Windows the removal fails while another process has the damaged
        # matrix mapped, and the build would then refuse only after a full scan and assembly.
        _remove_damaged(manifest)
        reused = _recalibrated(root, generation_id, lineage, embedder)
        parents = rescue_parents(store.iter_chunks()) if reused is None else []
    root.mkdir(parents=True, exist_ok=True)
    # The root holds view vectors of the corpus, like the `.recall` directory.
    write_ignore(root)
    profile = embedding_profile(embedder).fingerprint()
    if reused is not None:
        matrix, metadata, ordinary, counts = reused
    else:
        with AtomicViewStore(root / VIEW_STORE_NAME) as views:
            matrix, metadata, counts = assemble(
                parents, views, profile, int(embedder.dim), lambda groups: embed_document_groups(embedder, groups)
            )
        ordinary = len(parents)
        _refuse_if_corpus_moved(dsn, tenant, generation_id, embedder.dim, lineage, root, parents)
    if artifact_serves(artifact_manifest(root, generation_id, calibration_id), lineage, embedder):
        # A concurrent build of this generation published while this one assembled: removing its
        # artifact would leave a window with none, or on Windows a half-removed one.
        return f"in force for {generation_id}"
    if _published_calibration(dsn, tenant, generation_id) != calibration_id:
        # A calibration published while this one assembled: the build for it replaces the artifact,
        # which this one, finishing later, would otherwise delete and replace with its own for the
        # calibration no longer published (CCA STAKES-004).
        raise AtomicRescueArtifactError(
            f"the published calibration of generation {generation_id} changed while its atomic rescue "
            "was built, so it was not built; the build for the calibration now published replaces it, "
            f"or run `recall --tenant {tenant} generation rescue {generation_id}`"
        )
    _remove_damaged(manifest)
    try:
        write_atomic_rescue_artifact(
            manifest.parent,
            matrix=matrix,
            views=metadata,
            generation_id=generation_id,
            calibration_id=str(lineage["calibration_id"]),
            pipeline_fingerprint=str(lineage["pipeline_fingerprint"]),
            corpus_fingerprint=str(lineage["corpus_fingerprint"]),
            embedding_profile=embedding_profile_id(embedder),
            embedding_fingerprint=profile,
            ordinary_chunk_count=ordinary,
            source_commit=f"recall-rag {__version__}",
        )
    except OSError:
        # A concurrent build of this generation published first; the writer verified this build's
        # own before publishing, so only the winner's is checked here.
        if not manifest.is_file():
            raise
        load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
    # A forget that committed after the check above scanned the artifacts before this one existed.
    _refuse_if_corpus_moved(dsn, tenant, generation_id, embedder.dim, lineage, root, parents, written=manifest.parent)
    return (
        f"built for {generation_id}: {counts['views']} view(s) of {counts['parents']} chunk(s), "
        f"{counts['embedded_views']} newly embedded"
    )


def _remove_damaged(manifest: Path) -> None:
    """Remove the directory of `manifest`, which the caller found serves nothing: a damaged
    artifact of this very key. Raises when it stays, which Windows makes it do while a process has
    its matrix mapped."""
    if not manifest.parent.exists():
        return
    # This process's cache maps the matrix it last loaded, and Windows refuses to delete it.
    evict_atomic_rescue_artifacts([manifest.parent])
    shutil.rmtree(manifest.parent, ignore_errors=True)
    if manifest.parent.exists():
        raise AtomicRescueArtifactError(
            f"{manifest.parent} holds a damaged artifact and cannot be replaced "
            "while a process has its matrix open; stop that server and run `recall generation rescue`"
        )


def _published_calibration(dsn: str, tenant: str, generation_id: str) -> str:
    """The id of `generation_id`'s certified published calibration; refuses one that is not."""
    from recall.calibration_v2 import CalibrationRepository, CalibrationStatus

    resolution = CalibrationRepository(dsn, tenant, actor="recall-atomic-rescue").resolve(generation_id)
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        raise AtomicRescueArtifactError(
            f"generation {generation_id} has no certified published calibration "
            f"({resolution.status.value}), so its atomic rescue cannot be built"
        )
    return resolution.artifact.calibration_id


def artifact_serves(manifest: Path, lineage: Mapping[str, str | None], embedder: Embedder) -> bool:
    """Whether `manifest` is an artifact with exactly this lineage, verified by the serving loader."""
    if not manifest.is_file():
        return False
    try:
        load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
    except AtomicRescueArtifactError:
        return False
    return True


def _recalibrated(
    root: Path, generation_id: str, lineage: Mapping[str, str | None], embedder: Embedder
) -> tuple[object, list[dict[str, object]], int, dict[str, int]] | None:
    """The views of an artifact of `generation_id` whose only difference is its calibration, else
    None: ``(matrix, views, ordinary chunk count, counts)``.

    A new calibration of an unchanged corpus leaves every view as it was, so they are copied rather
    than assembled again from a full scan, as the Lite builder copies a sibling (CCA PERF-001).
    """
    import dataclasses

    import numpy as np

    for directory in generation_artifact_dirs(root, generation_id):
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            artifact = load_atomic_rescue_artifact(manifest)
            if artifact.calibration_id == lineage["calibration_id"]:
                continue
            artifact.assert_lineage(**{**lineage, "calibration_id": artifact.calibration_id}, embedder=embedder)
        except AtomicRescueArtifactError:
            continue
        break
    else:
        return None
    counts = {"parents": artifact.parent_count, "views": artifact.view_count, "embedded_views": 0}
    # Copied, because that artifact's directory is removed by a later prune, and Windows refuses to
    # delete a matrix that is still mapped.
    return (
        np.array(artifact.matrix),
        [dataclasses.asdict(view) for view in artifact.views],
        artifact.ordinary_chunk_count,
        counts,
    )


def _refuse_if_corpus_moved(
    dsn: str,
    tenant: str,
    generation_id: str,
    dim: int,
    lineage: Mapping[str, str | None],
    root: Path,
    parents: Sequence[tuple[str, str, int, str]],
    *,
    written: Path | None = None,
) -> None:
    """Refuse a build whose generation a forget changed while it ran, erasing what it stored.

    Such a forget erased the views and artifacts of its chunks before this build stored them, so it
    cannot find them; they are erased here, by the texts the generation no longer holds, as the
    Lite builder does (`recall.lite.atomic.ensure_rescue_artifact`, CCA BUG-004). `written` is the
    artifact this build published, which then holds them too.
    """
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
    from recall.generation_store import GenerationStore

    store = GenerationStore(dsn, dim, tenant=tenant)
    with store:
        store.set_fixed_generation(generation_id)
        if store.generation_binding().get("corpus_fingerprint") == lineage["corpus_fingerprint"]:
            return
        current = {chunk.text for chunk in store.iter_chunks()}
    stale = {chunk_view_digest(text) for *_, text in parents if text not in current}
    if stale:
        with AtomicViewStore(root / VIEW_STORE_NAME) as views:
            views.erase_digests(stale)
    if written is not None:
        evict_atomic_rescue_artifacts([written])
        shutil.rmtree(written, ignore_errors=True)
    raise AtomicRescueArtifactError(
        f"the corpus of generation {generation_id} changed while its atomic rescue was built (a "
        "forget landed), so it was not built; run it again"
    )


def _root(env: Mapping[str, str] | None) -> Path | None:
    """The configured root whatever the mode, since a rescue turned off keeps what it built."""
    root = artifact_root(env)
    return root if root is not None and root.is_dir() else None


def forget_roots(
    dsn: str, tenant: str, *, env: Mapping[str, str] | None = None
) -> tuple[list[Path], list[str], bool]:
    """The roots a forget of `tenant` erases under, the recorded ones this process cannot reach,
    and whether the record could not be read.

    The configured root, and every root a build of the tenant recorded (`build_artifact`): the
    wizard writes the root to the project `.env` and the server blocks only, so a CLI forget run
    elsewhere would otherwise erase nothing the rescue keeps (CCA SEC-003). A recorded root that is
    a directory here is used when it holds the view store or a generation folder, and skipped
    when it holds neither, since then a forget has nothing to erase there; one that is no
    directory here, or cannot be read, is reported. The configured root is returned whatever the
    record. Never raises for a root it cannot read: the forget runs this before deleting anything.
    """
    from recall.generations import GenerationManager

    usable: list[Path] = []
    unreachable: list[str] = []
    try:
        configured = _root(env)
    except OSError:  # Python before 3.14 raises PermissionError from `is_dir`
        unreachable.append(str(artifact_root(env)))
        configured = None
    if configured is not None:
        usable.append(configured)
    try:
        recorded = sorted(GenerationManager(dsn, tenant, actor="recall-atomic-rescue").rescue_roots())
    except Exception:  # BROAD-CATCH: fail-closed  # the configured root is still erased, and the caller warns
        _log.exception("the atomic rescue roots of tenant %r could not be read", tenant)
        return usable, unreachable, True
    for raw in recorded:
        if raw.replace("\\", "/").startswith("//"):
            # A UNC or device path: probing it would open a connection to whatever host the
            # database names (CCA SEC-001), so it is reported without being touched.
            unreachable.append(raw)
            continue
        path = Path(raw)
        if path in usable:
            continue
        held = _holds_rescue(path) if path.is_absolute() else None
        if held is None:
            unreachable.append(raw)
        elif held:
            usable.append(path)
    return usable, unreachable, False


def _holds_rescue(path: Path) -> bool | None:
    """Whether the directory `path` holds anything a forget erases (the view store, or a generation
    folder); None when it is no directory here or cannot be read (CCA review NEW-001: Python before
    3.14 raises PermissionError from `is_dir`)."""
    try:
        if not path.is_dir():
            return None
        return (path / VIEW_STORE_NAME).is_file() or any(entry.is_dir() for entry in path.iterdir())
    except OSError:
        return None


@contextmanager
def build_lock(root: Path, generation_id: str) -> Iterator[bool]:
    """Try, without waiting, to hold the lock on building `generation_id`'s artifact under `root`,
    and yield whether it is held; released on exit.

    The host servers of one tenant (`recall_mcp.rescue_builder`) take it so one process builds at
    a time: one that cannot take it leaves the build to the holder, whose artifact serves it too.
    An OS lock, so it dies with its process.
    """
    root.mkdir(parents=True, exist_ok=True)
    with open(_lock_path(root, generation_id), "ab") as fh:
        held = try_lock(fh)
        try:
            yield held
        finally:
            if held:
                unlock(fh)


def _lock_path(root: Path, generation_id: str) -> Path:
    return root / f".rescue-{generation_id}.lock"


def _backoff_path(root: Path, generation_id: str) -> Path:
    return root / f".rescue-{generation_id}.backoff.json"


def build_backoff(root: Path, generation_id: str, calibration_id: str) -> tuple[float, float] | None:
    """The backoff after a failed build of `generation_id`'s artifact for `calibration_id`, as
    ``(retry at, backoff seconds)`` on the wall clock, or None when none is recorded.

    Kept in a file beside the build lock, so every host server of the tenant waits it out: held per
    process, N servers each retried a build that always fails, N builds per window. A record for
    another calibration, or one that cannot be read, is no backoff.
    """
    try:
        record = json.loads(_backoff_path(root, generation_id).read_text(encoding="utf-8"))
        if record["calibration_id"] != calibration_id:
            return None
        return float(record["retry_at"]), float(record["backoff_s"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def set_build_backoff(
    root: Path, generation_id: str, calibration_id: str, backoff: tuple[float, float] | None
) -> None:
    """Record ``(retry at, backoff seconds)`` for `build_backoff`, or clear it with None. The caller
    holds `build_lock`, so two writers never race; the file is replaced whole, so a reader never
    sees half of one."""
    path = _backoff_path(root, generation_id)
    if backoff is None:
        path.unlink(missing_ok=True)
        return
    staging = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staging.write_text(
        json.dumps({"calibration_id": calibration_id, "retry_at": backoff[0], "backoff_s": backoff[1]}),
        encoding="utf-8",
    )
    os.replace(staging, path)


def _remove_build_state(root: Path, generation_id: str) -> None:
    """Remove the build lock file and backoff record of a generation `prune_superseded` removed,
    unless a process holds the lock: only one this process can lock itself is removed."""
    path = _lock_path(root, generation_id)
    if not path.is_file():
        return
    with open(path, "ab") as fh:
        if not try_lock(fh):
            return  # a build holds it
        _backoff_path(root, generation_id).unlink(missing_ok=True)
        if sys.platform != "win32":
            # Unlinked while held: a process that opened it already finds it locked until this one
            # closes it, and one that opens it later creates a new file.
            path.unlink()
            return
        unlock(fh)
    # Windows refuses to delete a file that is open, so it is released first; a process that opened
    # it since then makes the removal fail, and it stays for that process to lock.
    with suppress(OSError):
        path.unlink()


def prune_superseded(dsn: str, tenant: str, root: Path) -> int:
    """Remove the artifacts under `root` of this tenant's generations that cannot serve again,
    returning how many directories went.

    Run by the host server after it builds one (`recall_mcp.rescue_builder`): a desktop install runs
    no gc, and each upload promotes a generation with a full-size artifact. Kept: the active
    generation, the most recently retired one (a rollback's target), and every generation not yet
    retired or failed (one about to be promoted). Removed: older retired, failed, and those gc has
    already deleted. Best effort: a directory a process still has mapped (Windows) stays, and a later
    prune removes it. Only this tenant's generation ids are considered, though tenants share a root.
    """
    from recall.generations import GenerationManager
    from recall.lineage import GenerationState

    manager = GenerationManager(dsn, tenant, actor="recall-atomic-rescue")
    records = manager.list_generations()
    retired = sorted(
        (record for record in records if record.state is GenerationState.RETIRED),
        key=lambda record: record.retired_at or datetime.min.replace(tzinfo=UTC),
    )
    doomed = {record.generation_id for record in retired[:-1]}
    doomed |= {record.generation_id for record in records if record.state is GenerationState.FAILED}
    doomed |= manager.collected_generation_ids()
    removed = 0
    for generation_id in sorted(doomed):
        # A server killed while it wrote an artifact leaves its staging directory, `.<key>.<hex>`
        # (`write_atomic_rescue_artifact`), which no build of a removed generation will reuse.
        staging = [entry for entry in root.glob(f".{generation_id}.*") if entry.is_dir()]
        for directory in [*generation_artifact_dirs(root, generation_id), *staging]:
            try:
                removed += remove_artifact_directories([directory], root)
            except AtomicRescueArtifactError:
                _log.info("atomic rescue artifact %s is still open; a later prune removes it", directory)
        _remove_build_state(root, generation_id)
    return removed


def remove_artifacts(generation_ids: Collection[str], *, env: Mapping[str, str] | None = None) -> int:
    """Remove the artifacts of `generation_ids` (collected or forgotten), returning how many.

    Raises when one is left, which happens on Windows while a process has its matrix open.
    """
    root = _root(env)
    if root is None:
        return 0
    return remove_artifact_directories(
        [directory for gid in sorted(generation_ids) for directory in generation_artifact_dirs(root, gid)], root
    )


def erase_forgotten(
    generation_ids: Collection[str],
    texts: Sequence[str],
    sources: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Erase what the rescue keeps of forgotten chunks: ``(view rows erased, artifacts removed)``.

    Run by a forget after the chunks are deleted, with their texts read before. The views are
    erased by text digest, and every artifact of the tenant's `generation_ids` whose `views.json`
    names one of `sources`, or cannot be read, is removed: the forget changed that generation's
    corpus fingerprint, so the artifact no longer serves, and it keeps the forgotten chunks' view
    vectors.
    """
    from recall.atomic_view_store import AtomicViewStore, chunk_view_digest

    root = _root(env)
    if root is None:
        return 0, 0
    erased = 0
    views_path = root / VIEW_STORE_NAME
    if texts and views_path.is_file():
        with AtomicViewStore(views_path) as views:
            erased = views.erase_digests({chunk_view_digest(text) for text in texts})
    named = set(sources)
    doomed = [
        directory for gid in sorted(generation_ids)
        for directory in generation_artifact_dirs(root, gid)
        if artifact_names_any(directory, named)
    ]
    return erased, remove_artifact_directories(doomed, root)
