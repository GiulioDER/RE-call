"""Atomic rescue on PostgreSQL: build a generation's artifact before it can serve.

`recall.trust` applies the rescue to an unscoped search once RECALL_ATOMIC_RESCUE_MODE resolves to
`active` for the tenant (`recall._env.tenant_retrieval_environment`): it loads
`<RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT>/<generation_id>/manifest.json`, refuses it unless its lineage
is the serving one, and refuses the search when it is missing. So in active mode a generation must
have its artifact before it serves. `ensure_for_promotion` is called before every promotion in this
package (the wizard, `recall generation promote` and `rollback`, a desktop upload) and raises when
it cannot build, which refuses the promotion; `recall generation rescue` builds one by hand, which a
recalibrated generation needs, since an artifact names the calibration it serves beside.

The artifact is assembled as a lite store's is (`recall.lite.atomic`): the micro views of each chunk
text are embedded once and kept in `<root>/micro-views.sqlite` (`recall.atomic_view_store`), keyed
by embedder profile and text, so a generation embeds only the views of texts no earlier one held.
The first build embeds every view, about `w / 12` for a chunk of `w` words, which is why it costs
more than the index build. Generation ids are unique across tenants, so one root may serve several.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    resolve_atomic_rescue_manifest,
    write_atomic_rescue_artifact,
)
from recall.lite.atomic import ROOT_ENV, VIEW_STORE_NAME, configured_root
from recall.state_dir import write_ignore

if TYPE_CHECKING:
    from recall.embeddings import Embedder

__all__ = ["after_publish", "build_artifact", "ensure_for_promotion", "erase_forgotten", "remove_artifacts"]


def _active_root(tenant: str, env: Mapping[str, str] | None) -> Path | None:
    """The artifact root when the rescue is active for `tenant`, else None; refuses a relative one."""
    root = configured_root(tenant, env)
    if root is not None and not root.is_absolute():
        # Each process resolves it against its own working directory, so the build and the server
        # would write and read different roots.
        raise AtomicRescueArtifactError(f"{ROOT_ENV} must be an absolute path, not {str(root)!r}")
    return root


def ensure_for_promotion(
    dsn: str,
    tenant: str,
    generation_id: str,
    embedder: Callable[[], Embedder],
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Build `generation_id`'s artifact when the rescue is active for `tenant`; None when it is not.

    Raises when it cannot, so the caller refuses the promotion: promoted without one, the
    generation would refuse every unscoped search. `env` is the environment the server resolves
    its mode from; the default is this process's. `embedder` is called only when a build runs, so
    a promotion with the rescue off loads no model.
    """
    root = _active_root(tenant, env)
    if root is None:
        return None
    return build_artifact(dsn, tenant, generation_id, embedder(), root)


def after_publish(
    dsn: str,
    tenant: str,
    generation_id: str,
    embedder: Callable[[], Embedder],
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
        built = ensure_for_promotion(dsn, tenant, generation_id, embedder, env=env)
    except Exception as exc:  # BROAD-CATCH: error-translation  # the publication stands; the operator must see this
        return (
            f"WARNING: the atomic rescue is active for tenant {tenant!r} and its artifact for "
            f"{generation_id} could not be rebuilt for the new calibration ({exc}), so its unscoped "
            f"searches refuse; retry with `recall --tenant {tenant} generation rescue {generation_id}`"
        )
    return f"atomic rescue: {built}" if built else ""


def build_artifact(dsn: str, tenant: str, generation_id: str, embedder: Embedder, root: Path) -> str:
    """Build the artifact `generation_id` serves beside its published calibration, and describe it.

    An artifact already in force is kept; one built for another calibration is replaced. The
    calibration must be certified and published, as the trust gate requires before it reaches the
    rescue, and the embedder must be the one the generation was built with.
    """
    from recall import __version__
    from recall.atomic_view_store import AtomicViewStore, assemble
    from recall.calibration_v2 import CalibrationRepository, CalibrationStatus
    from recall.embedding_core import profile_lineage_mismatch
    from recall.embeddings import embed_document_groups, embedding_profile, embedding_profile_id
    from recall.generation_store import GenerationStore
    from recall.types import metadata_ordinal

    try:
        import threadpoolctl  # noqa: F401
    except ImportError as exc:
        # The loader needs it too, so without it every active search would refuse.
        raise AtomicRescueArtifactError(
            'atomic rescue requires the atomic optional dependency: pip install "recall-rag[atomic]"'
        ) from exc
    resolution = CalibrationRepository(dsn, tenant, actor="recall-atomic-rescue").resolve(generation_id)
    if resolution.status is not CalibrationStatus.CERTIFIED or resolution.artifact is None:
        raise AtomicRescueArtifactError(
            f"generation {generation_id} has no certified published calibration "
            f"({resolution.status.value}), so its atomic rescue cannot be built"
        )
    with GenerationStore(dsn, embedder.dim, tenant=tenant) as store:
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
        parents: list[tuple[str, str, int, str]] = []
        for chunk in store.iter_chunks():
            ordinal = metadata_ordinal(chunk.metadata)
            if ordinal is None:
                raise AtomicRescueArtifactError(f"chunk {chunk.id} has no integer ord, so it cannot be a rescue parent")
            parents.append((chunk.id, chunk.source, ordinal, chunk.text))
    parents.sort(key=lambda item: (item[1], item[2]))
    lineage = {
        "generation_id": generation_id,
        "calibration_id": resolution.artifact.calibration_id,
        "pipeline_fingerprint": binding.get("pipeline_fingerprint"),
        "corpus_fingerprint": binding.get("corpus_fingerprint"),
    }
    manifest = resolve_atomic_rescue_manifest(root, generation_id)
    if manifest.is_file():
        try:
            load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
            return f"in force for {generation_id}"
        except AtomicRescueArtifactError:
            pass  # another calibration's, or damaged: replaced below
    root.mkdir(parents=True, exist_ok=True)
    # The root holds view vectors of the corpus, like the `.recall` directory.
    write_ignore(root)
    profile = embedding_profile(embedder).fingerprint()
    with AtomicViewStore(root / VIEW_STORE_NAME) as views:
        matrix, metadata, counts = assemble(
            parents, views, profile, int(embedder.dim), lambda groups: embed_document_groups(embedder, groups)
        )
    if manifest.parent.exists():
        # This process's cache maps the matrix it last loaded, and Windows refuses to delete it.
        clear_atomic_rescue_artifact_cache()
        shutil.rmtree(manifest.parent, ignore_errors=True)
        if manifest.parent.exists():
            raise AtomicRescueArtifactError(
                f"{manifest.parent} holds an artifact for another calibration and cannot be replaced "
                "while a process has its matrix open; stop that server and run `recall generation rescue`"
            )
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
            ordinary_chunk_count=len(parents),
            source_commit=f"recall-rag {__version__}",
        )
    except OSError:
        # A concurrent build of this generation published first; the writer verified this build's
        # own before publishing, so only the winner's is checked here.
        if not manifest.is_file():
            raise
        load_atomic_rescue_artifact(manifest).assert_lineage(**lineage, embedder=embedder)
    return (
        f"built for {generation_id}: {counts['views']} view(s) of {counts['parents']} chunk(s), "
        f"{counts['embedded_views']} newly embedded"
    )


def _root(env: Mapping[str, str] | None) -> Path | None:
    """The configured root whatever the mode, since a rescue turned off keeps what it built."""
    import os

    value = (os.environ if env is None else env).get(ROOT_ENV, "").strip()
    root = Path(value).expanduser() if value else None
    # The builder refuses a relative root, so nothing was written under one.
    return root if root is not None and root.is_absolute() and root.is_dir() else None


def remove_artifacts(generation_ids: Collection[str], *, env: Mapping[str, str] | None = None) -> int:
    """Remove the artifacts of `generation_ids` (collected or forgotten), returning how many.

    Raises when one is left, which happens on Windows while a process has its matrix open.
    """
    root = _root(env)
    if root is None:
        return 0
    doomed = [resolve_atomic_rescue_manifest(root, gid).parent for gid in sorted(generation_ids)]
    doomed = [entry for entry in doomed if entry.is_dir()]
    if doomed:
        clear_atomic_rescue_artifact_cache()
    for entry in doomed:
        shutil.rmtree(entry, ignore_errors=True)
    left = [entry.name for entry in doomed if entry.exists()]
    if left:
        raise AtomicRescueArtifactError(
            f"atomic rescue artifact(s) {', '.join(left)} under {root} could not be removed while a "
            "process has their matrix open"
        )
    return len(doomed)


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
    names one of `sources` is removed: the forget changed that generation's corpus fingerprint, so
    the artifact no longer serves, and it keeps the forgotten chunks' view vectors.
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
        gid for gid in generation_ids
        if _names_any(resolve_atomic_rescue_manifest(root, gid).parent / "views.json", named)
    ]
    return erased, remove_artifacts(doomed, env=env)


def _names_any(views_json: Path, sources: set[str]) -> bool:
    """Whether the artifact's `views.json` names one of `sources`; False when it cannot be read."""
    try:
        views = json.loads(views_json.read_text(encoding="utf-8"))["views"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return any(isinstance(view, dict) and view.get("source") in sources for view in views)
