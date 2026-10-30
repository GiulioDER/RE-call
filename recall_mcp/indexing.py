"""MCP local filesystem indexing boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
import re
from typing import TYPE_CHECKING

from recall._env import strict_bool
from recall.cache import default_cache
from recall.context import context_policy_for_profile
from recall.embeddings import Embedder, embedding_profile_id
# The budget lives in `recall.index_budget`, shared with the lite store's automatic refresh; the
# constants stay importable from here (and from `recall_mcp.service`) for existing callers.
from recall.index_budget import DEFAULT_MAX_INDEX_BYTES as DEFAULT_MAX_INDEX_BYTES
from recall.index_budget import DEFAULT_MAX_INDEX_FILES as DEFAULT_MAX_INDEX_FILES
from recall.index_budget import budget_from_env, budget_refusal
from recall.errors import RecallError
from recall.index import Chunker, Indexer, ShadowIndexTarget, candidate_files, chunk_text
from recall.observability import get_logger
from recall.runtime_route import RouteConfigurationError, resolve_runtime_route
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall_mcp.models import IndexResult
from recall_mcp.settings import runtime_environment

if TYPE_CHECKING:
    from recall.control_plane import ControlPlane
    from recall.store import PgVectorStore

_log = get_logger("mcp.service")
REDACTED_PATH = "<server index root>"


def _scrub_paths(message: str, *paths: Path) -> str:
    """Replace server-side absolute paths in an indexing error."""
    for path in paths:
        raw = str(path)
        for form in (raw, raw.replace("\\", "\\\\")):
            if form:
                message = message.replace(form, REDACTED_PATH)
    return message


CandidateFiles = Callable[..., list[Path]]
CandidateFilesProvider = Callable[[], CandidateFiles]
_candidate_files_provider: CandidateFilesProvider | None = None


def _set_candidate_files_provider(provider: CandidateFilesProvider) -> None:
    """Register a compatibility provider without making this boundary import its facade."""
    global _candidate_files_provider
    _candidate_files_provider = provider


class IndexPreflightError(ValueError, RecallError):
    """Index request was refused before the indexer could write corpus state."""


def _is_lite(store: object) -> bool:
    from recall.lite import LiteStore

    return isinstance(store, LiteStore)


def index_memory(
    store: PgVectorStore,
    embedder: Embedder,
    path: str,
    on_measured: Callable[[int, int], None] | None = None,
    shadow_store: PgVectorStore | None = None,
    shadow_embedder: Embedder | None = None,
    control_plane: ControlPlane | None = None,
    glob: str | None = None,
    chunker: Chunker = chunk_text,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
    candidate_files_fn: CandidateFiles | None = None,
    lane_purpose: str = "index",
    source_namespace: str | None = None,
) -> IndexResult:
    """Index a markdown file or folder into memory; return counts plus a human message.

    On a lite store the whole run (read hashes, embed, write, prune, calibrate) holds the file's
    write lane, so two servers indexing one file take turns (`recall.lite.writer_lock`).
    """
    from recall.backends import write_session
    from recall.index_lock import single_writer

    if source_namespace is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", source_namespace):
            raise IndexPreflightError("source_namespace must contain 1 to 64 safe characters")
        if security_policy is not None:
            raise IndexPreflightError(
                "stable source namespaces are not supported with source security policies; "
                "use a reviewed source manifest instead"
            )

    with write_session(store, lane_purpose):
        guard = single_writer(store, env=env) if source_namespace is not None and not _is_lite(store) else nullcontext()
        with guard:
            return _index_memory_unlocked(
                store, embedder, path, on_measured, shadow_store, shadow_embedder, control_plane, glob,
                chunker, security_policy, security_context, env, candidate_files_fn,
                _publication_namespace=source_namespace,
            )


def _index_memory_unlocked(
    store: PgVectorStore,
    embedder: Embedder,
    path: str,
    on_measured: Callable[[int, int], None] | None = None,
    shadow_store: PgVectorStore | None = None,
    shadow_embedder: Embedder | None = None,
    control_plane: ControlPlane | None = None,
    glob: str | None = None,
    chunker: Chunker = chunk_text,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
    candidate_files_fn: CandidateFiles | None = None,
    _publication_namespace: str | None = None,
) -> IndexResult:
    """`index_memory` without the lite store's write lane; call `index_memory`."""
    values = dict(runtime_environment() if env is None else env)
    try:
        route = resolve_runtime_route(
            values,
            enterprise=strict_bool(
                values.get("RECALL_ENTERPRISE_CONTROL_PLANE"),
                name="RECALL_ENTERPRISE_CONTROL_PLANE",
            )
        )
    except RouteConfigurationError as exc:
        raise IndexPreflightError(str(exc)) from exc
    if route.uses_generation:
        if route.environment == "production":
            raise IndexPreflightError(
                "local filesystem indexing is development-only; production ingestion requires an "
                "immutable S3 manifest"
            )
        raise IndexPreflightError(
            "legacy filesystem indexing is disabled on the generation route; build an immutable "
            "manifest and use generation build"
        )

    root = Path(values.get("RECALL_INDEX_ROOT", ".")).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root):
        _log.warning("refused index path %r: outside the index root %s", path, root)
        raise IndexPreflightError(
            f"path {path!r} is outside the directory this server is allowed to index; "
            "an operator can widen it with RECALL_INDEX_ROOT."
        )
    if not target.exists():
        raise IndexPreflightError(f"path not found: {path!r}")

    try:
        budget = budget_from_env(values)
    except ValueError as exc:
        raise IndexPreflightError(str(exc)) from exc
    if candidate_files_fn is not None:
        files_fn = candidate_files_fn
    elif _candidate_files_provider is not None:
        files_fn = _candidate_files_provider()
    else:
        files_fn = candidate_files
    try:
        files = files_fn(target, glob) if glob is not None else files_fn(target)
    except (OSError, PermissionError) as exc:
        raise IndexPreflightError(str(exc)) from exc
    subject = f"index request for {path!r}"
    if (refusal := budget_refusal(subject, len(files), 0, budget)) is not None:
        raise IndexPreflightError(refusal)
    total_bytes = 0
    try:
        for file in files:
            try:
                total_bytes += file.stat().st_size
            except (FileNotFoundError, NotADirectoryError):
                continue
    except (OSError, PermissionError) as exc:
        raise IndexPreflightError(str(exc)) from exc
    if (refusal := budget_refusal(subject, len(files), total_bytes, budget)) is not None:
        raise IndexPreflightError(refusal)
    if security_policy is not None:
        if security_context is None:
            raise IndexPreflightError(
                "security_context is required when security_policy is configured"
            )
        # The names the indexer will store and decide on: RECALL_INDEX_ROOT-relative on a lite
        # store (`key_root` below), else relative to the indexed path. Deciding on other names
        # here would let a request pass this check, be charged, and be denied in the indexer.
        relative_paths = (
            [file.relative_to(root).as_posix() for file in files]
            if _is_lite(store)
            else [file.relative_to(target).as_posix() for file in files]
            if target.is_dir()
            else [file.name for file in files]
        )
        for relative in relative_paths:
            decision = security_policy.decide(relative, security_context)
            if not decision.allowed:
                raise IndexPreflightError(f"source {relative!r} denied: {decision.reason}")

    if _publication_namespace is not None:
        # Admission above measured the disposable upload. Publish only after all
        # guards passed, while the caller holds the corpus's writer boundary.
        from recall.desktop.uploads import _promote_staged, discard_staging

        if not target.is_dir():
            raise IndexPreflightError("namespaced uploads require a staged directory")
        stable = target.parent / f"sync-memory-{_publication_namespace}"
        destination = stable.resolve()
        published_files = [stable / file.relative_to(target) for file in files]
        if not destination.is_relative_to(root) or any(
            not file.resolve().is_relative_to(destination) for file in published_files
        ):
            raise IndexPreflightError("stable upload destination escapes the index root")
        try:
            _promote_staged(target, stable)
        finally:
            discard_staging(target)
        target, files = stable, published_files

    if on_measured is not None:
        on_measured(len(files), total_bytes)

    try:
        shadow_target = None
        if any(value is not None for value in (shadow_store, shadow_embedder, control_plane)):
            if shadow_store is None or shadow_embedder is None or control_plane is None:
                raise ValueError("shadow indexing requires store, embedder, and control plane")
            shadow_target = ShadowIndexTarget(
                store=shadow_store,
                embedder=shadow_embedder,
                control_plane=control_plane,
                context_policy=context_policy_for_profile(embedding_profile_id(shadow_embedder)),
            )
        # The shared embedding cache, as at every other indexing entry point: unchanged chunk
        # text is served from it rather than re-embedded. Opened only after preflight, so a
        # refused request never touches the cache file. `RECALL_EMBED_CACHE` moves or disables it.
        with default_cache() as cache:
            indexer = Indexer(
                store,
                embedder,
                chunker=chunker,
                cache=cache,
                context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
                shadow=shadow_target,
                security_policy=security_policy,
                security_context=security_context,
                env=values,
            )
            # Publication already holds this same guard; a second PostgreSQL
            # connection would contend with the caller's non-reentrant lock.
            index_path = indexer._index_path if _publication_namespace is not None else indexer.index_path
            stats = index_path(
                target,
                files=files,
                # A lite server names every file relative to RECALL_INDEX_ROOT, as its automatic
                # refresh does, so the name does not depend on which of the two indexed it first
                # (owner decision, 2026-10-13). Postgres servers keep the indexed path's names.
                key_root=root if _is_lite(store) else None,
            )
    except (RuntimeError, OSError, ValueError) as exc:
        _log.warning("index of %r failed: %s", path, exc)
        scrubbed = _scrub_paths(str(exc), target, root)
        raise type(exc)(scrubbed) from exc
    message = f"Indexed {stats.chunks} chunk(s) from {stats.files} file(s) into memory."
    if stats.skipped:
        message += f" {stats.skipped} file(s) were unchanged and not re-embedded."
    if stats.deleted:
        message += f" Pruned {stats.deleted} source(s) whose files are gone from disk."
    from recall.lite import LiteStore

    if isinstance(store, LiteStore):
        # A local install has nobody to run a calibration by hand: keep one in force whenever the
        # memory changed, so strict search starts answering on its own. A changed memory carries
        # its calibration forward when the threshold still re-verifies (milliseconds, no
        # embedding); a refit, when one is due, runs in the background so the write returns now.
        from recall.lite.calibration import ensure_calibrated

        outcome = ensure_calibrated(store, embedder, background=True)
        message += f" Calibration: {outcome.status.value}. {outcome.reason}"
    # The facts the indexed memos state, on a lite store or the legacy table (a generation store
    # records them when its generation is promoted). The rule is read from the process
    # environment, not `values`: it is applied again whenever facts are read, there, and a private
    # env here would let recording and reading disagree. The controller role's DSN is read from
    # `values`, the runtime environment with its secrets, as `recall_apply_fact` and forget read it.
    from recall.stated_facts import reconcile_stated_facts

    stated = reconcile_stated_facts(store, write_env=values)
    if stated.enabled:
        message += f" Stated facts: {stated.summary()}."
    return IndexResult(
        files=stats.files,
        chunks=stats.chunks,
        skipped=stats.skipped,
        deleted=stats.deleted,
        message=message,
    )
