"""The compiled graph for one store: when it is on, where it lives, and how it is built and erased.

Both backends call ``ensure_compiled_graph`` with the chunks of the corpus in force and its
generation binding; ``serving_lineage`` names the same lineage at search time, so a search loads
only the artifact built for the corpus it searches. A build never raises into the write or the
promotion that triggered it: the graph is an addition, and losing it must not lose data. A credit
stop, or a provider refusing the configuration, sets a backoff so that a Lite store does not call
the provider again on every write; `recall graph compile --force` clears it. A budget stop sets
none, since the cache keeps what was paid for and the next build goes further.

Each build records the root it writes under in the store, so a forget run in a process with
another root still erases there; the root must be absolute for the same reason. Publishing, and
caching a compile, are ordered against a forget's erasure by the tenant folder's publish mutex
and erasure counter (`_committer`), so a build cannot publish text a forget has already erased.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from recall.compiled_graph.artifact import (
    ArtifactUnconfigured,
    ArtifactDamaged,
    CompiledGraphArtifact,
    CompiledGraphArtifactError,
    artifact_dir,
    artifact_sources,
    load_artifact,
    stored_lineage,
    stored_owner,
    write_artifact,
)
from recall.compiled_graph.budget import Budget
from recall.compiled_graph.build import SourceInput, build_compiled_graph
from recall.compiled_graph.cache import CompileCache
from recall.compiled_graph.provider import (
    ENV_PREFIX,
    CompiledGraphConfigError,
    ProviderConfig,
    Transport,
    urllib_transport,
)
from recall._env import modes_by_tenant
from recall.compiled_graph.promotion import placements
from recall.compiled_graph.records import prompt_digest
from recall.embedding_core import Embedder
from recall.errors import RecallError
from recall.types import Chunk

_log = logging.getLogger("recall.compiled_graph")

MODE_ENV = f"{ENV_PREFIX}_MODE"
MODE_BY_TENANT_ENV = f"{ENV_PREFIX}_MODE_BY_TENANT"
ROOT_ENV = f"{ENV_PREFIX}_ARTIFACT_ROOT"
MODES = ("off", "shadow", "active")
CACHE_NAME = "compile-cache.sqlite"
BACKOFF_NAME = "credit-backoff"
EPOCH_NAME = "erasure-epoch"
#: Stops the next write would hit again: no credit, or a configuration the provider refuses. A
#: budget stop is not one: the cache keeps what it paid for, so the next build goes further.
BACKOFF_STOPS = ("credit", "provider")
#: How often a cache write or publish is retried while forgets change the folder under it.
COMMIT_ATTEMPTS = 5


class CommitContended(CompiledGraphArtifactError):
    """Forgets kept landing between a build's presence check and its lock; the build stops."""


class CompiledGraphErasureIncomplete(OSError, RecallError):
    """An erasure that could not finish everywhere; ``unreached`` names the roots it could not use."""

    def __init__(self, message: str, unreached: list[str]) -> None:
        super().__init__(message)
        self.unreached = unreached

Mode = Literal["off", "shadow", "active"]


def _parse_mode(value: str, variable: str) -> Mode:
    mode = value.strip().lower() or "off"
    if mode not in MODES:
        raise ValueError(f"{variable} must be one of {', '.join(MODES)}, not {value!r}")
    return mode  # type: ignore[return-value]  # narrowed by the membership check above


def compiled_graph_mode(env: Mapping[str, str], tenant: str | None) -> Mode:
    """The mode for ``tenant``: its entry in the per-tenant map, else the global mode.

    The whole map is parsed, so a malformed entry (``memory=off``) raises instead of leaving that
    tenant on the global mode, which may be ``active``. The MCP server refuses such a value at
    startup (``recall_mcp.settings``).
    """
    by_tenant = modes_by_tenant(env, MODE_BY_TENANT_ENV, MODES)
    chosen = by_tenant.get(tenant) if tenant is not None else None
    return _parse_mode(chosen if chosen is not None else env.get(MODE_ENV, ""), MODE_ENV)


def artifact_root(env: Mapping[str, str]) -> Path | None:
    """The configured root; refuses a relative one, which each process would resolve differently."""
    value = env.get(ROOT_ENV, "").strip()
    if not value:
        return None
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise CompiledGraphConfigError(f"{ROOT_ENV} must be an absolute path, not {value!r}")
    return root


def tenant_folder(root: Path, tenant: str) -> Path:
    return root / f"tenant-{hashlib.sha256(tenant.encode('utf-8')).hexdigest()[:16]}"


def _publish_lock(folder: Path) -> AbstractContextManager[None]:
    """The lock under which a build publishes or caches and a forget erases, per tenant folder.

    Nothing that takes a store's own lock may run under it: a forget holds its store's lock when
    it takes this one, so the other order would deadlock until this mutex's wait runs out.
    """
    from recall.lite.writer_lock import _mutex

    folder.mkdir(parents=True, exist_ok=True)
    return _mutex(folder, "compiled-graph-publish")


def _epoch(folder: Path) -> str:
    """The folder's erasure counter, which every erasure changes under the publish lock."""
    try:
        return (folder / EPOCH_NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _committer(
    folder: Path, all_present: Callable[[Sequence[str]], bool]
) -> Callable[[Sequence[str], Callable[[], None]], bool]:
    """Run a cache write or a publish only when no forget of its sources could have missed it.

    Presence is asked of the store first, outside the lock; under the lock only the erasure
    counter is compared. A forget that deleted the rows before the presence check is seen there;
    one after it changes the counter before it erases, so either this write is refused or the
    erasure that follows it removes what it wrote. A changed counter means only that some forget
    ran, of these sources or others, so presence is asked again: False is returned only when a
    source is gone, and ``CommitContended`` is raised when forgets keep landing in the window, so
    a still-present source is never silently dropped from a published artifact.
    """

    def commit(sources: Sequence[str], action: Callable[[], None]) -> bool:
        for _ in range(COMMIT_ATTEMPTS):
            before = _epoch(folder)
            if not all_present(sources):
                return False
            with _publish_lock(folder):
                if _epoch(folder) == before:
                    action()
                    return True
        raise CommitContended(f"forgets kept changing the compiled graph folder; tried {COMMIT_ATTEMPTS} times")

    return commit


def _artifact_dirs(folder: Path) -> list[Path]:
    return [path for path in folder.glob("cg-*") if path.is_dir()]


def _known_vectors(folder: Path, embedder_profile: str) -> dict[str, np.ndarray]:
    """Record vectors of earlier artifacts under the same embedder, keyed by record text."""
    from recall.compiled_graph.build import _record_text

    known: dict[str, np.ndarray] = {}
    for directory in _artifact_dirs(folder):
        lineage = stored_lineage(directory)
        if lineage is None or lineage.get("embedder") != embedder_profile:
            continue
        try:
            previous = load_artifact(folder, lineage)
        except CompiledGraphArtifactError:
            continue
        for record, vector in zip(previous.records, previous.vectors):
            known.setdefault(_record_text(record), vector)
    return known


def _prune_superseded(folder: Path, owner: str, keep: Path, serving: Mapping[str, str] | None) -> None:
    """Remove ``owner``'s other artifacts, keeping ``keep`` (the one built) and every artifact of
    the ``serving`` binding's generation and corpus, whatever embedder or model built it (a
    candidate may re-embed with another model while the active generation still serves).

    Another store sharing the tenant's folder keeps its own. Best effort: an artifact that cannot
    be removed now (a reader holds it open on Windows) is left for the next build.
    """
    for path in _artifact_dirs(folder):
        if path == keep or stored_owner(path) != owner:
            continue
        lineage = stored_lineage(path) or {}
        if serving is not None and (lineage.get("generation"), lineage.get("corpus")) == (
            str(serving.get("generation_id")),
            str(serving.get("corpus_fingerprint", "")),
        ):
            continue
        try:
            shutil.rmtree(path)
        except OSError as exc:
            _log.warning("a superseded compiled graph artifact was not removed: %s", exc)


def serving_lineage(
    tenant: str, binding: Mapping[str, str], embedder: Embedder, env: Mapping[str, str]
) -> dict[str, str] | None:
    """The lineage a search of this corpus loads; None without a generation or a model."""
    from recall.embeddings import embedding_profile_id

    generation, model = binding.get("generation_id"), env.get(f"{ENV_PREFIX}_MODEL", "").strip()
    if not generation or not model:
        return None
    return {
        "tenant": tenant,
        "generation": str(generation),
        "corpus": str(binding.get("corpus_fingerprint", "")),
        "embedder": embedding_profile_id(embedder),
        "model": model,
        "prompt": prompt_digest(),
    }


def body_from_chunks(chunks: Iterable[Chunk]) -> str:
    """The source text as its chunk offsets index it: each run of each chunk at its own start
    (`recall.compiled_graph.promotion.placements`).

    A chunker that normalised whitespace stored a chunk shorter or longer than its span, so each
    chunk is padded or clipped to its span. Characters no chunk covers are spaces.
    """
    placed: list[tuple[int, int, str]] = []
    for chunk in chunks:
        for start, end, offset in placements(chunk):
            placed.append((start, end, chunk.text[offset:]))
    if not placed:
        return ""
    body = [" "] * max(end for _, end, _ in placed)
    for start, end, text in sorted(placed):
        span = text[: end - start].ljust(end - start)
        body[start:end] = list(span)
    return "".join(body)


def _sources(chunks: Iterable[Chunk]) -> list[SourceInput]:
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)
    return [
        SourceInput(source, body_from_chunks(group), tuple(group))
        for source, group in sorted(by_source.items())
    ]


def ensure_compiled_graph(
    tenant: str,
    chunks: Iterable[Chunk],
    binding: Mapping[str, str],
    embedder: Embedder,
    *,
    env: Mapping[str, str],
    transport: Transport = urllib_transport,
    force: bool = False,
    retry_rejected: bool = False,
    all_present: Callable[[Sequence[str]], bool] = lambda sources: True,
    record_root: Callable[[Path], None] = lambda root: None,
    owner: str = "",
    serving_binding: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Build this corpus's artifact when the tenant's mode is on; a one-line outcome.

    ``all_present`` says whether the store still holds every one of the sources given; it is
    asked before each cache write and before the publish. ``record_root`` keeps the root in the
    store before anything is written under it. ``force`` clears a backoff; ``retry_rejected`` also
    sends again the sources whose refusal is cached. After a publish, ``owner``'s other artifacts
    are removed except the one ``serving_binding`` names (the binding built, by default).
    """
    try:
        if compiled_graph_mode(env, tenant) == "off":
            return ""
        root = artifact_root(env)
        if root is None:
            return f"compiled graph not built: {ROOT_ENV} is not set"
        lineage = serving_lineage(tenant, binding, embedder, env)
        if lineage is None:
            return f"compiled graph not built: {ENV_PREFIX}_MODEL and a generation are required"
        folder = tenant_folder(root, tenant)
        backoff = folder / BACKOFF_NAME
        if force:
            backoff.unlink(missing_ok=True)
        elif backoff.exists():
            reason = backoff.read_text(encoding="utf-8").strip() or "a stop"
            return (
                f"compiled graph not built: backoff after {reason} in force; run "
                "`recall graph compile --force` once the cause is fixed"
            )
        if (artifact_dir(folder, lineage) / "manifest.json").is_file():
            return f"compiled graph in force for {lineage['generation']}"
        config = ProviderConfig.from_env(env)
        budget = Budget.from_env(env)
        base = {key: value for key, value in lineage.items() if key not in ("model", "prompt")}
        sources = _sources(chunks)
        record_root(root)
        commit = _committer(folder, all_present)
        artifact, report = build_compiled_graph(
            sources, embedder, config, transport, CompileCache(folder / CACHE_NAME), budget,
            lineage=base, commit=commit, known_vectors=_known_vectors(folder, lineage["embedder"]),
            retry_rejected=retry_rejected, sleep=sleep,
        )
        spent = f"{report.calls} call(s), USD {report.spent_usd:.4f}"
        if artifact is None:
            if report.stopped in BACKOFF_STOPS:
                folder.mkdir(parents=True, exist_ok=True)
                backoff.write_text(f"a {report.stopped} stop\n", encoding="utf-8")
            return f"compiled graph not built: stopped on {report.stopped} after {spent}"
        def publish() -> None:
            published = write_artifact(folder, artifact, owner=owner)
            _prune_superseded(folder, owner, published, serving_binding)

        if not commit([source.source for source in sources], publish):
            return "compiled graph not built: a source was forgotten during the build"
        return (
            f"compiled graph built for {lineage['generation']}: {len(artifact.records)} record(s), "
            f"{report.sources_rejected} source(s) rejected, {report.sources_without_offsets} "
            f"source(s) skipped because no chunk carries character offsets, {spent}"
        )
    except (CompiledGraphConfigError, CompiledGraphArtifactError, ValueError, OSError) as exc:
        # The write or promotion that called this has already succeeded; it reports this line.
        _log.warning("compiled graph build for tenant %r did not complete: %s", tenant, exc)
        return f"compiled graph not built: {exc}"


def erase_forgotten_sources(
    sources: Collection[str],
    *,
    tenant: str,
    env: Mapping[str, str],
    recorded_roots: Iterable[str] = (),
) -> tuple[int, list[str]]:
    """Remove ``tenant``'s artifacts and cache rows holding ``sources``, under the configured root
    and every root a build recorded; the artifacts removed, and the recorded roots not reachable.

    Artifacts are immutable, so one holding a forgotten source is removed whole; the tenant then
    searches without the graph until the next build, which no longer sees the source. A leftover
    staging directory, or an artifact whose records cannot be read, may hold the text, so it is
    removed too. A root that cannot be used (a relative configured one, a recorded UNC or device
    path, which is never probed, one this host cannot read) is reported and the others are still
    erased; every root and artifact is visited before a failure is raised.
    """
    if not sources:
        return 0, []
    roots: list[Path] = []
    unreached: list[str] = []
    try:
        configured = artifact_root(env)
    except CompiledGraphConfigError:
        unreached.append(env.get(ROOT_ENV, "").strip())
        configured = None
    if configured is not None:
        roots.append(configured)
    for raw in recorded_roots:
        path = Path(raw)
        if path in roots:
            continue
        if raw.replace("\\", "/").startswith("//") or not path.is_absolute():
            # Probing a UNC or device path would open a connection to whatever host the record names.
            unreached.append(raw)
            continue
        try:
            usable = path.is_dir()
        except OSError:  # Python before 3.14 raises PermissionError from `is_dir`
            usable = False
        if usable:
            roots.append(path)
        else:
            unreached.append(raw)
    wanted = set(sources)
    removed = 0
    failures: list[str] = []
    for root in roots:
        try:
            removed += _erase_folder(tenant_folder(root, tenant), wanted, failures)
        except Exception as exc:  # BROAD-CATCH: fail-closed  # this root is reported and the others are still erased
            failures.append(f"{root}: {type(exc).__name__}: {exc}")
    if failures:
        raise CompiledGraphErasureIncomplete(
            "compiled graph erasure did not complete: " + "; ".join(failures), unreached
        )
    return removed, unreached


def _erase_folder(folder: Path, wanted: set[str], failures: list[str]) -> int:
    """Erase ``wanted`` from one tenant folder under its publish lock; the artifacts removed."""
    if not folder.is_dir():
        return 0
    removed = 0
    with _publish_lock(folder):
        # Before anything is erased, so a build that checked presence before this forget does
        # not publish or cache afterwards (`_committer`).
        (folder / EPOCH_NAME).write_text(uuid.uuid4().hex, encoding="utf-8")
        if (folder / CACHE_NAME).exists():
            try:
                CompileCache(folder / CACHE_NAME).erase_sources(wanted)
            except (sqlite3.Error, OSError) as exc:
                failures.append(f"{folder / CACHE_NAME}: {type(exc).__name__}: {exc}")
        for directory in [*_artifact_dirs(folder), *folder.glob(".cg-staging-*")]:
            try:
                # No manifest is an earlier erase that stopped part way (a write publishes one
                # by rename), and its text may be what that erase was forgetting.
                holds = (
                    directory.name.startswith(".")
                    or not (directory / "manifest.json").is_file()
                    or bool(artifact_sources(directory) & wanted)
                )
            except ArtifactDamaged:
                holds = True
            if not holds:
                continue
            try:
                # The manifest goes first: a load refuses a directory without one, so an rmtree
                # that stops part way leaves nothing any process (or its cache) can serve.
                (directory / "manifest.json").unlink(missing_ok=True)
                shutil.rmtree(directory)
            except OSError as exc:
                failures.append(f"{directory}: {exc}")
                continue
            removed += 1
    return removed


def load_serving_artifact(
    tenant: str, binding: Mapping[str, str], embedder: Embedder, env: Mapping[str, str]
) -> CompiledGraphArtifact:
    """The artifact a search of this corpus serves; raises a ``CompiledGraphArtifactError``."""
    lineage = serving_lineage(tenant, binding, embedder, env)
    root = artifact_root(env)
    if lineage is None or root is None:
        raise ArtifactUnconfigured(
            f"a search needs a generation, {ENV_PREFIX}_MODEL and {ROOT_ENV} to load the graph"
        )
    return load_artifact(tenant_folder(root, tenant), lineage)


class ServingStore(Protocol):
    """What ``LiteStore`` and ``GenerationStore`` both offer a build."""

    @property
    def tenant(self) -> str: ...

    def iter_chunks(self, batch_size: int = ...) -> Iterator[Chunk]: ...

    def generation_binding(self) -> dict[str, str]: ...

    def sources_present(self, sources: Sequence[str]) -> set[str]: ...

    def record_compiled_graph_root(self, root: Path) -> None: ...


def _store_owner(store: ServingStore) -> str:
    """Which store an artifact belongs to, so stores sharing a tenant's folder keep their own."""
    path = getattr(store, "path", None)
    if path is not None:
        identity = f"lite:{Path(path).resolve()}"
    else:
        from psycopg.conninfo import conninfo_to_dict

        # Without the password, which a hash of the manifest could otherwise be tested against.
        fields = conninfo_to_dict(str(getattr(store, "_dsn", "")))
        fields.pop("password", None)
        identity = "pg:" + ";".join(f"{key}={fields[key]}" for key in sorted(fields))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def ensure_for_store(
    store: ServingStore,
    embedder: Embedder,
    env: Mapping[str, str],
    *,
    force: bool = False,
    retry_rejected: bool = False,
    serving_binding: Mapping[str, str] | None = None,
) -> str:
    """``ensure_compiled_graph`` over the corpus ``store`` serves now; never raises.

    ``serving_binding`` is the binding searches load while this one is built (the active
    generation, for a PostgreSQL candidate); its artifact is kept when older ones are pruned.
    """
    try:
        if compiled_graph_mode(env, store.tenant) == "off":
            return ""
        return ensure_compiled_graph(
            store.tenant,
            store.iter_chunks(),
            store.generation_binding(),
            embedder,
            env=env,
            transport=urllib_transport,
            force=force,
            retry_rejected=retry_rejected,
            all_present=lambda sources: len(store.sources_present(list(sources))) == len(set(sources)),
            record_root=store.record_compiled_graph_root,
            owner=_store_owner(store),
            serving_binding=serving_binding,
        )
    except Exception as exc:  # BROAD-CATCH: error-translation  # the write that called this has succeeded; it logs this line
        _log.warning("compiled graph build for tenant %r did not complete: %s", store.tenant, exc)
        return f"compiled graph not built: {type(exc).__name__}: {exc}"


def estimate_compile(
    tenant: str, chunks: Iterable[Chunk], env: Mapping[str, str], *, retry_rejected: bool = False
) -> str:
    """What a build would send now, one attempt per uncached source priced as the budget reserves
    it (a source whose refusal is cached only with ``retry_rejected``); sends nothing."""
    from recall.compiled_graph.anchors import build_source_anchors, compile_payload, fit_payload
    from recall.compiled_graph.build import PAYLOAD_BUDGET_CHARS, without_offsets
    from recall.compiled_graph.cache import cache_key
    from recall.compiled_graph.provider import TEMPLATE_SLACK_TOKENS, request_body

    root = artifact_root(env)
    if root is None:
        return f"compiled graph: {ROOT_ENV} is not set"
    config = ProviderConfig.from_env(env)
    budget = Budget.from_env(env)
    cache = CompileCache(tenant_folder(root, tenant) / CACHE_NAME)
    pending = cached = refused = unplaced = 0
    worst = 0.0
    for source in _sources(chunks):
        anchors = build_source_anchors(source.source, source.body)
        if not anchors:
            unplaced += without_offsets(source)
            continue
        key = cache_key(config, source.source, hashlib.sha256(source.body.encode("utf-8")).hexdigest())
        if cache.get(key) is not None:
            cached += 1
            continue
        if not retry_rejected and cache.rejection(key) is not None:
            refused += 1
            continue
        payload = compile_payload(source.source, anchors)
        payload = fit_payload(payload, PAYLOAD_BUDGET_CHARS) or payload
        worst += budget.worst_case(len(request_body(config, payload)) + TEMPLATE_SLACK_TOKENS)
        pending += 1
    return (
        f"compiled graph dry run for tenant {tenant!r}: {pending} source(s) to compile, {cached} cached, "
        f"{refused} refused before (sent again only with --retry-rejected), "
        f"{unplaced} skipped because no chunk carries character offsets; "
        f"worst case USD {worst:.4f} for one attempt each, cap USD {budget.max_usd:.2f}; nothing was sent"
    )


def active_binding(store: ServingStore) -> dict[str, str] | None:
    """The binding the active generation serves, read before a store is fixed to a candidate;
    None on a first install, which has no active generation yet."""
    from recall.generation_types import NoActiveGeneration

    try:
        return store.generation_binding()
    except NoActiveGeneration:
        return None


def ensure_for_generation(
    dsn: str,
    tenant: str,
    generation_id: str,
    make_embedder: Callable[[], Embedder],
    env: Mapping[str, str],
) -> str:
    """Build ``generation_id``'s artifact before it is promoted; never raises into the promotion."""
    from recall.generation_store import GenerationStore

    try:
        if compiled_graph_mode(env, tenant) == "off":
            return ""
        embedder = make_embedder()
        store = GenerationStore(dsn, embedder.dim, tenant=tenant)
        with store:
            serving = active_binding(store)
            store.set_fixed_generation(generation_id)
            return ensure_for_store(store, embedder, env, serving_binding=serving)
    except Exception as exc:  # BROAD-CATCH: error-translation  # the promotion stands; the operator sees this line
        _log.warning("compiled graph build for %s did not complete: %s", generation_id, exc)
        return f"compiled graph not built for {generation_id}: {type(exc).__name__}"
