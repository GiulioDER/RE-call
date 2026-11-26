"""The compiled graph for one store: when it is on, where it lives, and how it is built and erased.

Both backends call ``ensure_compiled_graph`` with the chunks of the corpus in force and its
generation binding; ``serving_lineage`` names the same lineage at search time, so a search loads
only the artifact built for the corpus it searches. A build never raises into the write or the
promotion that triggered it: the graph is an addition, and losing it must not lose data. A credit
stop sets a backoff so that a Lite store does not call the provider again on every write.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from recall.compiled_graph.artifact import (
    CompiledGraphArtifact,
    CompiledGraphArtifactError,
    artifact_dir,
    load_artifact,
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
from recall.compiled_graph.records import prompt_digest
from recall.embedding_core import Embedder
from recall.types import Chunk

_log = logging.getLogger("recall.compiled_graph")

MODE_ENV = f"{ENV_PREFIX}_MODE"
MODE_BY_TENANT_ENV = f"{ENV_PREFIX}_MODE_BY_TENANT"
ROOT_ENV = f"{ENV_PREFIX}_ARTIFACT_ROOT"
MODES = ("off", "shadow", "active")
CACHE_NAME = "compile-cache.sqlite"
BACKOFF_NAME = "credit-backoff"

Mode = Literal["off", "shadow", "active"]


def _parse_mode(value: str, variable: str) -> Mode:
    mode = value.strip().lower() or "off"
    if mode not in MODES:
        raise ValueError(f"{variable} must be one of {', '.join(MODES)}, not {value!r}")
    return mode  # type: ignore[return-value]  # narrowed by the membership check above


def compiled_graph_mode(env: Mapping[str, str], tenant: str | None) -> Mode:
    """The mode for ``tenant``: its entry in the per-tenant map, else the global mode."""
    for entry in env.get(MODE_BY_TENANT_ENV, "").split(","):
        name, separator, value = entry.strip().partition(":")
        if separator and name.strip() == tenant:
            return _parse_mode(value, MODE_BY_TENANT_ENV)
    return _parse_mode(env.get(MODE_ENV, ""), MODE_ENV)


def artifact_root(env: Mapping[str, str]) -> Path | None:
    value = env.get(ROOT_ENV, "").strip()
    return Path(value).expanduser() if value else None


def tenant_folder(root: Path, tenant: str) -> Path:
    return root / f"tenant-{hashlib.sha256(tenant.encode('utf-8')).hexdigest()[:16]}"


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
    """The source text as its chunk offsets index it: each chunk at its own ``text_start``.

    A chunker that normalised whitespace stored a chunk shorter or longer than its span, so each
    chunk is padded or clipped to its span. Characters no chunk covers are spaces.
    """
    placed: list[tuple[int, int, str]] = []
    for chunk in chunks:
        start, end = chunk.metadata.get("text_start"), chunk.metadata.get("text_end")
        if isinstance(start, int) and isinstance(end, int) and end > start >= 0:
            placed.append((start, end, chunk.text))
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
    still_present: Callable[[str], bool] = lambda source: True,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Build this corpus's artifact when the tenant's mode is on; a one-line outcome, never raises."""
    try:
        if compiled_graph_mode(env, tenant) == "off":
            return ""
        root = artifact_root(env)
        if root is None:
            return ""
        lineage = serving_lineage(tenant, binding, embedder, env)
        if lineage is None:
            return f"compiled graph not built: {ENV_PREFIX}_MODEL and a generation are required"
        folder = tenant_folder(root, tenant)
        backoff = folder / BACKOFF_NAME
        if force:
            backoff.unlink(missing_ok=True)
        elif backoff.exists():
            return "compiled graph not built: credit backoff in force; run `recall graph compile`"
        if (artifact_dir(folder, lineage) / "manifest.json").is_file():
            return f"compiled graph in force for {lineage['generation']}"
        config = ProviderConfig.from_env(env)
        budget = Budget.from_env(env)
        base = {key: value for key, value in lineage.items() if key not in ("model", "prompt")}
        sources = _sources(chunks)
        artifact, report = build_compiled_graph(
            sources, embedder, config, transport, CompileCache(root / CACHE_NAME), budget,
            lineage=base, still_present=still_present, sleep=sleep,
        )
        spent = f"{report.calls} call(s), USD {report.spent_usd:.4f}"
        if artifact is None:
            if report.stopped == "credit":
                folder.mkdir(parents=True, exist_ok=True)
                backoff.write_text("credit stop\n", encoding="utf-8")
            return f"compiled graph not built: stopped on {report.stopped} after {spent}"
        if not all(still_present(source.source) for source in sources):
            return "compiled graph not built: a source was forgotten during the build"
        write_artifact(folder, artifact)
        return (
            f"compiled graph built for {lineage['generation']}: {len(artifact.records)} record(s), "
            f"{report.sources_rejected} source(s) rejected, {spent}"
        )
    except (CompiledGraphConfigError, CompiledGraphArtifactError, ValueError, OSError) as exc:
        # The write or promotion that called this has already succeeded; it reports this line.
        _log.warning("compiled graph build for tenant %r did not complete: %s", tenant, exc)
        return f"compiled graph not built: {exc}"


def erase_forgotten_sources(sources: Collection[str], *, env: Mapping[str, str]) -> int:
    """Remove every artifact and cache row holding ``sources``; the number of artifacts removed.

    Artifacts are immutable, so one holding a forgotten source is removed whole; the tenant then
    searches without the graph until the next build, which no longer sees the source.
    """
    root = artifact_root(env)
    if root is None or not sources or not root.exists():
        return 0
    wanted = set(sources)
    if (root / CACHE_NAME).exists():
        CompileCache(root / CACHE_NAME).erase_sources(wanted)
    removed = 0
    for records in root.glob("tenant-*/cg-*/records.jsonl"):
        import json

        holds = any(
            json.loads(line).get("source") in wanted
            for line in records.read_text(encoding="utf-8").splitlines()
            if line
        )
        if holds:
            shutil.rmtree(records.parent)
            removed += 1
    return removed


def load_serving_artifact(
    tenant: str, binding: Mapping[str, str], embedder: Embedder, env: Mapping[str, str]
) -> CompiledGraphArtifact:
    """The artifact a search of this corpus serves; raises a ``CompiledGraphArtifactError``."""
    lineage = serving_lineage(tenant, binding, embedder, env)
    root = artifact_root(env)
    if lineage is None or root is None:
        raise CompiledGraphArtifactError("no lineage or artifact root to load")
    return load_artifact(tenant_folder(root, tenant), lineage)


class ServingStore(Protocol):
    """What ``LiteStore`` and ``GenerationStore`` both offer a build."""

    @property
    def tenant(self) -> str: ...

    def iter_chunks(self, batch_size: int = ...) -> Iterator[Chunk]: ...

    def generation_binding(self) -> dict[str, str]: ...

    def sources_present(self, sources: Sequence[str]) -> set[str]: ...


def ensure_for_store(
    store: ServingStore, embedder: Embedder, env: Mapping[str, str], *, force: bool = False
) -> str:
    """``ensure_compiled_graph`` over the corpus ``store`` serves now."""
    if compiled_graph_mode(env, store.tenant) == "off" or artifact_root(env) is None:
        return ""
    return ensure_compiled_graph(
        store.tenant,
        store.iter_chunks(),
        store.generation_binding(),
        embedder,
        env=env,
        transport=urllib_transport,
        force=force,
        still_present=lambda source: bool(store.sources_present([source])),
    )


def estimate_compile(tenant: str, chunks: Iterable[Chunk], env: Mapping[str, str]) -> str:
    """What a build would send now, priced at its worst case; sends nothing."""
    from recall.compiled_graph.anchors import (
        build_source_anchors,
        compile_payload,
        encode_stored_data,
        fit_payload,
    )
    from recall.compiled_graph.build import PAYLOAD_BUDGET_CHARS
    from recall.compiled_graph.cache import cache_key
    from recall.compiled_graph.provider import TEMPLATE_SLACK_TOKENS
    from recall.compiled_graph.records import COMPILE_SYSTEM_PROMPT

    root = artifact_root(env)
    if root is None:
        return f"compiled graph: {ROOT_ENV} is not set"
    config = ProviderConfig.from_env(env)
    budget = Budget.from_env(env)
    cache = CompileCache(root / CACHE_NAME)
    pending = cached = 0
    worst = 0.0
    for source in _sources(chunks):
        anchors = build_source_anchors(source.source, source.body)
        if not anchors:
            continue
        body_hash = hashlib.sha256(source.body.encode("utf-8")).hexdigest()
        if cache.get(cache_key(config, source.source, body_hash)) is not None:
            cached += 1
            continue
        payload = compile_payload(source.source, anchors)
        payload = fit_payload(payload, PAYLOAD_BUDGET_CHARS) or payload
        request_bytes = (
            len(COMPILE_SYSTEM_PROMPT.encode("utf-8"))
            + len(encode_stored_data(payload).encode("utf-8"))
            + TEMPLATE_SLACK_TOKENS
        )
        worst += (
            request_bytes / 1000 * budget.cost_in_per_1k + budget.max_tokens / 1000 * budget.cost_out_per_1k
        )
        pending += 1
    return (
        f"compiled graph dry run for tenant {tenant!r}: {pending} source(s) to compile, {cached} cached; "
        f"worst case USD {worst:.4f} for one attempt each, cap USD {budget.max_usd:.2f}; nothing was sent"
    )


def ensure_for_generation(
    dsn: str,
    tenant: str,
    generation_id: str,
    make_embedder: Callable[[], Embedder],
    env: Mapping[str, str],
) -> str:
    """Build ``generation_id``'s artifact before it is promoted; never raises into the promotion."""
    if compiled_graph_mode(env, tenant) == "off" or artifact_root(env) is None:
        return ""
    from recall.generation_store import GenerationStore

    try:
        embedder = make_embedder()
        store = GenerationStore(dsn, embedder.dim, tenant=tenant)
        with store:
            store.set_fixed_generation(generation_id)
            return ensure_for_store(store, embedder, env)
    except Exception as exc:  # BROAD-CATCH: error-translation  # the promotion stands; the operator sees this line
        _log.warning("compiled graph build for %s did not complete: %s", generation_id, exc)
        return f"compiled graph not built for {generation_id}: {type(exc).__name__}"
