"""Rank authorized one-hop Jira destinations for the connector's pinned generation."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.supersession import store_supersession_graph
from recall.trust_gate import evaluate
from recall.trust_policy import dependency_mode_from_env
from recall.types import Chunk, RetrievalResult, ScoredChunk, TrustedHit, TrustedResult, metadata_ordinal
from recall_mcp.settings import jira_exact_id_base, jira_exact_id_source_prefix

KEY = re.compile(r"[A-Z][A-Z0-9]+-[0-9]+")
TITLE_NOISE = re.compile(r"https?://\S+|\b[A-Z][A-Z0-9]+-[0-9]+\b", re.IGNORECASE)
MAX_EXPORT_BYTES = 1024 * 1024


def rank_jira_documents(query: str, docs: list[str], api_key: str) -> tuple[list[int], int]:
    import httpx2

    try:
        response = httpx2.post(
            "https://api.voyageai.com/v1/rerank", headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "documents": docs, "model": "rerank-3", "truncation": False,
                  "return_documents": False},
            timeout=0.9, follow_redirects=False,
        )
        response.raise_for_status()
    except httpx2.HTTPError as error:
        raise ValueError("Voyage reranking unavailable") from error
    result = response.json()
    ranked = sorted(result["data"], key=lambda item: (-item["relevance_score"], item["index"]))
    order = [item["index"] for item in ranked]
    tokens = result["usage"]["total_tokens"]
    if (result["model"] != "rerank-3" or sorted(order) != list(range(len(docs)))
            or any(type(index) is not int for index in order)
            or any(not math.isfinite(item["relevance_score"]) for item in ranked)
            or type(tokens) is not int or tokens <= 0):
        raise ValueError("Invalid Voyage ranking")
    return order, tokens


def _verified_export(
    env: Mapping[str, str], tenant: str, generation: str
) -> tuple[str, dict[str, list[str]]] | None:
    filename = env.get("RECALL_JIRA_NATIVE_LINKS_FILE")
    if not filename:
        return None
    try:
        with Path(filename).open("rb") as stream:
            raw = stream.read(MAX_EXPORT_BYTES + 1)
        if len(raw) > MAX_EXPORT_BYTES:
            return None
        export = json.loads(raw)
        if not isinstance(export, dict) or export.get("tenant_id") != tenant or export.get("generation_id") != generation:
            return None
        base = jira_exact_id_base(export.get("site", ""))
        configured_base = jira_exact_id_base(env.get("RECALL_JIRA_EXACT_ID_BASE_URL", ""))
        prefix = jira_exact_id_source_prefix(env.get("RECALL_JIRA_EXACT_ID_SOURCE_PREFIX", ""))
        links = export.get("links")
        if not base or (configured_base and base != configured_base) or not isinstance(links, dict):
            return None
        if any(not KEY.fullmatch(key) or not isinstance(targets, list) or len(targets) > 64
               or any(not isinstance(target, str) or not KEY.fullmatch(target) for target in targets)
               for key, targets in links.items()):
            return None
        return (prefix + "/" if prefix else base + "/browse/"), links
    except (OSError, TypeError, ValueError, AttributeError):
        return None


def native_jira_related(
    store: Any, result: TrustedResult, security: SourceSecurityPolicy, access: AccessContext,
    env: Mapping[str, str],
) -> tuple[list[TrustedHit], int]:
    """Keep primary retrieval intact and rerank only within a separately authorized expansion."""
    api_key = env.get("VOYAGE_API_KEY", "")
    if (not api_key or not access.egress_allowed or result.abstained or not result.calibrated
            or result.staleness.stale or result.tenant_id != access.tenant
            or store.tenant != access.tenant or not result.generation_id):
        return [], 0
    verified = _verified_export(env, access.tenant, result.generation_id)
    read = getattr(store, "iter_scoped_chunks", None)
    hydrate = getattr(store, "chunk_by_id", None)
    if verified is None or not callable(read) or not callable(hydrate):
        return [], 0
    prefix, graph = verified
    is_export = prefix.startswith(("file://", "s3://"))
    suffix = ".md" if is_export else ""
    primary = {hit.chunk.source for hit in result.hits}
    support: Counter[str] = Counter()
    parents: dict[str, Chunk] = {}
    seeds: set[str] = set()
    for hit in result.hits[:8]:
        seed = hit.chunk
        if hit.verdict != "ok" or seed.source in seeds or not security.allows_stored_chunk(seed, access):
            continue
        if not seed.source.startswith(prefix) or not seed.source.endswith(suffix):
            continue
        key = seed.source[len(prefix):].removesuffix(suffix)
        if not KEY.fullmatch(key):
            continue
        seeds.add(seed.source)
        for target in set(graph.get(key, [])) - {key}:
            source = prefix + target + suffix
            if source not in primary:
                support[source] += 1
                parents.setdefault(source, seed)
    sources = sorted(support, key=lambda source: (-support[source], source))[:16]
    if not sources:
        return [], 0
    first: dict[str, Chunk] = {}
    wanted = set(sources)
    for chunk in read(security.scope_for(access), sources=sources, include_text=False):
        if chunk.source in wanted and security.allows_stored_chunk(chunk, access):
            prior = first.get(chunk.source)
            if prior is None or (metadata_ordinal(chunk.metadata) or 0, chunk.id) < (
                metadata_ordinal(prior.metadata) or 0, prior.id
            ):
                first[chunk.source] = chunk
    eligible = []
    for source in sources:
        if source not in first:
            continue
        chunk = hydrate(first[source].id)
        if chunk is not None and chunk.source == source and security.allows_stored_chunk(chunk, access):
            eligible.append(ScoredChunk(chunk, 0.0, score_kind="structural"))
        if len(eligible) == 4:
            break
    if not eligible:
        return [], 0

    instant = datetime.now(UTC)
    dependency_mode = dependency_mode_from_env(env)
    projection = None
    if dependency_mode == "enforce":
        reader = getattr(store, "dependency_projection", None)
        if not callable(reader):
            return [], 0
        projection = reader(as_of=instant, known_as_of=None)
        if projection is None:
            return [], 0
    trusted = evaluate(
        RetrievalResult(result.query, eligible, False, result.staleness),
        store_supersession_graph(store), None, instant, dependency_projection=projection,
        dependency_mode=dependency_mode, record_metrics=False,
    )
    targets = [hit for hit in trusted.hits if hit.verdict == "ok"]
    if not targets:
        return [], 0
    first_seed = parents[targets[0].chunk.source]
    docs = [first_seed.text[:1000]]
    for hit in targets:
        parent = parents[hit.chunk.source]
        title = " ".join(TITLE_NOISE.sub(" ", parent.text.partition("\n")[0]).split())[:160]
        text = hit.chunk.text[:1000]
        docs.append(text + "\nLinked from retrieved issue: " + title if title else text)
    try:
        order, tokens = rank_jira_documents(result.query, docs, api_key)
    except (ImportError, ValueError, TypeError, KeyError, AttributeError):
        return [], 0
    return [targets[i - 1] for i in order if i], tokens
