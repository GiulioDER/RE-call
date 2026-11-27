"""The compiled graph at search time: seeds by cosine, bounded promotion, exact order on failure.

``compiled_graph_step`` returns the transform ``HybridRetriever`` runs after fusion and the outcome
note the trust layer stamps on the result's diagnostics. In ``shadow`` the promotion is computed and
reported but the order is returned unchanged; in ``active`` at most two linked raw chunks move into
ranks 9 and 10. Search never calls a model.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from recall.compiled_graph.artifact import (
    ArtifactLineageMismatch,
    ArtifactMissing,
    ArtifactUnconfigured,
    CompiledGraphArtifact,
    CompiledGraphArtifactError,
)
from recall.compiled_graph.promotion import SEED_K, promote
from recall.compiled_graph.service import compiled_graph_mode, load_serving_artifact
from recall.embedding_core import Embedder
from recall.types import GraphTransform, ScoredChunk

_log = logging.getLogger("recall.compiled_graph")


@dataclass
class GraphOutcome:
    """What the step did on this search, as the ``compiled_graph`` diagnostics field reports it."""

    note: str = ""


def _seeds(artifact: CompiledGraphArtifact, query_vector: list[float]) -> list[tuple[str, tuple[str, ...]]]:
    query: np.ndarray = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if not len(artifact.records) or norm == 0.0:
        return []
    negated = -(artifact.vectors @ (query / norm))
    # The top SEED_K in the order a stable full sort gives, without sorting every record: every
    # record tied with the SEED_K-th is kept before the stable sort, so ties break by index.
    cut = negated[np.argpartition(negated, min(SEED_K, len(negated)) - 1)[:SEED_K]].max()
    candidates = np.flatnonzero(negated <= cut)
    order = candidates[np.argsort(negated[candidates], kind="stable")][:SEED_K]
    return [(artifact.records[i].key, artifact.records[i].edges) for i in order]


def compiled_graph_step(
    store: object,
    embedder: Embedder,
    env: Mapping[str, str],
    *,
    scope_is_empty: bool,
) -> tuple[GraphTransform | None, GraphOutcome]:
    """The graph transform for this search, or None with the reason in the outcome."""
    outcome = GraphOutcome()
    tenant = getattr(store, "tenant", None)
    binding_reader = getattr(store, "generation_binding", None)
    # The trust layer serves duck-typed stores too; one without a tenant or binding has no graph.
    if not isinstance(tenant, str) or not callable(binding_reader):
        return None, outcome
    mode = compiled_graph_mode(env, tenant)
    if mode == "off":
        return None, outcome
    if not scope_is_empty:
        # A scope is a hard filter: the graph must not reach outside it.
        outcome.note = "skipped:scoped"
        return None, outcome
    try:
        artifact = load_serving_artifact(tenant, binding_reader(), embedder, env)
    except ArtifactLineageMismatch:
        outcome.note = "skipped:lineage"
        return None, outcome
    except ArtifactMissing:
        outcome.note = "skipped:missing"
        return None, outcome
    except ArtifactUnconfigured:
        outcome.note = "skipped:unconfigured"
        return None, outcome
    except CompiledGraphArtifactError:
        outcome.note = "skipped:damaged"
        return None, outcome

    def transform(query_vector: list[float], ranked: list[ScoredChunk]) -> list[ScoredChunk]:
        try:
            promotion = promote(ranked, _seeds(artifact, query_vector))
        except (ValueError, IndexError) as exc:
            # The search answers exactly as without the graph; the reason is in the log.
            _log.warning("compiled graph step failed for tenant %r: %s", tenant, exc)
            outcome.note = "skipped:error"
            return ranked
        if mode == "shadow":
            outcome.note = f"shadow:would_promote={promotion.promoted_count}"
            return ranked
        outcome.note = f"active:promoted={promotion.promoted_count}"
        return promotion.hits

    return transform, outcome
