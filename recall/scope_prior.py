"""A bounded folder-affinity prior: the folder reorders results, and never removes any.

The idea a folder layer invites is routing: score the query against each folder, pick the best
one, search only there. This module deliberately does not do that, for two reasons that are worth
stating where the code is rather than in a design document nobody re-reads.

⛔ **Routing is unrecoverable and silent.** If the query routes to `python/` and the answer is in
`infra/`, the true chunk is not ranked badly, it is absent, and the result is indistinguishable
from a corpus that simply has no answer. A prior that only reorders keeps every candidate, so a
wrong folder guess costs rank rather than the answer.

⛔ **Routing moves the distribution the trust gate was calibrated on.** The certified threshold is
fitted over an unfiltered candidate pool. Pruning that pool changes the score distribution
underneath a gate that has no way to notice, which turns a calibration into a number that merely
looks calibrated. This module never touches `ScoredChunk.score` for the same reason: `recall.trust`
reads that field and compares it against the certified threshold, so a blended score there would
be a silently miscalibrated one. The blend lives in the SORT KEY and nowhere else.

What is left is modest by construction, which is the honest description: the prior can promote a
candidate the fusion ranked just below the cut, and cannot rescue one that was never retrieved.
Whether that is worth anything on a real corpus is an empirical question, pre-registered in
`docs/preregistrations/2026-08-28-folder-scope-and-prior.md` and unmeasured at the time of
writing. It is therefore OFF by default, and `ScopePrior.weight` must be set deliberately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from recall.scope import FACET_METADATA_KEY, folder_of
from recall.types import ScoredChunk

#: Above this, the prior can reorder a candidate past several ranks of genuine cosine evidence.
#: The cap exists because a weight is the kind of knob that gets raised until the metric moves,
#: and a prior strong enough to dominate retrieval is a router wearing a different hat.
MAX_WEIGHT = 0.5


@dataclass(frozen=True)
class ScopePrior:
    """How much the structural dimension is allowed to reorder a result, and along which axis.

    `weight` of 0.0 is off, and is the default everywhere. A disabled prior does not merely
    contribute nothing: `apply_scope_prior` returns the input list unchanged and identical, so a
    corpus with the feature off cannot differ by so much as a tie-break from one built before it
    existed.
    """

    dimension: str = "folder"
    weight: float = 0.0
    #: Folders smaller than this are dropped from the centroid set rather than trusted. A centroid
    #: over three chunks is nearly a single chunk and its cosine says more about that chunk than
    #: about the folder.
    min_chunks: int = 5

    def __post_init__(self) -> None:
        if self.dimension not in ("folder", "facet"):
            raise ValueError(
                f"dimension must be 'folder' or 'facet', got {self.dimension!r}"
            )
        if not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite, got {self.weight!r}")
        if self.weight < 0.0:
            # A negative weight is a request to rank AWAY from the query's own folder. That is
            # not a tuning direction, it is a sign error, and it has a plausible-looking effect
            # on any metric that is noisy enough.
            raise ValueError(f"weight must be >= 0, got {self.weight}")
        if self.weight > MAX_WEIGHT:
            raise ValueError(
                f"weight {self.weight} exceeds MAX_WEIGHT {MAX_WEIGHT}: above this the prior "
                f"outranks the retrieval evidence rather than tilting it, which is routing"
            )
        if self.min_chunks < 1:
            raise ValueError(f"min_chunks must be >= 1, got {self.min_chunks}")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


def scope_value_of(hit: ScoredChunk, dimension: str) -> str | None:
    """The chunk's value along `dimension`, or None when it has none.

    A chunk with no `file` metadata (a legacy row) has no folder, and a chunk whose author
    declared no facet has no facet. Both return None and are left at their retrieved rank rather
    than being assigned a default bucket: inventing a value would let the prior act on a fact the
    corpus does not contain.
    """
    metadata = hit.chunk.metadata or {}
    if dimension == "folder":
        file = metadata.get("file")
        return folder_of(file) if isinstance(file, str) and file else None
    if dimension == "facet":
        facet = metadata.get(FACET_METADATA_KEY)
        return facet.strip().lower() if isinstance(facet, str) and facet.strip() else None
    raise ValueError(f"unknown scope dimension {dimension!r}")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain cosine similarity, 0.0 when either side has no magnitude.

    A zero vector has no direction, so "how aligned is it" has no answer. Returning 0.0 rather
    than raising keeps a degenerate centroid (which `min_chunks` should already have excluded)
    from taking down a query.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors differ in length: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def affinities(
    query_vector: Sequence[float], centroids: Sequence[tuple[str, int, Sequence[float]]]
) -> dict[str, float]:
    """``{scope value: affinity in [0, 1]}`` for one query.

    The raw cosine is min-maxed across the folders present, so the prior expresses "which of THESE
    folders is this question about" rather than an absolute number. That matters because dense
    cosines over a single corpus occupy a narrow band: raw, the spread between the best and worst
    folder can be a few hundredths, and a weight large enough to matter against it would be large
    enough to dominate. Normalizing puts the decision in the weight rather than in the accident of
    how tightly this particular embedder packs its space.

    With fewer than two folders there is nothing to rank, so every affinity is 0.0 and the prior
    is inert. That is the right answer for a flat corpus, and it is why enabling the prior on one
    changes nothing rather than quietly ranking by folder size.
    """
    if len(centroids) < 2:
        return {value: 0.0 for value, _n, _vec in centroids}
    raw = {value: cosine(query_vector, vec) for value, _n, vec in centroids}
    low, high = min(raw.values()), max(raw.values())
    if high - low <= 0.0:
        return {value: 0.0 for value in raw}
    return {value: (score - low) / (high - low) for value, score in raw.items()}


def apply_scope_prior(
    hits: Sequence[ScoredChunk],
    affinity_by_value: Mapping[str, float],
    prior: ScopePrior,
) -> list[ScoredChunk]:
    """Reorder `hits` by ``rank_score + weight * affinity``, returning the same hits.

    **Same hits, same scores, different order.** Nothing is dropped, nothing is added, and no
    `ScoredChunk` is rebuilt, so the object the trust layer inspects is the object retrieval
    produced.

    The base is the hit's RANK in the incoming order, mapped to ``1.0`` for the first and ``0.0``
    for the last, rather than its cosine. Two reasons. A rank composes with a reranker, whose
    verdict is expressed as an order and not as a score this function can see, so blending against
    cosine would quietly overrule the reranker with the very number the reranker was brought in to
    improve on. And a rank has a known scale, where a cosine's usable range varies by embedder, so
    one weight means the same thing across corpora.

    The consequence is worth being explicit about: with `weight` w, a candidate can climb at most
    ``w * (len(hits) - 1)`` positions, and only past candidates whose affinity is lower. At a
    realistic 0.05 over a 100-candidate pool that is about five places. A prior that cannot move
    something five places cannot rescue it either, which is the honest ceiling on this feature.

    **The key carries affinity as a tie-break, and it has to.** Rank scores are evenly spaced, so
    a bonus of ``w * affinity`` lands EXACTLY level with the candidate ``w * (last)`` positions
    above rather than above it, and a stable sort then keeps the incumbent. Without the tie-break
    the prior would silently deliver one position less than its own documented bound at every
    weight, which is the kind of off-by-one that reads as "the prior does not work" in a
    measurement rather than as a bug. Affinity breaks the tie because the tie is a tie in rank and
    affinity is the only new evidence available to settle it.

    Genuine ties, where two hits carry the same affinity, keep their incoming order: `sorted` is
    stable and the key is the only thing consulted. So an inert prior (`weight` 0, or a corpus
    with a single folder) is an exact identity.
    """
    if not prior.enabled or len(hits) < 2:
        return list(hits)
    last = len(hits) - 1

    def key(indexed: tuple[int, ScoredChunk]) -> tuple[float, float]:
        index, hit = indexed
        base = (last - index) / last
        value = scope_value_of(hit, prior.dimension)
        # A hit with no value along this dimension gets no bonus, rather than the mean or the
        # best. It keeps its retrieved rank, which is the only claim the corpus supports for it.
        affinity = affinity_by_value.get(value, 0.0) if value is not None else 0.0
        return base + prior.weight * affinity, affinity

    ordered = sorted(enumerate(hits), key=key, reverse=True)
    return [hit for _index, hit in ordered]


__all__ = [
    "MAX_WEIGHT",
    "ScopePrior",
    "affinities",
    "apply_scope_prior",
    "cosine",
    "scope_value_of",
]
