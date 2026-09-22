"""Bounded rank based retrieval across physically isolated specialist tenants.

Federation is a logical request plan. Each leg remains responsible for its own store, embedding
profile, generation, calibration, and trust evaluation. This module only combines independently
ranked, trusted candidates; it never compares cosine values from different tenants.

The route planner is intentionally an adapter supplied by the caller. Workstream 1 can provide
that adapter without moving routing policy into this module or changing the single tenant control
path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field, replace
import hashlib
import os
import time
from typing import Literal

from recall.errors import RecallError
from recall.observability import METRICS
from recall.trust import is_trusted
from recall.types import TrustedHit, TrustedResult


FederationMode = Literal["off", "shadow", "active"]
InvalidLegPolicy = Literal["omit", "fail_closed"]
RetrieveLeg = Callable[[str, int], TrustedResult]

MAX_FEDERATION_LEGS = 8
MAX_FEDERATION_CONCURRENCY = 8
MAX_FEDERATION_CANDIDATE_K = 50
MAX_FEDERATION_RESULT_K = 50
MAX_FEDERATION_RRF_K = 1_000_000
MAX_FEDERATION_LATENCY_BUDGET_MS = 120_000


class FederationConfigurationError(ValueError, RecallError):
    """Raised when a federation request would exceed a configured safety bound."""


class FederationLegRejected(RuntimeError, RecallError):
    """Raised when the explicit fail closed policy rejects a tenant leg."""


@dataclass(frozen=True)
class FederationConfig:
    """Bound the work and define how a federation result may be served.

    ``off`` executes only the first leg, preserving the single tenant control path. ``shadow``
    executes the bounded plan and returns diagnostics while callers continue serving control
    results. ``active`` permits callers to serve the merged candidates.
    """

    mode: FederationMode = "off"
    max_legs: int = 2
    max_concurrency: int = 2
    candidate_k: int = 20
    result_k: int = 5
    primary_prefix: int = 3
    rescue_slots: int = 1
    rrf_k: int = 60
    invalid_leg_policy: InvalidLegPolicy = "omit"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "active"}:
            raise FederationConfigurationError("mode must be off, shadow, or active")
        for name in (
            "max_legs",
            "max_concurrency",
            "candidate_k",
            "result_k",
            "primary_prefix",
        ):
            if getattr(self, name) < 1:
                raise FederationConfigurationError(f"{name} must be >= 1")
        if self.max_concurrency > self.max_legs:
            raise FederationConfigurationError("max_concurrency must be <= max_legs")
        for name, maximum in (
            ("max_legs", MAX_FEDERATION_LEGS),
            ("max_concurrency", MAX_FEDERATION_CONCURRENCY),
            ("candidate_k", MAX_FEDERATION_CANDIDATE_K),
            ("result_k", MAX_FEDERATION_RESULT_K),
            ("primary_prefix", MAX_FEDERATION_RESULT_K),
            ("rrf_k", MAX_FEDERATION_RRF_K),
        ):
            if getattr(self, name) > maximum:
                raise FederationConfigurationError(f"{name} must be <= {maximum}")
        if self.candidate_k > MAX_FEDERATION_CANDIDATE_K:
            raise FederationConfigurationError(
                f"candidate_k must be <= {MAX_FEDERATION_CANDIDATE_K}"
            )
        if self.rrf_k < 0:
            raise FederationConfigurationError("rrf_k must be >= 0")
        if self.rescue_slots < 0:
            raise FederationConfigurationError("rescue_slots must be >= 0")
        if self.rescue_slots > self.result_k:
            raise FederationConfigurationError("rescue_slots must be <= result_k")
        if self.primary_prefix + self.rescue_slots > self.result_k:
            raise FederationConfigurationError(
                "primary_prefix plus rescue_slots must be <= result_k"
            )
        if self.invalid_leg_policy not in {"omit", "fail_closed"}:
            raise FederationConfigurationError(
                "invalid_leg_policy must be omit or fail_closed"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "FederationConfig":
        """Resolve the opt in configuration without selecting tenants.

        Tenant selection remains the route planner's responsibility. Invalid values refuse
        startup or request construction instead of silently widening the fanout.
        """

        values = os.environ if env is None else env

        def integer(name: str, default: int, *, minimum: int = 0) -> int:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise FederationConfigurationError(f"{name} must be an integer") from exc
            if value < minimum:
                raise FederationConfigurationError(f"{name} must be >= {minimum}")
            return value

        mode = values.get("RECALL_FEDERATION_MODE", "off").strip().lower()
        return cls(
            mode=mode,  # type: ignore[arg-type]
            max_legs=integer("RECALL_FEDERATION_MAX_LEGS", 2, minimum=1),
            max_concurrency=integer("RECALL_FEDERATION_MAX_CONCURRENCY", 2, minimum=1),
            candidate_k=integer("RECALL_FEDERATION_CANDIDATE_K", 20, minimum=1),
            result_k=integer("RECALL_FEDERATION_RESULT_K", 5, minimum=1),
            primary_prefix=integer("RECALL_FEDERATION_PRIMARY_PREFIX", 3, minimum=1),
            rescue_slots=integer("RECALL_FEDERATION_RESCUE_SLOTS", 1),
            rrf_k=integer("RECALL_FEDERATION_RRF_K", 60),
            invalid_leg_policy=values.get("RECALL_FEDERATION_INVALID_LEG", "omit").strip().lower(),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class FederationLeg:
    """One independently routed tenant retrieval operation.

    The first leg supplied to :func:`federate` is the primary control leg. The callback must
    return a ``TrustedResult`` produced by that tenant's own calibrated trust path.
    """

    tenant_id: str
    generation_id: str
    embedding_profile: str
    calibration_id: str | None
    calibration_status: str
    route: str
    retrieve: RetrieveLeg
    candidate_k: int | None = None
    pipeline_fingerprint: str | None = None
    corpus_fingerprint: str | None = None
    query_set_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "generation_id",
            "embedding_profile",
            "calibration_status",
            "route",
        ):
            if not getattr(self, name).strip():
                raise FederationConfigurationError(f"{name} must be non empty")
        if not callable(self.retrieve):
            raise FederationConfigurationError("retrieve must be callable")
        if self.candidate_k is not None and not 1 <= self.candidate_k <= MAX_FEDERATION_CANDIDATE_K:
            raise FederationConfigurationError(
                f"candidate_k must be between 1 and {MAX_FEDERATION_CANDIDATE_K}"
            )


@dataclass(frozen=True)
class FederatedCandidate:
    """A merged item with its complete tenant and rank lineage."""

    hit: TrustedHit
    tenant_id: str
    generation_id: str
    embedding_profile: str
    calibration_id: str
    calibration_status: str
    route: str
    leg_rank: int
    fused_rank: int
    fused_score: float
    pipeline_fingerprint: str | None = None
    corpus_fingerprint: str | None = None
    query_set_digest: str | None = None

    def lineage(self) -> dict[str, object]:
        """Return safe additive metadata for API and diagnostic adapters."""

        return {
            "tenant": self.tenant_id,
            "generation": self.generation_id,
            "embedding_profile": self.embedding_profile,
            "calibration": self.calibration_id,
            "calibration_status": self.calibration_status,
            "route": self.route,
            "leg_rank": self.leg_rank,
            "fused_rank": self.fused_rank,
            "fused_score": self.fused_score,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "query_set_digest": self.query_set_digest,
        }


@dataclass(frozen=True)
class FederationLegDiagnostics:
    tenant_id: str
    route: str
    embedding_profile: str
    generation_id: str
    candidate_count: int
    trusted_candidate_count: int
    rejected_candidate_count: int
    latency_ms: float
    accepted: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class FederationDiagnostics:
    mode: FederationMode
    fanout_count: int
    executed_leg_count: int
    candidate_count: int
    fused_candidate_count: int
    primary_prefix_count: int
    rescue_count: int
    rejected_leg_count: int
    rejected_candidate_count: int
    latency_ms: float
    max_concurrency: int
    legs: tuple[FederationLegDiagnostics, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FederationResult:
    """The bounded merge and its operator facing diagnostics.

    ``served`` is the control safe projection. A caller must check ``active`` before replacing
    its control result with it. In shadow mode the merge is computed, but ``active`` is false.
    """

    candidates: tuple[FederatedCandidate, ...]
    served: tuple[FederatedCandidate, ...]
    diagnostics: FederationDiagnostics
    active: bool


def _rrf(
    rankings: Sequence[Sequence[tuple[str, str]]], constant: int
) -> dict[tuple[str, str], float]:
    """Fuse independent best first ID rankings, never candidate scores."""

    scores: dict[tuple[str, str], float] = {}
    for ranking in rankings:
        for rank, candidate_id in enumerate(ranking, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (constant + rank)
    return scores


def _candidate_keys(hit: TrustedHit) -> tuple[str, tuple[str, str]]:
    """Return an ID key and a content key for cross tenant novelty checks."""

    digest = hashlib.sha256(hit.chunk.text.encode("utf-8")).hexdigest()
    return hit.chunk.id, (hit.chunk.source, digest)


def _leg_rejection(leg: FederationLeg, result: TrustedResult) -> str | None:
    """Check the result of one leg before it can contribute to the logical corpus."""

    if result.tenant_id != leg.tenant_id:
        return "tenant_lineage_mismatch"
    if result.generation_id != leg.generation_id:
        return "generation_lineage_mismatch"
    profile = result.diagnostics.embedding_profile
    if profile != leg.embedding_profile:
        return "embedding_lineage_mismatch"
    if result.trust_state != "trusted":
        return "trust_not_certified"
    if not result.calibrated:
        return "calibration_not_certified"
    if result.calibration_id != leg.calibration_id:
        return "calibration_lineage_mismatch"
    if result.calibration_status != leg.calibration_status:
        return "calibration_status_mismatch"
    if result.gap_warning:
        return "tenant_gap_warning"
    return None


def _run_leg(leg: FederationLeg, query: str, candidate_k: int) -> tuple[TrustedResult, float]:
    started = time.perf_counter()
    result = leg.retrieve(query, leg.candidate_k or candidate_k)
    return result, (time.perf_counter() - started) * 1000.0


def federate(
    query: str,
    legs: Sequence[FederationLeg],
    config: FederationConfig | None = None,
    *,
    latency_budget_ms: int | None = None,
) -> FederationResult:
    """Retrieve independently in bounded legs and merge by rank.

    The first leg is primary. It contributes the protected prefix and the control safe remainder.
    Secondary legs may contribute only independently trusted, calibrated, non gap, novel hits to
    the bounded rescue tail. A leg is never scored against another leg's cosine space.
    """

    selected = tuple(legs)
    settings = config or FederationConfig()
    if latency_budget_ms is not None and not 1 <= latency_budget_ms <= MAX_FEDERATION_LATENCY_BUDGET_MS:
        raise FederationConfigurationError(
            f"latency_budget_ms must be between 1 and {MAX_FEDERATION_LATENCY_BUDGET_MS}"
        )
    if not selected:
        raise FederationConfigurationError("at least one federation leg is required")
    if len(selected) > settings.max_legs:
        raise FederationConfigurationError(
            f"federation selected {len(selected)} legs, over max_legs={settings.max_legs}"
        )
    effective_legs = selected if settings.mode != "off" else selected[:1]
    results: dict[int, tuple[TrustedResult, float]] = {}
    timed_out: set[int] = set()
    started = time.perf_counter()
    if len(effective_legs) == 1 and latency_budget_ms is not None:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_leg, effective_legs[0], query, settings.candidate_k)
        try:
            done, pending = wait((future,), timeout=latency_budget_ms / 1000.0)
            if future in done:
                try:
                    results[0] = future.result()
                except Exception as exc:  # BROAD-CATCH: fail-open
                    if settings.invalid_leg_policy == "fail_closed":
                        raise FederationLegRejected(
                            f"tenant leg {effective_legs[0].tenant_id!r} failed: "
                            f"{type(exc).__name__}"
                        ) from exc
            else:
                timed_out = {0}
                future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    elif len(effective_legs) == 1:
        try:
            results[0] = _run_leg(effective_legs[0], query, settings.candidate_k)
        except Exception as exc:  # BROAD-CATCH: fail-open
            if settings.invalid_leg_policy == "fail_closed":
                raise FederationLegRejected(
                    f"tenant leg {effective_legs[0].tenant_id!r} failed: {type(exc).__name__}"
                ) from exc
    else:
        executor = ThreadPoolExecutor(max_workers=settings.max_concurrency)
        futures = {
            executor.submit(_run_leg, leg, query, settings.candidate_k): index
            for index, leg in enumerate(effective_legs)
        }
        try:
            if latency_budget_ms is None:
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:  # BROAD-CATCH: fail-open
                        if settings.invalid_leg_policy == "fail_closed":
                            raise FederationLegRejected(
                                f"tenant leg {effective_legs[index].tenant_id!r} failed: "
                                f"{type(exc).__name__}"
                            ) from exc
            else:
                done, pending = wait(futures, timeout=latency_budget_ms / 1000.0)
                timed_out = {futures[future] for future in pending}
                for future in done:
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:  # BROAD-CATCH: fail-open
                        if settings.invalid_leg_policy == "fail_closed":
                            raise FederationLegRejected(
                                f"tenant leg {effective_legs[index].tenant_id!r} failed: "
                                f"{type(exc).__name__}"
                            ) from exc
        finally:
            executor.shutdown(wait=latency_budget_ms is None, cancel_futures=True)

    leg_diagnostics: list[FederationLegDiagnostics] = []
    valid_hits: dict[int, list[TrustedHit]] = {}
    rejected_legs = 0
    rejected_candidates = 0
    for index, leg in enumerate(effective_legs):
        result_and_latency = results.get(index)
        if result_and_latency is None:
            rejected_legs += 1
            if settings.invalid_leg_policy == "fail_closed":
                timeout_reason = "latency_budget_exceeded" if index in timed_out else "leg_failed"
                raise FederationLegRejected(
                    f"tenant leg {leg.tenant_id!r} rejected: {timeout_reason}"
                )
            leg_diagnostics.append(
                FederationLegDiagnostics(
                    tenant_id=leg.tenant_id,
                    route=leg.route,
                    embedding_profile=leg.embedding_profile,
                    generation_id=leg.generation_id,
                    candidate_count=0,
                    trusted_candidate_count=0,
                    rejected_candidate_count=0,
                    latency_ms=0.0,
                    accepted=False,
                    rejection_reason=(
                        "latency_budget_exceeded" if index in timed_out else "leg_failed"
                    ),
                )
            )
            continue
        result, latency = result_and_latency
        request_k = leg.candidate_k or settings.candidate_k
        if len(result.hits) > request_k:
            result = replace(result, hits=result.hits[:request_k])
        reason = _leg_rejection(leg, result)
        trusted_hits = [hit for hit in result.hits if is_trusted(hit)]
        rejected_count = len(result.hits) - len(trusted_hits)
        if reason is not None:
            rejected_legs += 1
            rejected_count = len(result.hits)
            trusted_hits = []
        else:
            valid_hits[index] = trusted_hits
        rejected_candidates += rejected_count
        leg_diagnostics.append(
            FederationLegDiagnostics(
                tenant_id=leg.tenant_id,
                route=leg.route,
                embedding_profile=leg.embedding_profile,
                generation_id=leg.generation_id,
                candidate_count=len(result.hits),
                trusted_candidate_count=len(trusted_hits),
                rejected_candidate_count=rejected_count,
                latency_ms=round(latency, 3),
                accepted=reason is None,
                rejection_reason=reason,
            )
        )
        if settings.invalid_leg_policy == "fail_closed" and reason is not None:
            raise FederationLegRejected(
                f"tenant leg {leg.tenant_id!r} rejected: {reason}"
            )

    primary_hits = valid_hits.get(0, [])
    primary_content = {_candidate_keys(hit)[1] for hit in primary_hits}
    all_rankings: list[list[tuple[str, str]]] = []
    for index in sorted(valid_hits):
        tenant_id = effective_legs[index].tenant_id
        all_rankings.append([(tenant_id, hit.chunk.id) for hit in valid_hits[index]])
    fused_scores = _rrf(all_rankings, settings.rrf_k)
    fused_order = sorted(
        fused_scores,
        key=lambda candidate_id: (-fused_scores[candidate_id], candidate_id),
    )
    fused_rank = {candidate_id: rank for rank, candidate_id in enumerate(fused_order, start=1)}

    by_key: dict[tuple[str, tuple[str, str]], FederatedCandidate] = {}
    primary_by_key: dict[tuple[str, tuple[str, str]], FederatedCandidate] = {}
    secondary_candidates: list[FederatedCandidate] = []
    for index in sorted(valid_hits):
        leg = effective_legs[index]
        for leg_rank, hit in enumerate(valid_hits[index], start=1):
            id_key, content_key = _candidate_keys(hit)
            candidate = FederatedCandidate(
                hit=hit,
                tenant_id=leg.tenant_id,
                generation_id=leg.generation_id,
                embedding_profile=leg.embedding_profile,
                calibration_id=leg.calibration_id or "",
                calibration_status=leg.calibration_status,
                route=leg.route,
                leg_rank=leg_rank,
                fused_rank=fused_rank.get(
                    (leg.tenant_id, id_key), len(fused_order) + leg_rank
                ),
                fused_score=fused_scores.get((leg.tenant_id, id_key), 0.0),
                pipeline_fingerprint=leg.pipeline_fingerprint,
                corpus_fingerprint=leg.corpus_fingerprint,
                query_set_digest=leg.query_set_digest,
            )
            key = (id_key, content_key)
            previous = by_key.get(key)
            if previous is None or (candidate.fused_rank, candidate.tenant_id) < (
                previous.fused_rank,
                previous.tenant_id,
            ):
                by_key[key] = candidate
            if index == 0:
                primary_by_key[key] = candidate
            if (
                index != 0
                and 0 in valid_hits
                and content_key not in primary_content
            ):
                secondary_candidates.append(candidate)

    secondary_candidates = sorted(
        {(
            candidate.hit.chunk.id,
            candidate.hit.chunk.source,
            candidate.hit.chunk.text,
        ): candidate for candidate in secondary_candidates}.values(),
        key=lambda candidate: (
            candidate.fused_rank,
            candidate.leg_rank,
            candidate.tenant_id,
            candidate.hit.chunk.id,
        ),
    )
    primary_candidates = [
        primary_by_key[(_candidate_keys(hit)[0], _candidate_keys(hit)[1])]
        for hit in primary_hits
    ]
    result_limit = settings.result_k
    rescue_limit = min(settings.rescue_slots, result_limit)
    primary_limit = result_limit - rescue_limit
    primary_prefix_count = min(settings.primary_prefix, len(primary_hits), result_limit)
    primary_served = primary_candidates[:primary_limit]
    rescue = secondary_candidates[:rescue_limit]
    # Keep the primary segment contiguous, then append the bounded secondary rescue tail. The
    # prefix is therefore protected even when a caller asks for fewer primary slots than hits.
    served_raw = [*primary_served, *rescue][:result_limit]
    served = tuple(served_raw)
    diagnostics = FederationDiagnostics(
        mode=settings.mode,
        fanout_count=len(effective_legs),
        executed_leg_count=len(results),
        candidate_count=sum(len(hits) for hits in valid_hits.values()),
        fused_candidate_count=len(fused_order),
        primary_prefix_count=primary_prefix_count,
        rescue_count=len(rescue),
        rejected_leg_count=rejected_legs,
        rejected_candidate_count=rejected_candidates,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        max_concurrency=min(settings.max_concurrency, len(effective_legs)),
        legs=tuple(leg_diagnostics),
    )
    METRICS.observe("recall_federation_total_ms", diagnostics.latency_ms, mode=settings.mode)
    METRICS.increment("recall_federation_fanout_total", diagnostics.fanout_count)
    METRICS.increment("recall_federation_rejected_legs_total", diagnostics.rejected_leg_count)
    METRICS.increment(
        "recall_federation_rejected_candidates_total", diagnostics.rejected_candidate_count
    )
    for metric_leg in diagnostics.legs:
        METRICS.observe(
            "recall_federation_leg_ms", metric_leg.latency_ms, tenant=metric_leg.tenant_id
        )
        METRICS.increment(
            "recall_federation_leg_candidates_total",
            metric_leg.candidate_count,
            tenant=metric_leg.tenant_id,
        )
    return FederationResult(
        candidates=tuple(by_key.values()),
        served=served,
        diagnostics=diagnostics,
        active=settings.mode == "active",
    )


__all__ = [
    "FederationConfig",
    "FederationConfigurationError",
    "FederationLeg",
    "FederationLegDiagnostics",
    "FederationLegRejected",
    "FederationResult",
    "FederatedCandidate",
    "federate",
]
