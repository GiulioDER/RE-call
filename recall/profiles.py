"""Process-level retrieval profiles and bounded admission control."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Literal

RetrievalProfileName = Literal["legacy", "fast", "quality"]

#: The legacy profile's budget: a sentinel meaning "no budget is enforced", not a 24-day limit.
#: Kept as a large int rather than `None` so the timeout arithmetic has no special case, but it
#: must never reach a client as a number — see `RetrievalProfile.enforced_budget_ms`.
NO_BUDGET_SENTINEL_MS = 0x7FFFFFFF

#: Ceiling on `max_concurrency + queue_capacity`, refused at resolution.
#:
#: This exists because the MCP server sizes its worker pool FROM the profile. Before it did,
#: anyio's 40-token default was an accidental but real ceiling on how many OS threads the search
#: path could occupy; sizing the pool from the profile removes that ceiling and would otherwise
#: turn `RECALL_SEARCH_QUEUE` into an unvalidated thread-count knob. 256 is far above any
#: plausible per-process retrieval concurrency and far below a number that would exhaust a host.
MAX_ADMISSION_CAPACITY = 256


@dataclass(frozen=True)
class RetrievalProfile:
    """One fixed process cost profile.

    ``latency_budget_ms`` is not decoration and not a promotion-gate input. At request time it
    means exactly two things, both observable:

    1. **It bounds the admission wait.** A request that cannot acquire a running slot within the
       budget is shed with :class:`RetrievalOverloaded` *before anything is embedded*. Without
       this, ``queue_capacity`` bounds how many threads may be parked and says nothing about how
       long any of them waits, so a process could hold a client for minutes behind a slow
       reranked query while every counter looked healthy.
    2. **It labels an overrun.** A request that completes over budget is reported as such on the
       result surface and counted in ``METRICS``; see ``recall_mcp.service.search_memory``.

    What it deliberately does **not** do is abort a request in flight. There is no cancellation
    point inside a blocking cross-encoder ``predict``, so a mid-flight abort would pay the whole
    cost and then discard the answer: strictly worse than returning a slow one, and it would turn
    a latency regression into an availability incident. Shedding happens at the door, where it is
    free.
    """

    name: RetrievalProfileName
    candidate_k: int
    returned_k: int
    reranker: bool
    latency_budget_ms: int
    max_concurrency: int = 4
    queue_capacity: int = 16
    inference_threads: int | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_k, "candidate_k"),
            (self.returned_k, "returned_k"),
            (self.latency_budget_ms, "latency_budget_ms"),
            (self.max_concurrency, "max_concurrency"),
            (self.queue_capacity, "queue_capacity"),
        ):
            if value < 1:
                raise ValueError(f"{label} must be positive")

    @property
    def enforced_budget_ms(self) -> int | None:
        """The budget as a client should read it: `None` when no budget is enforced.

        The legacy profile carries a sentinel so the admission timeout needs no special case.
        Publishing that sentinel as a number invites a reader to do arithmetic on it, exactly as
        `RetrievalOverloaded` already refuses to hand it back as a retry hint.
        """
        return None if self.latency_budget_ms >= NO_BUDGET_SENTINEL_MS else self.latency_budget_ms

    @property
    def queue_identity(self) -> tuple[str, int, int, int]:
        """The fields that define an admission queue, and only those.

        Memoising a queue on the WHOLE profile made `inference_threads` part of its identity,
        which has nothing to do with admission: the exported `QUALITY_PROFILE` left it `None`
        while the resolver filled in `1`, so the two were unequal and each got its own pair of
        semaphores. The process-wide concurrency bound then quietly doubled, on the one profile
        where a surplus concurrent request costs the most.
        """
        return (self.name, self.max_concurrency, self.queue_capacity, self.latency_budget_ms)


#: Fast and quality carry SEPARATE concurrency budgets rather than one shared default.
#:
#: The two profiles run the same candidate pool by design (`docs/ENTERPRISE_RETRIEVAL.md`), so the
#: only cost difference is the reranker — and it is a large one. Giving both the same queue depth
#: would let quality make its clients wait roughly six times as long, because each parked request
#: may hold its slot for up to six times the budget. The numbers below hold
#: `queue_capacity * latency_budget_ms` within the same order of magnitude (fast 32 x 250 = 8000
#: slot-milliseconds, quality 8 x 1500 = 12000), which is the wait a saturated process actually
#: owes its clients. Slot-milliseconds, not CPU-seconds: `latency_budget_ms` bounds the WAIT, it
#: is not a service time, so nothing here is a claim about CPU consumed.
#:
#: ⚠️ These are a POLICY choice, not a measurement. Latency on this program is PENDING for want of
#: a 16-vCPU idle reference host, so no number here is claimed to be tuned to measured throughput.
#: `RECALL_SEARCH_CONCURRENCY` / `RECALL_SEARCH_QUEUE` override them per deployment.
FAST_PROFILE = RetrievalProfile("fast", 20, 5, False, 250, max_concurrency=8, queue_capacity=32)
#: `inference_threads=1` here, not left at the dataclass default, so the exported constant EQUALS
#: what `resolve_retrieval_profile` produces for an unconfigured quality process. A constant that
#: disagrees with its own resolver is a trap for anything that keys on the profile.
QUALITY_PROFILE = RetrievalProfile(
    "quality", 20, 5, True, 1500, max_concurrency=2, queue_capacity=8, inference_threads=1
)
#: Legacy keeps 4/16 and an effectively infinite budget: it is the pre-profile behaviour, and a
#: deployment that never opted into a profile must not acquire shedding it did not ask for.
LEGACY_PROFILE = RetrievalProfile("legacy", 20, 5, False, NO_BUDGET_SENTINEL_MS)


def resolve_retrieval_profile(env: dict[str, str] | None = None) -> RetrievalProfile:
    """Resolve a fixed process profile while preserving the legacy rerank switch."""
    values = os.environ if env is None else env
    selected = values.get("RECALL_RETRIEVAL_PROFILE", "").strip().lower()
    legacy_rerank = values.get("RECALL_RERANK", "").strip().lower()
    if selected not in {"", "fast", "quality"}:
        raise ValueError("RECALL_RETRIEVAL_PROFILE must be 'fast' or 'quality'")
    if selected and legacy_rerank:
        enabled = legacy_rerank in {"1", "true", "yes", "on"}
        if enabled != (selected == "quality"):
            raise ValueError(
                "RECALL_RETRIEVAL_PROFILE conflicts with the legacy RECALL_RERANK setting"
            )
    base = QUALITY_PROFILE if selected == "quality" else FAST_PROFILE if selected == "fast" else LEGACY_PROFILE

    def _positive(name: str, default: int) -> int:
        raw = values.get(name)
        # Empty means UNSET, not malformed. `.env.example` ships these keys with no value, and a
        # dotenv load puts an empty string in the environment rather than leaving the key out, so
        # reading it as a bad integer would make the shipped example refuse startup.
        if raw is None or not raw.strip():
            return default
        try:
            parsed = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
        if parsed < 1:
            raise ValueError(f"{name} must be positive")
        return parsed

    max_concurrency = _positive("RECALL_SEARCH_CONCURRENCY", base.max_concurrency)
    queue_capacity = _positive("RECALL_SEARCH_QUEUE", base.queue_capacity)
    if max_concurrency + queue_capacity > MAX_ADMISSION_CAPACITY:
        raise ValueError(
            f"admission capacity {max_concurrency + queue_capacity} exceeds the "
            f"{MAX_ADMISSION_CAPACITY} limit. The worker pool is sized from this number, so an "
            f"unbounded queue would be an unbounded thread count."
        )
    return RetrievalProfile(
        name=base.name,
        candidate_k=base.candidate_k,
        returned_k=base.returned_k,
        reranker=base.reranker,
        latency_budget_ms=base.latency_budget_ms,
        max_concurrency=max_concurrency,
        queue_capacity=queue_capacity,
        inference_threads=(
            _positive("RECALL_RERANK_THREADS", 1) if base.reranker else None
        ),
    )


class RetrievalOverloaded(RuntimeError):
    """The process has no safe capacity to begin another retrieval.

    ``reason`` is a stable machine-readable slug, not prose:

    * ``queue_full`` — every queue slot is taken; the request was refused without waiting.
    * ``budget_exhausted`` — the request queued, but could not start within
      ``latency_budget_ms``, so it was shed rather than served late.

    Both are RETRYABLE: nothing was consumed and no state changed. The message deliberately names
    only the profile and the numbers involved. It never carries the query, because a rejection is
    exactly the moment a message is most likely to be logged verbatim by a caller.
    """

    #: Ceiling on the retry hint, in seconds. The legacy profile's budget is a sentinel roughly
    #: 24 days long; handing that to a client as "retry after" would be worse than silence.
    MAX_RETRY_AFTER_SECONDS = 5.0

    def __init__(self, message: str, *, reason: str, retry_after_seconds: float) -> None:
        super().__init__(message)
        self.reason = reason
        #: Mirrors `recall_mcp.limits.RateLimited`: a structured, machine-readable field on the
        #: exception IS this codebase's retryable status. Nothing was consumed and no state
        #: changed, so the only question a client has is when to come back.
        self.retry_after_seconds = retry_after_seconds


class RetrievalAdmission:
    """A bounded process-local queue acquired before query embedding begins.

    Two semaphores, not one. ``_slots`` bounds how many requests may be *in the process at all*
    (running plus parked) and is taken non-blocking, so a saturated process refuses instantly
    instead of growing an unbounded pile of waiters. ``_running`` bounds how many may execute
    concurrently and is taken with the profile's latency budget as its timeout, so a parked
    request is shed rather than served arbitrarily late.
    """

    def __init__(self, profile: RetrievalProfile) -> None:
        self.profile = profile
        self._slots = threading.BoundedSemaphore(
            profile.max_concurrency + profile.queue_capacity
        )
        self._running = threading.BoundedSemaphore(profile.max_concurrency)

    @property
    def _retry_after_seconds(self) -> float:
        return min(
            self.profile.latency_budget_ms / 1000.0,
            RetrievalOverloaded.MAX_RETRY_AFTER_SECONDS,
        )

    def __enter__(self) -> None:
        if not self._slots.acquire(blocking=False):
            raise RetrievalOverloaded(
                f"retrieval queue is full for profile {self.profile.name!r} "
                f"({self.profile.max_concurrency} running + "
                f"{self.profile.queue_capacity} queued)",
                reason="queue_full",
                retry_after_seconds=self._retry_after_seconds,
            )
        # The queue slot is held from here on, so every failure path below MUST return it.
        # `__exit__` does not run when `__enter__` raises, which is why the release is explicit:
        # a leaked permit is permanent, and once all `max_concurrency + queue_capacity` of them
        # have leaked the process refuses every request forever while reporting itself as merely
        # busy. (Partial leakage degrades proportionally: after `queue_capacity` leaks the process
        # still serves `max_concurrency` concurrent requests, with no queue depth left.)
        #
        # `admitted` is bound BEFORE the acquire so the handler can tell "never acquired" from
        # "acquired", and releases the running permit in the second case.
        #
        # ⚠️ Honest limit: this does NOT close the interrupt window, and a later reader should not
        # believe it does. CPython's exception table for this block puts the only interruptible
        # point at the acquire call itself, where `admitted` is still False, so the branch below
        # cannot currently fire. The real window is INSIDE `Semaphore.acquire`, after it
        # decrements its counter and before it returns, which is not reachable from here. The
        # guard is kept because it is correct and costs nothing if CPython's codegen changes; it
        # is not a fix, and the residual leak is recorded in ENTERPRISE_PROGRAM_STATUS.md.
        admitted = False
        try:
            admitted = self._running.acquire(
                timeout=self.profile.latency_budget_ms / 1000.0
            )
        except BaseException:
            if admitted:
                self._running.release()
            self._slots.release()
            raise
        if not admitted:
            self._slots.release()
            raise RetrievalOverloaded(
                f"waited longer than the {self.profile.latency_budget_ms} ms budget of profile "
                f"{self.profile.name!r} for a running slot",
                reason="budget_exhausted",
                retry_after_seconds=self._retry_after_seconds,
            )

    def __exit__(self, *exc: object) -> None:
        self._running.release()
        self._slots.release()
