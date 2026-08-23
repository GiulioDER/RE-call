"""Profile resolution, bounded admission, and the request-time latency budget.

The behaviours pinned here are the ones an operator is promised by
`docs/ENTERPRISE_RETRIEVAL.md`: a process serves exactly one cost profile, a client cannot buy
its way onto the expensive one, and the process refuses work it cannot afford *before* it spends
anything on it. Each test below was shown red by mutating the module it covers; the mutations are
recorded in `docs/archive/ENTERPRISE_PROGRAM_STATUS.md`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from recall.profiles import (
    CODE_PROFILE,
    FAST_PROFILE,
    LEGACY_PROFILE,
    MAX_ADMISSION_CAPACITY,
    QUALITY_PROFILE,
    RetrievalAdmission,
    RetrievalOverloaded,
    RetrievalProfile,
    resolve_retrieval_profile,
)


# --------------------------------------------------------------------------------------------
# Profile resolution and conflict refusal
# --------------------------------------------------------------------------------------------


def test_fast_quality_and_code_match_the_published_specification() -> None:
    """The two numbers an operator reads off the doc are the two numbers the code uses."""
    assert (FAST_PROFILE.candidate_k, FAST_PROFILE.returned_k) == (20, 5)
    assert FAST_PROFILE.reranker is False
    assert FAST_PROFILE.latency_budget_ms == 250
    assert (QUALITY_PROFILE.candidate_k, QUALITY_PROFILE.returned_k) == (20, 5)
    assert QUALITY_PROFILE.reranker is True
    assert QUALITY_PROFILE.latency_budget_ms == 1500
    assert (CODE_PROFILE.candidate_k, CODE_PROFILE.returned_k) == (20, 5)
    assert CODE_PROFILE.reranker is True
    assert CODE_PROFILE.latency_budget_ms == 10_000
    assert CODE_PROFILE.max_concurrency == 1
    assert CODE_PROFILE.inference_threads == 1


def test_an_unknown_profile_name_is_refused() -> None:
    with pytest.raises(ValueError, match="must be 'fast', 'quality', or 'code'"):
        resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "premium"})


def test_legacy_rerank_switch_survives_only_while_no_profile_is_selected() -> None:
    """`RECALL_RERANK` keeps its old meaning, but only on the path that has no profile."""
    legacy = resolve_retrieval_profile({"RECALL_RERANK": "true"})
    assert legacy.name == "legacy"
    assert resolve_retrieval_profile({}).name == "legacy"


@pytest.mark.parametrize(
    ("profile", "rerank"),
    [
        ("fast", "true"),
        ("fast", "on"),
        ("quality", "false"),
        ("quality", "0"),
        ("code", "false"),
        ("code", "0"),
    ],
)
def test_a_conflicting_profile_and_legacy_switch_refuses(profile: str, rerank: str) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": profile, "RECALL_RERANK": rerank}
        )


@pytest.mark.parametrize(
    ("profile", "rerank"), [("fast", "off"), ("quality", "1"), ("code", "1")]
)
def test_an_agreeing_profile_and_legacy_switch_is_accepted(profile: str, rerank: str) -> None:
    """Agreement is not a conflict. Only a contradiction refuses."""
    resolved = resolve_retrieval_profile(
        {"RECALL_RETRIEVAL_PROFILE": profile, "RECALL_RERANK": rerank}
    )
    assert resolved.name == profile


def test_env_overrides_must_be_positive_integers() -> None:
    with pytest.raises(ValueError, match="RECALL_SEARCH_CONCURRENCY must be an integer"):
        resolve_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": "fast", "RECALL_SEARCH_CONCURRENCY": "many"}
        )
    with pytest.raises(ValueError, match="RECALL_SEARCH_QUEUE must be positive"):
        resolve_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": "fast", "RECALL_SEARCH_QUEUE": "0"}
        )


# --------------------------------------------------------------------------------------------
# Separate concurrency budgets
# --------------------------------------------------------------------------------------------


def test_quality_carries_a_smaller_concurrency_budget_than_fast() -> None:
    """The two profiles do not share an admission budget.

    Quality's per-request budget is six times fast's, so an equal concurrency budget would let it
    park roughly six times the CPU-seconds behind the same queue. The budgets are therefore set
    per profile rather than inherited from one default. The absolute values are a policy choice,
    not a measurement: latency on this program is PENDING for want of a reference host.
    """
    assert QUALITY_PROFILE.max_concurrency < FAST_PROFILE.max_concurrency
    assert QUALITY_PROFILE.queue_capacity < FAST_PROFILE.queue_capacity


def test_resolution_carries_each_profiles_own_budget_not_one_shared_default() -> None:
    fast = resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "fast"})
    quality = resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "quality"})
    assert (fast.max_concurrency, fast.queue_capacity) == (
        FAST_PROFILE.max_concurrency,
        FAST_PROFILE.queue_capacity,
    )
    assert (quality.max_concurrency, quality.queue_capacity) == (
        QUALITY_PROFILE.max_concurrency,
        QUALITY_PROFILE.queue_capacity,
    )
    assert fast.max_concurrency != quality.max_concurrency


def test_an_empty_override_means_unset_not_invalid() -> None:
    """A blank assignment puts an empty STRING in the environment, not nothing.

    `KEY=` in a systemd `EnvironmentFile`, a Kubernetes ConfigMap or a dotenv file all produce an
    empty value rather than omitting the key, so reading it as a malformed integer turns the
    ordinary way of writing "I am not setting this" into a startup refusal. Empty and absent
    must agree.
    """
    empty = resolve_retrieval_profile(
        {
            "RECALL_RETRIEVAL_PROFILE": "quality",
            "RECALL_SEARCH_CONCURRENCY": "",
            "RECALL_SEARCH_QUEUE": "   ",
            "RECALL_RERANK_THREADS": "",
        }
    )
    absent = resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "quality"})
    assert empty == absent


def test_an_operator_override_applies_to_the_selected_profile_only() -> None:
    env = {"RECALL_RETRIEVAL_PROFILE": "quality", "RECALL_SEARCH_CONCURRENCY": "3"}
    assert resolve_retrieval_profile(env).max_concurrency == 3
    assert QUALITY_PROFILE.max_concurrency != 3  # the module constant is not mutated


def test_admissions_for_two_profiles_are_isolated_from_each_other() -> None:
    """Saturating one profile's admission must not reject on the other's."""
    from recall_mcp.service import _admission

    tiny_fast = RetrievalProfile("fast", 20, 5, False, 250, max_concurrency=1, queue_capacity=1)
    tiny_quality = RetrievalProfile(
        "quality", 20, 5, True, 1500, max_concurrency=1, queue_capacity=1
    )
    fast_admission = _admission(tiny_fast)
    quality_admission = _admission(tiny_quality)
    assert fast_admission is not quality_admission
    with fast_admission:
        with quality_admission:  # the other profile still has its whole budget
            pass


# --------------------------------------------------------------------------------------------
# Bounded admission: the queue bounds latency, not merely parked threads
# --------------------------------------------------------------------------------------------


def _saturating_profile(budget_ms: int) -> RetrievalProfile:
    return RetrievalProfile(
        "fast", 20, 5, False, budget_ms, max_concurrency=1, queue_capacity=1
    )


def test_a_full_queue_is_rejected_without_waiting() -> None:
    admission = RetrievalAdmission(_saturating_profile(60_000))
    started = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with admission:
            started.set()
            release.wait(5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert started.wait(5)
    queued = threading.Thread(target=lambda: _park(admission, release), daemon=True)
    queued.start()
    time.sleep(0.2)  # let the queued request take the one queue slot

    t0 = time.perf_counter()
    with pytest.raises(RetrievalOverloaded) as excinfo:
        with admission:
            pass
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert excinfo.value.reason == "queue_full"
    assert elapsed_ms < 1000  # refused immediately, not after the 60 s budget
    release.set()
    worker.join(5)
    queued.join(5)


def _park(admission: RetrievalAdmission, release: threading.Event) -> None:
    try:
        with admission:
            release.wait(5)
    except RetrievalOverloaded:
        pass


def test_a_request_that_cannot_start_within_its_budget_is_rejected() -> None:
    """`latency_budget_ms` bounds the wait. Without it the queue caps parked threads only."""
    admission = RetrievalAdmission(_saturating_profile(120))
    started = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with admission:
            started.set()
            release.wait(5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert started.wait(5)

    t0 = time.perf_counter()
    with pytest.raises(RetrievalOverloaded) as excinfo:
        with admission:
            pass
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert excinfo.value.reason == "budget_exhausted"
    assert 100 <= elapsed_ms < 3000  # waited about the budget, then shed
    release.set()
    worker.join(5)


def test_a_budget_rejection_returns_its_queue_slot() -> None:
    """A shed request must not leak the slot it held, or the queue drains to zero permanently.

    The discriminator is the *reason* of the second rejection. If the first rejection leaked its
    queue slot, the second attempt is refused as `queue_full` without ever waiting; only a
    released slot lets it reach the budget wait again.
    """
    admission = RetrievalAdmission(_saturating_profile(120))
    started = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with admission:
            started.set()
            release.wait(5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert started.wait(5)

    reasons = []
    for _ in range(3):
        with pytest.raises(RetrievalOverloaded) as excinfo:
            with admission:
                pass
        reasons.append(excinfo.value.reason)
    assert reasons == ["budget_exhausted"] * 3
    release.set()
    worker.join(5)


def test_admission_is_reusable_after_a_normal_exit() -> None:
    admission = RetrievalAdmission(_saturating_profile(500))
    for _ in range(4):
        with admission:
            pass
    with admission:
        pass


def test_the_legacy_profile_waits_instead_of_shedding_under_contention() -> None:
    """Legacy behaviour is preserved: it QUEUES behind a busy slot rather than shedding.

    The earlier form of this test acquired an UNCONTENDED admission, which succeeds identically
    for a 24-day budget and a 1 ms one, so it could not fail on the property it named. The wait
    branch is only reached when the running slot is already taken, so the holder is the test.
    """
    assert LEGACY_PROFILE.latency_budget_ms > 24 * 60 * 60 * 1000
    admission = RetrievalAdmission(
        replace(LEGACY_PROFILE, max_concurrency=1, queue_capacity=1)
    )
    held = threading.Event()
    release = threading.Event()
    admitted = threading.Event()

    def hold() -> None:
        with admission:
            held.set()
            release.wait(10)

    def follow() -> None:
        with admission:
            admitted.set()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(5)
    waiter = threading.Thread(target=follow, daemon=True)
    waiter.start()
    # Still waiting, not shed: a finite budget of 0.3 s would already have raised by now.
    assert not admitted.wait(0.8)
    assert waiter.is_alive()
    release.set()
    assert admitted.wait(5), "legacy must be admitted once the slot frees, never shed"
    holder.join(5)
    waiter.join(5)
    assert not holder.is_alive() and not waiter.is_alive()


# --------------------------------------------------------------------------------------------
# The gate has to be REACHABLE, not merely correct
# --------------------------------------------------------------------------------------------


def test_admission_capacity_leaves_room_in_the_worker_pool() -> None:
    """The queue_full branch is unreachable if admission capacity equals the thread pool.

    Every MCP tool body runs inside `anyio.to_thread.run_sync`, so a request parked in
    `RetrievalAdmission` is holding a worker thread. anyio's default limiter is 40 tokens. Fast's
    8 + 32 is also 40, so the 41st concurrent search never reaches `__enter__` at all: it waits
    in anyio's limiter, which has no timeout, no budget and no counter. The shed that the whole
    design exists to produce would never fire, and `recall_retrieval_rejected_total` would sit at
    zero while clients waited unboundedly.

    This asserts the invariant rather than the constant, so it keeps holding if the profile
    numbers move.
    """
    from recall_mcp.server import worker_thread_budget

    for profile in (FAST_PROFILE, QUALITY_PROFILE, LEGACY_PROFILE):
        capacity = profile.max_concurrency + profile.queue_capacity
        assert worker_thread_budget(profile) > capacity, (
            f"{profile.name}: worker pool must exceed admission capacity {capacity}"
        )


def test_startup_raises_the_worker_pool_above_the_admission_capacity() -> None:
    """Asserting the invariant is not enforcing it. This runs the code that applies it."""
    import anyio
    import anyio.to_thread

    from recall_mcp.server import apply_worker_thread_budget

    async def _check() -> tuple[float, float]:
        before = anyio.to_thread.current_default_thread_limiter().total_tokens
        apply_worker_thread_budget(FAST_PROFILE)
        return before, anyio.to_thread.current_default_thread_limiter().total_tokens

    before, after = anyio.run(_check)
    assert before == 40, "anyio's documented default; if this moved, re-derive the invariant"
    assert after > FAST_PROFILE.max_concurrency + FAST_PROFILE.queue_capacity


def test_the_worker_pool_is_never_lowered() -> None:
    """A host that already sized its pool generously must not be shrunk by ours."""
    import anyio
    import anyio.to_thread

    from recall_mcp.server import apply_worker_thread_budget

    async def _check() -> tuple[float, float]:
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 500
        apply_worker_thread_budget(QUALITY_PROFILE)
        return 500, limiter.total_tokens

    before, after = anyio.run(_check)
    assert after == before


# --------------------------------------------------------------------------------------------
# Startup validation: a misconfigured process must refuse to start, not to serve
# --------------------------------------------------------------------------------------------


def test_startup_refuses_a_conflicting_profile_and_legacy_switch() -> None:
    from recall_mcp.service import startup_retrieval_profile

    with pytest.raises(ValueError, match="conflicts"):
        startup_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": "quality", "RECALL_RERANK": "false"}
        )


def test_startup_refuses_a_quality_profile_with_no_pinned_reranker() -> None:
    from recall_mcp.service import startup_retrieval_profile

    with pytest.raises(ValueError, match="RECALL_RERANK_PATH and RECALL_RERANK_SHA256"):
        startup_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "quality"})


def test_startup_refuses_a_reranker_artifact_that_is_not_the_pinned_one() -> None:
    """The digest must AGREE with the pin, not define it.

    Reading the operator's value and then verifying the tree against it proves the tree hashes to
    its own hash, which is true of every tree. The pin is the value chosen elsewhere that makes
    the comparison mean something.
    """
    from recall_mcp.service import startup_retrieval_profile

    with pytest.raises(ValueError, match="does not match the reranker pinned"):
        startup_retrieval_profile(
            {
                "RECALL_RETRIEVAL_PROFILE": "quality",
                "RECALL_RERANK_PATH": "/opt/recall-enterprise/models/ms-marco-MiniLM-L-6-v2",
                "RECALL_RERANK_SHA256": "0" * 64,
            }
        )


def test_startup_accepts_the_pinned_reranker_and_loads_nothing() -> None:
    """A configuration check that needed torch installed would not be a startup check."""
    from recall.rerank import PINNED_RERANKER_SHA256
    from recall_mcp.service import startup_retrieval_profile

    profile = startup_retrieval_profile(
        {
            "RECALL_RETRIEVAL_PROFILE": "quality",
            "RECALL_RERANK_PATH": "/nonexistent/path/that/is/never/opened",
            "RECALL_RERANK_SHA256": PINNED_RERANKER_SHA256.upper(),  # case-insensitive
        }
    )
    assert profile.name == "quality"
    assert profile.reranker is True
    assert profile.inference_threads == 1


def test_startup_accepts_the_code_profile_without_quality_artifact() -> None:
    from recall_mcp.service import startup_retrieval_profile

    profile = startup_retrieval_profile(
        {"RECALL_RETRIEVAL_PROFILE": "code", "RECALL_ACCEPT_REMOTE_MODEL_CODE": "1"}
    )
    assert profile.name == "code"
    assert profile.reranker is True
    assert profile.latency_budget_ms == 10_000
    assert profile.inference_threads == 1


def test_startup_refuses_code_profile_without_remote_model_code_opt_in() -> None:
    from recall_mcp.service import startup_retrieval_profile

    with pytest.raises(ValueError, match="RECALL_ACCEPT_REMOTE_MODEL_CODE"):
        startup_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "code"})


def test_startup_refuses_code_profile_with_non_coreb_model() -> None:
    from recall_mcp.service import startup_retrieval_profile

    with pytest.raises(ValueError, match="requires RECALL_RERANK_MODEL=coreb-code"):
        startup_retrieval_profile(
            {
                "RECALL_RETRIEVAL_PROFILE": "code",
                "RECALL_RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            }
        )


def test_startup_returns_the_digest_it_validated_not_the_one_it_was_given() -> None:
    """Validating a normalised value and returning the raw one re-opens the hole it closed.

    `verify_artifact` rejects on LENGTH before it lowercases, so a digest that is correct except
    for a trailing newline (what a dotenv literal block or a padded `.env` line produces) passes
    the startup comparison and then raises on the first search. That is exactly the "starts
    clean, refuses every search" failure `startup_retrieval_profile` exists to remove.
    """
    from recall.rerank import PINNED_RERANKER_SHA256
    from recall_mcp.service import _validate_quality_reranker_config

    model_path, digest = _validate_quality_reranker_config(
        {
            "RECALL_RERANK_PATH": "  /opt/recall-enterprise/models/ms-marco-MiniLM-L-6-v2  ",
            "RECALL_RERANK_SHA256": f"  {PINNED_RERANKER_SHA256.upper()}\n",
        }
    )
    assert digest == PINNED_RERANKER_SHA256
    assert len(digest) == 64
    assert model_path == "/opt/recall-enterprise/models/ms-marco-MiniLM-L-6-v2"


def test_each_module_constant_equals_what_its_own_resolver_produces() -> None:
    """A constant that disagrees with its own resolver is a trap for anything keyed on it.

    `QUALITY_PROFILE` left `inference_threads` at the dataclass default while the resolver filled
    in `1`, so the exported constant and the resolved profile were unequal objects describing the
    same configuration.
    """
    for name, constant in (
        ("fast", FAST_PROFILE),
        ("quality", QUALITY_PROFILE),
        ("code", CODE_PROFILE),
    ):
        assert resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": name}) == constant


def test_a_queue_is_identified_by_what_defines_a_queue() -> None:
    """`inference_threads` has nothing to do with admission and must not split a queue.

    Memoising the admission on the WHOLE profile made an unrelated field part of a queue's
    identity, so two profiles meaning one queue each got their own pair of semaphores and the
    process-wide concurrency bound silently doubled. Asserted on two profiles that differ ONLY in
    that field, so the fix cannot be satisfied by making the constants happen to agree.
    """
    import recall_mcp.service as service

    one = replace(QUALITY_PROFILE, inference_threads=1)
    four = replace(QUALITY_PROFILE, inference_threads=4)
    assert one != four  # genuinely different objects, so this is not comparing a thing to itself
    assert service._admission(one) is service._admission(four)
    # ...and a profile that differs in something a queue IS made of still gets its own.
    assert service._admission(one) is not service._admission(
        replace(QUALITY_PROFILE, max_concurrency=3)
    )


def test_an_admission_capacity_that_would_spawn_a_thread_army_is_refused() -> None:
    """The pool is now sized FROM the profile, so the profile must not be unbounded.

    Before the pool was sized from the profile, anyio's 40-token default was an accidental but
    real ceiling on how many OS threads a search path could occupy. Sizing the pool from the
    profile removes that ceiling, which turns an unvalidated operator variable into a
    thread-count knob: `RECALL_SEARCH_QUEUE=5000` would have asked for 5008 threads.
    """
    with pytest.raises(ValueError, match="admission capacity"):
        resolve_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": "fast", "RECALL_SEARCH_QUEUE": "5000"}
        )
    at_the_cap = resolve_retrieval_profile(
        {
            "RECALL_RETRIEVAL_PROFILE": "fast",
            "RECALL_SEARCH_CONCURRENCY": "8",
            "RECALL_SEARCH_QUEUE": str(MAX_ADMISSION_CAPACITY - 8),
        }
    )
    assert at_the_cap.max_concurrency + at_the_cap.queue_capacity == MAX_ADMISSION_CAPACITY


def test_a_cached_reranker_failure_does_not_grow_a_traceback_per_request() -> None:
    """Re-raising ONE exception instance accumulates frames, and those frames pin the query.

    Every re-raise appends the current frame to the same `__traceback__`. The retained frames
    hold their locals, which on this path include the caller's query text and the store, so a
    misconfigured quality server would grow memory per request AND retain user text for the
    process lifetime. The point of caching the failure was to stop doing work, not to start
    accumulating it.
    """
    import recall_mcp.service as service
    import recall_mcp.factories as factories

    def failing_factory(env=None):
        raise RuntimeError("model artifact checksum mismatch")

    service._reset_reranker_cache()
    original = factories._new_reranker
    factories._new_reranker = failing_factory
    depths = []
    try:
        for _ in range(5):
            try:
                service._build_reranker()
            except RuntimeError as exc:
                depth = 0
                tb = exc.__traceback__
                while tb is not None:
                    depth += 1
                    tb = tb.tb_next
                depths.append(depth)
    finally:
        factories._new_reranker = original
        service._reset_reranker_cache()
    # The first call raises through the factory and is one frame deeper; every call after it is
    # served from the cache and must be identical to the one before. Growth is the defect: the
    # unfixed code produced 3, 4, 5, 6, 7.
    assert len(set(depths[1:])) == 1, f"traceback grew across re-raises: {depths}"
    assert max(depths) <= depths[0], f"a cached re-raise deepened the traceback: {depths}"


def test_an_interrupt_during_a_cold_build_does_not_poison_the_reranker_forever() -> None:
    """`KeyboardInterrupt` is not a bad artifact. Caching it would make it permanent.

    Caching the failure is right for a configuration error, which is deterministic. A
    `BaseException` that arrives asynchronously says nothing about the artifact, and turning it
    into a permanent verdict would take a transient event and make it a process-lifetime outage.
    """
    import recall_mcp.service as service
    import recall_mcp.factories as factories

    attempts = []

    def interrupted_then_fine(env=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise KeyboardInterrupt
        return None

    service._reset_reranker_cache()
    original = factories._new_reranker
    factories._new_reranker = interrupted_then_fine
    try:
        with pytest.raises(KeyboardInterrupt):
            service._build_reranker()
        # Caught explicitly rather than allowed to propagate: a `KeyboardInterrupt` escaping a
        # test aborts the whole pytest session, and an aborted session is not a failed
        # assertion. The mutation that proves this test can fail has to produce a red test, not
        # a run that never finished.
        try:
            rebuilt = service._build_reranker()
        except BaseException as exc:  # noqa: BLE001 - the assertion IS that nothing is raised
            pytest.fail(f"a transient interrupt poisoned the cache: {exc!r}")
        assert rebuilt is None  # retried, not poisoned
    finally:
        factories._new_reranker = original
        service._reset_reranker_cache()
    assert len(attempts) == 2


def test_a_failed_reranker_construction_is_not_retried_on_every_request() -> None:
    """A broken artifact must fail fast, not re-hash a model tree per client request.

    Only successes were memoised, so a bad digest or a missing path re-ran the full artifact
    verification on every search, holding both the construction lock and a running slot. A
    misconfigured quality server turned each request into a multi-hundred-megabyte disk read.
    """
    import recall_mcp.service as service
    import recall_mcp.factories as factories

    calls = []

    def failing_factory(env=None):
        calls.append(1)
        raise RuntimeError("model artifact checksum mismatch")

    service._reset_reranker_cache()
    original = factories._new_reranker
    factories._new_reranker = failing_factory
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                service._build_reranker()
    finally:
        factories._new_reranker = original
        service._reset_reranker_cache()
    assert len(calls) == 1, f"the failed construction was retried {len(calls)} times"


def test_the_pinned_reranker_is_the_measured_ms_marco_minilm() -> None:
    """The pin is a fact recorded from a provisioned artifact, not a placeholder.

    Recomputed independently on VPS2 on 2026-08-05 over
    `/opt/recall-enterprise/models/ms-marco-MiniLM-L-6-v2`, and equal to the value
    `/opt/recall-enterprise/manifest.json` has carried since 2026-08-03.
    """
    from recall.rerank import (
        PINNED_RERANKER_MODEL,
        PINNED_RERANKER_REVISION,
        PINNED_RERANKER_SHA256,
    )

    assert PINNED_RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert PINNED_RERANKER_REVISION == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert PINNED_RERANKER_SHA256 == (
        "db6ad87969c7dc78320152e68a16118aeb4b2a6f7d8cc979c57f61ddb5e2ab2a"
    )
    assert len(PINNED_RERANKER_SHA256) == 64


def test_the_server_refuses_to_start_on_a_contradictory_profile(monkeypatch) -> None:
    """Not "refuses to serve". The lifespan must fail before it opens anything.

    Checked by what the refusal is NOT: if the profile check were placed after the DSN and store
    setup, this would raise something else (or nothing, on a healthy database), so the `match` is
    load-bearing rather than decoration.
    """
    import asyncio

    from recall_mcp.server import _make_lifespan

    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.setenv("RECALL_RERANK", "true")
    lifespan = _make_lifespan(None)

    async def _start() -> None:
        async with lifespan(None):  # type: ignore[arg-type]
            pass

    with pytest.raises(ValueError, match="conflicts"):
        asyncio.run(_start())
