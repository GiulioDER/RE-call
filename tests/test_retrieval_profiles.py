"""Profile resolution, bounded admission, and the request-time latency budget.

The behaviours pinned here are the ones an operator is promised by
`docs/ENTERPRISE_RETRIEVAL.md`: a process serves exactly one cost profile, a client cannot buy
its way onto the expensive one, and the process refuses work it cannot afford *before* it spends
anything on it. Each test below was shown red by mutating the module it covers; the mutations are
recorded in `docs/ENTERPRISE_PROGRAM_STATUS.md`.
"""
from __future__ import annotations

import threading
import time

import pytest

from recall.profiles import (
    FAST_PROFILE,
    LEGACY_PROFILE,
    QUALITY_PROFILE,
    RetrievalAdmission,
    RetrievalOverloaded,
    RetrievalProfile,
    resolve_retrieval_profile,
)


# --------------------------------------------------------------------------------------------
# Profile resolution and conflict refusal
# --------------------------------------------------------------------------------------------


def test_fast_and_quality_match_the_published_specification() -> None:
    """The two numbers an operator reads off the doc are the two numbers the code uses."""
    assert (FAST_PROFILE.candidate_k, FAST_PROFILE.returned_k) == (20, 5)
    assert FAST_PROFILE.reranker is False
    assert FAST_PROFILE.latency_budget_ms == 250
    assert (QUALITY_PROFILE.candidate_k, QUALITY_PROFILE.returned_k) == (20, 5)
    assert QUALITY_PROFILE.reranker is True
    assert QUALITY_PROFILE.latency_budget_ms == 1500


def test_an_unknown_profile_name_is_refused() -> None:
    with pytest.raises(ValueError, match="must be 'fast' or 'quality'"):
        resolve_retrieval_profile({"RECALL_RETRIEVAL_PROFILE": "premium"})


def test_legacy_rerank_switch_survives_only_while_no_profile_is_selected() -> None:
    """`RECALL_RERANK` keeps its old meaning, but only on the path that has no profile."""
    legacy = resolve_retrieval_profile({"RECALL_RERANK": "true"})
    assert legacy.name == "legacy"
    assert resolve_retrieval_profile({}).name == "legacy"


@pytest.mark.parametrize(
    ("profile", "rerank"),
    [("fast", "true"), ("fast", "on"), ("quality", "false"), ("quality", "0")],
)
def test_a_conflicting_profile_and_legacy_switch_refuses(profile: str, rerank: str) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        resolve_retrieval_profile(
            {"RECALL_RETRIEVAL_PROFILE": profile, "RECALL_RERANK": rerank}
        )


@pytest.mark.parametrize(
    ("profile", "rerank"), [("fast", "off"), ("quality", "1")]
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
    """`.env.example` ships these keys empty, and a dotenv load puts an empty STRING in the env.

    Reading that as a malformed integer would make the shipped example refuse startup, which is
    the failure mode a configuration example exists to prevent. Empty and absent must agree.
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


def test_the_legacy_profile_keeps_its_unbounded_wait() -> None:
    """Legacy behaviour is preserved: no budget, so no budget-driven shedding."""
    assert LEGACY_PROFILE.latency_budget_ms > 24 * 60 * 60 * 1000
    admission = RetrievalAdmission(LEGACY_PROFILE)
    with admission:
        pass


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
