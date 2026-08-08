"""`benchmarks._trust.bench_search`: the two-part trust configuration every arm retrieves under.

Needs no database, no `fastembed`, and no `membench`, so unlike the adapters it configures, this
runs in CI. That is deliberate: the decision under test is shared by four arms, three of which CI
cannot import at all, so testing the decision ONCE where it is reachable is the only way it gets
covered before a merge rather than after a published run.
"""
from __future__ import annotations

from typing import Any

import pytest

from recall.trust_policy import TrustMode, TrustPolicy


class _FakeEmbedder:
    """Minimal embedder: `embedding_profile_id` falls through to `name` when there is no profile."""

    name = "fake-profile-v1"
    dim = 8


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Intercept at `trusted_search`, the layer BENEATH the seam under test.

    Patching `research_search` would observe only what `bench_search` passes, and would stay green
    if `bench_search` ever stopped routing through the research seam — the exact regression this
    exists to catch. Patching the layer underneath means the assertions are made on what actually
    arrives at the trust layer, after every wrapper has had its turn, which is the only place the
    question "what policy did this query really run under" has an answer.
    """
    from recall.eval import _research_trust

    seen: dict[str, Any] = {}

    def _fake(store: Any, embedder: Any, query: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["query"] = query
        return "sentinel-result"

    monkeypatch.setattr(_research_trust, "trusted_search", _fake)
    return seen


def test_bench_search_runs_development_mode_with_an_explicit_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both halves, which is the whole point of there being one function for this.

    The policy alone leaves `calibration=None`, which `recall.trust` reads as "no threshold exists
    at all" and answers by blanking every verdict to `unverified` and forcing `abstained=False`.
    An arm reading `result.abstained` then measures a structural zero, and an arm filtering on
    `verdict == "ok"` measures a structural zero the other way: it cites nothing at all.
    """
    from recall.guards import DEFAULT_GAP_THRESHOLD

    from benchmarks._trust import bench_search

    seen = _capture(monkeypatch)
    result = bench_search(object(), _FakeEmbedder(), "q")

    assert result == "sentinel-result"  # the return value is passed through, not swallowed
    assert seen["policy"].mode is TrustMode.DEVELOPMENT
    assert seen["calibration"] is not None
    assert seen["calibration"].threshold == DEFAULT_GAP_THRESHOLD
    # Keyed to the embedding PROFILE, which is what a fitted calibration file is written under.
    assert seen["calibration"].embedder == "fake-profile-v1"


def test_a_caller_keeps_its_own_calibration_and_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults, not impositions. An arm that fits a real calibration must be able to use it.

    `benchmarks/beam/systems.py` loads a fitted artifact with `--calibration`, and an arm
    measuring strict-mode refusal needs its own policy. Both must survive this helper.
    """
    from recall.calibration import Calibration

    from benchmarks._trust import bench_search

    seen = _capture(monkeypatch)
    fitted = Calibration(embedder="fitted-v2", threshold=0.31)
    strict = TrustPolicy.strict_policy()
    bench_search(object(), _FakeEmbedder(), "q", calibration=fitted, policy=strict)

    assert seen["calibration"] is fitted
    assert seen["policy"] is strict


def test_other_search_arguments_are_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is a thin default-supplier, not a narrowing wrapper: `k`, `candidate_k`, `reranker`,
    `now` and the rest reach `trusted_search` untouched, or the arms lose their configuration."""
    from benchmarks._trust import bench_search

    seen = _capture(monkeypatch)
    reranker = object()
    bench_search(object(), _FakeEmbedder(), "q", k=9, candidate_k=77, reranker=reranker)

    assert (seen["k"], seen["candidate_k"]) == (9, 77)
    assert seen["reranker"] is reranker


def test_the_ladder_arm_queries_through_the_bench_trust_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one sibling adapter CI can import, exercised end-to-end through `query`.

    `benchmarks/membench/*` import `membench` at module scope and that package is deliberately not
    a dependency, so they are covered by the source-level rule in `test_bench_trust_policy.py`
    instead. The ladder adapter imports only `recall` and its own package, so its real behaviour is
    reachable here — and its `query` reads BOTH `result.abstained` and `verdict == "ok"`, which is
    the combination that makes the calibration half load-bearing.

    Built with `object.__new__` rather than the constructor, which wants a live database (and runs
    a non-empty-table guard against it). That idiom is already the convention for these adapters:
    `tests/test_membench_adapters.py` reads `system_version` off `object.__new__(cls)` for the same
    reason.
    """
    from benchmarks.ladder.systems.recall_system import RecallSystem

    seen = _capture(monkeypatch)
    system = object.__new__(RecallSystem)
    system._store = object()
    system._embedder = _FakeEmbedder()

    # `match=`, so this cannot absorb an AttributeError raised BEFORE the search by some attribute
    # a future `query` reads off the half-built object. That error would leave `seen` empty and
    # the assertions below reading a dict that was never populated.
    with pytest.raises(AttributeError, match=r"'str' object has no attribute 'hits'"):
        # `_capture` returns a string, so `.hits` raises — the search wiring is what is under test
        # here, not the response mapping, which needs a real `TrustedResult` to mean anything.
        system.query("q")

    assert seen["query"] == "q"
    assert seen["policy"].mode is TrustMode.DEVELOPMENT
    assert seen["calibration"] is not None


def test_a_none_calibration_is_replaced_rather_than_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`calibration=None` is the spelling that MEANS "no threshold exists" to `recall.trust`.

    `setdefault` leaves a present None alone, so an arm threading an optional
    `calibration: Calibration | None = None` straight through would silently reach the branch that
    forces `abstained=False` — the defect this helper exists to prevent, arriving by the one
    spelling that reads like "use the default". Absent and None must mean the same thing here.
    """
    from recall.guards import DEFAULT_GAP_THRESHOLD

    from benchmarks._trust import bench_search

    seen = _capture(monkeypatch)
    bench_search(object(), _FakeEmbedder(), "q", calibration=None)

    assert seen["calibration"] is not None
    assert seen["calibration"].threshold == DEFAULT_GAP_THRESHOLD


def test_a_none_policy_is_replaced_rather_than_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror of the case above, and it fails the other way: straight back to STRICT.

    `research_search` applies its own policy with `kwargs.setdefault`, which leaves a PRESENT None
    alone, and `recall.trust` then resolves `None or TrustPolicy()` to strict. So an arm threading
    an optional `policy: TrustPolicy | None = None` through this helper would refuse every
    question with INDEX_NOT_READY — the exact defect this module exists to prevent, restored by
    the one spelling that reads like "use the default". Guarding `calibration` but not `policy`
    left that open.
    """
    from benchmarks._trust import bench_search

    seen = _capture(monkeypatch)
    bench_search(object(), _FakeEmbedder(), "q", policy=None)

    assert seen["policy"] is not None
    assert seen["policy"].mode is TrustMode.DEVELOPMENT
