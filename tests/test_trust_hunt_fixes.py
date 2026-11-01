"""Regression tests for the 2026-10-31 CCA hunt of recall/trust.py, trust_policy.py, calibration.py.

No database and no network: the store and the retriever are fakes, as in `tests/test_trust.py`.
Each behaviour test names the production line it pins and the mutation or pre-fix code it was
seen to fail against.
"""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from recall.calibration import Calibration
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
FRESH = StalenessReport(
    stale=False, newest_indexed_at=NOW, age=timedelta(0), max_age=timedelta(days=2)
)
CAL = Calibration(embedder="test", threshold=0.5, scale=0.05)


def _hit(cid: str, file: str, score: float) -> ScoredChunk:
    chunk = Chunk(id=cid, source=file, text=f"text of {file}", metadata={"file": file, "ord": 0})
    return ScoredChunk(chunk=chunk, score=score, indexed_at=NOW)


def _result(hits: list[ScoredChunk]) -> RetrievalResult:
    return RetrievalResult(query="q", hits=hits, gap_warning=False, staleness=FRESH)


def _fake_retriever(monkeypatch, hits: list[ScoredChunk]) -> None:
    from recall import trust

    class _FakeRetriever:
        def __init__(self, store, embedder, **kwargs):
            del store, embedder, kwargs

        def search(self, query, k, source=None, scope=None, **kwargs):
            del query, k, source, scope, kwargs
            return _result(list(hits))

    monkeypatch.setattr(trust, "HybridRetriever", _FakeRetriever)


class _CertifiedStore:
    """A generation store whose resolver answers a certified artifact bound to generation g1."""

    tenant = "tenant-a"

    def __init__(self, supersession: dict[str, str] | None = None) -> None:
        self._supersession = supersession or {}

    def generation_binding(self) -> dict[str, str]:
        return {"tenant_id": "tenant-a", "generation_id": "g1"}

    def resolve_calibration(self):
        from recall.calibration_v2 import CalibrationStatus

        artifact = SimpleNamespace(runtime=CAL, calibration_id="cal-1", query_set_digest="qs")
        return SimpleNamespace(status=CalibrationStatus.CERTIFIED, artifact=artifact)

    def supersession(self):
        return dict(self._supersession), frozenset()


# STAKES-001 / NUM-001


def test_a_nan_cosine_maps_to_zero_confidence_not_one():
    """A NaN similarity is no similarity, so its confidence is the floor of the range.

    Red proof: against the pre-fix `Calibration.confidence` (recall/calibration.py, no NaN branch
    before the clamp), `max(-60.0, min(60.0, nan))` is 60.0 and this returns 1.0, failing the
    first assertion.
    """
    assert CAL.confidence(float("nan")) == 0.0
    # Control: finite inputs keep the sigmoid, exactly 0.5 at the threshold.
    assert CAL.confidence(0.5) == pytest.approx(0.5)
    assert CAL.confidence(0.9) > 0.99


# STAKES-002


def test_a_nan_dense_hit_is_never_served_as_trusted(monkeypatch):
    """A NaN cosine (pgvector, zero-norm vector) must not clear the certified threshold.

    Red proof: against the pre-fix `_trusted_search` (recall/trust.py, no
    `_without_non_finite_dense` before `evaluate`), the NaN hit compares False with
    `hit.score < threshold` in `trust_gate._verdict`, is verdicted `ok`, and the result is served
    `trusted`, `calibrated` and not abstained, failing the `abstained` assertion.
    """
    from recall.trust import trusted_search

    _fake_retriever(monkeypatch, [_hit("c1", "a/one.md", float("nan")), _hit("c2", "a/two.md", 0.2)])

    result = trusted_search(_CertifiedStore(), object(), "q", k=2, now=NOW)

    assert result.trust_state == "trusted"
    assert result.abstained is True
    assert not [hit for hit in result.hits if hit.verdict == "ok"]
    assert all(not math.isnan(hit.cosine) for hit in result.hits)
    # Control: the finite hit is still judged, below the threshold.
    assert [(hit.chunk.id, hit.verdict) for hit in result.hits] == [("c2", "low_confidence")]


def test_a_finite_certified_hit_is_still_served(monkeypatch):
    """Control for the NaN filter: a finite hit above the threshold is untouched."""
    from recall.trust import trusted_search

    _fake_retriever(monkeypatch, [_hit("c1", "a/one.md", 0.9)])

    result = trusted_search(_CertifiedStore(), object(), "q", k=1, now=NOW)

    assert result.abstained is False
    assert result.calibrated is True
    assert [hit.verdict for hit in result.hits] == ["ok"]


# SEC-001


def _policy_for_prefix_a():
    from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy

    policy = SourceSecurityPolicy((SourceRule("a"), SourceRule("b", principals=frozenset({"hr"}))))
    return policy, AccessContext("analyst", "tenant-a")


def test_a_successor_the_caller_may_not_read_is_withheld(monkeypatch):
    """A caller restricted to prefix `a` must not learn the name of a successor under `b`.

    Red proof: against the pre-fix `_trusted_search` (recall/trust.py, no
    `_withhold_unauthorized_successors` after `evaluate`), `superseded_by` carries `b/new.md` and
    the abstention reason names it, failing the first assertion.
    """
    from recall.trust import trusted_search

    _fake_retriever(monkeypatch, [_hit("c1", "a/old.md", 0.9)])
    policy, context = _policy_for_prefix_a()

    result = trusted_search(
        _CertifiedStore({"a/old.md": "b/new.md"}),
        object(),
        "q",
        k=1,
        now=NOW,
        security_policy=policy,
        access_context=context,
    )

    [hit] = result.hits
    assert hit.validity.superseded_by == "<withheld>"
    assert hit.verdict == "superseded"
    assert result.abstained is True
    assert "b/new.md" not in result.reason


def test_a_successor_the_caller_may_read_is_still_named(monkeypatch):
    """Control: withholding applies only to a successor the policy denies."""
    from recall.trust import trusted_search

    _fake_retriever(monkeypatch, [_hit("c1", "a/old.md", 0.9)])
    policy, context = _policy_for_prefix_a()

    result = trusted_search(
        _CertifiedStore({"a/old.md": "a/new.md"}),
        object(),
        "q",
        k=1,
        now=NOW,
        security_policy=policy,
        access_context=context,
    )

    [hit] = result.hits
    assert hit.validity.superseded_by == "a/new.md"
    assert "a/new.md" in result.reason


def test_the_withheld_sentinel_matches_the_records_api():
    """Nonbehavioral pin: trust and records must spell a withheld successor the same way."""
    from recall import records, trust

    assert trust._WITHHELD == records.WITHHELD


# STAKES-003


def test_a_string_mode_is_coerced_so_strict_stays_strict():
    """`TrustPolicy(mode="strict")` must refuse like the default, not degrade.

    Red proof: against the pre-fix `TrustPolicy` (recall/trust_policy.py, no `__post_init__`),
    `strict` tests `self.mode is TrustMode.STRICT`, the plain string is not the member, and the
    first assertion fails.
    """
    from recall.trust_policy import TrustMode, TrustPolicy

    assert TrustPolicy(mode="strict").strict is True
    assert TrustPolicy(mode="development").mode is TrustMode.DEVELOPMENT
    with pytest.raises(ValueError):
        TrustPolicy(mode="strcit")


# ENV-001


def test_the_dependency_env_value_is_normalised_and_named_when_wrong(monkeypatch):
    """`RECALL_DEPENDENCY_INVALIDATION` is matched like its sibling env keys, and named on error.

    Red proof: against the pre-fix `_trusted_search` (recall/trust.py, raw env read), `"OFF "` is
    not in `{"off", "enforce"}` and raises ValueError, failing the first search.
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    _fake_retriever(monkeypatch, [_hit("c1", "a/one.md", 0.9)])
    store = SimpleNamespace(supersession=lambda: ({}, frozenset()))

    result = trusted_search(
        store,
        object(),
        "q",
        k=1,
        now=NOW,
        calibration=CAL,
        policy=TrustPolicy.development(),
        env={"RECALL_DEPENDENCY_INVALIDATION": "OFF "},
    )
    assert [hit.verdict for hit in result.hits] == ["ok"]

    with pytest.raises(ValueError, match="RECALL_DEPENDENCY_INVALIDATION"):
        trusted_search(
            store,
            object(),
            "q",
            k=1,
            now=NOW,
            calibration=CAL,
            policy=TrustPolicy.development(),
            env={"RECALL_DEPENDENCY_INVALIDATION": "sometimes"},
        )


# DAT-001 / NUM-002 / BUG-001


def test_a_saved_calibration_reloads_with_the_verdict_it_was_written_with(tmp_path):
    """`certified` is recomputed on load, so the separability it is recomputed from round-trips.

    The samples put the 95% lower bound at 0.89997, just under MIN_SEPARABILITY 0.90.

    Red proof: against the pre-fix `_payload` (recall/calibration.py,
    `round(cal.separability, 4)`), 0.91946 is saved as 0.9195, whose lower bound is 0.90002, and
    the reloaded calibration is certified while the original is not, failing the assertion.
    """
    from recall.calibration import from_samples, load_for, save

    unanswerable = [0.3 + 0.001 * j for j in range(200)]
    answerable = [0.95] * 459 + [0.3035] * 23 + [0.3025] * 18
    cal = from_samples("bge", answerable, unanswerable)
    assert cal.certified is False

    back = load_for("bge", save(cal, tmp_path / "c.json"))

    assert back is not None
    assert back.separability == cal.separability
    assert back.certified is cal.certified


# CODE-002


def _parameter_names(function) -> set[str]:
    return set(inspect.signature(function).parameters)


def test_every_parameter_is_forwarded_through_both_pass_throughs(monkeypatch):
    """Guard for CODE-002: the snapshot re-entry and `trusted_search` forward every parameter.

    Nonbehavioral today (every name is forwarded), so it is not claimed as red proof against the
    pre-fix tree. Mutation check: deleting `env=env,` from the snapshot re-entry in
    `_trusted_search` fails the first assertion, and deleting `env=env,` from `forwarded` in
    `trusted_search` fails the second.
    """
    from recall import trust

    inner = trust._trusted_search
    expected = _parameter_names(inner) - {"store", "embedder", "query"}
    captured: list[dict[str, object]] = []

    def _record(store, embedder, query, **kwargs):
        captured.append(kwargs)
        return "sentinel"

    @contextmanager
    def _snapshot():
        yield

    monkeypatch.setattr(trust, "_trusted_search", _record)
    store = SimpleNamespace(snapshot=_snapshot)

    assert inner(store, object(), "q") == "sentinel"
    assert set(captured[-1]) == expected

    assert trust.trusted_search(store, object(), "q") == "sentinel"
    assert set(captured[-1]) == expected
    assert _parameter_names(trust.trusted_search) - {"ledger"} == _parameter_names(inner)


def test_the_candidate_trace_callback_sees_the_pool_trust_judged():
    """With a NaN dense hit, the trace handed to the callback and its trust result agree.

    `recall_mcp/reasoning_diagnostics.py` refuses a trace whose ids differ from the trusted ids.
    Mutation check (L5.5 finding on FIX-002): passing the unfiltered `candidate_trace` to the
    callback while evaluating the filtered pool fails the id assertion.
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    class _Store:
        tenant = "tenant-a"
        generation_id = "generation-a"

        def query_dense(self, vector, k, source=None):
            del vector, source
            return [_hit("dense-a", "a.md", float("nan")), _hit("both", "b.md", 0.8)][:k]

        def query_sparse(self, query, k, source=None, vec=None):
            del query, source, vec
            return [_hit("both", "b.md", 0.8)][:k]

        def newest_indexed_at(self):
            return NOW

        def supersession(self):
            return {}, frozenset()

    class _Embedder:
        dim = 3
        name = "test"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    traces = []
    trusted_search(
        _Store(),
        _Embedder(),
        "q",
        k=1,
        candidate_k=2,
        calibration=CAL,
        policy=TrustPolicy.development(),
        candidate_trace_callback=lambda raw, trusted, cal: traces.append((raw, trusted)),
    )

    [(raw, trusted)] = traces
    assert [hit.chunk.id for hit in raw.result.hits] == [hit.chunk.id for hit in trusted.hits]
    assert "dense-a" not in [hit.chunk.id for hit in raw.result.hits]


# Second hunt (run 20261031T204801Z) over the same files.


def test_a_nan_calibration_sample_is_refused_not_fitted():
    """STAKES-102: a NaN sample must not move the fitted threshold and still certify.

    Red proof: against the pre-fix `best_threshold` (recall/calibration.py, no finiteness check),
    `sorted()` leaves the NaN in place, the q95 ceiling is read from an unsorted list, and
    `from_samples` returns a certified calibration at 0.55 instead of raising, failing the
    `pytest.raises` assertion.
    """
    from recall.calibration import from_samples

    with pytest.raises(ValueError, match="finite"):
        from_samples("e", [0.8] * 40, [0.75] * 4 + [float("nan")] + [0.3] * 36)
    # Control: the same data without the NaN still fits.
    assert from_samples("e", [0.8] * 40, [0.75] * 4 + [0.3] * 36).threshold == pytest.approx(0.775)


def test_a_corrupt_calibration_file_falls_back_instead_of_raising(tmp_path):
    """DAT-101: an oversized integer or a deeply nested file is malformed, not a crash.

    Red proof: against the pre-fix `load_for` (recall/calibration.py, except tuples without
    OverflowError and RecursionError), the 400-digit threshold raises OverflowError from
    `float()`, failing the test with that exception.
    """
    from recall.calibration import load_for

    digits = "1" * 400
    huge = tmp_path / "huge.json"
    huge.write_text('{"embedder": "e", "threshold": %s, "scale": 0.05}' % digits, encoding="utf-8")
    assert load_for("e", huge) is None
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    assert load_for("e", deep) is None
    sep = tmp_path / "sep.json"
    sep.write_text(
        '{"embedder": "e", "threshold": 0.5, "scale": 0.05, "separability": %s}' % digits,
        encoding="utf-8",
    )
    loaded = load_for("e", sep)
    assert loaded is not None and loaded.separability is None


def test_a_dependency_the_caller_may_not_read_is_withheld(monkeypatch):
    """SEC-101: a transitive dependency under a denied prefix must not be named to the caller.

    Red proof: against the pre-fix `_trusted_search` (recall/trust.py, which withheld only
    `superseded_by`), the abstention reason names `hr/z.md`, failing the first assertion.
    """
    from recall.trust import trusted_search
    from recall.types import InvalidationReason

    reason = InvalidationReason(
        dependency="hr/z.md", cause="superseded", path=("a/x.md", "a/y.md", "hr/z.md")
    )
    projection = SimpleNamespace(
        edges=(),
        diagnostics=(),
        reason_for=lambda node: reason if node == "a/x.md" else None,
    )
    store = _CertifiedStore()
    store.dependency_projection = lambda as_of=None, known_as_of=None: projection
    _fake_retriever(monkeypatch, [_hit("c1", "a/x.md", 0.9)])
    policy, context = _policy_for_prefix_a()

    result = trusted_search(
        store,
        object(),
        "q",
        k=1,
        now=NOW,
        dependency_mode="enforce",
        security_policy=policy,
        access_context=context,
    )

    [hit] = result.hits
    assert "hr/z.md" not in result.reason
    assert hit.verdict == "dependency_invalidated"
    assert hit.invalidation is not None
    assert hit.invalidation.dependency == "<withheld>"
    assert hit.invalidation.path == ("a/x.md", "a/y.md", "<withheld>")


def test_a_readable_collided_successor_is_named(monkeypatch):
    """BUG-101 (+SEC-102): the policy decides on the root-relative file, not the absolute key.

    Red proof: against the pre-fix `_withhold_unauthorized_successors` (recall/trust.py, which
    asked the policy about the graph key `/two/README.md`), no rule matches the absolute path and
    the readable successor is reported as `<withheld>`, failing the assertion.
    """
    from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
    from recall.supersession import SupersessionGraph
    from recall.trust import trusted_search

    one = ScoredChunk(Chunk("r1", "/one/README.md", "old", {"file": "README.md"}), 0.9, NOW)
    two = ScoredChunk(Chunk("r2", "/two/README.md", "new", {"file": "README.md"}), 0.8, NOW)
    _fake_retriever(monkeypatch, [one, two])
    store = _CertifiedStore()
    graph = SupersessionGraph(
        edges={"/one/README.md": "/two/README.md"}, collided=frozenset({"README.md"})
    )
    store.supersession_graph = lambda: graph

    result = trusted_search(
        store,
        object(),
        "q",
        k=2,
        now=NOW,
        security_policy=SourceSecurityPolicy((SourceRule("README.md"),)),
        access_context=AccessContext("analyst", "tenant-a"),
    )

    by_id = {hit.chunk.id: hit for hit in result.hits}
    assert by_id["r1"].validity.superseded_by == "/two/README.md"


def test_a_transform_that_fills_an_empty_pool_is_still_judged_against_the_edges(monkeypatch):
    """STAKES-101: edges must be read for the pool `evaluate` judges, not only for retrieval's.

    Red proof: against the pre-fix `_trusted_search` (recall/trust.py, edges read only under
    `if result.hits:` before `pre_trust_transform`), the superseded `old.md` the transform adds is
    verdicted `ok`, failing the first assertion.
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustPolicy

    _fake_retriever(monkeypatch, [])
    store = SimpleNamespace(supersession=lambda: ({"old.md": "new.md"}, frozenset()))

    result = trusted_search(
        store,
        object(),
        "q",
        k=1,
        now=NOW,
        calibration=CAL,
        policy=TrustPolicy.development(),
        pre_trust_transform=lambda raw: _result([_hit("c1", "old.md", 0.9)]),
    )

    assert [(hit.chunk.id, hit.verdict) for hit in result.hits] == [("c1", "superseded")]
    assert result.abstained is True


def test_an_unrecognised_trust_mode_is_strict_and_says_so(caplog):
    """ENV-101: a misspelt RECALL_TRUST_MODE stays strict, and a warning names the variable.

    Red proof: against the pre-fix `TrustPolicy.from_env` (recall/trust_policy.py, silent
    fallback), no warning is logged, failing the second assertion.
    """
    import logging

    from recall.trust_policy import TrustPolicy

    with caplog.at_level(logging.WARNING, logger="recall.trust_policy"):
        policy = TrustPolicy.from_env({"RECALL_TRUST_MODE": "developmnet"})
    assert policy.strict is True
    assert any("RECALL_TRUST_MODE" in record.getMessage() for record in caplog.records)
    assert "developmnet" not in caplog.text
    # Control: the two real values and an empty value log nothing.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="recall.trust_policy"):
        TrustPolicy.from_env({"RECALL_TRUST_MODE": "Development"})
        TrustPolicy.from_env({"RECALL_TRUST_MODE": "strict"})
        TrustPolicy.from_env({})
    assert not caplog.records


def test_the_empty_sample_threshold_matches_the_uncalibrated_floor():
    """Nonbehavioral pin for CODE-108: `best_threshold([], [])` and the uncalibrated floor agree."""
    from recall.calibration import best_threshold
    from recall.guards import DEFAULT_GAP_THRESHOLD

    assert best_threshold([], []) == DEFAULT_GAP_THRESHOLD


def test_a_clean_gap_narrower_than_the_grid_keeps_the_threshold_above_the_ceiling():
    """BUG-103 (+NUM-102): rounding down must not land on or below the unanswerable ceiling.

    Red proof: against the pre-fix `best_threshold` (recall/calibration.py, a bare
    `_floor_to_grid((floor + ceiling) / 2)`), the first case returns 0.5, below the ceiling
    0.5001, failing the first assertion.
    """
    from recall.calibration import best_threshold

    narrow = best_threshold([0.5009] * 25, [0.5001] * 25)
    assert 0.5001 < narrow <= 0.5009
    assert best_threshold([0.022] * 20, [0.020] * 20) == pytest.approx(0.021)
    # The grid step above the ceiling (0.501), not the midpoint (0.5008).
    assert best_threshold([0.5012] * 25, [0.5004] * 25) == pytest.approx(0.501, abs=1e-12)
    # A ceiling one float below a grid point: the step above must not round back onto it.
    edge = 0.009999999999999998
    assert best_threshold([0.01] * 25, [edge] * 25) > edge
    # A gap of one float: only the floor itself lies above the ceiling.
    ceiling = 0.5
    floor = math.nextafter(ceiling, 1.0)
    assert best_threshold([floor] * 25, [ceiling] * 25) == floor
    # Controls: a gap wider than the grid still rounds the midpoint down, and overlapping
    # classes still bisect the overlap.
    assert best_threshold([0.8] * 25, [0.3] * 25) == pytest.approx(0.55)
    assert best_threshold([0.5] * 25, [0.6] * 25) == pytest.approx(0.55)
