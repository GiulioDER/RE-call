"""The BEAM arm's calibration must REACH the retrieval call, not merely be accepted.

Prior work: searched with ``docs_search(source_type="memory", ...)``.
[[project-recall-beam-bestconfig-blocked-2026-07-28]] measured the failure this guards: the shipped
0.50 cosine floor starves **14 of 60 questions (23.3%)** on voyage-4-large, against 7% on
text-embedding-3-small and 0% on bge-small. No prior test covered the wiring.

Why a test and not an inspection
--------------------------------
`trusted_search` resolves a calibration from the store through ``resolve_calibration`` — a method
that lives on the generation store and NOT on the ``PgVectorStore`` this arm hands it. So the
lookup misses silently, the status falls to ``"missing"``, and the run proceeds on the 0.50
constant with nothing in the output saying so. A calibration that is constructed, stored on the
system object, and then never passed down would look identical from the outside: the flag would be
accepted, the log line would print a threshold, and every question would still run on the default.

That is the shape these tests exist to make impossible. They assert on the kwargs the retrieval
call actually receives.
"""
from __future__ import annotations

import pytest

from benchmarks.beam.systems import BeamRecallSystem


class _Calibration:
    """Stands in for `recall.calibration.Calibration` — identity is what is asserted on."""

    threshold = 0.3
    certified = False


class _Hit:
    def __init__(self) -> None:
        self.chunk = type("C", (), {"text": "a memory", "metadata": {"file": "turn_0000.md"}})()


class _Result:
    abstained = False
    hits: list = []


class _FakeStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> "_FakeStore":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture()
def captured(monkeypatch):
    """Patch the two things `retrieve` imports, and capture research_search's kwargs."""
    seen: dict = {}

    def _fake_research_search(store, embedder, question, **kwargs):
        seen.update(kwargs)
        seen["question"] = question
        return _Result()

    import recall.eval._research_trust as trust_mod
    import recall.store as store_mod

    monkeypatch.setattr(store_mod, "PgVectorStore", _FakeStore)
    monkeypatch.setattr(trust_mod, "research_search", _fake_research_search)
    return seen


def _system(**kwargs) -> BeamRecallSystem:
    system = BeamRecallSystem("postgresql://unused/db", embedder_name="hashing", **kwargs)
    # `retrieve` refuses before ingest; the tenant is all it needs and ingest would want a corpus.
    system._tenant = "beam-1m-0"
    return system


def test_calibration_reaches_research_search(captured) -> None:
    """The assertion that makes the flag mean something."""
    cal = _Calibration()
    _system(calibration=cal).retrieve("where did they study?")

    assert "calibration" in captured, "research_search was called without a calibration kwarg"
    assert captured["calibration"] is cal, "a DIFFERENT calibration object arrived"


def test_no_calibration_passes_none_explicitly(captured) -> None:
    """The default path must still pass the kwarg, so the two states are distinguishable."""
    _system().retrieve("where did they study?")

    assert "calibration" in captured
    assert captured["calibration"] is None


def test_describe_records_which_gate_produced_the_numbers(captured) -> None:
    """An artifact whose false-abstention rate cannot be interpreted is not a result."""
    calibrated = _system(calibration=_Calibration()).describe()["calibration"]
    assert calibrated["source"] == "explicit"
    assert calibrated["threshold"] == 0.3
    assert calibrated["certified"] is False

    uncalibrated = _system().describe()["calibration"]
    assert uncalibrated["source"] == "none"
    assert uncalibrated["certified"] is False
    assert "0.50" in str(uncalibrated["threshold"])


def test_the_other_retrieval_knobs_still_arrive(captured) -> None:
    """Threading one kwarg through must not displace the ones already relied on."""
    _system(k=45, candidate_k=250).retrieve("q")

    assert captured["k"] == 45
    assert captured["candidate_k"] == 250
    assert captured["reranker"] is None
    assert captured["entailment"] is None
