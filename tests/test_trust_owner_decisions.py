"""Regression tests for the owner's 2026-11-01 decisions on the trust hunt residue.

ENV-002 (development under production warned about), ENV-105 (the dependency mode validated at
startup), the trust gate's non-finite dense score, and DOC-003 (a projection read that raises is
an outage). No database and no network: stores and the retriever are fakes, as in
`tests/test_trust_hunt_fixes.py`.

Red proof for every behaviour test below was taken on the test host against the parent commit
`d56661c71` (the pre-fix production code, this file unchanged); each docstring names the line and
the assertion that failed there.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.calibration import Calibration
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
FRESH = StalenessReport(
    stale=False, newest_indexed_at=NOW, age=timedelta(0), max_age=timedelta(days=2)
)
CAL = Calibration(embedder="test", threshold=0.5, scale=0.05)
DEV_UNDER_PRODUCTION = {"RECALL_TRUST_MODE": "development", "RECALL_ENV": "production"}


def _hit(cid: str, file: str, score: float) -> ScoredChunk:
    chunk = Chunk(id=cid, source=file, text=f"text of {file}", metadata={"file": file, "ord": 0})
    return ScoredChunk(chunk=chunk, score=score, indexed_at=NOW)


def _clear_trust_env(monkeypatch) -> None:
    for name in ("RECALL_TRUST_MODE", "RECALL_ENV", "RECALL_DEPENDENCY_INVALIDATION"):
        monkeypatch.delenv(name, raising=False)


# ENV-002


def _production_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "recall.trust_policy" and "RECALL_ENV=production" in record.getMessage()
    ]


def test_development_under_production_is_warned_about_and_still_honoured(caplog):
    """`TrustPolicy.from_env` warns, naming both variables, for the pair and for neither alone.

    The owner chose warn only (2026-11-01): `recall setup` writes the pair on purpose, because
    `RECALL_ENV=production` also selects `GenerationStore`. Red proof: against the pre-fix
    `TrustPolicy.from_env` (recall/trust_policy.py, no production check at all), nothing is logged
    for the pair and the first assertion fails (`[] == 1` warnings).
    """
    import logging

    from recall.trust_policy import TrustMode, TrustPolicy

    with caplog.at_level(logging.WARNING, logger="recall.trust_policy"):
        policy = TrustPolicy.from_env(dict(DEV_UNDER_PRODUCTION))
    warnings = _production_warnings(caplog)
    assert len(warnings) == 1
    assert "RECALL_TRUST_MODE=development" in warnings[0]
    # No behaviour change: the relaxed gate is still what the operator gets.
    assert policy.mode is TrustMode.DEVELOPMENT

    # Padded the way `env_is_production` normalises it, as a systemd EnvironmentFile pads it.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="recall.trust_policy"):
        TrustPolicy.from_env({"RECALL_TRUST_MODE": " Development ", "RECALL_ENV": "Production "})
    assert len(_production_warnings(caplog)) == 1

    # Controls: either variable alone, and strict in production, log nothing.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="recall.trust_policy"):
        assert TrustPolicy.from_env({"RECALL_TRUST_MODE": "development"}).strict is False
        assert TrustPolicy.from_env({"RECALL_ENV": "production"}).strict is True
        assert TrustPolicy.from_env({"RECALL_TRUST_MODE": "strict", "RECALL_ENV": "production"}).strict
    assert not caplog.records


def test_the_server_warns_about_development_under_production_at_startup(caplog):
    """The MCP settings snapshot, built once at server start, warns and keeps the relaxed policy.

    Red proof: against pre-fix `TrustPolicy.from_env`, `Settings.from_env` logs nothing for the
    pair and the first assertion fails (`[] == 1` warnings).
    """
    import logging

    from recall_mcp.settings import Settings

    with caplog.at_level(logging.WARNING, logger="recall.trust_policy"):
        settings = Settings.from_env(dict(DEV_UNDER_PRODUCTION))
    assert len(_production_warnings(caplog)) == 1
    assert settings.trust_policy.strict is False


# ENV-105


def test_an_unknown_dependency_mode_is_refused_at_startup(monkeypatch, capsys):
    """Both the server snapshot and a non-searching CLI command refuse `sometimes`.

    Red proof: against pre-fix `_validate_runtime_options` (recall_mcp/settings.py) and `_main`
    (recall/cli.py), neither reads RECALL_DEPENDENCY_INVALIDATION, so the first
    `pytest.raises(ValueError)` fails with "DID NOT RAISE".
    """
    from recall import cli
    from recall_mcp.settings import Settings

    with pytest.raises(ValueError, match="RECALL_DEPENDENCY_INVALIDATION"):
        Settings.from_env({"RECALL_DEPENDENCY_INVALIDATION": "sometimes"})
    # Control: the two modes, normalised as the search path normalises them, start.
    Settings.from_env({"RECALL_DEPENDENCY_INVALIDATION": " Enforce"})
    Settings.from_env({"RECALL_DEPENDENCY_INVALIDATION": "off"})

    _clear_trust_env(monkeypatch)
    monkeypatch.setenv("RECALL_DEPENDENCY_INVALIDATION", "sometimes")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["schema", "grants", "--role", "recall_app"])
    assert "RECALL_DEPENDENCY_INVALIDATION" in str(excinfo.value.code)
    assert capsys.readouterr().out == ""


# Owner decision 11: trust_gate._verdict


@pytest.mark.parametrize("score", [math.nan, math.inf])
def test_a_non_finite_dense_score_is_low_confidence(score):
    """The verdict itself treats a non-finite cosine as below the threshold.

    `trusted_search` already drops NaN dense hits before `evaluate`; this pins the gate for any
    other caller of `evaluate`. Red proof: against the pre-fix `_verdict` (recall/trust_gate.py,
    `hit.score < threshold` alone), NaN compares False and +inf is not below 0.5, so both are
    `ok` and the first assertion fails for both parameters.
    """
    from recall.trust_gate import _verdict

    assert _verdict(_hit("c1", "a.md", score), {}, 0.5, NOW)[0] == "low_confidence"
    # Control: a finite score either side of the threshold keeps its verdict.
    assert _verdict(_hit("c2", "b.md", 0.9), {}, 0.5, NOW)[0] == "ok"
    assert _verdict(_hit("c3", "c.md", 0.1), {}, 0.5, NOW)[0] == "low_confidence"


# DOC-003


class _EnforcingStore:
    """A certified generation store with dependency invalidation enforced."""

    tenant = "tenant-a"

    def __init__(self, projection) -> None:
        self._projection = projection

    def generation_binding(self) -> dict[str, str]:
        return {"tenant_id": "tenant-a", "generation_id": "g1"}

    def resolve_calibration(self):
        from recall.calibration_v2 import CalibrationStatus

        artifact = SimpleNamespace(runtime=CAL, calibration_id="cal-1", query_set_digest="qs")
        return SimpleNamespace(status=CalibrationStatus.CERTIFIED, artifact=artifact)

    def dependency_invalidation_mode(self) -> str:
        return "enforce"

    def dependency_projection(self, *, as_of, known_as_of):
        del as_of, known_as_of
        return self._projection()

    def supersession(self):
        return {}, frozenset()


def _fake_retriever(monkeypatch) -> None:
    from recall import trust

    class _FakeRetriever:
        def __init__(self, store, embedder, **kwargs):
            del store, embedder, kwargs

        def search(self, query, k, source=None, scope=None, **kwargs):
            del query, k, source, scope, kwargs
            return RetrievalResult(
                query="q", hits=[_hit("c1", "a.md", 0.9)], gap_warning=False, staleness=FRESH
            )

    monkeypatch.setattr(trust, "HybridRetriever", _FakeRetriever)


def test_a_projection_read_that_raises_is_unavailable_under_every_policy(monkeypatch):
    """An exception reading the projection refuses as DEPENDENCY_UNAVAILABLE, development too.

    Red proof: against the pre-fix `_trusted_search` (recall/trust.py, the projection `except`
    set `DEPENDENCY_GRAPH_NOT_READY`), the development search degrades instead of refusing and
    `pytest.raises(TrustRefusal)` fails with "DID NOT RAISE".
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal

    _fake_retriever(monkeypatch)

    def _down():
        raise ConnectionError("projection table unreachable")

    for policy in (TrustPolicy.development(), TrustPolicy.strict_policy()):
        with pytest.raises(TrustRefusal) as excinfo:
            trusted_search(_EnforcingStore(_down), object(), "q", k=1, now=NOW, policy=policy)
        assert excinfo.value.code is TrustFailureCode.DEPENDENCY_UNAVAILABLE
        assert excinfo.value.generation_id == "g1"
        assert excinfo.value.calibration_id == "cal-1"


def test_a_missing_projection_stays_graph_not_ready(monkeypatch):
    """Control for the outage path: an absent projection is still DEPENDENCY_GRAPH_NOT_READY.

    Nonbehavioral control, exempt from red proof: it pins the half of DOC-003 the fix leaves
    unchanged, so the pre-fix code passes it too, by design.
    """
    from recall.trust import trusted_search
    from recall.trust_policy import TrustFailureCode, TrustPolicy, TrustRefusal

    _fake_retriever(monkeypatch)
    store = _EnforcingStore(lambda: None)

    degraded = trusted_search(
        store, object(), "q", k=1, now=NOW, policy=TrustPolicy.development()
    )
    assert degraded.trust_state == "degraded"
    assert degraded.failure_code == TrustFailureCode.DEPENDENCY_GRAPH_NOT_READY.value
    with pytest.raises(TrustRefusal) as excinfo:
        trusted_search(store, object(), "q", k=1, now=NOW, policy=TrustPolicy.strict_policy())
    assert excinfo.value.code is TrustFailureCode.DEPENDENCY_GRAPH_NOT_READY


def test_the_calibration_doc_lists_every_failure_code():
    """docs/CALIBRATION.md's failure code table has one row per `TrustFailureCode`, and says so.

    Red proof: against the pre-fix docs/CALIBRATION.md ("The six failure codes", no
    `DEPENDENCY_GRAPH_NOT_READY` row), the row set misses one code and the first assertion fails.
    """
    from recall.trust_policy import TrustFailureCode

    doc = (Path(__file__).resolve().parents[1] / "docs" / "CALIBRATION.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^### The (\w+) failure codes\n(.*?)^### ", doc, flags=re.M | re.S)
    assert match is not None
    rows = set(re.findall(r"^\| `([A-Z_]+)` \|", match.group(2), flags=re.MULTILINE))
    assert rows == {code.value for code in TrustFailureCode}
    words = {6: "six", 7: "seven", 8: "eight", 9: "nine"}
    assert match.group(1) == words[len(TrustFailureCode)]
