"""The host MCP server builds a missing atomic rescue artifact in the background
(`recall_mcp.rescue_builder`, docs/DESKTOP_ATOMIC_RESCUE_DESIGN.md).

No database: the store is a `GenerationStore` subclass that only answers its binding and its
calibration, the build is a recorder, and the check of whether an artifact serves is replaced, so
each test sets exactly the state it is about.

Red proofs, each run on Linux and failing in its own assertion:
- B1, B3, B4 and B5 against the module with an empty `after_retrieval` and `stop` (before the
  implementation): "no build for a missing artifact", "assert [] == ['gen_1']" (B3 and B5) and
  "a failed build ran again inside its backoff" read as the first build never starting.
- B2 with `_artifact_serves` ignored in `after_retrieval`: "a build started for an artifact that
  serves".
- B6 with `from_env` ignoring RECALL_ATOMIC_RESCUE_WHEN_MISSING: "a builder for refuse".
- B7 with the `after_retrieval()` call removed from `_run_retrieval_tool`: "the search did not
  ask the builder".
- B6, its prune half, with the `prune` default removed from `from_env`: "a host build would never
  remove superseded artifacts".
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.calibration_v2 import CalibrationStatus
from recall.generation_store import GenerationStore
from recall_mcp import rescue_builder as rb
from tests.test_calibration_carry_forward import _CarryEmbedder
from tests.test_lite_mcp import lite_env  # noqa: F401  # the fixture is used by name


class _Store(GenerationStore):
    """Only what the builder asks: the tenant, the DSN, the binding and the calibration."""

    def __init__(self, generation_id: str = "gen_1", calibration_id: str = "cal_1") -> None:  # noqa: D107
        self._tenant = "t"
        self._dsn = "postgresql://x"
        self._gen, self._cal = generation_id, calibration_id

    def generation_binding(self) -> dict[str, str]:  # type: ignore[override]
        return {"generation_id": self._gen, "pipeline_fingerprint": "p", "corpus_fingerprint": "c"}

    def resolve_calibration(self):  # type: ignore[override]  # noqa: ANN201
        return SimpleNamespace(
            status=CalibrationStatus.CERTIFIED, artifact=SimpleNamespace(calibration_id=self._cal)
        )


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Build:
    """Records each build; `gate` holds it open, `fail` makes it raise."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail = fail

    def __call__(self, dsn: str, tenant: str, generation_id: str, embedder: object, root: Path) -> str:
        self.calls.append(generation_id)
        self.gate.wait(5)
        if self.fail:
            raise RuntimeError("embedder down")
        return f"built for {generation_id}"


def _builder(tmp_path: Path, build: _Build, clock: _Clock, store: _Store | None = None) -> rb.RescueBuilder:
    return rb.RescueBuilder(
        store or _Store(), _CarryEmbedder(), tmp_path / "atomic", build=build, prune=lambda *_: 0, clock=clock
    )


@pytest.fixture(autouse=True)
def _missing(monkeypatch) -> None:
    monkeypatch.setattr(rb, "_artifact_serves", lambda *a, **k: False)


def _settle(builder: rb.RescueBuilder) -> None:
    thread = builder._thread
    if thread is not None:
        thread.join(5)


def test_a_missing_artifact_starts_one_build(tmp_path: Path) -> None:
    """B1: one build per missing key, however many searches arrive while it runs."""
    build, clock = _Build(), _Clock()
    build.gate.clear()
    builder = _builder(tmp_path, build, clock)

    builder.after_retrieval()
    clock.now += 31
    builder.after_retrieval()
    build.gate.set()
    _settle(builder)

    assert build.calls == ["gen_1"], f"no build for a missing artifact, or more than one: {build.calls}"


def test_a_present_artifact_starts_none(tmp_path: Path, monkeypatch) -> None:
    """B2."""
    monkeypatch.setattr(rb, "_artifact_serves", lambda *a, **k: True)
    build = _Build()
    builder = _builder(tmp_path, build, _Clock())

    builder.after_retrieval()
    _settle(builder)

    assert build.calls == [], "a build started for an artifact that serves"


def test_a_second_builder_skips_a_build_another_holds(tmp_path: Path) -> None:
    """B3: two servers of one tenant (Claude Code and Codex) see the same missing artifact."""
    first_build, second_build = _Build(), _Build()
    first_build.gate.clear()
    first, second = _builder(tmp_path, first_build, _Clock()), _builder(tmp_path, second_build, _Clock())

    first.after_retrieval()
    deadline = time.monotonic() + 5
    while not first_build.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    second.after_retrieval()
    _settle(second)
    first_build.gate.set()
    _settle(first)

    assert first_build.calls == ["gen_1"]
    assert second_build.calls == [], "a second builder built what another held"


def test_a_failed_build_backs_off(tmp_path: Path) -> None:
    """B4: 60 s after the first failure, then doubling, so a broken build does not run per search."""
    build, clock = _Build(fail=True), _Clock()
    builder = _builder(tmp_path, build, clock)

    builder.after_retrieval()
    _settle(builder)
    clock.now += 31
    builder.after_retrieval()
    _settle(builder)
    assert build.calls == ["gen_1"], "a failed build ran again inside its backoff"
    clock.now += 30  # 61 s after the failure
    builder.after_retrieval()
    _settle(builder)
    assert build.calls == ["gen_1", "gen_1"]
    clock.now += 61  # inside the doubled 120 s
    builder.after_retrieval()
    _settle(builder)
    assert build.calls == ["gen_1", "gen_1"], "the backoff did not double"


def test_stop_waits_for_a_running_build(tmp_path: Path) -> None:
    """B5: a server shutting down lets the build finish, then starts no other."""
    build = _Build()
    build.gate.clear()
    builder = _builder(tmp_path, build, _Clock())
    builder.after_retrieval()
    threading.Timer(0.3, build.gate.set).start()

    builder.stop(timeout_s=5)

    assert builder._thread is None or not builder._thread.is_alive(), "stop returned while the build ran"
    builder.after_retrieval()
    assert build.calls == ["gen_1"]


@pytest.mark.parametrize(
    "env",
    [
        {"RECALL_ATOMIC_RESCUE_MODE": "active", "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "refuse"},
        {"RECALL_ATOMIC_RESCUE_MODE": "active"},
        {"RECALL_ATOMIC_RESCUE_MODE": "off", "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip"},
        {"RECALL_ATOMIC_RESCUE_MODE": "active", "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip", "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": ""},
    ],
)
def test_from_env_is_none_off_the_desktop_path(tmp_path: Path, env: dict[str, str]) -> None:
    """B6: only an active rescue with `skip` and a root gets a builder."""
    values = {"RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path), **env}

    assert rb.RescueBuilder.from_env(_Store(), _CarryEmbedder(), values) is None, "a builder for refuse"


def test_from_env_builds_for_the_desktop_path(tmp_path: Path) -> None:
    """B6, the other side, and a Lite store never gets one."""
    from recall.lite import LiteStore

    values = {
        "RECALL_ATOMIC_RESCUE_MODE": "active",
        "RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip",
        "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT": str(tmp_path),
    }

    builder = rb.RescueBuilder.from_env(_Store(), _CarryEmbedder(), values)
    assert isinstance(builder, rb.RescueBuilder)
    from recall.generation_rescue import prune_superseded

    assert builder._prune is prune_superseded, "a host build would never remove superseded artifacts"
    with LiteStore(tmp_path / "memory.db", dim=8) as lite:
        assert rb.RescueBuilder.from_env(lite, _CarryEmbedder(), values) is None


def test_the_server_checks_after_a_search_and_stops_at_shutdown(lite_env, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001, F811
    """B7: the lifespan creates the builder, `recall_search` asks it after retrieving, and the
    shutdown stops it before the store closes. Red proof: with the `after_retrieval()` call removed
    from `_run_retrieval_tool`, "the search did not ask the builder"."""
    import json

    from tests.test_lite_calibration import _write_memos
    from tests.test_lite_mcp import QUESTION, _serve

    events: list[str] = []

    class _Recorder:
        def after_retrieval(self) -> None:
            events.append("checked")

        def stop(self, timeout_s: float = 10.0) -> None:
            events.append("stopped")

    monkeypatch.setattr(rb.RescueBuilder, "from_env", classmethod(lambda cls, *a, **k: _Recorder()))
    _write_memos(tmp_path / "mem", 48)

    async def body(state, call):  # noqa: ANN001, ANN202
        json.loads(await call("recall_index", path=str(tmp_path / "mem")))
        await call("recall_search", query=QUESTION)
        assert events == ["checked"], f"the search did not ask the builder: {events}"

    _serve(lite_env, body)
    assert events == ["checked", "stopped"], f"the shutdown did not stop the builder: {events}"
