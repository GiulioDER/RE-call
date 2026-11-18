"""The six minor findings the final review of #1339 (the atomic rescue on desktop installs) deferred.

No database: the builder's store is `tests.test_rescue_builder._Store`, and `build_artifact`'s
store, calibration repository and manager are the fakes of `tests.test_atomic_rescue_default_on_cca`.

Red proofs, each run on Linux and failing in its own assertion (the mutations are of this PR's code,
the baselines are master at `292127550`):
- D1 against `292127550`: "a second server re-ran a failed build inside the first one's backoff".
- D2 with `_backing_off` dropping its `<= now + backoff` bound: "a clock that went back held the
  build off"; its second half with `after_retrieval`'s debounce dropping its `0 <=`: "a clock that
  went back stopped the checks".
- D3 with `_artifact_serves` passing `calibration_id=None` in the lineage: "a build started for
  an artifact that serves" (the check through the real loader, which every other builder test
  replaces, so the public `artifact_serves` is exercised end to end).
- D4 against `292127550`: "the build scanned the corpus before refusing a damaged directory it
  cannot remove". Also checked once on Windows (2026-11-17, not committed) with a real mapping: a
  helper process holding the damaged matrix, no `rmtree` stub; `292127550` scanned and then
  refused (`assert [1] == []`), this change refuses before the scan.
- D5 against `292127550`: "removing another artifact made the serving one reload"; its second
  assertion with `evict_atomic_rescue_artifacts` made a no-op: "a removed artifact stayed in the
  cache".
- D6 against `292127550`: "a removed generation's lock file was left"; its held-lock assertion with
  `_remove_build_state` ignoring `try_lock`'s answer: "a lock another holder has was removed".
- D7 against `292127550`: "under skip the progress line says the rescue stays off".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import recall.atomic_rescue as atomic_rescue
from recall.atomic_rescue import AtomicRescueArtifactError, load_atomic_rescue_artifact
from tests.test_atomic_rescue_default_on_cca import _Corpus, _publish, corpus  # noqa: F401  # fixture by name
from tests.test_calibration_carry_forward import _CarryEmbedder
from tests.test_rescue_builder import _Build, _builder, _Clock, _settle, _Store


def _serving_store() -> _Store:
    """A builder store whose binding is the one the `corpus` fixture builds artifacts for."""

    class _Bound(_Store):
        def generation_binding(self) -> dict[str, str]:  # type: ignore[override]
            return {"generation_id": "gen_1", "pipeline_fingerprint": "pipe-1", "corpus_fingerprint": "corpus-1"}

    return _Bound()


def test_a_failed_build_backs_off_every_server(tmp_path: Path, monkeypatch) -> None:
    """D1 (Minor 1): the backoff was per process, so with N host servers a build that always fails
    ran N times per window. Invariant: a second server sharing the root waits out the first one's
    backoff, and the doubling is shared too."""
    import recall_mcp.rescue_builder as rb

    monkeypatch.setattr(rb, "_artifact_serves", lambda *a, **k: False)
    clock = _Clock()
    first_build, second_build = _Build(fail=True), _Build(fail=True)
    first, second = _builder(tmp_path, first_build, clock), _builder(tmp_path, second_build, clock)

    first.after_retrieval()
    _settle(first)
    clock.now += 5
    second.after_retrieval()
    _settle(second)
    assert second_build.calls == [], "a second server re-ran a failed build inside the first one's backoff"

    clock.now += 56  # 61 s after the first failure: the second server retries, and fails
    second.after_retrieval()
    _settle(second)
    assert second_build.calls == ["gen_1"]
    clock.now += 61  # inside the doubled 120 s
    first.after_retrieval()
    _settle(first)
    assert first_build.calls == ["gen_1"], "the doubling was not shared"


def test_a_backoff_further_off_than_itself_is_ignored(tmp_path: Path, monkeypatch) -> None:
    """D2: the shared backoff is on the wall clock, which can go back; a retry time further off
    than the backoff itself must not hold the build off for as long as the clock moved."""
    import recall_mcp.rescue_builder as rb
    from recall.generation_rescue import set_build_backoff

    monkeypatch.setattr(rb, "_artifact_serves", lambda *a, **k: False)
    clock, build = _Clock(), _Build()
    root = tmp_path / "atomic"
    root.mkdir()
    set_build_backoff(root, "gen_1", "cal_1", (clock.now + 86_400, 60.0))
    builder = _builder(tmp_path, build, clock)

    builder.after_retrieval()
    _settle(builder)

    assert build.calls == ["gen_1"], "a clock that went back held the build off"
    clock.now -= 3_600  # the clock goes back an hour: the next search still checks
    builder.after_retrieval()
    _settle(builder)
    assert build.calls == ["gen_1", "gen_1"], "a clock that went back stopped the checks"


def test_the_serving_check_reads_a_real_artifact(corpus: _Corpus, tmp_path: Path) -> None:  # noqa: F811
    """D3 (Minor 5): the builder asked `recall.generation_rescue._serves`, a private name, and every
    builder test replaced the check, so a rename would have left each server rebuilding, or never
    building, with nothing but a logged warning. Invariant: an artifact `build_artifact` wrote is
    seen as serving through the public `artifact_serves`, and no build starts."""
    from recall.generation_rescue import build_artifact

    root = tmp_path / "atomic"
    assert build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), root).startswith("built for gen_1")
    build = _Build()
    builder = _builder(tmp_path, build, _Clock(), store=_serving_store())

    builder.after_retrieval()
    _settle(builder)

    assert build.calls == [], "a build started for an artifact that serves"


def test_a_damaged_directory_that_stays_refuses_before_the_scan(corpus: _Corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    """D4 (Minor 2): a directory of the very key being built that serves nothing was removed only
    after the full scan and assembly, and on Windows the removal fails while another process maps
    its matrix, so the build refused having done all of that work. The removal failing is simulated
    by an `rmtree` that removes nothing. Invariant: it refuses before scanning."""
    import recall.generation_rescue as rescue

    _publish(tmp_path / "gen_1.cal_1", "cal_1")  # names generation `gen_1.cal_1`, so it serves nothing here
    scanned: list[int] = []
    real = rescue.rescue_parents
    monkeypatch.setattr(rescue, "rescue_parents", lambda chunks: (scanned.append(1), real(chunks))[1])
    monkeypatch.setattr(rescue.shutil, "rmtree", lambda path, **kw: None)

    with pytest.raises(AtomicRescueArtifactError, match="cannot be replaced"):
        rescue.build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)

    assert scanned == [], "the build scanned the corpus before refusing a damaged directory it cannot remove"


class _Manager:
    """Generations `prune_superseded` reads: one active, one failed (so removed)."""

    def __init__(self, dsn: str, tenant: str, **_: object) -> None:
        pass

    def rescue_prune_snapshot(self) -> tuple[tuple[Any, ...], set[str], set[str]]:
        from recall.lineage import GenerationState

        records = (
            SimpleNamespace(generation_id="gen_active", state=GenerationState.ACTIVE, retired_at=None),
            SimpleNamespace(generation_id="gen_failed", state=GenerationState.FAILED, retired_at=None),
        )
        return records, {"gen_active"}, set()


def test_a_prune_keeps_the_serving_artifact_cached(tmp_path: Path, monkeypatch) -> None:
    """D5 (Minor 4): each prune cleared this process's whole artifact cache, so the next search
    reloaded and re-hashed the matrix still serving. Invariant: only the removed directories leave
    the cache."""
    import recall.generations as generations
    from recall.generation_rescue import prune_superseded

    monkeypatch.setattr(generations, "GenerationManager", _Manager)
    _publish(tmp_path / "gen_active.cal_1", "cal_1")
    _publish(tmp_path / "gen_failed.cal_1", "cal_1")
    loads: list[Path] = []
    real = atomic_rescue._load_atomic_rescue_artifact
    monkeypatch.setattr(atomic_rescue, "_load_atomic_rescue_artifact", lambda path: (loads.append(path), real(path))[1])
    atomic_rescue.clear_atomic_rescue_artifact_cache()
    try:
        load_atomic_rescue_artifact(tmp_path / "gen_active.cal_1" / "manifest.json")
        load_atomic_rescue_artifact(tmp_path / "gen_failed.cal_1" / "manifest.json")

        assert prune_superseded("postgresql://x", "t", tmp_path) == 1
        load_atomic_rescue_artifact(tmp_path / "gen_active.cal_1" / "manifest.json")

        assert len(loads) == 2, "removing another artifact made the serving one reload"
        doomed = (tmp_path / "gen_failed.cal_1").resolve()
        assert not [path for path in atomic_rescue._ARTIFACT_CACHE if path.is_relative_to(doomed)], (
            "a removed artifact stayed in the cache"
        )
    finally:
        atomic_rescue.clear_atomic_rescue_artifact_cache()


def test_dat_001_prune_keeps_lock_identity_and_clears_idle_backoff(tmp_path: Path, monkeypatch) -> None:
    """Red on 31e6841d: prune split lock contention on POSIX or changed the idle file ID on Windows."""
    import os

    import recall.generations as generations
    from recall.generation_rescue import build_lock, prune_superseded
    from recall.lite.writer_lock import try_lock

    monkeypatch.setattr(generations, "GenerationManager", _Manager)
    for generation_id in ("gen_active", "gen_failed"):
        with build_lock(tmp_path, generation_id):
            pass
    failed_lock = tmp_path / ".rescue-gen_failed.lock"
    backoff = tmp_path / ".rescue-gen_failed.backoff.json"
    backoff.write_text("{}", encoding="utf-8")
    with failed_lock.open("ab") as delayed:
        original = os.fstat(delayed.fileno())
        prune_superseded("postgresql://x", "t", tmp_path)
        with build_lock(tmp_path, "gen_failed") as held:
            assert held
            assert not try_lock(delayed), "the delayed and later openers acquired separate locks"
        assert try_lock(delayed)
        with build_lock(tmp_path, "gen_failed") as held:
            assert not held, "the later opener did not contend with the delayed descriptor"

    assert not backoff.exists(), "a removed generation's backoff was left"
    assert (tmp_path / ".rescue-gen_active.lock").exists(), "a kept generation's lock file was removed"

    # Windows normally denies unlink while the delayed descriptor is open. Check the file
    # identity after an idle prune as well, when Windows lets the old implementation unlink it.
    prune_superseded("postgresql://x", "t", tmp_path)
    with build_lock(tmp_path, "gen_failed") as held:
        assert held
        assert os.path.samestat(original, failed_lock.stat()), "prune replaced the generation lock's file identity"

    backoff.write_text("{}", encoding="utf-8")
    with build_lock(tmp_path, "gen_failed") as held:
        assert held
        prune_superseded("postgresql://x", "t", tmp_path)
        assert backoff.is_file(), "prune cleared the backoff of a generation with a live holder"


def test_the_wizard_says_what_happens_to_an_unbuilt_rescue_under_skip(tmp_path: Path, monkeypatch) -> None:
    """D7 (Minor 3): a reused corpus whose rescue rebuild fails was reported as "so it stays off",
    while under RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip it is registered active and its server builds
    it. Invariant: the line says so under skip, and still says it stays off under refuse."""
    import recall.wizard.headless as headless
    from recall.lite.atomic import ROOT_ENV
    from recall.wizard.headless import _RealServices, run_headless
    from tests.test_wizard_state import _config, _CountingSpy

    config = _config(tmp_path)
    state_path = tmp_path / "s.json"
    run_headless(config, services=_CountingSpy(), state_path=state_path)

    class _Unbuildable(_CountingSpy, _RealServices):
        def __init__(self) -> None:
            _CountingSpy.__init__(self)
            self.config = config
            self._embedder = None
            self.atomic_env = {}

        def configure_fact_boundary(self, *args: object, **kwargs: object) -> None:
            pass

        def has_legacy_rows(self, *args: object) -> bool:
            return False

        def smoke(self, block: Any) -> Any:
            raise RuntimeError("no database in this test")

        def _rescue_builder(self, tenant: str):  # noqa: ANN202
            def refuse(generation_id: str) -> str:
                raise ValueError("the embedding provider refused the request")

            return refuse

    def lines(when_missing: str) -> list[str]:
        env = {"RECALL_ATOMIC_RESCUE_MODE": "active", ROOT_ENV: str(tmp_path / "atomic"),
               "RECALL_ATOMIC_RESCUE_WHEN_MISSING": when_missing}
        monkeypatch.setattr(headless, "_atomic_rescue_env", lambda config: (dict(env), ""))
        said: list[str] = []
        run_headless(config, services=_Unbuildable(), state_path=state_path, progress=said.append)
        return [line for line in said if "not rebuilt" in line]

    skipped = lines("skip")
    assert skipped, "no line reported the rescue that was not rebuilt"
    assert not [line for line in skipped if "stays off" in line], (
        f"under skip the progress line says the rescue stays off: {skipped}"
    )
    assert all("until its server builds it" in line for line in skipped), skipped
    refused = lines("refuse")
    assert refused and all("stays off" in line for line in refused), refused
