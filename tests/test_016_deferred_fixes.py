"""Behaviour fixes of the CCA deferred pass before 0.16.0 (run 20261118T225356Z, parent 20261118T201647Z).

- D1 BUG-003: `recall calibration auto --json` keeps stdout one JSON document when the rescue rebuilds.
- D2 BUG-004: `recall generation rollback` with no previous generation refuses instead of rolling
  back with no target check.
- D3 CODE-012: a promote whose embedder cannot be built says it was not promoted, and why.
- D4 DAT-001: a cache that cannot erase does not replace the failure that stopped a build.
- D5 ENV-002: a server of several tenants under `skip` warns that it builds no artifact.
- D6 ENV-003: the project `.env` names every wizard tenant's rescue mode, `off` included.
- D7 PERF-002: the dashboard's Lite reader does not read every vector a second time to build a graph.
- D8 PERF-003: the dashboard's Lite View 1 is not assembled again for an unchanged file.
- D9 PERF-006: a damaged artifact is not hashed again on every search.
- D9b PERF-006: a read that failed once (a transient OSError) is not remembered as a damaged artifact.
- D10 PERF-007: a PostgreSQL forget creates no lock file for a generation that has no artifact.
- D11 SEC-002: an authenticated server's forget names no host path of a rescue root.
- D12 STAKES-005: a view store that cannot be erased does not keep the artifacts holding the same vectors.
- D13 BUG-005: under prefers-reduced-motion the graph is laid out, then held still.
- D14 DEPLOY-001: a frozen installer says the folder it runs from must stay.

Red proof, run on the test host 2026-11-19 against `01dcfe907` (the parent run's fixes, before
this pass) with this file added, each failing in its own assertion: D1 at "stdout is not one JSON
document" (`_rescue_after_publish`); D2 at "a rollback with no target to check was attempted" (the
rollback branch of `_cmd_generation`); D3 at "'not promoted' in" (`_rescue_before_serving`); D4 at
"the build's failure was replaced by OperationalError" (`GenerationManager.build`); D6 at "a tenant
pinned off was left to the shell's" (`_cli_rescue_env`); D7 at "every vector was read 2 times"
(`recall.lite.graph.graph_of`); D8 at "an unchanged file's import scan ran 2 times"
(`lite_db.corpus_graph`); D9 at "the damaged matrix was verified 3 times"
(`load_atomic_rescue_artifact`); D10 at "lock files for generations with nothing to remove"
(`generation_rescue.erase_forgotten`); D12 at "the artifact holding the forgotten vectors was kept"
(the same); D13 at "the layout never ran under reduced motion", for `rebuild` and `updateMotion`.
D5, D11 and D14 use new names (an ImportError or TypeError there, which is not proof), so each was
proved by mutating the fix: `_rescue_skip_warning` returning "" fails D5 at "'builds no missing'
in ''"; `_forget_memory_unlocked` joining the root paths whatever `name_rescue_roots` says fails D11
at the message naming `drive-not-mounted`; `_frozen_folder_note` returning "" fails D14 at its
second assertion. D9b was added after the differential review found the first PERF-006 cache
remembering a transient read failure (the loader turns an OSError into the same error type): with
that first version it fails at "a read that failed once kept refusing the artifact". Every test
passes with the fixes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.atomic_rescue import AtomicRescueArtifactError
from recall.lite.atomic import ROOT_ENV
from tests.test_atomic_rescue_default_on_cca import _publish
from tests.test_generation_cache_erasure_release import build_case  # noqa: F401  # a fixture


def _args(command: str, **values: object) -> argparse.Namespace:
    values.setdefault("generation_cmd", command)
    return argparse.Namespace(dsn="postgresql://x", tenant="p-docs", embedder="hashing", **values)


def test_calibration_auto_json_keeps_stdout_one_document(monkeypatch, capsys) -> None:
    """D1."""
    import recall.calibration_v2 as calibration
    import recall.drift as drift
    import recall.generation_rescue as rescue
    from recall.cli_commands.calibration_cmd import _cmd_calibration

    monkeypatch.setattr(calibration, "CalibrationRepository", lambda dsn, tenant: object())
    outcome = SimpleNamespace(
        action="recalibrated", calibration_id="cal_new", published=True, reason="drift",
        to_dict=lambda: {"action": "recalibrated"},
    )
    monkeypatch.setattr(drift, "auto_recalibrate", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(rescue, "after_publish", lambda *args, **kwargs: "atomic rescue: built for gen_live")

    _cmd_calibration(_args("auto", calibration_cmd="auto", generation="gen_live", publish=True, json=True))

    captured = capsys.readouterr()
    try:
        parsed = json.loads(captured.out)
    except ValueError:
        parsed = None
    assert parsed == {"action": "recalibrated"}, f"stdout is not one JSON document: {captured.out!r}"
    assert "atomic rescue: built" in captured.err


class _NoPrevious:
    calls: list[str] = []

    def __init__(self, dsn: str, tenant: str, **_: object) -> None:
        self.last_stated_facts = 0

    def previous_generation_id(self) -> None:
        return None

    def rollback(self, **kwargs: object) -> str:
        _NoPrevious.calls.append(f"rollback {kwargs}")
        return "gen_retired"


def test_a_rollback_with_no_previous_generation_refuses(monkeypatch) -> None:
    """D2."""
    import recall.generations as generations
    from recall.cli_commands.generation_cmd import _cmd_generation

    monkeypatch.setattr(generations, "GenerationManager", _NoPrevious)
    _NoPrevious.calls = []
    refused = ""
    try:
        _cmd_generation(_args("rollback"))
    except SystemExit as exc:
        refused = str(exc)
    assert _NoPrevious.calls == [], "a rollback with no target to check was attempted"
    assert "not rolled back" in refused


class _Promote:
    def __init__(self, dsn: str, tenant: str, **_: object) -> None:
        self.last_stated_facts = 0

    def promote(self, generation_id: str, **_: object) -> None:
        raise AssertionError("promoted although the rescue could not be built")


def test_a_promote_whose_embedder_cannot_be_built_says_it_was_not_promoted(monkeypatch) -> None:
    """D3."""
    import recall.cli_commands._shared as shared
    import recall.cli_commands.generation_cmd as generation_cmd
    import recall.generation_rescue as rescue
    import recall.generations as generations

    def no_embedder(name: str) -> object:
        raise SystemExit("embedder construction failed: the hosted API is unreachable")

    monkeypatch.setattr(shared, "_make_embedder", no_embedder)
    monkeypatch.setattr(generation_cmd, "_make_embedder", no_embedder)
    monkeypatch.setattr(generations, "GenerationManager", _Promote)
    monkeypatch.setattr(rescue, "ensure_for_promotion", lambda dsn, tenant, gid, make_embedder, **_: make_embedder())

    with pytest.raises(SystemExit) as refused:
        generation_cmd._cmd_generation(_args("promote", generation_id="gen_new", unsafe_development_promotion=False))
    message = str(refused.value)
    assert "not promoted" in message and "hosted API is unreachable" in message, message


def test_a_cache_that_cannot_erase_keeps_the_build_failure(build_case, monkeypatch) -> None:  # noqa: F811
    """D4."""
    from recall.cache import EmbeddingCache

    state = build_case

    def write(*args: object) -> int:
        raise RuntimeError("synthetic write failure")

    def erase(cache: object, digests: object) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(state.manager, "_write_source", write)
    monkeypatch.setattr(state.manager, "fail", lambda *args: None)
    monkeypatch.setattr(EmbeddingCache, "erase_digests", erase)
    with pytest.raises(Exception) as raised:
        state.build()
    assert str(raised.value) == "synthetic write failure", f"the build's failure was replaced by {raised.value!r}"


def test_a_server_of_several_tenants_warns_that_skip_builds_nothing() -> None:
    """D5."""
    from recall_mcp.server_lifespan import _rescue_skip_warning

    skip = {"RECALL_ATOMIC_RESCUE_WHEN_MISSING": "skip"}
    assert "builds no missing" in _rescue_skip_warning(None, skip)
    assert _rescue_skip_warning(object(), skip) == "", "a server of one store builds, so it does not warn"
    assert _rescue_skip_warning(None, {"RECALL_ATOMIC_RESCUE_WHEN_MISSING": "refuse"}) == ""


def test_the_project_env_names_every_wizard_tenant_rescue_mode() -> None:
    """D6."""
    from recall.wizard.headless import _cli_rescue_env
    from recall.wizard.wiring import ServerBlock

    blocks = (
        ServerBlock(name="p-docs", tenant="p-docs", env={"RECALL_ATOMIC_RESCUE_MODE": "active"}, rationale="r"),
        ServerBlock(name="p-memory", tenant="p-memory", env={"RECALL_ATOMIC_RESCUE_MODE": "off"}, rationale="r"),
    )
    assert _cli_rescue_env(blocks)["RECALL_ATOMIC_RESCUE_MODE_BY_TENANT"] == "p-docs:active,p-memory:off", (
        "a tenant pinned off was left to the shell's RECALL_ATOMIC_RESCUE_MODE in the CLI"
    )


def _lite_file(tmp_path: Path):  # noqa: ANN202
    from recall.lite import LiteStore
    from recall.types import Chunk

    store = LiteStore(tmp_path / "memory.db", dim=4)
    store.replace_sources(
        ["a.py", "b.md"],
        [
            Chunk(id="a1", source="a.py", text="import os\nprint('alpha bravo')", metadata={}),
            Chunk(id="b1", source="b.md", text="charlie delta echo", metadata={}),
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    return store


def test_the_lite_dashboard_reads_the_vectors_once(tmp_path: Path, monkeypatch) -> None:
    """D7. The write cleared the stored graph, so the reader builds one in memory."""
    from recall.dashboard import lite_db

    store = _lite_file(tmp_path)
    tenant, dsn = store.tenant, f"sqlite:///{store.path}"
    store.close()
    statements: list[str] = []

    class Traced(lite_db._ReadConnection):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.set_trace_callback(statements.append)

    monkeypatch.setattr(lite_db, "_ReadConnection", Traced)
    lite_db._CORPORA.clear()
    lite_db._GRAPHS.clear()
    graph = lite_db.corpus_graph(dsn, tenant)

    assert graph["nodes"], graph
    reads = [s for s in statements if "embedding" in s and "FROM chunks" in s]
    assert len(reads) == 1, f"every vector was read {len(reads)} times: {reads}"


def test_the_lite_dashboard_does_not_assemble_an_unchanged_graph_again(tmp_path: Path, monkeypatch) -> None:
    """D8."""
    from recall.dashboard import lite_db

    store = _lite_file(tmp_path)
    tenant, dsn = store.tenant, f"sqlite:///{store.path}"
    store.close()
    scans: list[int] = []
    real = lite_db._python_imports

    def counted(chunks, files):  # noqa: ANN001, ANN202
        scans.append(len(chunks))
        return real(chunks, files)

    monkeypatch.setattr(lite_db, "_python_imports", counted)
    lite_db._CORPORA.clear()
    first = lite_db.corpus_graph(dsn, tenant)
    second = lite_db.corpus_graph(dsn, tenant)

    assert first == second
    assert len(scans) == 1, f"an unchanged file's import scan ran {len(scans)} times"


def test_a_damaged_artifact_is_not_hashed_again_on_every_search(tmp_path: Path, monkeypatch) -> None:
    """D9. A repaired matrix is verified again (a control)."""
    import recall.atomic_rescue as atomic_rescue

    artifact = tmp_path / "gen_1"
    _publish(artifact, "cal_1")
    matrix = artifact / "matrix.npy"
    good = matrix.read_bytes()
    matrix.write_bytes(good[:-1] + bytes([good[-1] ^ 0xFF]))
    atomic_rescue.clear_atomic_rescue_artifact_cache()
    loads: list[Path] = []
    real = atomic_rescue._load_atomic_rescue_artifact

    def counted(path):  # noqa: ANN001, ANN202
        loads.append(path)
        return real(path)

    monkeypatch.setattr(atomic_rescue, "_load_atomic_rescue_artifact", counted)
    for _ in range(3):
        with pytest.raises(AtomicRescueArtifactError):
            atomic_rescue.load_atomic_rescue_artifact(artifact / "manifest.json")
    assert len(loads) == 1, f"the damaged matrix was verified {len(loads)} times"

    matrix.write_bytes(good)
    atomic_rescue.load_atomic_rescue_artifact(artifact / "manifest.json")
    assert len(loads) == 2, "a repaired matrix was refused from memory"
    atomic_rescue.clear_atomic_rescue_artifact_cache()


def test_a_read_that_failed_once_is_not_remembered(tmp_path: Path, monkeypatch) -> None:
    """D9b. A transient read failure (a scanner holding the matrix) is not a damaged artifact: the
    next load reads again, and under refuse a remembered one would refuse every search for a minute."""
    import recall.atomic_rescue as atomic_rescue

    artifact = tmp_path / "gen_1"
    _publish(artifact, "cal_1")
    atomic_rescue.clear_atomic_rescue_artifact_cache()
    real, calls = atomic_rescue._sha256, []

    def held_once(path):  # noqa: ANN001, ANN202
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("the file is being scanned")
        return real(path)

    monkeypatch.setattr(atomic_rescue, "_sha256", held_once)
    with pytest.raises(AtomicRescueArtifactError):
        atomic_rescue.load_atomic_rescue_artifact(artifact / "manifest.json")
    loaded = None
    try:
        loaded = atomic_rescue.load_atomic_rescue_artifact(artifact / "manifest.json")
    except AtomicRescueArtifactError:
        pass
    atomic_rescue.clear_atomic_rescue_artifact_cache()
    assert loaded is not None, "a read that failed once kept refusing the artifact"


def test_a_forget_creates_no_lock_for_a_generation_without_an_artifact(tmp_path: Path) -> None:
    """D10."""
    from recall.generation_rescue import erase_forgotten

    _publish(tmp_path / "gen_1", "cal_1")
    erased, removed = erase_forgotten(
        ["gen_1", "gen_2", "gen_3"], [], ["a.md"], env={ROOT_ENV: str(tmp_path)}
    )
    locks = sorted(path.name for path in tmp_path.glob(".rescue-*.lock"))
    assert removed == 1 and not (tmp_path / "gen_1").exists()
    assert locks == [".rescue-gen_1.lock"], f"lock files for generations with nothing to remove: {locks}"


def test_an_authenticated_forget_names_no_rescue_root_path(tmp_path: Path, monkeypatch) -> None:
    """D11. The local form still names it (a control)."""
    from recall_mcp.lifecycle import forget_memory

    monkeypatch.delenv(ROOT_ENV, raising=False)
    results = {}
    for named in (True, False):
        store = _lite_file(tmp_path / str(named))
        gone = tmp_path / str(named) / "drive-not-mounted" / "atomic"
        store.record_rescue_root(gone.resolve())
        try:
            results[named] = forget_memory(store, ["b.md"], name_rescue_roots=named).message
        finally:
            store.close()
    assert str(gone.parent.name) in results[True], results[True]
    assert "drive-not-mounted" not in results[False] and "1 root(s) the server log names" in results[False], results[False]


def test_a_view_store_that_cannot_be_erased_keeps_no_artifact(tmp_path: Path, monkeypatch) -> None:
    """D12."""
    from recall.atomic_view_store import AtomicViewStore
    from recall.generation_rescue import erase_forgotten
    from recall.lite.atomic import VIEW_STORE_NAME

    _publish(tmp_path / "gen_1", "cal_1")
    with AtomicViewStore(tmp_path / VIEW_STORE_NAME) as views:
        views.put("profile", "d" * 64, [[1.0, 0.0, 0.0, 0.0]])
        views.commit()

    def busy(self: object, digests: object) -> int:
        raise sqlite3.OperationalError("a reader blocked the checkpoint")

    monkeypatch.setattr(AtomicViewStore, "erase_digests", busy)
    try:
        erase_forgotten(["gen_1"], ["the forgotten text"], ["a.md"], env={ROOT_ENV: str(tmp_path)})
    except Exception as exc:  # noqa: BLE001  # the type is asserted below, after the artifact
        raised: Exception | None = exc
    else:
        raised = None
    assert not (tmp_path / "gen_1").exists(), "the artifact holding the forgotten vectors was kept"
    assert isinstance(raised, AtomicRescueArtifactError) and "views" in str(raised), raised


_REDUCED_MOTION = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const src = require('node:fs').readFileSync(process.argv[1], 'utf8');
function pick(start, end) {
  const i = src.indexOf(start), j = src.indexOf(end, i);
  assert(i >= 0 && j > i, 'missing ' + start);
  return src.slice(i, j);
}
const settle = src.includes('  function settle(') ? pick('  function settle(', '  function tick(') : '';
const reheat = pick('    // Reheat only as much', '    if (hops > 0 && selected)');
const motion = pick('  function updateMotion(', '  toggles.motion.addEventListener');
const failures = [];
for (const [name, body, alpha] of [['rebuild', reheat, 0], ['updateMotion', 'updateMotion();', 0.5]]) {
  const context = vm.createContext({
    alpha, ticks: 0, wasEmpty: true, placed: false, selected: null, flight: null, cameraAdjusted: false,
    reducedMotion: {matches: true}, toggles: {motion: {}}, wantsMotion: true, moving: () => false,
    cam: {}, needsDraw: false, fitDistance() {},
  });
  vm.runInContext('function tick() { ticks++; alpha *= 0.985; }\n' + settle + motion + body, context);
  if (context.ticks === 0) failures.push(name + ': the layout never ran under reduced motion');
  if (context.alpha !== 0) failures.push(name + ': still animating, alpha ' + context.alpha);
}
assert.deepEqual(failures, []);
"""


def test_reduced_motion_lays_the_graph_out_then_holds_it_still() -> None:
    """D13."""
    import shutil
    import subprocess

    from tests.test_dashboard_graph import STATIC

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute graph renderer regressions")
    done = subprocess.run(
        [node, "-e", _REDUCED_MOTION, str(STATIC / "graph.js")], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr


def test_a_frozen_installer_says_its_folder_must_stay(tmp_path: Path, monkeypatch) -> None:
    """D14. An install run from Python says nothing (a control)."""
    import sys

    from recall.desktop.install_ui import _frozen_folder_note

    monkeypatch.delattr(sys, "frozen", raising=False)
    assert _frozen_folder_note() == ""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bundle" / "recall-install.exe"))
    note = _frozen_folder_note()
    assert "bundle" in note and "Keep this folder" in note, note
