"""Two behaviours the design rests on that only another PROCESS can show (final review, Important 3):

- X1: a second process holding the build lock makes this server's builder skip, neither waiting
  nor building, as two servers of one tenant (Claude Code and Codex) do.
- X2: a recalibration builds while another process still has the old artifact's matrix mapped.
  Windows refuses to delete a mapped file, which is why artifacts are keyed by calibration: the
  new build goes beside the old one instead of replacing it.

Each runs a helper process with `sys.executable`, so it means most on Windows, where the lock is
`msvcrt.locking` and a mapped file cannot be removed; on Linux it still checks the same contract.

Red proof, run on Windows (2026-11-17), each a mutation, failing in its own assertion:
- X2 with `build_artifact` writing the old `<generation_id>` key again: "the recalibration could
  not build while another process mapped the old matrix" (Windows refused the replacement:
  "cannot be replaced while a process has its matrix open").
- X1 with `recall.generation_rescue.build_lock` yielding True whatever `_try_lock` returns: "a
  second builder built what another process held".
Both pass on Windows and on Linux with the code as it is.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from recall.atomic_rescue import clear_atomic_rescue_artifact_cache
from tests.test_atomic_rescue_default_on_cca import _Corpus, corpus  # noqa: F401  # fixture by name
from tests.test_calibration_carry_forward import _CarryEmbedder

ROOT = Path(__file__).resolve().parents[1]


def _helper(code: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready", "the helper process did not start"
    return process


def _release(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("go\n")
    process.stdin.flush()
    process.wait(10)


def test_a_lock_another_process_holds_makes_the_builder_skip(tmp_path: Path, monkeypatch) -> None:
    """X1."""
    from tests.test_rescue_builder import _Build, _builder, _Clock
    import recall_mcp.rescue_builder as rb

    monkeypatch.setattr(rb, "_artifact_serves", lambda *a, **k: False)
    root = tmp_path / "atomic"
    root.mkdir()
    lock = root / ".rescue-gen_1.lock"
    holder = _helper(
        "import sys\n"
        "from recall.lite.writer_lock import _try_lock\n"
        f"fh = open({str(lock)!r}, 'ab')\n"
        "assert _try_lock(fh)\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    try:
        build = _Build()
        builder = _builder(tmp_path, build, _Clock())
        started = time.monotonic()
        builder.after_retrieval()
        thread = builder._thread
        assert thread is not None
        thread.join(5)
        assert not thread.is_alive(), "the builder waited on a lock another process holds"
        assert time.monotonic() - started < 5
        assert build.calls == [], "a second builder built what another process held"
    finally:
        _release(holder)


def test_a_recalibration_builds_while_another_process_maps_the_old_matrix(corpus: _Corpus, tmp_path: Path) -> None:  # noqa: F811
    """X2."""
    from recall.generation_rescue import artifact_manifest, build_artifact

    embedder = _CarryEmbedder()
    assert build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path).startswith("built for gen_1")
    old = artifact_manifest(tmp_path, "gen_1", "cal_1")
    mapper = _helper(
        "import sys\n"
        "from recall.atomic_rescue import load_atomic_rescue_artifact\n"
        f"artifact = load_atomic_rescue_artifact({str(old)!r})\n"
        "float(artifact.matrix[0][0])\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    try:
        corpus.repository.calibration_id = "cal_2"  # type: ignore[attr-defined]
        clear_atomic_rescue_artifact_cache()
        try:
            report = build_artifact("postgresql://x", "t", "gen_1", embedder, tmp_path)
        except Exception as exc:  # noqa: BLE001  # the failure text is the red proof
            raise AssertionError(
                f"the recalibration could not build while another process mapped the old matrix: {exc}"
            ) from exc
        assert report.startswith("built for gen_1"), report
        assert old.is_file(), "the mapped artifact was removed from under its reader"
        assert artifact_manifest(tmp_path, "gen_1", "cal_2").is_file()
    finally:
        _release(mapper)
        clear_atomic_rescue_artifact_cache()
