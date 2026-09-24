"""The progress reporter `recall generation build` writes to stderr.

Why these exist: on 2026-09-24 a VPS2 project refresh embedded for 45 minutes and printed nothing,
which was indistinguishable from a hang. Each test below names the silence it prevents, and each
was watched to fail against a mutation of the line that implements it (recorded per test).
No database, no model: the reporter is pure formatting plus one thread.
"""

from __future__ import annotations

import io
import time

from recall.build_progress import BuildCounters, BuildPlan, BuildProgressReporter, format_plan


def _plan(**overrides: object) -> BuildPlan:
    values: dict[str, object] = {
        "total": 554,
        "reusable": 493,
        "pipeline_fingerprint": "new0fingerprint0",
        "generations_with_fingerprint": 1,
        "active_generation_id": "gen_active",
        "active_pipeline_fingerprint": "new0fingerprint0",
        "cache_enabled": True,
    }
    values.update(overrides)
    return BuildPlan(**values)  # type: ignore[arg-type]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_plan_states_reused_versus_embedded_before_anything_is_embedded() -> None:
    """Invariant: the first line says how many objects can be reused and how many must be embedded.

    Red proof: `tests/test_build_progress.py::test_the_plan_states_reused_versus_embedded_before_anything_is_embedded`
    failed on `assert "at least 61 to chunk and embed" in line` with `format_plan` mutated to
    `to_embed = plan.total` (the line then read "at least 554"), and passed restored.
    """
    line = format_plan(_plan())

    assert line.startswith("[build] plan: 554 objects")
    assert "up to 493 reusable" in line
    assert "at least 61 to chunk and embed" in line
    assert "FULL RE-EMBED" not in line


def test_a_moved_pipeline_fingerprint_is_announced_as_a_full_re_embed() -> None:
    """Invariant: when nothing can be reused because the fingerprint moved, the plan says so loudly.

    That is the expensive surprise: a routine refresh that silently becomes a full re-embed.
    Red proof: `tests/test_build_progress.py::test_a_moved_pipeline_fingerprint_is_announced_as_a_full_re_embed`
    failed on `assert "FULL RE-EMBED" in line` with `BuildPlan.full_re_embed` mutated to compare
    `generations_with_fingerprint != 0`, and passed restored.
    """
    line = format_plan(
        _plan(
            reusable=0,
            generations_with_fingerprint=0,
            active_pipeline_fingerprint="old0fingerprint0",
        )
    )

    assert "FULL RE-EMBED" in line
    assert "all 554 objects will be chunked and embedded" in line
    assert "gen_active (old0fingerpr)" in line


def test_a_progress_line_every_n_objects_and_never_a_carriage_return() -> None:
    """Invariant: one plain line per `every` objects, and none in between.

    Red proof: `tests/test_build_progress.py::test_a_progress_line_every_n_objects_and_never_a_carriage_return`
    failed on `assert [...] == [25, 50]` (it got `[]`) with `due_by_count` mutated to `False` in
    `BuildProgressReporter.update`, and passed restored.
    """
    stream = io.StringIO()
    clock = _Clock()
    with BuildProgressReporter(stream, every=25, interval=0, heartbeat=False, clock=clock) as r:
        for done in range(60):
            r.update(BuildCounters(total=60, done=done, reused=done), stage="reading")

    lines = stream.getvalue().splitlines()
    assert [int(line.split("object ")[1].split("/")[0]) for line in lines] == [25, 50]
    assert "\r" not in stream.getvalue()
    assert all(line.startswith("[build] object ") for line in lines)


def test_a_progress_line_at_least_every_interval_even_between_counts() -> None:
    """Invariant: a slow build still reports every `interval` seconds, not only every N objects.

    Red proof: `tests/test_build_progress.py::test_a_progress_line_at_least_every_interval_even_between_counts`
    failed on `assert len(lines) == 1` (it got 0) with `due_by_time` mutated to `False` in
    `BuildProgressReporter.update`, and passed restored.
    """
    stream = io.StringIO()
    clock = _Clock()
    with BuildProgressReporter(stream, every=0, interval=30, heartbeat=False, clock=clock) as r:
        clock.now = 10
        r.update(BuildCounters(total=554, done=3), stage="embedding", current="s3://b/a.md")
        assert stream.getvalue() == ""
        clock.now = 31
        r.update(BuildCounters(total=554, done=4), stage="embedding", current="s3://b/b.md")

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert "object 4/554" in lines[0]
    assert "elapsed 31s" in lines[0]
    assert "now embedding s3://b/b.md" in lines[0]


def test_the_heartbeat_speaks_while_one_embedding_call_holds_the_loop() -> None:
    """Invariant: a single long embedding call cannot silence the log.

    This is the observed failure itself: the build loop was inside one embed call, so no
    per-object line could fire. Red proof:
    `tests/test_build_progress.py::test_the_heartbeat_speaks_while_one_embedding_call_holds_the_loop`
    failed on `assert heard` with the guard in `BuildProgressReporter.__enter__` inverted to
    `if not self.heartbeat and ...` (no thread is created), and passed restored. Removing only
    `self._thread.start()` is NOT valid proof: it fails earlier, in `join()` inside `__exit__`.
    """
    stream = io.StringIO()
    with BuildProgressReporter(stream, every=0, interval=0.05) as r:
        r.update(BuildCounters(total=554, done=7), stage="embedding", current="s3://b/big.md")
        deadline = time.monotonic() + 5
        heard = False
        while time.monotonic() < deadline and not heard:
            time.sleep(0.02)
            heard = "still working: embedding s3://b/big.md for " in stream.getvalue()

    assert heard
    assert "object 7/554" in stream.getvalue()


def test_a_broken_stream_never_fails_the_build() -> None:
    """Invariant: progress observes the build; a closed or failing log cannot abort it.

    Red proof: `tests/test_build_progress.py::test_a_broken_stream_never_fails_the_build` failed
    with `OSError: disk full` raised out of `planned` when the `try` in
    `BuildProgressReporter._write` was removed, and passed restored.
    """

    class _Broken(io.StringIO):
        def write(self, s: str) -> int:
            raise OSError("disk full")

    with BuildProgressReporter(_Broken(), every=1, interval=0, heartbeat=False) as r:
        r.planned(_plan())
        r.update(BuildCounters(total=1, done=1), stage="reading")
        r.finished(BuildCounters(total=1, done=1))
