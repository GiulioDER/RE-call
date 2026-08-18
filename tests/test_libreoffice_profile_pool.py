"""The LibreOffice profile pool: reuse, isolation under concurrency, recovery, and the escape hatch.

These tests never launch LibreOffice. What is under test is which ``-env:UserInstallation`` each
conversion is handed, and that is decided entirely by `recall.extraction`, so a fake
``subprocess.run`` that records the argument and writes the output file exercises the real logic in
milliseconds. The measured timings that justify the pool live in
``docs/preregistrations/2026-08-18-libreoffice-profile-reuse.md`` and need a real binary.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from recall import extraction
from recall.extraction import DocumentExtractionError, extract_document


@pytest.fixture(autouse=True)
def _reset_pool() -> None:
    """The pool is process-wide module state, so a test that leaves it populated poisons the next."""
    extraction._discard_all_profiles()
    extraction._profile_pool = None
    yield
    extraction._discard_all_profiles()
    extraction._profile_pool = None


def _profile_of(command: list[str]) -> str:
    for argument in command:
        if argument.startswith("-env:UserInstallation="):
            return argument.split("=", 1)[1]
    raise AssertionError(f"no user installation in {command}")


def _fake_soffice(recorded: list[str], *, fail_times: int = 0) -> object:
    """Stand in for `subprocess.run`, recording the profile and writing the converted file."""
    state = {"failures": fail_times}
    lock = threading.Lock()

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        with lock:
            recorded.append(_profile_of(command))
            should_fail = state["failures"] > 0
            if should_fail:
                state["failures"] -= 1
        if should_fail:
            raise subprocess.CalledProcessError(1, command)
        outdir = Path(command[command.index("--outdir") + 1])
        source = Path(command[-1])
        (outdir / f"{source.stem}.txt").write_text("LEGACY-DOC-TEST", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


@pytest.fixture
def fake_libreoffice(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(extraction, "_libreoffice_executable", lambda: "soffice")
    monkeypatch.setattr(extraction.subprocess, "run", _fake_soffice(recorded))
    return recorded


def _doc(tmp_path: Path, name: str = "sample.doc") -> Path:
    path = tmp_path / name
    path.write_bytes(b"legacy bytes")
    return path


def test_serial_extractions_reuse_one_profile(tmp_path: Path, fake_libreoffice: list[str]) -> None:
    """The whole point of the pool: call two onwards must not pay a profile bootstrap.

    Measured with a real binary, this is the difference between about 6s and about 3s per call.
    """
    path = _doc(tmp_path)

    for _ in range(5):
        assert "LEGACY-DOC-TEST" in extract_document(path, path.read_bytes()).text

    assert len(fake_libreoffice) == 5
    assert len(set(fake_libreoffice)) == 1, fake_libreoffice


def test_concurrent_extractions_never_share_a_profile(
    tmp_path: Path, fake_libreoffice: list[str]
) -> None:
    """Two conversions against one profile do not both succeed, so overlap must mean distinct ones.

    Measured 2026-08-18 with LibreOffice 26.2.5.2: of two `soffice` processes started at the same
    moment against one profile, exactly one converts and the other exits 1 silently. The pool is
    what keeps that from happening, so this pins the property rather than the timing.
    """
    workers = 4
    monkeypatched_profiles: list[str] = []
    barrier = threading.Barrier(workers)
    hold = threading.Event()

    real_run = extraction.subprocess.run

    def run(command: list[str], **kwargs: object) -> object:
        # Hold every worker inside the conversion at once, so all four profiles are checked out
        # simultaneously. Without this the calls could serialise by luck and still look isolated.
        monkeypatched_profiles.append(_profile_of(command))
        barrier.wait(timeout=30)
        hold.wait(timeout=30)
        return real_run(command, **kwargs)

    extraction.subprocess.run = run  # type: ignore[assignment]
    try:
        paths = [_doc(tmp_path, f"sample{index}.doc") for index in range(workers)]
        errors: list[BaseException] = []

        def extract(path: Path) -> None:
            try:
                extract_document(path, path.read_bytes())
            except BaseException as exc:  # noqa: BLE001 - the assertion is that this stays empty
                errors.append(exc)

        threads = [threading.Thread(target=extract, args=(path,)) for path in paths]
        for thread in threads:
            thread.start()
        # Every worker is now parked inside its conversion holding a distinct profile.
        while len(monkeypatched_profiles) < workers:
            pass
        hold.set()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        extraction.subprocess.run = real_run  # type: ignore[assignment]

    assert errors == []
    assert len(monkeypatched_profiles) == workers
    assert len(set(monkeypatched_profiles)) == workers, monkeypatched_profiles


def test_pool_size_bounds_concurrent_profiles(
    tmp_path: Path, fake_libreoffice: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`RECALL_LIBREOFFICE_PROFILES` caps how many conversions may be in flight at once."""
    monkeypatch.setenv("RECALL_LIBREOFFICE_PROFILES", "2")
    extraction._profile_pool = None

    path = _doc(tmp_path)
    for _ in range(3):
        extract_document(path, path.read_bytes())

    assert extraction._profiles().qsize() == 2
    assert len(set(fake_libreoffice)) == 1  # serial work still reuses one, LIFO not FIFO


def test_failed_conversion_discards_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `soffice` that died can leave a lock behind, so a failure must not be inherited.

    Without this the next call reuses a poisoned profile and fails too, and so does every call after
    it: a transient fault would become a permanent one for the life of the process.
    """
    recorded: list[str] = []
    monkeypatch.setattr(extraction, "_libreoffice_executable", lambda: "soffice")
    monkeypatch.setattr(extraction.subprocess, "run", _fake_soffice(recorded, fail_times=1))

    path = _doc(tmp_path)
    with pytest.raises(DocumentExtractionError, match="could not extract"):
        extract_document(path, path.read_bytes())
    assert "LEGACY-DOC-TEST" in extract_document(path, path.read_bytes()).text

    assert len(recorded) == 2
    assert recorded[0] != recorded[1], "the failed conversion's profile was reused"
    assert not Path(recorded[0].removeprefix("file:///")).exists()


def test_escape_hatch_restores_a_profile_per_call(
    tmp_path: Path, fake_libreoffice: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`RECALL_LIBREOFFICE_SHARED_PROFILE=0` returns to the old behaviour, for a stuck deployment."""
    monkeypatch.setenv("RECALL_LIBREOFFICE_SHARED_PROFILE", "0")

    path = _doc(tmp_path)
    for _ in range(3):
        extract_document(path, path.read_bytes())

    assert len(set(fake_libreoffice)) == 3, fake_libreoffice
