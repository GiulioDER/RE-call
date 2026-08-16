"""What the installer must know about the machine before it offers the user a choice.

`recall.setup.probe_hardware` already reports CPU, GPU, CUDA, free disk and internet. It does not
report RAM, does not know whether Docker exists or is running, and cannot tell a python.org
interpreter from the Microsoft Store build whose path sandboxing makes indexing silently find
nothing. An installer that cannot see those cannot decide what to offer.

The governing rule, inherited from `_probe_gpu`: **a probe never raises.** Every branch of every
probe here is wrapped, because the alternative is an install that dies on a machine whose `torch`
is half-installed or whose `docker` binary hangs. "Unknown" is a real and useful answer; a
traceback during preflight is not.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from recall.wizard import probe as P


# --------------------------------------------------------------------------------------
# The governing rule: nothing here raises, whatever the underlying call does
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func",
    [
        P.probe_ram,
        P.probe_docker,
        P.probe_wsl2,
        P.probe_virtualization,
        P.probe_cuda_vram,
    ],
)
def test_a_probe_never_raises_even_when_everything_below_it_explodes(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break every mechanism a probe could use, then call it."""

    def boom(*args, **kwargs):
        raise RuntimeError("the underlying mechanism is broken")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(P.shutil, "which", boom)
    monkeypatch.setattr(P, "_read_proc_meminfo", boom)
    monkeypatch.setattr(P, "_windows_memory_status", boom)
    monkeypatch.setattr(P, "_torch", boom)

    func()  # must return, not raise


def test_probe_system_never_raises_and_always_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(P.shutil, "which", boom)
    monkeypatch.setattr(P, "_torch", boom)

    result = P.probe_system()
    assert isinstance(result, P.SystemProbe)
    assert result.python_executable  # always knowable
    assert result.docker_running is False  # unknown collapses to "not usable", never to True


# --------------------------------------------------------------------------------------
# RAM
# --------------------------------------------------------------------------------------


def test_ram_on_this_machine_is_plausible() -> None:
    """Unmocked, on whatever host runs the suite. A wrong unit is the likely bug here."""
    total, available = P.probe_ram()
    if total is None:
        pytest.skip(f"no RAM probe for platform {sys.platform!r}")
    assert total > 512 * 1024**2, "less than 512 MB total means the units are wrong"
    assert total < 8 * 1024**4, "more than 8 TB total means the units are wrong"
    if available is not None:
        assert 0 < available <= total


def test_ram_reads_proc_meminfo_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        P,
        "_read_proc_meminfo",
        lambda: {"MemTotal": 16_000_000 * 1024, "MemAvailable": 8_000_000 * 1024},
    )
    total, available = P.probe_ram()
    assert total == 16_000_000 * 1024
    assert available == 8_000_000 * 1024


def test_ram_is_unknown_rather_than_zero_when_it_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None and 0 must stay distinguishable: 0 would fail every threshold as if the box were tiny."""
    monkeypatch.setattr(sys, "platform", "some-future-os")
    assert P.probe_ram() == (None, None)


# --------------------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------------------


def test_docker_absent_is_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P.shutil, "which", lambda _: None)
    assert P.probe_docker() == (False, False)


def test_docker_installed_but_daemon_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented Windows failure: installing Docker Desktop does not start it."""
    monkeypatch.setattr(P.shutil, "which", lambda _: "C:/bin/docker.exe")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "Cannot connect to the Docker daemon"),
    )
    assert P.probe_docker() == (True, False)


def test_docker_installed_and_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P.shutil, "which", lambda _: "C:/bin/docker.exe")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "Server Version: 27", "")
    )
    assert P.probe_docker() == (True, True)


def test_a_hanging_docker_is_treated_as_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docker info` against a dead daemon can hang. Preflight must not hang with it."""

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="docker info", timeout=k.get("timeout", 5))

    monkeypatch.setattr(P.shutil, "which", lambda _: "C:/bin/docker.exe")
    monkeypatch.setattr(subprocess, "run", timeout)
    assert P.probe_docker() == (True, False)


# --------------------------------------------------------------------------------------
# The Microsoft Store Python trap
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "executable, expected",
    [
        (r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe", True),
        (r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12\python.exe", True),
        (r"C:\Python312\python.exe", False),
        ("/usr/bin/python3", False),
    ],
)
def test_store_python_is_detected_by_path(
    monkeypatch: pytest.MonkeyPatch, executable: str, expected: bool
) -> None:
    """Documented in site/troubleshooting.html: the Store build sandboxes and rewrites file paths,
    so `recall index` on a real folder finds nothing and reports success."""
    monkeypatch.setattr(sys, "executable", executable)
    assert P.probe_store_python() is expected


def test_store_python_survives_a_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "executable", "")
    assert P.probe_store_python() is False


# --------------------------------------------------------------------------------------
# CUDA VRAM, which is what decides whether SPLADE is worth offering
# --------------------------------------------------------------------------------------


def test_cuda_vram_is_none_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "_torch", lambda: None)
    monkeypatch.setattr(P.shutil, "which", lambda _: None)
    assert P.probe_cuda_vram() is None


def test_cuda_vram_from_nvidia_smi_when_torch_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with a driver but no torch still has a GPU worth reporting."""
    monkeypatch.setattr(P, "_torch", lambda: None)
    monkeypatch.setattr(P.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "24576\n", "")
    )
    assert P.probe_cuda_vram() == 24576 * 1024**2


def test_cuda_vram_ignores_unparseable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "_torch", lambda: None)
    monkeypatch.setattr(P.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "N/A\n", "")
    )
    assert P.probe_cuda_vram() is None


def test_splade_is_only_offered_with_enough_vram() -> None:
    """The user's rule: no capable GPU means the question is never asked.

    `sparse_choices` in the existing wizard gates on `cuda_available` alone, which would offer
    SPLADE on a 2 GB laptop GPU that cannot load the encoder.
    """
    assert P.splade_is_feasible(cuda_available=False, vram_bytes=None) is False
    assert P.splade_is_feasible(cuda_available=True, vram_bytes=None) is False
    assert P.splade_is_feasible(cuda_available=True, vram_bytes=2 * 1024**3) is False
    assert P.splade_is_feasible(cuda_available=True, vram_bytes=P.SPLADE_MIN_VRAM_BYTES) is True


# --------------------------------------------------------------------------------------
# Composition: SystemProbe carries the existing HardwareProbe rather than replacing it
# --------------------------------------------------------------------------------------


def test_system_probe_embeds_the_existing_hardware_probe(tmp_path: Path) -> None:
    """`recall.setup` stays the authority on CPU/GPU/disk/internet; this adds to it."""
    from recall.setup import HardwareProbe

    result = P.probe_system(path=tmp_path)
    assert isinstance(result.hardware, HardwareProbe)
    assert result.hardware.free_bytes > 0
    assert result.hardware.cpu_count == os.cpu_count()


def test_summary_names_every_blocker_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_why_unavailable` in setup.py reports the conditions that actually failed, not the usual
    one. Preflight owes the reader the same, or they fix the wrong thing."""
    monkeypatch.setattr(P.shutil, "which", lambda _: None)
    monkeypatch.setattr(P, "probe_ram", lambda: (2 * 1024**3, 1 * 1024**3))
    monkeypatch.setattr(sys, "executable", r"C:\WindowsApps\python.exe")

    blockers = P.blockers(P.probe_system())
    joined = " ".join(blockers).lower()
    assert "docker" in joined
    assert "memory" in joined or "ram" in joined
    assert "store" in joined


def test_a_running_docker_proves_virtualization_without_asking_systeminfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`systeminfo` takes tens of seconds; a running container engine already answered.

    Asserted by breaking the slow path outright: if the shortcut were removed, this raises rather
    than quietly taking longer, so the test cannot pass for the wrong reason.
    """

    def must_not_be_called(*a, **k):
        raise AssertionError("systeminfo was run even though Docker is already running")

    monkeypatch.setattr(P, "_run", must_not_be_called)
    assert P.probe_virtualization(docker_running=True) is True


def test_virtualization_is_unknown_rather_than_false_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown must not become a blocker; `blockers` only refuses on an explicit False."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert P.probe_virtualization() is None
    assert P.blockers(
        P.SystemProbe(
            hardware=P.probe_hardware(),
            total_ram_bytes=None,
            available_ram_bytes=None,
            docker_installed=True,
            docker_running=True,
            wsl2_available=None,
            virtualization_enabled=None,
            cuda_vram_bytes=None,
            python_executable="/usr/bin/python3",
            python_is_store_build=False,
        )
    ) == []


def test_no_blockers_on_a_healthy_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "probe_ram", lambda: (32 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(P, "probe_docker", lambda: (True, True))
    monkeypatch.setattr(P, "probe_store_python", lambda: False)
    assert P.blockers(P.probe_system()) == []
