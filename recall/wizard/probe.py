"""What the installer needs to know about the machine, beyond what `recall.setup` already probes.

`recall.setup.probe_hardware` reports CPU count, GPU name, CUDA availability, free disk and
internet, and it stays the authority on those. `SystemProbe` carries one rather than replacing it,
and adds the four facts an installer cannot proceed without:

* **RAM.** Nothing in the tree measured it, so every choice was gated on disk alone. A machine can
  clear a 4 GB disk floor and still be unable to hold a reranker and a database at once.
* **Docker.** Installed and running are different states, and the second is the documented Windows
  surprise: installing Docker Desktop does not start it.
* **Which Python this is.** The Microsoft Store build sandboxes and rewrites file paths, so
  `recall index` on a real folder finds nothing and exits 0. That is the worst class of failure
  this project has, because it is silent.
* **How much VRAM.** `cuda_available` alone says a GPU exists, not that it can run what is being
  offered.

**Every probe returns rather than raises.** This follows `_probe_gpu`, which wraps every branch: a
half-installed `torch`, a hung `docker`, a `/proc` that is not there. Unknown is a useful answer and
is represented as `None`; a traceback during preflight ends an install over a diagnostic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recall.setup import HardwareProbe, probe_hardware

#: `docker info` against a dead daemon can sit for a long time. Preflight is allowed to be wrong
#: about Docker; it is not allowed to hang, so the probe is bounded and a timeout reads as "not
#: running" rather than as an error.
DOCKER_TIMEOUT_SECONDS = 8.0

#: Below this the install is refused rather than merely warned about: PostgreSQL in a container,
#: an embedder and the OS do not coexist under it.
MIN_RAM_BYTES = 4 * 1024**3

#: Below this the install proceeds but the heavier options are withheld. A cross-encoder reranker
#: and a 1.2 GB embedder on top of a database want headroom this names.
COMFORTABLE_RAM_BYTES = 8 * 1024**3

#: The SPLADE encoder plus its vocabulary-width activations. Gating on `cuda_available` alone,
#: which is what `recall.setup.sparse_choices` does today, offers SPLADE on a 2 GB laptop GPU that
#: cannot load it — an option that is visible, selectable, and fails later.
SPLADE_MIN_VRAM_BYTES = 6 * 1024**3

#: Path fragments that identify the Microsoft Store interpreter. Matched case-insensitively
#: against `sys.executable`. `WindowsApps` covers both the per-user shim directory and the
#: `Program Files\WindowsApps` install root.
_STORE_PYTHON_MARKERS = ("windowsapps",)


@dataclass(frozen=True)
class SystemProbe:
    """Everything preflight established, with `None` meaning "could not tell"."""

    hardware: HardwareProbe
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    docker_installed: bool
    docker_running: bool
    wsl2_available: bool | None
    virtualization_enabled: bool | None
    cuda_vram_bytes: int | None
    python_executable: str
    python_is_store_build: bool


def _torch() -> Any | None:
    """`torch` if it imports, else None. Its own import can fail in more ways than ImportError."""
    try:
        import torch
    except Exception:
        return None
    return torch


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    """Run `command`, returning None on any failure at all. Never raises."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None


def _read_proc_meminfo() -> dict[str, int]:
    """`/proc/meminfo` as bytes per key. Empty when unreadable."""
    values: dict[str, int] = {}
    try:
        raw = Path("/proc/meminfo").read_text(encoding="utf-8")
    except Exception:
        return values
    for line in raw.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            # Every numeric line in /proc/meminfo is in kB, and the ones without a unit are counts
            # rather than sizes, so they are skipped rather than guessed at.
            if len(parts) > 1 and parts[1].lower() == "kb":
                values[key.strip()] = int(parts[0]) * 1024
        except ValueError:
            continue
    return values


def _windows_memory_status() -> tuple[int, int] | None:
    """Total and available physical bytes via GlobalMemoryStatusEx. None on any failure.

    ctypes rather than psutil, so this adds no dependency to a package whose install footprint is
    the thing the wizard is trying to keep small.
    """
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def probe_ram() -> tuple[int | None, int | None]:
    """Total and available physical memory in bytes, `(None, None)` when it cannot be read.

    `None` rather than `0` deliberately: zero would compare as a real measurement below every
    threshold, so an unreadable value would silently refuse the install on a machine that is fine.
    """
    try:
        if sys.platform == "win32":
            return _windows_memory_status() or (None, None)
        if sys.platform.startswith("linux"):
            info = _read_proc_meminfo()
            total = info.get("MemTotal")
            available = info.get("MemAvailable", info.get("MemFree"))
            return (total or None), (available or None)
        if sys.platform == "darwin":
            result = _run(["sysctl", "-n", "hw.memsize"], timeout=5.0)
            if result and result.returncode == 0:
                return int(result.stdout.strip()), None
    except Exception:
        return None, None
    return None, None


def probe_docker() -> tuple[bool, bool]:
    """`(installed, running)`. A timeout or any error reads as installed-but-not-running.

    The two are kept apart because the remedies differ and the second is the documented Windows
    surprise: installing Docker Desktop does not start it, and the error a user then sees names
    the daemon rather than the application they need to launch.
    """
    try:
        if not shutil.which("docker"):
            return False, False
    except Exception:
        return False, False
    result = _run(["docker", "info", "--format", "{{.ServerVersion}}"], DOCKER_TIMEOUT_SECONDS)
    return True, bool(result and result.returncode == 0)


def probe_wsl2() -> bool | None:
    """Whether a WSL2 distribution is available. `None` off Windows, where it does not apply."""
    try:
        if sys.platform != "win32":
            return None
        if not shutil.which("wsl"):
            return False
    except Exception:
        return None
    result = _run(["wsl", "--status"], timeout=10.0)
    return bool(result and result.returncode == 0)


def probe_virtualization(*, docker_running: bool = False) -> bool | None:
    """Whether hardware virtualization is enabled. `None` when it cannot be determined.

    Docker Desktop cannot run without it, and it is disabled in firmware often enough to be worth
    naming before the Docker install fails with a message about Hyper-V.

    `docker_running` short-circuits the whole probe, for both speed and accuracy. A running Linux
    container engine is *proof* that virtualization is enabled, whereas `systeminfo` is an
    inference from text that takes ten to twenty seconds to produce. Asking the slow question after
    the fast one has already answered it would make preflight feel frozen for no added certainty.
    """
    if docker_running:
        return True
    try:
        if sys.platform != "win32":
            return None
    except Exception:
        return None
    # `systeminfo` is slow but present on every Windows edition, and its wording is stable.
    result = _run(["systeminfo"], timeout=60.0)
    if not result or result.returncode != 0:
        return None
    text = result.stdout.lower()
    if "virtualization enabled in firmware: yes" in text:
        return True
    if "virtualization enabled in firmware: no" in text:
        return False
    # A machine already running Hyper-V reports the requirements as met by the hypervisor instead.
    if "hyper-v requirements" in text and "hypervisor has been detected" in text:
        return True
    return None


def probe_cuda_vram() -> int | None:
    """Total VRAM of device 0 in bytes, `None` when there is no CUDA device or no way to ask."""
    try:
        torch = _torch()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    return int(torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                pass
        if not shutil.which("nvidia-smi"):
            return None
    except Exception:
        return None
    result = _run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        timeout=10.0,
    )
    if not result or result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        # nvidia-smi reports MiB under `nounits`.
        return int(first.strip()) * 1024**2
    except ValueError:
        return None


def probe_store_python() -> bool:
    """Whether this interpreter is the Microsoft Store build.

    Its path sandboxing rewrites file access, so indexing a real folder finds nothing and reports
    success — the failure mode `site/troubleshooting.html` documents as "Python works, but indexing
    a folder finds nothing". Detected by path because that is what the troubleshooting page tells a
    user to check, and the two agreeing matters more than a cleverer test.
    """
    try:
        executable = (sys.executable or "").lower()
    except Exception:
        return False
    return any(marker in executable for marker in _STORE_PYTHON_MARKERS)


def splade_is_feasible(*, cuda_available: bool, vram_bytes: int | None) -> bool:
    """Whether SPLADE is worth offering on this machine at all.

    Asked before the question is shown, not after it is answered. Unknown VRAM counts as NOT
    feasible, which is the opposite of how liveness is treated for a session claim: there, a false
    "dead" costs somebody's work, so unknown means alive. Here a false "feasible" costs a user a
    selectable option that fails at query time, so unknown means withhold.
    """
    if not cuda_available or vram_bytes is None:
        return False
    return vram_bytes >= SPLADE_MIN_VRAM_BYTES


def probe_system(path: Path | None = None) -> SystemProbe:
    """Everything above, in one call. Never raises."""
    try:
        hardware = probe_hardware(path)
    except Exception:
        hardware = HardwareProbe(
            cpu_count=os.cpu_count(),
            gpu=None,
            cuda_available=False,
            free_bytes=0,
            internet=False,
            fastembed_available=False,
            sentence_transformers_available=False,
        )
    total, available = probe_ram()
    docker_installed, docker_running = probe_docker()
    try:
        executable = sys.executable or ""
    except Exception:
        executable = ""
    return SystemProbe(
        hardware=hardware,
        total_ram_bytes=total,
        available_ram_bytes=available,
        docker_installed=docker_installed,
        docker_running=docker_running,
        wsl2_available=probe_wsl2(),
        virtualization_enabled=probe_virtualization(docker_running=docker_running),
        cuda_vram_bytes=probe_cuda_vram(),
        python_executable=executable,
        python_is_store_build=probe_store_python(),
    )


def blockers(system: SystemProbe) -> list[str]:
    """Every condition that would stop the install, each naming its own remedy.

    Reports all of them rather than the first. `_why_unavailable` in `recall/setup.py` exists
    because a note naming the usual blocker, to somebody whose real blocker is different, sends
    them to fix the wrong thing; the same applies here and more expensively, because these are
    fixed before the install rather than during it.

    An unknown value is never a blocker. Refusing to install because a probe could not read
    `systeminfo` would fail closed on a diagnostic.
    """
    found: list[str] = []
    if system.total_ram_bytes is not None and system.total_ram_bytes < MIN_RAM_BYTES:
        gib = system.total_ram_bytes / 1024**3
        found.append(
            f"This machine reports {gib:.1f} GB of memory, below the {MIN_RAM_BYTES // 1024**3} GB "
            "needed to run PostgreSQL and an embedding model together."
        )
    if not system.docker_installed:
        found.append(
            "Docker was not found. RE-call stores its index in PostgreSQL with pgvector, which "
            "the installer runs as a container."
        )
    elif not system.docker_running:
        found.append(
            "Docker is installed but its daemon is not responding. Installing Docker Desktop does "
            "not start it: launch it once and let it finish starting."
        )
    if system.python_is_store_build:
        found.append(
            f"This is the Microsoft Store build of Python ({system.python_executable}). It "
            "sandboxes file paths, so indexing a folder finds nothing and still reports success. "
            "Install Python from python.org instead."
        )
    if system.virtualization_enabled is False:
        found.append(
            "Hardware virtualization is disabled in firmware. Docker Desktop cannot start "
            "without it; enable VT-x/AMD-V in the BIOS."
        )
    return found
