"""Windows process ownership for benchmark sessions.

The wrapper reads a release byte before starting the CLI. Its parent assigns it
to a job first, so every later descendant inherits that job. Closing the last
job handle kills its members, including descendants of an already exited CLI.

Prior work: extends the existing `claude_exec.py` subprocess boundary, using the
Windows API references below for benchmark process ownership.

https://learn.microsoft.com/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject
https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

KILL_ON_JOB_CLOSE = 0x2000
EXTENDED_LIMIT_INFORMATION = 9
PROCESS_SET_QUOTA_AND_TERMINATE = 0x0101


def _last_error() -> OSError:
    if sys.platform == "win32":
        return ctypes.WinError(ctypes.get_last_error())
    return OSError("Windows job APIs are unavailable on this platform")


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJob:
    """Own a job handle; assign the waiting wrapper before releasing its CLI."""

    _api: Any
    _handle: Any

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows jobs require Windows")
        api: Any = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], handle),
            "SetInformationJobObject": ([handle, ctypes.c_int32, ctypes.c_void_p, dword], boolean),
            "OpenProcess": ([dword, boolean, dword], handle),
            "AssignProcessToJobObject": ([handle, handle], boolean),
            "TerminateJobObject": ([handle, dword], boolean),
            "CloseHandle": ([handle], boolean),
        }
        for name, (args, result) in signatures.items():
            function = getattr(api, name)
            function.argtypes, function.restype = args, result
        self._api = api
        self._handle = api.CreateJobObjectW(None, None)
        if not self._handle:
            raise _last_error()
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(
            self._handle, EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = _last_error()
            self.close()
            raise error

    @staticmethod
    def command(command: Sequence[str]) -> list[str]:
        # Isolated Python startup avoids site hooks before job assignment.
        return [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--child", *command]

    def assign(self, pid: int) -> None:
        process = self._api.OpenProcess(PROCESS_SET_QUOTA_AND_TERMINATE, False, pid)
        if not process:
            raise _last_error()
        try:
            if not self._api.AssignProcessToJobObject(self._handle, process):
                raise _last_error()
        finally:
            self._api.CloseHandle(process)

    def terminate(self) -> None:
        if self._handle and not self._api.TerminateJobObject(self._handle, 1):
            raise _last_error()

    def close(self) -> None:
        if self._handle:
            handle, self._handle = self._handle, None
            if not self._api.CloseHandle(handle):
                raise _last_error()


def _child(command: Sequence[str]) -> int:
    if sys.stdin.buffer.read(1) != b"1":
        return 2
    return subprocess.call(list(command), stdin=subprocess.DEVNULL)  # noqa: S603 - argv, no shell


if __name__ == "__main__":
    if sys.argv[1:2] != ["--child"] or len(sys.argv) < 3:
        raise SystemExit(2)
    raise SystemExit(_child(sys.argv[2:]))
