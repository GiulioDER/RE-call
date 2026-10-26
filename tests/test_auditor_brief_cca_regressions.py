"""CCA regressions for the auditor brief's handling of untrusted Git tree paths."""

from __future__ import annotations

import hashlib
import json
import stat
import time
from pathlib import PureWindowsPath
from types import SimpleNamespace
from typing import Any

import pytest

from recall_hooks import auditor_brief


@pytest.mark.parametrize(
    "malicious",
    [r"\\attacker.invalid\share\x.py", r"D:\outside\x.py", r"..\outside\x.py"],
    ids=["unc", "drive", "backslash-traversal"],
)
def test_decoded_git_path_is_confined_before_filesystem_access(
    monkeypatch: pytest.MonkeyPatch, malicious: str,
) -> None:
    r"""SEC-001: a C-quoted Git filename must not become an outside filesystem request.

    Node: tests/test_auditor_brief_cca_regressions.py::
    test_decoded_git_path_is_confined_before_filesystem_access.
    Baseline: 00a124ba211ea79b72427e51f61ed94ac62ec950, targeting
    `changed_lines` -> `_collect` -> `_small`. Predict the baseline fails the explicit
    "decoded Git filename reached an outside filesystem path" assertion: `_small`
    calls stat after Windows joining has discarded or escaped the checkout root.
    The primary CCA runner records the actual red/green receipt on the testbench.

    This calls the real public scope flow and real diff parser. PureWindowsPath models
    Windows joining on Linux; module-local OS and path doubles intercept every filesystem
    operation, and fake Git processes supply all bytes. Even the unfixed implementation
    cannot touch a real host. The benign Python file must still produce its symbol anchor.
    """
    root = PureWindowsPath("C:/checkout")
    benign = "pkg/safe.py"
    source = b"def safe():\n    return 1\n"
    filesystem_calls: list[tuple[str, PureWindowsPath]] = []
    blob_requests: list[bytes] = []
    git_calls: list[tuple[str, ...]] = []

    def filesystem(operation: str, path: PureWindowsPath) -> SimpleNamespace:
        path = PureWindowsPath(path)
        filesystem_calls.append((operation, path))
        if path == root / "pkg":
            return SimpleNamespace(st_size=0, st_mode=stat.S_IFDIR, st_reparse_tag=0)
        if path == root / benign:
            return SimpleNamespace(st_size=len(source), st_mode=stat.S_IFREG, st_reparse_tag=0)
        # Record rather than raise AssertionError here: the baseline must fail the
        # behavior assertion below, after its normal missing-file handler has run.
        raise FileNotFoundError(str(path))

    class WindowsPathDouble(PureWindowsPath):
        def resolve(self, *args: Any, **kwargs: Any) -> WindowsPathDouble:
            filesystem_calls.append(("resolve", self))
            assert self == root, "scope resolved an unexpected path"
            return self

        def stat(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return filesystem("stat", self)

    # json.dumps emits the C escapes used by Git for these ASCII backslash names.
    quoted = json.dumps("b/" + malicious)
    diff = (
        f"diff --git {json.dumps('a/' + malicious)} {quoted}\n"
        f"--- /dev/null\n+++ {quoted}\n@@ -0,0 +1,2 @@\n+def bad():\n+    pass\n"
        f"diff --git a/{benign} b/{benign}\n"
        f"--- /dev/null\n+++ b/{benign}\n@@ -0,0 +1,2 @@\n+def safe():\n+    return 1\n"
    ).encode()

    class GitProcessDouble:
        returncode = 0

        def __init__(self, args: tuple[str, ...]) -> None:
            self.args = args

        def communicate(
            self, input: bytes | None = None, timeout: float | None = None,
        ) -> tuple[bytes, bytes]:
            command = self.args[0]
            if command == "rev-parse":
                output = root.as_posix().encode() + b"\n"
            elif command == "config":
                output = b"https://github.com/example/audited.git\n"
            elif command == "merge-base":
                output = b"1" * 40 + b"\n"
            elif command == "diff":
                output = diff
            else:
                assert self.args == ("cat-file", "--batch"), "unexpected Git operation"
                assert input is not None
                blob_requests.append(input)
                assert input == f"HEAD:{benign}\n".encode(), "an outside blob was requested"
                output = b"2" * 40 + f" blob {len(source)}\n".encode() + source + b"\n"
            return output, b""

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            raise AssertionError("a completed fake Git process was killed")

    def start_git(cwd: str, *args: str, stdin: bool = False) -> GitProcessDouble:
        assert cwd == str(root)
        assert stdin == (args[0] == "cat-file")
        git_calls.append(args)
        return GitProcessDouble(args)

    def unexpected_log(reason: str) -> None:
        raise AssertionError(f"scope unexpectedly logged a failure: {reason}")

    monkeypatch.setattr(auditor_brief, "Path", WindowsPathDouble)
    monkeypatch.setattr(
        auditor_brief, "os",
        SimpleNamespace(name="nt", lstat=lambda path: filesystem("lstat", path)),
    )
    monkeypatch.setattr(auditor_brief, "_start_git", start_git)
    monkeypatch.setattr(auditor_brief, "_log", unexpected_log)

    repo, anchors = auditor_brief.scope(
        str(root), "Audit this branch", deadline=time.monotonic() + 30,
    )

    assert repo == "example/audited"
    by_file = {anchor["file"]: anchor for anchor in anchors}
    assert by_file[malicious] == {
        "file": malicious, "symbol": None, "code_sha": auditor_brief.UNKNOWN_SHA,
    }, "the real Git parser lost the decoded malicious filename"
    outside = [
        (operation, str(path))
        for operation, path in filesystem_calls
        if (path != root and not path.is_relative_to(root)) or ".." in path.parts
    ]
    assert not outside, f"decoded Git filename reached an outside filesystem path: {outside}"
    assert blob_requests == [f"HEAD:{benign}\n".encode()]
    assert by_file[benign] == {
        "file": benign,
        "symbol": "safe",
        "code_sha": hashlib.sha256(source.rstrip(b"\n")).hexdigest(),
    }
    assert any(args[0] == "diff" for args in git_calls)
    assert any(
        (operation, root / benign) in filesystem_calls for operation in ("stat", "lstat")
    )
