"""PR988 ENV-001: generate a usable Windows cache recovery command without executing it.

Red/green execution against baseline 00a124ba2 and the repair belongs to the primary on the testbench.
These tests inspect generated text only; they never invoke a shell or a generated command.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from recall_hooks import auditor_brief


@pytest.mark.parametrize(
    ("package_name", "executable"),
    [
        ("Example package root", r"C:\Program Files\Example Python\python.exe"),
        ("Example's package root", r"C:\Program Files\Example's Python\python.exe"),
    ],
)
def test_windows_recovery_command_quotes_paths_and_restores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
    executable: str,
) -> None:
    """ENV-001: Windows recovery uses PowerShell syntax, the intended copy, and safe literals.

    Baseline: 00a124ba2. Target: recall_hooks.auditor_brief.warm_command. Intended red failure:
    the original POSIX-only command fails the PowerShell-label assertion. Later assertions catch
    lost apostrophe escaping, a wrong interpreter/root, missing invocation syntax or a missing
    finally restoration. sys.platform is patched without changing os.name or executing the text.
    """
    package_parent = tmp_path / package_name
    module_file = package_parent / "recall_hooks" / "auditor_brief.py"
    expected_root = str(package_parent.resolve())
    monkeypatch.setattr(auditor_brief, "__file__", str(module_file))
    monkeypatch.setattr(auditor_brief.sys, "platform", "win32")
    monkeypatch.setattr(auditor_brief.sys, "executable", executable)

    command = auditor_brief.warm_command()

    assert re.search(r"\bpowershell\b", command, re.IGNORECASE), (
        "Windows recovery command does not identify its PowerShell syntax"
    )
    root_literal = "'" + expected_root.replace("'", "''") + "'"
    executable_literal = "'" + executable.replace("'", "''") + "'"
    assignment = re.search(r"\$env:PYTHONPATH\s*=\s*" + re.escape(root_literal), command)
    assert assignment is not None, (
        "recovery command did not quote the intended package root as a PowerShell literal"
    )
    invocation = re.search(
        r"&\s+" + re.escape(executable_literal) + r"\s+-P\s+-m\s+recall_hooks\s+warm-auditor-brief\b",
        command,
    )
    assert invocation is not None, (
        "recovery command did not safely invoke the intended interpreter and module"
    )

    saved = re.search(
        r"(?P<name>\$[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\$env:PYTHONPATH\b",
        command,
    )
    assert saved is not None, "recovery command did not save the caller's PYTHONPATH"
    guarded = re.search(r"\btry\s*\{", command, re.IGNORECASE)
    restored = re.search(r"\bfinally\s*\{(?P<body>.*?)\}", command, re.IGNORECASE | re.DOTALL)
    assert guarded is not None and restored is not None, (
        "recovery command lacks try/finally environment restoration"
    )
    assert saved.start() < guarded.start() < assignment.start() < invocation.start() < restored.start(), (
        "recovery command does not bracket its environment change and invocation with restoration"
    )
    assert re.search(
        r"\$env:PYTHONPATH\s*=\s*" + re.escape(saved.group("name")) + r"\b",
        restored.group("body"),
    ), "recovery command did not restore the saved PYTHONPATH in finally"
