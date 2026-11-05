"""Every console script turns off the working-directory executable search first (M2).

Invariant: each entry point in `pyproject.toml`'s `[project.scripts]` calls
`recall._env.refuse_current_directory_executables()`, or (in `recall_hooks`, which does not import
`recall` on every hook) sets `NoDefaultCurrentDirectoryInExePath` itself, before anything else it
does. The failure mode is CCA's `injection` shape: Windows CreateProcess finds a bare `git`,
`powershell` or `taskkill` in the current directory before PATH, so a `git.exe` committed at the
root of a repository runs instead of git. Before this, two of the ten entry points did it;
`recall-mcp`, started by the client in the session's directory and running `git` through the
indexer, did not.

Red proof (2026-11-04): the call deleted from `recall_mcp.server.main` made
`test_every_console_script_refuses_cwd_executables_first` fail naming `recall-mcp`; on master it
fails in the same assertion naming nine of the ten, all but `recall`
(`recall-hooks` set the variable inside `_dispatch`, after reading its arguments).
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = "NoDefaultCurrentDirectoryInExePath"


def _scripts() -> dict[str, str]:
    with (REPO / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def _function(target: str) -> ast.FunctionDef:
    module, _, name = target.partition(":")
    base = REPO / Path(*module.split("."))
    path = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{target} names no module-level function")


def _hardens(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        func = statement.value.func
        return getattr(func, "id", getattr(func, "attr", "")) == "refuse_current_directory_executables"
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        return isinstance(target, ast.Subscript) and ast.unparse(target) == f"os.environ['{ENV}']"
    return False


def _refuses_first(function: ast.FunctionDef) -> bool:
    """Whether the hardening is the first statement that does anything: only a docstring, imports
    and `global` declarations may precede it."""
    for statement in function.body:
        if _hardens(statement):
            return True
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue  # the docstring
        if isinstance(statement, ast.Import | ast.ImportFrom | ast.Global):
            continue
        return False
    return False


def test_every_console_script_refuses_cwd_executables_first() -> None:
    late = sorted(name for name, target in _scripts().items() if not _refuses_first(_function(target)))
    assert late == []


def test_the_scripts_table_is_read() -> None:
    """Negative control on the table itself: an empty or misread table would pass the test above."""
    scripts = _scripts()
    assert {"recall", "recall-mcp", "recall-hooks"} <= set(scripts)
    assert len(scripts) >= 10


def test_the_shared_call_sets_the_variable(monkeypatch) -> None:
    import os

    from recall._env import refuse_current_directory_executables

    monkeypatch.delenv(ENV, raising=False)
    refuse_current_directory_executables()
    assert os.environ[ENV] == "1"
