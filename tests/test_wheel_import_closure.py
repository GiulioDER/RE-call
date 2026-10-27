"""No module the wheel ships imports a first-party module the wheel leaves out.

Invariant: for every `.py` file under `[tool.hatch.build.targets.wheel] packages` that the wheel's
`exclude` list does not remove, every first-party module it imports, or names as a dotted string
(the form `importlib.import_module` takes, as in `recall.__init__._LAZY_EXPORTS` and
`recall.cli._COMMAND_REGISTRATIONS`), is itself in the wheel. First party means a top-level
directory of this repository with an `__init__.py`: the shipped packages, and also `benchmarks`,
`recall_interop`, `recall_consistency`, `examples` and `tests`, none of which the wheel carries.

Failure mode caught: a module is added to `exclude` while shipped code still imports it, or shipped
code starts importing research or repository-only code. Both pass every test in a checkout, where
the whole tree is importable, and fail only for a user who installed the wheel, as an
`ImportError` on the first call that reaches the import.

Imports under `if TYPE_CHECKING:` are exempt, because they never run. Imports inside functions are
NOT exempt: a lazy import of an excluded module is the same `ImportError`, only later. The
`exclude` patterns are matched with `fnmatch` plus a directory prefix, which is how hatchling reads
the entries this file holds today; `test_the_scan_sees_both_sides_of_the_boundary` fails if an
entry stops matching anything, so a pattern this reading misses cannot pass silently. Checked
against a real wheel built by hatchling 1.32.4 on 2026-10-02: the wheel and `_shipped_files` hold
the same 270 `.py` files, none missing either way.

Red proof, recorded 2026-10-02 on a Linux test host, one mutation per run on a copy of the tree,
each failing in the named test's leak assertion:

- `"recall/eval/calibrate.py"` added to the wheel `exclude` in `pyproject.toml`:
  `test_no_shipped_module_imports_code_the_wheel_leaves_out` fails naming
  `recall/calibration_v2.py:806` and `recall/setup.py:28` (eager) and `recall/setup.py:1250` (lazy).
- a function appended to `recall/wizard/queryset.py` that runs `from recall.eval import harness`:
  the same test fails naming that line, so lazy imports are seen.
- `import benchmarks.claim_gate` appended to `recall_mcp/evidence_cards.py`: the same test fails,
  so a top-level tree the wheel never ships is seen.
- `"run_harness": "recall.eval.harness"` added to `recall.__init__._LAZY_EXPORTS`:
  `test_no_shipped_module_names_left_out_code_as_a_module_string` fails naming that line.
- `"recall/eval/does_not_exist.py"` added to the wheel `exclude`:
  `test_the_scan_sees_both_sides_of_the_boundary` fails naming the entry.

Control: the second mutation's import placed under `if TYPE_CHECKING:` passes, as intended.
Unmodified tree: all three pass.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _wheel_config() -> dict[str, Any]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel: dict[str, Any] = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    return wheel


def _first_party() -> frozenset[str]:
    return frozenset(path.parent.name for path in ROOT.glob("*/__init__.py"))


def _excluded(rel: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(rel, pattern) or rel.startswith(pattern.rstrip("/") + "/")
        for pattern in patterns
    )


def _ships(rel: str, config: dict[str, Any]) -> bool:
    return rel.split("/", 1)[0] in config["packages"] and not _excluded(rel, config.get("exclude", []))


def _shipped_files(config: dict[str, Any]) -> list[Path]:
    files = [path for package in config["packages"] for path in (ROOT / package).rglob("*.py")]
    return sorted(path for path in files if _ships(path.relative_to(ROOT).as_posix(), config))


def _module_name(path: Path) -> tuple[str, bool]:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def _resolve(module: str, first_party: frozenset[str]) -> str | None:
    """Return the repository path that defines `module`, or None for anything not first party."""

    parts = module.split(".")
    if parts[0] not in first_party:
        return None
    for candidate in (ROOT.joinpath(*parts).with_suffix(".py"), ROOT.joinpath(*parts, "__init__.py")):
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    return None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imported_names(tree: ast.Module, module: str, is_package: bool) -> Iterator[tuple[int, str]]:
    """Yield `(line, dotted name)` for every import that can run, eager or lazy.

    `from a import b` yields both `a` and `a.b`, since `b` may be a submodule; a name that is an
    attribute rather than a module simply does not resolve to a file.
    """

    def visit(nodes: list[ast.stmt]) -> Iterator[tuple[int, str]]:
        for node in nodes:
            if isinstance(node, ast.If) and _is_type_checking(node.test):
                yield from visit(node.orelse)
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield node.lineno, alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.split(".") if is_package else module.split(".")[:-1]
                    base = base[: len(base) - (node.level - 1)]
                    source = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    source = node.module or ""
                yield node.lineno, source
                for alias in node.names:
                    yield node.lineno, f"{source}.{alias.name}"
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    yield from visit([child])
                elif isinstance(child, ast.excepthandler):
                    yield from visit(child.body)
                elif isinstance(child, ast.match_case):
                    yield from visit(child.body)

    yield from visit(tree.body)


_DOTTED = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")


def _string_module_names(tree: ast.Module) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _DOTTED.match(node.value):
            yield node.lineno, node.value


def _leaks(names_of: Any) -> tuple[list[str], int]:
    config = _wheel_config()
    first_party = _first_party()
    leaks: list[str] = []
    resolved = 0
    for path in _shipped_files(config):
        module, is_package = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, name in names_of(tree, module, is_package):
            target = _resolve(name, first_party)
            if target is None:
                continue
            resolved += 1
            if not _ships(target, config):
                leaks.append(f"{path.relative_to(ROOT).as_posix()}:{line} -> {name} ({target})")
    return sorted(set(leaks)), resolved


def test_no_shipped_module_imports_code_the_wheel_leaves_out() -> None:
    leaks, resolved = _leaks(_imported_names)
    assert resolved > 0, "the scan resolved no first-party import, so it checked nothing"
    assert leaks == [], (
        "these wheel-shipped modules import first-party code the wheel does not ship, which an "
        "installed copy raises ImportError on:\n" + "\n".join(leaks)
    )


def test_no_shipped_module_names_left_out_code_as_a_module_string() -> None:
    leaks, resolved = _leaks(lambda tree, _module, _is_package: _string_module_names(tree))
    assert resolved > 0, "the scan resolved no dotted module string, so it checked nothing"
    assert leaks == [], (
        "these wheel-shipped modules name first-party code the wheel does not ship as a dotted "
        "module string (an importlib target), which an installed copy cannot import:\n"
        + "\n".join(leaks)
    )


def test_the_scan_sees_both_sides_of_the_boundary() -> None:
    """Every `exclude` entry removes at least one file, and the shipped set is not empty.

    An entry that matches nothing is either stale or written in a form this file's matching does
    not read, and either way the two tests above would check nothing for it.
    """

    config = _wheel_config()
    candidates = [
        path.relative_to(ROOT).as_posix()
        for package in config["packages"]
        for path in (ROOT / package).rglob("*")
        if path.is_file()
    ]
    dead = [
        pattern for pattern in config.get("exclude", []) if not any(_excluded(rel, [pattern]) for rel in candidates)
    ]
    assert dead == [], f"wheel exclude entries that match no file: {dead}"
    assert _shipped_files(config), "the wheel ships no Python file, so the scan checked nothing"
