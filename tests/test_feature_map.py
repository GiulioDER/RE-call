"""The feature map (``tools/feature_map.py``): docs/FEATURE_MAP.md stays generated from
docs/feature_map.toml, and that source stays true of the code.

Invariant: every console script, ``recall`` command, MCP tool and shipped module is claimed by a
feature; every name and glob a feature lists exists; and the committed page equals the render.
Failure mode caught: a map written once and left to drift, which is the largest class of defect in
this repository's audit history (stale documentation).

Red proof, 2026-11-25, one mutation to ``tools/feature_map.py`` per test, run alone and restored;
each failed at its own assertion:

- ``render`` dropping each feature's summary: ``test_the_committed_map_is_current``.
- the unclaimed-module line removed from ``problems``: ``test_a_module_no_feature_claims_is_reported``.
- the unclaimed-surface line removed: ``test_a_command_tool_or_script_no_feature_claims_is_reported``.
- the "does not exist" append removed: ``test_a_name_that_does_not_exist_is_reported``.
- the "matches no tracked file" append removed: ``test_a_glob_that_matches_nothing_is_reported``.
- the duplicate-id append removed: ``test_a_duplicate_id_is_reported``.
- ``cli_commands`` reading the module name instead of the command set:
  ``test_commands_and_tools_are_read_from_the_source``.
- ``--which`` matching ``modules`` only: ``test_which_names_the_features_of_a_file``.
- ``--check`` returning 0 for a stale page: ``test_check_refuses_a_stale_page``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import textwrap
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "feature_map", Path(__file__).parents[1] / "tools" / "feature_map.py"
)
assert SPEC and SPEC.loader
fmap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fmap)

FEATURE = """
[[feature]]
id = "search"
area = "Read path"
name = "Search"
summary = "Find things."
scripts = ["recall"]
cli = ["search", "index"]
mcp = ["recall_search", "recall_index"]
modules = ["recall/*", "recall_mcp/*"]
tests = ["tests/test_search.py"]
"""


def _repo(tmp_path: Path, toml: str = FEATURE, extra: dict[str, str] | None = None) -> Path:
    files = {
        "pyproject.toml": '[project]\nname = "x"\n[project.scripts]\nrecall = "recall.cli:main"\n',
        "recall/__init__.py": "",
        "recall/cli.py": textwrap.dedent("""
            _COMMAND_REGISTRATIONS: tuple = (
                (frozenset({"search"}), "recall.cli_commands.search", "register"),
                (frozenset({"index"}), "recall.cli_commands.index", "register"),
            )
        """),
        "recall_mcp/tool_surface.py": 'ALL_TOOL_NAMES: frozenset[str] = frozenset({"recall_search", "recall_index"})\n',
        "tests/test_search.py": "",
        "docs/feature_map.toml": toml,
        **(extra or {}),
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def _problems(root: Path) -> list[str]:
    return fmap.problems(fmap.load(root), root, fmap.tracked(root))


def test_the_committed_map_is_current() -> None:
    assert fmap.main(["--check"]) == 0


def test_a_module_no_feature_claims_is_reported(tmp_path: Path) -> None:
    root = _repo(tmp_path, FEATURE.replace('"recall/*"', '"recall/cli.py", "recall/__init__.py"'),
                 {"recall/orphan.py": ""})
    assert _problems(root) == ["module `recall/orphan.py` is in no feature"]


def test_a_command_tool_or_script_no_feature_claims_is_reported(tmp_path: Path) -> None:
    toml = FEATURE.replace('cli = ["search", "index"]', 'cli = ["search"]').replace(
        'mcp = ["recall_search", "recall_index"]', 'mcp = ["recall_search"]').replace('scripts = ["recall"]\n', "")
    assert _problems(_repo(tmp_path, toml)) == [
        "scripts `recall` is in no feature", "cli `index` is in no feature", "mcp `recall_index` is in no feature",
    ]


def test_a_name_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    toml = FEATURE.replace('mcp = ["recall_search", "recall_index"]', 'mcp = ["recall_search", "recall_index", "recall_gone"]')
    assert _problems(_repo(tmp_path, toml)) == ["search: mcp `recall_gone` does not exist"]


def test_a_glob_that_matches_nothing_is_reported(tmp_path: Path) -> None:
    toml = FEATURE.replace('tests = ["tests/test_search.py"]', 'tests = ["tests/test_search.py", "tests/test_moved*.py"]')
    assert _problems(_repo(tmp_path, toml)) == ["search: tests `tests/test_moved*.py` matches no tracked file"]


def test_a_duplicate_id_is_reported(tmp_path: Path) -> None:
    second = FEATURE.replace('cli = ["search", "index"]\n', "").replace('mcp = ["recall_search", "recall_index"]\n', "")
    assert _problems(_repo(tmp_path, FEATURE + second.replace('scripts = ["recall"]\n', ""))) == ["search: duplicate id"]


def test_commands_and_tools_are_read_from_the_source(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert (fmap.cli_commands(root), fmap.mcp_tools(root)) == ({"search", "index"}, {"recall_search", "recall_index"})


def test_which_names_the_features_of_a_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo(tmp_path)
    assert fmap.main(["--which", "tests/test_search.py"], root) == 0
    assert capsys.readouterr().out == "search: Search\n"
    assert fmap.main(["--which", "elsewhere/file.py"], root) == 1


def test_check_refuses_a_stale_page(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert fmap.main([], root) == 0
    assert fmap.main(["--check"], root) == 0
    page = root / fmap.OUTPUT
    page.write_text(page.read_text(encoding="utf-8").replace("Find things.", "Find other things."), encoding="utf-8")
    assert fmap.main(["--check"], root) == 1
