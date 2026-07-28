"""Every Dependabot `ignore` rule must still describe a cap that exists in `pyproject.toml`.

An ignore rule is a claim: "this dependency is capped on purpose, stop proposing past it." Lift the
cap and forget the rule, and the claim quietly inverts — Dependabot goes on withholding updates for
a dependency nothing constrains any more, and the symptom is *silence*. No red PR, no failing job;
just a package that stops being offered. That is strictly worse than the noise the rule removed,
because noise is visible.

So the rule and the pin are checked against each other. The reverse direction is deliberately NOT
checked: a new cap without an ignore rule produces a weekly PR someone has to close, which is
annoying and self-announcing. Only the silent failure gets a test.

Parsed with regexes rather than a YAML/TOML library on purpose — the `test` and `typecheck` jobs
install `.[dev]` only, and neither PyYAML nor a TOML reader is declared there. A guard that needs a
new dependency to run is a guard that gets deleted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
PYPROJECT = ROOT / "pyproject.toml"

_IGNORED = re.compile(r'^\s*-\s*dependency-name:\s*"([^"]+)"', re.MULTILINE)


def _ignored_names() -> list[str]:
    return _IGNORED.findall(DEPENDABOT.read_text(encoding="utf-8"))


def _dependency_text() -> str:
    """The parts of pyproject.toml that actually DECLARE dependencies, comments removed.

    Scoped rather than whole-file, because a bare `"name"` elsewhere is indistinguishable from an
    uncapped requirement. Two live examples in this file: `"mcp"` appears in `keywords`, and
    `"mem0ai"` appears inside a prose comment as `importlib.metadata.version("mem0ai")`. Matching
    the whole file reported both as declarations carrying no upper bound.
    """
    kept: list[str] = []
    in_optional = False   # inside [project.optional-dependencies]
    in_array = False      # inside a `dependencies = [` ... `]` array
    for raw in PYPROJECT.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0] if raw.lstrip().startswith("#") else raw
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and "=" not in stripped:
            in_optional = stripped == "[project.optional-dependencies]"
            in_array = False
            continue
        if in_optional:
            kept.append(line)
            continue
        if re.match(r"^dependencies\s*=", stripped):
            in_array = True
            depth = 0
        if in_array:
            kept.append(line)
            # Bracket DEPTH, not "contains a ]": the first element of `[project].dependencies` is
            # `"psycopg[binary]>=3.3.4"`, whose own bracket used to close the region on line one
            # and hide every core dependency after it (pgvector among them) from both checks.
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                in_array = False
    return "\n".join(kept)


def _requirements(name: str) -> list[str]:
    """Every declared requirement string for `name`, across dependencies and every extra."""
    text = _dependency_text()
    # Requirement strings are quoted and start with the exact package name, followed by a version
    # operator or the closing quote. The name boundary matters: `mcp` must not match `mcp-foo`.
    pattern = re.compile(rf'"{re.escape(name)}(?![-\w.])([^"]*)"')
    return [m.group(1) for m in pattern.finditer(text)]


def test_there_are_ignore_rules_to_check() -> None:
    assert _ignored_names(), "no ignore rules found; the regex or the file layout probably changed"


@pytest.mark.parametrize("name", _ignored_names(), ids=str)
def test_ignored_dependency_is_actually_declared(name: str) -> None:
    assert _requirements(name), (
        f"dependabot.yml ignores {name!r}, but pyproject.toml declares no such dependency. "
        f"Either the name is misspelled — in which case the rule does nothing and the PRs keep "
        f"coming — or the dependency was dropped and the rule should go with it."
    )


@pytest.mark.parametrize("name", _ignored_names(), ids=str)
def test_ignored_dependency_still_has_an_upper_bound(name: str) -> None:
    """The rule exists to defend a ceiling. No ceiling, no reason to withhold updates.

    EVERY declaration must carry the bound, not merely one of them. `mcp` is declared twice —
    once in the `mcp` extra and once in `dev` — and both sites carry a comment insisting they be
    kept in step. Asserting only that SOME declaration is capped let the `dev` copy lose its `<2`
    silently, which is the copy `test` and `typecheck` actually install.
    """
    specs = _requirements(name)
    assert specs, f"no requirement strings found for {name!r}"
    uncapped = [s for s in specs if "<" not in s and "==" not in s]
    assert not uncapped, (
        f"dependabot.yml ignores {name!r}, but these requirement strings in pyproject.toml carry "
        f"no upper bound: {uncapped} (all declarations: {specs}). A cap that holds in one extra "
        f"and not another is not a cap — the uncapped one is what some install path resolves. If "
        f"the cap was lifted deliberately, delete the ignore rule in the same change; leaving it "
        f"withholds real updates and nothing reports that it is happening."
    )
