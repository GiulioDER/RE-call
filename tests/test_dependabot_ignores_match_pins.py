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


def _requirements(name: str) -> list[str]:
    """Every declared requirement string for `name`, across dependencies and every extra."""
    text = PYPROJECT.read_text(encoding="utf-8")
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
    """The rule exists to defend a ceiling. No ceiling, no reason to withhold updates."""
    specs = _requirements(name)
    capped = [s for s in specs if "<" in s or "==" in s]
    assert capped, (
        f"dependabot.yml ignores {name!r}, but none of its requirement strings in "
        f"pyproject.toml carry an upper bound: {specs}. If the cap was lifted deliberately, "
        f"delete the ignore rule in the same change — leaving it withholds real updates and "
        f"nothing reports that it is happening."
    )
