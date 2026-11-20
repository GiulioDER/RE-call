"""The skills-only plugin under `directory-plugin/`, built for the Anthropic Claude directory.

The directory refused the full plugin (0.15.0, 2026-11-08) for LOCAL_CODE_HOOK_COMMAND and
LOCAL_CODE_MCP_SERVER: hooks and MCP servers that run on the user's computer are accepted only
from Verified organizations. Skills are accepted. So the one property this plugin exists to keep
is that it carries nothing that runs on its own, and a later edit that adds a hook or a server
"for convenience" would turn an accepted listing into a refused one without any local check
saying so: `claude plugin validate` passes either way.

Red proof, run 2026-11-20 against deliberate mutations of `directory-plugin/`:
- `test_the_plugin_is_skills_only`: adding `"mcpServers": {"memory": {"command": "recall-mcp"}}`
  to `.claude-plugin/plugin.json` failed at the manifest-key assertion; creating
  `hooks/hooks.json` failed at the file-shape assertion.
- `test_no_skill_runs_a_shell_line_when_it_loads`: adding a line "!`recall --version`" to
  `skills/install-re-call/SKILL.md` failed at its assertion.
- `test_the_plugin_install_lines_are_the_ones_setup_prints`: changing `recall@re-call` to
  `recall@recall` in that skill failed at its assertion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from recall.claude_code import PLUGIN_INSTALL_LINES

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / "directory-plugin"
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
FULL_MANIFEST = REPO / "plugin" / ".claude-plugin" / "plugin.json"

#: Manifest keys that make Claude Code start something on the user's machine.
LOCAL_CODE_KEYS = {"hooks", "mcpServers", "lspServers"}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _skills() -> list[Path]:
    return sorted((PLUGIN / "skills").glob("*/SKILL.md"))


def test_the_plugin_is_skills_only() -> None:
    assert not LOCAL_CODE_KEYS & set(_manifest()), "a hook or server here is refused by the directory"

    files = {p.relative_to(PLUGIN).as_posix() for p in PLUGIN.rglob("*") if p.is_file()}
    allowed = {".claude-plugin/plugin.json", ".claude-plugin/icon.png", "README.md"}
    allowed |= {p.relative_to(PLUGIN).as_posix() for p in _skills()}
    assert files == allowed, f"unexpected files in the directory plugin: {sorted(files - allowed)}"


def test_no_skill_runs_a_shell_line_when_it_loads() -> None:
    """A "!`command`" line in a skill runs at load time, which is local code like a hook."""
    for skill in _skills():
        text = skill.read_text(encoding="utf-8")
        assert not re.search(r"!`", text), f"{skill} runs a command when it loads"


def test_every_skill_has_front_matter_with_a_description() -> None:
    assert len(_skills()) == 2
    for skill in _skills():
        text = skill.read_text(encoding="utf-8")
        header = text.split("---", 2)[1]
        assert f"name: {skill.parent.name}\n" in header
        assert re.search(r"^description: \S", header, re.M), f"{skill} has no description"


def test_every_skill_front_matter_parses_as_yaml() -> None:
    """The directory's validator flagged an unquoted description holding ": " (2026-11-20).

    Claude Code's loader accepted it, so only a strict YAML reader shows the defect.

    Red proof, 2026-11-20: run against `7db648a6a`, where the `install-re-call` description was
    unquoted, this failed at the safe_load call ("mapping values are not allowed here").
    """
    yaml = pytest.importorskip("yaml")
    for skill in _skills():
        header = skill.read_text(encoding="utf-8").split("---", 2)[1]
        data = yaml.safe_load(header)
        assert data["name"] == skill.parent.name
        assert isinstance(data["description"], str)


def test_the_name_is_distinct_from_the_full_plugin_and_the_taken_name() -> None:
    """The directory treats names differing only in case or punctuation as the same name."""
    name = _manifest()["name"]
    folded = re.sub(r"[^a-z0-9]", "", name.lower())
    assert folded != "recall"
    assert name != json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))["name"]


def test_the_license_matches_the_full_plugin() -> None:
    full = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    assert _manifest()["license"] == full["license"]


def test_the_plugin_install_lines_are_the_ones_setup_prints() -> None:
    text = (PLUGIN / "skills" / "install-re-call" / "SKILL.md").read_text(encoding="utf-8")
    for line in PLUGIN_INSTALL_LINES:
        assert line in text


def test_the_readme_clears_the_directory_minimum() -> None:
    """The directory blocks a README under 40 words, not counting code blocks."""
    text = (PLUGIN / "README.md").read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    assert len(prose.split()) >= 40
