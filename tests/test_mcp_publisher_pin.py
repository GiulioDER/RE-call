"""The PR gate must run the same mcp-publisher the registry publish runs.

`ci.yml`'s mcp-metadata job exists so a server.json the registry would refuse fails on the PR,
not at publish time after PyPI already has the release. PR 664 pinned the publish workflow to one
checksummed release while the PR gate kept downloading `releases/latest`, so the two validators
drifted five releases apart with nothing to report it (CCA run 20261029T210323Z, DEP-001).

Red proof: against ci.yml at 8c296fd31 (the `releases/latest` curl) this test fails in its
`assert ci_pin == publish_pin` with `None != ('v1.8.1', 'a06c...')`; restored, it passes.
"""

from __future__ import annotations

import pathlib
import re

WORKFLOWS = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows"
_VERSION = re.compile(r"^\s*MCP_PUBLISHER_VERSION:\s*(\S+)\s*$", re.MULTILINE)
_SHA256 = re.compile(r"^\s*MCP_PUBLISHER_SHA256:\s*([0-9a-f]{64})\s*$", re.MULTILINE)


def _pin(name: str) -> tuple[str, str] | None:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    version, digest = _VERSION.findall(text), _SHA256.findall(text)
    if len(version) != 1 or len(digest) != 1:
        return None
    return version[0], digest[0]


def test_the_pr_gate_validates_with_the_publisher_that_publishes() -> None:
    publish_pin = _pin("publish-mcp.yml")
    ci_pin = _pin("ci.yml")

    assert publish_pin is not None
    assert ci_pin == publish_pin
    assert "releases/latest" not in (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
