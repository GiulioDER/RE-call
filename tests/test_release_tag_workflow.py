"""The publish workflow must wire the release tag guard before building artifacts."""

from __future__ import annotations

import pathlib


def test_release_workflow_guards_tag_builds_before_building() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    guard = workflow.index("name: Verify release tag matches package metadata")
    build = workflow.index("name: Build sdist and wheel")
    guarded_section = workflow[guard:build]

    assert "if: startsWith(github.ref, 'refs/tags/')" in guarded_section
    assert 'run: python scripts/verify_release_tag.py "$GITHUB_REF_NAME"' in guarded_section
