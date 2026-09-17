"""Keep the repository's legal and packaging metadata on one license."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LICENSE_ID = "PolyForm-Noncommercial-1.0.0"
LICENSE_NAME = "PolyForm Noncommercial License 1.0.0"


def test_project_metadata_declares_the_spdx_license_and_license_files() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["license"] == LICENSE_ID
    assert set(project["license-files"]) >= {"LICENSE", "NOTICE"}


def test_citation_and_plugin_metadata_use_the_same_license_identifier() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^license:\s*" + re.escape(LICENSE_ID) + r"\s*$", citation, re.M)

    for path in (
        ROOT / "plugin" / ".claude-plugin" / "plugin.json",
        ROOT / "codex-plugin" / ".codex-plugin" / "plugin.json",
    ):
        assert json.loads(path.read_text(encoding="utf-8"))["license"] == LICENSE_ID


def test_public_legal_surface_names_the_license_and_preserves_historical_terms() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commercial = " ".join(
        (ROOT / "COMMERCIAL_LICENSE.md").read_text(encoding="utf-8").split()
    )

    assert LICENSE_NAME in license_text
    assert "Required Notice: Copyright 2026 Giulio D'Erme" in notice
    assert LICENSE_NAME in readme
    assert "COMMERCIAL_LICENSE.md" in readme
    assert "Releases published under Apache 2.0 remain available under Apache 2.0" in commercial
