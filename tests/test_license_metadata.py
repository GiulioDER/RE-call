"""Keep the repository's legal and packaging metadata on one license."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LICENSE_ID = "AGPL-3.0-only"
LICENSE_NAME = "GNU Affero General Public License v3.0"
LICENSE_TITLE = "GNU AFFERO GENERAL PUBLIC LICENSE"


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
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    commercial = " ".join(
        (ROOT / "COMMERCIAL_LICENSE.md").read_text(encoding="utf-8").split()
    )

    assert LICENSE_TITLE in license_text
    assert "Version 3, 19 November 2007" in license_text
    assert "Copyright 2026 Giulio D'Erme" in notice
    assert LICENSE_NAME in readme
    assert "COMMERCIAL_LICENSE.md" in readme
    assert "Releases published under Apache 2.0 remain available under Apache 2.0" in commercial
    assert "0.14.0 remains available under the PolyForm Noncommercial License 1.0.0" in commercial


def test_no_current_surface_still_describes_the_retired_noncommercial_license() -> None:
    """Every place that states RE-call's current license must state the new one.

    The 0.14.0 relicensing touched 34 files, and a footer or a docstring left behind would tell a
    reader the wrong terms. Historical records (the changelog and COMMERCIAL_LICENSE.md, which
    name 0.14.0's license on purpose) are the only places PolyForm may still appear.
    """
    surfaces = [
        ROOT / "README.md",
        ROOT / "NOTICE",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "MODEL_LICENSES.md",
        ROOT / "docs" / "WIZARD.md",
        ROOT / "recall" / "sparse.py",
        ROOT / "benchmarks" / "store_latency_share.py",
        ROOT / "scripts" / "encode_sparse.py",
        *sorted((ROOT / "site").rglob("*.html")),
    ]
    stale = [str(p.relative_to(ROOT)) for p in surfaces if "PolyForm" in p.read_text(encoding="utf-8")]
    assert stale == []


def test_contributions_are_licensed_to_the_copyright_holder_under_the_cla() -> None:
    """Dual licensing needs every contribution to be relicensable by the copyright holder.

    Without the grant in CLA.md, a single merged outside patch would be available only under the
    AGPL, and the commercial license could no longer cover the whole work.
    """
    cla = " ".join((ROOT / "CLA.md").read_text(encoding="utf-8").split())
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "Giulio D'Erme" in cla
    assert "sublicense" in cla
    assert "license terms other than the AGPL" in cla
    assert "CLA.md" in contributing
    assert "CLA.md" in (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")


#: Commits that are cited elsewhere as fixed versions: release 0.14.0, and the builds that served the
#: Agent Memory Leaderboard Cycle 2 runs. All were published under PolyForm Noncommercial.
POLYFORM_ERA_FIXED_COMMITS = {
    "c3c944ce3379ed68053a260baf2697703f2e210c": "v0.14.0",
    "5d82b5166d61d6b0478273621017d55b137fd9df": "aml-c2-full1-multimodal",
    "1f8666df2b04c34efd33153f0751c624fb17b8f6": "aml-c2-full1-coding",
    "3eb447c4ea1f04f41f941ad8f089c9a02106e763": "aml-c2-full1-textual",
    "efb79146821ba751bb56694c7f04d55105823c02": "aml-c2-full1-textual-early",
    "225e7509eedf130314e1eab2b06a412d14dc1aae": "aml-c2-full2 build",
}


def test_polyform_era_versions_are_also_granted_under_the_agpl() -> None:
    """Every PolyForm era version, and each cited fixed commit by name, is also AGPL-3.0.

    An open-source eligibility check reads the LICENSE file at the commit it was given, and at
    these commits that file is PolyForm. The grant is what makes them open source, so a commit
    missing from it would leave that entry's terms in doubt.
    """
    grant = " ".join((ROOT / "LICENSE_GRANT_AGPL.md").read_text(encoding="utf-8").split())
    commercial = " ".join((ROOT / "COMMERCIAL_LICENSE.md").read_text(encoding="utf-8").split())

    assert "AGPL-3.0-only" in grant
    assert "3e1adc25393b8f48721d797915aaf05ee33cf87d" in grant
    assert "does not withdraw it" in grant
    missing = [name for sha, name in POLYFORM_ERA_FIXED_COMMITS.items() if sha not in grant]
    assert missing == []
    assert "LICENSE_GRANT_AGPL.md" in commercial
