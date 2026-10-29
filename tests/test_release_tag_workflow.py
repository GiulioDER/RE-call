"""The publish workflow must wire the release tag guard before building artifacts."""

from __future__ import annotations

import pathlib
import re

WORKFLOW = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"


def _jobs(workflow: str) -> dict[str, str]:
    """Each top-level job's text, keyed by job id (the two-space keys under `jobs:`)."""
    body = workflow[workflow.index("\njobs:\n") :]
    starts = list(re.finditer(r"^  ([A-Za-z0-9_-]+):\s*$", body, re.MULTILINE))
    return {
        match.group(1): body[match.start() : starts[i + 1].start() if i + 1 < len(starts) else None]
        for i, match in enumerate(starts)
    }


def test_release_workflow_guards_tag_builds_before_building() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    guard = workflow.index("name: Verify release tag matches package metadata")
    build = workflow.index("name: Build sdist and wheel")
    guarded_section = workflow[guard:build]

    assert "if: startsWith(github.ref, 'refs/tags/')" in guarded_section
    assert 'run: python scripts/verify_release_tag.py "$GITHUB_REF_NAME"' in guarded_section


def test_only_the_attest_job_can_mint_an_oidc_token_for_provenance() -> None:
    """A job-level `id-token: write` reaches every step of that job, run-time tools included.

    PR 665 put `id-token: write` and `attestations: write` on the build job, which also runs
    `uv build` and `uvx twine`, so a compromised build or check tool could mint a token and sign
    provenance for bytes of its choosing (CCA run 20261029T210323Z, SEC-001). The attestation
    now runs alone in `attest`, and `publish` waits for it so nothing ships unattested.

    Red proof: against release.yml at 8c296fd31 this fails in its first assertion, because the
    build job declares `id-token: write`; restored, it passes. Comment lines are dropped first,
    so a comment naming the permission is not mistaken for the grant.
    """
    jobs = {
        name: "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        for name, text in _jobs(WORKFLOW.read_text(encoding="utf-8")).items()
    }

    assert not re.search(r"^\s*(id-token|attestations):", jobs["build"], re.MULTILINE)
    assert "attest-build-provenance" not in jobs["build"]
    attest = jobs["attest"]
    assert re.search(r"^\s*id-token:\s*write\s*$", attest, re.MULTILINE)
    assert re.search(r"^\s*attestations:\s*write\s*$", attest, re.MULTILINE)
    assert "attest-build-provenance" in attest
    assert re.search(r"needs:\s*build\b", attest)
    assert "run:" not in attest
    assert re.search(r"needs:\s*\[[^\]]*\battest\b[^\]]*\]", jobs["publish"])
