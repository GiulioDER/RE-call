"""The publish workflow must wire the release tag guard before building artifacts."""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

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


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("sha256sum") is None,
    reason="needs bash and sha256sum, as the release runner has",
)
def test_release_checksums_verify_beside_the_downloaded_assets(tmp_path: pathlib.Path) -> None:
    """SHA256SUMS must check out in a directory holding the assets as the Release names them.

    `gh release create` uploads each file by its basename, so a user downloads
    `recall_rag-X.whl` and SHA256SUMS side by side. The step wrote `dist/packages/...` paths,
    and `sha256sum -c` found none of them (CCA deferred run 20261029T222748Z, DEPLOY-001).
    This runs the workflow's own command, then verifies the way a user would. Red proof:
    against release.yml at 8c296fd31 the returncode assertion fails; restored, it passes.
    """
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow[workflow.index("name: Generate release checksums") :]
    command = re.search(r"^\s*run:\s*(.+)$", step, re.MULTILINE).group(1)  # type: ignore[union-attr]
    build = tmp_path / "build"
    (build / "dist" / "packages").mkdir(parents=True)
    for name in ("recall_rag-9.9.9-py3-none-any.whl", "recall_rag-9.9.9.tar.gz"):
        (build / "dist" / "packages" / name).write_bytes(name.encode())
    subprocess.run(["bash", "-c", command], cwd=build, check=True)

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for asset in [*(build / "dist" / "packages").iterdir(), build / "dist" / "SHA256SUMS"]:
        shutil.copy(asset, downloads / asset.name)
    verified = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=downloads, capture_output=True, text=True
    )

    assert verified.returncode == 0, verified.stdout + verified.stderr
