"""The Windows installer workflow, asserted for the properties that fail SILENTLY.

A workflow is the one artefact in this repository whose bugs cannot be caught by running it: a
mistake in a `permissions:` block does nothing visible, and a mistake in an `if:` gate only shows up
on the one tag push a year where it matters. Nothing tested this file at all, and three separate
findings against it in one audit round were about exactly that.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / ".github/workflows/windows-installer.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_the_build_job_never_holds_a_repo_writable_token() -> None:
    """⛔ **A job-level `permissions:` block applies to EVERY step in that job.**

    The attach step was briefly added to the build job with `contents: write` and a comment claiming
    the grant was "scoped to this step". It is not, and a step-level `if:` gates execution rather
    than the token — GitHub Actions has no step-level `permissions` key at all. So the third-party
    signing action, both upload actions, and the step that runs repository code under PyInstaller
    all held a token that could write to the repository, on same-repo pull requests too. The
    fork-PR argument offered for it covers only forks.

    Three auditors reported it independently. The write grant now lives in a separate job that runs
    nothing but a download and an upload.
    """
    jobs = _workflow()["jobs"]

    assert jobs["build"]["permissions"]["contents"] == "read", (
        "the job that runs third-party actions and builds repository code must not be able to "
        "write to the repository"
    )
    assert "attach" in jobs, "the write-capable work belongs in its own job"
    assert jobs["attach"]["permissions"] == {"contents": "write"}, (
        "and that job should ask for nothing else"
    )
    assert jobs["attach"]["needs"] == "build" or "build" in jobs["attach"]["needs"]


def test_a_test_signed_bundle_can_never_reach_a_public_release() -> None:
    """⛔ **One predicate decides both the signing policy and the attach, or they can disagree.**

    The signing step selects `release-signing` on a push and `test-signing` otherwise. Gating the
    attach on the tag REF alone would let a `workflow_dispatch` aimed at a tag sign with the test
    policy while `startsWith(github.ref, 'refs/tags/v')` stayed true — attaching a test-signed
    binary to a public release, which is the one outcome code signing exists to prevent.
    """
    workflow = _workflow()
    gate = workflow["jobs"]["attach"]["if"]

    assert "github.event_name == 'push'" in gate, (
        "the attach must be gated on the same fact the signing policy is chosen by"
    )
    assert "needs.build.outputs.signed == 'true'" in gate, (
        "and an unsigned or failed build must never publish an asset"
    )

    signing = [
        step
        for step in workflow["jobs"]["build"]["steps"]
        if "signpath" in str(step.get("uses", "")).lower()
    ]
    assert signing, "the build must still sign"
    policy = str(signing[0]["with"]["signing-policy-slug"])
    assert "release-signing" in policy and "test-signing" in policy, (
        "both policies must appear in one expression; two separate decisions are what drift"
    )
    assert "github.event_name == 'push'" in policy, (
        "and it must be the SAME predicate the attach gate uses"
    )


def test_the_tag_is_never_interpolated_into_a_script_body() -> None:
    """⚠️ `${{ }}` is substituted into the script TEXT before any shell parses it.

    A tag carrying shell metacharacters would therefore become script rather than data. Tags here
    are pushed by maintainers, so this is a small hole — and a free one to close, which is the right
    trade on the one job in this workflow that holds a repo-writable token.
    """
    for job_name, job in _workflow()["jobs"].items():
        for step in job["steps"]:
            script = step.get("run")
            if not script:
                continue
            assert "${{ github.ref_name }}" not in script, (
                f"{job_name}/{step.get('name', '?')}: pass the tag through `env:` and read it as "
                "$TAG, rather than substituting it into the script before the shell sees it"
            )
