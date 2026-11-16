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
    assert "needs.sign.result == 'success'" in gate, (
        "and an unsigned or failed build must never publish an asset"
    )

    signing = [
        step
        for step in workflow["jobs"]["sign"]["steps"]
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


def test_a_tag_without_signpath_warns_and_still_attaches_nothing() -> None:
    """Owner decision 2026-10-28: until SignPath is configured, a tag push warns instead of failing.

    Invariant: on `push` with `vars.SIGNPATH_ORGANIZATION_ID` empty, the warning step runs and the
    refusal step does not, while a manual run without `allow_unsigned` is still refused and the
    `attach` job still requires a signed bundle. Failure caught: every tag from v0.10.0 to v0.14.0
    failed this job only because the credentials did not exist yet. Red against the workflow
    before the change: "a tag without SignPath has no warning step".
    """
    steps = {step.get("name"): step for step in _workflow()["jobs"]["build"]["steps"]}
    warn = steps.get("Note that this release carries no Windows installer")
    assert warn is not None, "a tag without SignPath has no warning step"
    assert "vars.SIGNPATH_ORGANIZATION_ID == ''" in warn["if"] and "github.event_name == 'push'" in warn["if"]
    assert "exit 1" not in warn["run"]
    refuse = steps["Refuse an unsigned release"]["if"]
    assert "github.event_name == 'workflow_dispatch'" in refuse, "a tag push is refused again"
    assert "inputs.allow_unsigned != true" in refuse, "a manual unsigned run is no longer refused"
    assert "needs.sign.result == 'success'" in _workflow()["jobs"]["attach"]["if"]


def _runs(job: dict) -> list[str]:
    return [str(step["run"]) for step in job["steps"] if "run" in step]


def test_the_signing_credential_never_shares_a_job_with_an_install() -> None:
    """⛔ **The job that installs packages holds no signing credential, and the job that signs
    installs nothing.**

    Invariant: `SIGNPATH_API_TOKEN` and `id-token: write` appear only in a job with no checkout and
    no `run:` step that installs or executes Python. Failure caught: both sat in `build`, after a
    `pip install` of the newest packages on PyPI and a PyInstaller run, so code from a compromised
    release could rewrite the bundle before it was uploaded for signing and ship under the project's
    signature (the bee-js and lwd-uri-util reports of 2026-11-09 and 2026-11-10 are that shape).
    Red proof, 2026-11-12: `tests/test_installer_workflow.py::
    test_the_signing_credential_never_shares_a_job_with_an_install` against origin/master's
    windows-installer.yml (528d662a3) fails at the first assertion naming `build`.
    """
    for name, job in _workflow()["jobs"].items():
        text = yaml.safe_dump(job)
        # The `secrets.` reference, not the bare name: build's warning steps name the secret in text.
        holds = (
            "secrets.SIGNPATH_API_TOKEN" in text
            or job.get("permissions", {}).get("id-token") == "write"
        )
        if not holds:
            continue
        uses = [str(step.get("uses", "")) for step in job["steps"]]
        assert not any(u.startswith("actions/checkout@") for u in uses), (
            f"{name} holds the signing credential and checks out the repository"
        )
        for script in _runs(job):
            assert not any(word in script for word in ("pip", "uv ", "python", "PyInstaller")), (
                f"{name} holds the signing credential and installs or runs Python: {script[:80]}"
            )


def test_the_installer_is_built_only_from_hashed_packages() -> None:
    """⛔ **Every package frozen into the installer is pinned by hash.**

    Invariant: external installs require hashes. The local checkout is built with its
    dependencies and build backend already installed, without index access. Failure caught: the job
    ran `pip install ".[desktop,mcp,fastembed,documents]"` and `pip install "pyinstaller>=6.10"`,
    resolving whatever PyPI served that minute, with no lock and no cooldown. Red proof,
    2026-11-12: `tests/test_installer_workflow.py::test_the_installer_is_built_only_from_hashed_packages`
    against origin/master's windows-installer.yml (528d662a3) fails on the unhashed install.
    SEC-002 checkout control was red-proved by removing --no-index from the fixed
    workflow's local checkout install. It fails with "can still resolve unhashed build inputs".
    """
    scripts = _runs(_workflow()["jobs"]["build"])
    installs = [
        line.strip()
        for script in scripts
        for line in script.splitlines()
        if "pip install" in line or "locked_install.sh" in line
    ]
    assert installs, "the build installs nothing, so this test would observe nothing"
    for line in installs:
        if line.split()[-1] == ".":
            assert all(flag in line.split() for flag in (
                "--no-index", "--no-deps", "--no-build-isolation",
            )), "the checkout can still resolve unhashed build inputs (SEC-002)"
            continue
        assert "--require-hashes" in line or "locked_install.sh" in line, (
            f"unhashed install in the installer build: {line}"
        )


def test_the_lock_exporter_uses_the_pinned_actions_checksum_table() -> None:
    """SEC-001. The exporter must use executable bytes reviewed with the action.

    Red proof uses ac67de930a666e2264850c2c7cbcf5b1f2190540's workflow.
    Its setup-uv step has no version input and fails the latest-known assertion.
    """
    steps = _workflow()["jobs"]["build"]["steps"]
    exporter = next(step for step in steps if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"))
    assert exporter.get("with", {}).get("version") == "latest-known", (
        "the lock exporter can download an unreviewed uv release"
    )


def test_the_installer_cannot_resolve_unhashed_build_dependencies() -> None:
    """SEC-002. Every external build input is a hashed wheel.

    Red proof uses ac67de930a666e2264850c2c7cbcf5b1f2190540's workflow
    and install helper. Its editable checkout install leaves build isolation on
    and fails the explicit no-build-isolation assertion.
    """
    root = WORKFLOW.parents[2]
    scripts = _runs(_workflow()["jobs"]["build"])
    if any("locked_install.sh" in script for script in scripts):
        scripts.append((root / ".github/scripts/locked_install.sh").read_text(encoding="utf-8"))
    commands = [
        line.strip().split()
        for script in scripts
        for line in script.splitlines()
        if line.strip().startswith("pip install ")
    ]
    checkout = [command for command in commands if command[-1] == "."]
    assert checkout, "the checkout install was not inspected"
    for command in checkout:
        assert "--no-build-isolation" in command, "the checkout opens an unhashed build environment"
        assert "--no-deps" in command and "--no-index" in command
        assert "-e" not in command, "editable builds need an additional unmanaged backend requirement"
    external = [command for command in commands if command not in checkout]
    assert external, "the external build inputs were not inspected"
    for command in external:
        assert "--require-hashes" in command
        assert "--only-binary=:all:" in command, "an sdist can open a second build environment"
    requirements = root / ".github/requirements/pyinstaller.txt"
    assert "hatchling==" in requirements.read_text(encoding="utf-8")
    scripts_text = "\n".join(_runs(_workflow()["jobs"]["build"]))
    assert scripts_text.index(".github/requirements/pyinstaller.txt") < scripts_text.index(
        "--no-build-isolation"
    ), "the backend must be installed before building the checkout"


@pytest.mark.parametrize("path", [
    "uv.lock",
    "pyproject.toml",
    ".github/requirements/pyinstaller.txt",
])
def test_dependency_changes_select_the_windows_installer(path: str) -> None:
    """BUG-001. Dependency-only PRs must reach the Windows bundle selftest.

    Red proof uses ac67de930a666e2264850c2c7cbcf5b1f2190540's workflow.
    Each input misses all four original path filters.
    """
    from fnmatch import fnmatchcase

    workflow = _workflow()
    events = workflow.get("on", workflow.get(True))
    assert events is not None, "the workflow has no event configuration"
    assert any(fnmatchcase(path, pattern) for pattern in events["pull_request"]["paths"]), (
        f"{path} changes cannot select the Windows installer"
    )


def test_dependency_prs_without_secrets_keep_their_installer_build_unsigned() -> None:
    """BUG-001. Adding dependency paths must preserve secret-free PR builds.

    Red proof uses the pinned workflow's organization-only signing condition.
    It omits both PR secret-availability guards asserted below.
    """
    gate = _workflow()["jobs"]["sign"]["if"]
    assert "github.event_name != 'pull_request'" in gate
    assert "github.event.pull_request.head.repo.full_name == github.repository" in gate
    assert "github.actor != 'dependabot[bot]'" in gate
