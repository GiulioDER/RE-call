"""`scripts/release.py`: the bump, and every precondition that stops a bad one.

The script's whole value is that it refuses. A bumper that always bumps is barely better than
`sed`, because the failures worth preventing are not "I mistyped the number", they are "I released
a version PyPI already has", "I released from a feature branch", and "I bumped six of seven files".
So most of this file is about refusals.

⚠️ **These tests never touch the repository's own files.** Every one copies the version sites into
`tmp_path` and points the module at that. A test that exercised a bumper in place would rewrite the
version of the checkout running it, and a failure partway through would leave it half-bumped.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import release  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def sandbox(tmp_path, monkeypatch) -> pathlib.Path:
    """A throwaway tree holding real copies of every version site, in a real git repository.

    Real copies rather than fixtures, because the patterns in `SITES` are matched against the exact
    syntax these files use, and a hand-written stand-in would let a pattern rot while the test kept
    passing against the shape it was written for.
    """
    for site in release.SITES:
        source = REPO / site.path
        target = tmp_path / site.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [999.0.0] - 2026-01-01\n\n- a section for the target version\n",
        encoding="utf-8",
        newline="\n",
    )

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    _git("init", "-q", "-b", "master", ".")
    _git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false",
         "commit", "-q", "--allow-empty", "-m", "base")

    monkeypatch.setattr(release, "REPO", tmp_path)
    return tmp_path


def _version_of(sandbox: pathlib.Path, path: str) -> str:
    document = json.loads((sandbox / path).read_text(encoding="utf-8"))
    return document.get("version") or document["plugins"][0]["version"]


# --------------------------------------------------------------------------------------------
# The bump itself.
# --------------------------------------------------------------------------------------------


def test_apply_bumps_every_site_including_the_pin(sandbox) -> None:
    """⛔ `server.json` carries the version THREE times and only two are `"version":` keys.

    The third is the package pin inside `runtimeArguments`. Missing it publishes a registry entry
    that advertises the new version while telling every client to install the previous one, which
    is the silent version skew `recall/wizard/stack.py` documents at length. The first draft of the
    script did exactly that, and `test_the_bumper_recognises_the_version_actually_in_each_file`
    caught it on the occurrence count.
    """
    release.apply("1.2.3")

    server = (sandbox / "server.json").read_text(encoding="utf-8")
    assert server.count('"version": "1.2.3"') == 2
    assert "recall-rag[mcp,fastembed]==1.2.3" in server
    assert "0.9" not in server, "an old version survived somewhere in server.json"

    assert 'version = "1.2.3"' in (sandbox / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "1.2.3"' in (sandbox / "recall/__init__.py").read_text(encoding="utf-8")
    assert "version: 1.2.3" in (sandbox / "CITATION.cff").read_text(encoding="utf-8")
    assert _version_of(sandbox, "plugin/.claude-plugin/plugin.json") == "1.2.3"
    assert _version_of(sandbox, ".claude-plugin/marketplace.json") == "1.2.3"


def test_the_bumped_files_are_still_valid_json(sandbox) -> None:
    """A regex bump that breaks the syntax is worse than no bump: it fails at publish time."""
    release.apply("1.2.3")
    for site in release.SITES:
        if site.path.endswith(".json"):
            json.loads((sandbox / site.path).read_text(encoding="utf-8"))


def test_apply_writes_lf_endings(sandbox) -> None:
    """This repository pins LF in every writer. A platform newline rewrites every line."""
    release.apply("1.2.3")
    for site in release.SITES:
        assert b"\r\n" not in (sandbox / site.path).read_bytes(), site.path


def test_a_bump_does_not_touch_prose_versions(sandbox) -> None:
    """⚠️ A bare version in prose is a historical measurement, not a declaration.

    `recall/wizard/stack.py` is full of them, and rewriting one falsifies a record of what was
    measured. The patterns are anchored on surrounding syntax for exactly this reason.
    """
    note = sandbox / "recall" / "__init__.py"
    original = note.read_text(encoding="utf-8")
    note.write_text(
        original + '\n# Measured on 0.9.8: the installer could not open its own window.\n',
        encoding="utf-8",
        newline="\n",
    )

    release.apply("1.2.3")
    after = note.read_text(encoding="utf-8")
    assert "Measured on 0.9.8" in after, "a prose version was rewritten"
    assert '__version__ = "1.2.3"' in after


# --------------------------------------------------------------------------------------------
# The refusals, which are the point.
# --------------------------------------------------------------------------------------------


def test_refuses_a_version_that_is_not_x_y_z(sandbox) -> None:
    with pytest.raises(release.Refusal, match="X.Y.Z"):
        release.check_preconditions("0.9.9rc1", allow_dirty=True)


def test_refuses_going_backwards(sandbox) -> None:
    """⛔ PyPI refuses a version it has seen, and it refuses it AFTER the tag exists.

    So the failure lands with a tag pushed, a GitHub Release possibly created, and nothing on PyPI.
    """
    with pytest.raises(release.Refusal, match="not greater"):
        release.check_preconditions("0.0.1", allow_dirty=True)


def test_refuses_the_version_already_declared(sandbox) -> None:
    current = release.current_version()
    with pytest.raises(release.Refusal, match="already says"):
        release.check_preconditions(current, allow_dirty=True)


def test_refuses_when_the_tag_already_exists(sandbox) -> None:
    """Re-cutting an existing tag is how a published version gets re-pointed at different code."""
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "tag", "v999.0.0"],
        cwd=sandbox,
        check=True,
        capture_output=True,
    )
    with pytest.raises(release.Refusal, match="already exists"):
        release.check_preconditions("999.0.0", allow_dirty=True)


def test_refuses_a_dirty_tree_unless_told_otherwise(sandbox) -> None:
    """A release commit should contain the bump and nothing else."""
    (sandbox / "stray.txt").write_text("uncommitted", encoding="utf-8")

    with pytest.raises(release.Refusal, match="dirty"):
        release.check_preconditions("999.0.0", allow_dirty=False)

    assert release.check_preconditions("999.0.0", allow_dirty=True) is not None


def test_refuses_without_a_changelog_section(sandbox) -> None:
    """`release.yml` builds the GitHub Release body from it, falling back to generated notes."""
    (sandbox / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8", newline="\n")
    with pytest.raises(release.Refusal, match="CHANGELOG"):
        release.check_preconditions("999.0.0", allow_dirty=True)


def test_refuses_a_breaking_change_shipped_as_a_patch_bump(sandbox) -> None:
    """A BREAKING note under [Unreleased] plus a patch bump is refused: pre-1.0 the break goes
    in the MINOR, or the registry pin ships it to clients disguised as a patch."""
    current = release.current_version()  # 0.9.8 from the copied pyproject
    major, minor, patch = (int(p) for p in current.split("."))
    patch_bump = f"{major}.{minor}.{patch + 1}"
    (sandbox / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n### Security\n\n* **BREAKING: a scope now required.**\n\n"
        f"## [{patch_bump}] - 2026-01-01\n\n- the section\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(release.Refusal, match="BREAKING"):
        release.check_preconditions(patch_bump, allow_dirty=True)

    # The same BREAKING note is fine on a minor bump.
    minor_bump = f"{major}.{minor + 1}.0"
    (sandbox / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n### Security\n\n* **BREAKING: a scope now required.**\n\n"
        f"## [{minor_bump}] - 2026-01-01\n\n- the section\n",
        encoding="utf-8",
        newline="\n",
    )
    assert release.check_preconditions(minor_bump, allow_dirty=True) is not None


def test_a_non_master_branch_is_a_note_and_not_a_refusal(sandbox) -> None:
    """⚠️ Deliberately advisory. A hotfix cut from a branch is legitimate, and refusing it would
    make the script something you work around rather than something you use."""
    subprocess.run(["git", "checkout", "-q", "-b", "hotfix"], cwd=sandbox, check=True,
                   capture_output=True)
    notes = release.check_preconditions("999.0.0", allow_dirty=True)
    assert any("master" in note for note in notes)


def test_a_site_whose_shape_changed_is_refused_not_skipped(sandbox) -> None:
    """⛔ The failure this script exists to prevent is a PARTIAL bump.

    So a file that no longer matches its expected occurrence count stops the release rather than
    being quietly left behind at the old version.
    """
    server = sandbox / "server.json"
    document = json.loads(server.read_text(encoding="utf-8"))
    del document["version"]
    server.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(release.Refusal, match="matched"):
        release.plan("1.2.3")


def test_a_site_carrying_two_different_versions_is_refused(sandbox) -> None:
    """Already-drifted files are the input this script is most likely to meet, and bumping one of
    two mismatched values would hide the drift rather than fix it."""
    server = sandbox / "server.json"
    server.write_text(
        server.read_text(encoding="utf-8").replace('"version": "', '"version": "9.', 1),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(release.Refusal, match="several versions"):
        release.plan("1.2.3")


# --------------------------------------------------------------------------------------------
# The output, which crashed a real run after the files were already written.
# --------------------------------------------------------------------------------------------


def test_the_cli_prints_its_closing_advice_on_a_cp1252_console(sandbox, tmp_path) -> None:
    """⛔ It did not, and it failed at the worst possible moment.

    The closing advice carries a `⚠️`. A Windows console is cp1252, so `print` raised
    `UnicodeEncodeError` with seven files already bumped and the tag instructions never shown: a
    traceback that reads as though the bump itself failed.

    Run in a subprocess with `PYTHONIOENCODING=cp1252`, because the encoding is a property of the
    interpreter's streams and cannot be faked convincingly in-process.

    ⚠️ **`--apply` is load-bearing, and the first version of this test omitted it.** The closing
    advice is only printed on the apply path, so a dry run never reaches the character that
    crashes. Removing the `reconfigure` from the script left this test green, which is the
    "guard that cannot fail" this repository has a name for. Caught by mutation.

    ⛔ **The script is COPIED into the sandbox and run from there**, because `release.py` resolves
    `REPO` from its own `__file__`. Running the repository's copy with `cwd=sandbox` would have
    bumped this checkout's real version files the moment `--apply` was added.
    """
    import os
    import shutil

    (sandbox / "scripts").mkdir(exist_ok=True)
    shutil.copy(REPO / "scripts" / "release.py", sandbox / "scripts" / "release.py")

    result = subprocess.run(
        [sys.executable, "scripts/release.py", "999.0.0", "--allow-dirty", "--apply"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )

    assert "UnicodeEncodeError" not in (result.stderr or ""), result.stderr
    assert result.returncode == 0, result.stderr
    # The advice actually printed, rather than the process merely surviving.
    assert "permanent" in result.stdout
    # And it really did apply, so the crash path was genuinely exercised.
    assert 'version = "999.0.0"' in (sandbox / "pyproject.toml").read_text(encoding="utf-8")
    # ...in the sandbox ONLY. This assertion is the one that would have caught the copy running
    # against the real checkout, which is a failure a passing test would otherwise have hidden.
    assert 'version = "999.0.0"' not in (REPO / "pyproject.toml").read_text(encoding="utf-8")


def test_the_dry_run_writes_nothing(sandbox, capsys) -> None:
    """The default has to be safe: a bumper whose default mutates is one you run once by accident."""
    before = {site.path: (sandbox / site.path).read_bytes() for site in release.SITES}

    assert release.main(["999.0.0", "--allow-dirty"]) == 0

    for path, content in before.items():
        assert (sandbox / path).read_bytes() == content, f"{path} changed during a dry run"
    assert "Nothing written" in capsys.readouterr().out
