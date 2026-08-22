"""`pyproject.toml` and `recall.__version__` must agree, because two things read different ones.

⛔ **This is not bookkeeping tidiness.** The two values feed different machinery:

* `pyproject.toml` builds the wheel, so it decides what version exists on PyPI.
* `recall.__version__` is what `recall/wizard/stack.py` pins into the Dockerfile it generates
  (`recall-rag[...]=={version}`) and what `_default_image` scopes the image tag to.

So a drift does not produce a cosmetic mismatch. It produces an installer that provisions a
container pinned to a version that either does not exist on PyPI — a stack that fails to build,
during somebody's first install — or exists and is a DIFFERENT recall than the one building the
generations, which is the silent version skew `dockerfile_text`'s own docstring says the pin exists
to prevent.

Found while releasing 0.9.7: bumping `pyproject.toml` left `__init__.py` at 0.9.6, and nothing in
the suite would have said so.
"""

from __future__ import annotations

import pathlib
import tomllib

import recall


def _declared() -> str:
    root = pathlib.Path(__file__).resolve().parent.parent
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document["project"]["version"])


def test_the_package_and_the_project_declare_the_same_version() -> None:
    assert recall.__version__ == _declared(), (
        f"recall.__version__ is {recall.__version__} and pyproject.toml says {_declared()}. The "
        "wheel is built from pyproject, and the installer pins recall.__version__ into the "
        "Dockerfile it writes, so this drift ships a stack that installs the wrong recall or none."
    )


def test_every_hand_maintained_copy_of_the_version_is_accounted_for() -> None:
    """⚠️ **There are FOUR copies, not two, and I asserted two.**

    The commit that added this file claimed the version "lived in two places". It lives in four:
    `pyproject.toml`, `recall/__init__.py`, `server.json` (three times) and `CITATION.cff`. CI found
    the two I had missed, because `tests/test_smoke.py` already guarded them — so the repository's
    coverage was better than my assessment of it, and the genuinely unguarded one was
    `recall/__init__.py`, which is exactly the one that slipped.

    This test fails when a NEW copy appears that nothing checks. It does not re-assert what
    `test_smoke.py` already covers; it asserts that the set of files carrying the version is the set
    somebody has thought about.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    declared = _declared()
    # `plugin/.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json` are the Claude Code
    # plugin and the marketplace entry that distributes it. Both are checked by
    # `tests/test_claude_code_plugin.py::test_plugin_and_marketplace_versions_track_the_package`,
    # which asserts each equals `recall.__version__`, so they meet this list's bar: somebody checks
    # that they agree.
    #
    # ⚠️ A stale copy here is worse than a stale one elsewhere, because a marketplace entry is
    # CACHED by every user who has added the marketplace. `/plugin marketplace update` is the only
    # thing that refreshes it, so a version this repository advertises wrongly keeps being served
    # from other people's machines after the fix lands here.
    known = {
        "pyproject.toml",
        "recall/__init__.py",
        "server.json",
        "CITATION.cff",
        "uv.lock",
        "plugin/.claude-plugin/plugin.json",
        ".claude-plugin/marketplace.json",
    }

    carrying: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".json", ".toml", ".cff", ".lock"}:
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in {".git", "build", "dist", "__pycache__", ".venv"} for part in path.parts):
            continue
        if relative.startswith(("tests/", "docs/", "results/", "benchmarks/")):
            continue
        # Skipped for the same reason `benchmarks/` is, and it is only outside that tree because
        # this repository keeps its scripts together. `agent_ab_build_workspaces.py` GENERATES a
        # benchmark fixture whose whole task is "bump recall/version.py from 0.9.7 to 0.9.8", so
        # the literal it carries is fixture DATA and must never track the package version: if the
        # package reached 0.9.8 and something helpfully updated this, the task would ask an agent
        # to bump a file that already says 0.9.8 and every session would pass for the wrong
        # reason. Not added to `known` above, because `known` means "somebody checks these agree"
        # and this one must NOT agree.
        if relative == "scripts/agent_ab_build_workspaces.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # The quoted or `version:` form only. A bare `0.9.7` inside prose is a historical
        # measurement — `recall/wizard/stack.py` is full of them — and rewriting those would
        # falsify a record of what was measured.
        if f'"{declared}"' in text or f"version: {declared}" in text or f"=={declared}" in text:
            carrying.add(relative)

    assert carrying <= known, (
        f"these files carry the version and nothing is known to check them: {sorted(carrying - known)}. "
        "Add them to `known` here only after adding a test that asserts they agree; the point of "
        "this list is that every copy has been thought about."
    )


def test_the_generated_dockerfile_pins_the_version_that_will_be_published() -> None:
    """The consequence, asserted rather than described.

    `dockerfile_text` takes the version from `recall.__version__`. If that is not the version
    `pyproject.toml` publishes, the generated stack pins something PyPI may not have.
    """
    from recall.wizard.stack import dockerfile_text

    assert "recall-rag[" in dockerfile_text()
    assert f"=={_declared()}" in dockerfile_text(), (
        "the Dockerfile the installer writes must pin the version this project publishes"
    )
