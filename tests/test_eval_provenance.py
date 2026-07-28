"""The block every result artifact embeds so it can be checked later.

No result JSON in results/ records the corpus it was measured against. That makes the
2026-07-27 doubled-corpus failure undetectable retroactively on every published number,
including postfix_pool20.json and postfix_abstention.json.

Pure functions only — this must be testable without a database, because it is the part that
has to work on every runner.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from recall.eval.provenance import _git_dirty, _git_sha, provenance_block


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A freshly initialised repo with one tracked file (`tracked.txt`, "v1\\n") committed.

    Skips (not fails) where `git` itself is not on PATH, matching how the rest of this file
    degrades to None instead of asserting a specific tool is present.
    """
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")

    (tmp_path / "tracked.txt").write_text("v1\n", encoding="utf-8")

    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)

    run("init", "-q")
    run("-c", "user.email=test@example.com", "-c", "user.name=Test", "add", "tracked.txt")
    run(
        "-c", "user.email=test@example.com", "-c", "user.name=Test",
        "commit", "-q", "-m", "initial",
    )
    return tmp_path


def test_block_carries_the_row_count_and_where_it_came_from():
    b = provenance_block(corpus_rows=5882, table="locomo_chunks", tenants=["locomo-conv-26"])
    assert b["corpus_rows"] == 5882
    assert b["table"] == "locomo_chunks"
    assert b["tenants"] == ["locomo-conv-26"]


def test_git_sha_is_present_so_a_result_names_the_tree_that_made_it():
    """Absent a repo it must degrade to None, never to a wrong or invented sha."""
    b = provenance_block(corpus_rows=1, table="t", tenants=[])
    assert "git_sha" in b
    assert b["git_sha"] is None or isinstance(b["git_sha"], str)


def test_git_dirty_is_present_so_a_clean_sha_cannot_be_mistaken_for_a_clean_tree():
    """Same tristate shape as `git_sha`: absent a repo it must degrade to None, never guess."""
    b = provenance_block(corpus_rows=1, table="t", tenants=[])
    assert "git_dirty" in b
    assert b["git_dirty"] is None or isinstance(b["git_dirty"], bool)


def test_git_dirty_is_none_exactly_when_git_sha_is_none(tmp_path: Path):
    """"Dirty relative to nothing" is not a fact — outside a repo both must degrade together.

    A lone `git_dirty=False` next to `git_sha=None` would read as "this untracked run was
    clean", which asserts something no commit backs. `tmp_path` is pytest's own per-test scratch
    directory — freshly created, so it is never itself inside a git work tree.
    """
    assert _git_sha(cwd=tmp_path) is None, "sanity check: tmp_path must not be inside a repo"
    assert _git_dirty(cwd=tmp_path) is None


def test_git_dirty_true_on_a_dirty_tree_false_on_a_clean_tree(git_repo: Path):
    """The decisive case for I2: prove `git_dirty` actually flips, and `git_sha` alone would not.

    Before this fix, `_git_sha()` returned the same clean-looking short sha whether the tree
    that produced a result was pristine or under active edit — "a wrong sha is worse than an
    absent one" (module docstring), and an unflagged dirty tree is exactly that: a sha that
    looks trustworthy and is not.
    """
    sha_clean = _git_sha(cwd=git_repo)
    dirty_clean = _git_dirty(cwd=git_repo)
    assert sha_clean is not None, "a freshly committed repo must resolve a sha"
    assert dirty_clean is False, "nothing changed since the commit — must read clean"

    # Edit the tracked file without committing: the exact "worktree under active edit" scenario
    # the fix exists for. A fresh eval run against this tree would produce real numbers that
    # this commit's tree does not contain.
    (git_repo / "tracked.txt").write_text("v2 — uncommitted edit\n", encoding="utf-8")

    sha_dirty = _git_sha(cwd=git_repo)
    dirty_dirty = _git_dirty(cwd=git_repo)

    assert sha_dirty == sha_clean, (
        "git_sha names the nearest commit and is not expected to change — this is exactly why "
        "git_sha alone cannot carry dirtiness; git_dirty must"
    )
    assert dirty_dirty is True, "a tracked-file edit since the commit must read dirty"
    assert dirty_clean != dirty_dirty, "clean and dirty states must be distinguishable"


def test_git_dirty_ignores_untracked_files(git_repo: Path):
    """Matches what `git describe --dirty` itself checks — tracked content only.

    A stray untracked file (a downloaded corpus, a log) sitting next to a clean checkout should
    not make a measured run look like it ran against edited code.
    """
    (git_repo / "untracked_scratch.log").write_text("noise\n", encoding="utf-8")

    assert _git_dirty(cwd=git_repo) is False


def test_tenants_are_sorted_so_two_runs_of_one_config_produce_equal_blocks():
    """Dict ordering must not make a diff of two identical runs look like a change."""
    a = provenance_block(corpus_rows=2, table="t", tenants=["b", "a"])
    b = provenance_block(corpus_rows=2, table="t", tenants=["a", "b"])
    assert a["tenants"] == b["tenants"] == ["a", "b"]


def test_a_zero_row_corpus_is_representable_not_swallowed():
    """0 is a real, alarming value. It must not be dropped as falsy."""
    b = provenance_block(corpus_rows=0, table="t", tenants=[])
    assert b["corpus_rows"] == 0


def test_every_locomo_runner_embeds_the_provenance_block():
    """A runner that writes a result JSON must embed the block.

    Checked by import rather than by running the benchmark: each run costs tens of minutes, so
    a test that ran one would never be run. This catches the realistic regression — a new
    runner, or a refactor that drops the call — which is exactly how the count failed to reach
    #103's artifacts in the first place.
    """
    import inspect

    from recall.eval import locomo, locomo_abstention, locomo_entailment_sweep

    for mod in (locomo, locomo_abstention, locomo_entailment_sweep):
        src = inspect.getsource(mod)
        assert "provenance_block(" in src, (
            f"{mod.__name__} writes a result artifact but does not embed provenance_block — "
            f"its output could not be told apart from a doubled-corpus run"
        )
