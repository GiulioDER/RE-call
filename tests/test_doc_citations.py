"""The documentation citation checker: that it catches drift, and that it stays quiet otherwise.

Both halves matter and the second is the one that decides whether the check survives. A citation
checker that cries wolf gets switched off, and then the coverage is gone rather than merely noisy.
That is not a hypothetical here: the FIRST implementation of this checker matched each citation
against a backticked anchor near it, and produced **33 findings on a tree whose citations had just
been repaired by hand**, nearly all of them correct citations flagged wrongly, because one
documentation line routinely carries several symbols and several citations with no reliable pairing
between them. It was replaced with the git line mapping the tests below exercise.

The failure this exists to prevent is real and recurring: one commit (`79a0d6ed`) shifted
`recall/index.py` by 26 lines and broke 23 citations across 5 documents at once.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# ⚠️ `scripts/` must be importable before the module below is executed, because it does
# `from check_citation_anchors import FROZEN_PREFIXES`. Running the checker as CI runs it
# (`python scripts/check_doc_citations.py`) puts `scripts/` on `sys.path[0]` for free; pytest does
# not, and importing by file location does not either.
#
# Without this line the module still imported, for a reason that is pure accident:
# `tests/test_citation_anchors.py` sorts before this file, and it registers
# `sys.modules["check_citation_anchors"]` as a side effect of its own setup. So a full-suite run
# passed while `pytest tests/test_doc_citations.py` on its own failed at COLLECTION, which is the
# worst shape for this: CI stayed green, and anyone running one file, renaming either file, or
# randomising order got an error that says nothing about what is wrong. Measured on master before
# this line was added: alone, 1 collection error; preceded by the anchors file, 59 passed.
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "check_doc_citations", REPO / "scripts" / "check_doc_citations.py"
)
assert _spec and _spec.loader
citations = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_citations"] = citations
_spec.loader.exec_module(citations)


# --- the arithmetic, which is where a wrong answer would be most confidently wrong ---------------

def test_a_line_before_every_hunk_does_not_move() -> None:
    """Edits later in a file cannot move a citation earlier in it."""
    assert citations.resolve(5, [(10, 0, 10, 26)]) == 5


def test_an_insertion_above_pushes_a_line_down() -> None:
    """The measured case: #381 inserted 26 lines and every citation below moved by 26."""
    assert citations.resolve(671, [(430, 0, 430, 26)]) == 697


def test_a_deletion_above_pulls_a_line_up() -> None:
    assert citations.resolve(100, [(10, 5, 10, 0)]) == 95


def test_several_hunks_accumulate() -> None:
    assert citations.resolve(500, [(10, 0, 10, 3), (100, 4, 103, 0), (200, 0, 199, 10)]) == 509


def test_a_line_inside_a_changed_hunk_is_unresolvable() -> None:
    """Stronger than a move: what was cited may not exist any more, so no number is offered."""
    assert citations.resolve(12, [(10, 5, 10, 5)]) is None


def test_a_line_at_the_last_position_of_a_hunk_is_still_inside_it() -> None:
    """An off-by-one here would report a confident wrong destination for an edited line."""
    assert citations.resolve(14, [(10, 5, 10, 5)]) is None
    assert citations.resolve(15, [(10, 5, 10, 5)]) == 15


# --- what counts as a citation -------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "see `recall/index.py:420`",
        "see `recall/index.py:420-478`",
        "see `recall_mcp/service.py:1659`",
        "see `recall/migrations/sql/0001_v08_baseline.sql:12`",
    ],
)
def test_a_repo_relative_citation_is_recognised(text: str) -> None:
    assert citations.CITATION.search(text) is not None


@pytest.mark.parametrize("text", ["see `cli.py:487`", "see `promotion.py:184`", "see `fix.py:24`"])
def test_a_bare_filename_is_not_treated_as_a_citation(text: str) -> None:
    """Documents use `cli.py:487` as shorthand for a module under discussion.

    Resolving those against the repository root reported "does not exist" for seven such
    references in the first implementation. A citation must carry a directory to be checkable, so
    the shorthand is left alone rather than failed.
    """
    assert citations.CITATION.search(text) is None


def test_a_prose_colon_is_not_a_citation() -> None:
    assert citations.CITATION.search("`recall/index.py` line 420") is None


# --- the policy file, which is what keeps frozen records out of the check ------------------------

def test_the_policy_declares_the_frozen_zone_and_its_marker_still_exists() -> None:
    """A marker that stops matching would silently un-freeze a registered prediction.

    `frozen_line` raises in that case rather than checking the zone, and this asserts the declared
    markers are all still present, so the raise is a real backstop and not the normal path.
    """
    exempt, zones = citations.load_policy()
    assert zones, "the frozen-zone declaration disappeared from docs/citation-policy.toml"
    for zone in zones:
        doc = REPO / zone["path"]
        assert doc.exists(), f"policy names a document that does not exist: {zone['path']}"
        boundary = citations.frozen_line(doc, zone["path"], zones)
        assert boundary is not None and boundary > 1
    assert exempt, "the exemption list disappeared"
    for rule in exempt:
        assert rule.get("reason"), f"exemption without a stated reason: {rule['path']}"


def test_a_vanished_marker_refuses_rather_than_passing() -> None:
    """The guard's own failure path, exercised: a stale policy must stop the run, not the check."""
    zone = {"path": "docs/citation-policy.toml", "marker": "## a marker that is not there"}
    with pytest.raises(SystemExit, match="frozen-zone marker"):
        citations.frozen_line(REPO / "docs" / "citation-policy.toml", "docs/citation-policy.toml",
                              [zone])


# --- the repository itself ------------------------------------------------------------------------

def test_every_citation_in_the_docs_still_resolves() -> None:
    """The gate. Fails when a commit moves code out from under a documented line number.

    Documents with uncommitted edits are skipped by the checker, so this passes locally while docs
    are being written and is only fully binding on a clean tree, which is what CI has.
    """
    findings, _skipped = citations.check()
    assert not findings, "\n" + "\n".join(f.render() for f in findings)


# --- a frozen PREFIX must not swallow a zone the policy declares live ----------------------------

def _init_repo(repo: Path) -> None:
    for args in (
        ("init", "--quiet"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
        ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _commit(repo: Path, message: str, *paths: str) -> None:
    subprocess.run(["git", "-C", str(repo), "add", *(paths or ("-A",))],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", message],
                   check=True, capture_output=True)


def _repo_with_docs(repo: Path, count: int) -> None:
    """A throwaway repo with `count` clean, committed documents that all cite one file."""
    (repo / "docs").mkdir(parents=True)
    (repo / "recall").mkdir(parents=True)
    (repo / "recall" / "thing.py").write_text(
        "import os\n\nTARGET = 1\n", encoding="utf-8", newline="\n"
    )
    for index in range(count):
        (repo / "docs" / f"d{index}.md").write_text(
            f"# d{index}\n\ncites `recall/thing.py:3`\n", encoding="utf-8", newline="\n"
        )
    _init_repo(repo)
    _commit(repo, "base", "docs", "recall")


def _count_spawns(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the git subcommand of every process the checker starts, in order."""
    seen: list[str] = []
    real_run = subprocess.run

    def counting(*args, **kwargs):
        argv = list(args[0]) if args else []
        # ["git", "-C", <repo>, "<subcommand>", ...]
        seen.append(argv[3] if len(argv) > 3 else "?")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(citations.subprocess, "run", counting)
    return seen


def test_the_status_query_is_one_spawn_for_the_whole_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One `git status` for the run, not one per document. This is a COST test, and it is load bearing.

    Process creation is this checker's entire cost on Windows: measured 2026-08-20, 178 spawns at
    141ms each with 98% of wall clock inside `subprocess.run`. The per document `status` calls were
    61 of those. That is not merely slow, because the cost is nonlinear in machine load: under 12
    way CPU saturation the same run took 289s rather than 25s, and the test that drives this checker
    carries no timeout of its own, so it runs under the 120s default and a loaded machine turns a
    passing check into a killed run.

    ⚠️ Asserted as `== 1`, not `< len(docs)`. A bound that merely beats the old number would go on
    passing if someone reintroduced a per document call for a subset of documents, which is exactly
    how this kind of regression comes back.
    """
    repo = tmp_path / "repo"
    _repo_with_docs(repo, count=6)
    monkeypatch.setattr(citations, "REPO", repo)
    monkeypatch.setattr(citations, "POLICY", repo / "docs" / "absent-policy.toml")

    spawns = _count_spawns(monkeypatch)
    citations.check()

    assert spawns.count("status") == 1, (
        f"expected exactly ONE `git status` for the whole tree, got {spawns.count('status')} "
        f"across 6 documents. Spawn sequence: {spawns}"
    )


def test_a_dirty_document_is_still_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batching the query must not change the ANSWER: an edited document is still not checked.

    The reason is in `check`: the baseline is the last commit that touched the document, so
    citations sitting in the working tree are newer than that baseline and would be reported as
    drift for having just been repaired. Six hand corrected citations were all reported stale that
    way, and all six findings vanished on commit.
    """
    repo = tmp_path / "repo"
    _repo_with_docs(repo, count=3)
    (repo / "docs" / "d1.md").write_text(
        "# d1\n\nedited in the working tree, citing `recall/thing.py:3`\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(citations, "REPO", repo)
    monkeypatch.setattr(citations, "POLICY", repo / "docs" / "absent-policy.toml")

    _findings, skipped = citations.check()

    assert skipped == ["docs/d1.md"], (
        f"only the edited document may be skipped, got {skipped}"
    )


def test_a_dirty_path_containing_a_space_is_recognised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the query is `--porcelain -z` and not `--porcelain`.

    ⚠️ Verified against git rather than assumed: plain `--porcelain` renders this path as
    `"docs/a doc with spaces.md"`, **with the quotes**, so a membership test against the raw path
    misses it. The document would then be checked against a baseline that no longer describes it,
    and the findings would be reported against the wrong text. `-z` emits the path raw and
    NUL separated, which is the only form that survives spaces, non ASCII bytes and `core.quotepath`.
    """
    repo = tmp_path / "repo"
    _repo_with_docs(repo, count=1)
    spaced = repo / "docs" / "a doc with spaces.md"
    spaced.write_text("# spaced\n", encoding="utf-8", newline="\n")
    _commit(repo, "add the spaced document", "docs")
    spaced.write_text("# spaced, now edited\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(citations, "REPO", repo)

    assert "docs/a doc with spaces.md" in citations.dirty_paths(), (
        "a dirty path containing a space was not recognised, which is what plain `--porcelain` "
        "does: it quotes the path and the membership test silently misses it"
    )


def test_an_untracked_document_inside_an_untracked_directory_is_recognised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the query carries `-uall`, and the one case where batching changed the ANSWER.

    ⚠️ The call this replaced passed a pathspec, and a pathspec forces git to name the file. Asked
    about the whole tree instead, git collapses untracked directories and reports `docs/notes/`
    rather than the documents inside it, so every one of them would fall out of the membership test.
    Verified against git rather than reasoned about: without `-uall` the entry really is the
    directory.

    This is the only behavioural difference the batching introduced, which is why it gets its own
    test rather than being folded into the rename one.
    """
    repo = tmp_path / "repo"
    _repo_with_docs(repo, count=1)
    (repo / "docs" / "notes").mkdir()
    (repo / "docs" / "notes" / "fresh.md").write_text(
        "# fresh, never committed\n", encoding="utf-8", newline="\n"
    )

    monkeypatch.setattr(citations, "REPO", repo)

    assert "docs/notes/fresh.md" in citations.dirty_paths(), (
        "an untracked document inside an untracked directory was not recognised. Without `-uall` "
        "git reports the collapsed directory `docs/notes/` and the document is invisible here"
    )


def test_a_rename_records_both_of_its_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-z` puts the ORIGINAL path in the next field, and both names have to land in the set.

    Being in the set means "do not check this document". Naming both is the conservative direction:
    the alternative is checking one of them against a baseline that no longer describes it.
    """
    repo = tmp_path / "repo"
    _repo_with_docs(repo, count=2)
    subprocess.run(
        ["git", "-C", str(repo), "mv", "docs/d0.md", "docs/renamed.md"],
        check=True, capture_output=True,
    )

    monkeypatch.setattr(citations, "REPO", repo)
    dirty = citations.dirty_paths()

    assert "docs/renamed.md" in dirty, f"the new name is missing from {sorted(dirty)}"
    assert "docs/d0.md" in dirty, f"the original name is missing from {sorted(dirty)}"


def test_a_declared_live_zone_is_still_checked_under_a_frozen_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[[frozen_above]]` must survive `FROZEN_PREFIXES`, or it is dead config that reads as live.

    ⚠️ This is a regression test for a defect that was ON MASTER, found by building this repository
    and looking rather than by reading the loop. `FROZEN_PREFIXES` was checked BEFORE `frozen_line`,
    so a document under `docs/preregistrations/` was skipped whole and its declared live tail was
    never read. Both `[[frozen_above]]` entries live under that prefix, so the entire mechanism was
    unreachable, and **zero** findings came back for a stale citation in a zone the policy called
    live.

    Nothing reported the loss: the existing marker test calls `frozen_line` directly, so it kept
    passing, and the policy file kept describing a live zone that nothing checked. That combination,
    a silent coverage loss behind a still-green guard, is why this asserts on the FINDING and not on
    the classification.
    """
    repo = tmp_path / "repo"
    (repo / "recall").mkdir(parents=True)
    (repo / "docs" / "preregistrations").mkdir(parents=True)

    (repo / "recall" / "thing.py").write_text(
        "import os\n\nTARGET = 1\n", encoding="utf-8", newline="\n"
    )
    # The citation sits BELOW the marker: the zone the policy declares live.
    (repo / "docs" / "preregistrations" / "p.md").write_text(
        "# P\n\nthe prediction, citing `recall/thing.py:3`\n\n## Result\n\n"
        "the measurement, citing `recall/thing.py:3`\n",
        encoding="utf-8", newline="\n",
    )
    (repo / "docs" / "citation-policy.toml").write_text(
        '[[frozen_above]]\n'
        'path = "docs/preregistrations/p.md"\n'
        'marker = "## Result"\n'
        'reason = "test"\n',
        encoding="utf-8", newline="\n",
    )

    _init_repo(repo)
    _commit(repo, "base")
    # Move the cited line in a commit that does NOT touch the document, so the drift is real and the
    # document's baseline still predates it.
    (repo / "recall" / "thing.py").write_text(
        "import os\nimport sys\nimport json\n\nTARGET = 1\n", encoding="utf-8", newline="\n"
    )
    _commit(repo, "shift the cited line by two", "recall/thing.py")

    monkeypatch.setattr(citations, "REPO", repo)
    monkeypatch.setattr(citations, "POLICY", repo / "docs" / "citation-policy.toml")

    findings, skipped = citations.check()

    assert not skipped, skipped
    assert len(findings) == 1, (
        "expected exactly one finding, from the LIVE tail below the marker. Zero means the frozen "
        f"prefix swallowed the declared zone; two means the frozen head was checked too. got: "
        f"{[(f.doc_line, f.detail) for f in findings]}"
    )
    # Line 7 is the citation below `## Result`; line 3 is the one above it, which must stay silent.
    assert findings[0].doc_line == 7, (
        f"the finding came from line {findings[0].doc_line}, not the live tail's line 7. A finding "
        "from line 3 would mean the frozen PREDICTION was checked, which is the opposite failure."
    )
    assert "moved +2" in findings[0].detail, findings[0].detail
