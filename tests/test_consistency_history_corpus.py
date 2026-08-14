"""The emitter turns git history into the corpus shape a real agent memory has."""
from __future__ import annotations

from pathlib import Path

import pytest

import recall_consistency.history_corpus as history_corpus
from recall_consistency.history_corpus import (
    Revision,
    memo_stem,
    revisions,
    tracked_markdown,
    write_history_corpus,
)
from tests.consistency_helpers import (
    directory_then_file_repo,
    non_ascii_path_repo,
    symlink_then_file_repo,
    two_commit_repo,
)


def _repo(tmp_path: Path) -> Path:
    return two_commit_repo(tmp_path, "recall@5 is 0.92\n", "recall@5 is 0.945\n")


def test_revisions_returns_every_version_oldest_first(tmp_path: Path) -> None:
    revs = revisions(_repo(tmp_path), "notes.md")
    assert [r.body.strip() for r in revs] == ["recall@5 is 0.92", "recall@5 is 0.945"]


def test_tracked_markdown_lists_repo_relative_paths(tmp_path: Path) -> None:
    assert tracked_markdown(_repo(tmp_path)) == ["notes.md"]


def test_the_newer_memo_declares_supersession_over_the_older(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    revs = revisions(repo, "notes.md")
    out = tmp_path / "corpus"

    written = write_history_corpus(repo, ["notes.md"], out)

    assert len(written) == 2
    old_stem = memo_stem(revs[0].path, revs[0].date, revs[0].sha)
    new_stem = memo_stem(revs[1].path, revs[1].date, revs[1].sha)
    older = (out / f"{old_stem}.md").read_text(encoding="utf-8")
    newer = (out / f"{new_stem}.md").read_text(encoding="utf-8")
    assert "supersedes:" not in older
    assert f"valid_from: {revs[0].date}" in older
    assert f"supersedes: {old_stem}" in newer
    assert "recall@5 is 0.945" in newer


def test_memos_are_written_with_lf_endings(tmp_path: Path) -> None:
    """This repo is eol=lf. A scripted writer that emits CRLF churns every future diff."""
    written = write_history_corpus(_repo(tmp_path), ["notes.md"], tmp_path / "corpus")
    assert b"\r\n" not in written[0].read_bytes()


def test_a_revision_where_the_path_held_a_directory_is_skipped(tmp_path: Path) -> None:
    """`git show <sha>:<dir>` exits zero and prints a tree listing. That is not memo text.

    Delete the mode check and this test must go red: it is the only thing standing
    between a tree listing and the corpus.
    """
    revs = revisions(directory_then_file_repo(tmp_path), "notes.md")

    assert [r.body.strip() for r in revs] == ["recall@5 is 0.945"]


def test_a_revision_where_the_path_held_a_symlink_is_skipped(tmp_path: Path) -> None:
    """`cat-file -t` reports `blob` for a symlink same as a real file, so the mode is the only
    thing that separates them. Delete the mode check and this test must go red.
    """
    revs = revisions(symlink_then_file_repo(tmp_path), "notes.md")

    assert [r.body.strip() for r in revs] == ["recall@5 is 0.945"]


def test_a_non_ascii_tracked_path_is_returned_and_its_revisions_are_read(tmp_path: Path) -> None:
    """`core.quotePath` escapes non-ASCII paths on a plain listing; `-z` must not.

    Revert `tracked_markdown` to a newline-split listing and this goes red: the path comes back
    quoted and octal-escaped, and `revisions()` on that literal reads zero revisions.
    """
    repo = non_ascii_path_repo(tmp_path)

    assert tracked_markdown(repo) == ["café.md"]
    revs = revisions(repo, "café.md")
    assert [r.body.strip() for r in revs] == ["recall@5 is 0.92", "recall@5 is 0.945"]


def test_a_git_failure_names_the_failing_command_and_the_repository(tmp_path: Path) -> None:
    """Fatal is only useful if the operator can tell which document to exclude and re-run."""
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        revisions(not_a_repo, "notes.md")

    assert "git log" in str(excinfo.value)
    assert str(not_a_repo) in str(excinfo.value)


def test_memo_stem_refuses_a_path_that_would_split_frontmatter() -> None:
    with pytest.raises(ValueError) as excinfo:
        memo_stem(f"notes{chr(0x2028)}pwned.md", "2026-08-14", "abc1234")

    assert "line break" in str(excinfo.value)


def test_write_history_corpus_skips_one_poisoned_path_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _revisions(_repo: Path, rel_path: str) -> list[Revision]:
        return [
            Revision(
                path=rel_path,
                sha="1111111",
                date="2026-08-13",
                body="recall@5 is 0.92\n",
            ),
            Revision(
                path=rel_path,
                sha="2222222",
                date="2026-08-14",
                body="recall@5 is 0.945\n",
            ),
        ]

    monkeypatch.setattr(history_corpus, "revisions", _revisions)
    out = tmp_path / "corpus"

    written = write_history_corpus(
        tmp_path,
        [f"bad{chr(0x2028)}path.md", "good.md"],
        out,
    )

    assert len(written) == 2
    assert all(path.name.startswith("good__") for path in written)
    assert not list(out.glob(f"*{chr(0x2028)}*"))
    assert "skipping history corpus" in capsys.readouterr().err
