"""The parts of `recall.generation_build` the CLI cannot reach, and therefore cannot pin.

`tests/test_generation_build_assembly.py` drives this module through `recall generation build` and
is the regression net for the extraction. It can only express what argparse can express, which
leaves one new capability untested: `commit_root`. The CLI has exactly two settings for it, `"."`
and `None`, so the wizard's case — stamping the repository that actually holds the corpus, which
is not the directory the process was started in — is reachable from nowhere else.

Untested new capability is the gap that matters here. The commit is written onto every chunk of a
calibrated generation and is what a later reader uses to say where a hit came from, so a wrong one
is not a missing record: it is a confident and false one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from recall.generation_build import BuildRequest, build_provenance, chunker_for, embedder_identity
from recall.index import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS


def _git_repo(root: Path) -> str:
    """A real repository with one commit, so `head_commit` has something to read."""
    run = lambda *args: subprocess.run(  # noqa: E731 - a local alias, not an exported helper
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "test")
    run("config", "commit.gpgsign", "false")
    (root / "a.md").write_text("body\n", encoding="utf-8")
    run("add", "a.md")
    run("commit", "-q", "-m", "first")
    return run("rev-parse", "--short", "HEAD").stdout.strip()


def test_the_commit_is_read_from_the_root_the_caller_names(tmp_path: Path) -> None:
    """The wizard indexes somebody else's repository from its own working directory.

    Stamping the process's cwd would record the wizard's own commit on every chunk of the user's
    corpus — provenance that is present, well-formed and wrong, which is worse than absent. Two
    distinct repositories are built here precisely so a cwd-derived answer cannot pass: the
    assertion is equality with the CORPUS's sha, not merely that some sha was recorded.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    corpus_sha = _git_repo(corpus)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    elsewhere_sha = _git_repo(elsewhere)
    assert corpus_sha != elsewhere_sha, "the fixture cannot distinguish the two roots"

    stamped = build_provenance(BuildRequest(commit_root=str(corpus)))

    assert stamped["indexed_commit"] == corpus_sha


def test_no_commit_root_stamps_no_commit(tmp_path: Path) -> None:
    """`None` is how a caller says 'do not record one', and must not become a lookup of `"."`."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _git_repo(corpus)

    assert "indexed_commit" not in build_provenance(BuildRequest(commit_root=None))


def test_a_root_outside_any_repository_stamps_nothing_rather_than_a_placeholder(
    tmp_path: Path,
) -> None:
    """A stored placeholder is indistinguishable later from a commit that was genuinely read."""
    assert "indexed_commit" not in build_provenance(BuildRequest(commit_root=str(tmp_path)))


def test_an_unset_project_is_omitted_rather_than_stored_as_none(tmp_path: Path) -> None:
    provenance = build_provenance(BuildRequest(project=None, commit_root=None))
    assert provenance == {}

    named = build_provenance(BuildRequest(project="p", commit_root=None))
    assert named == {"project": "p"}


def test_the_request_defaults_are_the_repository_chunking_defaults() -> None:
    """The wizard builds from `BuildRequest()` with no flags, so its defaults ARE the pipeline.

    Pinned against `recall.index`'s constants rather than against the literals 800 and 80, so a
    change to the repository's chunking default cannot leave the wizard silently on the old one.
    """
    request = BuildRequest()

    assert request.chunker == "text"
    assert request.max_chars == DEFAULT_MAX_CHARS
    assert request.overlap == DEFAULT_OVERLAP_CHARS

    _, identity = chunker_for(request)
    assert dict(identity.configuration) == {
        "max_chars": DEFAULT_MAX_CHARS,
        "overlap": DEFAULT_OVERLAP_CHARS,
    }


def test_the_identity_helpers_do_not_read_the_environment(monkeypatch) -> None:
    """The wizard drives several corpora in one process, so no step may depend on a global.

    Not a style preference: `RECALL_ENV` and `RECALL_LOCAL_ALLOWLIST` both change what a build
    does, and both are process-wide. Every seam takes a parameter instead, and this is the arm
    that fails if one of them grows an `os.environ` read.
    """

    class _Embedder:
        name = "BAAI/bge-small-en-v1.5"
        dim = 384

    monkeypatch.setenv("RECALL_ENV", "production")
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", "/nowhere")

    identity = embedder_identity(_Embedder(), BuildRequest(unverified=True))

    assert identity.provider == "fastembed"
    assert identity.unverified_reason == "explicit development build"
