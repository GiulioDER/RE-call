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

import pytest

from recall.generation_build import (
    BuildRequest,
    build_provenance,
    chunker_for,
    embedder_identity,
    pipeline_identity,
)
from recall.index import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS
from recall.lineage import IndexManifestV1
from recall.manifest import load_inventory


class _FakeEmbedder:
    """A resolved non-hashing embedder, shaped only as far as the identity helpers reach."""

    name = "BAAI/bge-small-en-v1.5"
    dim = 384


def _git_repo(root: Path) -> str:
    """A real repository with one commit, so `head_commit` has something to read.

    The content is derived from the directory name, and that is not cosmetic. A commit object is
    a pure function of its tree, author, committer, parents, message and timestamp, so two repos
    built from identical content in the same second get the SAME sha — the timestamp has one-second
    resolution and everything else here was constant. The full suite caught it: two repositories
    that must be distinguishable came out as `fbe4868` twice. Distinct content makes them differ
    by construction rather than by how fast the machine ran.
    """
    run = lambda *args: subprocess.run(  # noqa: E731 - a local alias, not an exported helper
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "test")
    run("config", "commit.gpgsign", "false")
    # The FULL path, not `root.name`. A leaf name is unique for the two call sites that exist
    # today and collides again the moment a later test builds two repos called the same thing
    # under different parents, which is the identical flake one level along. `tmp_path` is unique
    # per test, so the full path cannot collide whatever a later caller does.
    (root / "a.md").write_text(f"body of {root}\n", encoding="utf-8")
    run("add", "a.md")
    run("commit", "-q", "-m", f"first commit in {root}")
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
    """A stored placeholder is indistinguishable later from a commit that was genuinely read.

    The precondition is asserted rather than assumed. This is an absence check with two ways to
    pass without exercising anything: `git rev-parse` walks UP to find a `.git`, so the result
    depends on no ancestor of pytest's tmp directory being a repository, and `head_commit` swallows
    `OSError`, so it would also pass on a runner with no git binary at all. The probe below rules
    out both, and fails loudly rather than silently, because a missing binary raises here.
    """
    probe = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        # Matching `head_commit`, which bounds its own call for the same reason: a git that blocks
        # on a credential helper or a stalled filesystem would hang the run rather than fail it,
        # and a TimeoutExpired here is as loud as the FileNotFoundError this probe relies on.
        timeout=15,
    )
    assert probe.returncode != 0, (
        f"{tmp_path} is inside a repository, so this test cannot demonstrate an absence"
    )

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


def test_an_unknown_chunker_is_refused_rather_than_silently_treated_as_text() -> None:
    """The validation argparse used to provide, which did not travel with the extraction.

    `ChunkerKind` is a `Literal` and erased at runtime. Without a check, `chunker="Code"` selects
    the text chunker AND records `recall.chunk_text`, so the callable and the immutable record
    agree with each other while both disagree with what was asked. Nothing downstream can see
    that, which is what makes it worth a refusal rather than a default.

    Case is included on purpose: a plausible hand edit of a resumable state file, not a random
    string, is the input this exists to catch.
    """
    for bad in ("Code", "TEXT", "codee", "", "chunk_code"):
        with pytest.raises(ValueError, match="chunker must be"):
            BuildRequest(chunker=bad)  # type: ignore[arg-type]

    # And the two real values still construct, so the guard is not simply refusing everything.
    assert BuildRequest(chunker="text").chunker == "text"
    assert BuildRequest(chunker="code").chunker == "code"


def test_the_chunker_decision_point_refuses_a_request_that_bypassed_the_constructor() -> None:
    """The constructor guard does not cover the path the wizard's resumable state will use.

    A frozen dataclass rebuilt by `copy.deepcopy`, or by any deserialiser that restores `__dict__`
    directly, never runs `__post_init__`. So `chunker_for` has to refuse too: otherwise a request
    that arrived from a file selects the TEXT chunker and records `recall.chunk_text`, the callable
    and the record agreeing with each other and both disagreeing with what was asked.

    `object.__new__` here stands in for any such reconstruction. The point is not that anybody
    calls it, but that a value can reach `chunker_for` without passing the constructor.
    """
    import dataclasses

    smuggled = object.__new__(BuildRequest)
    for field in dataclasses.fields(BuildRequest):
        object.__setattr__(smuggled, field.name, getattr(BuildRequest(), field.name))
    object.__setattr__(smuggled, "chunker", "Code")

    with pytest.raises(ValueError, match="unknown chunker"):
        chunker_for(smuggled)


def test_a_handed_in_chunker_identity_is_the_one_recorded() -> None:
    """`build_generation` holds the pair already, so it must not silently get a second one.

    `is` rather than `==`, because a recomputed identity would compare equal today and that is
    exactly what would hide the single-call property being lost again. The no-argument form is
    checked alongside, so the convenience path is not left to rot the way it did once already.
    """
    # `unverified` because a non-hashing embedder with no revision and no digest cannot form an
    # identity at all; the flag is incidental here and the chunker half is the subject.
    request = BuildRequest(chunker="code", max_chars=321, unverified=True)
    _, identity = chunker_for(request)

    handed_in = pipeline_identity(_FakeEmbedder(), request, identity)
    assert handed_in.chunker is identity

    computed = pipeline_identity(_FakeEmbedder(), request)
    assert computed.chunker == identity
    assert computed.embedder == handed_in.embedder


def test_a_directly_constructed_reader_needs_no_allowlist_variable(
    tmp_path: Path, monkeypatch
) -> None:
    """The precondition the whole no-environment design rests on, measured rather than reasoned.

    Both the design plan and the handoff state that `RECALL_LOCAL_ALLOWLIST` is mandatory for a
    `file://` manifest. It is mandatory only for `LocalObjectReader.from_environment`, which is
    what `reader_for_manifest` calls. `manager.build` takes the reader as a parameter, so a caller
    that already knows its corpus root passes one and the variable never enters the picture.

    That matters beyond tidiness: the wizard drives three corpora in one process, and a
    process-global allowlist would have to be set and restored around each build, which is the
    shape that produces a green run for the wrong reason. Both halves are asserted here, because
    "the direct reader works" is only interesting alongside "the environment one refuses".
    """
    import json

    from recall.manifest import LocalObjectReader, reader_for_manifest
    from recall.wizard.inventory import build_inventory

    monkeypatch.delenv("RECALL_LOCAL_ALLOWLIST", raising=False)

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("# T\n\nbody\n", encoding="utf-8")
    inventory = tmp_path / "inv.json"
    inventory.write_text(json.dumps(build_inventory(root, "**/*.md")), encoding="utf-8")
    manifest = IndexManifestV1("t", "v1", load_inventory(inventory))

    with pytest.raises(ValueError, match="RECALL_LOCAL_ALLOWLIST"):
        reader_for_manifest(manifest)

    reader = LocalObjectReader([root])
    assert reader.fetch(manifest.objects[0]).data
    reader.verify(manifest)  # raises if any object fails its digest


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
