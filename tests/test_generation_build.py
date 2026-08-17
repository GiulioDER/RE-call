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

import dataclasses
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.generation_build import (
    BuildRequest,
    build_generation,
    build_provenance,
    chunker_for,
    embedder_identity,
    pipeline_identity,
)
from recall.index import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, chunk_text
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


def test_a_chunk_size_that_cannot_describe_a_real_pipeline_is_refused() -> None:
    """The sizes go into the same immutable record as the chunker, from the same edited file.

    `max_chars=0` is the one with teeth, and it fails differently on each path, which is why
    neither caught it: `chunk_code` returns the whole text as ONE chunk while the record claims
    `max_chars=0`, and the text path raises only inside `manager.build`, after `manager.create` has
    already written the generation row.

    A non-int is refused before the range is compared, because the range test alone let a float
    through: `json.loads('{"max_chars": 800.0}')` passed `< 1`, recorded `{"max_chars": 800.0}`, and
    raised `TypeError` from a slice inside `manager.build` — after `manager.create` had written the
    row. `True` is refused for the same reason: it was recorded as `True` while running as 1.
    """
    for max_chars in (0, -5):
        with pytest.raises(ValueError, match="max_chars must be at least 1"):
            BuildRequest(max_chars=max_chars)

    with pytest.raises(ValueError, match="overlap cannot be negative"):
        BuildRequest(overlap=-1)

    for bad in (800.0, "800", True, None):
        with pytest.raises(ValueError, match="max_chars must be an int"):
            BuildRequest(max_chars=bad)  # type: ignore[arg-type]
    for bad in (80.0, "80", False):
        with pytest.raises(ValueError, match="overlap must be an int"):
            BuildRequest(overlap=bad)  # type: ignore[arg-type]

    assert BuildRequest(max_chars=1, overlap=0).max_chars == 1


def test_the_recorded_overlap_is_the_one_the_chunker_will_actually_use() -> None:
    """An overlap above the clamp must be recorded clamped, not as asked for.

    This is the finding I got wrong by measuring the wrong observable. I checked that an oversized
    overlap produces identical CHUNKS, concluded there was no harm, and recorded the requested value
    — but chunk equality is not the property this module protects. `PipelineIdentity.fingerprint`
    is defined as covering every input that can change chunks, and a calibration binds to it, so two
    configurations producing byte-identical chunks and different fingerprints make a calibration
    measured against one `CALIBRATION_STALE` against the other.

    Default-reachable, not exotic: the clamp is `max_chars // 4`, so `--max-chars 200` with the
    default overlap of 80 already runs at 50.
    """
    recorded = chunker_for(BuildRequest(max_chars=200, overlap=80))[1].configuration
    assert recorded["overlap"] == 50, "the default overlap at max_chars=200 runs at max_chars // 4"

    # An oversized request records the clamp, not the request.
    assert chunker_for(BuildRequest(max_chars=100, overlap=500))[1].configuration["overlap"] == 25

    # Below the clamp, the requested value is faithful and must NOT be altered.
    assert chunker_for(BuildRequest(max_chars=800, overlap=80))[1].configuration["overlap"] == 80

    # And the callable is bound to the same value the record names, so the pair still agree.
    #
    # Asserted STRUCTURALLY, on the partial's keywords, not by comparing outputs. Outputs cannot
    # see this: `chunk_text` applies the same clamp internally, so binding 80 and binding 50 produce
    # identical chunks and an output comparison holds either way. That left the reason for clamping
    # the binding at all — that the record's correctness must not depend on another module's
    # internal clamp continuing to exist — asserted by nothing. If `_split_oversized` ever loses its
    # clamp, now that `effective_overlap` is a named function and the recording caller applies it,
    # this is the line that notices.
    text = ("para. " * 60 + "\n\n") * 12
    callable_, identity = chunker_for(BuildRequest(max_chars=200, overlap=80))
    assert callable_.keywords["overlap"] == identity.configuration["overlap"]
    assert callable_(text) == chunk_text(
        text, max_chars=200, overlap=identity.configuration["overlap"]
    )


def test_two_requests_that_chunk_identically_record_one_fingerprint() -> None:
    """The consequence, stated as the property rather than as the mechanism.

    Without the clamp in the record these two fingerprinted differently while producing the same
    chunks, which is the whole harm: the wizard calibrates one generation and serves another.

    ⚠️ The premise is measured on `chunk_text` DIRECTLY, not through the two partials. Comparing the
    partials was the obvious way to write it and the fix made it a tautology: both are now bound to
    the clamped value, so they carry byte-identical keywords and the comparison cannot fail for any
    input — it passed on `text=""`, which is exactly the fixture inadequacy its own message claims to
    rule out. Before the fix the two partials carried 80 and 50 and the comparison meant something.
    A guard whose subject changes under it stops being a guard.
    """
    text = ("para. " * 60 + "\n\n") * 12
    assert chunk_text(text, max_chars=200, overlap=80) == chunk_text(
        text, max_chars=200, overlap=50
    ), "the fixture has no oversized block, so it cannot show the fingerprint problem"

    asked = chunker_for(BuildRequest(max_chars=200, overlap=80))
    clamped = chunker_for(BuildRequest(max_chars=200, overlap=50))
    assert asked[1] == clamped[1]


def test_the_embedder_is_examined_before_the_chunker() -> None:
    """The evaluation order `pipeline_for` chose deliberately, pinned rather than commented.

    Before the extraction, a request invalid in both its embedder identity and its chunker surfaced
    the embedder's `LineageError`. An intermediate shape flipped that to the chunker's `ValueError`,
    which reads as a no-op and is not: it changes which of two problems an operator is told about
    first. Only a comment recorded the decision, and that comment had already gone stale, quoting a
    message no longer present anywhere in the tree.
    """
    from recall.generation_build import pipeline_for
    from recall.lineage import LineageError

    # Invalid embedder identity (no revision, no digest, no reason) AND an invalid chunker.
    smuggled = object.__new__(BuildRequest)
    for field in dataclasses.fields(BuildRequest):
        object.__setattr__(smuggled, field.name, getattr(BuildRequest(), field.name))
    object.__setattr__(smuggled, "chunker", "Code")

    with pytest.raises(LineageError):
        pipeline_for(_FakeEmbedder(), smuggled)


def test_the_chunker_decision_point_also_refuses_a_smuggled_chunk_size() -> None:
    """The sizes need the same second gate the chunker got, and for the same reason.

    A request restored without `__init__` never runs `__post_init__`, so validating only in the
    constructor left the deserialisation path this module's docstring names as the threat model
    entirely uncovered for two of the three fields.
    """
    import dataclasses

    smuggled = object.__new__(BuildRequest)
    for field in dataclasses.fields(BuildRequest):
        object.__setattr__(smuggled, field.name, getattr(BuildRequest(), field.name))
    object.__setattr__(smuggled, "max_chars", 0)

    with pytest.raises(ValueError, match="max_chars must be at least 1"):
        chunker_for(smuggled)


def test_the_chunker_decision_point_refuses_a_request_that_bypassed_the_constructor() -> None:
    """The constructor guard does not cover the path the wizard's resumable state will use.

    A frozen dataclass rebuilt by `copy.deepcopy`, or by any deserialiser that restores `__dict__`
    directly, never runs `__post_init__`. So `chunker_for` has to refuse too: otherwise a request
    that arrived from a file selects the TEXT chunker and records `recall.chunk_text`, the callable
    and the record agreeing with each other and both disagreeing with what was asked.

    `object.__new__` here stands in for any such reconstruction. The point is not that anybody
    calls it, but that a value can reach `chunker_for` without passing the constructor.

    The message is the SAME one the constructor gives, and that is deliberate. The two gates
    previously validated in opposite orders, so a request invalid in both its chunker and its sizes
    reported a different reason depending on which gate caught it. They share one validator now.
    """
    import dataclasses

    smuggled = object.__new__(BuildRequest)
    for field in dataclasses.fields(BuildRequest):
        object.__setattr__(smuggled, field.name, getattr(BuildRequest(), field.name))
    object.__setattr__(smuggled, "chunker", "Code")

    with pytest.raises(ValueError, match="chunker must be one of") as from_gate:
        chunker_for(smuggled)

    with pytest.raises(ValueError, match="chunker must be one of") as from_constructor:
        BuildRequest(chunker="Code")  # type: ignore[arg-type]

    assert str(from_gate.value) == str(from_constructor.value)

    # And a request invalid in BOTH ways reports the same field first at either gate, rather than
    # depending on which one happened to catch it.
    object.__setattr__(smuggled, "max_chars", 0)
    with pytest.raises(ValueError, match="chunker must be one of"):
        chunker_for(smuggled)


def test_the_callable_and_the_record_come_from_one_chunker_for_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property `pipeline_for` exists for, pinned by COUNTING rather than by comparing.

    The previous attempt asserted that a handed-in `ChunkerIdentity` was the object recorded, using
    `is` on the theory that object identity would catch a silent recompute. It did not: reverting
    `build_generation` to call `pipeline_identity` without the identity, which restores exactly the
    double `chunker_for` call the change existed to remove, left all 43 tests green. `is` pins which
    object was used, not how many times the function ran, and that test never went through
    `build_generation` at all.

    Counting the calls kills both mutants. The agreement assertion is kept alongside, because a
    single call is only interesting if what comes out of it is what gets used.
    """
    import recall.generation_build as module

    # `commit_root=None` so this does not spawn a real `git rev-parse` in the pytest working
    # directory. The sibling test file stubs `head_commit` for exactly that reason; the chunker call
    # count is the subject here and the commit stamp is incidental to it.
    request = BuildRequest(chunker="code", max_chars=321, unverified=True, commit_root=None)
    calls: list[BuildRequest] = []
    real = module.chunker_for

    def counting(req: BuildRequest) -> object:
        calls.append(req)
        return real(req)

    monkeypatch.setattr(module, "chunker_for", counting)

    captured: dict[str, object] = {}

    class _Manager:
        def create(self, manifest: object, pipeline: object, *, allow_unverified: bool) -> object:
            captured["pipeline"] = pipeline
            return SimpleNamespace(generation_id="gen_count")

        def build(
            self,
            generation_id: str,
            reader: object,
            embedder: object,
            chunker: object,
            *,
            provenance: object = None,
        ) -> object:
            captured["chunker"] = chunker
            return SimpleNamespace(generation_id=generation_id)

    build_generation(_Manager(), object(), object(), _FakeEmbedder(), request)

    assert len(calls) == 1, f"chunker_for ran {len(calls)} times, not once"

    # And the record describes the callable that was handed over.
    pipeline = captured["pipeline"]
    source = "def f():\n    return 1\n\n\n" * 60
    assert pipeline.chunker == real(request)[1]  # type: ignore[attr-defined]
    assert captured["chunker"](source) == real(request)[0](source)  # type: ignore[operator]


def test_the_identity_only_helper_delegates_rather_than_rebuilding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pipeline_identity` must stay a delegation, and equality cannot show that.

    The first version of this asserted `pipeline_identity(...) == pipeline_for(...)[1]`, and a
    mutant that rebuilt the record inline survived it: an independently constructed identity
    compares equal, so the assertion held while the module went back to two construction sites,
    which is the drift the whole arrangement exists to prevent. Same mistake as the `is`-versus-
    count one a round earlier, one level along — the property is structural, so it has to be
    measured structurally.
    """
    import recall.generation_build as module

    request = BuildRequest(chunker="code", max_chars=321, unverified=True)
    calls: list[BuildRequest] = []
    real = module.pipeline_for

    def counting(embedder: object, req: BuildRequest) -> object:
        calls.append(req)
        return real(embedder, req)

    monkeypatch.setattr(module, "pipeline_for", counting)

    identity = pipeline_identity(_FakeEmbedder(), request)

    assert len(calls) == 1, "pipeline_identity rebuilt the record instead of delegating"
    assert identity == real(_FakeEmbedder(), request)[1]


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
