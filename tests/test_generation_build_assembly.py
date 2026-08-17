"""Pin what `recall generation build` writes into a generation's immutable lineage record.

Phase 3 of the installation wizard has to build generations too, and the hundred lines that turn
CLI flags into a `PipelineIdentity` live inline in `cli.py`'s `main()`. Reimplementing them in the
wizard is how the two silently diverge, so they are about to be extracted into one function both
callers use. Nothing pinned them before this file: `hashing-md5-bow-v1`, the provider defaults, the
`unverified_reason` conditional and the two chunker names are all written verbatim into a record
the system treats as immutable evidence, and every one of them could have been changed by the
extraction without a single test noticing.

These are characterization tests. They were written against the unmodified `cli.py` and passed
there first; the extraction is correct exactly insofar as they keep passing unchanged.

No database. `GenerationManager.__init__` only parses its arguments, so `create` and `build` can be
intercepted and the assembly observed without anything connecting.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.cli import main as cli_main
from recall.generations import BuildStats, GenerationManager
from recall.index import chunk_text
from recall.lineage import IndexManifestV1
from recall.manifest import load_inventory
from recall.wizard.inventory import write_inventory

#: Local host, so `require_secure_dsn` does not refuse the published `recall:recall` credentials,
#: and port 1, where nothing listens on any platform — if an interception ever fails to bite, the
#: test fails connecting rather than quietly writing to a real database.
DSN = "postgresql://recall:recall@127.0.0.1:1/recall"

TENANT = "assembly-tenant"


class _NotHashing:
    """Stands in for a resolved fastembed embedder, so the `else` arm is genuinely reached.

    Every real alternative to `hashing` downloads weights, so without this the non-hashing branch
    of the provider default can only be entered by passing `--embedder-provider` explicitly, which
    is the one input that stops the default from being observed at all.
    """

    name = "BAAI/bge-small-en-v1.5"
    dim = 384


def _run_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *flags: str,
    embedder: str = "hashing",
    resolved: object | None = None,
) -> dict[str, Any]:
    """Drive the real CLI path and capture what it hands `GenerationManager`.

    Every patch is scoped to this one call. `monkeypatch` is function-scoped, so a `setattr` left
    standing leaked a stub embedder into the *second* call of any test that runs two flag sets and
    compares them — which is most of them, and which silently turned a comparison into a
    tautology before it turned into a failure.
    """
    # `exist_ok` because several tests below call this twice to compare two flag sets, and a
    # second call must land on the same corpus: the point of comparison is the flags.
    root = tmp_path / "corpus"
    root.mkdir(exist_ok=True)
    (root / "a.md").write_text("# Title\n\nbody text\n", encoding="utf-8")

    inventory = tmp_path / "inv.json"
    write_inventory(root, inventory, "**/*.md")
    manifest = IndexManifestV1(TENANT, "2026-01-01", load_inventory(inventory))
    manifest_path = tmp_path / "man.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_create(
        self: GenerationManager,
        manifest: Any,
        pipeline: Any,
        *,
        allow_unverified: bool = False,
    ) -> Any:
        captured["manifest"] = manifest
        captured["pipeline"] = pipeline
        captured["allow_unverified"] = allow_unverified
        captured["environment"] = self.environment
        return SimpleNamespace(generation_id="gen_assembly")

    def fake_build(
        self: GenerationManager,
        generation_id: str,
        reader: Any,
        build_embedder: Any,
        chunker: Any,
        *,
        provenance: Any = None,
    ) -> BuildStats:
        captured["reader"] = reader
        captured["embedder"] = build_embedder
        captured["chunker"] = chunker
        captured["provenance"] = provenance
        return BuildStats("gen_assembly", 1, 1, 0, 0, 0, 0)

    with monkeypatch.context() as patch:
        patch.setenv("RECALL_LOCAL_ALLOWLIST", str(root))
        patch.setattr(GenerationManager, "create", fake_create)
        patch.setattr(GenerationManager, "build", fake_build)
        if resolved is not None:
            patch.setattr("recall.cli._make_embedder", lambda name: resolved)

        cli_main(
            [
                "--serving-dsn",
                DSN,
                "--tenant",
                TENANT,
                "--embedder",
                embedder,
                "generation",
                "build",
                str(manifest_path),
                *flags,
            ]
        )
    assert "pipeline" in captured, "the build never reached GenerationManager.create"
    return captured


def test_the_hashing_embedder_carries_its_own_provider_and_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hashing` is deterministic and shipped, so it names itself rather than borrowing fastembed's.

    Both strings are literals in the assembly and both land in an immutable record. A digest is
    never computed for it, so the revision is the entire identity.
    """
    identity = _run_build(tmp_path, monkeypatch)["pipeline"].embedder

    assert identity.provider == "recall"
    assert identity.revision == "hashing-md5-bow-v1"
    assert identity.model == "hashing-64"
    assert identity.dimension == 64
    assert identity.artifact_digest is None


def test_a_non_hashing_embedder_defaults_to_the_fastembed_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `else` arm. It must NOT inherit the hashing revision, which would be a false identity.

    `hashing` is the only embedder that needs no download, so the arm is reached with a resolved
    stub rather than by passing `--embedder-provider`: an explicit provider is precisely the input
    that prevents the default from being observed.
    """
    # `--unverified-development` is required, not incidental: with no revision, no digest and no
    # reason, `EmbedderIdentity` refuses to be constructed at all. That refusal is why the wizard
    # needs `wizard.identity` to resolve a digest, and it is the state this arm has to sit in to
    # observe the provider default.
    identity = _run_build(
        tmp_path, monkeypatch, "--unverified-development", resolved=_NotHashing()
    )["pipeline"].embedder

    assert identity.provider == "fastembed"
    assert identity.revision is None, "the hashing revision must not leak into another embedder"
    assert identity.model == "BAAI/bge-small-en-v1.5"
    assert identity.dimension == 384


def test_an_explicit_provider_and_revision_beat_the_hashing_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`provider or "recall"` and `revision or "hashing-md5-bow-v1"`, in that direction.

    Inverting either to `"recall" or provider` would ignore the operator's flag while still
    producing a well-formed record, which is the failure this pins.
    """
    identity = _run_build(
        tmp_path,
        monkeypatch,
        "--embedder-provider",
        "operator-provider",
        "--embedder-revision",
        "operator-revision",
    )["pipeline"].embedder

    assert identity.provider == "operator-provider"
    assert identity.revision == "operator-revision"


def test_an_artifact_digest_reaches_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of `wizard.identity`: a digest alone makes the identity verifiable."""
    digest = "f" * 64
    identity = _run_build(
        tmp_path, monkeypatch, "--embedder-artifact-digest", digest
    )["pipeline"].embedder

    assert identity.artifact_digest == digest


def test_unverified_development_stamps_a_reason_only_when_nothing_else_identifies_the_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unverified_reason` is conditioned on THREE things, and each clause is pinned separately.

    A record carrying both a real revision and "explicit development build" would be
    self-contradicting provenance, so the two `and not` clauses are the interesting part rather
    than the flag itself.
    """
    # The positive case, which needs an embedder that supplies no revision of its own.
    bare = _run_build(
        tmp_path, monkeypatch, "--unverified-development", resolved=_NotHashing()
    )["pipeline"].embedder
    assert bare.unverified_reason == "explicit development build"

    # `and not revision`: `hashing` supplies one, so the flag alone must not stamp a reason.
    with_revision = _run_build(
        tmp_path, monkeypatch, "--unverified-development"
    )["pipeline"].embedder
    assert with_revision.revision == "hashing-md5-bow-v1"
    assert with_revision.unverified_reason is None

    # `and not artifact_digest`: a digest identifies the embedder on its own.
    with_digest = _run_build(
        tmp_path,
        monkeypatch,
        "--unverified-development",
        "--embedder-artifact-digest",
        "a" * 64,
        resolved=_NotHashing(),
    )["pipeline"].embedder
    assert with_digest.unverified_reason is None

    # And without the flag at all, no reason. Necessarily on `hashing`: an embedder with no
    # revision, no digest and no reason is refused outright by `EmbedderIdentity`, so this is the
    # only shape in which "the flag is what stamps it" can be observed from the negative side.
    assert (
        _run_build(tmp_path, monkeypatch)["pipeline"].embedder.unverified_reason is None
    )


def test_unverified_development_is_passed_through_to_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`allow_unverified` is a separate argument from the identity, and gates a different guard."""
    assert _run_build(tmp_path, monkeypatch)["allow_unverified"] is False
    assert (
        _run_build(tmp_path, monkeypatch, "--unverified-development")["allow_unverified"]
        is True
    )


def test_the_text_chunker_records_its_name_version_and_both_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _run_build(
        tmp_path, monkeypatch, "--chunker", "text", "--max-chars", "200", "--overlap", "20"
    )
    chunker = captured["pipeline"].chunker

    assert chunker.algorithm == "recall.chunk_text"
    assert chunker.schema_version == 1
    assert dict(chunker.configuration) == {"max_chars": 200, "overlap": 20}


def test_the_code_chunker_records_no_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`chunk_code` takes no overlap, so recording one would describe a pipeline that never ran."""
    chunker = _run_build(
        tmp_path, monkeypatch, "--chunker", "code", "--max-chars", "300", "--overlap", "20"
    )["pipeline"].chunker

    assert chunker.algorithm == "recall.chunk_code"
    assert chunker.schema_version == 1
    assert dict(chunker.configuration) == {"max_chars": 300}


def test_the_chunker_callable_is_bound_to_the_same_parameters_it_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lineage record and the function that produced the chunks must not disagree.

    A partial built with the defaults while the record carries the operator's flags is a
    well-formed generation whose provenance is a lie, and no assertion about the record alone can
    see it. Checked against an independently-parameterised `chunk_text`, and against the default,
    so neither "the flags were ignored" nor "the wrong values were bound" can pass.
    """
    text = "para one. " * 400
    chunker = _run_build(
        tmp_path, monkeypatch, "--chunker", "text", "--max-chars", "200", "--overlap", "20"
    )["chunker"]

    assert chunker(text) == chunk_text(text, max_chars=200, overlap=20)
    assert chunker(text) != chunk_text(text)


def test_the_code_chunker_callable_is_bound_to_the_parameter_it_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same check on the other arm, which survived mutation without it.

    Dropping `max_chars` from the code branch's `functools.partial` left every test green: the
    identity still recorded the operator's value and only the callable fell back to the default.
    That is the precise failure the sibling test above exists to catch, and enumerating the
    mutations from the source rather than from the tests is what surfaced that only one of the two
    branches was covered.
    """
    from recall.index import chunk_code

    source = "def f():\n    return 1\n\n\n" * 200
    chunker = _run_build(
        tmp_path, monkeypatch, "--chunker", "code", "--max-chars", "150"
    )["chunker"]

    assert chunker(source) == chunk_code(source, max_chars=150)
    assert chunker(source) != chunk_code(source)


def test_provenance_carries_the_project_and_the_commit_unless_they_are_withheld(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same stamps `recall index` writes. A calibrated generation is the only path the
    wizard can use, so a generation without them cannot say where a hit came from.

    `head_commit` is stubbed rather than left to read the real repository. The CLI stamps
    `head_commit(".")`, which is the pytest process's own working directory, so asserting merely
    that a commit was recorded made this test depend on the checkout having a `.git` — it fails in
    an sdist extract, and in any container where git refuses with "dubious ownership". Stubbing
    also upgrades the assertion from "something was recorded" to "the value flowed through".

    The stub RECORDS its argument rather than discarding it. A `lambda root: "c0ffee1"` made the
    assertion pass for `commit_root="."`, `""`, `/nowhere` and `str(tmp_path)` alike, so it
    silently removed the only coverage of what the CLI passes: mutating `cli.py` to send
    `"/definitely/not/a/repo"` left the whole file green. Stubbing away an environment dependency
    is right; stubbing away the argument is how the fix for one gap opened another.
    """
    roots: list[str | None] = []

    def stub(root: str | None) -> str:
        roots.append(root)
        return "c0ffee1"

    monkeypatch.setattr("recall.generation_build.head_commit", stub)

    default = _run_build(tmp_path, monkeypatch)["provenance"]
    assert "project" not in default, "an unset --project must not be stamped as None"
    assert default["indexed_commit"] == "c0ffee1"

    named = _run_build(tmp_path, monkeypatch, "--project", "my-project")["provenance"]
    assert named["project"] == "my-project"

    unstamped = _run_build(tmp_path, monkeypatch, "--no-commit-stamp")["provenance"]
    assert "indexed_commit" not in unstamped

    # Read at the END, so all three arms are covered rather than only the first, and so inserting
    # a fourth arm above cannot leave the assertion silently describing a subset. Two entries, not
    # three: the `--no-commit-stamp` arm must not call `head_commit` at all.
    assert roots == [".", "."], "the CLI reads the commit from its own working directory"


def test_the_cli_defaults_and_the_request_defaults_describe_the_same_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent sources of the same defaults, cross-checked against each other.

    `argparse` carries the literals 800 and 80; `BuildRequest` carries `recall.index`'s constants.
    The CLI always passes explicit values, so a drift between the two would never surface there —
    it would surface as the wizard, which builds from a bare `BuildRequest()`, silently producing
    a different pipeline fingerprint than the documented default, and therefore a generation no
    existing calibration binds to.
    """
    from recall.generation_build import BuildRequest, chunker_for

    from_cli = _run_build(tmp_path, monkeypatch)["pipeline"].chunker
    from_request = chunker_for(BuildRequest())[1]

    assert from_cli == from_request


def test_a_local_manifest_builds_under_development_and_reads_through_the_local_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two constraints the wizard inherits: a `file://` manifest is refused in production, and it
    is read by the local reader rather than the S3 one."""
    from recall.manifest import LocalObjectReader

    captured = _run_build(tmp_path, monkeypatch)

    assert captured["environment"] == "development"
    assert isinstance(captured["reader"], LocalObjectReader)
