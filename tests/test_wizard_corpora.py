"""The three tenant definitions, and the couplings that make them safe.

The design's corpus table is four independent-looking columns that are not independent at all:
`RECALL_ENV=production` is the only switch that routes search through `GenerationStore`, and
`recall index` refuses to run under that same switch. So "calibrated" and "writable" are mutually
exclusive, and a spec that claims both describes an install that cannot work. Recording that as a
table in prose leaves the wizard to discover it as a runtime refusal three steps later; recording it
as a constructor invariant makes it unbuildable.

🔁 That second clause used to read "the same switch refuses local filesystem indexing", which
`corpora.py` had already retracted by name while this file kept the disproven wording — one
changeset disagreeing with itself. The refusal lives in `recall index`'s own dispatch and in
`setup.py`'s auto-index; `Indexer` reads no such variable, which is why the wizard must impose the
rule on itself rather than inherit it.

The embedder is deliberately NOT a per-corpus field. All three tenants share one database and one
`chunks`-family schema, and the dimension is welded to the table, so two embedders across corpora is
not a configuration the wizard should be able to express. `CorpusPlan` takes one embedder for the
whole plan, which is the "prefer impossible over detectable" form of that constraint.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from recall.trust_policy import TrustMode
from recall.wizard.corpora import (
    CorpusPlan,
    CorpusSpec,
    code_corpus,
    default_plan,
    docs_corpus,
    memory_corpus,
)


def test_the_three_corpora_match_the_design_table(tmp_path: Path) -> None:
    """The documented layout, asserted field by field rather than trusted to the factories."""
    plan = default_plan(
        embedder="fastembed",
        docs_root=tmp_path / "default-docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "default-memory",
    )

    by_tenant = {c.tenant: c for c in plan.corpora}
    assert set(by_tenant) == {"default-docs", "default-code", "default-memory"}

    docs = by_tenant["default-docs"]
    assert docs.glob == "**/*.md"
    assert docs.chunker == "text"
    assert docs.calibrated is True
    assert docs.serving_environment == "production"
    assert docs.trust_mode is TrustMode.STRICT
    assert docs.writable is False

    code = by_tenant["default-code"]
    assert code.glob == "**/*.py"
    assert code.chunker == "code"
    assert code.calibrated is True
    assert code.serving_environment == "production"
    assert code.trust_mode is TrustMode.STRICT
    assert code.writable is False

    memory = by_tenant["default-memory"]
    assert memory.glob == "**/*.md"
    assert memory.chunker == "text"
    assert memory.calibrated is False
    assert memory.serving_environment == "development"
    assert memory.trust_mode is TrustMode.DEVELOPMENT
    assert memory.writable is True


def test_memory_stays_uncalibrated_on_purpose(tmp_path: Path) -> None:
    """Not an oversight, and the reason belongs beside the definition.

    A fresh memory directory holds one file and cannot meet the 20-per-class certification floor, so
    a calibrated memory tenant would refuse every search with `CALIBRATION_MISSING` forever. It is
    also the one corpus that must accept writes, which production mode refuses outright.
    """
    memory = memory_corpus(tmp_path / "memory")

    assert memory.calibrated is False
    assert memory.writable is True
    assert memory.trust_mode is TrustMode.DEVELOPMENT


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"calibrated": True, "writable": True}, "cannot be both calibrated and writable"),
        (
            {"calibrated": True, "serving_environment": "development"},
            "a calibrated corpus is served under production",
        ),
        # `calibrated=False` is not incidental: the base spec is calibrated, and the
        # calibrated-and-writable guard sits ABOVE this one, so without it the earlier guard
        # absorbs the input and this branch never runs while the test still passes.
        (
            {"calibrated": False, "writable": True, "serving_environment": "production"},
            "refuses to run under production",
        ),
        # `serving_environment` moved to development on purpose: the uncalibrated-in-production
        # guard sits above the strict guard, so leaving the base's production value here would have
        # this case testing that one instead. `calibrated=False` is what does the work.
        (
            {
                "calibrated": False,
                "serving_environment": "development",
                "trust_mode": TrustMode.STRICT,
            },
            "an uncalibrated corpus cannot be served strictly",
        ),
        # The seventh combination, and it fails worse than the other six: no generation under
        # production means `GenerationStore.snapshot` raises from outside `trusted_search`'s try
        # block, so the caller gets a bare exception rather than a refusal.
        (
            {"calibrated": False, "trust_mode": TrustMode.DEVELOPMENT},
            "cannot be served under production",
        ),
        ({"tenant": "   "}, "tenant must be non-empty"),
        ({"serving_environment": "staging"}, "serving_environment must be one of"),
        ({"chunker": "Code"}, "chunker must be one of"),
        # The converse of the uncalibrated-and-strict guard, which was missing entirely: every
        # other guard refuses a corpus claiming MORE than it can deliver, this one refuses a
        # calibrated corpus claiming LESS. Nothing enforced the docstring's own statement that a
        # calibrated corpus is served strictly.
        (
            {"trust_mode": TrustMode.DEVELOPMENT},
            "a calibrated corpus is served strictly",
        ),
        ({"chunker": "prose", "calibrated": False, "serving_environment": "development",
          "trust_mode": TrustMode.DEVELOPMENT}, "chunker must be one of"),
    ],
)
def test_a_contradictory_spec_is_unbuildable(
    tmp_path: Path, overrides: dict[str, object], expected: str
) -> None:
    """Each coupling refused at construction, with a message naming the reason.

    These are not stylistic checks. `RECALL_ENV=production` both routes search through
    `GenerationStore` and refuses filesystem indexing, so every one of these combinations describes
    an install that fails at a later step with a message about something else.
    """
    fields: dict[str, object] = {
        "tenant": "default-docs",
        "root": tmp_path,
        "glob": "**/*.md",
        "chunker": "text",
        "calibrated": True,
        "serving_environment": "production",
        "trust_mode": TrustMode.STRICT,
        "writable": False,
    }
    fields.update(overrides)

    with pytest.raises(ValueError, match=expected):
        CorpusSpec(**fields)  # type: ignore[arg-type]


def test_the_valid_combinations_do_build(tmp_path: Path) -> None:
    """The allow path, so the invariants above cannot be satisfied by refusing everything.

    A guard that blocks legitimate configuration gets deleted, which loses the coverage entirely.
    """
    assert docs_corpus(tmp_path).calibrated is True
    assert code_corpus(tmp_path).chunker == "code"
    assert memory_corpus(tmp_path).writable is True


def test_one_embedder_for_the_whole_plan(tmp_path: Path) -> None:
    """The dimension is welded to the table, so two embedders is not expressible by design.

    Asserted as a property of the API rather than of a value: `CorpusPlan` takes a single embedder
    and `CorpusSpec` has no embedder field at all, so a per-corpus embedder cannot be written down.
    A test that merely checked "the three specs agree" would pass on an API that let them disagree.
    """
    plan = default_plan(
        embedder="fastembed",
        docs_root=tmp_path / "default-docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "default-memory",
    )

    assert plan.embedder == "fastembed"
    # On the FIELD SET, not `hasattr`. `hasattr` also matches a property or a method, so it would
    # keep passing if `embedder` arrived as a computed attribute, which is the same drift wearing a
    # different hat.
    assert "embedder" not in {f.name for f in dataclasses.fields(CorpusSpec)}


def test_two_corpora_cannot_share_a_tenant(tmp_path: Path) -> None:
    """A tenant has ONE active generation, so two corpora on it cannot both be served.

    🔁 This docstring used to say re-indexing prunes what is absent from the new manifest, so the
    second corpus deletes the first. That was false twice over: `_prune_vanished` is scoped to the
    indexed root (its own docstring says the scoping exists so one root's re-index cannot delete
    another's rows), and it prunes only what is gone from DISK, never what is missing from a
    manifest. The real hazard is supersession: promoting the second generation leaves the first
    indexed but unsearchable, and `rollback()` exists precisely because those rows are still there.
    Quieter than deletion, not smaller.
    """
    with pytest.raises(ValueError, match="tenant .* appears twice"):
        CorpusPlan(
            embedder="fastembed",
            corpora=(docs_corpus(tmp_path / "a"), docs_corpus(tmp_path / "b")),
        )


def test_an_empty_plan_is_refused() -> None:
    """Nothing to build is a configuration error, not a no-op install that reports success."""
    with pytest.raises(ValueError, match="at least one corpus"):
        CorpusPlan(embedder="fastembed", corpora=())


def test_an_empty_embedder_is_refused(tmp_path: Path) -> None:
    """The embedder decides the schema dimension, so a blank one is not a default to fall back on.

    It reaches `schema apply --dim` and `resolve_embedder`, and an empty string there produces a
    failure that names neither the wizard nor the field the user left blank.
    """
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="embedder must be non-empty"):
            CorpusPlan(embedder=blank, corpora=(docs_corpus(tmp_path),))


def test_the_calibrated_view_excludes_the_writable_tenant(tmp_path: Path) -> None:
    """The pipeline iterates this, so including `memory` would drive it down a path it must not take.

    Asserted by naming the tenants rather than by counting them: a count is satisfied by the wrong
    two, and "two of three" is exactly the shape that reads as correct while selecting the wrong
    subset. Plan order is asserted too, because the pipeline reports progress in it.
    """
    plan = default_plan(
        embedder="fastembed",
        docs_root=tmp_path / "default-docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "default-memory",
    )

    assert [c.tenant for c in plan.calibrated] == ["default-docs", "default-code"]
    assert all(c.calibrated for c in plan.calibrated)
    assert "default-memory" not in [c.tenant for c in plan.calibrated]


def test_a_trust_mode_spelled_as_a_plain_string_cannot_bypass_the_strict_guard(
    tmp_path: Path,
) -> None:
    """`TrustMode` is a `StrEnum`, so `"strict" == TrustMode.STRICT` while `is` is False.

    The uncalibrated-and-strict guard compared with `is`, and `trust_mode` was the one field with no
    validation at all, so passing the plain string sailed past the guard and produced exactly the
    tenant that answers INDEX_NOT_READY forever. Measured before the fix: that spec constructed with
    `type(trust_mode) is str`.

    Both halves are pinned here, because either alone leaves the guard spellable-past: the value is
    COERCED to the enum, and the comparison is `==`.
    """
    with pytest.raises(ValueError, match="cannot be served strictly"):
        CorpusSpec(
            tenant="default-memory",
            root=tmp_path,
            glob="**/*.md",
            chunker="text",
            calibrated=False,
            serving_environment="development",
            trust_mode="strict",  # type: ignore[arg-type]
            writable=True,
        )

    # A legitimate string spelling is coerced rather than rejected, so config read from a file works.
    coerced = memory_corpus(tmp_path)
    assert coerced.trust_mode is TrustMode.DEVELOPMENT
    assert isinstance(
        CorpusSpec(
            tenant="default-memory",
            root=tmp_path,
            glob="**/*.md",
            chunker="text",
            calibrated=False,
            serving_environment="development",
            trust_mode="development",  # type: ignore[arg-type]
            writable=True,
        ).trust_mode,
        TrustMode,
    )

    with pytest.raises(ValueError, match="trust_mode must be one of"):
        CorpusSpec(
            tenant="default-memory",
            root=tmp_path,
            glob="**/*.md",
            chunker="text",
            calibrated=False,
            serving_environment="development",
            trust_mode="paranoid",  # type: ignore[arg-type]
            writable=True,
        )


def test_a_wrongly_typed_corpora_argument_names_the_mistake(tmp_path: Path) -> None:
    """A `str` is iterable, so `corpora="default-docs"` survived the tuple coercion.

    It then failed inside the duplicate-tenant loop with `AttributeError: 'str' object has no
    attribute 'tenant'`, which names neither the argument nor the mistake, in a module where every
    other guard says exactly what is wrong.
    """
    with pytest.raises(TypeError, match="corpora must be an iterable of CorpusSpec"):
        CorpusPlan(embedder="fastembed", corpora="default-docs")  # type: ignore[arg-type]


def test_the_absolute_root_guard_has_the_same_meaning_on_both_platforms() -> None:
    """Pins what `is_absolute()` accepts per platform, which no CI job can currently observe.

    `Path.is_absolute()` is platform-polymorphic: `PureWindowsPath("/docs")` is NOT absolute (no
    drive) while `PurePosixPath("/docs")` is. Every CI job that installs the package and imports
    `recall.wizard` runs ubuntu-latest, and the one windows-latest job deliberately skips
    `pip install -e .`, so the guard protecting provenance on the SHIPPED target is exercised only
    under POSIX rules. The existing refusal test uses `Path("default-docs")`, which is relative under both
    rule sets and therefore cannot tell them apart.

    This asserts the flavour semantics directly, so the meaning is pinned without a Windows runner.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    # The form that differs, and the reason a Git Bash user on Windows can be surprised.
    assert PurePosixPath("/docs").is_absolute() is True
    assert PureWindowsPath("/docs").is_absolute() is False

    # Native absolute forms, both flavours, including UNC which `recall.manifest` supports.
    assert PureWindowsPath(r"C:\corpus").is_absolute() is True
    assert PureWindowsPath("//server/share/corpus").is_absolute() is True
    assert PurePosixPath("/srv/corpus").is_absolute() is True

    # Drive-relative is NOT absolute, which is the trap `C:docs` sets.
    assert PureWindowsPath("C:docs").is_absolute() is False

    # And whatever the platform, a plainly relative root is refused by the real guard.
    with pytest.raises(ValueError, match="root must be absolute"):
        docs_corpus(Path("default-docs"))


def test_a_relative_root_is_refused_because_it_would_stamp_the_wizards_own_commit() -> None:
    """The provenance bug `commit_root` exists to prevent, reintroduced one layer up.

    `head_commit` shells out to `git -C <path>`, which resolves a relative path against the calling
    process's working directory and then walks upward looking for a repository. Measured before this
    guard: `docs_corpus(Path("default-docs")).build_request()` stamped THIS repository's HEAD onto what is
    nominally the user's corpus. Absent provenance is safe; present-and-wrong is not.
    """
    with pytest.raises(ValueError, match="root must be absolute"):
        docs_corpus(Path("default-docs"))

    with pytest.raises(ValueError, match="root must be absolute"):
        memory_corpus(Path("default-memory"))


def test_a_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    """Absent is allowed, because the wizard creates the memory directory. A FILE is not."""
    a_file = tmp_path / "not-a-dir.md"
    a_file.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="exists and is not a directory"):
        docs_corpus(a_file)

    # And an absent absolute root still builds, since the wizard may create it.
    assert memory_corpus(tmp_path / "does-not-exist-yet").root.name == "does-not-exist-yet"


def test_a_list_of_corpora_becomes_a_tuple_so_the_plan_cannot_be_widened(tmp_path: Path) -> None:
    """`frozen=True` blocks rebinding, not mutation, and the annotation is not a runtime constraint.

    A list was accepted, and appending to it reinstated the duplicate tenant `CorpusPlan` exists to
    refuse — after construction, past the check. A tenant has one active generation, so the
    duplicate does not merge: promoting the second leaves the first indexed but unsearchable.
    """
    plan = CorpusPlan(embedder="fastembed", corpora=[docs_corpus(tmp_path / "default-docs")])

    assert isinstance(plan.corpora, tuple)
    with pytest.raises(AttributeError):
        plan.corpora.append(docs_corpus(tmp_path / "other"))  # type: ignore[attr-defined]
    assert hash(plan), "a frozen plan must stay hashable, which a list field silently prevents"


def test_the_build_request_carries_the_corpus_chunker_and_root(tmp_path: Path) -> None:
    """The seam into `generation_build`, and the reason `commit_root` exists.

    The CLI stamps `head_commit(".")`, its own working directory. The wizard indexes somebody
    else's repository, so it must stamp THAT root or record provenance that is present,
    well-formed and wrong.
    """
    code = code_corpus(tmp_path / "repo")
    request = code.build_request(project="my-project")

    assert request.chunker == "code"
    assert request.project == "my-project"
    assert request.commit_root == str(tmp_path / "repo")


def test_an_uncalibrated_corpus_has_no_build_request(tmp_path: Path) -> None:
    """`memory` is indexed through the legacy `chunks` table and never becomes a generation.

    Returning a `BuildRequest` for it would invite a caller to build one, which is the mistake that
    produces a promoted generation nothing can calibrate.
    """
    with pytest.raises(ValueError, match="memory.*is not calibrated"):
        memory_corpus(tmp_path / "memory").build_request()


# ----------------------------------------------------------------------------------------------
# Project scoping: the tenant name the desktop UI already uses
# ----------------------------------------------------------------------------------------------


def test_the_default_project_produces_the_tenants_the_ui_already_expects() -> None:
    """`RuntimeProfile.default_tenant` is "default", so this install is one the UI can see.

    A wizard that named its tenants anything else would build a parallel set the desktop app could
    not list, which is the same silent split as two databases one layer up.
    """
    from recall.wizard.corpora import DEFAULT_PROJECT, tenant_for

    assert DEFAULT_PROJECT == "default"
    assert tenant_for(DEFAULT_PROJECT, "docs") == "default-docs"
    assert tenant_for(DEFAULT_PROJECT, "code") == "default-code"
    assert tenant_for(DEFAULT_PROJECT, "memory") == "default-memory"


def test_a_project_may_contain_hyphens() -> None:
    """`rsplit("-", 1)` recovers the scope, which is how the desktop reads a tenant back."""
    from recall.wizard.corpora import tenant_for

    tenant = tenant_for("my-side-project", "code")
    assert tenant == "my-side-project-code"
    assert tenant.rsplit("-", 1) == ["my-side-project", "code"]


@pytest.mark.parametrize("kind", ["docs", "code", "memory"])
def test_a_project_ending_in_a_kind_is_refused(kind: str) -> None:
    """`my-docs` plus `-docs` is indistinguishable from the docs corpus of a project called `my`.

    Two different corpora resolving to one tenant name is the kind of collision that silently
    merges somebody's notes into somebody else's project.
    """
    from recall.wizard.corpora import tenant_for

    with pytest.raises(ValueError, match=f"must not end in -{kind}"):
        tenant_for(f"my-{kind}", "docs")


@pytest.mark.parametrize("bad", ["my project", "proj/ect", "pro:ject", "pro$ject"])
def test_a_project_that_is_not_a_valid_service_name_is_refused(bad: str) -> None:
    """The tenant becomes a Docker Compose service name (`recall-{tenant}`).

    Refused here rather than at `docker compose up`, where the error names a generated file the
    user never wrote and cannot correct.
    """
    from recall.wizard.corpora import tenant_for

    with pytest.raises(ValueError, match="only letters, digits"):
        tenant_for(bad, "docs")


def test_an_empty_project_is_refused() -> None:
    from recall.wizard.corpora import tenant_for

    with pytest.raises(ValueError, match="project must be non-empty"):
        tenant_for("   ", "docs")


def test_the_plan_scopes_every_corpus_to_the_same_project(tmp_path: Path) -> None:
    """One project, three tenants, and nothing left on the old bare naming."""
    from recall.wizard.corpora import default_plan

    plan = default_plan(
        embedder="hashing",
        docs_root=tmp_path / "d",
        code_root=tmp_path / "c",
        memory_root=tmp_path / "m",
        project="myapp",
    )

    assert [spec.tenant for spec in plan.corpora] == ["myapp-docs", "myapp-code", "myapp-memory"]


def test_a_data_folder_that_is_blank_or_relative_is_refused() -> None:
    """⛔ `Path("")` normalises to `Path(".")`, so a blank answer produced a RELATIVE data_root.

    `load_config` then refused it as non-absolute: an installer handing the user a validation error
    about its own output. The GUI reaches this by clearing the field, and would have written
    `wizard.json` into whatever directory the process happened to start in. The three corpus roots
    below it were already guarded; this one was not, and the fix shipped with no test.
    """
    from recall.wizard.questions import build_config

    base = {
        "project": "acme",
        "database": "new",
        "embedder": "fastembed",
        "corpus_version": "2026-01-01",
        "docs_root": "C:/docs" if os.name == "nt" else "/docs",
        "code_root": "C:/code" if os.name == "nt" else "/code",
        "memory_root": "C:/memory" if os.name == "nt" else "/memory",
    }

    for blank in ("", "   "):
        with pytest.raises(ValueError, match="data folder is required"):
            build_config({**base, "data_root": blank})

    for relative in ("recall-data", "./recall-data", "../recall-data"):
        with pytest.raises(ValueError, match="absolute path"):
            build_config({**base, "data_root": relative})

    # Control: an absolute path, and a `~` that expands to one, are both accepted — so this is not
    # a test that passes because `build_config` refuses everything.
    absolute = "C:/recall" if os.name == "nt" else "/recall"
    assert build_config({**base, "data_root": absolute})["data_root"]
    assert Path(build_config({**base, "data_root": "~/recall"})["data_root"]).is_absolute()


def test_the_terminal_interview_refuses_rather_than_traces_back(tmp_path: Path) -> None:
    """⛔ `recall/cli.py` handles `InteractiveRefusal`, not `ValueError`.

    The data-folder refusal above is reachable by TYPING a relative path, unlike the corpus-root
    ones whose prompts offer a non-empty default. So the terminal interview ended in a traceback
    with every answer already given thrown away, on the ordinary mistake rather than an exceptional
    one. The GUI was never affected: `install_ui._start_install` catches `ValueError` itself.
    """
    from recall.wizard.interactive import InteractiveRefusal, Prompter, ask_config
    from recall.wizard.questions import question_plan

    # Answer every question with its own default, except the data folder, which gets the relative
    # path a person can plausibly type. Driving the real plan rather than a fixed list means this
    # does not go stale the next time a question is added.
    relative = "recall-data"

    plan = question_plan(default_root=tmp_path)
    data_question = next(question for question in plan if question.key == "data_root")

    def read(prompt: str) -> str:
        # Keyed on the real question's own prompt text, so this cannot drift into answering nothing
        # and passing for the wrong reason. Everything else takes its default.
        return relative if data_question.prompt in prompt else ""

    with pytest.raises(InteractiveRefusal, match="absolute path"):
        ask_config(
            Prompter(read=read, write=lambda _text: None),
            default_root=tmp_path,
            probe=lambda *_args, **_kwargs: None,
        )


def test_the_desktop_corpus_prefix_is_reserved_against_a_wizard_install() -> None:
    """⛔ A prefix that decides what gets ABANDONED must not be something a user can wander into.

    `_release_superseded` selects generations to abandon by `corpus_version` prefix, so that a
    desktop upload never reclaims a corpus a wizard install built and deliberately left READY.
    `corpus_version` reaches the pipeline straight from user config with no other validation, so a
    copy-pasted value carrying the prefix would opt that corpus into the reclaim: abandon, then gc,
    then the chunk rows cascade away. The two constants are pinned equal here because they live in
    different packages on purpose and would otherwise drift silently.
    """
    from recall.wizard.pipeline import _RESERVED_CORPUS_PREFIX
    from recall_mcp.service import _DESKTOP_CORPUS_PREFIX

    assert _RESERVED_CORPUS_PREFIX == _DESKTOP_CORPUS_PREFIX, (
        "the reservation and the reclaim filter must be the same string; if they drift, the "
        "reservation stops protecting anything and nothing says so"
    )
