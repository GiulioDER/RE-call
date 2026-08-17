"""The three tenant definitions, and the couplings that make them safe.

The design's corpus table is four independent-looking columns that are not independent at all:
`RECALL_ENV=production` is the only switch that routes search through `GenerationStore`, and the
same switch refuses local filesystem indexing. So "calibrated" and "writable" are mutually
exclusive, and a spec that claims both describes an install that cannot work. Recording that as a
table in prose leaves the wizard to discover it as a runtime refusal three steps later; recording it
as a constructor invariant makes it unbuildable.

The embedder is deliberately NOT a per-corpus field. All three tenants share one database and one
`chunks`-family schema, and the dimension is welded to the table, so two embedders across corpora is
not a configuration the wizard should be able to express. `CorpusPlan` takes one embedder for the
whole plan, which is the "prefer impossible over detectable" form of that constraint.
"""

from __future__ import annotations

import dataclasses
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
        docs_root=tmp_path / "docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
    )

    by_tenant = {c.tenant: c for c in plan.corpora}
    assert set(by_tenant) == {"docs", "code", "memory"}

    docs = by_tenant["docs"]
    assert docs.glob == "**/*.md"
    assert docs.chunker == "text"
    assert docs.calibrated is True
    assert docs.serving_environment == "production"
    assert docs.trust_mode is TrustMode.STRICT
    assert docs.writable is False

    code = by_tenant["code"]
    assert code.glob == "**/*.py"
    assert code.chunker == "code"
    assert code.calibrated is True
    assert code.serving_environment == "production"
    assert code.trust_mode is TrustMode.STRICT
    assert code.writable is False

    memory = by_tenant["memory"]
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
        "tenant": "docs",
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
        docs_root=tmp_path / "docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
    )

    assert plan.embedder == "fastembed"
    # On the FIELD SET, not `hasattr`. `hasattr` also matches a property or a method, so it would
    # keep passing if `embedder` arrived as a computed attribute, which is the same drift wearing a
    # different hat.
    assert "embedder" not in {f.name for f in dataclasses.fields(CorpusSpec)}


def test_two_corpora_cannot_share_a_tenant(tmp_path: Path) -> None:
    """A manifest is bound to one tenant, and re-indexing a tenant PRUNES what is not in it.

    So two corpora on one tenant do not merge, they delete each other. Refused where it is cheap to
    refuse rather than discovered as a corpus that silently lost half its sources.
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
        docs_root=tmp_path / "docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
    )

    assert [c.tenant for c in plan.calibrated] == ["docs", "code"]
    assert all(c.calibrated for c in plan.calibrated)
    assert "memory" not in [c.tenant for c in plan.calibrated]


def test_a_relative_root_is_refused_because_it_would_stamp_the_wizards_own_commit() -> None:
    """The provenance bug `commit_root` exists to prevent, reintroduced one layer up.

    `head_commit` shells out to `git -C <path>`, which resolves a relative path against the calling
    process's working directory and then walks upward looking for a repository. Measured before this
    guard: `docs_corpus(Path("docs")).build_request()` stamped THIS repository's HEAD onto what is
    nominally the user's corpus. Absent provenance is safe; present-and-wrong is not.
    """
    with pytest.raises(ValueError, match="root must be absolute"):
        docs_corpus(Path("docs"))

    with pytest.raises(ValueError, match="root must be absolute"):
        memory_corpus(Path("memory"))


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
    refuse — after construction, past the check. Re-indexing a tenant prunes what is absent from the
    new manifest, so a duplicate does not merge, it deletes.
    """
    plan = CorpusPlan(embedder="fastembed", corpora=[docs_corpus(tmp_path / "docs")])

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
