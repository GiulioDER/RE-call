"""Pure dependency projection regression coverage."""

from __future__ import annotations

import pytest

from recall.dependency_invalidation import build_dependency_projection
from recall.types import Chunk


def _chunk(
    source: str,
    *,
    authority: str = "tool_observation",
    depends_on: list[str] | None = None,
) -> Chunk:
    graph: dict[str, object] = {"authority": authority}
    if depends_on is not None:
        graph["depends_on"] = depends_on
    return Chunk(source, source, source, {"file": source, "recall_graph": graph})


def test_direct_and_transitive_invalidations_are_retained() -> None:
    projection = build_dependency_projection(
        [_chunk("dependent.md", depends_on=["prerequisite.md"]), _chunk("prerequisite.md")],
        tenant_id="tenant",
        generation_id="generation",
        base_states={"dependent.md": "current", "prerequisite.md": "expired"},
    )

    assert projection.reason_for("prerequisite.md") is not None
    reason = projection.reason_for("dependent.md")
    assert reason is not None
    assert reason.dependency == "prerequisite.md"
    assert reason.cause == "expired"
    assert reason.path == ("dependent.md", "prerequisite.md")


def test_model_inference_is_not_invalidated_by_temporal_state() -> None:
    projection = build_dependency_projection(
        [_chunk("inference.md", authority="model_inference")],
        tenant_id="tenant",
        generation_id="generation",
        base_states={"inference.md": "expired"},
    )

    # The authority assertion is not decoration. `reason_for` returns None both when a source was
    # judged and not invalidated AND when it never entered the projection at all, so on its own it
    # would pass just as happily against a build that silently dropped the chunk.
    assert projection.authorities["inference.md"] == "model_inference"
    assert projection.reason_for("inference.md") is None
    assert dict(projection.invalidations) == {}


def _placed(cid: str, file: str, source: str, depends_on: list[str] | None = None) -> Chunk:
    graph: dict[str, object] = {"authority": "tool_observation"}
    if depends_on is not None:
        graph["depends_on"] = depends_on
    return Chunk(cid, source, file, {"file": file, "recall_graph": graph})


def test_a_merged_node_is_classified_by_every_copy_not_the_first() -> None:
    """Node tests/test_dependency_invalidation.py::test_a_merged_node_is_classified_by_every_copy_not_the_first.

    Invariant: a projection built without `collided` (as the reasoning graph builds one) keeps
    two roots' copies of `README.md` in one node, and that node is classified by all its copies:
    it is in the dependent's root because one copy is, and is no upload because one copy is not,
    whichever chunk sorts first. Red by mutation of
    `recall.dependency_invalidation.build_dependency_projection` reading a node's root and upload
    status from its first chunk only (the rule's first version): fails on "an upload sorting
    first stripped the dependency" (2026-10-22 review of the per-copy rule, R1).
    """
    projection = build_dependency_projection(
        [
            _placed("a-upload", "README.md", "/m/uploads/tenant/job1/README.md"),
            _placed("b-memory", "README.md", "/z/README.md"),
            _placed("c-x", "x.md", "/z/x.md", depends_on=["README.md"]),
        ],
        tenant_id="tenant",
        generation_id="generation",
    )
    assert [(e.dependent, e.prerequisite) for e in projection.edges] == [("x.md", "README.md")], (
        f"an upload sorting first stripped the dependency: {projection.diagnostics}"
    )
    assert projection.reason_for("x.md") is None


def test_an_escaped_file_uri_stays_in_its_own_root() -> None:
    """Node tests/test_dependency_invalidation.py::test_an_escaped_file_uri_stays_in_its_own_root.

    Invariant: a node's root is read on the decoded location, so an upload `my notes.md` stored as
    a percent-escaped `file://` URI shares its job's root with its sibling `base.md` and resolves
    to it. Red by mutation of `recall.dependency_invalidation.build_dependency_projection` taking
    the root of the raw source: fails on "an escaped name lost its own root" (the encoded source
    did not end in the decoded file, so the name was a root of its own and the dependency, held
    only by an upload, went unresolved; 2026-10-22 review of the per-copy rule, R2).
    """
    job = "file:///C:/idx/uploads/tenant/job1"
    projection = build_dependency_projection(
        [
            _placed("base", "base.md", f"{job}/base.md"),
            _placed("notes", "my notes.md", f"{job}/my%20notes.md", depends_on=["base.md"]),
        ],
        tenant_id="tenant",
        generation_id="generation",
    )
    assert projection.reason_for("my notes.md") is None, (
        f"an escaped name lost its own root: {projection.reason_for('my notes.md')}"
    )
    assert [(e.dependent, e.prerequisite) for e in projection.edges] == [("my notes.md", "base.md")]


@pytest.mark.parametrize(
    "root,other_root",
    [("s3://one/root", "s3://two/root"), ("file:///one", "file:///two")],
    ids=["s3", "file"],
)
def test_encoded_generation_file_resolves_its_own_roots_copy(root: str, other_root: str) -> None:
    """Node tests/test_dependency_invalidation.py::test_encoded_generation_file_resolves_its_own_roots_copy.

    Encoded generation names must resolve their own root's prerequisite.
    Red-proved against 50cb206e1a4e593d495fca931570d6c303016154:
    _source_root's encoded/decoded mismatch failed the edge assertion.
    """
    file = "my%20notes.md"
    own = f"{root}/base.md"
    other = f"{other_root}/base.md"
    projection = build_dependency_projection(
        [
            _placed("own", "base.md", own),
            _placed("other", "base.md", other),
            _placed("notes", file, f"{root}/{file}", depends_on=["base.md"]),
        ],
        tenant_id="tenant",
        generation_id="generation",
        collided=frozenset({"base.md"}),
        base_states={own: "expired", other: "current"},
    )
    assert [(e.dependent, e.prerequisite) for e in projection.edges] == [(file, own)], (
        f"an encoded generation filename lost its own root: {projection.diagnostics}"
    )
    assert projection.dependencies[file] == ("base.md",)
    reason = projection.reason_for(file)
    assert reason is not None and reason.cause == "expired" and reason.dependency == own


@pytest.mark.parametrize(
    "root,other_root,source_name",
    [
        ("/one", "/two", "my%20notes.md"),
        ("s3://one/root", "s3://two/root", "my%2520notes.md"),
        ("file:///one", "file:///two", "my%2520notes.md"),
    ],
    ids=["plain-path", "s3-literal-percent", "file-literal-percent"],
)
def test_literal_percent_filenames_keep_their_own_root(root: str, other_root: str, source_name: str) -> None:
    """Node tests/test_dependency_invalidation.py::test_literal_percent_filenames_keep_their_own_root.

    Literal percent filenames must retain their own root.
    Red-proved by making _source_root unconditionally unquote(file);
    the edge assertion detected the second decoding.
    """
    file = "my%20notes.md"
    own = f"{root}/base.md"
    projection = build_dependency_projection(
        [
            _placed("own", "base.md", own),
            _placed("other", "base.md", f"{other_root}/base.md"),
            _placed("notes", file, f"{root}/{source_name}", depends_on=["base.md"]),
        ],
        tenant_id="tenant",
        generation_id="generation",
        collided=frozenset({"base.md"}),
    )
    assert [(e.dependent, e.prerequisite) for e in projection.edges] == [(file, own)], (
        f"a literal percent filename was decoded again: {projection.diagnostics}"
    )
    assert projection.reason_for(file) is None
