"""A context-mode profile must be indexable through the promotion harness.

`Indexer` defaults to `ContextPolicy()`, mode "none" / `raw-v1`, and REFUSES an embedder whose
profile declares anything else. `_indexed_store` never passed one, so every context profile was
unindexable through this harness and a campaign comparing context modes could not run.

Measured before the fix, on the deployment host, against the real corpus: all three context arms
died with `embedding profile context 'context-document-v1' does not match index context 'raw-v1'`
before a single question was scored, while the `raw-v1` baseline scored 110 of 110.

This pins the policy that `_indexed_store` derives, not the indexing itself: the derivation is the
part that was missing and the part a future edit can drop again. Indexing is covered against real
PostgreSQL elsewhere.
"""
from __future__ import annotations

import pytest

from recall.context import ContextPolicy, context_policy_for_profile
from recall.embedding_registry import context_version_for, registered_profile


@pytest.mark.parametrize(
    "profile_id,expected_mode",
    [
        ("bge-small-symmetric-v1", "none"),
        ("bge-small-asymmetric-v1", "none"),
        ("bge-small-context-document-v1", "document"),
        ("bge-small-context-section-v1", "section"),
        ("bge-small-context-neighbor-v1", "neighbor"),
    ],
)
def test_the_derived_policy_matches_what_indexer_demands(profile_id, expected_mode):
    """The exact comparison `Indexer.__init__` makes, run against every registered profile.

    Indexer computes `context_version_for(policy.mode, policy.version)` and refuses unless it
    equals the embedder profile's own `context_version`. If those two disagree for any profile,
    that profile cannot be indexed at all.
    """
    policy = context_policy_for_profile(profile_id)
    assert policy.mode == expected_mode
    assert context_version_for(policy.mode, policy.version) == registered_profile(
        profile_id
    ).context_version


def test_the_default_policy_cannot_index_a_context_profile():
    """The failure the fix removes, stated as a fact rather than as a story.

    `ContextPolicy()` is what `_indexed_store` used to leave `Indexer` with. Against a context
    profile it produces a version mismatch, which is exactly the refusal measured on the host.
    """
    default = ContextPolicy()
    assert context_version_for(default.mode, default.version) == "raw-v1"
    for profile_id in (
        "bge-small-context-document-v1",
        "bge-small-context-section-v1",
        "bge-small-context-neighbor-v1",
    ):
        assert registered_profile(profile_id).context_version != "raw-v1"


def test_indexed_store_passes_a_profile_derived_policy_to_the_indexer(monkeypatch):
    """The wiring itself: `_indexed_store` must hand `Indexer` the arm's own policy.

    Deleting the `context_policy=` argument makes this fail, which is what makes it a guard rather
    than a restatement of the two tests above.
    """
    import argparse

    import recall.eval.promotion.__main__ as cli

    seen: dict[str, object] = {}

    class StubEmbedder:
        dim = 3
        name = "stub"
        profile = registered_profile("bge-small-context-section-v1").identity(
            artifact_digest="a" * 64
        )

        def embed(self, texts):
            return [[0.0] * 3 for _ in texts]

    class StubStore:
        def __init__(self, *a, **kw):
            pass

        def ensure_schema(self):
            pass

        def drop_table(self):
            pass

        def close(self):
            pass

    class StubStats:
        chunks = 7

    class StubIndexer:
        def __init__(self, store, embedder, *, context_policy=ContextPolicy(), **kw):
            seen["policy"] = context_policy

        def index_path(self, path, glob=None):
            seen["glob"] = glob
            return StubStats()

    monkeypatch.setattr(cli, "uuid", __import__("uuid"))
    import recall.index
    import recall.store
    import recall_mcp.service

    monkeypatch.setattr(recall.index, "Indexer", StubIndexer)
    monkeypatch.setattr(recall.store, "PgVectorStore", StubStore)
    monkeypatch.setattr(recall_mcp.service, "make_embedder", lambda _name: StubEmbedder())

    args = argparse.Namespace(
        dsn="postgresql:///unused", embedder="fastembed", corpus_dir=".", glob="**/*.rst"
    )

    class Adapter:
        name = "stub"

    with cli._indexed_store(args, Adapter()):
        pass

    policy = seen["policy"]
    assert policy.mode == "section", "the Indexer was handed the default raw policy, not the arm's"
    assert context_version_for(policy.mode, policy.version) == "context-section-v1"


def test_the_glob_reaches_the_indexer_and_an_empty_index_is_refused(monkeypatch, tmp_path):
    """Two defects measured on the host in one run.

    The default glob is `**/*.md`; the PEPs corpus is `.rst`, so indexing wrote ZERO chunks and
    all 110 questions abstained against an empty index. The gate caught it downstream as a vacuous
    arm and blamed the label space, which was not the fault, after four arms had each paid for a
    full embedding pass.
    """
    import argparse

    import recall.eval.promotion.__main__ as cli
    import recall.index
    import recall.store
    import recall_mcp.service

    seen: dict[str, object] = {}

    class StubEmbedder:
        dim = 3
        name = "stub"
        profile = registered_profile("bge-small-symmetric-v1").identity(artifact_digest="a" * 64)

    class StubStore:
        def __init__(self, *a, **kw): pass
        def ensure_schema(self): pass
        def drop_table(self): pass
        def close(self): pass

    class EmptyStats:
        chunks = 0

    class StubIndexer:
        def __init__(self, store, embedder, **kw): pass
        def index_path(self, path, glob=None):
            seen["glob"] = glob
            return EmptyStats()

    monkeypatch.setattr(recall.index, "Indexer", StubIndexer)
    monkeypatch.setattr(recall.store, "PgVectorStore", StubStore)
    monkeypatch.setattr(recall_mcp.service, "make_embedder", lambda _n: StubEmbedder())

    class Adapter:
        name = "peps"

    args = argparse.Namespace(
        dsn="postgresql:///unused", embedder="fastembed", corpus_dir=str(tmp_path),
        glob="**/*.rst",
    )
    with pytest.raises(SystemExit) as exit_signal:
        with cli._indexed_store(args, Adapter()):
            pass

    assert seen["glob"] == "**/*.rst", "the --glob argument never reached index_path"
    message = str(exit_signal.value)
    assert "produced no chunks" in message
    assert "--glob" in message, "the refusal must name the knob that fixes it"
