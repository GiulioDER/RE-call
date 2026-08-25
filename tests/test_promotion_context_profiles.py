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
        ("bge-large-context-section-v1", "section"),
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


# ------------------------------------------------- the contract, exercised rather than restated


@pytest.mark.parametrize(
    "profile_id",
    [
        "bge-small-symmetric-v1",
        "bge-small-asymmetric-v1",
        "bge-small-context-document-v1",
        "bge-small-context-section-v1",
        "bge-small-context-neighbor-v1",
        "bge-large-context-section-v1",
    ],
)
def test_a_real_indexer_accepts_every_registered_profile_with_its_derived_policy(profile_id):
    """Constructs the REAL `Indexer`, which the tests above only described.

    The CCA bug auditor showed the gap: mutating `Indexer.__init__` back to
    `expected_context = "raw-v1"` — literally the defect this file exists about — left all eight
    tests green, because they re-implemented the comparison instead of making it. `Indexer.__init__`
    touches only `store` and `embedder`, so a stub store is enough and no database is needed.
    """
    from recall.embeddings import EmbeddingProfile
    from recall.index import Indexer

    registered = registered_profile(profile_id)
    profile: EmbeddingProfile = registered.identity(artifact_digest="b" * 64)

    class StubStore:
        def source_content_hashes(self):
            return {}

    class StubEmbedder:
        dim = registered.dimension
        name = registered.model_name

        def __init__(self):
            self.profile = profile

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

    # No pytest.raises: the point is that this does NOT raise. Under the mutation it does.
    Indexer(
        StubStore(),
        StubEmbedder(),
        context_policy=context_policy_for_profile(profile_id),
    )


def test_a_real_indexer_refuses_a_profile_whose_policy_does_not_match():
    """The other half. Without this, the test above passes against an Indexer that checks nothing."""
    from recall.index import Indexer

    profile = registered_profile("bge-small-context-section-v1").identity(artifact_digest="b" * 64)

    class StubStore:
        def source_content_hashes(self):
            return {}

    class StubEmbedder:
        dim = 384
        name = "stub"

        def __init__(self):
            self.profile = profile

    # The default this test relies on. After the redundant test was deleted, a mutant setting
    # ContextPolicy's default mode to 'document' survived the ENTIRE suite: 2356 passed,
    # byte-identical to baseline. Both Indexer and ShadowIndexTarget default to it.
    assert context_version_for(ContextPolicy().mode, ContextPolicy().version) == "raw-v1"
    with pytest.raises(ValueError, match="does not match index context"):
        Indexer(StubStore(), StubEmbedder(), context_policy=ContextPolicy())


# ------------------------------------------------------------------- the over-broad glob refusal


@pytest.mark.parametrize(
    "glob",
    [
        "*", "**", "**/*", "**/*.*", "*.*", "docs/*", "", "**/",
        # Every one of these was MEASURED passing the first version of the guard and
        # pulling .env / .npmrc / tokens.json out of a seeded tree. The first four
        # reconstruct a wildcard with a character class or a "?"; the last matches
        # dotfiles EXCLUSIVELY and was the most dangerous pattern of all.
        "**/*.?*", "**/*.[a-z]*", "**/*.[!x]*", "**/*.j*", "**/[.]*",
        "docs/*.*", "**/.*", "**/*.", "**/README",
        # A dotfile basename is NOT an extension, but the anchored regex reads `.env` as
        # `\.` + `env` at end of string and ACCEPTED it, walking to the real file. This is
        # the hole the leading-dot clause was deleted into and then restored to close.
        "**/.env", ".env", "**/.npmrc", "**/.git-credentials",
        # Windows separator: `rsplit("/")` alone leaves a final component that does not
        # start with a dot, while `Path.glob` still returns the file.
        "**\\.env", "nested\\.env",
        # `$` matches before a trailing newline; `\Z` does not.
        "**/*.md\n",
        # A `?` or a class AS the extension, so widening the extension class is caught.
        "**/*.?", "**/*.[a-z]", "**/*.[!x]",
    ],
)
def test_a_glob_without_a_literal_extension_is_refused(glob):
    """The empty-index refusal is one-sided: it cannot fire when the glob is too BROAD, because
    that yields chunks > 0. `**/*` was measured pulling `.env`, `tokens.json` and `id_rsa` into the
    candidate set."""
    from recall.eval.promotion.__main__ import _refuse_overbroad_glob

    with pytest.raises(SystemExit) as exit_signal:
        _refuse_overbroad_glob(glob)
    assert "--glob" in str(exit_signal.value)
    assert "extension" in str(exit_signal.value)


@pytest.mark.parametrize(
    "glob",
    [
        "**/*.rst", "**/*.md", "*.txt", "corpus/**/*.rst",
        # Named in the docstring as validated, so pin them rather than assert them.
        "**/*.py", "**/*.tar.gz",
        # A class or a `?` EARLIER in the component is fine: the literal tail bounds it.
        "**/[a-z]*.md", "**/?.md",
    ],
)
def test_a_glob_naming_an_extension_is_accepted(glob):
    """It must not refuse the patterns the harness actually uses, or it is just noise."""
    from recall.eval.promotion.__main__ import _refuse_overbroad_glob

    _refuse_overbroad_glob(glob)


def test_indexed_store_actually_calls_the_overbroad_refusal(monkeypatch, tmp_path):
    """The wiring, not the predicate.

    A mutation sweep caught this gap: deleting the `_refuse_overbroad_glob(args.glob)` CALL from
    `_indexed_store` left all 24 tests green, because every other test invokes the predicate
    directly. A test that pins a function does not pin its caller — the same shape as an artifact
    test that cannot see its emitter losing a field.
    """
    import argparse

    import recall.eval.promotion.__main__ as cli
    import recall.index
    import recall.store
    import recall_mcp.service

    opened = {"n": 0}

    class StubEmbedder:
        dim = 3
        name = "stub"
        profile = registered_profile("bge-small-symmetric-v1").identity(artifact_digest="a" * 64)

    class StubStore:
        def __init__(self, *a, **kw):
            opened["n"] += 1

        def ensure_schema(self): pass
        def drop_table(self): pass
        def close(self): pass

    class StubIndexer:
        def __init__(self, store, embedder, **kw): pass
        def index_path(self, path, glob=None):
            raise AssertionError("indexing must not be reached with an over-broad glob")

    monkeypatch.setattr(recall.index, "Indexer", StubIndexer)
    monkeypatch.setattr(recall.store, "PgVectorStore", StubStore)
    monkeypatch.setattr(recall_mcp.service, "make_embedder", lambda _n: StubEmbedder())

    class Adapter:
        name = "peps"

    args = argparse.Namespace(
        dsn="postgresql:///unused", embedder="fastembed", corpus_dir=str(tmp_path), glob="**/*",
    )
    with pytest.raises(SystemExit) as exit_signal:
        with cli._indexed_store(args, Adapter()):
            pass
    assert "extension" in str(exit_signal.value)
    # The refusal is a pure predicate over argv and now runs BEFORE the model load and before any
    # database work. Measured order used to be make_embedder -> connect -> CREATE TABLE -> refuse.
    assert opened["n"] == 0, "the operator's database was touched for a request already refused"
