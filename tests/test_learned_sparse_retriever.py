"""`HybridRetriever(sparse_backend=...)`: is the learned sparse leg actually wired in?

The first test here is the one that matters. Without it, every other test in this feature passes
with `sparse_backend` connected to nothing at all.
"""

from __future__ import annotations

import pytest

from recall.retriever import HybridRetriever
from recall.sparse import SparseProfile
from recall.types import Chunk
from tests.conftest import requires_db

PROFILE_ID = "kw-test"


class KeywordSparseEncoder:
    """A real, deterministic sparse encoder: one term id per distinct word.

    Not a mock of the system under test — it is a genuine implementation of the encoder protocol
    the retriever depends on, chosen over a real SPLADE checkpoint so these tests need no 500MB
    download and no network. The retrieval path it drives is the production one.
    """

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {self._vocabulary[w]: 1.0 for w in text.lower().split() if w in self._vocabulary}
            for text in texts
        ]


class StubEmbedder:
    """Constant dense vectors, so the DENSE leg cannot be what distinguishes the arms below."""

    dim = 64
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 64 for _ in texts]


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source=f"/c/{cid}.md", text=text, metadata={"file": f"{cid}.md"})


@requires_db
def test_switching_the_sparse_backend_changes_which_documents_come_back(make_store) -> None:
    """THE load-bearing test: prove `sparse_backend` is connected to something.

    Both documents are dense-identical (constant embedder), so the dense leg cannot separate
    them. `lexical` searches the TEXT, where only `alpha` contains "aardvark". The learned sparse
    arm is given a corpus where only `beta` carries that query's term. So the two backends must
    disagree — and if the parameter were ignored, they would agree and this test would fail.

    A test that merely asserted "splade returns something" would pass with the parameter wired to
    nothing, which is the failure this whole file exists to exclude.
    """
    store = make_store(64)
    store.upsert(
        [_chunk("alpha", "aardvark facts"), _chunk("beta", "unrelated prose")],
        [[0.1] * 64, [0.1] * 64],
    )
    # Deliberately INVERTED against the text: the learned sparse index says "beta", the words say
    # "alpha". Only a genuinely separate leg can produce the inverted answer.
    store.upsert_sparse(PROFILE_ID, {"beta": {7: 3.0}})
    encoder = KeywordSparseEncoder({"aardvark": 7})

    def _retriever(backend: str) -> HybridRetriever:
        return HybridRetriever(
            store, StubEmbedder(), sparse_backend=backend,
            sparse_encoder=encoder if backend != "lexical" else None,
        )

    lexical = [h.chunk.id for h in _retriever("lexical").search("aardvark", k=5).hits]
    splade = [h.chunk.id for h in _retriever("splade").search("aardvark", k=5).hits]

    # Both arms return both documents — the dense leg is still on and, with a constant embedder,
    # it retrieves everything. What the sparse leg decides is the ORDER: whichever document also
    # gets sparse RRF credit is ranked first. So the discriminating assertion is the top hit, and
    # if `sparse_backend` were ignored both arms would answer "alpha" and this would fail.
    assert lexical[0] == "alpha"
    assert splade[0] == "beta"


@requires_db
def test_selecting_splade_without_an_encoder_is_refused(make_store) -> None:
    """A backend that cannot encode its own queries is a misconfiguration, not a silent fallback.

    Falling back to lexical here would mean an operator who asked for learned sparse gets
    keyword search and a set of numbers that look like a SPLADE result.
    """
    store = make_store(64)

    with pytest.raises(ValueError, match="sparse_encoder"):
        HybridRetriever(store, StubEmbedder(), sparse_backend="splade")


@requires_db
@pytest.mark.parametrize("spelling", ["production", "Production", " production "])
def test_the_learned_sparse_leg_refuses_to_run_in_production(
    make_store, monkeypatch, spelling
) -> None:
    """The sidecar has no FOREIGN KEY, so nothing cascades on Postgres's side.

    Erasure is wired now (the delete paths scrub `recall_sparse_v1` in the same transaction),
    but a corpus encoded before that fix may carry orphaned rows, and SPLADE weights over the
    vocabulary are partially reconstructable content. Until an orphan sweep exists, production
    stays refused. Every spelling of "production" must refuse: the old bare compare meant
    `RECALL_ENV=Production` silently DISABLED this gate through a capital letter.
    """
    monkeypatch.setenv("RECALL_ENV", spelling)
    store = make_store(64)

    with pytest.raises(RuntimeError, match="orphan"):
        HybridRetriever(
            store, StubEmbedder(), sparse_backend="splade",
            sparse_encoder=KeywordSparseEncoder({"aardvark": 7}),
        )
