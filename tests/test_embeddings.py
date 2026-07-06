import math

from recall.embeddings import Embedder, HashingEmbedder


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_hashing_embedder_satisfies_protocol():
    emb = HashingEmbedder(dim=64)
    assert isinstance(emb, Embedder)
    assert emb.dim == 64
    assert emb.name == "hashing-64"


def test_hashing_embedder_is_deterministic():
    emb = HashingEmbedder(dim=32)
    a = emb.embed(["the quick brown fox"])[0]
    b = emb.embed(["the quick brown fox"])[0]
    assert a == b
    assert len(a) == 32


def test_hashing_embedder_similar_text_closer_than_unrelated():
    emb = HashingEmbedder(dim=256)
    q = emb.embed(["database caching decision"])[0]
    near = emb.embed(["we made a caching decision for the database"])[0]
    far = emb.embed(["penguins waddle across antarctic ice"])[0]
    assert _cosine(q, near) > _cosine(q, far)
