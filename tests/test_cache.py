from recall.cache import EmbeddingCache, cache_key, embed_with_cache


class CountingEmbedder:
    """Deterministic embedder that records exactly which texts it was asked to embed."""

    dim = 2
    name = "counting"

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]


def test_cache_returns_cached_vector_without_re_embedding(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    emb = CountingEmbedder()

    first = embed_with_cache(emb, ["alpha", "beta"], cache)
    assert emb.embedded == ["alpha", "beta"]  # both embedded on the cold cache

    second = embed_with_cache(emb, ["alpha", "beta"], cache)
    assert emb.embedded == ["alpha", "beta"]  # unchanged: nothing re-embedded
    assert second == first


def test_cache_embeds_only_the_misses_preserving_order(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    emb = CountingEmbedder()

    embed_with_cache(emb, ["alpha"], cache)  # warm "alpha"
    emb.embedded.clear()

    out = embed_with_cache(emb, ["alpha", "gamma", "alpha"], cache)
    assert emb.embedded == ["gamma"]  # only the miss hit the embedder
    assert out == [[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]]  # order preserved, cached values reused


def test_cache_none_is_plain_embed(tmp_path):
    emb = CountingEmbedder()
    out = embed_with_cache(emb, ["x", "y"], None)
    assert emb.embedded == ["x", "y"]
    assert out == [[1.0, 1.0], [1.0, 1.0]]


def test_cache_key_separates_embedders_and_dims():
    assert cache_key("a", 2, "hello") != cache_key("b", 2, "hello")
    assert cache_key("a", 2, "hello") != cache_key("a", 3, "hello")
    assert cache_key("a", 2, "hello") == cache_key("a", 2, "hello")
