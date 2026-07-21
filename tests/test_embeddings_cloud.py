import os

import pytest

from recall.embeddings import Embedder, VoyageEmbedder

requires_voyage = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"), reason="no VOYAGE_API_KEY"
)


def test_voyage_requires_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        VoyageEmbedder()


@requires_voyage
def test_voyage_roundtrip():
    emb = VoyageEmbedder()
    vecs = emb.embed(["hello world"])
    assert isinstance(emb, Embedder)
    assert emb.dim > 0 and len(vecs) == 1 and len(vecs[0]) == emb.dim


class FakeVoyageClient:
    """Records the batches it was asked to embed; optionally fails the first N calls."""

    def __init__(self, fail_times: int = 0, error: Exception | None = None):
        self.batches: list[list[str]] = []
        self._fail_times = fail_times
        self._error = error or RuntimeError("RateLimitError: slow down")

    def embed(self, texts, model):
        self.batches.append(list(texts))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._error
        return type("R", (), {"embeddings": [[float(len(t))] for t in texts]})()


def test_embed_batched_splits_into_provider_sized_batches():
    from recall.embeddings import _embed_batched

    client = FakeVoyageClient()
    texts = [f"t{i}" for i in range(250)]
    vecs = _embed_batched(client, "voyage-3", texts, batch_size=100)
    assert [len(b) for b in client.batches] == [100, 100, 50]
    assert len(vecs) == 250


def test_embed_batched_preserves_input_order():
    from recall.embeddings import _embed_batched

    client = FakeVoyageClient()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vecs = _embed_batched(client, "voyage-3", texts, batch_size=2)
    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_embed_batched_retries_a_transient_failure():
    from recall.embeddings import _embed_batched

    client = FakeVoyageClient(fail_times=2)
    slept: list[float] = []
    vecs = _embed_batched(
        client, "voyage-3", ["a"], batch_size=10, max_retries=3, sleep=slept.append
    )
    assert len(vecs) == 1
    assert len(client.batches) == 3  # two failures then success
    assert slept == [1.0, 2.0]  # exponential backoff


def test_embed_batched_gives_up_after_max_retries():
    import pytest as _pytest

    from recall.embeddings import _embed_batched

    client = FakeVoyageClient(fail_times=99)
    with _pytest.raises(RuntimeError):
        _embed_batched(client, "voyage-3", ["a"], batch_size=10, max_retries=2, sleep=lambda _: None)
    assert len(client.batches) == 2


def test_embed_batched_does_not_retry_a_permanent_error():
    import pytest as _pytest

    from recall.embeddings import _embed_batched

    client = FakeVoyageClient(fail_times=99, error=RuntimeError("AuthenticationError: bad key"))
    with _pytest.raises(RuntimeError, match="Authentication"):
        _embed_batched(client, "voyage-3", ["a"], batch_size=10, max_retries=4, sleep=lambda _: None)
    assert len(client.batches) == 1  # an invalid key will never succeed — fail fast


def test_embed_batched_empty_input_makes_no_call():
    from recall.embeddings import _embed_batched

    client = FakeVoyageClient()
    assert _embed_batched(client, "voyage-3", [], batch_size=10) == []
    assert client.batches == []


def test_embed_batched_rejects_a_nonpositive_retry_budget():
    import pytest as _pytest

    from recall.embeddings import _embed_batched

    # a zero budget would skip the attempt loop entirely and fall through to an unbound result
    with _pytest.raises(ValueError, match="max_retries"):
        _embed_batched(FakeVoyageClient(), "voyage-3", ["a"], max_retries=0)


def test_embed_batched_rejects_a_short_batch_response():
    """A provider returning fewer vectors than texts must fail loudly, not silently misalign.

    The Embedder contract is positional: chunk i pairs with vector i. If one batch comes back
    short, every later chunk is stored against its neighbour's vector — and the only downstream
    check is on the TOTAL count, which a compensating long batch would satisfy.
    """
    import pytest as _pytest

    from recall.embeddings import _embed_batched

    class ShortClient:
        def embed(self, texts, model):
            return type("R", (), {"embeddings": [[1.0]] * (len(texts) - 1)})()

    with _pytest.raises(RuntimeError, match="2 embeddings for 3 texts"):
        _embed_batched(ShortClient(), "voyage-3", ["a", "b", "c"], batch_size=3)


def test_embed_batched_rejects_a_nonpositive_batch_size():
    import pytest as _pytest

    from recall.embeddings import _embed_batched

    # range(0, n, -5) is empty -> silently returns [], caught only far downstream
    with _pytest.raises(ValueError, match="batch_size"):
        _embed_batched(FakeVoyageClient(), "voyage-3", ["a"], batch_size=-5)
