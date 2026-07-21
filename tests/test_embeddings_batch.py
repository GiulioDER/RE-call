import pytest

from recall.embeddings import batched_embed, retry_with_backoff


def test_batched_embed_splits_and_preserves_order():
    calls: list[list[str]] = []

    def embed_batch(batch):
        calls.append(list(batch))
        # a distinct vector per text so order is verifiable
        return [[float(len(t))] for t in batch]

    texts = [f"t{i}" * (i + 1) for i in range(10)]
    out = batched_embed(texts, embed_batch, batch_size=3)

    assert len(calls) == 4  # 3 + 3 + 3 + 1
    assert [len(c) for c in calls] == [3, 3, 3, 1]
    assert out == [[float(len(t))] for t in texts]  # order preserved end-to-end


def test_batched_embed_char_budget_cuts_batches():
    calls: list[list[str]] = []

    def embed_batch(batch):
        calls.append(list(batch))
        return [[0.0] for _ in batch]

    # each text is 10 chars; budget 25 -> at most 2 per batch even though batch_size is huge
    texts = ["x" * 10 for _ in range(5)]
    batched_embed(texts, embed_batch, batch_size=100, max_batch_chars=25)
    assert [len(c) for c in calls] == [2, 2, 1]


def test_batched_embed_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError):
        batched_embed(["a"], lambda b: [[0.0]], batch_size=0)


def test_retry_recovers_from_transient_failure():
    attempts = {"n": 0}
    slept: list[float] = []

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("429 too many requests")
        return "ok"

    out = retry_with_backoff(flaky, attempts=3, base_delay=0.01, sleep=slept.append)
    assert out == "ok"
    assert attempts["n"] == 3
    assert len(slept) == 2  # slept before each of the two retries


def test_retry_reraises_nontransient_immediately():
    attempts = {"n": 0}

    def bad():
        attempts["n"] += 1
        raise ValueError("401 unauthorized")  # not transient

    with pytest.raises(ValueError):
        retry_with_backoff(bad, attempts=5, sleep=lambda _s: None)
    assert attempts["n"] == 1  # gave up without retrying


def test_retry_reraises_after_exhausting_attempts():
    def always():
        raise RuntimeError("503 unavailable")

    with pytest.raises(RuntimeError):
        retry_with_backoff(always, attempts=3, base_delay=0.0, sleep=lambda _s: None)


def test_batched_embed_rejects_a_short_batch_response():
    """A provider returning fewer vectors than texts must fail loudly, not silently misalign.

    The contract is positional: chunk i pairs with vector i. If one batch comes back short,
    every later chunk is stored against its neighbour's vector — and the only downstream check
    is the TOTAL count, which a compensating long batch would satisfy.
    """
    import pytest

    from recall.embeddings import batched_embed

    def short(batch):
        return [[1.0]] * (len(batch) - 1)

    with pytest.raises(RuntimeError, match="2 embeddings for 3 texts"):
        batched_embed(["a", "b", "c"], short, batch_size=3)
