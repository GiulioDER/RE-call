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


class _RecordingModel:
    """Stands in for a fastembed `TextEmbedding`, recording how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def passage_embed(self, texts, **kwargs):
        self.calls.append({"n": len(list(texts)), **kwargs})
        return [[0.0, 1.0] for _ in texts]

    # Some profiles name the passage encoder `embed` rather than `passage_embed`.
    embed = passage_embed


def _embedder_with(model, mode="passage_embed"):
    """A `FastEmbedEmbedder` around a stub, without constructing a real model.

    `__init__` eagerly builds a `TextEmbedding` and downloads weights, which is precisely what a
    unit test of one method must not do. Only the two attributes `embed_passages` reads are set.
    """
    from recall.embeddings import FastEmbedEmbedder

    embedder = object.__new__(FastEmbedEmbedder)
    embedder._model = model
    embedder._passage_mode = mode
    return embedder


def test_the_passage_batch_is_unbounded_by_default(monkeypatch):
    """⚠️ The default must not move. Unset, the whole list goes to fastembed as before."""
    monkeypatch.delenv("RECALL_FASTEMBED_BATCH", raising=False)
    model = _RecordingModel()

    _embedder_with(model).embed_passages(["a", "b", "c"])

    assert len(model.calls) == 1
    assert "batch_size" not in model.calls[0], (
        "passing a batch size when the variable is unset would change the default for every "
        "existing corpus, which this knob exists to avoid"
    )


def test_the_batch_size_is_passed_to_fastembed_when_set(monkeypatch):
    """⛔ **bge-large at fastembed's default batch of 256 asks for a 1.24 GB allocation.**

    Measured while indexing a real memory corpus: onnxruntime's arena refused with
    `Failed to allocate memory for requested buffer of size 1239040000`, part way through, after
    several projects had already been written. A hard failure mid-index, not a slow run.

    Passed as fastembed's OWN parameter rather than by slicing the list here. An earlier version
    sliced and called the encoder once per slice, and at size 16 that wedged: 152 of 208 files in,
    eight cores pinned, no database writes for three minutes, and every server connection idle in
    `ClientRead`, so the stall was in this process rather than on I/O.
    """
    monkeypatch.setenv("RECALL_FASTEMBED_BATCH", "16")
    model = _RecordingModel()

    vectors = _embedder_with(model).embed_passages(["a", "b", "c"])

    assert len(model.calls) == 1, "one call: fastembed does its own batching internally"
    assert model.calls[0]["batch_size"] == 16
    assert len(vectors) == 3


def test_a_backend_that_rejects_the_argument_still_embeds(monkeypatch):
    """The variable is a memory guard, not a contract: an encoder without it must still work."""
    monkeypatch.setenv("RECALL_FASTEMBED_BATCH", "8")

    class _Strict:
        def __init__(self) -> None:
            self.calls = 0

        def passage_embed(self, texts):
            self.calls += 1
            return [[0.0, 1.0] for _ in texts]

    model = _Strict()
    vectors = _embedder_with(model).embed_passages(["a", "b"])

    assert len(vectors) == 2, "a TypeError from the batch argument must not fail the embed"
    # ONE call, not two: Python raises `TypeError` while BINDING the arguments, before the function
    # body runs, so the rejected attempt never reaches the counter. My first version of this
    # assertion expected two and was wrong about the language, not about the code.
    assert model.calls == 1


def test_a_junk_value_does_not_crash_an_index(monkeypatch):
    """A mistyped variable must degrade to the default rather than abort a long index run."""
    monkeypatch.setenv("RECALL_FASTEMBED_BATCH", "not-a-number")
    model = _RecordingModel()

    vectors = _embedder_with(model).embed_passages(["a", "b"])

    assert len(vectors) == 2


def test_a_bad_batch_value_is_reported_once_not_once_per_batch(monkeypatch, caplog):
    """⛔ `_batch_size_from_env` runs once per BATCH, so an unconditional warning is log spam.

    During an index of a real corpus that is hundreds of identical lines, which buries anything
    worth reading. The docstring claimed "warns once" while the code warned every time, which is
    the mismatch this fix closes rather than the wording.
    """
    import logging

    from recall.embeddings import _WARNED_BATCH_VALUES, _batch_size_from_env

    _WARNED_BATCH_VALUES.clear()
    monkeypatch.setenv("RECALL_FASTEMBED_BATCH", "not-a-number")

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert _batch_size_from_env() is None

    warnings = [r for r in caplog.records if "RECALL_FASTEMBED_BATCH" in r.getMessage()]
    assert len(warnings) == 1, f"warned {len(warnings)} times for one bad value"

    # A DIFFERENT bad value is still worth reporting: the setting may have been corrected badly.
    monkeypatch.setenv("RECALL_FASTEMBED_BATCH", "-4")
    with caplog.at_level(logging.WARNING):
        assert _batch_size_from_env() is None
    warnings = [r for r in caplog.records if "RECALL_FASTEMBED_BATCH" in r.getMessage()]
    assert len(warnings) == 2
