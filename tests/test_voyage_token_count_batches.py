"""Token counting retains bounded encodings without changing document token counts."""

import hashlib
import sys
import types

import pytest

from recall.embedding_providers import voyage
from recall.embedding_providers.voyage import _voyage_token_counter as load_counter


def test_token_counting_releases_encodings_between_batches(monkeypatch):
    """Runs without optional SDKs. ecbbabd936 fails peak<=128 (257 live encodings)."""
    retention = {"live": 0, "peak": 0}

    class Encoding:
        def __init__(self, text):
            self.ids = list(range(len(text)))
            retention["live"] += 1
            retention["peak"] = max(retention["peak"], retention["live"])

        def __del__(self):
            retention["live"] -= 1

    class Tokenizer:
        padding = None

        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def no_truncation(self):
            pass

        def to_str(self):
            return "counting-test-tokenizer"

        def encode_batch(self, texts):
            return [Encoding(text) for text in texts]

    module = types.ModuleType("tokenizers")
    module.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    counter = load_counter("voyage-context-4")
    assert counter is not None
    texts = ["x" * (index % 7) for index in range(257)]
    counts = counter(texts)
    assert retention["peak"] <= 128
    assert retention["live"] == 0
    assert counts == [len(text) for text in texts]


@pytest.mark.parametrize("padding", [None, "longest", "fixed"])
def test_token_counting_bounds_encoding_retention_without_changing_counts(monkeypatch, padding):
    """Invariant: at most128 encodings retained, exact counts/order/identity preserved.

    Red proof: unchanged test on ecbbabd936af90164fe2cb25ba38ce23c320de80 fails
    the maximum batch-size assertion (259 >128) for all three padding modes.
    The full-group reference includes longest-batch and fixed padding; the long
    text crosses the counting boundary, so per-batch padding alone is insufficient.
    """
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[PAD]": 0, "[UNK]": 1, "word": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    if padding == "longest":
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", pad_to_multiple_of=8)
    elif padding == "fixed":
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=16)
    texts = ["word"] * 128 + [" ".join(["word"] * 19), ""] + ["word"] * 129
    expected = [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
    expected_small = [len(encoding.ids) for encoding in tokenizer.encode_batch(texts[:3])]
    serialized = tokenizer.to_str()
    batches = []
    retention = {"live": 0, "peak": 0}

    class Encoding:
        def __init__(self, raw):
            self.raw = raw
            retention["live"] += 1
            retention["peak"] = max(retention["peak"], retention["live"])

        @property
        def ids(self):
            return self.raw.ids

        def __del__(self):
            retention["live"] -= 1

    class ObservedTokenizer:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def encode_batch(self, batch):
            batches.append(list(batch))
            return [Encoding(raw) for raw in tokenizer.encode_batch(batch)]

    class Factory:
        @staticmethod
        def from_pretrained(model):
            assert model == "voyageai/voyage-context-4"
            return ObservedTokenizer()

    module = types.ModuleType("tokenizers")
    module.Tokenizer = Factory
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    monkeypatch.setattr(voyage, "_package_version", lambda name: "test-runtime")
    counter = load_counter("voyage-context-4")
    assert isinstance(counter, voyage._IdentifiedTokenCounter)
    counts = counter(texts)

    assert max(map(len, batches)) <= 128
    assert retention["peak"] <= 128
    assert retention["live"] == 0
    assert counts == expected
    assert [text for batch in batches for text in batch] == texts
    artifact = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert counter.identity == f"tokenizers:test-runtime:sha256:{artifact}"
    assert tokenizer.to_str() == serialized
    assert counter([]) == []
    assert counter(texts[:3]) == expected_small
