"""A large Add's embedding time is spent on Voyage, not on cache keys, and its requests overlap.

Measured 2026-09-24 on C9 with one 600,000 character Add (451.9 s in total): each of the 7,169
atomic view keys encoded the whole 1.35 M character group, 20 ms per key, about 144 s of CPU
for Context 4, and Code 4 took the same group-keyed path because `LockedEmbedder` always
exposes `embed_document_groups`. Every Voyage request was also sent one after another. Each
behaviour test records the mutation it was proved red against, with this file unchanged.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import recall.embedding_registry as registry
import recall.embeddings as embeddings
import recall_aml.embedding_lock as embedding_lock
from recall.embeddings import VoyageContextualizedEmbedder, batched_embed
from recall_aml.embedding_lock import CachedEmbedder, LockedEmbedder


class InFlight:
    """Counts how many calls overlap, with a short sleep so overlap is observable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.calls = 0

    def __enter__(self) -> "InFlight":
        with self._lock:
            self.current += 1
            self.calls += 1
            self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *exc: object) -> None:
        time.sleep(0.05)
        with self._lock:
            self.current -= 1


def vector(text: str, *context: str) -> list[float]:
    return [float(len(text)), float(sum(len(item) for item in context))]


class PlainProvider:
    """A non-contextual provider: a text's vector depends on that text alone."""

    dim = 2
    name = "plain-test"
    profile = None

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [vector(text) for text in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class ContextProvider(PlainProvider):
    """A contextual provider: a text's vector depends on its whole group."""

    name = "context-test"

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        self.embedded.extend(text for group in groups for text in group)
        return [[vector(text, *group) for text in group] for group in groups]


def no_lock(_path: Path) -> Any:
    from contextlib import nullcontext

    return nullcontext()


def test_context_cache_keys_do_not_encode_the_group(monkeypatch, tmp_path):
    """Red proof: `CachedEmbedder.embed_document_groups` keyed with the old
    `value={"group": group, "ordinal": ordinal}`. This test then failed on the key size bound
    (every key carried the whole 200 text group)."""
    sizes: list[int] = []
    real = embedding_lock._structured_cache_key

    def measuring(**kwargs: Any) -> str:
        sizes.append(len(json.dumps(kwargs["value"])))
        return real(**kwargs)

    monkeypatch.setattr(embedding_lock, "_structured_cache_key", measuring)
    cached = CachedEmbedder(ContextProvider(), tmp_path / "cache.sqlite")
    group = [f"atomic view {i} " + "word " * 20 for i in range(200)]

    cached.embed_document_groups([group])

    assert len(sizes) == len(group)
    assert max(sizes) < 200


def test_context_cache_still_separates_groups_and_reuses_a_repeated_one(tmp_path):
    """Red proof: `_group_digest` made to return a constant. This test then failed on
    `assert provider.embedded == group_b` (the second group was served the first group's
    vectors from the cache)."""
    provider = ContextProvider()
    cached = CachedEmbedder(provider, tmp_path / "cache.sqlite")
    group_a = ["shared text", "only in a"]
    group_b = ["shared text", "only in b"]

    first = cached.embed_document_groups([group_a])
    provider.embedded.clear()
    again = cached.embed_document_groups([group_a])
    assert provider.embedded == []
    assert again == first

    cached.embed_document_groups([group_b])
    assert provider.embedded == group_b


def test_a_non_contextual_stack_caches_per_text_across_calls(tmp_path):
    """Red proof: `CachedEmbedder.embed_passages` checking the outermost wrapper again
    (`callable(getattr(self._inner, "embed_document_groups", None))`). This test then failed on
    `assert provider.embedded == ["new neighbour"]` (the repeated text was embedded again)."""
    provider = PlainProvider()
    cached = CachedEmbedder(
        LockedEmbedder(provider, tmp_path / "lock", lock_factory=no_lock), tmp_path / "c.sqlite"
    )

    cached.embed_passages(["repeated window", "first neighbour"])
    provider.embedded.clear()
    vectors = cached.embed_passages(["repeated window", "new neighbour"])

    assert provider.embedded == ["new neighbour"]
    assert vectors == [vector("repeated window"), vector("new neighbour")]


def test_batched_embed_sends_batches_concurrently_in_input_order():
    """Red proof: `batched_embed` with its `if max_workers > 1:` branch disabled. This test then
    failed on `assert flight.peak > 1`."""
    flight = InFlight()

    def embed_batch(batch: list[str]) -> list[list[float]]:
        with flight:
            return [vector(text) for text in batch]

    texts = [f"text {i:03d}" + "x" * i for i in range(40)]
    vectors = batched_embed(texts, embed_batch, batch_size=5, max_workers=4)

    assert vectors == [vector(text) for text in texts]
    assert flight.calls == 8
    assert flight.peak > 1


class FakeContextClient:
    def __init__(self, flight: InFlight) -> None:
        self.flight = flight

    def contextualized_embed(self, *, inputs: list[list[str]], **_: Any) -> Any:
        from types import SimpleNamespace

        with self.flight:
            return SimpleNamespace(
                results=[
                    SimpleNamespace(embeddings=[vector(text, *group) for text in group])
                    for group in inputs
                ]
            )


def context_embedder(parallel: int, flight: InFlight) -> VoyageContextualizedEmbedder:
    embedder = object.__new__(VoyageContextualizedEmbedder)
    embedder._client = FakeContextClient(flight)
    embedder._model = "voyage-context-4"
    embedder._output_dimension = 2
    embedder._max_request_inputs = 1000
    embedder._max_request_chunks = 16_000
    embedder._max_request_chars = 300
    embedder._max_retries = 1
    embedder._max_parallel_requests = parallel
    return embedder


def test_context_requests_overlap_and_return_the_sequential_vectors():
    """Red proof: `embed_document_groups` with its parallel branch condition changed to
    `self._max_parallel_requests > 99`. This test then failed on `assert parallel.peak > 1`."""
    groups = [[f"g{g} chunk {i} " + "w" * 30 for i in range(6)] for g in range(4)]
    sequential = InFlight()
    parallel = InFlight()

    expected = context_embedder(1, sequential).embed_document_groups(groups)
    actual = context_embedder(4, parallel).embed_document_groups(groups)

    assert actual == expected
    assert sequential.peak == 1
    assert parallel.calls == sequential.calls > 1
    assert parallel.peak > 1


def test_the_registry_passes_the_parallel_setting_to_both_voyage_backends(monkeypatch):
    """Red proof: `max_parallel_requests=_voyage_parallel_requests(env)` removed from the
    `voyage-context` constructor call in `RegisteredProfile._construct`. This test then failed
    on the context profile's recorded setting (missing, so the default 1)."""
    recorded: dict[str, int] = {}

    def recorder(label: str):
        def build(**kwargs: Any) -> Any:
            recorded[label] = kwargs.get("max_parallel_requests", 1)
            raise RuntimeError("stop after construction arguments")

        return build

    monkeypatch.setattr(registry, "VoyageEmbedder", recorder("voyage"))
    monkeypatch.setattr(registry, "VoyageContextualizedEmbedder", recorder("context"))
    env = {"VOYAGE_API_KEY": "k", "RECALL_VOYAGE_PARALLEL_REQUESTS": "4"}
    for profile in ("voyage-code-4-v1", "voyage-context-4-v1"):
        try:
            embeddings.resolve_registered_embedder(profile, env)
        except RuntimeError:
            pass

    assert recorded == {"voyage": 4, "context": 4}


def test_the_parallel_setting_is_validated():
    """Nonbehavioural control of the setting's parser: default 1, bounds 1 to 16."""
    assert registry._voyage_parallel_requests(None) == 1
    assert registry._voyage_parallel_requests({"RECALL_VOYAGE_PARALLEL_REQUESTS": "4"}) == 4
    for bad in ("0", "17", "x"):
        try:
            registry._voyage_parallel_requests({"RECALL_VOYAGE_PARALLEL_REQUESTS": bad})
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
