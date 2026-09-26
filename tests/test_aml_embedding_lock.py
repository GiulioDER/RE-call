"""Safety proofs for serialized hosted embedding calls."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

import pytest

from recall_aml.config import HostedSettings
from recall_aml.embedding_lock import (
    CachedEmbedder,
    CachedMultimodalEmbedder,
    LockedEmbedder,
    LockedMultimodalEmbedder,
)
from recall_aml.variants import variant
from scripts.aml_release_manifest import build_manifest


class _TextEmbedder:
    dim = 3
    name = "test-text"
    profile = "test-profile"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.events.append("embed")
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.events.append("query")
        return [1.0, 0.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.events.append("passages")
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        self.events.append("groups")
        return [[[1.0, 0.0, 0.0] for _ in group] for group in groups]


class _MultimodalEmbedder:
    dim = 3
    profile = "test-multimodal"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed_documents(self, inputs):
        self.events.append("documents")
        return [[0.0, 1.0, 0.0] for _ in inputs]

    def embed_query(self, value):
        self.events.append("multimodal-query")
        return [0.0, 1.0, 0.0]


class _IndependentTextEmbedder:
    dim = 3
    name = "test-independent-text"
    profile = "test-independent-profile"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.events.append("embed")
        return [[float(len(text)), 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.events.append("query")
        return [float(len(text)), 1.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.events.append("passages")
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def _recording_lock(events: list[str]):
    @contextmanager
    def lock(_path: Path):
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    return lock


def test_locked_text_embedder_guards_every_supported_provider_call(tmp_path: Path) -> None:
    """Mutation proof: direct delegation fails the exact lock ordering assertions.

    A query is the one call left outside the lock, since 2026-09-26; see
    ``test_a_query_does_not_wait_behind_a_held_passage_lock``.
    """
    events: list[str] = []
    wrapped = LockedEmbedder(
        _TextEmbedder(events), tmp_path / "embed.lock", lock_factory=_recording_lock(events)
    )

    assert wrapped.dim == 3
    assert wrapped.name == "test-text"
    assert wrapped.profile == "test-profile"
    wrapped.embed(["a"])
    wrapped.embed_query("q")
    wrapped.embed_passages(["p"])
    wrapped.embed_document_groups([["g"]])

    assert events == [
        "lock-enter",
        "embed",
        "lock-exit",
        "query",
        "lock-enter",
        "passages",
        "lock-exit",
        "lock-enter",
        "groups",
        "lock-exit",
    ]


def test_locked_multimodal_embedder_guards_document_and_query_calls(tmp_path: Path) -> None:
    """Mutation proof: omitting the document wrapper call loses its lock event pair.

    The query is outside the lock since 2026-09-26, as for text queries.
    """
    events: list[str] = []
    wrapped = LockedMultimodalEmbedder(
        _MultimodalEmbedder(events),
        tmp_path / "embed.lock",
        lock_factory=_recording_lock(events),
    )

    wrapped.embed_documents([{"content": []}])
    wrapped.embed_query("q")

    assert events == [
        "lock-enter",
        "documents",
        "lock-exit",
        "multimodal-query",
    ]


def _held_lock(events: list[str]):
    """A provider lock that some other caller (an Add's passage batch) is holding.

    The factory waits briefly and records that it had to, then proceeds, so a caller that takes
    the lock is seen waiting rather than failing with a lock error.
    """
    held = threading.Lock()
    held.acquire()

    @contextmanager
    def lock(_path: Path):
        if held.acquire(timeout=0.2):
            held.release()
        else:
            events.append("waited-on-held-lock")
        yield

    return lock


def test_a_query_does_not_wait_behind_a_held_passage_lock(tmp_path: Path) -> None:
    """A Search's query embedding never queues behind an Add's passage embedding.

    Red proof, 2026-09-26: wrapping the body of ``LockedEmbedder.embed_query`` in
    ``recall_aml/embedding_lock.py`` in ``with self._lock_factory(self._path):`` again (the code
    before this change) failed ``assert events == ["query"]`` with
    ``['waited-on-held-lock', 'query']``. The same mutation of
    ``LockedMultimodalEmbedder.embed_query`` failed the multimodal assertion the same way.
    """
    events: list[str] = []
    text = LockedEmbedder(
        _TextEmbedder(events), tmp_path / "embed.lock", lock_factory=_held_lock(events)
    )

    assert text.embed_query("q") == [1.0, 0.0, 0.0]
    assert events == ["query"]
    text.embed_passages(["p"])
    assert events == ["query", "waited-on-held-lock", "passages"]

    events.clear()
    multimodal = LockedMultimodalEmbedder(
        _MultimodalEmbedder(events), tmp_path / "embed.lock", lock_factory=_held_lock(events)
    )

    assert multimodal.embed_query("q") == [0.0, 1.0, 0.0]
    assert events == ["multimodal-query"]


def test_cached_text_embedder_reuses_exact_passages_and_queries_across_process_instances(
    tmp_path: Path,
) -> None:
    """Exact Code4 vectors are reused while query and passage key spaces stay isolated.

    Red proof receipt ``aml-hosted-embedding-cache-01`` targets
    ``CachedEmbedder.embed_passages``. Replacing its cached call with direct delegation made the
    second provider record ``passages`` and failed the final event assertion.
    """
    first_events: list[str] = []
    second_events: list[str] = []
    path = tmp_path / "shared.sqlite"
    first = CachedEmbedder(_IndependentTextEmbedder(first_events), path)
    second = CachedEmbedder(_IndependentTextEmbedder(second_events), path)

    assert first.embed_passages(["same"]) == second.embed_passages(["same"])
    assert first.embed_query("same") == second.embed_query("same")

    assert first_events == ["passages", "query"]
    assert second_events == []


def test_cached_context_embedder_binds_vectors_to_the_complete_document_group(
    tmp_path: Path,
) -> None:
    """Context4 reuse is allowed only when the whole ordered session is identical.

    Red proof receipt ``aml-hosted-embedding-cache-02`` targets
    ``CachedEmbedder.embed_document_groups``. Keying only on individual text made the reordered
    second group a false hit and failed the second provider event assertion.
    """
    first_events: list[str] = []
    second_events: list[str] = []
    path = tmp_path / "context.sqlite"
    first = CachedEmbedder(_TextEmbedder(first_events), path)
    second = CachedEmbedder(_TextEmbedder(second_events), path)

    first.embed_document_groups([["shared", "tail-a"]])
    second.embed_document_groups([["shared", "tail-a"]])
    second.embed_document_groups([["tail-a", "shared"]])

    assert first_events == ["groups"]
    assert second_events == ["groups"]


def test_cached_multimodal_embedder_reuses_documents_without_aliasing_queries(
    tmp_path: Path,
) -> None:
    """MM2 document and query vectors use distinct structured cache domains.

    Red proof receipt ``aml-hosted-embedding-cache-03`` targets
    ``CachedMultimodalEmbedder.embed_documents``. Direct provider delegation made the second
    instance record ``documents`` and failed the final event assertion.
    """
    first_events: list[str] = []
    second_events: list[str] = []
    path = tmp_path / "multimodal.sqlite"
    first = CachedMultimodalEmbedder(_MultimodalEmbedder(first_events), path)
    second = CachedMultimodalEmbedder(_MultimodalEmbedder(second_events), path)
    document = {"content": [{"type": "text", "text": "same"}]}

    assert first.embed_documents([document]) == second.embed_documents([document])
    assert first.embed_query("same") == second.embed_query("same")

    assert first_events == ["documents", "multimodal-query"]
    assert second_events == []


def test_production_wrapper_order_keeps_cache_hits_outside_provider_lock(
    monkeypatch, tmp_path: Path
) -> None:
    """The cache must be outermost so only provider misses acquire the cross process lock.

    Red proof receipt ``aml-hosted-embedding-cache-04`` targets the wrapper construction in
    ``_resolve_hosted_embedders``. Reversing the two wrapper assignments makes the recorded
    order ``cache, lock`` and fails before any provider call is made.
    """
    from recall_aml import __main__ as hosted_main

    events: list[tuple[str, str]] = []

    @contextmanager
    def no_lock(_path):
        yield

    def resolve(_profile: str, _source):
        return _TextEmbedder([])

    def cached(inner, _path):
        events.append(("cache", type(inner).__name__))
        return ("cached", inner)

    def locked(inner, _path):
        events.append(("lock", type(inner).__name__))
        return ("locked", inner)

    monkeypatch.setattr(hosted_main, "embedding_call_lock", no_lock)
    monkeypatch.setattr(hosted_main, "resolve_registered_embedder", resolve)
    monkeypatch.setattr(hosted_main, "CachedEmbedder", cached)
    monkeypatch.setattr(hosted_main, "LockedEmbedder", locked)
    settings = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        variant_name="A0_raw",
        voyage_api_key="voyage-key",
        embedding_lock_path=tmp_path / "embed.lock",
        embedding_cache_path=tmp_path / "embed.sqlite",
    )

    hosted_main._resolve_hosted_embedders(settings, variant(settings.variant_name))

    assert events == [("lock", "_TextEmbedder"), ("cache", "tuple")]


def test_hosted_settings_loads_the_shared_embedding_lock(monkeypatch) -> None:
    """The VPS2 path must reach startup construction and every wrapped live call."""
    monkeypatch.setenv("RECALL_AML_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("RECALL_AML_API_KEY", "secret")
    monkeypatch.setenv("RECALL_AML_GIT_COMMIT", "abc123")
    monkeypatch.setenv("RECALL_AML_AUTHORIZED_USER_ID", "embedding-test-user")
    monkeypatch.setenv("RECALL_AML_EMBED_LOCK_PATH", "/srv/locks/embed.lock")
    monkeypatch.setenv("RECALL_AML_EMBED_CACHE_PATH", "/srv/cache/embeddings.sqlite")

    settings = HostedSettings.from_env()

    assert settings.embedding_lock_path == Path("/srv/locks/embed.lock")
    assert settings.embedding_cache_path == Path("/srv/cache/embeddings.sqlite")


def test_hosted_environment_refuses_to_start_without_the_shared_embedding_lock(monkeypatch) -> None:
    """Hosted production must fail closed when the cross process lock is absent.

    Red proof receipt ``aml-hosted-embedding-lock-01`` targets the environment contract in
    ``HostedSettings.from_env``. Removing the required lock check lets a production configuration
    construct successfully with an unbounded provider call path.
    """
    monkeypatch.setenv("RECALL_AML_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("RECALL_AML_API_KEY", "secret")
    monkeypatch.setenv("RECALL_AML_GIT_COMMIT", "abc123")
    monkeypatch.delenv("RECALL_AML_EMBED_LOCK_PATH", raising=False)

    with pytest.raises(RuntimeError, match="RECALL_AML_EMBED_LOCK_PATH"):
        HostedSettings.from_env()


def test_embedder_constructor_probes_run_inside_the_shared_lock(
    monkeypatch, tmp_path: Path
) -> None:
    """Moving either resolver outside the constructor lock fails the event ordering."""
    from recall_aml import __main__ as hosted_main

    events: list[str] = []
    primary = _TextEmbedder(events)
    context = _TextEmbedder(events)

    @contextmanager
    def lock(_path: Path | None):
        events.append("constructor-lock-enter")
        try:
            yield
        finally:
            events.append("constructor-lock-exit")

    def resolve(profile: str, _source):
        events.append("resolve:" + profile)
        return context if profile == "voyage-context-4-v1" else primary

    monkeypatch.setattr(hosted_main, "embedding_call_lock", lock)
    monkeypatch.setattr(hosted_main, "resolve_registered_embedder", resolve)
    settings = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        variant_name="C7_routed_specialists",
        voyage_api_key="voyage-key",
        embedding_lock_path=tmp_path / "embed.lock",
    )

    hosted_main._resolve_hosted_embedders(settings, variant(settings.variant_name))

    assert events == [
        "constructor-lock-enter",
        "resolve:voyage-code-4-v1",
        "resolve:voyage-context-4-v1",
        "constructor-lock-exit",
    ]


def test_hosted_embedder_resolution_propagates_voyage_timeout(monkeypatch, tmp_path: Path) -> None:
    from recall_aml import __main__ as hosted_main

    captured: list[dict[str, str]] = []
    embedder = _TextEmbedder([])

    def resolve(_profile: str, source: dict[str, str]):
        captured.append(source)
        return embedder

    monkeypatch.setenv("RECALL_VOYAGE_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(hosted_main, "resolve_registered_embedder", resolve)
    settings = HostedSettings(
        "postgresql://unused",
        "secret",
        "abc123",
        variant_name="A0_raw",
        voyage_api_key="voyage-key",
    )

    hosted_main._resolve_hosted_embedders(settings, variant(settings.variant_name))

    assert captured == [
        {
            "VOYAGE_API_KEY": "voyage-key",
            "RECALL_VOYAGE_TIMEOUT_SECONDS": "12",
            "RECALL_VOYAGE_PARALLEL_REQUESTS": "4",
        }
    ]


def test_vps_setup_binds_a_validated_unit_and_shared_embedding_lock() -> None:
    """C6 and C7 must coexist while serializing all Voyage embedding traffic."""
    script = (
        Path(__file__).parents[1] / "scripts" / "aml_experience_vps2_setup.sh"
    ).read_text(encoding="utf-8")

    assert 'RECALL_AML_EXPERIMENT_UNIT:-recall-aml-experiment.service' in script
    assert '^recall-aml-[a-z0-9][a-z0-9-]*\\.service$' in script
    assert 'RECALL_AML_EMBED_LOCK_PATH=%s' in script
    assert '/home/sentiment/recall-repos/.locks/embed.lock' in script
    assert 'RECALL_AML_EMBED_CACHE_PATH=%s' in script
    assert '/home/sentiment/recall-repos/.cache/aml-hosted-embeddings.sqlite' in script
    assert 'systemctl --user restart "$service_unit"' in script


def test_c7_release_manifest_binds_embedding_lock_source(tmp_path: Path) -> None:
    """The immutable C7 receipt must cover the safety wrapper used in production."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = Path(__file__).parents[1]
    from scripts.aml_release_manifest import (
        BOUND_REPOSITORY_ARTIFACTS,
        SPECIALIST_PREREGISTRATION,
    )

    required = {
        *BOUND_REPOSITORY_ARTIFACTS.values(),
        SPECIALIST_PREREGISTRATION,
        Path("recall_aml/code4.py"),
        Path("recall/store.py"),
        Path("recall_aml/retrieval.py"),
        Path("recall_aml/service.py"),
        Path("recall_aml/specialists.py"),
        Path("recall_aml/storage.py"),
        Path("recall_aml/multimodal.py"),
        Path("recall_aml/embedding_lock.py"),
        Path("scripts/aml_c7_qualification.py"),
    }
    for relative in required:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
    wheel = repo / "dist/recall_rag.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"specialist wheel")

    manifest = build_manifest(
        repo_root=repo,
        wheel_path=wheel,
        commit="f" * 40,
        variant_name="C7_routed_specialists",
    )

    assert Path(str(manifest["artifacts"]["embedding_lock_source"]["path"])) == Path(
        "recall_aml/embedding_lock.py"
    )
