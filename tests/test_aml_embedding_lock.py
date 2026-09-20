"""Safety proofs for serialized hosted embedding calls."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from recall_aml.config import HostedSettings
from recall_aml.embedding_lock import LockedEmbedder, LockedMultimodalEmbedder
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
    """Mutation proof: direct delegation fails the exact lock ordering assertions."""
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
        "lock-enter",
        "query",
        "lock-exit",
        "lock-enter",
        "passages",
        "lock-exit",
        "lock-enter",
        "groups",
        "lock-exit",
    ]


def test_locked_multimodal_embedder_guards_document_and_query_calls(tmp_path: Path) -> None:
    """Mutation proof: omitting either wrapper call loses its lock event pair."""
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
        "lock-enter",
        "multimodal-query",
        "lock-exit",
    ]


def test_hosted_settings_loads_the_shared_embedding_lock(monkeypatch) -> None:
    """The VPS2 path must reach startup construction and every wrapped live call."""
    monkeypatch.setenv("RECALL_AML_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("RECALL_AML_API_KEY", "secret")
    monkeypatch.setenv("RECALL_AML_GIT_COMMIT", "abc123")
    monkeypatch.setenv("RECALL_AML_EMBED_LOCK_PATH", "/srv/locks/embed.lock")

    settings = HostedSettings.from_env()

    assert settings.embedding_lock_path == Path("/srv/locks/embed.lock")


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


def test_vps_setup_binds_a_validated_unit_and_shared_embedding_lock() -> None:
    """C6 and C7 must coexist while serializing all Voyage embedding traffic."""
    script = (
        Path(__file__).parents[1] / "scripts" / "aml_experience_vps2_setup.sh"
    ).read_text(encoding="utf-8")

    assert 'RECALL_AML_EXPERIMENT_UNIT:-recall-aml-experiment.service' in script
    assert '^recall-aml-[a-z0-9][a-z0-9-]*\\.service$' in script
    assert 'RECALL_AML_EMBED_LOCK_PATH=%s' in script
    assert '/home/sentiment/recall-repos/.locks/embed.lock' in script
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
