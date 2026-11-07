"""fastembed's models live in the platform cache directory, not the temp directory.

fastembed's own default is `<temp>/fastembed_cache`. Found by the 0.15.0 Windows suite run: Windows
cleanup had emptied that directory but left the download metadata, so fastembed trusted the
metadata and failed every load with NO_SUCHFILE instead of downloading again. 18 real-model tests
failed until the entry was deleted by hand, and a Windows user hits the same broken cache.

The loader is stubbed because `fastembed` is an optional extra that CI's `test` job does not
install, and a test that skipped there would prove nothing.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from recall.embedding_providers.local import FastEmbedEmbedder, default_fastembed_cache_dir


class _StubTextEmbedding:
    instances: list["_StubTextEmbedding"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        _StubTextEmbedding.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] + [0.0] * 383 for _ in texts]

    query_embed = embed
    passage_embed = embed


@pytest.fixture
def stub_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_StubTextEmbedding]:
    _StubTextEmbedding.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _StubTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return _StubTextEmbedding


def _platform_env(tmp_path: Path) -> dict[str, str]:
    """An environment naming this platform's cache root, and nothing else."""
    key = "LOCALAPPDATA" if os.name == "nt" else "XDG_CACHE_HOME"
    return {key: str(tmp_path / "cache-root")}


def test_the_default_is_the_platform_cache_directory(tmp_path: Path) -> None:
    assert default_fastembed_cache_dir(_platform_env(tmp_path)) == (
        tmp_path / "cache-root" / "recall" / "fastembed"
    )


def test_fastembed_cache_path_still_wins(tmp_path: Path) -> None:
    env = {**_platform_env(tmp_path), "FASTEMBED_CACHE_PATH": str(tmp_path / "chosen")}
    assert default_fastembed_cache_dir(env) == tmp_path / "chosen"


def test_an_unpinned_model_is_loaded_from_the_platform_cache(
    stub_fastembed, monkeypatch, tmp_path: Path
) -> None:
    """Red proof: removed the `default_fastembed_cache_dir` branch in `FastEmbedEmbedder.__init__`
    (so no `cache_dir` was passed, fastembed's temp default); this failed at the `cache_dir`
    assertion (it was None)."""
    for key, value in _platform_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    FastEmbedEmbedder()
    kwargs = stub_fastembed.instances[-1].kwargs
    assert kwargs.get("cache_dir") == str(tmp_path / "cache-root" / "recall" / "fastembed")


def test_the_late_interaction_reranker_uses_the_same_cache(monkeypatch, tmp_path: Path) -> None:
    """Red proof: removed the `cache_dir is None` default in `LateInteractionReranker.from_pretrained`;
    this failed at the `cache_dir` assertion (it was None)."""
    from recall.rerank import LateInteractionReranker

    seen: dict[str, object] = {}

    class _StubLateInteraction:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    module = types.ModuleType("fastembed")
    module.LateInteractionTextEmbedding = _StubLateInteraction  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    for key, value in _platform_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    LateInteractionReranker.from_pretrained()
    assert seen.get("cache_dir") == str(tmp_path / "cache-root" / "recall" / "fastembed")


def test_an_explicit_env_does_not_hide_the_processs_cache_choice(
    stub_fastembed, monkeypatch, tmp_path: Path
) -> None:
    """fastembed reads `FASTEMBED_CACHE_PATH` from the process environment, and a server builds the
    embedder from a narrower mapping. Red proof: passed `self._env` to
    `default_fastembed_cache_dir` (the first version of this change); this failed at the
    `cache_dir` assertion (it named the platform cache, not the chosen one)."""
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "chosen"))
    FastEmbedEmbedder(env={})
    assert stub_fastembed.instances[-1].kwargs.get("cache_dir") == str(tmp_path / "chosen")
