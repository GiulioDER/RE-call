"""PERF001: original449aa0a1 re-embedded six of six warm-cache passages.

Hold complete synthetic identity fixed, compare distinct store builds, and
retain cold/disabled controls. Six notes stay below the calibration floor, so
query embeddings cannot contaminate passage work. No elapsed-time claim.
"""

from __future__ import annotations

from recall.embeddings import HashingEmbedder, EmbeddingProfile
from recall.lite.setup import run_lite_setup
from tests.test_lite_calibration import _write_memos
from tests.test_lite_setup_boundaries import isolated_lite_environment  # noqa: F401


class CountingHashing(HashingEmbedder):
    def __init__(self):
        super().__init__(dim=384)
        self.calls = []
        self.profile = EmbeddingProfile(
            profile_id="test-cache-hashing-v1",
            model_name=self.name,
            artifact_digest="1" * 64,
            dimension=self.dim,
            query_mode="legacy",
            passage_mode="legacy",
        )

    def embed(self, texts):
        self.calls.extend(texts)
        return super().embed(texts)


def test_warm_cache_reuses_passages_in_a_new_store(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    _write_memos(root / "notes", 6)
    embedder = CountingHashing()

    def build(filename):
        embedder.calls.clear()
        report = run_lite_setup(
            project_root=root,
            memory=root / "notes",
            embedder=embedder,
            embedder_name="hashing:384",
            store_path=root / filename,
            register=False,
        )
        assert report.chunks == 6, "invalid corpus control"
        return len(embedder.calls)

    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    disabled_first = build("off-a.db")
    disabled_second = build("off-b.db")
    assert disabled_first == disabled_second and disabled_first > 0, (
        "invalid disabled cache control"
    )
    monkeypatch.setenv("RECALL_EMBED_CACHE", str(tmp_path / "cache.sqlite"))
    cold = build("warm-a.db")
    warm = build("warm-b.db")
    assert cold == disabled_first, "invalid cold cache control"
    assert warm == 0, f"warm setup re-embedded cached passages: {warm} of {cold}"
