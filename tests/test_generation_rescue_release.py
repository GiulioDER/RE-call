"""Generation rescue cleanup and reuse at their normal builder entry point."""

from pathlib import Path

import pytest

from recall.generation_types import NoActiveGeneration
from recall.generation_rescue import build_artifact
from tests.test_atomic_rescue_default_on_cca import corpus  # noqa: F401
from tests.test_calibration_carry_forward import _CarryEmbedder


def test_collection_during_publication_removes_this_builds_artifact(corpus, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    from recall import generation_rescue, generation_store

    publish = generation_rescue.write_atomic_rescue_artifact
    select = generation_store.GenerationStore.set_fixed_generation
    collected = False

    def selected(store, generation_id):
        if collected:
            raise NoActiveGeneration("synthetic generation collection")
        select(store, generation_id)

    def publish_then_collect(*args, **kwargs):
        nonlocal collected
        result = publish(*args, **kwargs)
        collected = True
        return result

    monkeypatch.setattr(generation_store.GenerationStore, "set_fixed_generation", selected)
    monkeypatch.setattr(generation_rescue, "write_atomic_rescue_artifact", publish_then_collect)
    with pytest.raises(NoActiveGeneration, match="synthetic generation collection"):
        build_artifact("postgresql://x", "t", "gen_1", _CarryEmbedder(), tmp_path)
    assert not list(tmp_path.glob("gen_1.*/manifest.json")), "collection left a newly published orphan artifact"
