"""Provenance must reach the GENERATION build, not only `recall index`.

`recall/generations.py` does not use `Indexer` at all: it has its own chunking and embedding loop.
So adding `--project` to `recall index` left the generation path unstamped, and the generation path
is the one calibration requires (`recall calibration calibrate --generation`). A corpus could
therefore be calibrated and served with no record of which project produced each chunk, which is the
exact thing provenance was added for.

This is the same partial-fix shape an audit caught earlier on this branch, where a policy reached
two of four call sites. Two entry points build corpora; both must stamp.
"""

from __future__ import annotations

import pytest

from recall.generations import with_provenance


def test_provenance_is_applied_over_frontmatter() -> None:
    """A document must not be able to relabel its own origin.

    Chunk metadata starts as the document's frontmatter. If frontmatter won, any indexed file could
    claim to come from another project, and provenance would be an assertion made by the data
    rather than a record made by the builder.
    """
    merged = with_provenance({"project": "impostor", "kept": 1}, {"project": "recall"})
    assert merged["project"] == "recall"
    assert merged["kept"] == 1


def test_empty_provenance_leaves_metadata_untouched() -> None:
    meta = {"a": 1, "b": 2}
    assert with_provenance(meta, {}) == meta


def test_it_does_not_mutate_the_input() -> None:
    """The caller reuses `metadata` across chunks of one document; mutating it would accumulate."""
    meta = {"a": 1}
    with_provenance(meta, {"project": "recall"})
    assert meta == {"a": 1}


def test_build_accepts_provenance() -> None:
    """The parameter has to exist on the real signature, or none of the above is reachable."""
    import inspect

    from recall.generations import GenerationManager

    params = inspect.signature(GenerationManager.build).parameters
    assert "provenance" in params, (
        "GenerationManager.build takes no provenance, so a calibrated generation carries no record "
        "of which project produced each chunk."
    )
    assert params["provenance"].default in (None, {}), "provenance must be optional"


@pytest.mark.parametrize("field", ["project", "indexed_commit"])
def test_both_fields_survive(field: str) -> None:
    merged = with_provenance({}, {"project": "recall", "indexed_commit": "abc1234"})
    assert field in merged
