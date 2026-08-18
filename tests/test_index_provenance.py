"""Every chunk must say which project produced it, and against which commit.

The memory and code corpora are about to hold material from several repositories at once. A hit
that cannot say where it came from is a hit a reader has to go and verify by hand, which is the
cost retrieval was supposed to remove. Two fields carry that:

``project``
    The name the operator gives the run. Not derived from the directory, because a worktree's
    directory name is not its project any more than it is its branch.

``indexed_commit``
    The repository's HEAD at index time. This is the field that makes staleness *detectable*
    rather than merely suspected: without it, a chunk indexed from code that has since been
    rewritten is indistinguishable from a current one, and it cannot be reconstructed afterwards.

Frontmatter cannot carry either. `Indexer` builds chunk metadata from `document.meta`, which is the
document's own frontmatter, and **code files have none** — which is exactly the corpus this is for.
"""

from __future__ import annotations

import pytest

from recall.index import Indexer


class _Embedder:
    dim = 8
    name = "test-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


def test_indexer_accepts_provenance() -> None:
    """The constructor must take it. Everything else depends on this."""
    idx = Indexer(
        store=None,  # type: ignore[arg-type]
        embedder=_Embedder(),
        project="recall",
        indexed_commit="deadbee",
    )
    assert idx.provenance == {"project": "recall", "indexed_commit": "deadbee"}


def test_provenance_defaults_to_empty_not_to_a_guess() -> None:
    """Absent provenance must be ABSENT, never inferred from the path.

    A guessed project name is worse than a missing one: it reads as authoritative and is wrong for
    every worktree, whose directory name is not its project.
    """
    idx = Indexer(store=None, embedder=_Embedder())  # type: ignore[arg-type]
    assert idx.provenance == {}


def test_a_partial_provenance_is_kept_partial() -> None:
    """Recording the project without a commit is legitimate; a corpus may not be a git repo."""
    idx = Indexer(store=None, embedder=_Embedder(), project="notes")  # type: ignore[arg-type]
    assert idx.provenance == {"project": "notes"}


@pytest.mark.parametrize("field", ["project", "indexed_commit"])
def test_frontmatter_cannot_override_provenance(field: str) -> None:
    """A document must not be able to relabel its own origin.

    Chunk metadata merges `**meta` from frontmatter. If frontmatter won, any indexed file could
    claim to come from another project, and provenance would be an assertion by the data rather
    than a record by the indexer. Provenance is applied AFTER, so the indexer's value stands.
    """
    idx = Indexer(
        store=None,  # type: ignore[arg-type]
        embedder=_Embedder(),
        project="recall",
        indexed_commit="deadbee",
    )
    frontmatter = {field: "impostor", "other": "kept"}
    merged = idx.apply_provenance(frontmatter)
    assert merged[field] == idx.provenance[field], "frontmatter overrode the indexer's provenance"
    assert merged["other"] == "kept", "provenance clobbered unrelated frontmatter"


def test_head_commit_is_read_from_the_repo_not_invented() -> None:
    """The helper returns a real short SHA in a repo, and None outside one.

    None rather than a placeholder: "unknown" stored as a value is indistinguishable, later, from a
    commit that happened to be named that, and it makes the absent case look recorded.
    """
    import pathlib

    from recall.index import head_commit

    here = pathlib.Path(__file__).resolve().parent
    sha = head_commit(here)
    assert sha and len(sha) >= 7 and all(c in "0123456789abcdef" for c in sha)
    assert head_commit(pathlib.Path(pathlib.Path(here.anchor))) is None
