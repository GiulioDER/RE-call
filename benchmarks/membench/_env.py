"""Retrieval configuration for both mem-bench adapters, built from environment variables.

Prior-work status for this whole package is recorded once in `__init__.py`: relocation of existing
adapters, not a new experiment, and `docs_search` was unavailable (VPS2 down).

Shared rather than copied. Each adapter used to construct its own embedder, reranker and k values
from the same twelve lines, which is precisely the hazard their own comments warn about one level
down ("a second class would be a second thing that could differ"): two arms that are supposed to be
identical except for the embedder must not have two places where that could stop being true.

**Env-selected, not subclassed, for the same reason.** The default arm and the tuned arm run this
same function; the only difference between them is what the environment says. A separate "tuned"
class would be a second code path, and a second code path is a second thing to explain when the
numbers differ.

| variable | default | meaning |
|---|---|---|
| `MEMBENCH_EMBEDDER` | fastembed default | `voyage:<model>` for Voyage, otherwise a fastembed name |
| `MEMBENCH_RERANKER` | none | `voyage:<model>`; anything else is ignored |
| `MEMBENCH_K` | 5 | hits returned |
| `MEMBENCH_CANDIDATE_K` | 20 | pool reranked down to `k` |
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from recall.embeddings import FastEmbedEmbedder, VoyageEmbedder

_VOYAGE = "voyage:"


@dataclass(frozen=True)
class RetrievalConfig:
    embedder: object
    reranker: object | None
    k: int
    candidate_k: int


def _build_embedder():
    spec = os.environ.get("MEMBENCH_EMBEDDER", "")
    if spec.startswith(_VOYAGE):
        return VoyageEmbedder(spec[len(_VOYAGE):])
    return FastEmbedEmbedder(spec) if spec else FastEmbedEmbedder()


def _build_reranker():
    """`None` unless `MEMBENCH_RERANKER` names a Voyage model.

    Imported lazily and from `benchmarks.voyage_rerank`, the copy that is ON MASTER. The previous
    version of this did `sys.path.insert(0, "/opt/recall-ladder-v4/benchmarks/ladder/systems")` --
    an absolute path on a host that no longer exists, pointing at a branch-only duplicate of the
    same file. A tuned re-run would have died on the import before scoring anything.
    """
    spec = os.environ.get("MEMBENCH_RERANKER", "")
    if not spec.startswith(_VOYAGE):
        return None
    from benchmarks.voyage_rerank import VoyageReranker

    return VoyageReranker(spec[len(_VOYAGE):])


def from_env() -> RetrievalConfig:
    return RetrievalConfig(
        embedder=_build_embedder(),
        reranker=_build_reranker(),
        k=int(os.environ.get("MEMBENCH_K", "5")),
        candidate_k=int(os.environ.get("MEMBENCH_CANDIDATE_K", "20")),
    )


def system_version() -> str:
    """The version of the `recall` package ABOUT TO RUN, read live.

    Read at call time, never stored as a constant: a constant would be a second place the version is
    written down, and it would go stale exactly like the `--system-version 0.6.0` that mem-bench's
    `sysversion` contract now exists to stop. RE-call's own `test_smoke.py` pins
    `recall.__version__` equal to the declared project version, so this is the same string a
    reproducer gets from `pip show`.
    """
    import recall

    return recall.__version__
