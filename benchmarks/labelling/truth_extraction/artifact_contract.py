"""Validation for the truth-extraction census artifact, applied at the write site.

The census is the artifact every later recall number is read against. Its counts and its lists
are the same facts written twice — a summary a reader quotes and a body a reader recomputes from.
If they disagree, the artifact is not merely wrong, it is unfalsifiable: nothing in it says which
of the two is the typo. So the disagreement is refused at write time, when it costs nothing.

Pattern follows `benchmarks/artifact_contract.py`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

_REQUIRED_PROVENANCE = ("peps_sha", "clone_date", "recall_commit", "generated_at", "invocation")


def validate_census(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent, attributable census."""
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("census payload requires a _provenance block")
    for field in _REQUIRED_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"census _provenance requires {field}")

    edges = payload.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("census edges must be an array")
    if payload.get("n_header_edges") != len(edges):
        raise ValueError(
            f"n_header_edges {payload.get('n_header_edges')!r} disagrees with "
            f"{len(edges)} entries in edges"
        )

    restatements = payload.get("restatements")
    if not isinstance(restatements, Mapping):
        raise ValueError("census restatements must be an object")
    if payload.get("n_restated_in_prose") != len(restatements):
        raise ValueError(
            f"n_restated_in_prose {payload.get('n_restated_in_prose')!r} disagrees with "
            f"{len(restatements)} entries in restatements"
        )

    # The recall ceiling is a proportion of the header edges. A restated count that reaches or
    # passes the edge count means the restatement detector matched something that is not
    # distinctly inside the gold set — with zero edges and one restatement, or an equal count
    # squeezed down to a single shared edge, the ratio is at or past 100% either way.
    if len(restatements) >= len(edges) and len(restatements) > 0:
        raise ValueError(
            f"n_restated_in_prose ({len(restatements)}) cannot exceed n_header_edges "
            f"({len(edges)}) — the recall ceiling cannot exceed 100%"
        )


__all__ = ["validate_census"]
