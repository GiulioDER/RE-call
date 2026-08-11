"""Census of supersession evidence in `python/peps`. No model, no human judgement.

This runs before any arm and is arm-independent, so it belongs in the preregistration's
`## Already measured` section rather than among its predictions.

The number it exists to publish is `n_restated_in_prose / n_header_edges`: the fraction of
authored header edges that the body ALSO states in prose. That ratio is the hard ceiling on
recall for any extractor that reads prose, because an edge no sentence states cannot be found by
reading sentences. Measured at `5981b2a`: **8 of 47, or 17.0%.** Publish it, or every recall
number below it reads as a model failure when it is a corpus fact.

`restates` requires the marker and the partner reference in ONE sentence. Whole-body
co-occurrence would report 26 of 47 instead, by pairing a marker in one section with a reference
in another — the loose matching `recall/fix.py` records as producing garbage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarks.labelling.truth_extraction.artifact_contract import validate_census
from benchmarks.labelling.truth_extraction.peps_header import (
    Edge,
    edges_from_fields,
    header_fields,
    restates,
    split_header,
)
from recall.eval.provenance import generated_at, model_stack
from recall.lint import CLOSURE_MARKERS


@dataclass(frozen=True)
class Census:
    n_files: int
    n_header_edges: int
    n_prose_marker_files: int
    n_marker_without_header: int
    n_restated_in_prose: int
    edges: tuple[Edge, ...]
    restatements: dict[str, str]
    marker_without_header: tuple[str, ...]
    file_digests: dict[str, str]


def compute_census(peps_dir: Path) -> Census:
    """Count everything the labelled set is built from. Reads files; decides nothing."""
    files = sorted(peps_dir.glob("pep-*.rst"))
    if not files:
        raise SystemExit(
            f"no pep-*.rst under {peps_dir} — pass the nested 'peps/' directory of the clone, "
            f"not the repository root"
        )

    bodies: dict[str, str] = {}
    digests: dict[str, str] = {}
    edges: set[Edge] = set()
    marker_files: list[str] = []

    for path in files:
        raw = path.read_bytes()
        digests[path.name] = hashlib.sha256(raw).hexdigest()
        head, body = split_header(raw.decode("utf-8", errors="replace"))
        bodies[path.stem] = body
        edges |= edges_from_fields(path.stem, header_fields(head))
        if CLOSURE_MARKERS.search(body):
            marker_files.append(path.stem)

    # An edge is restated if EITHER end states it. The successor's body saying "replaces PEP 386"
    # is as much a prose statement of the relation as the predecessor's "superseded by PEP 440".
    restatements: dict[str, str] = {}
    for edge in sorted(edges):
        for holder, partner in (
            (edge.superseded, edge.successor),
            (edge.successor, edge.superseded),
        ):
            sentence = restates(bodies.get(holder, ""), partner)
            if sentence:
                restatements[f"{edge.superseded}->{edge.successor}"] = sentence.strip()
                break

    in_an_edge = {e.superseded for e in edges} | {e.successor for e in edges}
    without_header = tuple(sorted(s for s in marker_files if s not in in_an_edge))

    return Census(
        n_files=len(files),
        n_header_edges=len(edges),
        n_prose_marker_files=len(marker_files),
        n_marker_without_header=len(without_header),
        n_restated_in_prose=len(restatements),
        edges=tuple(sorted(edges)),
        restatements=restatements,
        marker_without_header=without_header,
        file_digests=digests,
    )


def census_payload(
    census: Census,
    *,
    peps_sha: str,
    clone_date: str,
    recall_commit: str,
    invocation: str,
) -> dict:
    """The committed artifact: counts, bodies, and the provenance that makes them checkable."""
    ceiling = census.n_restated_in_prose / census.n_header_edges if census.n_header_edges else 0.0
    return {
        "n_files": census.n_files,
        "n_header_edges": census.n_header_edges,
        "n_prose_marker_files": census.n_prose_marker_files,
        "n_marker_without_header": census.n_marker_without_header,
        "n_restated_in_prose": census.n_restated_in_prose,
        "recall_ceiling": round(ceiling, 4),
        "edges": [{"superseded": e.superseded, "successor": e.successor} for e in census.edges],
        "restatements": census.restatements,
        "marker_without_header": list(census.marker_without_header),
        "file_digests": census.file_digests,
        "_provenance": {
            "peps_sha": peps_sha,
            "clone_date": clone_date,
            "recall_commit": recall_commit,
            "generated_at": generated_at(),
            "model_stack": model_stack(),
            "invocation": invocation,
            "note": (
                "Arm-independent: no model and no human judgement. recall_ceiling is the "
                "fraction of authored header edges also stated in prose, and is the hard upper "
                "bound on recall for any prose extractor."
            ),
        },
    }


def write_census(path: Path, payload: Mapping[str, object]) -> None:
    """Validate, then write. A payload that fails validation leaves no file behind."""
    validate_census(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peps-dir", type=Path, required=True,
                        help="the nested peps/ directory of a python/peps clone")
    parser.add_argument("--peps-sha", required=True, help="git rev-parse HEAD of that clone")
    parser.add_argument("--clone-date", required=True, help="ISO date the clone was taken")
    parser.add_argument("--out", type=Path, default=Path("results/truth_extraction/census.json"))
    args = parser.parse_args()

    recall_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    census = compute_census(args.peps_dir)
    payload = census_payload(
        census,
        peps_sha=args.peps_sha,
        clone_date=args.clone_date,
        recall_commit=recall_commit,
        invocation=" ".join(["python", "-m", "benchmarks.labelling.truth_extraction.census",
                             *sys.argv[1:]]),
    )
    write_census(args.out, payload)
    print(f"{args.out}")
    print(f"  n_files                 {census.n_files}")
    print(f"  n_header_edges          {census.n_header_edges}")
    print(f"  n_prose_marker_files    {census.n_prose_marker_files}")
    print(f"  n_marker_without_header {census.n_marker_without_header}")
    print(f"  n_restated_in_prose     {census.n_restated_in_prose}"
          f"  <- recall ceiling {payload['recall_ceiling']:.1%}")


if __name__ == "__main__":
    main()
