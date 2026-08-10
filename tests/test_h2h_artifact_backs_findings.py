"""§9d's table and its artifact must agree, cell by cell.

The defect this exists for is not a wrong number, it is a *missing mechanism*. §9d was the
only published table in this repository with no committed artifact: `results/ARTIFACTS.md`
had no entry for it, and no path under `results/` named the comparison. Nothing could have
checked it, so nothing did, and row 4's `paired p` sat at `0.00018` for two weeks when the
value is `0.00017` — row 2's p, copied down a row. It was found by the first run of
`benchmarks/h2h_artifact.py`, i.e. the first moment a check became possible.

Rebuilding the artifact needs the ~123 MB of raw runs, which are deliberately not in the
repository, so this cannot re-derive anything. What it *can* do, entirely from committed
files, is assert the two committed representations of the same result agree — and that is
exactly the failure that occurred. A transcription error has to survive two places now.

When the archived outcome vectors are restored locally, it also asserts they reconstruct each row's
2x2 table. The source tree keeps the compact artifact and vector declaration; the raw vectors live
outside git.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.claim_gate import matches

_ROOT = Path(__file__).resolve().parent.parent
_FINDINGS = _ROOT / "results" / "FINDINGS.md"
_ARTIFACT = _ROOT / "results" / "head_to_head" / "paired_accuracy.json"
_VECTORS = _ROOT / "results" / "head_to_head" / "outcomes"

#: A §9d body row: | generator | budget | judge | **RE-call** | Mem0 | p |
#:
#: Each numeric cell carries a claim-gate marker (`<!--@ artifact.json#key -->`), so the pattern
#: has to step over one. The marker is OPTIONAL here: this test asserts the table agrees with the
#: artifact, and `test_published_numbers_have_artifacts` is what asserts the markers exist.
#: Requiring them in both would make one defect fail two guards for the same reason.
_MARKER = r"(?:\s*<!--@[^>]*-->)?"
_ROW = re.compile(
    r"^\|\s*(gpt-4o(?:-mini)?)\s*\|\s*([^|<]+?)\s*\|\s*(gpt-4o(?:-mini)?)\s*\|\s*"
    rf"\*\*([\d.]+)\*\*{_MARKER}\s*\|\s*([\d.]+){_MARKER}\s*\|\s*([\d.]+){_MARKER}\s*\|\s*$",
    re.MULTILINE,
)


def _table_rows() -> list[tuple[str, str, str, str, str, str]]:
    """The five §9d rows, parsed out of the section rather than out of the whole file.

    Scoped to the section: FINDINGS.md is ~1,800 lines and other tables carry the same
    column shape, so an unscoped match would silently gather rows from a different result
    and compare them against this artifact.
    """
    text = _FINDINGS.read_text(encoding="utf-8")
    start = text.index("### 9d.")
    end = text.index("\n### ", start + 1)
    return _ROW.findall(text[start:end])


def _rounds_to(derived: float, published: str) -> bool:
    """The repo's own published-number rule, not a second one written here.

    `claim_gate.matches` is what gates every number in FINDINGS.md already; asserting these
    cells under a different rounding convention would let a cell pass one check and fail the
    other, which is worse than either check alone.
    """
    return matches(published, derived)


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert _ARTIFACT.exists(), (
        f"{_ARTIFACT.relative_to(_ROOT)} is missing. §9d publishes a five-row table; without "
        "this artifact nothing in the repository can check it, which is how the row-4 p-value "
        "error survived. Rebuild with benchmarks/h2h_artifact.py --runs <raw>."
    )
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


def test_the_table_and_the_artifact_have_the_same_rows(artifact: dict) -> None:
    rows = _table_rows()
    assert len(rows) == 5, f"expected §9d's five rows, parsed {len(rows)}"
    assert len(artifact["rows"]) == len(rows)


@pytest.mark.parametrize("index", range(5))
def test_each_published_cell_matches_the_artifact(artifact: dict, index: int) -> None:
    generator, budget, judge, recall_pub, mem0_pub, p_pub = _table_rows()[index]
    row = list(artifact["rows"].values())[index]

    assert row["generator"] == generator, f"row {index}: generator moved"
    assert row["judge"] == judge, f"row {index}: judge moved"
    assert budget.split()[0] in row["budget"], f"row {index}: budget moved"

    assert _rounds_to(row["recall"]["accuracy"], recall_pub), (
        f"row {index} ({row['row_id']}): FINDINGS §9d says RE-call {recall_pub}, artifact has "
        f"{row['recall']['accuracy']:.6f}"
    )
    assert _rounds_to(row["mem0"]["accuracy"], mem0_pub), (
        f"row {index} ({row['row_id']}): FINDINGS §9d says Mem0 {mem0_pub}, artifact has "
        f"{row['mem0']['accuracy']:.6f}"
    )
    assert _rounds_to(row["mcnemar"]["p_value"], p_pub), (
        f"row {index} ({row['row_id']}): FINDINGS §9d says p {p_pub}, artifact has "
        f"{row['mcnemar']['p_value']:.6e}. This is the assertion that catches a p copied "
        f"from an adjacent row."
    )


@pytest.mark.parametrize("index", range(5))
def test_the_outcome_vectors_reconstruct_the_reported_2x2(artifact: dict, index: int) -> None:
    """The archived vectors must reproduce the counts when they are restored locally."""
    row = list(artifact["rows"].values())[index]
    vector = _VECTORS / f"{row['row_id']}.jsonl"
    if not vector.exists():
        pytest.skip("head-to-head outcome vectors are archived outside the source tree")

    both = recall_only = mem0_only = neither = 0
    for line in vector.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        r, m = bool(record["recall"]), bool(record["mem0"])
        if r and m:
            both += 1
        elif r:
            recall_only += 1
        elif m:
            mem0_only += 1
        else:
            neither += 1

    test = row["mcnemar"]
    assert both + recall_only + mem0_only + neither == test["n_paired"]
    # a_only is McNemar's b cell (RE-call right, Mem0 wrong); b_only is c.
    assert recall_only == test["a_only"], f"{row['row_id']}: discordant b cell disagrees"
    assert mem0_only == test["b_only"], f"{row['row_id']}: discordant c cell disagrees"
    assert both == test["both_correct"]
    assert neither == test["both_wrong"]


def test_the_as_shipped_replicates_are_both_recorded(artifact: dict) -> None:
    """The README quoted the higher of two runs; both must stay visible.

    A single replicate here is not a smaller artifact, it is a different claim: the
    published margin range "+0.046 to +0.056" IS the two replicates, so dropping one turns
    a measured spread into an unexplained interval.
    """
    shipped = artifact["as_shipped_embedder"]
    replicates = shipped["recall_replicates"]
    assert len(replicates) == 2, "both as-shipped RE-call runs must be recorded"
    accuracies = sorted(r["accuracy"] for r in replicates)
    assert accuracies[1] - accuracies[0] > 0.005, (
        "the two replicates differed by ~1.04 points; if that spread has collapsed, either "
        "the runs changed or the same run was recorded twice"
    )
    margins = sorted(r["margin_over_mem0"] for r in replicates)
    assert _rounds_to(margins[0], "0.046") and _rounds_to(margins[1], "0.056"), (
        f"FINDINGS §9d publishes the as-shipped margin as +0.046 to +0.056; the artifact's "
        f"replicates give {margins[0]:.4f} to {margins[1]:.4f}"
    )


def test_the_outcomes_directory_holds_exactly_the_declared_vectors(artifact: dict) -> None:
    """No orphans, no gaps when the archived vector directory is restored.

    The first commit of this artifact shipped NINE vector files for seven declared rows. An
    earlier key derivation had written two of the as-shipped replicates under different
    names, and renaming the key left the originals behind. They were byte-valid, sat beside
    the live files, and backed nothing.

    That is the quieter half of the same defect the collision guard covers: a stale vector
    does not overwrite anything, it accumulates, and a reader has no way to tell which of
    two plausible files is the evidence. `write_artifact` now prunes; this asserts it.
    """
    declared = set(artifact["outcome_vectors"])
    on_disk = {p.stem for p in _VECTORS.glob("*.jsonl")}
    assert declared, "the artifact must declare its vectors, or this test is vacuous"
    if not on_disk:
        pytest.skip("head-to-head outcome vectors are archived outside the source tree")
    assert on_disk == declared, (
        f"outcomes/ and the artifact disagree.\n"
        f"  orphaned (on disk, not declared): {sorted(on_disk - declared)}\n"
        f"  missing  (declared, not on disk): {sorted(declared - on_disk)}"
    )
    # And the declared set really is the rows plus both replicates, not some subset that
    # happens to match a truncated directory.
    assert declared >= set(artifact["rows"]), "every table row needs its vector"
    assert sum(1 for k in declared if k.startswith("as_shipped__")) == 2
