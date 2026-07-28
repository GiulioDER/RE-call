"""Every committed result artifact must say which configuration produced it.

`results/locomo_abstention.json` reads `calibrated: 0.5269`; `RESULTS.md` §7b publishes 0.574.
Both are correct — they are the pre- and post-#81/#84 configurations — and for a while nothing in
the filename said so. Worse, the marker was on the *post*-fix files (`postfix_`), so the absence of
a marker read as "the result" when it meant "the older one".

That is the same failure as a threshold that returns a plausible number on data that cannot support
it: nothing errors, and the reader gets a wrong answer with no signal. The fix is a `_provenance`
block inside each artifact, and these tests are what stop the next artifact from arriving without
one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"
GENERATIONS = {"pre-#81/#84", "post-#81/#84"}
STATUSES = {"current", "superseded", "not re-measured"}


def _artifacts() -> list[Path]:
    """Committed LOCOMO/cosine result artifacts — the ones a reader could mistake for each other.

    This is an allowlist of directories, not a sweep, so a NEW results subdirectory escapes the
    guard until someone adds it here — which is exactly what happened to `results/wrrf/`. Making it
    `RESULTS.rglob("*.json")` is the robust shape, but `results/gap/` (20 BEIR artifacts) predates
    the convention and would fail immediately; stamping that family is separate work.
    """
    paths = [
        *RESULTS.glob("locomo*.json"),
        *(RESULTS / "locomo").glob("*.json"),
        *(RESULTS / "locomo_rerank").glob("*.json"),
        *(RESULTS / "cosine").glob("*.json"),
        *(RESULTS / "wrrf").glob("*.json"),
        *(RESULTS / "beam_voyage").glob("*.json"),
    ]
    return sorted(p for p in paths if p.is_file())


def test_there_are_artifacts_to_check() -> None:
    assert len(_artifacts()) >= 12, "glob found too few artifacts; this suite would be vacuous"


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_artifact_declares_its_configuration(path: Path) -> None:
    prov = json.loads(path.read_text(encoding="utf-8")).get("_provenance")
    assert prov is not None, f"{path.name} has no _provenance block"
    assert prov["generation"] in GENERATIONS
    assert prov["status"] in STATUSES
    assert prov["note"].strip()
    assert prov["backs"], "an artifact nothing cites should be deleted, not kept unexplained"


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_superseded_artifacts_name_their_successor(path: Path) -> None:
    prov = json.loads(path.read_text(encoding="utf-8"))["_provenance"]
    successor = prov.get("superseded_by")
    if prov["status"] == "superseded":
        assert successor, f"{path.name} is superseded but names no successor"
        assert (RESULTS.parent / successor).is_file(), f"{path.name} -> missing {successor}"
    else:
        assert successor is None, (
            f"{path.name} is '{prov['status']}' but names a successor — a live artifact with a "
            f"successor is a contradiction the reader has to resolve"
        )


def test_the_pre_fix_artifacts_are_the_ones_this_repo_documents() -> None:
    """Pins the count FINDINGS §9a states, so the prose and the tree cannot drift again.

    §9a said "the two pre-fix artifacts this repo still holds" and stayed at two after #111
    committed three more. The number is now in a test.
    """
    pre = {
        p.name
        for p in _artifacts()
        if json.loads(p.read_text(encoding="utf-8"))["_provenance"]["generation"] == "pre-#81/#84"
    }
    assert pre == {
        "locomo_abstention.json",
        "locomo_entailment_sweep.json",
        "locomo_fastembed_k5.json",
        "depth_curve_pool20.json",
        "depth_curve_pool100.json",
    }


def test_provenance_did_not_displace_the_measurements() -> None:
    """A stamped artifact must still carry its own payload, not just its label."""
    for path in _artifacts():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) - {"_provenance"}, f"{path.name} has provenance and nothing else"


def test_artifacts_index_lists_every_committed_artifact() -> None:
    """`ARTIFACTS.md` is the map; an artifact missing from it is invisible to a reader."""
    index = (RESULTS / "ARTIFACTS.md").read_text(encoding="utf-8")
    for path in _artifacts():
        assert path.name in index, f"{path.name} is committed but absent from ARTIFACTS.md"
