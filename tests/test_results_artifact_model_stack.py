"""Every result artifact must say which MODEL STACK produced it, or admit it does not know.

`_provenance` records which CODE produced an artifact — the `#81`/`#84` generation. For anything
that passes through a model that is not enough, and `results/locomo/postfix_abstention.json` is the
proof: it publishes four abstention modes, two of them route through a QNLI cross-encoder, and it
names no `sentence-transformers`, `transformers` or `torch` version at all.

Re-running it on a corpus asserted clean by row count moved the `entail` row by **0.0525** while
the calibrated thresholds — fit directly on the distribution a doubled corpus would move —
reproduced *exactly*. So the corpus was not the variable, and nothing recorded in the artifact says
what was. Two of its four published rows cannot be reproduced by anyone, on any machine, and the
file gives no signal that this is so.

The sentinel is deliberate. Artifacts produced before this convention genuinely do not know their
stack, and inventing plausible versions for them would be worse than the gap: it would make an
unreproducible row look reproducible. So they carry `"unrecorded"`, and the set allowed to carry it
is **pinned by name** below — a new artifact cannot join them by omission.

Note what is already pinned and is therefore NOT the gap: `recall.entailment.DEFAULT_QNLI_REVISION`
fixes the judge's Hub commit, so the weights are immutable. `git log -S` puts that pin at 2026-07-18,
before the artifact above. The gap is the stack that runs the model, not the model.

`generated_at` is checked here too, and shares the sentinel and the grandfather list rather than
getting a second copy of both. It answers the companion question, and the one that cost the most to
answer without it: deciding whether that artifact predated the double-index guard meant reading git
for the commit that ADDED it — and a commit date is when someone committed, not when the run
happened. For `3ee36ed` those differ by an unknown amount, which is the gap that let a 07-26 run and
a 07-28 guard pass each other. Two facts identify a run: which stack, and when.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from recall.eval.provenance import TRACKED, model_stack

RESULTS = Path(__file__).resolve().parent.parent / "results"
UNRECORDED = "unrecorded"

#: Artifacts that predate the convention. They may declare `"unrecorded"`; nothing else may.
#: Shrinks when one is re-run and re-stamped. It must never grow.
PREDATES_THE_CONVENTION = {
    "arm_A_rrf_pool20.json",
    "arm_C_rrf_pool100.json",
    "baseline.json",
    "baseline_verified.json",
    "depth_curve_pool100.json",
    "depth_curve_pool20.json",
    "distributions.json",
    "doubled_pool100.json",
    "doubled_pool20.json",
    "ksweep.json",
    "locomo_abstention.json",
    "locomo_entailment_sweep.json",
    "locomo_fastembed_k5.json",
    "postfix_abstention.json",
    "postfix_pool20.json",
    "rerank_modern.json",
    "rerank_shipped.json",
}


def _artifacts() -> list[Path]:
    paths = [
        *RESULTS.glob("locomo*.json"),
        *(RESULTS / "locomo").glob("*.json"),
        *(RESULTS / "locomo_rerank").glob("*.json"),
        *(RESULTS / "cosine").glob("*.json"),
        *(RESULTS / "wrrf").glob("*.json"),
        *(RESULTS / "beam_voyage").glob("*.json"),
    ]
    return sorted(p for p in paths if p.is_file())


def _declared_stack(payload: dict) -> object | None:
    """The stack, from either place it can live.

    Runners emit it top-level alongside `elapsed_s`, because it is runtime fact. The
    hand-written sentinel sits inside `_provenance` with the other status fields.
    """
    if "stack" in payload:
        return payload["stack"]
    return payload.get("_provenance", {}).get("stack")


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_artifact_declares_its_model_stack(path: Path) -> None:
    declared = _declared_stack(json.loads(path.read_text(encoding="utf-8")))
    assert declared is not None, (
        f"{path.name} declares no model stack. Every number here passes through an embedder, a "
        f"cross-encoder, or both, and their versions decide it. Emit `model_stack()` from the "
        f"runner, or record {UNRECORDED!r} if the run genuinely predates this."
    )


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_only_grandfathered_artifacts_may_say_unrecorded(path: Path) -> None:
    declared = _declared_stack(json.loads(path.read_text(encoding="utf-8")))
    if declared != UNRECORDED:
        return
    assert path.name in PREDATES_THE_CONVENTION, (
        f"{path.name} is new but declares its stack {UNRECORDED!r}. A new run knows its own "
        f"versions — call `recall.eval.provenance.model_stack()`. The sentinel exists for runs "
        f"that cannot know, not as a way to skip recording."
    )


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_a_recorded_stack_names_versions(path: Path) -> None:
    declared = _declared_stack(json.loads(path.read_text(encoding="utf-8")))
    if declared == UNRECORDED or declared is None:
        return
    assert isinstance(declared, dict) and declared, f"{path.name}: stack is not a version mapping"
    unknown = set(declared) - set(TRACKED)
    assert not unknown, f"{path.name} records untracked packages {sorted(unknown)}"
    for name, ver in declared.items():
        assert isinstance(ver, str) and ver, f"{path.name}: {name} has no version string"


def _declared_time(payload: dict) -> object | None:
    if "generated_at" in payload:
        return payload["generated_at"]
    return payload.get("_provenance", {}).get("generated_at")


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_artifact_declares_when_it_ran(path: Path) -> None:
    declared = _declared_time(json.loads(path.read_text(encoding="utf-8")))
    assert declared is not None, (
        f"{path.name} does not say when it ran. Without it, 'did this predate guard X?' has to be "
        f"answered from git — which records when a file was COMMITTED, not when the run happened. "
        f"Emit `generated_at()` from the runner, or record {UNRECORDED!r}."
    )


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_only_grandfathered_artifacts_may_not_know_when_they_ran(path: Path) -> None:
    declared = _declared_time(json.loads(path.read_text(encoding="utf-8")))
    if declared != UNRECORDED:
        return
    assert path.name in PREDATES_THE_CONVENTION, (
        f"{path.name} is new but does not say when it ran. A run knows its own clock — call "
        f"`recall.eval.provenance.generated_at()`."
    )


@pytest.mark.parametrize("path", _artifacts(), ids=lambda p: p.name)
def test_a_recorded_time_is_an_utc_timestamp(path: Path) -> None:
    """A date nobody can parse is the same as no date, and reads as if it were one."""
    declared = _declared_time(json.loads(path.read_text(encoding="utf-8")))
    if declared == UNRECORDED or declared is None:
        return
    assert isinstance(declared, str), f"{path.name}: generated_at is not a string"
    parsed = datetime.fromisoformat(declared)  # raises on anything non-ISO-8601
    assert parsed.tzinfo is not None, (
        f"{path.name}: {declared!r} has no timezone. A naive stamp cannot be compared against a "
        f"commit date without guessing which machine produced it."
    )


def test_the_grandfather_list_matches_the_tree() -> None:
    """A name left here after its artifact is gone silently re-permits the sentinel."""
    stale = PREDATES_THE_CONVENTION - {p.name for p in _artifacts()}
    assert not stale, f"grandfathered names with no artifact: {sorted(stale)}"


def test_model_stack_resolves_installed_and_skips_absent() -> None:
    """Behaviour of the helper, not a claim about what happens to be installed.

    An earlier version asserted `model_stack()` was non-empty. That passed locally, where the
    torch stack is present, and failed in CI, which installs `.[dev]` and none of the tracked
    packages — so it was testing the environment and calling it a code test. `pytest` is installed
    wherever this runs, by construction.
    """
    assert model_stack(("pytest",)).get("pytest"), "an installed package must resolve"
    assert model_stack(("recall-no-such-package",)) == {}, "an absent package must be omitted"


def test_tracked_is_not_silently_empty() -> None:
    """`TRACKED` going empty would make every stack `{}` and every check above vacuous."""
    assert TRACKED and all(isinstance(name, str) and name for name in TRACKED)
