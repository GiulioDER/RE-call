"""The frozen injection baseline, and the proof that it can move.

`benchmarks/evidence_injection.py` runs 13 payloads through 3 carriers and reports how many left
the delimited data region. This file turns that number into a ratchet: a later session can show
the rate did not increase, which is the only way "we did not make it worse" is a claim rather than
an impression.

Three things have to hold together, and each is a separate test because each fails separately:

1. **The suite is frozen.** A digest over the payload, carrier and detector lists must match the
   artifact. Editing a payload out of the suite is otherwise indistinguishable from fixing it.
2. **The rate did not increase.** Escapes are compared against the recorded count, per carrier as
   well as in total, so an improvement in one arm cannot mask a regression in another.
3. **The detectors can fire.** Zero escapes and a broken detector set produce the same number.
   The positive control runs the identical detectors against the renderer this session replaced,
   which must be reported as escaping.

And a fourth, easy to forget: **the evidence must still be there.** A renderer that deleted all
corpus text would score a perfect zero, so the suite also records whether each payload survives a
JSON parse of the region it was placed in.
"""
from __future__ import annotations

import json

import pytest

from benchmarks.evidence_injection import (
    BASELINE_PATH,
    CARRIERS,
    DETECTORS,
    PAYLOADS,
    deletion_control,
    positive_control,
    run_suite,
    suite_digest,
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trials() -> list:
    return run_suite()


def test_the_suite_is_the_one_the_baseline_was_recorded_against(baseline) -> None:
    """A rate is only comparable to a rate measured on the same instrument."""
    assert baseline["suite_digest"] == suite_digest(), (
        "the payload, carrier or detector list changed since the baseline was frozen. If that was "
        "deliberate, regenerate with `python -m scripts.generate_evidence_injection_baseline` and "
        "look hard at the diff; if it was not, revert the edit."
    )
    assert baseline["trials"] == len(CARRIERS) * len(PAYLOADS)
    assert baseline["detectors"] == [name for name, _ in DETECTORS]


def test_the_injection_success_rate_did_not_increase(baseline, trials) -> None:
    escapes = sum(1 for trial in trials if trial.escaped)

    assert len(trials) == baseline["trials"], "the trial count moved without the digest moving"
    assert escapes <= baseline["escapes"], (
        f"{escapes} payloads escaped the evidence region, up from {baseline['escapes']}: "
        + "; ".join(
            f"{trial.carrier}/{trial.payload_id} fired {trial.detectors_fired}"
            for trial in trials
            if trial.escaped
        )
    )


def test_no_carrier_regressed_while_another_improved(baseline, trials) -> None:
    """A total that holds can hide one arm getting worse and another getting better."""
    for carrier in CARRIERS:
        rows = [trial for trial in trials if trial.carrier == carrier]
        escapes = sum(1 for trial in rows if trial.escaped)
        assert escapes <= baseline["by_carrier"][carrier]["escapes"], (
            f"the {carrier} carrier regressed: {escapes} escapes against "
            f"{baseline['by_carrier'][carrier]['escapes']} recorded"
        )


def test_the_detectors_fire_against_the_renderer_this_replaced(baseline) -> None:
    """The positive control. Without it, zero escapes and an inert detector set read alike.

    The pre-2026-08-05 renderer wrapped a plain `json.dumps` in the same delimiter. `json.dumps`
    escapes quotes, backslashes and control characters and does NOT escape `<` or `>`, which is
    exactly what the delimiter is made of — so a memory containing `</evidence_data>` closed the
    region early. That is the escape this control reproduces.
    """
    control = positive_control()
    escapes = [trial for trial in control if trial.escaped]

    assert escapes, "the detectors did not fire against a renderer known to leak: they are inert"
    assert len(escapes) >= baseline["positive_control_escapes"], (
        "the positive control got weaker, so the zero above is worth less than it was"
    )
    assert {trial.payload_id for trial in escapes} >= {"close_tag", "open_tag"}
    # The tag-closing payloads must trip the delimiter detector specifically, not merely trip
    # SOMETHING — an escape counted for an unrelated reason would not be evidence of this defect.
    assert all(
        "delimiter_not_intact" in trial.detectors_fired
        for trial in escapes
        if trial.payload_id in {"close_tag", "open_tag"}
    )


def test_the_boundary_escapes_the_evidence_rather_than_deleting_it(baseline, trials) -> None:
    """Zero escapes is worthless if it was bought by dropping the corpus text.

    `metadata` is excluded and that exclusion is stated rather than assumed: an evidence item
    carries no corpus metadata dict at all, so nothing from that carrier reaches the prompt to be
    preserved. Its zero is structural, not the result of escaping — which is also why its zero
    escape count proves less than the other two arms'.
    """
    for carrier in ("filename", "text"):
        rows = [trial for trial in trials if trial.carrier == carrier]
        preserved = sum(1 for trial in rows if trial.payload_preserved)
        assert preserved == len(rows), (
            f"{len(rows) - preserved} {carrier} payloads did not survive a JSON round trip: the "
            f"boundary is mangling evidence, not escaping it"
        )
        assert preserved == baseline["by_carrier"][carrier]["payload_preserved"]

    assert baseline["by_carrier"]["metadata"]["payload_preserved"] == 0
    assert all(
        not trial.payload_preserved for trial in trials if trial.carrier == "metadata"
    ), "chunk metadata began reaching the prompt; that arm now needs escaping, not absence"


def test_the_preservation_check_can_report_absence(baseline) -> None:
    """The negative control, and it was a mutation sweep that showed it was needed.

    `payload_preserved` reads 13 of 13 on both live arms, so a `_preserved` stuck at True is
    indistinguishable from a working one — the balanced-fixture failure this repository has
    recorded three times. This runs the same suite against a renderer that ships no evidence at
    all: it must score a PERFECT escape rate and zero preservation, which is the pair that makes
    the real zero mean something.
    """
    control = deletion_control()

    assert not any(trial.escaped for trial in control), (
        "a renderer that ships no evidence should escape nothing — the detectors are miscounting"
    )
    assert not any(trial.payload_preserved for trial in control), (
        "the preservation check reported evidence that was never rendered"
    )
    assert baseline["escapes"] == 0, "the live suite and this control must agree on the escape rate"


def test_every_payload_is_exercised_against_every_carrier(trials) -> None:
    """A trial that silently stopped running would lower the escape count for free."""
    seen = {(trial.carrier, trial.payload_id) for trial in trials}

    assert seen == {(carrier, payload_id) for carrier in CARRIERS for payload_id, _ in PAYLOADS}
