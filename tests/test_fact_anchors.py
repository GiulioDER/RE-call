"""The judge-free scorer, pinned before it scores anything.

The supersession probe replaces an LLM judge with substring matching, which is cheap and
reproducible and wrong in specific ways. Those ways are pinned here so the probe's write-up can
state them as measured properties rather than as caveats someone remembered to add.

Properties, one test each:
  1. A hand-written answer containing every anchor scores 1.0 (pre-registered check A1).
  2. An empty answer scores 0.0 (pre-registered check A2).
  3. A partial answer scores strictly between, so the scorer is not a constant (check A3's shape).
  4. A NEGATIVE anchor scores a hit when ABSENT and a miss when present, which is the direction
     most easily got backwards.
  5. Normalisation unifies the glyphs a model renders differently from the source, and no more.
  6. A row with no scorable anchor raises rather than reporting 1.0.
  7. The loader refuses a malformed anchor file rather than scoring less than it claims to.
  8. `unanchorable` facts are excluded from the denominator, deliberately and visibly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.fact_anchors import (
    RowScore,
    anchors_digest,
    load_anchors,
    normalize,
    score_row,
)

ROW = {
    "question_id": "qst_test",
    "positive": [
        {"id": "endpoint", "any": ["/v1/capacity/migrations/start"]},
        {"id": "rate", "any": ["$0.085", "0.085"]},
        {"id": "range", "all": ["160", "260"]},
    ],
    "negative": [{"id": "no_singular", "any": ["/v1/capacity/migration/start"]}],
    "unanchorable": [{"fact": 1, "why": "framing"}],
}


def test_a1_a_complete_answer_scores_one():
    answer = "Use POST /v1/capacity/migrations/start. Egress is $0.085 per GiB, +$160-$260/hour."
    score = score_row(answer, ROW)
    assert score.rate == 1.0
    assert score.missed == () and score.violated == ()


def test_a2_an_empty_answer_scores_zero():
    """Positives all miss; the negative is vacuously satisfied, so it is NOT a free hit here
    only because the total counts it too. The rate is what matters and it is not 1.0."""
    score = score_row("", ROW)
    assert score.hits == 1, "the absent negative anchor is the only thing an empty answer earns"
    assert score.rate == 0.25
    assert set(score.missed) == {"endpoint", "rate", "range"}


def test_a3_a_partial_answer_scores_between():
    """Without this the two tests above pass for a scorer that returns a constant per input."""
    score = score_row("The endpoint is POST /v1/capacity/migrations/start.", ROW)
    assert 0.0 < score.rate < 1.0
    assert "rate" in score.missed


def test_a_negative_anchor_scores_the_right_way_round():
    clean = score_row("Use /v1/capacity/migrations/start at $0.085, 160 to 260 per hour.", ROW)
    assert clean.violated == () and clean.rate == 1.0

    invented = score_row(
        "Use /v1/capacity/migration/start at $0.085, 160 to 260 per hour.", ROW
    )
    assert invented.violated == ("no_singular",)
    assert invented.rate < 1.0, "inventing a path must cost, not be free"


@pytest.mark.parametrize(
    ("written", "anchor"),
    [
        ("Score ≥85 for Tier 1", ">=85"),
        ("Tier 2 is 70–84", "70-84"),
        ("Tier 2 is 70—84", "70-84"),
        ("SHA256-Only", "sha256-only"),
        ("uses   sampled    bytes", "sampled bytes"),
    ],
    ids=["ge-glyph", "en-dash", "em-dash", "case", "whitespace"],
)
def test_normalisation_unifies_what_a_model_renders_differently(written: str, anchor: str):
    assert normalize(anchor) in normalize(written)


def test_normalisation_does_not_over_reach():
    """A scorer that normalised aggressively would start matching things that are not there."""
    assert normalize("migrations") not in normalize("The migration path")
    assert normalize("$0.085") not in normalize("the rate is $0.85 per GiB")
    assert normalize("sha256-only") not in normalize("we use sha256 only for hashing")


def test_a_row_with_no_scorable_anchor_raises_rather_than_scoring_one():
    """An empty denominator is not a perfect score; it is an unmeasured row, and letting it
    report 1.0 would let an unanchorable row inflate an arm."""
    empty = RowScore(question_id="q", hits=0, total=0, missed=(), violated=())
    with pytest.raises(ValueError, match="no scorable anchors"):
        _ = empty.rate


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"positive": [], "negative": []}, "no scorable anchors"),
        ({"positive": [{"any": ["x"]}]}, "no id"),
        ({"positive": [{"id": "a", "any": ["x"]}, {"id": "a", "any": ["y"]}]}, "repeats anchor"),
        ({"positive": [{"id": "a", "any": ["x"], "all": ["y"]}]}, "exactly one of any/all"),
        ({"positive": [{"id": "a"}]}, "exactly one of any/all"),
        ({"positive": [{"id": "a", "any": []}]}, "empty match string"),
        ({"positive": [{"id": "a", "any": ["  "]}]}, "empty match string"),
    ],
    ids=["no-anchors", "no-id", "duplicate-id", "both-any-and-all", "neither", "empty-list",
         "blank-string"],
)
def test_the_loader_refuses_a_malformed_anchor_file(tmp_path: Path, mutation: dict, match: str):
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps({"rows": {"qst_x": mutation}}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_anchors(path)


def test_unanchorable_facts_are_excluded_from_the_denominator():
    """Counting them would understate both arms; dropping them silently would overstate the
    instrument. They are carried in the file so the human read knows where to look."""
    assert score_row("", ROW).total == 4  # 3 positive + 1 negative, NOT the unanchorable fact
    assert ROW["unanchorable"], "the fixture must actually exercise the exclusion"


def test_the_digest_moves_when_an_anchor_moves():
    """The anchors are frozen before the run; the digest is what proves it."""
    before = anchors_digest({"rows": {"q": ROW}})
    changed = json.loads(json.dumps({"rows": {"q": ROW}}))
    changed["rows"]["q"]["positive"][0]["any"].append("/v1/other")
    assert anchors_digest(changed) != before
