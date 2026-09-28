"""`scripts/aml_w3_history_check.py`: the scoring that decides W3's go/no-go, on a fixture.

Red proof in the docstring.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w3_history_check as w3  # noqa: E402


def test_an_update_is_right_only_when_the_latest_value_cites_the_updated_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: a linked update pair counts as correct only when its key's latest fact cites the
    ``updated_info`` turn, and a contradiction pair counts as detected only when classed conflict.

    Red proof: counting an update pair as correct whenever the kind is ``update`` (dropping the
    ``updated_info`` turn check) scores the second, wrongly ordered pair as right and fails the
    update_correct assertion.
    """
    pairs = [
        {"type": "knowledge_update", "left": [10], "right": [20]},
        {"type": "knowledge_update", "left": [30], "right": [40]},
        {"type": "contradiction_resolution", "left": [50], "right": [60]},
    ]
    monkeypatch.setattr(w3, "pairs_of", lambda row: pairs)
    record = {"conversation": 0, "facts": [
        {"key": "repo|commits", "value": "150", "relation": "state", "turns": [10]},
        {"key": "repo|commits", "value": "165", "relation": "state", "turns": [20]},
        # The old value is restated after the update (turn 45), so by turn order the latest fact
        # is the old value and does not cite the updated turn (40): the pair must count as wrong.
        {"key": "api|latency", "value": "400ms", "relation": "state", "turns": [30]},
        {"key": "api|latency", "value": "250ms", "relation": "state", "turns": [40]},
        {"key": "api|latency", "value": "400ms", "relation": "state", "turns": [45]},
        {"key": "flask|routes", "value": "never written routes", "relation": "never", "turns": [50]},
        {"key": "flask|routes", "value": "a homepage route", "relation": "state", "turns": [60]},
    ]}
    result = w3.check([{}], [record])
    assert result["knowledge_update"] == {"linked": 2, "update_correct": 0.5, "update_read_as_conflict": 0.0}
    assert result["contradiction_resolution"]["conflict_detected"] == 1.0
