"""NUM-001 consumer regressions, red on 33759a686 before the endpoint fix."""
from __future__ import annotations

import json
import sys

import pytest

from benchmarks.mtrag.probe import arm_r_costs


@pytest.mark.parametrize("pay,unans,label,expected_mean,expected_caught", [
    ([0.2], [0.3, 0.4], "IN-SAMPLE", "0.6667", "100.0%"),
    ([0.2], [0.4, 0.4], "IN-SAMPLE", "0.6667", "100.0%"),
    ([0.2], [0.2], "PREREGISTERED", "0.5000", "100.0%"),
])
def test_cost_report_includes_all_abstain_interval(tmp_path, monkeypatch, capsys, pay, unans,
                                                 label, expected_mean, expected_caught):
    artifact = tmp_path / "scores.json"
    artifact.write_text(json.dumps({
        "rows": [{"cls": cls, "top1_cosine": score, "gap_warning": False}
                 for cls, scores in (("ANSWERABLE", pay), ("UNANSWERABLE", unans)) for score in scores],
        "separability": 0.5, "separability_ci": [0.1, 0.9], "embedder": "stub", "threshold": 0.2,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["arm_r_costs", "--artifact", str(artifact)])
    assert arm_r_costs.main() == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if label in line)
    columns = line.split()
    assert columns[-2] == expected_mean
    assert columns[-3] == expected_caught
