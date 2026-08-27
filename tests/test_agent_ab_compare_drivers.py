"""The driver-equivalence comparison: band arithmetic and refusals."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ab_compare_drivers import METRIC_PATHS, compare  # noqa: E402

pytestmark = pytest.mark.benchharness


def _analysis(**overrides) -> dict:
    base = {
        "run_id": "r",
        "admission": {"admitted_pairs": 54},
        "mechanism": {"search_rate": 1.0, "governing_memo_rate": 0.674},
        "primary_per_task": {"mean_delta": 0.208},
        "primary_per_pair": {"delta_mean": 0.196},
        "control": {"on_mean": 1.0, "off_mean": 1.0},
        "cost": {
            "input_tokens": {"delta_median": 106946.0},
            "wall_time_ms": {"delta_median": 36461.9},
        },
    }
    for key, value in overrides.items():
        base[key] = value
    return base


def _bands(**overrides) -> dict:
    bands = {
        "admitted_pairs": {"min": 48},
        "search_rate": {"max_abs_diff": 0.15},
        "governing_memo_rate": {"max_abs_diff": 0.15},
        "per_task_mean_delta": {"max_abs_diff": 0.15},
        "per_pair_delta_mean": {"max_abs_diff": 0.15},
        "control_on_mean": {"max_abs_diff": 0.125},
        "control_off_mean": {"max_abs_diff": 0.125},
        "input_tokens_delta_median": {"max_abs_diff": 55000},
        "wall_time_delta_median_ms": {"record_only": True},
    }
    bands.update(overrides)
    return bands


def test_identical_runs_are_equivalent_and_recorded_metrics_never_falsify() -> None:
    run = _analysis(cost={"input_tokens": {"delta_median": 106946.0}, "wall_time_ms": {"delta_median": 900000.0}})
    summary = compare(run, _analysis(), _bands())
    assert summary["verdict"] == "equivalent"
    assert summary["outside_band"] == []
    assert summary["metrics"]["wall_time_delta_median_ms"]["verdict"] == "recorded"


def test_a_metric_outside_its_band_falsifies_equivalence() -> None:
    run = _analysis(mechanism={"search_rate": 0.6, "governing_memo_rate": 0.674})
    summary = compare(run, _analysis(), _bands())
    assert summary["verdict"] == "not_equivalent"
    assert "search_rate" in summary["outside_band"]
    assert summary["metrics"]["search_rate"]["inside_band"] is False


def test_an_admitted_pair_floor_breach_is_a_wiring_void_not_a_falsification() -> None:
    run = _analysis(admission={"admitted_pairs": 20})
    summary = compare(run, _analysis(), _bands())
    assert summary["verdict"] == "wiring_void"


def test_a_metric_without_a_committed_band_is_refused() -> None:
    bands = _bands()
    del bands["search_rate"]
    with pytest.raises(SystemExit, match="search_rate"):
        compare(_analysis(), _analysis(), bands)


def test_a_band_for_an_unknown_metric_is_refused() -> None:
    with pytest.raises(SystemExit, match="does not compute"):
        compare(_analysis(), _analysis(), _bands(mystery_metric={"max_abs_diff": 1}))


def test_underscore_keys_in_the_bands_file_are_commentary_not_metrics() -> None:
    summary = compare(_analysis(), _analysis(), _bands(_comment="why these bands"))
    assert summary["verdict"] == "equivalent"


def test_a_metric_missing_from_the_run_voids_the_wiring_rather_than_falsifying() -> None:
    """A comparison that never ran must not publish as a measured non-equivalence.

    An absent key (a renamed field, a schema drift between analyzer versions) used to land in
    `outside_band`, so the summary would report the drivers as not equivalent when the truth was
    that the number was never computed.
    """
    run = _analysis(mechanism={"governing_memo_rate": 0.674})  # search_rate key removed
    summary = compare(run, _analysis(), _bands())
    assert summary["verdict"] == "wiring_void"
    assert summary["metrics"]["search_rate"]["verdict"] == "missing_value"
    assert "search_rate" not in summary["outside_band"]


def test_a_metric_missing_from_the_baseline_voids_too() -> None:
    baseline = _analysis(primary_per_task={})
    summary = compare(_analysis(), baseline, _bands())
    assert summary["verdict"] == "wiring_void"
    assert summary["metrics"]["per_task_mean_delta"]["verdict"] == "missing_value"


def test_a_non_numeric_metric_value_voids_instead_of_crashing() -> None:
    run = _analysis(primary_per_task={"mean_delta": "n/a"})
    summary = compare(run, _analysis(), _bands())
    assert summary["verdict"] == "wiring_void"
    assert summary["metrics"]["per_task_mean_delta"]["verdict"] == "missing_value"


def test_every_compared_metric_has_a_key_path() -> None:
    payload = _analysis()
    for metric, path in METRIC_PATHS.items():
        value = payload
        for key in path:
            assert key in value, f"{metric} path {path} does not resolve in the fixture"
            value = value[key]
