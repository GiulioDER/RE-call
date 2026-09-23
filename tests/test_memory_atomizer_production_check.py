"""Offline invariants of the production-path memory check.

Red proofs (2026-09-23, each mutation applied to ``scripts/memory_atomizer_production_check.py``
and reverted):

* ``test_only_gold_sources_are_excluded`` failed when ``gold_sources`` also collected ``source``.
* ``test_arms_differ_only_in_atomic_keys`` failed when the ``off`` arm kept a placement key from
  the base environment (the ``pop`` removed).
* ``test_summary_counts_paired_changes_and_latency`` failed when ``top5_changed`` compared the arm
  with itself.
"""

from __future__ import annotations

from scripts import memory_atomizer_production_check as prod


def test_only_gold_sources_are_excluded() -> None:
    records = [
        {"rows": [{"gold_sources": ["recall/a.md"], "candidates": [{"source": "recall/b.md"}]}]},
        {"gold_source": "steel/c.md"},
    ]
    assert prod.gold_sources(records) == {"recall/a.md", "steel/c.md"}


def test_arms_differ_only_in_atomic_keys() -> None:
    base = {"RECALL_ENV": "production", "RECALL_ATOMIC_RESCUE_PLACEMENT": "fused", "X": "1"}
    off = prod.arm_environment(base, "off", "/root")
    dense = prod.arm_environment(base, "dense", "/root")
    fused = prod.arm_environment(base, "fused", "/root")
    assert off["RECALL_ATOMIC_RESCUE_MODE"] == "off"
    assert "RECALL_ATOMIC_RESCUE_PLACEMENT" not in off
    assert dense["RECALL_ATOMIC_RESCUE_PLACEMENT"] == "dense"
    assert fused["RECALL_ATOMIC_RESCUE_PLACEMENT"] == "fused"
    for values in (off, dense, fused):
        assert values["RECALL_ENV"] == "production" and values["X"] == "1"
    assert base["RECALL_ATOMIC_RESCUE_PLACEMENT"] == "fused"


def _arm(exact, top5, atomic=None):
    return {
        "exact": exact, "source": exact, "abstained": False, "trust_state": "trusted",
        "top5": top5, "atomic_ms": atomic, "total_ms": 10.0, "error": None,
    }


def test_summary_counts_paired_changes_and_latency() -> None:
    rows = [
        {"off": _arm(8, ["a", "b"]), "dense": _arm(6, ["a", "c"], 30.0), "fused": _arm(6, ["a", "b"], 20.0)},
        {"off": _arm(1, ["x"]), "dense": _arm(2, ["y"], 40.0), "fused": _arm(1, ["x"], 25.0)},
    ]
    report = prod.summarise(rows)
    assert report["paired"]["dense"]["exact@6"] == {"gains": 1, "losses": 0, "net": 1}
    assert report["paired"]["dense"]["exact@1"] == {"gains": 0, "losses": 1, "net": -1}
    assert report["paired"]["fused"]["exact@1"] == {"gains": 0, "losses": 0, "net": 0}
    assert report["paired"]["dense"]["top5_changed"] == 2
    assert report["paired"]["fused"]["top5_changed"] == 0
    assert report["latency_ms"]["fused"]["p95"] == 25.0
    assert report["latency_ms"]["fused"]["atomic_stage_runs"] == 2
