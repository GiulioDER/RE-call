"""Compare a driver-equivalence replication against its archived baseline, band by band.

    python scripts/agent_ab_compare_drivers.py \
        --run benchmarks/artifacts/agent_ab/agent-ab-sdk-replication-001/analysis.json \
        --baseline ~/.claude/archive/agent-ab-skill-001/analysis.json \
        --bands benchmarks/agent_ab/sdk-replication-bands.json \
        --out results/agent_ab/agent_ab_sdk_replication_<date>.json

Reads two `analysis.json` files as written by `scripts/agent_ab_analyze_tasks.py`, applies the
COMMITTED equivalence bands, and emits one compact summary JSON: per metric,
`{baseline, replication, difference, band, inside_band}`. The bands file is part of the
preregistration and must be committed before the replication runs; this script refuses to invent
a band for a metric that has none, because a band chosen after seeing both numbers is not an
equivalence criterion, it is a rationalisation.

Two kinds of row, stated in the bands file itself:

- `max_abs_diff`: a falsifier. The replication value must sit within that distance of the
  baseline value.
- `min` (admitted pairs only): a floor. Falling below it is a WIRING result that voids the run,
  not an equivalence failure.
- `record_only: true`: recorded, never falsifying. Wall time and RE-call latency go here, because
  the driver changes both mechanically (SDK spawn overhead; arrival-clock latency basis).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Where each compared metric lives inside an `analysis.json`, as a key path.
METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "admitted_pairs": ("admission", "admitted_pairs"),
    "search_rate": ("mechanism", "search_rate"),
    "governing_memo_rate": ("mechanism", "governing_memo_rate"),
    "per_task_mean_delta": ("primary_per_task", "mean_delta"),
    "per_pair_delta_mean": ("primary_per_pair", "delta_mean"),
    "control_on_mean": ("control", "on_mean"),
    "control_off_mean": ("control", "off_mean"),
    "input_tokens_delta_median": ("cost", "input_tokens", "delta_median"),
    "wall_time_delta_median_ms": ("cost", "wall_time_ms", "delta_median"),
}


def _lookup(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def compare(
    run: dict[str, Any], baseline: dict[str, Any], bands: dict[str, Any]
) -> dict[str, Any]:
    """Build the per-metric comparison. Pure, so the test can drive it with fixtures."""

    bands = {key: value for key, value in bands.items() if not key.startswith("_")}
    unknown = sorted(set(bands) - set(METRIC_PATHS))
    if unknown:
        raise SystemExit(f"bands name metrics this script does not compute: {unknown}")
    missing = sorted(set(METRIC_PATHS) - set(bands))
    if missing:
        raise SystemExit(
            f"no band declared for: {missing}. Every compared metric needs a committed band "
            f"(or an explicit record_only: true) BEFORE the run; add it to the bands file."
        )

    rows: dict[str, Any] = {}
    falsified: list[str] = []
    wiring_void = False
    for metric, path in METRIC_PATHS.items():
        band = bands[metric]
        base_value = _lookup(baseline, path)
        run_value = _lookup(run, path)
        row: dict[str, Any] = {"baseline": base_value, "replication": run_value}
        if base_value is not None and run_value is not None:
            row["difference"] = run_value - base_value
        if band.get("record_only"):
            row["verdict"] = "recorded"
        elif "min" in band:
            row["band"] = {"min": band["min"]}
            ok = run_value is not None and run_value >= band["min"]
            row["verdict"] = "ok" if ok else "wiring_void"
            wiring_void = wiring_void or not ok
        elif "max_abs_diff" in band:
            row["band"] = {"max_abs_diff": band["max_abs_diff"]}
            ok = (
                base_value is not None
                and run_value is not None
                and abs(run_value - base_value) <= band["max_abs_diff"]
            )
            row["inside_band"] = ok
            row["verdict"] = "equivalent" if ok else "outside_band"
            if not ok:
                falsified.append(metric)
        else:
            raise SystemExit(
                f"band for {metric!r} declares none of max_abs_diff/min/record_only"
            )
        rows[metric] = row

    if wiring_void:
        verdict = "wiring_void"
    elif falsified:
        verdict = "not_equivalent"
    else:
        verdict = "equivalent"
    return {
        "baseline_run_id": baseline.get("run_id"),
        "replication_run_id": run.get("run_id"),
        "metrics": rows,
        "outside_band": falsified,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="the replication's analysis.json")
    parser.add_argument("--baseline", required=True, help="the archived baseline's analysis.json")
    parser.add_argument(
        "--bands", required=True, help="the COMMITTED equivalence bands JSON (preregistered)"
    )
    parser.add_argument("--out", default=None, help="where to write the summary JSON")
    args = parser.parse_args()

    paths = {name: Path(getattr(args, name)).expanduser() for name in ("run", "baseline", "bands")}
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"--{name} does not exist: {path}")
    summary = compare(
        json.loads(paths["run"].read_text(encoding="utf-8")),
        json.loads(paths["baseline"].read_text(encoding="utf-8")),
        json.loads(paths["bands"].read_text(encoding="utf-8")),
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    print(rendered)
    return 0 if summary["verdict"] == "equivalent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
