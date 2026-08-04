"""Ablate one rule at a time and require the named test to go red.

Each entry removes EXACTLY ONE rule and names the test that exists to catch it. The distinction
matters: a mutation that deletes something adjacent produces a red that pins the adjacent thing,
so the red is evidence only about the rule the mutation actually removed.

Run: uv run --extra dev python scripts/ablate_store_latency_guards.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "recall" / "store.py"
OBS = ROOT / "recall" / "observability.py"
TESTS = "tests/test_store_query_latency.py"

DENSE_TIMED = """        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_DENSE):
            return self._query_dense(vector, k, source)"""

SPARSE_TIMED = """        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_SPARSE):
            return self._query_sparse(text, k, source, vec)"""

TIMER_BODY = """        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)"""

#: (label, file, old, new, test that MUST fail)
ABLATIONS = [
    (
        "dense leg is timed at all",
        STORE,
        DENSE_TIMED,
        "        return self._query_dense(vector, k, source)",
        "test_query_dense_records_one_sample_per_call",
    ),
    (
        "sparse leg is timed at all",
        STORE,
        SPARSE_TIMED,
        "        return self._query_sparse(text, k, source, vec)",
        "test_the_two_legs_are_separate_series",
    ),
    (
        "the timer records ELAPSED time, not a constant",
        OBS,
        "self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)",
        "self.observe(name, 0.0, **labels)",
        "test_recorded_latency_tracks_injected_delay",
    ),
    (
        "a rejected call is excluded from the distribution",
        STORE,
        """        if k <= 0:
            raise ValueError("k must be a positive int")
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_DENSE):
            return self._query_dense(vector, k, source)""",
        """        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_DENSE):
            if k <= 0:
                raise ValueError("k must be a positive int")
            return self._query_dense(vector, k, source)""",
        "test_a_rejected_call_records_nothing",
    ),
    (
        "a raising call is still recorded",
        OBS,
        TIMER_BODY,
        """        start = time.perf_counter()
        yield
        self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)""",
        "test_a_failing_query_is_still_timed",
    ),
    (
        "drain CLEARS the ring",
        OBS,
        "samples = list(self._histograms.pop(key, ()))",
        "samples = list(self._histograms.get(key, ()))",
        "test_drain_isolates_consecutive_measurements",
    ),
    (
        "drain clears the TOTAL as well as the ring",
        OBS,
        "total = self._histogram_totals.pop(key, 0)",
        "total = self._histogram_totals.get(key, 0)",
        "test_drain_reports_evicted_samples_rather_than_hiding_them",
    ),
]


def run(test: str) -> int:
    return subprocess.run(
        [sys.executable, "-m", "pytest", f"{TESTS}::{test}", "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=os.environ,
    ).returncode


def main() -> int:
    if not os.environ.get("RECALL_TEST_DSN"):
        print("RECALL_TEST_DSN must point at a throwaway database", file=sys.stderr)
        return 2

    print(f"baseline: {len(ABLATIONS)} rules, requiring the suite green before ablating")
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, env=os.environ,
    )
    if baseline.returncode != 0:
        print(baseline.stdout[-2000:], file=sys.stderr)
        print("FAIL: the suite is not green before ablation; fix that first", file=sys.stderr)
        return 2

    reds = 0
    for label, path, old, new, test in ABLATIONS:
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            print(f"  SKIP  {label}: anchor matched {original.count(old)} times, expected 1")
            continue
        path.write_text(original.replace(old, new), encoding="utf-8")
        try:
            code = run(test)
        finally:
            path.write_text(original, encoding="utf-8")
        verdict = "RED " if code != 0 else "GREEN"
        if code != 0:
            reds += 1
        else:
            print(f"  {verdict} {label}  <-- {test} DID NOT CATCH THIS")
            continue
        print(f"  {verdict} {label}  ({test})")

    print(f"\n{reds}/{len(ABLATIONS)} ablations caught")
    return 0 if reds == len(ABLATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
