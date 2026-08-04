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


def run(test: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", f"{TESTS}::{test}", "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=os.environ,
    )
    return proc.returncode, proc.stdout + proc.stderr


def collectable(tests: list[str]) -> tuple[bool, str]:
    """Every named test must resolve BEFORE any mutation.

    Without this, a renamed or misspelled test id makes pytest exit 4 forever, and an exit-code
    check reads that as "the guard caught the mutation" for the rest of the file's life. The
    ablation would report a perfect score having never run an assertion.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "--collect-only", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=os.environ,
    )
    collected = proc.stdout
    missing = [t for t in tests if t not in collected]
    return (not missing), ", ".join(missing)


def main() -> int:
    if not os.environ.get("RECALL_TEST_DSN"):
        print("RECALL_TEST_DSN must point at a throwaway database", file=sys.stderr)
        return 2

    # Refuse to mutate a dirty tree: the restore below rewrites these files from a snapshot taken
    # at mutation time, so an unrelated uncommitted edit would be silently reverted to it.
    for target in (STORE, OBS):
        if subprocess.run(
            ["git", "diff", "--quiet", "--", str(target)], cwd=ROOT
        ).returncode != 0:
            print(f"FAIL: {target.name} has uncommitted changes; commit or stash first",
                  file=sys.stderr)
            return 2

    ok, missing = collectable([test for *_, test in ABLATIONS])
    if not ok:
        print(f"FAIL: these ablation tests do not resolve: {missing}", file=sys.stderr)
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

    reds, inconclusive = 0, 0
    for label, path, old, new, test in ABLATIONS:
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            print(f"  SKIP  {label}: anchor matched {original.count(old)} times, expected 1")
            continue
        path.write_text(original.replace(old, new), encoding="utf-8")
        try:
            code, output = run(test)
        finally:
            path.write_text(original, encoding="utf-8")
            # Verify the restore, do not assume it. A partial write here leaves shipped library
            # source mutated, and two of these mutations read as plausible code in a later diff.
            if path.read_text(encoding="utf-8") != original:
                print(f"FATAL: {path} was not restored; run `git checkout -- {path}`",
                      file=sys.stderr)
                return 3

        # Exit 1 AND a real failure line. pytest exits 2 on a collection error, 3 internal,
        # 4 usage, 5 nothing-collected — all non-zero, none of them an assertion that fired.
        # Counting those as RED is how an ablation reports a perfect score having proved nothing.
        if code == 1 and "1 failed" in output:
            reds += 1
            print(f"  RED   {label}  ({test})")
        elif code == 0:
            print(f"  GREEN {label}  <-- {test} DID NOT CATCH THIS")
        else:
            inconclusive += 1
            print(f"  INCONCLUSIVE {label}: pytest exit {code}, no failure line")
            print("    " + output.strip().splitlines()[-1] if output.strip() else "")

    print(f"\n{reds}/{len(ABLATIONS)} ablations caught"
          + (f", {inconclusive} INCONCLUSIVE" if inconclusive else ""))
    return 0 if reds == len(ABLATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
