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
GEN = ROOT / "recall" / "generation_store.py"
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
    (
        "a subclass cannot silently drop the timing by overriding the PUBLIC method",
        GEN,
        "def _query_dense(",
        "def query_dense(",
        "test_no_subclass_overrides_the_timed_public_query_methods",
    ),
    (
        "the per-search freshness round trip is timed",
        STORE,
        "with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_META):",
        "if True:",
        "test_newest_indexed_at_is_timed_as_a_store_leg",
    ),
    (
        "snapshot() reveals truncation, not just the drain path",
        OBS,
        '"truncated": observed > len(samples),',
        '"truncated": False,',
        "test_snapshot_reveals_truncation_like_the_drain_does",
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
    for target in (STORE, OBS, GEN):
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
        original = path.read_bytes()
        # Anchors are written with "\n", but these files are CRLF in the working copy (git
        # normalises on commit; .gitattributes pins LF in the index). `read_text` used to hide
        # that by translating newlines. Comparing BYTES does not, so the anchor must be re-encoded
        # with the newline the file actually uses. Switching to bytes WITHOUT this dropped 5 of 10
        # rules to SKIP: the fix for the restore check broke the anchor matching.
        newline = "\r\n" if b"\r\n" in original else "\n"
        old_b = old.replace("\n", newline).encode("utf-8")
        new_b = new.replace("\n", newline).encode("utf-8")
        if original.count(old_b) != 1:
            print(f"  SKIP  {label}: anchor matched {original.count(old_b)} times, expected 1")
            continue
        path.write_bytes(original.replace(old_b, new_b))
        try:
            code, output = run(test)
            # Retry ONCE on a non-verdict, INSIDE the try — the mutation must still be applied.
            # (Placing this after the `finally` re-ran the test against RESTORED source, which
            # passes, so every transient scored GREEN "not caught". The retry has to happen while
            # the code under test is still mutated or it is not a retry of the same experiment.)
            #
            # Why a retry at all: back-to-back pytest processes contend for the schema migration
            # advisory lock, and the loser raises ConcurrentMigrator during fixture setup. That is
            # a real error, but about scheduling, not about the rule under test.
            if code != 0 and "1 failed" not in output:
                code, output = run(test)
        finally:
            path.write_bytes(original)
            # Verify the restore, do not assume it. A partial write here leaves shipped library
            # source mutated, and two of these mutations read as plausible code in a later diff.
            # The check sets a flag rather than returning: a `return` inside `finally` discards
            # any in-flight exception, which would hide the very failure that broke the restore.
            # BYTES, not text: read_text/write_text translate newlines, so a text
            # comparison cannot observe a newline-only non-restore at all.
            restored = path.read_bytes() == original
        if not restored:
            print(f"FATAL: {path} was not restored; run `git checkout -- {path}`", file=sys.stderr)
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
