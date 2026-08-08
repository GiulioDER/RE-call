"""Ablate one rule at a time and require the named test to go red.

Each entry removes EXACTLY ONE rule and names the test that exists to catch it. The distinction
matters: a mutation that deletes something adjacent produces a red that pins the adjacent thing,
so the red is evidence only about the rule the mutation actually removed.

⚠️ This sweep MUTATES `recall/*.py` in place while it runs, so no other pytest run may be in
flight against this checkout: a concurrent suite reads half-mutated source and goes
non-deterministically red, blaming whatever it happened to read. The dirty-tree refusal in
`main()` only covers the inverse direction (a dirty tree before the sweep starts).

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
SPARSE = ROOT / "recall" / "sparse.py"
HOSTLOAD = ROOT / "recall" / "eval" / "hostload.py"
BENCH = ROOT / "benchmarks" / "store_latency_share.py"
TESTS = "tests/test_store_query_latency.py"

DENSE_TIMED = """        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_DENSE):
            return self._query_dense(vector, k, source)"""

SPARSE_TIMED = """        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_SPARSE):
            return self._query_sparse(text, k, source, vec)"""

#: The declaration the tuple-drift rule ablates. A CONSTANT because the tuple is multi-line: this
#: anchor was left at the old single-line spelling when the learned sparse leg widened the tuple,
#: and a stale anchor does not fail the sweep loudly — it prints SKIP and that rule silently stops
#: being exercised, which is the same class of defect the rule itself exists to catch.
TIMED_TUPLE = """TIMED_PUBLIC_METHODS = (
    "query_dense",
    "query_sparse",
    "query_learned_sparse",
    "newest_indexed_at",
)"""

TIMER_BODY = """        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)"""

#: (label, file, old, new, test_file, test that MUST fail)
ABLATIONS = [
    (
        "dense leg is timed at all",
        STORE,
        DENSE_TIMED,
        "        return self._query_dense(vector, k, source)",
        TESTS,
        "test_query_dense_records_one_sample_per_call",
    ),
    (
        "sparse leg is timed at all",
        STORE,
        SPARSE_TIMED,
        "        return self._query_sparse(text, k, source, vec)",
        TESTS,
        "test_the_two_legs_are_separate_series",
    ),
    (
        "the timer records ELAPSED time, not a constant",
        OBS,
        "self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)",
        "self.observe(name, 0.0, **labels)",
        TESTS,
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
        TESTS,
        "test_a_rejected_call_records_nothing",
    ),
    (
        "a raising call is still recorded",
        OBS,
        TIMER_BODY,
        """        start = time.perf_counter()
        yield
        self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)""",
        TESTS,
        "test_a_failing_query_is_still_timed",
    ),
    (
        "drain CLEARS the ring",
        OBS,
        "samples = list(self._histograms.pop(key, ()))",
        "samples = list(self._histograms.get(key, ()))",
        TESTS,
        "test_drain_isolates_consecutive_measurements",
    ),
    (
        "drain clears the TOTAL as well as the ring",
        OBS,
        "total = self._histogram_totals.pop(key, 0)",
        "total = self._histogram_totals.get(key, 0)",
        TESTS,
        "test_drain_reports_evicted_samples_rather_than_hiding_them",
    ),
    (
        "a subclass cannot silently drop the timing by overriding the PUBLIC method",
        GEN,
        "def _query_dense(",
        "def query_dense(",
        TESTS,
        "test_no_subclass_overrides_the_timed_public_query_methods",
    ),
    (
        "the per-search freshness round trip is timed",
        STORE,
        "with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_META):",
        "if True:",
        TESTS,
        "test_newest_indexed_at_is_timed_as_a_store_leg",
    ),
    (
        # Mutate the TUPLE, not a timer. Removing a timer would also turn that leg's own test
        # red, so the red would not be evidence about THIS rule alone — the sweep's docstring
        # rule that each entry must remove exactly one rule.
        "TIMED_PUBLIC_METHODS stays in step with the real timer call sites",
        STORE,
        TIMED_TUPLE,
        """TIMED_PUBLIC_METHODS = (
    "query_dense",
    "query_sparse",
    "query_learned_sparse",
)""",
        TESTS,
        "test_timed_public_methods_matches_the_actual_timer_call_sites",
    ),
    (
        # Same shape as the entry above, one level along: the tuple that says which legs a
        # caller must DRAIN, rather than which methods are timed. Dropping a leg here is what
        # actually happened when the learned sparse leg was added, in two callers at once.
        "STORE_QUERY_LEGS stays in step with the labels the timers emit",
        STORE,
        "STORE_QUERY_LEGS = (LEG_DENSE, LEG_SPARSE, LEG_LEARNED_SPARSE, LEG_META)",
        # Drops LEG_LEARNED_SPARSE specifically, so the mutation reproduces the drift that
        # actually happened rather than an equivalent one.
        "STORE_QUERY_LEGS = (LEG_DENSE, LEG_SPARSE, LEG_META)",
        TESTS,
        "test_store_query_legs_matches_the_actual_timer_labels",
    ),
    (
        "snapshot() reveals truncation, not just the drain path",
        OBS,
        '"truncated": observed > len(samples),',
        '"truncated": False,',
        TESTS,
        "test_snapshot_reveals_truncation_like_the_drain_does",
    ),
    (
        "the sparse coverage guard refuses a partial sidecar",
        SPARSE,
        "    if encoded == total:\n        return",
        "    if encoded <= total:\n        return",
        "tests/test_sparse_indexing.py",
        "test_coverage_refuses_a_half_encoded_corpus",
    ),
    (
        "the host load guard refuses a busy host",
        HOSTLOAD,
        "    if load is None or allow_busy or load <= ceiling:\n        return load",
        "    return load",
        "tests/test_hostload.py",
        "test_a_busy_host_is_refused",
    ),
    (
        # The plan's original anchor was the CLI loop's `measure(..., sparse_backend=backend,
        # ...)` call site (main()'s per-config sweep). That call site is real, but the named
        # test below calls `measure()` directly with an explicit `sparse_backend="splade"`
        # argument, so it never executes main() and cannot observe a hardcode there: measured,
        # that anchor scores GREEN, not RED. This anchor is the one INSIDE measure() that
        # actually reaches the test: the value `measure()` stamps into the `LegSplit` it
        # returns. Verified against the current tree with the test failing on
        # `assert split.sparse_backend == "splade"` before this entry was accepted.
        "measure() reports the SELECTED backend in its LegSplit, not a hardcoded one",
        BENCH,
        "    return LegSplit(\n        candidate_k=candidate_k,\n"
        "        reranked=reranker is not None,\n        sparse_backend=sparse_backend,",
        "    return LegSplit(\n        candidate_k=candidate_k,\n"
        '        reranked=reranker is not None,\n        sparse_backend="lexical",',
        "tests/test_store_latency_splade_arm.py",
        "test_the_splade_arm_reports_a_learned_fire_rate_and_no_lexical_leg",
    ),
    (
        "an explicitly requested cuda device is refused when unusable",
        SPARSE,
        "    if requested == \"cuda\" and report.refusal:",
        "    if False and report.refusal:",
        "tests/test_sparse_device.py",
        "test_requesting_cuda_explicitly_raises_rather_than_falling_back",
    ),
]


def run(test_file: str, test: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", f"{test_file}::{test}", "-q", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=os.environ,
    )
    return proc.returncode, proc.stdout + proc.stderr


def collectable(tests: list[tuple[str, str]]) -> tuple[bool, str]:
    """Every named test must resolve BEFORE any mutation, in the file it is claimed to live in.

    Without this, a renamed or misspelled test id makes pytest exit 4 forever, and an exit-code
    check reads that as "the guard caught the mutation" for the rest of the file's life. The
    ablation would report a perfect score having never run an assertion. `tests` is (test_file,
    test) pairs rather than bare names because entries now span more than one file: a name that
    collects fine in the wrong file is the same silent-SKIP failure this check exists to catch.
    """
    missing: list[str] = []
    for test_file in sorted({tf for tf, _ in tests}):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "--collect-only", "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, env=os.environ,
        )
        collected = proc.stdout
        missing.extend(
            f"{tf}::{t}" for tf, t in tests if tf == test_file and t not in collected
        )
    return (not missing), ", ".join(missing)


def main() -> int:
    if not os.environ.get("RECALL_TEST_DSN"):
        print("RECALL_TEST_DSN must point at a throwaway database", file=sys.stderr)
        return 2

    # Refuse to mutate a dirty tree: the restore below rewrites these files from a snapshot taken
    # at mutation time, so an unrelated uncommitted edit would be silently reverted to it.
    for target in (STORE, OBS, GEN, SPARSE, HOSTLOAD, BENCH):
        if subprocess.run(
            ["git", "diff", "--quiet", "--", str(target)], cwd=ROOT
        ).returncode != 0:
            print(f"FAIL: {target.name} has uncommitted changes; commit or stash first",
                  file=sys.stderr)
            return 2

    ok, missing = collectable([(test_file, test) for *_, test_file, test in ABLATIONS])
    if not ok:
        print(f"FAIL: these ablation tests do not resolve: {missing}", file=sys.stderr)
        return 2

    test_files = sorted({test_file for *_, test_file, _ in ABLATIONS})
    print(f"baseline: {len(ABLATIONS)} rules, requiring the suite green before ablating")
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", *test_files, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, env=os.environ,
    )
    if baseline.returncode != 0:
        print(baseline.stdout[-2000:], file=sys.stderr)
        print("FAIL: the suite is not green before ablation; fix that first", file=sys.stderr)
        return 2

    reds, inconclusive = 0, 0
    for label, path, old, new, test_file, test in ABLATIONS:
        original = path.read_bytes()
        # Anchors are written with "\n", but these files are CRLF in the working copy (git
        # normalises on commit; .gitattributes pins LF in the index). `read_text` used to hide
        # that by translating newlines. Comparing BYTES does not, so the anchor must be re-encoded
        # with the newline the file actually uses. Switching to bytes WITHOUT this took 4 of the 10
        # rules to SKIP and a 5th to INCONCLUSIVE: the fix for the restore check broke the
        # anchor matching. Assumes a uniformly-lined file; a mixed-ending file would need a
        # per-anchor sniff.
        newline = "\r\n" if b"\r\n" in original else "\n"
        old_b = old.replace("\n", newline).encode("utf-8")
        new_b = new.replace("\n", newline).encode("utf-8")
        if original.count(old_b) != 1:
            print(f"  SKIP  {label}: anchor matched {original.count(old_b)} times, expected 1")
            continue
        path.write_bytes(original.replace(old_b, new_b))
        try:
            code, output = run(test_file, test)
            # Retry ONCE on a non-verdict, INSIDE the try — the mutation must still be applied.
            # (Placing this after the `finally` re-ran the test against RESTORED source, which
            # passes, so every transient scored GREEN "not caught". The retry has to happen while
            # the code under test is still mutated or it is not a retry of the same experiment.)
            #
            # Why a retry at all: back-to-back pytest processes contend for the schema migration
            # advisory lock, and the loser raises ConcurrentMigrator during fixture setup. That is
            # a real error, but about scheduling, not about the rule under test.
            if code != 0 and "1 failed" not in output:
                code, output = run(test_file, test)
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
