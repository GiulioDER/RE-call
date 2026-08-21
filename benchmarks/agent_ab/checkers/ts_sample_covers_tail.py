"""Does the sampler reach the tail of the corpus, or only ever its head?

Sorting a random sample and then stopping at a budget is not sampling. Both halves are usually
added for good reasons, the sort for stable output order and the stop for a size cap, and together
they mean only the lowest-indexed part of the sample is ever considered. Measured twice on this
repository's own docs: **3 of 51 files** where sampling and continuing reached 21.

The fixture is sized so the budget genuinely binds. 1815 chunks over 51 files, mean text 584
characters, 20 chunks requested and 4000 characters allowed, so roughly six survive the cap and
which six is the whole question.

The checker runs 120 seeds and counts distinct files across the union. A sampler that decides
membership before sorting reaches nearly all 51; one that sorts and then breaks reaches a handful,
and no amount of luck closes that gap over 120 runs. The threshold sits at 25, far from both.
"""

from __future__ import annotations

from pathlib import Path

from ..tasksuccess import CheckResult
from ._run import run_python

SCRIPT = "scripts/sample.py"
SEEDS = 120
#: Union of distinct files that must appear. Head-biased samplers land near 3 to 8; a correct one
#: lands near 51. Nothing sits at 25 by accident.
MIN_FILES = 25
BUDGET_CHARS = 4000
#: A little slack over the stated budget, so a sampler is not failed for a trailing newline or a
#: header line. It is nowhere near enough to let an unbudgeted sampler through: printing all 20
#: chunks would be roughly 11,000 characters.
BUDGET_TOLERANCE = 200


def check(workdir: Path) -> CheckResult:
    script = workdir / SCRIPT
    if not script.is_file():
        return CheckResult(False, f"{SCRIPT} was not written")

    files: set[str] = set()
    over_budget = 0
    empty = 0
    failures = 0
    longest = 0
    first_error = ""
    for seed in range(SEEDS):
        result = run_python(SCRIPT, "--seed", str(seed), cwd=workdir, timeout_s=60)
        if result.timed_out or result.returncode != 0:
            failures += 1
            if not first_error:
                first_error = (result.stderr or result.stdout)[-400:]
            continue
        out = result.stdout
        longest = max(longest, len(out))
        if len(out) > BUDGET_CHARS + BUDGET_TOLERANCE:
            over_budget += 1
        lines = [line for line in out.splitlines() if line.strip()]
        if not lines:
            empty += 1
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                files.add(parts[1].strip())

    detail = {
        "seeds": SEEDS,
        "distinct_files": len(files),
        "min_files": MIN_FILES,
        "over_budget_runs": over_budget,
        "empty_runs": empty,
        "failed_runs": failures,
        "longest_output_chars": longest,
        "first_error": first_error,
        "files": sorted(files)[:60],
    }
    if failures > SEEDS // 10:
        return CheckResult(False, f"{failures} of {SEEDS} runs failed", detail)
    if empty > SEEDS // 10:
        return CheckResult(False, f"{empty} of {SEEDS} runs printed nothing", detail)
    if over_budget > SEEDS // 10:
        return CheckResult(
            False,
            f"{over_budget} of {SEEDS} runs blew the {BUDGET_CHARS} character budget",
            detail,
        )
    if len(files) < MIN_FILES:
        return CheckResult(
            False,
            f"reached only {len(files)} of 51 files over {SEEDS} seeds: head bias",
            detail,
        )
    return CheckResult(True, f"reached {len(files)} of 51 files within budget", detail)
