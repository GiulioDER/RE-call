"""One rule for every benchmark arm: retrieve through the research seam, never `trusted_search`.

Why this file exists as a SOURCE scan rather than as behavioural tests
---------------------------------------------------------------------

When retrieval began failing closed, `TrustPolicy`'s default became strict, and a benchmark tenant
has no generation and no published calibration — so every arm still calling `trusted_search`
without a policy began raising `TrustRefusal: INDEX_NOT_READY` before retrieval ran. The arm scored
nothing. `benchmarks/systems.py` was fixed in #244; three sibling adapters carried the identical
defect.

Every one of those adapters is unreachable from CI:

* `benchmarks/membench/recall_isolation.py` and `recall_temporal.py` import `membench` at module
  scope, and mem-bench is a SEPARATE repo that `pyproject.toml` deliberately does not depend on
  ("NOT a dependency of this one"). CI installs `.[dev]`, so importing them raises.
* `benchmarks/systems.py`'s integration tests are `@requires_fastembed`, and `fastembed` is not in
  `[dev]` either. That is precisely why #244's breakage survived four days of green pipelines.

A behavioural test for any of them would therefore be a test that CI skips, which is the failure
mode being closed here, reproduced in the guard meant to close it. Reading the source needs no
import, no database, no optional extra, and no network, so it runs everywhere — and it covers arms
that do not exist yet, which is what a per-adapter test cannot do.

Precedent: `tests/test_bench_systems.py::test_the_attribute_path_close_walks_still_exists_in_mem0`
pins mem0's attribute chain by reading its source for the same reason (importing it is too
expensive there, impossible here).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Repo root, resolved from this file so the scan does not depend on the working directory.
_BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"

#: The forbidden import. `recall.trust.trusted_search` applies `TrustPolicy()` — STRICT — to any
#: caller that does not pass a policy, which is correct for a serving path and fatal for a
#: benchmark. `recall.eval._research_trust.research_search` is the seam that exists for harnesses,
#: and its own docstring says so: "If you are writing a *serving* path and you find yourself
#: importing this, that is the bug." This is the converse, and it has no legitimate exception —
#: an arm that deliberately wants to MEASURE strict-mode refusal still goes through
#: `research_search` and passes `policy=` explicitly, which that function supports by design
#: ("A caller that passes its own `policy` keeps it").
_FORBIDDEN = ("recall.trust", "trusted_search")


def _benchmark_modules() -> list[Path]:
    return sorted(p for p in _BENCHMARKS.rglob("*.py") if "__pycache__" not in p.parts)


def _imports_trusted_search(source: str) -> bool:
    """True if this module can reach `recall.trust.trusted_search`, by ANY spelling.

    Parsed, not grepped: a grep for the name matches its own prose, and every one of these modules
    discusses `trusted_search` at length in comments and docstrings. The distinction between
    naming a function and importing it is exactly what a parser gets right and a substring search
    gets wrong, and getting it wrong in the permissive direction would make this guard silent.

    Binding the MODULE counts, not just the function. `from recall import trust` followed by
    `trust.trusted_search(...)` reaches the identical strict-defaulting function, and that is the
    spelling a future adapter is most likely to arrive by — it is already the idiom elsewhere in
    this suite. An earlier version of this detector checked only `from recall.trust import
    trusted_search` and `import recall.trust`, and missed four working routes.

    `importlib.import_module("recall.trust")` is NOT detected and cannot be, short of running the
    module. That is an accepted limit: this guard is here to catch the ordinary omission that has
    now happened four times, not an adversary. Anyone reaching for `importlib` to get past it has
    left the class of mistake this exists to prevent.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            # `from recall.trust import trusted_search` — and `import *`, which binds it too.
            if node.module == _FORBIDDEN[0]:
                if any(alias.name in (_FORBIDDEN[1], "*") for alias in node.names):
                    return True
            # `from recall import trust [as t]` — binds the module, so the function is one
            # attribute access away.
            if node.module == "recall" and any(alias.name == "trust" for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            # `import recall.trust [as rt]`.
            if any(alias.name == _FORBIDDEN[0] for alias in node.names):
                return True
    return False


def _research_search_names(tree: ast.AST) -> set[str]:
    """Every local name bound to `research_search`, including aliases.

    Matching the hardcoded string would let `import research_search as rs` walk straight past,
    which is the same alias trick `_imports_trusted_search` was just hardened against. Two rules
    guarding one mistake should not disagree about how it can be spelled.
    """
    names = {"research_search"}  # the attribute-chain form, `_research_trust.research_search(...)`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "recall.eval._research_trust":
            for alias in node.names:
                if alias.name == "research_search":
                    names.add(alias.asname or alias.name)
    return names


def _research_search_without_calibration(source: str) -> bool:
    """True if this module calls `research_search(...)` without a usable `calibration`.

    The import rule above covers only the POLICY half of a two-part requirement. An arm that
    imports `research_search` and calls it bare gets development mode with `calibration=None`,
    which `recall.trust` reads as "no threshold exists at all": it rewrites every verdict to
    ``unverified`` and forces ``abstained=False``. A verdict-filtering arm then cites nothing and
    scores zero everywhere; an abstention-reading arm reports a dead gate as a perfect one. Both
    are silent. Catching only the import would leave exactly half the defect enforceable.

    A literal ``calibration=None`` counts as MISSING, not as supplied. That distinction is the
    whole rule: `research_search` applies its own policy with `kwargs.setdefault` and does nothing
    at all to the calibration, so writing the keyword with None is byte-for-byte the defect, in the
    spelling that looks like compliance. An earlier draft of this file accepted it, and asserted
    that acceptance in a test whose comment claimed `_trust` would normalise it — which is true
    only for calls that go through `bench_search`, and these are exactly the calls that do not.

    ⚠️ **Limit, and it is load-bearing rather than theoretical.** A calibration passed as a
    VARIABLE cannot be judged here: `benchmarks/beam/systems.py` writes
    ``calibration=self._calibration``, which is None on any run without ``--calibration``, and this
    rule reports it as compliant. That arm discloses its uncalibrated default in its own
    `describe()`, so the gap is documented rather than hidden, but a green scan is not evidence
    that every arm's threshold is live.

    The same hole has one further edge: ``research_search(s, e, q, **kw, calibration=None)`` is
    NOT reported, because the splat is met first and a splatted call is unjudgeable by
    construction. It needs both spellings in one call to hide, so it is a corner of the limit
    above rather than a separate one, and it is written down here for the same reason as the
    rest — a guard's blind spots belong next to the guard, not in a reviewer's memory.
    """
    tree = ast.parse(source)
    names = _research_search_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in names:
            continue
        for kw in node.keywords:
            # `**kwargs` may carry it and the parser cannot see inside, so a splat is treated as
            # satisfied rather than reported as a call this rule cannot judge. `_trust.py`'s own
            # call is the reason that branch exists; `benchmarks/systems.py` also relies on it and
            # is covered behaviourally instead, by `tests/test_bench_systems.py`.
            if kw.arg is None:
                break
            if kw.arg == "calibration":
                if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                    return True  # written, but written as the defect
                break
        else:
            return True
    return False


def test_no_benchmark_arm_retrieves_through_the_strict_default() -> None:
    """A benchmark that calls `trusted_search` with no policy refuses every question.

    Not a style rule. `benchmarks/systems.py` shipped this defect and the entire RE-call arm of
    the published head-to-head returned `TrustRefusal: INDEX_NOT_READY` for every LOCOMO question
    until #244. The three modules this test was written to catch had it too.
    """
    offenders = [
        p.relative_to(_BENCHMARKS.parent).as_posix()
        for p in _benchmark_modules()
        if _imports_trusted_search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "these benchmark modules import `recall.trust.trusted_search`, whose default policy is "
        "STRICT, so every query refuses with INDEX_NOT_READY against an uncalibrated benchmark "
        "corpus. Retrieve through `benchmarks._trust.bench_search` instead: "
        f"{offenders}"
    )


def test_no_benchmark_arm_retrieves_without_an_explicit_calibration() -> None:
    """The other half of the rule, and the half that fails silently rather than loudly.

    A strict refusal is at least an exception someone reads. This half returns a normal-looking
    result whose verdicts have all been blanked and whose abstention flag has been forced, so the
    arm publishes a number that is an artefact of the trust layer being unavailable. Two of the
    four arms filter on `verdict == "ok"` and would score 0.0000 across the board.
    """
    offenders = [
        p.relative_to(_BENCHMARKS.parent).as_posix()
        for p in _benchmark_modules()
        if _research_search_without_calibration(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "these benchmark modules call `research_search` with no `calibration=`, so the trust layer "
        "blanks every verdict to `unverified` and forces `abstained=False`. Retrieve through "
        f"`benchmarks._trust.bench_search` instead: {offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "research_search(store, emb, q, k=10)",
        "research_search(store, emb, q, now=early)",
        "_research_trust.research_search(store, emb, q)",
        # Written, but written AS the defect. `research_search` does nothing to the calibration —
        # only the policy gets a `setdefault` — so this is identical to omitting it, in the
        # spelling that looks like compliance.
        "research_search(store, emb, q, k=5, calibration=None)",
        # Aliased, which the import rule already catches for its own name and this one must too.
        "from recall.eval._research_trust import research_search as rs\nrs(store, emb, q, k=1)",
    ],
)
def test_the_calibration_detector_catches_a_bare_call(source: str) -> None:
    assert _research_search_without_calibration(source)


@pytest.mark.parametrize(
    "source",
    [
        "research_search(store, emb, q, calibration=cal)",
        "research_search(store, emb, q, **kwargs)",
        "bench_search(store, emb, q)",
        "trusted_search(store, emb, q)",  # a different rule's business, not this one's
        "from recall.eval._research_trust import research_search as rs\nrs(s, e, q, calibration=c)",
    ],
)
def test_the_calibration_detector_does_not_fire_on_a_supplied_one(source: str) -> None:
    assert not _research_search_without_calibration(source)


def test_the_scan_actually_reads_the_benchmark_tree() -> None:
    """A guard whose corpus is empty passes forever. Pin that it found real modules.

    `_benchmark_modules()` resolves a path; a rename of `benchmarks/`, or a `rglob` that silently
    matches nothing, would leave `offenders == []` and the rule above permanently green over
    nothing at all.
    """
    modules = {p.name for p in _benchmark_modules()}
    assert len(modules) > 20, f"the benchmark scan found only {len(modules)} modules"
    # The four arms this rule exists for, named so a move renames them here too.
    for expected in ("systems.py", "recall_isolation.py", "recall_temporal.py", "recall_system.py"):
        assert expected in modules, f"{expected} is no longer where this guard looks for it"


@pytest.mark.parametrize(
    "source",
    [
        "from recall.trust import trusted_search",
        "from recall.trust import trusted_search, evaluate",
        "import recall.trust",
        # Every one of these reaches the same strict-defaulting function, and an earlier version of
        # this detector missed all four. `from recall import trust` is not exotic — it is already
        # the idiom at `tests/test_cca_deferred_second_pass.py`, so it is the spelling a future
        # adapter is most likely to arrive by. For three of the four arms this scan is the ONLY
        # protection, because CI cannot import them, so a hole here is not partial coverage.
        "from recall import trust",
        "from recall import trust as t",
        "from recall.trust import *",
        "import recall.trust as rt",
    ],
)
def test_the_detector_catches_every_spelling_of_the_import(source: str) -> None:
    """The guard must fail on the real thing, or it is decorative."""
    assert _imports_trusted_search(source)


@pytest.mark.parametrize(
    "source",
    [
        "from recall.eval._research_trust import research_search",
        "from benchmarks._trust import bench_search",
        '"""A docstring that mentions trusted_search and recall.trust at length."""',
        "# a comment about trusted_search\nx = 1",
        'x = "trusted_search"',
        "from recall.trust_policy import TrustPolicy",  # a NEIGHBOURING module, not the forbidden one
    ],
)
def test_the_detector_does_not_fire_on_mentions_or_neighbours(source: str) -> None:
    """And it must not fire on prose, or every one of these heavily-commented modules trips it."""
    assert not _imports_trusted_search(source)
