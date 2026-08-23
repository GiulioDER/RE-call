"""The published ATM-Bench harness must stay the program that produced the published run.

`docs/ATM_BENCH.md` points a leaderboard submission at `benchmarks/atm_full_run.py` as the
reproduction pointer for the 2026-08-21 run. That pointer is worth exactly as much as the promise
that the file has not moved since, and a promise kept by memory is not kept: the same repository
has already published a figure carried forward from a report that truncated rather than rounded,
and caught it by hand.

So this suite pins two different things, because they fail in two different ways:

1. **The bytes.** A lint fix, a rename or a "small clarification" makes the published harness a
   different program from the one that ran, silently, and the submission keeps pointing at it.
2. **The binding.** Freezing the file does NOT freeze the library it calls. `master` can drop an
   attribute the frozen harness reads and every test here except the hash pin stays green while
   the published harness becomes unrunnable. That is not hypothetical: `max_dense_score` was
   added on the run's own private branch and had never existed on `master`, so the first thing
   this suite did was fail.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import RetrievalDiagnostics, RetrievalResult
from tests.fakes import FakeEmbedder, FakeStore

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "benchmarks" / "atm_full_run.py"
RECORD = REPO_ROOT / "results" / "atm" / "atm_harness_20260823.json"


def _record() -> dict[str, Any]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _published_files() -> list[tuple[str, dict[str, Any]]]:
    return sorted(_record()["harness_files"].items())


@pytest.mark.parametrize("relative_path", [name for name, _ in _published_files()])
def test_published_harness_is_byte_identical_to_the_run_it_documents(relative_path: str) -> None:
    """The hash pin. One constant, in the artifact, not duplicated into this file.

    The recorded SHA-256 came from the run commit's blob, so a match here is a match with the
    code that produced QS 68.4264, not merely with whatever was committed alongside this test.
    """
    expected = _record()["harness_files"][relative_path]
    raw = (REPO_ROOT / relative_path).read_bytes()

    assert hashlib.sha256(raw).hexdigest() == expected["sha256"], (
        f"{relative_path} has changed since it was published as the ATM-Bench reproduction "
        f"pointer. If the change is deliberate, the submission at "
        f"{_record()['run']['leaderboard_pull_request']} no longer points at the code that ran: "
        f"publish the new file under a new name and leave this one alone, rather than updating "
        f"the hash."
    )
    assert len(raw) == expected["bytes"]


def test_the_publication_record_claims_the_files_match_the_run_commit() -> None:
    """`identical_to_run_commit` is the claim the document repeats; it must not be quietly false.

    Deliberately NOT re-derived from git: commit `6c0ec26b` lives on a branch that was never
    pushed, so `git rev-parse 6c0ec26b:...` succeeds on the machine that ran the benchmark and
    fails in CI and in every clone. The hash above is the portable half of the evidence.
    """
    for name, entry in _published_files():
        assert entry["identical_to_run_commit"] is True, name
        assert entry["git_blob"] == entry["git_blob_at_run_commit"], name


def _keywords_passed_to(call_name: str) -> set[str]:
    """Every keyword the frozen runner passes to `call_name`, read out of its AST.

    Reading the source rather than importing and calling it: the runner's own call sites need a
    database, an embedding provider and an answer provider, none of which a unit test has.
    """
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != call_name:
            continue
        found.update(kw.arg for kw in node.keywords if kw.arg is not None)
    return found


@pytest.mark.parametrize(
    "target, call_name",
    [
        (HybridRetriever.__init__, "HybridRetriever"),
        (PgVectorStore.__init__, "PgVectorStore"),
    ],
)
def test_the_library_still_accepts_what_the_frozen_harness_passes(
    target: Any, call_name: str
) -> None:
    """A parameter rename is a source-compatible refactor everywhere except here."""
    passed = _keywords_passed_to(call_name)
    assert passed, f"no {call_name}(...) call found in {RUNNER.name}; this test has gone vacuous"

    accepted = set(inspect.signature(target).parameters)
    missing = sorted(passed - accepted)

    assert not missing, (
        f"{call_name} no longer accepts {missing}, which the published ATM harness passes. The "
        f"harness is frozen, so this is fixed in the library or the reproduction pointer is dead."
    )


def test_the_library_still_supplies_every_field_the_frozen_harness_reads() -> None:
    """The attribute side of the same problem, and the one that actually fired."""
    reads = {
        RetrievalDiagnostics: {"max_dense_score", "reranking_ran"},
        RetrievalResult: {"hits", "gap_warning", "diagnostics"},
    }
    for owner, fields in reads.items():
        available = {field.name for field in owner.__dataclass_fields__.values()}
        assert fields <= available, f"{owner.__name__} lost {sorted(fields - available)}"


def test_search_reports_the_best_dense_score_it_saw() -> None:
    """`max_dense_score` is populated, not merely declared.

    A default of `None` on a frozen dataclass satisfies the attribute test above while writing a
    column of nulls into `retrieval.jsonl`, which is the failure that looks like a working run.
    """
    store = FakeStore(dense=[("a", 0.91), ("b", 0.44)], sparse=[("c", 0.30)])
    retriever = HybridRetriever(store, FakeEmbedder(), sparse_backend="lexical")

    result = retriever.search("what is x", k=3)

    assert result.diagnostics.max_dense_score == pytest.approx(0.91)


class _PassThroughReranker:
    """`search_fused` refuses to run unreranked, and the reranking is not what is under test."""

    def rerank(self, query: str, hits: list[Any]) -> list[Any]:
        return list(hits)


def test_search_fused_reports_it_too() -> None:
    """The second construction site. The two paths build their diagnostics independently."""
    store = FakeStore(dense=[("a", 0.88)], sparse=[("c", 0.30)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_PassThroughReranker(), sparse_backend="lexical"
    )

    result = retriever.search_fused("what is x", ["earlier turn"], k=3)

    assert result.diagnostics.max_dense_score == pytest.approx(0.88)


def test_an_empty_dense_leg_reports_none_rather_than_raising() -> None:
    """`max(..., default=None)` rather than `max(...)`: a sparse-only corpus must not crash."""
    store = FakeStore(dense=[], sparse=[("c", 0.30)])
    retriever = HybridRetriever(store, FakeEmbedder(), sparse_backend="lexical")

    assert retriever.search("what is x", k=3).diagnostics.max_dense_score is None


def test_the_frozen_files_are_exempt_from_the_style_gates() -> None:
    """The exemption and the freeze have to travel together.

    Without the exemption, `ruff check .` fails on files nobody may fix; with the exemption and
    without the pin, the files are unfrozen and unlinted at once. This asserts the pair, so a
    future cleanup that removes one gets told about the other.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name, _ in _published_files():
        assert f'"{name}"' in pyproject, f"{name} is pinned but not exempt from ruff"
    assert "atm_(full_run|bench)" in pyproject, "the frozen harness is not exempt from mypy"
