# Claim→Artifact and Self-Ablation Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a number from being published without a committed artifact behind it, and stop a benchmark run whose mechanism under test is provably inert.

**Architecture:** Two independent guards, both ported from the CCA `/audit-fix` pipeline. Guard 1 is a pytest gate over four documents: numbers carry an HTML-comment marker resolving to a `results/*.json` key, and today's unmarked numbers sit in a shrink-only baseline. Guard 2 is a `$0` retrieval-only preflight that disables each configured mechanism, re-retrieves, and refuses the run if the output does not change.

**Tech Stack:** Python 3.10+, pytest, stdlib only for guard 1 (`re`, `json`, `ast`, `operator`, `dataclasses`). Guard 2 uses the existing `recall.retriever.HybridRetriever`.

**Spec:** [`docs/superpowers/specs/2026-07-29-claim-artifact-and-arm-differs-guards-design.md`](../specs/2026-07-29-claim-artifact-and-arm-differs-guards-design.md)

## Global Constraints

- Branch: `guards/claim-artifact-and-arm-differs`, already created off `master` at `728ccd3`.
- **mypy runs over `benchmarks/` with `disallow_untyped_defs = true`** (`pyproject.toml:189-190`). Every function added under `benchmarks/` and `recall/` needs full annotations, including `-> None`.
- `ruff check .` must pass.
- **No `eval()` or `compile()` anywhere in this work.** The `derived:` marker is evaluated by walking the AST and applying `operator` functions to literal nodes. An expression evaluator that reaches `eval` is a code-execution path inside a documentation gate.
- **Guard 1 code lives in `benchmarks/`, not `recall/`.** The wheel ships `packages = ["recall", "recall_mcp"]` (`pyproject.toml:150`); a documentation gate must not be shipped to users. `benchmarks/` is the repo's established home for dev-only, type-checked, test-imported modules.
- **Guard 1 must not require a database.** It runs in CI alongside the offline tests. No `PgVectorStore`, no network, no API keys.
- **Read files with newline normalisation** (`read_text(encoding="utf-8").replace("\r\n", "\n")`). This repo is developed on Windows and CI runs Linux; `CLAIMS_BASELINE.json` is a generated artifact and must come out identical on both.
- Existing constants to reuse, not redefine: `recall.retriever.DEFAULT_CANDIDATE_K = 20`, `benchmarks.systems.DEFAULT_K = 5`.
- Commit after every task. Conventional-commit subjects (`feat:`, `test:`, `docs:`).

---

## File Structure

| File | Responsibility |
|---|---|
| `benchmarks/claim_gate.py` (create) | Scan a markdown document into `Claim` records; parse markers; resolve a claim against a `results/*.json` artifact. Pure functions, no I/O beyond reading the files named. |
| `results/WITHDRAWN.json` (create) | The retracted-figure registry. |
| `results/CLAIMS_BASELINE.json` (create, generated) | Frozen per-document multiset of today's unmarked numbers. |
| `scripts/generate_claims_baseline.py` (create) | One-shot generator for the baseline. Kept so the file can be regenerated on the OS that consumes it. |
| `tests/test_published_numbers_have_artifacts.py` (create) | The gate itself, plus every detection-path test. |
| `recall/eval/arm_check.py` (create) | `Verdict`, `ablation_verdicts`, `enforce`. Shipped, because it is eval machinery a user re-running the benchmark needs. |
| `tests/test_arm_check.py` (create) | Detection-path tests for guard 2, against stub retrievers. |
| `benchmarks/systems.py` (modify) | Add `RecallSystem.ablation_preflight`. |
| `benchmarks/run.py` (modify) | `--allow-inert-arm`, `--ablation-sample`, preflight call, provenance stamping. |
| `benchmarks/beam/run.py` (modify) | Same three, on the BEAM harness. |

---

## Task 1: Claim scanner — markers, exclusions, extraction

**Files:**
- Create: `benchmarks/claim_gate.py`
- Test: `tests/test_published_numbers_have_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Marker`, `Claim`, `EXCLUSIONS`, `GATED_DOCS`, `REPO_ROOT`, `RESULTS_ROOT`, `mask_excluded(text) -> str`, `scan_text(text, doc) -> list[Claim]`, `scan_document(path) -> list[Claim]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_published_numbers_have_artifacts.py`:

```python
"""The claim gate: a number in a published document must resolve to a committed artifact.

`results/ARTIFACTS.md` and `test_results_artifact_provenance.py` already enforce the other
direction — an artifact must declare what it is. Nothing stopped a number appearing in a document
that no artifact contains, which is how three defects reached publication on 2026-07-29: a loss
published as a tie, a figure derivable from nothing, and a count that contradicted its own summary.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.claim_gate import Claim, Marker, RESULTS_ROOT, scan_text


def test_a_bare_decimal_is_an_unmarked_claim() -> None:
    claims = scan_text("the shipped reranker reaches 0.777 overall.", doc="x.md")
    assert [(c.text, c.marker) for c in claims] == [("0.777", None)]


def test_a_bare_integer_is_an_unmarked_claim() -> None:
    """The `usable: 1` beside a published `n=17` defect was an integer."""
    claims = scan_text("The clean subset holds 17 records.", doc="x.md")
    assert [c.text for c in claims] == ["17"]


def test_an_artifact_marker_binds_to_the_number_before_it() -> None:
    claims = scan_text(
        "reaches **0.777** <!--@ locomo_rerank/rerank_shipped.json # depth_curve.5.overall.hit -->",
        doc="x.md",
    )
    assert len(claims) == 1
    marker = claims[0].marker
    assert marker is not None
    assert marker.kind == "artifact"
    assert marker.artifact == "locomo_rerank/rerank_shipped.json"
    assert marker.key == "depth_curve.5.overall.hit"


def test_citation_pending_derived_and_withdrawn_markers_parse() -> None:
    text = (
        "a 0.467 <!--@ citation-pending: no artifact retains this -->\n"
        "b 0.106 <!--@ derived: 0.777 - 0.671 -->\n"
        "c 0.945 <!--@ withdrawn: README withdrawn list -->\n"
    )
    kinds = [c.marker.kind for c in scan_text(text, doc="x.md") if c.marker]
    assert kinds == ["citation-pending", "derived", "withdrawn"]


def test_excluded_spans_hide_their_digits() -> None:
    text = (
        "code `k = 5` and ```\nblock 3.14\n``` and https://example.com/9.9 and "
        "v0.7.0 and 2026-07-29 and #1987 and hit@5 and bge-large-en-v1.5 and 2026"
    )
    assert scan_text(text, doc="x.md") == []


def test_line_numbers_are_one_based() -> None:
    claims = scan_text("nothing here\nbut 0.33 here\n", doc="x.md")
    assert [c.line for c in claims] == [2]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.claim_gate'`

- [ ] **Step 3: Write the implementation**

Create `benchmarks/claim_gate.py`:

```python
"""Scan a published markdown document into numeric claims and their citation markers.

The gate this feeds is a port of CCA's `NUM-*` rule: a numeric finding carries an execution
artifact or it is escalated, because a wrong number reads exactly as fluently as a right one.
Here the artifact is a committed `results/*.json` and the escalation is an explicit
`citation-pending` marker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

#: The documents that carry load-bearing numbers a reader could act on. Process records
#: (PREREGISTRATION, REVIEW, ARTICLE_DRAFT, CHANGELOG, docs/*.md) are deliberately absent: they are
#: not published results, and a guard that ships mostly-exempted stays that way.
GATED_DOCS: tuple[str, ...] = (
    "results/RESULTS.md",
    "results/FINDINGS.md",
    "README.md",
    "benchmarks/SUITE-DESIGN.md",
)

#: Spans masked before numbers are extracted.
#:
#: THIS TABLE IS THE GUARD'S SOFT SPOT. Every row is a place a real claim could hide — a count
#: written inside a code span is invisible to this gate. It is kept in one place, with a stated
#: reason per row, so it can be audited as a set rather than accumulating silently.
EXCLUSIONS: tuple[tuple[str, str, int], ...] = (
    ("fenced code — code, not prose", r"^```.*?^```", re.MULTILINE | re.DOTALL),
    ("inline code — code, not prose", r"`[^`\n]*`", 0),
    ("url — path and query components", r"https?://\S+", 0),
    ("html comment — the marker is the citation", r"<!--.*?-->", re.DOTALL),
    ("iso date — timestamp", r"\d{4}-\d{2}-\d{2}", 0),
    ("semver — identifier", r"\bv?\d+\.\d+\.\d+\b", 0),
    ("model version suffix — identifier", r"-v\d+(?:\.\d+)?\b", 0),
    ("issue ref — identifier", r"#\d+", 0),
    ("section ref — identifier", r"§\s*\d+[a-z]?", 0),
    ("metric depth suffix — identifier, e.g. hit@5", r"@\d+", 0),
    ("retrieval budget — configuration, e.g. k=5", r"\b[kn]\s*=\s*\d+", 0),
    ("year — timestamp", r"\b(?:19|20)\d{2}\b", 0),
    ("ordered list marker — structure", r"^\s{0,3}\d+\.\s", re.MULTILINE),
    ("table rule — structure", r"^\s*\|[\s:|-]+\|\s*$", re.MULTILINE),
    ("footnote — structure", r"\[\^\d+\]", 0),
)

MARKER_RE = re.compile(
    r"<!--@\s*(?:"
    r"(?P<pending>citation-pending)\s*:\s*(?P<pending_note>[^>]*?)"
    r"|(?P<derived>derived)\s*:\s*(?P<expr>[^>]*?)"
    r"|(?P<withdrawn>withdrawn)\s*:\s*(?P<ref>[^>]*?)"
    r"|(?P<artifact>[\w./-]+\.json)\s*#\s*(?P<key>[\w.-]+)"
    r")\s*-->"
)

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Marker:
    """A citation attached to one published number."""

    kind: str  # "artifact" | "citation-pending" | "derived" | "withdrawn"
    artifact: str | None = None
    key: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Claim:
    """One number as it appears in a document. `text` is the LITERAL digit string as published."""

    doc: str
    line: int
    text: str
    marker: Marker | None


def mask_excluded(text: str) -> str:
    """Replace every excluded span with spaces of equal length, preserving all offsets."""
    masked = text
    for _reason, pattern, flags in EXCLUSIONS:
        masked = re.sub(pattern, lambda m: " " * (m.end() - m.start()), masked, flags=flags)
    return masked


def _marker_from(match: re.Match[str]) -> Marker:
    if match.group("pending"):
        return Marker(kind="citation-pending", note=(match.group("pending_note") or "").strip())
    if match.group("derived"):
        return Marker(kind="derived", note=(match.group("expr") or "").strip())
    if match.group("withdrawn"):
        return Marker(kind="withdrawn", note=(match.group("ref") or "").strip())
    return Marker(kind="artifact", artifact=match.group("artifact"), key=match.group("key"))


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_of(offset: int, line_starts: list[int]) -> int:
    """1-based line number for a character offset."""
    low, high = 0, len(line_starts) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if line_starts[mid] <= offset:
            low = mid
        else:
            high = mid - 1
    return low + 1


def _marker_for(
    number_end: int, line: int, markers: list[tuple[int, Marker]], line_starts: list[int]
) -> Marker | None:
    """The first marker starting after `number_end` on the same line, if any."""
    for start, marker in markers:
        if start >= number_end and _line_of(start, line_starts) == line:
            return marker
    return None


def scan_text(text: str, doc: str) -> list[Claim]:
    """Every numeric claim in `text`, each bound to the marker that follows it on its line.

    A marker binds to the NEAREST PRECEDING number on the same line. Two numbers on one line need
    two markers; that is deliberate, because a single marker covering both would let one of them
    drift unchecked.
    """
    masked = mask_excluded(text)
    line_starts = _line_starts(text)
    markers: list[tuple[int, Marker]] = [
        (m.start(), _marker_from(m)) for m in MARKER_RE.finditer(text)
    ]
    claims: list[Claim] = []
    for number in NUMBER_RE.finditer(masked):
        line = _line_of(number.start(), line_starts)
        claims.append(
            Claim(
                doc=doc,
                line=line,
                text=number.group(0),
                marker=_marker_for(number.end(), line, markers, line_starts),
            )
        )
    return claims


def scan_document(path: Path) -> list[Claim]:
    """Scan a repo-relative document. Newlines are normalised so Windows and Linux agree."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return scan_text(text, doc=str(path).replace("\\", "/"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`
Expected: 6 passed.

- [ ] **Step 5: Type-check and lint**

Run: `cd ~/Documents/recall && mypy benchmarks/claim_gate.py && ruff check benchmarks/claim_gate.py`
Expected: `Success: no issues found in 1 source file`, no ruff output.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/claim_gate.py tests/test_published_numbers_have_artifacts.py
git commit -m "feat(claim-gate): scan published docs into numeric claims and markers"
```

---

## Task 2: Artifact resolution and the match rule

**Files:**
- Modify: `benchmarks/claim_gate.py`
- Test: `tests/test_published_numbers_have_artifacts.py`

**Interfaces:**
- Consumes: `Claim`, `Marker` from Task 1.
- Produces: `ClaimError`, `lookup(payload, key) -> object`, `matches(published, actual) -> bool`, `resolve(claim, results_root) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_published_numbers_have_artifacts.py`:

```python
import json

import pytest

from benchmarks.claim_gate import ClaimError, matches, resolve


def test_match_rule_rounds_to_the_published_precision() -> None:
    assert matches("0.777", 0.77714)
    assert matches("0.78", 0.7771)
    assert matches("17", 17)


def test_match_rule_rejects_the_suite_design_defect() -> None:
    """SUITE-DESIGN published 0.533 where the cell is 0.536 — a loss printed as a tie."""
    assert not matches("0.533", 0.536)


def test_match_rule_rejects_a_non_number() -> None:
    assert not matches("17", "17")
    assert not matches("1", True)


def _write_artifact(root: Path, payload: dict) -> None:
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "sub" / "a.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_accepts_a_matching_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path, {"depth": {"5": {"hit": 0.7771}}})
    claim = Claim("x.md", 1, "0.777", Marker("artifact", artifact="sub/a.json", key="depth.5.hit"))
    resolve(claim, tmp_path)  # does not raise


def test_resolve_rejects_a_mismatching_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path, {"depth": {"5": {"hit": 0.536}}})
    claim = Claim("x.md", 1, "0.533", Marker("artifact", artifact="sub/a.json", key="depth.5.hit"))
    with pytest.raises(ClaimError, match="0.536"):
        resolve(claim, tmp_path)


def test_resolve_rejects_a_missing_artifact(tmp_path: Path) -> None:
    claim = Claim("x.md", 1, "0.777", Marker("artifact", artifact="sub/missing.json", key="a"))
    with pytest.raises(ClaimError, match="no such artifact"):
        resolve(claim, tmp_path)


def test_resolve_rejects_a_missing_key(tmp_path: Path) -> None:
    _write_artifact(tmp_path, {"depth": {}})
    claim = Claim("x.md", 1, "0.777", Marker("artifact", artifact="sub/a.json", key="depth.5.hit"))
    with pytest.raises(ClaimError, match="no key"):
        resolve(claim, tmp_path)


def test_citation_pending_needs_a_reason(tmp_path: Path) -> None:
    resolve(Claim("x.md", 1, "0.467", Marker("citation-pending", note="no artifact")), tmp_path)
    with pytest.raises(ClaimError, match="reason"):
        resolve(Claim("x.md", 1, "0.467", Marker("citation-pending", note="")), tmp_path)


def test_withdrawn_needs_a_retraction_reference(tmp_path: Path) -> None:
    resolve(Claim("x.md", 1, "0.945", Marker("withdrawn", note="README list")), tmp_path)
    with pytest.raises(ClaimError, match="retraction"):
        resolve(Claim("x.md", 1, "0.945", Marker("withdrawn", note="")), tmp_path)


def test_derived_checks_the_arithmetic(tmp_path: Path) -> None:
    resolve(Claim("x.md", 1, "0.106", Marker("derived", note="0.777 - 0.671")), tmp_path)
    with pytest.raises(ClaimError, match="derived"):
        resolve(Claim("x.md", 1, "0.200", Marker("derived", note="0.777 - 0.671")), tmp_path)


def test_derived_refuses_anything_that_is_not_literal_arithmetic(tmp_path: Path) -> None:
    """The evaluator walks the AST and applies `operator` functions. It must not reach names,
    calls, attributes or subscripts — a documentation gate is not a place to execute code."""
    for hostile in ("__import__('os').getcwd()", "open('x')", "a + 1", "[1][0]"):
        with pytest.raises(ClaimError, match="literal arithmetic|does not parse"):
            resolve(Claim("x.md", 1, "1.0", Marker("derived", note=hostile)), tmp_path)


def test_an_unmarked_claim_does_not_resolve(tmp_path: Path) -> None:
    with pytest.raises(ClaimError, match="unmarked"):
        resolve(Claim("x.md", 1, "0.777", None), tmp_path)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`
Expected: FAIL — `ImportError: cannot import name 'ClaimError'`

- [ ] **Step 3: Write the implementation**

Append to `benchmarks/claim_gate.py` (and add `import ast`, `import json`, `import operator`, `from typing import Any, Callable` to the imports at the top):

```python
class ClaimError(Exception):
    """A published number that does not resolve. The gate raises this and reports every one."""


def lookup(payload: Any, key: str) -> Any:
    """Walk a dotted path into a decoded JSON payload. JSON object keys are always strings, so a
    numeric level like `depth_curve.5` works without special-casing."""
    node = payload
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ClaimError(f"no key {key!r} in artifact")
        node = node[part]
    return node


def matches(published: str, actual: object) -> bool:
    """True when `actual` rounds to exactly the digits published.

    Rounding to the PUBLISHED precision, rather than comparing floats within a tolerance, is what
    catches a cell printed as 0.533 when the artifact holds 0.536: the published string says how
    precisely the author claimed to know it.
    """
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    if "." in published:
        decimals = len(published.split(".", 1)[1])
        return f"{float(actual):.{decimals}f}" == published
    return float(actual).is_integer() and str(int(actual)) == published


#: Binary operators the `derived:` marker may use. Anything else is refused.
_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node: ast.AST, expression: str) -> float:
    """Evaluate one literal-arithmetic AST node.

    Deliberately NOT `eval`/`compile`: this walks the tree and applies `operator` functions, so
    there is no code-execution path at all — not a validated one, none. A documentation gate that
    can run arbitrary code is a worse problem than the one it solves.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ClaimError(f"derived expression {expression!r} is not literal arithmetic")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, expression)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _eval_node(node.left, expression), _eval_node(node.right, expression)
        )
    raise ClaimError(f"derived expression {expression!r} is not literal arithmetic")


def _evaluate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ClaimError(f"derived expression {expression!r} does not parse") from exc
    return _eval_node(tree.body, expression)


def resolve(claim: Claim, results_root: Path) -> None:
    """Raise `ClaimError` unless this claim is backed as its marker promises."""
    marker = claim.marker
    if marker is None:
        raise ClaimError(f"{claim.doc}:{claim.line} {claim.text} is unmarked")

    if marker.kind == "citation-pending":
        if not (marker.note or "").strip():
            raise ClaimError(
                f"{claim.doc}:{claim.line} citation-pending needs a reason — a figure with no "
                f"artifact AND no stated reason is just an unmarked number"
            )
        return

    if marker.kind == "withdrawn":
        if not (marker.note or "").strip():
            raise ClaimError(
                f"{claim.doc}:{claim.line} withdrawn needs a retraction reference — a retracted "
                f"number with no pointer to its retraction is just a wrong number"
            )
        return

    if marker.kind == "derived":
        value = _evaluate(marker.note or "")
        if not matches(claim.text, value):
            raise ClaimError(
                f"{claim.doc}:{claim.line} {claim.text} is not the derived value of "
                f"{marker.note!r} ({value})"
            )
        return

    if marker.artifact is None or marker.key is None:
        raise ClaimError(f"{claim.doc}:{claim.line} artifact marker is missing its path or key")
    path = results_root / marker.artifact
    if not path.is_file():
        raise ClaimError(f"{claim.doc}:{claim.line} no such artifact: {marker.artifact}")
    actual = lookup(json.loads(path.read_text(encoding="utf-8")), marker.key)
    if not matches(claim.text, actual):
        raise ClaimError(
            f"{claim.doc}:{claim.line} published {claim.text} but "
            f"{marker.artifact}#{marker.key} holds {actual}"
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`
Expected: all passed.

- [ ] **Step 5: Type-check and lint**

Run: `cd ~/Documents/recall && mypy benchmarks/claim_gate.py && ruff check benchmarks/claim_gate.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/claim_gate.py tests/test_published_numbers_have_artifacts.py
git commit -m "feat(claim-gate): resolve markers against artifacts, rounding to published precision"
```

---

## Task 3: The withdrawn registry

**Files:**
- Create: `results/WITHDRAWN.json`
- Modify: `benchmarks/claim_gate.py`
- Test: `tests/test_published_numbers_have_artifacts.py`

**Interfaces:**
- Consumes: `Claim`, `ClaimError` from Tasks 1-2.
- Produces: `load_withdrawn(results_root) -> dict[str, dict[str, str]]`, `check_withdrawn(claims, withdrawn) -> list[ClaimError]`.

**Populating the registry — read carefully.** `README.md:137-176` is the withdrawn list. Only the *retracted* figures go in the registry; the *corrections* published alongside them are live results and must NOT be listed. The retracted values are `0.945` (real-corpus recall@5), `0.615` (LOCOMO hit@5 pre-fix anchor), and from `results/FINDINGS.md` §9a the withdrawn pool-100 `0.5957`. Values such as `0.33`, `0.705`, `0.624` and `0.798` appear in the same paragraphs but are the **live** replacements — listing them would make the gate demand a withdrawn marker on current results.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_published_numbers_have_artifacts.py`:

```python
from benchmarks.claim_gate import check_withdrawn, load_withdrawn


def test_a_withdrawn_value_may_not_appear_bare() -> None:
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    errors = check_withdrawn([Claim("x.md", 3, "0.945", None)], withdrawn)
    assert len(errors) == 1
    assert "withdrawn" in str(errors[0])


def test_a_withdrawn_value_passes_with_a_withdrawn_marker() -> None:
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim("x.md", 3, "0.945", Marker("withdrawn", note="README withdrawn list"))
    assert check_withdrawn([claim], withdrawn) == []


def test_a_withdrawn_value_passes_when_legitimately_re_measured() -> None:
    """Same digits, arrived at from a committed artifact — a different figure that reads the same."""
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim("x.md", 3, "0.945", Marker("artifact", artifact="a.json", key="hit"))
    assert check_withdrawn([claim], withdrawn) == []


def test_a_citation_pending_marker_does_not_excuse_a_withdrawn_figure() -> None:
    """"We have not sourced it yet" is not the same statement as "this was retracted"."""
    withdrawn = {"0.945": {"figure": "real-corpus recall@5", "retraction_ref": "README"}}
    claim = Claim("x.md", 3, "0.945", Marker("citation-pending", note="later"))
    assert len(check_withdrawn([claim], withdrawn)) == 1


def test_the_registry_on_disk_is_well_formed() -> None:
    withdrawn = load_withdrawn(RESULTS_ROOT)
    assert withdrawn, "an empty registry would make the withdrawn rule vacuous"
    for value, entry in withdrawn.items():
        assert value == value.strip()
        assert entry["figure"].strip()
        assert entry["retraction_ref"].strip()


def test_the_registry_holds_literal_digit_strings_not_floats() -> None:
    """0.615 and its artifact's 0.6152 are different strings; float comparison would conflate a
    retracted figure with a live one at a different precision."""
    for value in load_withdrawn(RESULTS_ROOT):
        assert isinstance(value, str)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -k withdrawn -v`
Expected: FAIL — `ImportError: cannot import name 'check_withdrawn'`

- [ ] **Step 3: Create the registry**

Create `results/WITHDRAWN.json`:

```json
{
  "_note": "Retracted figures. A value here may never appear bare in a gated document and may never sit in CLAIMS_BASELINE.json. Values are the LITERAL digit string as published, not floats.",
  "0.945": {
    "figure": "real-corpus recall@5 — used document headings as queries, which is known-item retrieval",
    "retraction_ref": "README.md 'Claims that were withdrawn'"
  },
  "0.615": {
    "figure": "LOCOMO hit@5 pre-fix anchor — checkable since #111, still not evidence for the HNSW-build-noise claim it was used for",
    "retraction_ref": "README.md 'Claims that were withdrawn'; results/FINDINGS.md 9a"
  },
  "0.5957": {
    "figure": "pool-100 depth column — measured on a doubled corpus",
    "retraction_ref": "results/FINDINGS.md 9a retraction notice; results/wrrf/FINDINGS_pool100_contamination.md"
  }
}
```

- [ ] **Step 4: Write the implementation**

Append to `benchmarks/claim_gate.py`:

```python
def load_withdrawn(results_root: Path) -> dict[str, dict[str, str]]:
    """The retracted-figure registry, minus its `_note` header."""
    payload = json.loads((results_root / "WITHDRAWN.json").read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def check_withdrawn(
    claims: list[Claim], withdrawn: dict[str, dict[str, str]]
) -> list[ClaimError]:
    """A retracted figure may never appear bare.

    It passes only with a `withdrawn:` marker (it is being discussed AS a retraction) or with an
    artifact marker (the same digits arrived from a committed artifact, and are therefore a
    different figure that happens to read the same). `citation-pending` does NOT excuse it: "not
    sourced yet" and "this was retracted" are different statements about a number.

    This is the document-side counterpart of deleting `postfix_pool100.json` outright rather than
    annotating it — an annotated wrong number in a table is still a number someone can read off it.
    """
    errors: list[ClaimError] = []
    for claim in claims:
        if claim.text not in withdrawn:
            continue
        kind = claim.marker.kind if claim.marker else None
        if kind in {"withdrawn", "artifact"}:
            continue
        errors.append(
            ClaimError(
                f"{claim.doc}:{claim.line} {claim.text} is a withdrawn figure "
                f"({withdrawn[claim.text]['figure']}) and carries no withdrawn marker"
            )
        )
    return errors
```

- [ ] **Step 5: Run to verify they pass**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add results/WITHDRAWN.json benchmarks/claim_gate.py tests/test_published_numbers_have_artifacts.py
git commit -m "feat(claim-gate): withdrawn-figure registry, so a retraction cannot re-enter a table"
```

---

## Task 4: Baseline generation and the ratchet

**Files:**
- Create: `scripts/generate_claims_baseline.py`
- Create: `results/CLAIMS_BASELINE.json` (generated)
- Modify: `benchmarks/claim_gate.py`
- Test: `tests/test_published_numbers_have_artifacts.py`

**Interfaces:**
- Consumes: `scan_document`, `GATED_DOCS`, `load_withdrawn` from Tasks 1-3.
- Produces: `unmarked_counts(claims) -> dict[str, int]`, `build_baseline() -> dict[str, dict[str, int]]`, `load_baseline(results_root) -> dict[str, dict[str, int]]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_published_numbers_have_artifacts.py`:

```python
from benchmarks.claim_gate import build_baseline, load_baseline, unmarked_counts

#: The ratchet. This number may only ever go DOWN. Lower it when you mark a number; a change that
#: raises it is a change that added an uncited number to a published document.
MAX_BASELINE_ENTRIES = 0  # replaced in Step 4 with the generated total


def test_unmarked_counts_ignores_marked_numbers() -> None:
    claims = [
        Claim("x.md", 1, "0.33", None),
        Claim("x.md", 2, "0.33", None),
        Claim("x.md", 3, "0.77", Marker("citation-pending", note="why")),
    ]
    assert unmarked_counts(claims) == {"0.33": 2}


def test_every_baseline_entry_is_still_present_and_still_unmarked() -> None:
    """Dead-entry test. Marking a number forces deleting its baseline row, so the file shrinks."""
    assert load_baseline(RESULTS_ROOT) == build_baseline(), (
        "CLAIMS_BASELINE.json no longer matches the documents. If you MARKED a number, remove its "
        "row here and lower MAX_BASELINE_ENTRIES. If you ADDED an unmarked number, mark it instead."
    )


def test_the_baseline_never_grows() -> None:
    total = sum(sum(counts.values()) for counts in load_baseline(RESULTS_ROOT).values())
    assert total <= MAX_BASELINE_ENTRIES


def test_no_withdrawn_value_hides_in_the_baseline() -> None:
    """The known-bad figures may not sit in the ratchet — that is where they would be invisible."""
    withdrawn = set(load_withdrawn(RESULTS_ROOT))
    for doc, counts in load_baseline(RESULTS_ROOT).items():
        assert not (withdrawn & set(counts)), f"{doc} baselines a withdrawn figure"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -k baseline -v`
Expected: FAIL — `ImportError: cannot import name 'build_baseline'`

- [ ] **Step 3: Write the implementation**

Append to `benchmarks/claim_gate.py`:

```python
def unmarked_counts(claims: list[Claim]) -> dict[str, int]:
    """How many times each unmarked number appears. Keyed by the literal digit string, NOT by line
    number — line numbers drift under any prose edit and would make the baseline unmaintainable."""
    counts: dict[str, int] = {}
    for claim in claims:
        if claim.marker is None:
            counts[claim.text] = counts.get(claim.text, 0) + 1
    return dict(sorted(counts.items()))


def build_baseline() -> dict[str, dict[str, int]]:
    """The current unmarked-number multiset across every gated document."""
    return {doc: unmarked_counts(scan_document(Path(doc))) for doc in GATED_DOCS}


def load_baseline(results_root: Path) -> dict[str, dict[str, int]]:
    payload = json.loads((results_root / "CLAIMS_BASELINE.json").read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if not key.startswith("_")}
```

Create `scripts/generate_claims_baseline.py`:

```python
"""Regenerate `results/CLAIMS_BASELINE.json`.

Run this ONLY when deliberately re-freezing the baseline, and LOOK AT THE SIZE OF THE DIFF before
committing: a regeneration that silently drops entries turns the ratchet into a rubber stamp. The
generator normalises newlines, so it produces an identical file on Windows and Linux.
"""
from __future__ import annotations

import json
from typing import Any

from benchmarks.claim_gate import RESULTS_ROOT, build_baseline


def main() -> int:
    baseline = build_baseline()
    total = sum(sum(counts.values()) for counts in baseline.values())
    payload: dict[str, Any] = {
        "_note": (
            "Frozen multiset of unmarked numbers per gated document. This file may only SHRINK. "
            "Mark a number and remove its row; never add a row."
        )
    }
    payload.update(baseline)
    path = RESULTS_ROOT / "CLAIMS_BASELINE.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} - {total} unmarked numbers across {len(baseline)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the baseline and pin the ratchet**

Run: `cd ~/Documents/recall && python -m scripts.generate_claims_baseline`
Expected: `wrote .../CLAIMS_BASELINE.json - N unmarked numbers across 4 documents`

Take the printed `N` and set `MAX_BASELINE_ENTRIES = N` in the test file, replacing the `0`.

**Sanity-check the diff before continuing.** Run `git diff --stat results/CLAIMS_BASELINE.json` and confirm the file is substantial — it should hold hundreds of entries across four documents totalling ~2,977 lines of prose. A near-empty baseline means the exclusion table is eating real prose, not that the documents are clean. Report the number either way.

- [ ] **Step 5: Run to verify they pass**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`
Expected: all passed. If `test_no_withdrawn_value_hides_in_the_baseline` fails, that is a REAL finding: a retracted figure is sitting bare in a published document. Fix it by adding a `<!--@ withdrawn: ... -->` marker at that site, then regenerate the baseline and lower `MAX_BASELINE_ENTRIES`.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add scripts/generate_claims_baseline.py results/CLAIMS_BASELINE.json benchmarks/claim_gate.py tests/test_published_numbers_have_artifacts.py
git commit -m "feat(claim-gate): freeze today's unmarked numbers into a shrink-only baseline"
```

---

## Task 5: Arm the gate over the four documents

**Files:**
- Modify: `tests/test_published_numbers_have_artifacts.py`
- Modify: `results/RESULTS.md`, `results/FINDINGS.md`, `README.md`, `benchmarks/SUITE-DESIGN.md` (markers only — no numbers changed)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: the live gate.

- [ ] **Step 1: Write the gate test**

Append to `tests/test_published_numbers_have_artifacts.py`:

```python
from benchmarks.claim_gate import GATED_DOCS, scan_document


@pytest.mark.parametrize("doc", GATED_DOCS)
def test_every_marked_claim_resolves(doc: str) -> None:
    """A marker that does not resolve is worse than no marker: it reads as verified."""
    failures: list[str] = []
    for claim in scan_document(Path(doc)):
        if claim.marker is None:
            continue
        try:
            resolve(claim, RESULTS_ROOT)
        except ClaimError as exc:
            failures.append(str(exc))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("doc", GATED_DOCS)
def test_no_new_unmarked_numbers(doc: str) -> None:
    baseline = load_baseline(RESULTS_ROOT).get(doc, {})
    current = unmarked_counts(scan_document(Path(doc)))
    changed = {
        value: count for value, count in current.items() if count != baseline.get(value, 0)
    }
    assert not changed, (
        f"{doc}: these numbers are new or changed since the baseline: {changed}. Add a marker — "
        f"`<!--@ <artifact>.json # <key> -->`, or `<!--@ citation-pending: <reason> -->` if no "
        f"artifact retains it — and lower MAX_BASELINE_ENTRIES."
    )


@pytest.mark.parametrize("doc", GATED_DOCS)
def test_no_bare_withdrawn_figures(doc: str) -> None:
    errors = check_withdrawn(scan_document(Path(doc)), load_withdrawn(RESULTS_ROOT))
    assert not errors, "\n".join(str(e) for e in errors)
```

- [ ] **Step 2: Run and fix real violations**

Run: `cd ~/Documents/recall && pytest tests/test_published_numbers_have_artifacts.py -v`

`test_no_new_unmarked_numbers` should pass immediately — the baseline was generated from these exact documents in Task 4. The two that may fail are `test_no_bare_withdrawn_figures` and `test_every_marked_claim_resolves`. Both are real findings. Fix each by editing the document to add the correct marker.

**Do not change any published number to make a test pass.** If a number turns out to be wrong, that is a separate finding to report to the user — not a repair to make inside a green-the-suite loop. A wrong number edited into a right one is indistinguishable in the final diff from a citation you added.

- [ ] **Step 3: Mark the 0.467 explicitly**

The spec names this one: our `0.467` is not derivable from any committed artifact. Find every site:

```bash
cd ~/Documents/recall && grep -n "0\.467" results/RESULTS.md results/FINDINGS.md README.md benchmarks/SUITE-DESIGN.md
```

At each site, append:

```markdown
<!--@ citation-pending: no committed artifact retains this cell; re-derive or retract -->
```

Then remove `0.467` from its `results/CLAIMS_BASELINE.json` rows and lower `MAX_BASELINE_ENTRIES` by the number of occurrences removed. This is the ratchet working as designed: a known-bad figure leaves the baseline and becomes an explicit, visible pending citation.

- [ ] **Step 4: Run the full suite**

Run: `cd ~/Documents/recall && pytest -q`
Expected: green, and **no test in this file skipped**. A skip in a gating job is an absent guard.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/recall
git add tests/test_published_numbers_have_artifacts.py results/ README.md benchmarks/SUITE-DESIGN.md
git commit -m "feat(claim-gate): arm the gate over RESULTS, FINDINGS, README and SUITE-DESIGN"
```

---

## Task 6: The self-ablation module

**Files:**
- Create: `recall/eval/arm_check.py`
- Test: `tests/test_arm_check.py`

**Interfaces:**
- Consumes: `recall.retriever.HybridRetriever`.
- Produces: `DEFAULT_SAMPLE`, `METRIC_CLASSES`, `InertArmError`, `Verdict(mechanism: str, verdict: str, sampled: int, differing: int)` with `.as_dict()`, `ablation_verdicts(store, embedder, questions, *, k, candidate_k, reranker, use_sparse) -> list[Verdict]`, `enforce(verdicts, *, metric_class, allow_inert) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arm_check.py`:

```python
"""The self-ablation preflight: disable the mechanism, re-retrieve, require the output to differ.

This is CCA's red-state proof transposed. There, a test claimed as proof of a fix is re-run against
the PRE-fix code and must fail; a test that passes both ways proves nothing. Here, an arm claiming
to measure a mechanism is re-run with the mechanism OFF and must return something different; an arm
that returns the same thing either way measured nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from recall.eval.arm_check import InertArmError, Verdict, ablation_verdicts, enforce


def test_enforce_passes_when_every_mechanism_differs() -> None:
    enforce([Verdict("reranker", "DIFFERS", 25, 19)], metric_class="set", allow_inert=False)


def test_enforce_blocks_identical_on_any_metric_class() -> None:
    verdicts = [Verdict("reranker", "IDENTICAL", 25, 0)]
    with pytest.raises(InertArmError, match="IDENTICAL"):
        enforce(verdicts, metric_class="set", allow_inert=False)
    with pytest.raises(InertArmError, match="IDENTICAL"):
        enforce(verdicts, metric_class="ranked", allow_inert=False)


def test_enforce_blocks_set_identical_only_for_set_metrics() -> None:
    """Same ids, different order: inert for hit@k, live for a rank-sensitive metric."""
    verdicts = [Verdict("reranker", "SET_IDENTICAL", 25, 0)]
    with pytest.raises(InertArmError, match="SET_IDENTICAL"):
        enforce(verdicts, metric_class="set", allow_inert=False)
    enforce(verdicts, metric_class="ranked", allow_inert=False)


def test_allow_inert_lets_it_through() -> None:
    enforce([Verdict("reranker", "IDENTICAL", 25, 0)], metric_class="set", allow_inert=True)


def test_enforce_rejects_an_unknown_metric_class() -> None:
    """The caller declares its metric class; inference would hand a new harness the permissive
    branch by default."""
    with pytest.raises(ValueError, match="metric_class"):
        enforce([], metric_class="whatever", allow_inert=False)


def test_verdicts_are_json_serialisable() -> None:
    """They are stamped into the artifact's `_provenance`, so they must survive json.dumps."""
    payload = [Verdict("reranker", "DIFFERS", 25, 19).as_dict()]
    assert json.loads(json.dumps(payload)) == [
        {"mechanism": "reranker", "verdict": "DIFFERS", "sampled": 25, "differing": 19}
    ]


@dataclass
class _Chunk:
    id: str
    text: str = ""
    source: str = ""


@dataclass
class _Hit:
    chunk: _Chunk
    score: float = 0.5
    indexed_at: object = None


class _StubStore:
    """Returns a fixed dense list and a fixed sparse list, sliced to the requested k."""

    def __init__(self, dense: list[str], sparse: list[str]) -> None:
        self._dense = dense
        self._sparse = sparse

    def query_dense(self, qvec: object, k: int, source: object = None) -> list[_Hit]:
        return [_Hit(_Chunk(cid), 1.0 - i / 100) for i, cid in enumerate(self._dense[:k])]

    def query_sparse(
        self, query: str, k: int, source: object = None, vec: object = None
    ) -> list[_Hit]:
        return [_Hit(_Chunk(cid), 0.5) for cid in self._sparse[:k]]

    def newest_indexed_at(self) -> object:
        return None


class _StubEmbedder:
    dim = 3
    name = "stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 1.0] for _ in texts]


class _ReversingReranker:
    """Reverses the pool — changes the SET only when the pool is wider than k."""

    def rerank(self, query: str, hits: list[_Hit]) -> list[_Hit]:
        return list(reversed(hits))


class _NoOpReranker:
    def rerank(self, query: str, hits: list[_Hit]) -> list[_Hit]:
        return list(hits)


def test_reranker_over_a_wide_pool_differs() -> None:
    store = _StubStore(dense=[f"d{i}" for i in range(20)], sparse=[])
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20,
        reranker=_ReversingReranker(), use_sparse=False,
    )
    assert [v.verdict for v in verdicts] == ["DIFFERS"]


def test_reranker_on_a_pool_equal_to_k_cannot_change_the_set() -> None:
    """candidate_k == k on a dense-only arm: the pool IS the answer, so reranking reorders it but
    cannot change which ids survive. This is the documented inert-reranker case."""
    store = _StubStore(dense=[f"d{i}" for i in range(5)], sparse=[])
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=5,
        reranker=_ReversingReranker(), use_sparse=False,
    )
    assert [v.verdict for v in verdicts] == ["SET_IDENTICAL"]


def test_a_reranker_that_changes_nothing_is_identical() -> None:
    store = _StubStore(dense=[f"d{i}" for i in range(20)], sparse=[])
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20,
        reranker=_NoOpReranker(), use_sparse=False,
    )
    assert [v.verdict for v in verdicts] == ["IDENTICAL"]


def test_a_sparse_leg_that_contributes_nothing_is_identical() -> None:
    """The sparse leg was silently inert for a whole artifact generation, pre-#81/#84."""
    dense = [f"d{i}" for i in range(20)]
    store = _StubStore(dense=dense, sparse=dense)  # sparse returns exactly what dense returns
    verdicts = ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20, reranker=None, use_sparse=True,
    )
    assert [(v.mechanism, v.verdict) for v in verdicts] == [("sparse", "IDENTICAL")]


def test_no_mechanisms_configured_yields_no_verdicts() -> None:
    store = _StubStore(dense=[f"d{i}" for i in range(20)], sparse=[])
    assert ablation_verdicts(
        store, _StubEmbedder(), ["q"], k=5, candidate_k=20, reranker=None, use_sparse=False,
    ) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Documents/recall && pytest tests/test_arm_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'recall.eval.arm_check'`

- [ ] **Step 3: Write the implementation**

Create `recall/eval/arm_check.py`:

```python
"""Refuse a benchmark run whose mechanism under test provably changes nothing.

The rule of thumb "candidate_k == k renders the reranker inert" is exactly true only when the
REALIZED fused pool equals k: `HybridRetriever` reranks the whole fused pool and truncates to k
afterwards, and a hybrid pool can reach 2 * candidate_k. So inertness is measured at runtime rather
than asserted from configuration — which also catches inertness nobody predicted.

Retrieval only: no generator, no judge, so this costs nothing and runs ahead of all LLM spend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from recall.retriever import HybridRetriever

#: Questions sampled by default. Deterministic (the first N of the caller's list, no RNG), so a
#: preflight verdict is reproducible for a given slice.
DEFAULT_SAMPLE = 25

METRIC_CLASSES = ("set", "ranked")


class InertArmError(RuntimeError):
    """The arm under test does not differ from the arm with its mechanism disabled."""


@dataclass(frozen=True)
class Verdict:
    """One mechanism's ablation result."""

    mechanism: str
    verdict: str  # "DIFFERS" | "SET_IDENTICAL" | "IDENTICAL"
    sampled: int
    differing: int

    def as_dict(self) -> dict[str, Any]:
        """For stamping into an artifact's `_provenance`."""
        return {
            "mechanism": self.mechanism,
            "verdict": self.verdict,
            "sampled": self.sampled,
            "differing": self.differing,
        }


def _ids(retriever: HybridRetriever, query: str, k: int) -> list[str]:
    return [hit.chunk.id for hit in retriever.search(query, k=k).hits]


def _compare(baseline: list[list[str]], ablated: list[list[str]]) -> tuple[str, int]:
    """Aggregate per-question comparisons into one verdict plus a differing count."""
    set_differs = sum(1 for a, b in zip(baseline, ablated) if set(a) != set(b))
    if set_differs:
        return "DIFFERS", set_differs
    order_differs = sum(1 for a, b in zip(baseline, ablated) if a != b)
    if order_differs:
        return "SET_IDENTICAL", order_differs
    return "IDENTICAL", 0


def ablation_verdicts(
    store: Any,
    embedder: Any,
    questions: Sequence[str],
    *,
    k: int,
    candidate_k: int,
    reranker: Any = None,
    use_sparse: bool = True,
) -> list[Verdict]:
    """One verdict per CONFIGURED mechanism. A mechanism that is off yields no verdict."""
    baseline = HybridRetriever(
        store, embedder, reranker=reranker, use_sparse=use_sparse, candidate_k=candidate_k
    )
    base_ids = [_ids(baseline, q, k) for q in questions]

    verdicts: list[Verdict] = []
    if reranker is not None:
        without_rerank = HybridRetriever(
            store, embedder, reranker=None, use_sparse=use_sparse, candidate_k=candidate_k
        )
        verdict, differing = _compare(base_ids, [_ids(without_rerank, q, k) for q in questions])
        verdicts.append(Verdict("reranker", verdict, len(questions), differing))
    if use_sparse:
        without_sparse = HybridRetriever(
            store, embedder, reranker=reranker, use_sparse=False, candidate_k=candidate_k
        )
        verdict, differing = _compare(base_ids, [_ids(without_sparse, q, k) for q in questions])
        verdicts.append(Verdict("sparse", verdict, len(questions), differing))
    return verdicts


def enforce(verdicts: Sequence[Verdict], *, metric_class: str, allow_inert: bool) -> None:
    """Raise `InertArmError` when a configured mechanism is inert for the metric being reported.

    `metric_class` is DECLARED by the caller, not inferred: inference would hand a new harness the
    permissive branch by default, which is the failure mode this guard exists to prevent.
    """
    if metric_class not in METRIC_CLASSES:
        raise ValueError(f"metric_class must be one of {METRIC_CLASSES}, got {metric_class!r}")
    if allow_inert:
        return
    blocking = {"IDENTICAL"} if metric_class == "ranked" else {"IDENTICAL", "SET_IDENTICAL"}
    bad = [v for v in verdicts if v.verdict in blocking]
    if bad:
        detail = "; ".join(f"{v.mechanism}={v.verdict} over {v.sampled} questions" for v in bad)
        raise InertArmError(
            f"arm is inert for a '{metric_class}' metric: {detail}. The run would measure nothing. "
            f"Widen candidate_k, fix the mechanism, or pass --allow-inert-arm to record it "
            f"deliberately."
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd ~/Documents/recall && pytest tests/test_arm_check.py -v`
Expected: 12 passed.

If `test_reranker_on_a_pool_equal_to_k_cannot_change_the_set` returns `DIFFERS` rather than
`SET_IDENTICAL`, do **not** adjust the expectation to match. Read `recall/retriever.py:105-125`
and work out which pool the retriever actually built — the test encodes the spec's claim about
inertness, and a disagreement here is a finding about the premise, not about the test.

- [ ] **Step 5: Type-check and lint**

Run: `cd ~/Documents/recall && mypy recall/eval/arm_check.py && ruff check recall/eval/arm_check.py tests/test_arm_check.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add recall/eval/arm_check.py tests/test_arm_check.py
git commit -m "feat(arm-check): self-ablation verdicts — disable the mechanism, require a difference"
```

---

## Task 7: Wire the preflight into the LOCOMO harness

**Files:**
- Modify: `benchmarks/systems.py` (add `RecallSystem.ablation_preflight` after `retrieve`, ~line 245)
- Modify: `benchmarks/run.py` (CLI ~line 430, ingest loop ~line 475, payload ~line 505)

**Interfaces:**
- Consumes: `ablation_verdicts`, `enforce`, `DEFAULT_SAMPLE` from Task 6.
- Produces: `RecallSystem.ablation_preflight(questions, *, sample, metric_class, allow_inert) -> list[dict[str, Any]]`.

- [ ] **Step 1: Add the adapter method**

In `benchmarks/systems.py`, inside `class RecallSystem`, immediately after `retrieve`:

```python
    def ablation_preflight(
        self,
        questions: list[str],
        *,
        sample: int,
        metric_class: str,
        allow_inert: bool,
    ) -> list[dict[str, Any]]:
        """Refuse the run if a configured mechanism changes nothing. Retrieval only — no LLM spend.

        Opens its own store exactly as `retrieve` does, because the store is per-call here and the
        preflight must query the index this arm will actually use.
        """
        from recall.eval.arm_check import DEFAULT_SAMPLE, ablation_verdicts, enforce
        from recall.retriever import DEFAULT_CANDIDATE_K
        from recall.store import PgVectorStore

        if self._tenant is None:
            raise RuntimeError("RecallSystem.ablation_preflight() called before ingest()")
        sampled = questions[: sample or DEFAULT_SAMPLE]
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
        ) as store:
            verdicts = ablation_verdicts(
                store,
                self._embedder,
                sampled,
                k=self._k,
                candidate_k=DEFAULT_CANDIDATE_K,
                reranker=self._reranker,
                use_sparse=True,
            )
        enforce(verdicts, metric_class=metric_class, allow_inert=allow_inert)
        return [v.as_dict() for v in verdicts]
```

`DEFAULT_CANDIDATE_K` is used because `RecallSystem.retrieve` calls `trusted_search` without a
`candidate_k`, so that is the pool this arm actually retrieves at. If that ever changes, this must
change with it or the preflight measures a different configuration from the run.

- [ ] **Step 2: Add the CLI flags**

In `benchmarks/run.py`, immediately after the `--reranker` argument block:

```python
    p.add_argument(
        "--ablation-sample",
        type=_positive_int,
        default=25,
        help=(
            "questions sampled by the inert-arm preflight (default 25). Taken deterministically "
            "from the head of the question list, so the verdict is reproducible for a slice."
        ),
    )
    p.add_argument(
        "--allow-inert-arm",
        action="store_true",
        help=(
            "record an inert mechanism instead of refusing the run. The override AND every verdict "
            "are stamped into the artifact, so a run let through cannot read as clean afterwards."
        ),
    )
```

- [ ] **Step 3: Call the preflight**

In `benchmarks/run.py`, declare the accumulator next to `outcomes: list[Outcome] = []`:

```python
    ablation: list[dict[str, Any]] = []
```

Then inside `for position, conv in enumerate(convs):`, between `system.ingest(conv)` and
`conv_outcomes, _ = run_arm(...)`:

```python
        if position == 0 and args.arm == "recall":
            # After index build, before the FIRST generator call: retrieval-only, so an inert arm
            # is caught before a single token is spent. BEAM best-config ran out of credits at
            # 5/60; a post-hoc check would have spent them first.
            ablation = system.ablation_preflight(
                [str(q["question"]) for q in conv_questions],
                sample=args.ablation_sample,
                metric_class="set",
                allow_inert=args.allow_inert_arm,
            )
            print(f"ablation preflight: {ablation}", flush=True)
```

- [ ] **Step 4: Stamp it into the artifact**

In `benchmarks/run.py`, after `payload = _results_payload(...)` and before `path.write_text(...)`:

```python
    payload["ablation_preflight"] = {
        "verdicts": ablation,
        "allow_inert_arm": bool(args.allow_inert_arm),
        "sample": args.ablation_sample,
    }
```

- [ ] **Step 5: Verify**

Run: `cd ~/Documents/recall && python -m benchmarks.run --help`
Expected: `--ablation-sample` and `--allow-inert-arm` appear.

Run: `cd ~/Documents/recall && mypy benchmarks/run.py benchmarks/systems.py && ruff check benchmarks/run.py benchmarks/systems.py`
Expected: clean.

Run: `cd ~/Documents/recall && pytest tests/test_bench_run.py tests/test_bench_systems.py -q`
Expected: green. If a test asserts the exact set of artifact top-level keys, update it to include
`ablation_preflight` and say so in the commit message — that is a contract change, not a fixup.

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/recall
git add benchmarks/run.py benchmarks/systems.py tests/
git commit -m "feat(bench): refuse a LOCOMO run whose mechanism is provably inert"
```

---

## Task 8: Wire the preflight into the BEAM harness

**Files:**
- Modify: `benchmarks/beam/run.py` (CLI ~line 436, construction ~line 627)

**Interfaces:**
- Consumes: `ablation_verdicts`, `enforce` from Task 6; the same flag names as Task 7.
- Produces: nothing new — the same preflight on the second harness.

- [ ] **Step 1: Read the two call sites before editing**

Run: `cd ~/Documents/recall && sed -n '220,270p' benchmarks/beam/systems.py && sed -n '430,450p;615,640p' benchmarks/beam/run.py`

Note `benchmarks/beam/systems.py:239` — `self._candidate_k = max(candidate_k if candidate_k is not None else k, k)`. This clamp is what makes `candidate_k == k` reachable by default, and it is exactly the configuration the preflight exists to catch. **Do not change the clamp.** The guard reports the consequence; silently widening the pool would change published behaviour under cover of adding a guard.

- [ ] **Step 2: Add the same two CLI flags**

In `benchmarks/beam/run.py`, next to the existing `--reranker` argument, add the identical `--ablation-sample` and `--allow-inert-arm` blocks from Task 7 Step 2. Use the same flag names and the same help text so the two harnesses do not diverge.

- [ ] **Step 3: Call the preflight after index build**

In `benchmarks/beam/run.py`, after the system is constructed with `candidate_k=args.candidate_k` and after its corpus is indexed, call the preflight with `metric_class="set"` (BEAM reports nugget coverage, a set metric), sampling `args.ablation_sample` questions from the head of the BEAM question list. Follow Task 7 Step 3's shape exactly:

```python
    ablation = system.ablation_preflight(
        [str(q["question"]) for q in questions],
        sample=args.ablation_sample,
        metric_class="set",
        allow_inert=args.allow_inert_arm,
    )
    print(f"ablation preflight: {ablation}", flush=True)
```

If the BEAM system class has no `ablation_preflight`, add one to `benchmarks/beam/systems.py`
mirroring Task 7 Step 1, but passing `candidate_k=self._candidate_k` (BEAM's system stores its own
clamped pool size, unlike the LOCOMO adapter which retrieves at `DEFAULT_CANDIDATE_K`).

- [ ] **Step 4: Verify**

Run: `cd ~/Documents/recall && python -m benchmarks.beam.run --help`
Expected: both new flags appear.

Run: `cd ~/Documents/recall && mypy benchmarks/beam/run.py benchmarks/beam/systems.py && ruff check benchmarks/beam/ && pytest tests/test_bench_beam_candidate_k.py tests/test_bench_beam_cutoff_and_coverage.py -q`
Expected: clean and green.

- [ ] **Step 5: Full suite**

Run: `cd ~/Documents/recall && pytest -q && ruff check . && mypy`
Expected: all green.

- [ ] **Step 6: Commit and open the PR**

```bash
cd ~/Documents/recall
git add benchmarks/beam/
git commit -m "feat(beam): same inert-arm preflight on the BEAM harness"
git push -u origin guards/claim-artifact-and-arm-differs
gh pr create --title "Two CCA-ported guards: claim-to-artifact, and self-ablation" --body-file docs/superpowers/specs/2026-07-29-claim-artifact-and-arm-differs-guards-design.md
```

---

## Verification checklist

Before claiming this done, run this and paste the output:

```bash
cd ~/Documents/recall && pytest -q && ruff check . && mypy
```

Then confirm each by inspection, not by assumption:

- [ ] `results/CLAIMS_BASELINE.json` holds hundreds of entries, not a handful. A near-empty baseline means the exclusion table is eating real prose, and the guard would be a rubber stamp. Report the actual count.
- [ ] No test in `tests/test_published_numbers_have_artifacts.py` or `tests/test_arm_check.py` is skipped. A skip in a gating job is an absent guard.
- [ ] `0.467` carries a `citation-pending` marker and is absent from the baseline.
- [ ] `MAX_BASELINE_ENTRIES` equals the generated total.
- [ ] `git diff --stat` on the four gated documents shows marker additions only — **no published number changed value**. If one did, stop and report it.
- [ ] No `eval(` or `compile(` anywhere in the diff: `git diff | grep -nE '\b(eval|compile)\('` returns nothing.
