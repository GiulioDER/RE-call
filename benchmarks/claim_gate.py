"""Scan a published markdown document into numeric claims and their citation markers.

The gate this feeds is a port of CCA's `NUM-*` rule: a numeric finding carries an execution
artifact or it is escalated, because a wrong number reads exactly as fluently as a right one.
Here the artifact is a committed `results/*.json` and the escalation is an explicit
`citation-pending` marker.

Prior work: searched `docs_search(source_type='memory', query='claim gate numeric claims markdown
scanner artifact citation marker')` before writing this file. No prior implementation of this
document-side scanner exists. Nearest hits were `project-cca-numeric-oracle-2026-07-21.md` (CCA's
own internal `numeric` claim type for arithmetic FINDINGS during audit review — a different
system, not a markdown-claim-vs-artifact scanner) and
`reference-recall-docs-evidence-tier-convention-2026-07-26.md` (the evidence-tier convention this
gate enforces, already the design's stated basis). This module and its design were reviewed as
part of an already-approved spec (`.git/sdd/`), so this is Task 1 of that plan, not a fresh
experiment.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    ("fenced code — code, not prose", r"```.*?```", re.DOTALL),
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
    number_end: int,
    line: int,
    markers: list[tuple[int, Marker]],
    line_starts: list[int],
    other_number_starts: list[int],
) -> Marker | None:
    """The first marker starting after `number_end` on the same line, if no OTHER number's

    occurrence lies strictly between `number_end` and that marker's start — otherwise the marker
    binds to that nearer number instead, not to this one.
    """
    for start, marker in markers:
        if start < number_end or _line_of(start, line_starts) != line:
            continue
        if any(number_end < other_start < start for other_start in other_number_starts):
            return None
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
    numbers = list(NUMBER_RE.finditer(masked))
    number_starts = [number.start() for number in numbers]
    claims: list[Claim] = []
    for number in numbers:
        line = _line_of(number.start(), line_starts)
        other_starts = [start for start in number_starts if start != number.start()]
        claims.append(
            Claim(
                doc=doc,
                line=line,
                text=number.group(0),
                marker=_marker_for(number.end(), line, markers, line_starts, other_starts),
            )
        )
    return claims


def scan_document(path: Path) -> list[Claim]:
    """Scan a repo-relative document. Newlines are normalised so Windows and Linux agree."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return scan_text(text, doc=str(path).replace("\\", "/"))


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
