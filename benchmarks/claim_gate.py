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
    # `n=` is DELIBERATELY absent from this row: it is a sample size, a claim about the data, not
    # configuration — the design spec's motivating defect was exactly a `results/gap/summary.json`
    # reading `usable: 1` beside a published `n=17` that this table's old combined `[kn]` pattern
    # made invisible to the gate. Only `k=` (a retrieval budget the caller chose) is excluded here.
    # The `(?:,\d{3})*` tail matters: without it `k=1,536` masks only `k=1` and leaves `,536` for
    # NUMBER_RE, which then publishes `536` as a claim — the mask leaking a fragment of the number
    # it exists to hide. No gated document writes a grouped `k=` today; the tail keeps it that way.
    ("retrieval budget — configuration, e.g. k=5", r"\bk\s*=\s*\d+(?:,\d{3})*", 0),
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

#: Comma-grouped integer (e.g. `1,536`) tried FIRST so a triplet-grouped sample size scans as one
#: token instead of shredding at each comma into `1` / `536` — an edit from `n=1,536` to `n=2,536`
#: must be a visible change to the gate, not an invisible one hiding behind an untouched `536`.
#: Order is load-bearing: Python's `re` alternation tries branches left-to-right at each position
#: and commits to the first one that matches, so the plain-digit fallback must come second or it
#: would win first and consume only the leading digits before the first comma.
#:
#: Space-grouped numbers (`n=1 982`) are NOT recognised as one token — deliberately punted rather
#: than guessed at, because a space also separates two genuinely distinct numbers and there is no
#: local rule that tells the two cases apart.
#:
#: Measure the hole rather than describing it as hypothetical: **21 space-grouped numbers** are
#: published across the gated documents today (RESULTS 9, FINDINGS 8, README 4) against 47
#: comma-grouped ones — and the SAME figure appears both ways in all three, `1 536` beside
#: `1,536`. So one published sample size is a single claim in one sentence and `1` + `536` in
#: another. That is the size of what this branch chose not to solve.
#:
#: An optional leading sign is captured into the token: `results/FINDINGS.md` and
#: `results/RESULTS.md` print negative deltas as `−0.512` (Unicode minus, U+2212), never ASCII
#: `-`; the reproduction that motivated this fix used ASCII `-`. Both are handled here — handling
#: only one would leave the other silently unmarked.
#:
#: The regex itself just consumes an optional `-`/`−` immediately ahead of the digits, exactly
#: like the comma-grouping and decimal parts beside it — no lookbehind, no context. Whether a
#: consumed sign is a REAL sign (and not, say, the `-` in a hyphenated range or a hyphenated
#: identifier) is decided AFTER matching, in `scan_text`, against the ORIGINAL unmasked `text` —
#: not against `masked`, and not in this pattern. That split is load-bearing, not a style choice:
#: an earlier version of this fix put the disqualifying lookbehind directly in the regex, run
#: against `masked`. `EXCLUSIONS`'s year row replaces `2026` with four spaces, so in `masked`
#: `2026-07` becomes `    -07` — the character immediately before that `-` is now a SPACE, not
#: the digit `6` it actually is in the document, and a lookbehind checked against `masked` reads
#: the exposed `-` as a genuine sign, turning a bare year-month fragment into a fabricated
#: negative claim. Masking replaces characters, it does not remove them, so `text[start - 1]` — the
#: ORIGINAL character, checked in `scan_text` — still sees the `6` and correctly disqualifies it.
#: The rule itself: a sign is real when the character immediately before it is neither an ASCII
#: digit nor an ASCII letter. That is what tells `-0.065` (preceded by a space) apart from
#: `0.36-0.43` (preceded by the digit `6` — a hyphenated range, not a sign) and from `gpt-4` or
#: `a-1` (preceded by a letter). A disqualified sign is not dropped silently: the claim keeps the
#: digits, unsigned, starting one character later — exactly the token this regex would have
#: produced before this fix existed.
#:
#: Stated limit: a `-` preceded by ANOTHER `-` still qualifies as a sign under this rule (the
#: character before it is punctuation, not alnum) — a markdown anchor slug like
#: `#quickstart--2-minutes...` (rendered from an em dash in the source heading) reads as `-2`.
#: `EXCLUSIONS` has no row for bare `<a href="#...">` fragments (only `https?://` URLs are
#: excluded), so this was already an unmasked, uncited digit before this fix — sign capture only
#: changes its REPRESENTATION, from an unsigned baseline entry to a signed one. It is not a real
#: claim in either form, and both read as inert once baselined.
NUMBER_RE = re.compile(r"[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−]?\d+(?:\.\d+)?")

#: Characters `NUMBER_RE` may consume as a leading sign. Kept alongside the pattern so
#: `_disqualify_sign`'s ASCII-alnum lookback rule (see `scan_text`) stays visibly paired with what
#: the regex actually treats as a sign.
_SIGN_CHARS = "-−"


def _sign_is_real(text: str, start: int) -> bool:
    """True when the sign character `NUMBER_RE` consumed at `start` is a genuine sign.

    Checked against the ORIGINAL `text`, never `masked` — see the note on `NUMBER_RE` for why
    that distinction is load-bearing. Nothing before `start` (start of document or start of line,
    since a newline is not alnum either) counts as a qualifying predecessor, so a sign at the very
    top of a line is real.
    """
    if start == 0:
        return True
    before = text[start - 1]
    return not (before.isascii() and before.isalnum())


@dataclass(frozen=True)
class Marker:
    """A citation attached to one published number."""

    kind: str  # "artifact" | "citation-pending" | "derived" | "withdrawn"
    artifact: str | None = None
    key: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Claim:
    """One number as it appears in a document. `text` is the LITERAL digit string as published,

    including a leading sign when the document prints one (`-0.065` or `−0.065` — see the sign
    rule documented on `NUMBER_RE`). An unsigned claim has no leading sign character at all;
    there is no separate "positive" marker.
    """

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
    # (start, end, literal claim text) — `start`/`text` are adjusted below when a consumed sign
    # turns out not to be real; `end` never changes, since a disqualified sign is dropped from the
    # FRONT of the token, not the back.
    numbers: list[tuple[int, int, str]] = []
    for number in NUMBER_RE.finditer(masked):
        start, end, value = number.start(), number.end(), number.group(0)
        if value[0] in _SIGN_CHARS and not _sign_is_real(text, start):
            start, value = start + 1, value[1:]
        numbers.append((start, end, value))
    number_starts = [start for start, _end, _value in numbers]
    claims: list[Claim] = []
    for start, end, value in numbers:
        line = _line_of(start, line_starts)
        other_starts = [s for s in number_starts if s != start]
        claims.append(
            Claim(
                doc=doc,
                line=line,
                text=value,
                marker=_marker_for(end, line, markers, line_starts, other_starts),
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
    # Comma grouping is a formatting choice ("1,536" vs "1536"), not part of the published
    # precision — strip it before comparing digits so a comma-grouped sample size matches the
    # plain int an artifact holds. The Unicode minus is likewise a formatting choice for the same
    # sign ("−0.512" vs "-0.512") — normalise it to ASCII so it lines up with the "-" that
    # `f"{float(actual):...}"` always produces for a negative Python float below.
    normalized = published.replace(",", "").replace("−", "-")
    if "." in normalized:
        decimals = len(normalized.split(".", 1)[1])
        return f"{float(actual):.{decimals}f}" == normalized
    return float(actual).is_integer() and str(int(actual)) == normalized


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
    # `MARKER_RE`'s artifact path is `[\w./-]+\.json`, which permits `../`. This cannot fabricate
    # a claim by itself — the value still has to match below — but without this check a marker
    # could cite a real file outside `results/` that was never committed as a result. Resolve
    # before comparing so a symlink or `..` segment cannot land outside `results_root`.
    resolved_root = results_root.resolve()
    path = (results_root / marker.artifact).resolve()
    if not path.is_relative_to(resolved_root):
        raise ClaimError(
            f"{claim.doc}:{claim.line} artifact path escapes results_root: {marker.artifact!r}"
        )
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
