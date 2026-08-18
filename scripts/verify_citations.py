"""Verify that every `path:line` source citation in the markdown documentation still points at
the code it claims.

WHY THIS EXISTS, AND WHY IT IS NOT AN EXISTENCE CHECK.

Line numbers drift faster than a document can be written. Measured 2026-08-18: while
`docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` and its pre-registration were being written and merged,
three separate concurrent merges to master moved cited lines. One commit adding 197 lines to
`recall/index.py` moved 13 of 13 checked citations; a later merge moved another; #375 moved
`recall/embedding_registry.py:223` to `:228`. Each was caught by hand, and each was caught only
because someone happened to re-check. A merged document asserting `file.py:123 does X` when line
123 is a closing paren is a silent correctness defect.

The first version of this verifier reported "41 citations, 0 broken" because every cited line
existed. Three of those citations still pointed at the wrong code: a bare `try:`, a docstring's
closing quotes, and a tenant-mismatch error rather than the production identity gate it was
claimed to be. **An existence check is not a correctness check.** A line number that is merely
in range carries no information at all, because every line number below the file's length is in
range, including all the wrong ones.

So this verifies an ANCHOR. The convention it enforces is the one the documentation already
follows without being told to: a citation is written next to the identifier it is about, in
backticks.

    `SERVABLE_ACTIVE_STATES = frozenset({"ready", "active"})` (`recall/control_plane.py:34`)

`SERVABLE_ACTIVE_STATES` is the anchor. If the citation is right, that token is in the same scope
as line 34. If a merge moves the definition, the token leaves that scope and the citation is
reported STALE together with the line the anchor is on NOW, so the fix is mechanical.

The verdicts:

  OK            an anchor sits in the enclosing scope of the cited line (see `scope_span`).
  STALE         a distinctive anchor exists in the cited file, but not in that scope. FAILS.
  UNVERIFIABLE  no distinctive anchor could be derived from the prose, or the only one that fits
                is too common to certify. Reported, never silently passed, and held to a
                committed ceiling so the count can only fall.
  FROZEN        a stale citation in `docs/archive/`, printed but not failed: an archive records
                what was true when it was written.
  EXTERNAL      a document whose paths are in somebody else's repository.

UNVERIFIABLE is not a pass. It is the honest answer for a citation this check cannot decide, and
it is the reason the tool does not report a comforting "0 broken": that number was a lie the
first time it was printed. It is also not a loophole, because the ceiling is a ratchet, so a
citation that goes stale in a way the anchor rule cannot diagnose still fails the build by
raising the count.

WHAT THIS DOES NOT CATCH, measured rather than assumed. A citation that drifts but stays inside
the same function is not detected: a 40-line displacement is caught in 40% of cases, which
falsified a pre-registered prediction of >=90%. Displaced out of its enclosing scope, 70.9% of
citations are caught (61 of 86, measured 2026-08-18). So this finds the structural staleness that
motivated it -- a cited line that has become a closing paren, an `import`, a blank line, or part
of a different function -- and does not pretend to more. Full numbers, including the falsified
prediction, are in docs/preregistrations/2026-08-18-citation-anchor-verification.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Documentation that gets scanned. Citations elsewhere (docstrings, comments) are out of scope:
#: they sit beside the code they describe and move with it under most edits, whereas a document
#: in `docs/` is edited on a different clock from the file it cites, which is the drift this
#: check is about.
DOC_GLOB = "docs/**/*.md"

#: A cited path must start with one of these. This is what distinguishes a source citation from
#: a prose mention of a directory, and it deliberately excludes `docs/` itself: a doc citing a
#: doc by line number is a different problem with a different fix.
CITED_ROOTS = (
    "recall",
    "recall_mcp",
    "recall_interop",
    "recall_consistency",
    "tests",
    "benchmarks",
    "scripts",
)

#: How far from the cited line an anchor may sit when no enclosing scope can be computed -- a
#: non-Python file, or a Python file that does not parse. Three lines each way absorbs a
#: decorator or a signature that wraps, without being wide enough to make a moved definition look
#: present.
#:
#: A fixed window is the WRONG unit whenever a scope is available, and both directions of wrong
#: were measured on this corpus before the scope rule existed. Too tight: `recall/index.py:447`
#: cites a line inside `_index_fingerprint` whose name is 27 lines up at the `def`, and a citation
#: into a function body is a claim about that function. Too loose: `recall/frontmatter.py:12` is
#: an `import` line and the keys it is cited for are 5 lines below, which a window of 10 would
#: have called fine. No window separates those two, because they are the same distance apart.
#: The enclosing scope does separate them, which is why it is the unit and this is the fallback.
ANCHOR_WINDOW = 3

#: An anchor token must occur on at most this many lines of the cited file. This is the whole
#: strength of the check. A token that appears on 200 lines tells you nothing when it turns up
#: inside a 7-line window, so matching it would manufacture a green. Tokens above the threshold
#: are discarded as anchor candidates, which pushes the citation towards UNVERIFIABLE -- the
#: correct answer -- rather than towards a false OK.
MAX_ANCHOR_LINES = 6

#: Minimum length of an anchor token. Below this a token is almost never distinctive, and the
#: occurrence threshold would be doing all the work on an identifier like `db` or `id`.
MIN_ANCHOR_LENGTH = 4

#: Tokens common enough in Python or in prose that their presence near a line is not evidence of
#: anything. The occurrence threshold above already removes most of these in most files; this
#: list covers the small file where `None` genuinely appears only three times.
ANCHOR_STOPWORDS = frozenset({
    "None", "True", "False", "self", "cls", "def", "class", "return", "raise", "import",
    "from", "with", "async", "await", "yield", "assert", "lambda", "pass", "elif", "else",
    "while", "break", "continue", "except", "finally", "global", "nonlocal", "print",
    "str", "int", "bool", "dict", "list", "set", "tuple", "float", "bytes", "type", "object",
    "Any", "Optional", "Union", "List", "Dict", "Tuple", "Callable", "Iterable", "Sequence",
    "this", "that", "then", "than", "they", "them", "there", "these", "those", "when", "which",
    "what", "will", "would", "should", "could", "have", "here", "into", "only", "same",
    "does", "done", "case", "code", "line", "file", "path", "name", "text", "data", "test",
    "tests", "value", "values", "note", "also", "each", "both", "over", "under", "after",
    "before", "because", "still", "never", "always", "every", "some", "more", "most", "less",
    "very", "just", "like", "such", "used", "uses", "make", "made", "must", "none", "true",
    "false",
})

#: A full citation: a path under one of the cited roots, then `:line` or `:start-end`, optionally
#: wrapped in backticks. The backticks are optional because a handful of citations are written
#: bare inside a sentence, and a check that only saw the backticked ones would report a clean
#: sweep of the subset it happened to parse.
CITATION_RE = re.compile(
    r"`?(?P<path>(?:" + "|".join(CITED_ROOTS) + r")/[A-Za-z0-9_./-]+?\.[A-Za-z0-9_]+)"
    r":(?P<start>\d+)(?:-(?P<end>\d+))?`?"
)

#: A continuation citation. `recall/index.py:671` and `:690` is an established shorthand in these
#: documents: the second reference inherits the path from the last full citation earlier on the
#: same line. Parsing it matters because these are exactly the references a reader trusts by
#: association and nobody re-checks.
CONTINUATION_RE = re.compile(r"`:(?P<start>\d+)(?:-(?P<end>\d+))?`")

#: Backticked code spans, from which anchor tokens are extracted.
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")

#: Identifier-shaped tokens inside a code span.
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: An explicit anchor, for a citation whose prose cannot supply one -- a line whose meaning is
#: structural (a `try:`, a `return` in a chain of guards) and has no identifier worth quoting.
#: Written on the citation's line or either neighbour:  <!-- cite-anchor: TOKEN -->
EXPLICIT_ANCHOR_RE = re.compile(r"<!--\s*cite-anchor:\s*(?P<anchor>[^>]+?)\s*-->")

#: Fence delimiters. Citations inside a fenced block are skipped: the block is usually a shell
#: transcript or a diff, where the surrounding backtick convention that supplies anchors does not
#: apply, so every one of them would report UNVERIFIABLE and drown the real ones.
FENCE_RE = re.compile(r"^\s*(```|~~~)")

#: Documents whose citations are a record of what was true when they were written, and which are
#: therefore not maintained against today's line numbers. Their citations are still parsed, still
#: verified and still printed under FROZEN, so a stale one stays visible; it just does not fail
#: the build, because the remedy would be to edit an archive until it agreed with the present,
#: which is the opposite of what an archive is for.
#:
#: Deliberately ONLY `docs/archive/`. Pre-registrations were considered and rejected: they are
#: written continuously and cite code heavily, so freezing them would put a large and growing
#: share of the corpus outside the gate, and the standing rule they live under protects
#: PREDICTIONS from revision, not pointers. A stale line number in a live pre-registration is
#: fixed like any other.
FROZEN_PREFIXES = ("docs/archive/",)

#: A document-level opt-out, for prose whose `path:line` references belong to SOMEBODY ELSE'S
#: repository. It must be **alone on its own line and outside any code fence** -- see
#: `is_external`, and the self-inflicted bug recorded there.
#:
#: This is not a convenience. `docs/their-harness-parity.md` is the procedure for running RE-call
#: inside mem0's benchmark harness, and it cites `benchmarks/locomo/run.py:417` and
#: `benchmarks/beam/run.py:690` in THEIR tree. The first does not exist here, which is loud. The
#: second DOES exist here and is a completely unrelated file, so without this marker the check
#: quietly verifies a claim about mem0's code against ours and reports a verdict on it. A wrong
#: answer delivered confidently is the failure this whole tool exists to prevent, so it must not
#: be the tool's own behaviour.
EXTERNAL_MARKER = "<!-- citations: external -->"

OK = "OK"
STALE = "STALE"
FROZEN = "FROZEN"
EXTERNAL = "EXTERNAL"
UNVERIFIABLE = "UNVERIFIABLE"
MISSING_FILE = "MISSING_FILE"
OUT_OF_RANGE = "OUT_OF_RANGE"

#: Verdicts that fail the check outright. UNVERIFIABLE is absent on purpose and is governed by
#: the ratchet instead; FROZEN is absent because the document is not maintained.
FAILING = (STALE, MISSING_FILE, OUT_OF_RANGE)


@dataclass(frozen=True)
class Citation:
    """One `path:line` reference found in a document."""

    doc: str
    doc_line: int
    path: str
    start: int
    end: int

    def __str__(self) -> str:
        span = f"{self.start}" if self.start == self.end else f"{self.start}-{self.end}"
        return f"{self.path}:{span}"


@dataclass(frozen=True)
class Result:
    """The verdict on one citation, and enough detail to fix it without looking anything up."""

    citation: Citation
    status: str
    detail: str
    anchor: str | None = None

    @property
    def failed(self) -> bool:
        return self.status in FAILING


def strip_fenced_blocks(lines: list[str]) -> list[str]:
    """Blank out fenced code blocks, preserving line numbering.

    Numbering is preserved rather than the lines removed so that a reported document line number
    still matches what an editor shows, which is the difference between a mechanical fix and an
    archaeology session.
    """
    out: list[str] = []
    fence: str | None = None
    for line in lines:
        match = FENCE_RE.match(line)
        if fence is None and match:
            fence = match.group(1)
            out.append("")
            continue
        if fence is not None:
            out.append("")
            if match and match.group(1) == fence:
                fence = None
            continue
        out.append(line)
    return out


def extract_citations(doc: str, lines: list[str]) -> list[Citation]:
    """Find every citation in one document, given its fence-stripped lines."""
    found: list[Citation] = []
    for number, line in enumerate(lines, start=1):
        last_path: str | None = None
        # Full and continuation citations are collected together and re-sorted by position, so a
        # continuation resolves against the full citation that actually precedes it in the line
        # rather than against whichever pattern happened to be scanned first.
        marks: list[tuple[int, str | None, str, str | None]] = []
        for match in CITATION_RE.finditer(line):
            marks.append(
                (match.start(), match.group("path"), match.group("start"), match.group("end"))
            )
        for match in CONTINUATION_RE.finditer(line):
            marks.append((match.start(), None, match.group("start"), match.group("end")))
        for _position, path, start_text, end_text in sorted(marks, key=lambda mark: mark[0]):
            if path is None:
                if last_path is None:
                    continue
                path = last_path
            else:
                last_path = path
            start = int(start_text)
            end = int(end_text) if end_text else start
            found.append(Citation(doc=doc, doc_line=number, path=path, start=start, end=end))
    return found


def context_lines(lines: list[str], index: int) -> str:
    """The prose a citation's anchors may be drawn from.

    The citation's own line plus one either side, skipping a blank line, a heading, or a table
    row. A table row is its own context: the row above is a different claim about a different
    file, and letting its identifiers leak in is how a check starts agreeing with itself.
    """
    line = lines[index]
    if line.lstrip().startswith("|"):
        return line
    chunk = [line]
    for step in (-1, 1):
        neighbour = index + step
        if not 0 <= neighbour < len(lines):
            continue
        candidate = lines[neighbour]
        if not candidate.strip() or candidate.lstrip().startswith(("#", "|")):
            continue
        chunk.append(candidate)
    return "\n".join(chunk)


def anchor_candidates(context: str) -> list[str]:
    """Identifier-shaped tokens from the backticked spans in `context`, best first.

    Longest first, because a longer identifier is a stronger anchor and the report should name
    the strongest one it tried. The citations themselves are stripped first: `recall`, `index`
    and `py` are tokens of the path, not evidence about the line.
    """
    explicit = [match.group("anchor").strip() for match in EXPLICIT_ANCHOR_RE.finditer(context)]
    stripped = CONTINUATION_RE.sub(" ", CITATION_RE.sub(" ", context))
    spans = CODE_SPAN_RE.findall(stripped)

    # A whole span with no whitespace is a stronger anchor than the pieces it splits into, and
    # some are ONLY strong whole: `bge-small-symmetric-v1` tokenises to `bge`, `small`,
    # `symmetric`, `v1`, none of which identifies the profile the sentence is about.
    whole = [span.strip() for span in spans if span.strip() and not re.search(r"\s", span)]
    tokens: list[str] = []
    for span in spans:
        tokens.extend(TOKEN_RE.findall(span))

    ordered = sorted(set(whole) | set(tokens), key=lambda token: (-len(token), token))
    return explicit + [
        token
        for token in ordered
        if len(token) >= MIN_ANCHOR_LENGTH and token not in ANCHOR_STOPWORDS
    ]


def anchor_lines(source_lines: list[str], token: str) -> list[int]:
    """1-based line numbers of `token` in the cited file.

    A word boundary rather than a substring: `promote` is not evidence for a line that says
    `promote_generation`, and treating it as evidence is how a citation to a renamed function
    keeps passing. The LEADING boundary admits an underscore, because a parameter and the
    attribute it is stored on are the same thing under two names, and requiring an exact match
    reported `recall_mcp/stores.py:154` stale over `self._generation_mode`. The trailing boundary
    stays strict, where the prefix/suffix confusion is real.

    A token containing a character that cannot appear in an identifier -- `bge-small-symmetric-v1`
    -- is matched literally, since there is no word boundary to speak of.
    """
    if TOKEN_RE.fullmatch(token):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9_])")
    else:
        pattern = re.compile(re.escape(token))
    return [n for n, line in enumerate(source_lines, start=1) if pattern.search(line)]


def scope_span(source: str, line: int, total: int) -> tuple[int, int]:
    """The line span an anchor for `line` may occupy.

    The INNERMOST enclosing `def`/`class` -- decorators included -- because a citation into a
    function body is a claim about that function, and the function's name is at its `def`. Failing
    that (module-level code), the innermost enclosing statement, which keeps a citation to one
    item of a long module-level tuple from borrowing the next tuple's contents. Either is widened
    to at least `ANCHOR_WINDOW` lines each way.

    **Innermost only, and deliberately not the enclosing classes.** An earlier version also
    accepted an anchor sitting on any enclosing scope's header line, on the reasoning that "line
    1776 is inside class `PgVectorStore`" is a true and checkable claim. It is true, and it is
    nearly vacuous for a 2000-line class -- and it was measured doing real damage: 3 of 88 passing
    citations rested on it, and one of those, `recall/store.py:686`, was a citation to a comment
    about `close()` in a document claiming it showed `delete_sources()`. The rule that was
    supposed to reduce false alarms was concealing a defect of exactly the kind this tool exists
    to find. Dropping it cost 3 verdicts and recovered that one.

    A file that does not parse falls back to the plain window. That is a real degradation and it
    is silent by design: a syntax error in the repository is not this check's business to report,
    and refusing to verify would turn one broken file into a wall of citation failures.
    """
    low, high = max(1, line - ANCHOR_WINDOW), min(total, line + ANCHOR_WINDOW)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return low, high

    definitions: list[tuple[int, int]] = []
    statements: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            definitions.append((start, node.end_lineno or node.lineno))
        elif isinstance(node, ast.stmt) and node.end_lineno is not None:
            statements.append((node.lineno, node.end_lineno))

    containing = [d for d in definitions if d[0] <= line <= d[1]]
    if not containing:
        containing = [s for s in statements if s[0] <= line <= s[1]]
    if not containing:
        return low, high
    start, end = min(containing, key=lambda span: span[1] - span[0])
    return min(low, start), max(high, min(total, end))


def verify(citation: Citation, context: str, root: Path) -> Result:
    """Decide one citation."""
    target = root / citation.path
    if not target.is_file():
        return Result(citation, MISSING_FILE, f"{citation.path} does not exist")
    source = target.read_text(encoding="utf-8", errors="replace")
    source_lines = source.splitlines()
    if citation.end > len(source_lines):
        return Result(
            citation,
            OUT_OF_RANGE,
            f"{citation.path} has {len(source_lines)} lines, cited line {citation.end} is past "
            f"the end",
        )

    total = len(source_lines)
    if target.suffix == ".py":
        low, high = scope_span(source, citation.start, total)
        # A cited RANGE must admit anchors across the whole range, not just its first line.
        if citation.end != citation.start:
            end_low, end_high = scope_span(source, citation.end, total)
            low, high = min(low, end_low), max(high, end_high)
    else:
        low = max(1, citation.start - ANCHOR_WINDOW)
        high = min(total, citation.end + ANCHOR_WINDOW)

    tried: list[tuple[str, list[int]]] = []
    weak_on_cited_line: list[str] = []
    for token in anchor_candidates(context):
        hits = anchor_lines(source_lines, token)
        if not hits:
            # Not a token of THIS file. Sentences routinely cite two files at once, so a token
            # that is absent here is evidence about the other one, not a broken anchor.
            continue
        if len(hits) > MAX_ANCHOR_LINES:
            # Too common to certify a citation. It can still WITHHOLD an accusation, which is a
            # weaker claim and needs stronger evidence: the token must be on the CITED LINE
            # itself, not merely somewhere in its scope.
            #
            # That distinction is load-bearing and was measured. On the scope-wide version,
            # `recall/embedding_registry.py:223` -- a closing `),` whose profile moved to 228, the
            # exact defect this tool was built for -- was downgraded from STALE to UNVERIFIABLE by
            # an incidental common token elsewhere in the same tuple, and the report stopped
            # saying where the anchor had gone. On the cited-line version it is STALE again, while
            # `recall/control_plane.py:802`, which literally IS `def cutover(...)`, still
            # correctly escapes the accusation. A reader who opens the file at the cited line
            # lands on that line, so a token sitting on it is evidence of a kind that a token
            # thirty lines away is not.
            if any(citation.start <= hit <= citation.end for hit in hits):
                weak_on_cited_line.append(token)
            continue
        if any(low <= hit <= high for hit in hits):
            return Result(
                citation, OK, f"anchor `{token}` found in scope {low}-{high}", anchor=token
            )
        tried.append((token, hits))

    if weak_on_cited_line:
        # Positive evidence that something the sentence names is ON the cited line, just not
        # evidence strong enough to certify it. Calling this STALE would be a false accusation:
        # `recall/control_plane.py:802` is literally `def cutover(...)` and the sentence is about
        # `cutover()`, but the name is common enough in that file to fail the distinctiveness
        # cap, so the verdict would rest on an unrelated token from the same sentence. "Cannot
        # decide" is the truthful answer, and it is the difference between a check that reports
        # what it knows and one that reports what it guessed.
        return Result(
            citation,
            UNVERIFIABLE,
            f"`{weak_on_cited_line[0]}` is on the cited line but occurs too often in "
            f"{citation.path} to certify it; quote something more specific, or add "
            f"<!-- cite-anchor: TOKEN -->",
        )

    if not tried:
        return Result(
            citation,
            UNVERIFIABLE,
            "no distinctive anchor in the surrounding prose; quote the identifier this line is "
            "about in backticks, or add <!-- cite-anchor: TOKEN -->",
        )

    # Report up to three anchors rather than picking one. Every single-anchor rule tried here
    # was wrong somewhere: longest sent a reader chasing `recall-enterprise` in a sentence about
    # `replay_pending`, and fewest-hits promoted a bare `grep` over `ReasoningProviderPorts`.
    # Which token the sentence is really about is a judgement, and the report should hand the
    # reader the candidates instead of guessing on their behalf.
    ranked = sorted(tried, key=lambda pair: (len(pair[1]), -len(pair[0])))[:3]
    shown = "; ".join(
        f"`{token}` on line " + ", ".join(str(hit) for hit in hits[:5])
        for token, hits in ranked
    )
    token, hits = ranked[0]
    nearest = min(hits, key=lambda hit: abs(hit - citation.start))
    return Result(
        citation,
        STALE,
        f"no anchor within lines {low}-{high} (the scope of cited line {citation.start}). "
        f"Anchors are elsewhere: {shown}. Nearest is line {nearest}.",
        anchor=token,
    )


def is_external(lines: list[str]) -> bool:
    """Whether a document has opted its citations out, given its FENCE-STRIPPED lines.

    The marker must be **alone on its own line**, and fenced blocks are already blank by the time
    this sees them. Both conditions are scar tissue. The first version tested
    `EXTERNAL_MARKER in text`, and the pre-registration for this very tool then described the
    marker in a sentence -- which silently exempted all 8 of that document's citations, including
    the deliberately stale one it was recording. A document that documents the opt-out must not
    thereby take it, and neither must a tutorial that shows it in a code block.
    """
    return any(line.strip() == EXTERNAL_MARKER for line in lines)


def collect(root: Path) -> list[Result]:
    """Verify every citation in every scanned document."""
    results: list[Result] = []
    for path in sorted(root.glob(DOC_GLOB)):
        doc = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = strip_fenced_blocks(text.splitlines())
        external = is_external(lines)
        frozen = doc.startswith(FROZEN_PREFIXES)
        for citation in extract_citations(doc, lines):
            if external:
                # Not verified at all, rather than verified and excused. Reading somebody else's
                # line 690 out of our file would produce a verdict, and any verdict here is noise.
                results.append(Result(citation, EXTERNAL, "another repository's tree"))
                continue
            context = context_lines(lines, citation.doc_line - 1)
            result = verify(citation, context, root)
            if frozen and result.failed:
                result = Result(result.citation, FROZEN, result.detail, result.anchor)
            results.append(result)
    return results


#: The committed ceiling on UNVERIFIABLE citations. A data file rather than a constant so that
#: the number and the reason for it are one diff, and so that lowering it is the ordinary outcome
#: of anchoring a citation.
BASELINE_PATH = "docs/citation_unverifiable_baseline.txt"


def read_ceiling(root: Path) -> int:
    """The committed ceiling, or 0 if there is no baseline file.

    Absent means zero, not unlimited: a missing baseline is a checkout problem, and the failure
    mode of guessing "unlimited" is a gate that passes because its own configuration went
    missing.
    """
    path = root / BASELINE_PATH
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return int(stripped)
    return 0


def render(
    results: list[Result], ceiling: int, *, list_unverifiable: bool = False
) -> tuple[str, bool]:
    """Build the report, and say whether the check passed."""
    counts = Counter(result.status for result in results)
    failures = [result for result in results if result.failed]
    unverifiable = [result for result in results if result.status == UNVERIFIABLE]

    out: list[str] = []
    for status in FAILING:
        group = [result for result in results if result.status == status]
        if not group:
            continue
        out.append(f"{status} ({len(group)}):")
        for result in group:
            out.append(
                f"  {result.citation.doc}:{result.citation.doc_line}: "
                f"{result.citation} -- {result.detail}"
            )
        out.append("")

    stale_frozen = [result for result in results if result.status == FROZEN]
    if stale_frozen:
        out.append(f"FROZEN ({len(stale_frozen)}, not maintained, does not fail):")
        for result in stale_frozen:
            out.append(
                f"  {result.citation.doc}:{result.citation.doc_line}: "
                f"{result.citation} -- {result.detail}"
            )
        out.append("")

    # The UNVERIFIABLE list goes LAST and, at or below the ceiling, is summarised rather than
    # enumerated. It is the longest section by far (53 entries here), and printing it above the
    # actionable findings buried them: a CI log where the one line that needs acting on sits under
    # fifty that do not is a log nobody reads to the end. The COUNT is always in the summary
    # below, so nothing is hidden -- only the roll-call is deferred until it is worth acting on.
    over_ceiling = len(unverifiable) > ceiling
    if unverifiable and (over_ceiling or list_unverifiable):
        out.append(f"UNVERIFIABLE ({len(unverifiable)}, ceiling {ceiling}):")
        for result in unverifiable:
            out.append(
                f"  {result.citation.doc}:{result.citation.doc_line}: {result.citation}"
            )
        out.append("")

    summary = " ".join(
        f"{status}={counts.get(status, 0)}"
        for status in (OK, STALE, UNVERIFIABLE, FROZEN, EXTERNAL, MISSING_FILE, OUT_OF_RANGE)
    )
    out.append(f"{len(results)} citations: {summary}")

    if over_ceiling:
        out.append("")
        out.append(f"FAIL: {len(unverifiable)} unverifiable citations, ceiling is {ceiling}.")
        out.append("Anchor the new citation by quoting, in backticks in the same sentence, the")
        out.append("identifier the cited line is about, or add <!-- cite-anchor: TOKEN -->.")
        out.append(f"The ceiling in {BASELINE_PATH} is a ratchet: lower it as citations get")
        out.append("anchored, never raise it.")
    if failures:
        out.append("")
        out.append(f"FAIL: {len(failures)} citations no longer point at the code they claim.")

    return "\n".join(out), not (failures or over_ceiling)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify path:line citations in docs/*.md")
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root to scan (default: this checkout)"
    )
    parser.add_argument(
        "--unverifiable-ceiling",
        type=int,
        default=None,
        help="fail if more than this many citations are UNVERIFIABLE "
        "(default: the committed baseline)",
    )
    parser.add_argument(
        "--list-unverifiable",
        action="store_true",
        help="enumerate the unverifiable citations even when they are within the ceiling",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    ceiling = read_ceiling(root) if args.unverifiable_ceiling is None else args.unverifiable_ceiling

    results = collect(root)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "doc": result.citation.doc,
                        "doc_line": result.citation.doc_line,
                        "citation": str(result.citation),
                        "status": result.status,
                        "detail": result.detail,
                        "anchor": result.anchor,
                    }
                    for result in results
                ],
                indent=2,
            )
        )
        return 0

    report, passed = render(results, ceiling, list_unverifiable=args.list_unverifiable)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
