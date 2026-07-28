"""Which `FINDINGS.md` sections make an abstention claim, and whether the inventory knows it.

`results/INSTRUMENT_STATUS.md` tracks, for every abstention claim this project has published,
whether that claim is still checkable. It is hand-written, and `FINDINGS.md` is grown
continuously by other sessions — so the inventory has gone stale three times in one day: once
when a merged artifact falsified a row, twice when a new section appeared that the inventory
did not know existed. The second failure mode is the one this module exists to close.

**What this module does NOT do.** It does not decide whether a claim is still true, whether an
artifact backs it, or what status a row should carry. That requires judging whether a specific
JSON file describes the code as it ships today — a parser cannot do that, and a generator that
guessed would manufacture exactly the false confidence `INSTRUMENT_STATUS.md` exists to expose.

**What it does instead**, mechanically:

1. Parse `FINDINGS.md`'s numbered headings (`## N. Title`, `### Nx. Title`) into `Section`s.
2. Classify which of those sections' *body text* discusses abstention, via `ABSTENTION_KEYWORDS`
   minus a deliberate, commented `EXCLUDED_SECTIONS` list (sections that use the words only in
   passing — the LOCOMO section intro, a backward reference to an unrelated postmortem, and so
   on; each entry says why).
3. Parse `INSTRUMENT_STATUS.md`'s table for the section ids its rows already cover.
4. Diff the two: `missing_sections()` returns the abstention-claiming ids the table does not
   cover yet, so a test can fail naming them instead of the inventory silently going stale again.

The STATUS a human then assigns to a newly-discovered row — `current`, `stale`, `unfalsifiable`,
a retraction, whatever honestly fits — stays entirely outside this module, by design.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINDINGS_PATH = REPO_ROOT / "results" / "FINDINGS.md"
INSTRUMENT_STATUS_PATH = REPO_ROOT / "results" / "INSTRUMENT_STATUS.md"

# `## 2. Title`, `### 2b. Title`, `## 5b. Title`, `### 9h. Title` ... Heading DEPTH (## vs ###)
# does not reliably track id depth in this document -- "5b" is a "##" heading, not "###" -- so
# both levels are accepted and the id itself is what defines a section, not where it sits in the
# heading hierarchy. Un-numbered headings ("## What this document establishes", "#### Why quoting
# one depth was a mistake") never match, so they fall inside whichever numbered section precedes
# them rather than starting a new one -- which is what lets a "####" sub-heading exist at all.
_HEADING_RE = re.compile(r"^#{2,3} (\d+[a-z]?)\.\s+(.*)$", re.MULTILINE)

# Case-insensitive substring terms that mark a section's BODY as discussing abstention behaviour.
# Chosen by reading results/FINDINGS.md rather than guessed: every term below is a lexical form
# that actually occurs there. "abstain" and "abstention" are listed separately on purpose -- one
# is a verb ("abstains", "abstaining"), the other a noun ("Abstention (category 5)"), and neither
# is a substring of the other, so a filter checking only one silently misses the other.
ABSTENTION_KEYWORDS: tuple[str, ...] = (
    "abstain",          # verb form: "abstains", "abstained", "abstaining on half..."
    "abstention",       # noun form: "Abstention (category 5)", "abstention accuracy"
    "false-abstain",    # refusing an answerable question -- the cost abstention is traded against
    "false-confident",  # answering wrong with high confidence -- the failure abstention prevents
    "fcr",              # "False Confidence Rate", this repo's name for the gap-guard's failure mode
    "separability",     # AUC / Mann-Whitney separation of the answerable vs unanswerable scores
    "gap threshold",    # the cosine-gap abstention gate ("a fixed gap threshold does NOT transfer")
    "not_entailed",     # the entailment judge's verdict label feeding the entailment abstain path
)

# Sections whose BODY matches a keyword above but do not themselves publish a checkable
# abstention claim -- kept as an explicit, commented list rather than a broader/narrower keyword
# set, per the design this module implements: a bare keyword filter with no exclusion list would
# either over-report forever (every passing mention becomes a row) or get quietly loosened until
# it reports nothing (someone drops a keyword to silence a false positive and a real gap goes
# with it). Every entry here is a human decision, not a parser miss.
#
# NOT listed: the "What this document establishes" preamble, which also mentions abstention only
# in passing. It carries no number ("## What this document establishes", no "N. " prefix), so
# `_HEADING_RE` never turns it into a candidate `Section` in the first place -- there is no id to
# put in this dict. Named here so a reader auditing this list for gaps does not go looking for it.
EXCLUDED_SECTIONS: dict[str, str] = {
    "9": (
        "LOCOMO section intro: describes the abstention CATEGORY the benchmark tests (why the "
        "adversarial split exists) without measuring anything itself. The claim is in 9b "
        "(accuracy) and 9c (judge sweep), both already tracked."
    ),
    "2b": (
        "Explains a bug fix to best_threshold's fitting rule; its own text names where the "
        "resulting claim lives -- 'section 6 has the measurements that drove the change and "
        "what it cost' -- and section 6 is already tracked."
    ),
    "4": (
        "The trust-layer section explicitly disclaims its own local abstain column as evidence "
        "('The n=2 abstain column is not evidence in either direction') and names where the real "
        "claim is measured instead: 'section 2's separability analysis, 5b's n=100 arm, and "
        "9-10' -- all already tracked."
    ),
    "9f": (
        "Compares Mem0's published-LOCOMO protocol against this repo's own; 'abstention is "
        "forbidden' / 'abstention is the measured behaviour' is one row of that comparison "
        "table, restating 9b/9d's already-tracked setup rather than reporting a new measurement."
    ),
    "9k": (
        "RLS false-alarm postmortem; 'the 4-question abstention sample' is cited only as one of "
        "three past bad-probe examples while diagnosing an unrelated table-visibility bug -- no "
        "abstention number is asserted in this section."
    ),
    "11": (
        "Reranking section; 'Abstention is unaffected (0.00 on all three arms, n=446)' "
        "re-confirms 9b's already-tracked default-mode rate under a retrieval-focused change -- "
        "it is not a new claim about abstention itself."
    ),
}


@dataclass(frozen=True)
class Section:
    """One numbered `FINDINGS.md` heading and the text between it and the next numbered one.

    `body` deliberately excludes the heading line itself -- classification reads body text "not
    just its title" (a title can advertise a topic a section never measures, and vice versa; see
    9h vs 9i, where only one of the two says "abstention" in its own title).
    """

    id: str
    title: str
    body: str


def parse_findings_sections(text: str) -> list[Section]:
    """Every numbered heading in `FINDINGS.md`, in document order.

    A section runs from just after its heading to just before the next NUMBERED heading --
    un-numbered sub-headings in between (`#### Why quoting one depth was a mistake`) stay inside
    the body of whichever numbered section contains them.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        sections.append(Section(id=m.group(1), title=m.group(2).strip(), body=body))
    return sections


def section_mentions_abstention(section: Section) -> bool:
    """Whether any `ABSTENTION_KEYWORDS` term occurs (case-insensitively) in `section.body`."""
    haystack = section.body.lower()
    return any(keyword in haystack for keyword in ABSTENTION_KEYWORDS)


def _id_sort_key(section_id: str) -> tuple[int, str]:
    """Sorts ids numerically-then-alphabetically: 2, 2b, 5, 5b, 9, 9a, ..., 9p, 10, 10b, 11."""
    m = re.match(r"(\d+)([a-z]*)", section_id)
    assert m is not None, f"not a section id: {section_id!r}"
    return (int(m.group(1)), m.group(2))


def abstention_claiming_sections(sections: list[Section]) -> list[str]:
    """Ids of sections that make an abstention claim: keyword-matched, minus `EXCLUDED_SECTIONS`.

    Raises if `EXCLUDED_SECTIONS` names an id that is not among `sections` -- an exclusion for a
    heading that was since renamed or removed is not a no-op, it is a silently stale entry, which
    is exactly the failure mode this design exists to make loud instead of quiet.
    """
    by_id = {s.id: s for s in sections}
    unknown = set(EXCLUDED_SECTIONS) - set(by_id)
    if unknown:
        raise ValueError(
            f"EXCLUDED_SECTIONS names section id(s) absent from FINDINGS.md: {sorted(unknown)} "
            f"-- the heading was renamed, renumbered, or removed; update the exclusion list"
        )
    claiming = [
        s.id
        for s in sections
        if section_mentions_abstention(s) and s.id not in EXCLUDED_SECTIONS
    ]
    return sorted(claiming, key=_id_sort_key)


# Matches the first cell of a markdown table row: `| <cell> | ...` -> `<cell>`. Deliberately
# requires the cell to contain no literal "|" -- every claim cell in this table is prose, and a
# regex that tried to handle escaped pipes inside it would be solving a problem this file does
# not have.
_FIRST_CELL_RE = re.compile(r"^\s*\|\s*([^|]*?)\s*\|")

# A claim cell's leading section id, e.g. "§2 fixed gap threshold..." or "§9b LOCOMO abstention...".
_CLAIM_ID_RE = re.compile(r"§(\d+[a-z]?)\b")

# Convention (documented in INSTRUMENT_STATUS.md's own header): a claim cell reading e.g.
# "§10 LongMemEval, all rows" covers not just the bare id but every lettered sub-id of it that
# exists in FINDINGS.md (10, 10b, 10c, 10d, ...) -- one row standing for a whole numbered family,
# rather than requiring the row to enumerate every sub-id by hand.
_ALL_ROWS_MARKER = "all rows"


def _first_cell(table_row_line: str) -> str:
    m = _FIRST_CELL_RE.match(table_row_line)
    return m.group(1) if m else ""


def _table_rows(status_text: str) -> list[str]:
    """Data rows (header and separator excluded) of the claim/status/artifact/notes table.

    Scoped to the one contiguous block of "|"-led lines whose first row's first cell reads
    "claim" -- not every "|"-led line in the document -- so a future second table elsewhere in
    `INSTRUMENT_STATUS.md` (e.g. inside "Known gaps") cannot silently feed rows into this check.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in status_text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    for block in blocks:
        if len(block) >= 2 and _first_cell(block[0]).strip().lower() == "claim":
            return block[2:]  # skip the header row and the `|---|---|...` separator
    return []


def parse_covered_ids(status_text: str, known_ids: Iterable[str]) -> set[str]:
    """Section ids `INSTRUMENT_STATUS.md`'s table claims to cover.

    Reads only each row's CLAIM column (first cell). A cross-reference in a row's NOTES column
    (e.g. "also quoted at FINDINGS.md section 10c") documents a relationship between two claims;
    it does not assert that row is where section 10c's OWN checkability is tracked, so notes-
    column mentions are deliberately not scanned.
    """
    known_ids = set(known_ids)
    covered: set[str] = set()
    for row in _table_rows(status_text):
        claim_cell = _first_cell(row)
        ids = _CLAIM_ID_RE.findall(claim_cell)
        if not ids:
            continue
        covered.update(ids)
        if _ALL_ROWS_MARKER in claim_cell.lower():
            for base in ids:
                covered.update(
                    i for i in known_ids
                    if i == base or (i.startswith(base) and i[len(base):].isalpha())
                )
    return covered


def missing_sections(findings_text: str, status_text: str) -> list[str]:
    """Abstention-claiming `FINDINGS.md` section ids that `INSTRUMENT_STATUS.md` does not cover.

    This is the whole point of the module: everything above is plumbing for this one diff. A
    non-empty result means `FINDINGS.md` grew a section INSTRUMENT_STATUS.md has never seen --
    exactly the failure mode that made the inventory go stale twice in one day.
    """
    sections = parse_findings_sections(findings_text)
    all_ids = [s.id for s in sections]
    claiming = set(abstention_claiming_sections(sections))
    covered = parse_covered_ids(status_text, all_ids)
    return sorted(claiming - covered, key=_id_sort_key)


def load_missing_sections() -> list[str]:
    """`missing_sections()` against the checked-in `FINDINGS.md` / `INSTRUMENT_STATUS.md`."""
    findings_text = FINDINGS_PATH.read_text(encoding="utf-8")
    status_text = INSTRUMENT_STATUS_PATH.read_text(encoding="utf-8")
    return missing_sections(findings_text, status_text)
