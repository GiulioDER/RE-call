"""Parse PEP RFC822 headers and detect prose restatements of a header edge.

The header is the gold label. `Superseded-By:` and `Replaces:` describe the SAME relation from
opposite ends, so both are normalised to one `Edge(superseded, successor)` and deduplicated —
otherwise a pair that declares the relation at both ends counts twice and inflates the
denominator every recall number is measured against.

Direction is the hazard this module exists to get right, and it is the same one `fix.py:24-30`
documents for memos: `Replaces:` is active voice, so the declaring document is the SUCCESSOR;
`Superseded-By:` is passive, so the declaring document is the one being replaced. Inverting
either would declare the live document stale and demote it beneath the one it replaced.

Pure and file-free, so the direction rule is testable on strings alone.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from recall.lint import CLOSURE_MARKERS

#: A PEP header block is RFC822: it ends at the first blank line.
_HEADER_END = re.compile(r"\n\s*\n")

#: How a PEP is cited in prose. Three forms, all of which appear in the corpus:
#:   :pep:`287`   the Sphinx role, the modern convention
#:   PEP 287      plain prose
#:   pep-0287     a file-name-shaped reference
_REF = re.compile(r"(?::pep:`|PEP\s*|pep-)(\d{1,4})", re.IGNORECASE)


def split_header(text: str) -> tuple[str, str]:
    """``(header_block, body)``. A file with no blank line is all header and no body."""
    match = _HEADER_END.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end() :]


def header_fields(head: str) -> dict[str, str]:
    """RFC822 fields, folding continuation lines into the field they continue.

    `Replaces: 245,\\n  246` is one field with two values. Reading the continuation as a new
    field would silently drop PEP 246 from the gold set — a missing positive, which is invisible
    in a precision number and lowers recall for a reason that has nothing to do with the model.
    """
    fields: dict[str, str] = {}
    key: str | None = None
    for line in head.split("\n"):
        if not line.strip():
            continue
        if line[0].isspace() and key is not None:
            fields[key] += " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip()
    return fields


def pep_refs(text: str) -> set[str]:
    """Every PEP cited in `text`, as zero-padded stems.

    Zero-padding is what makes `PEP 5`, `PEP 0005` and `pep-0005` one document rather than three.
    """
    return {f"pep-{int(m.group(1)):04d}" for m in _REF.finditer(text)}


@dataclass(frozen=True, order=True)
class Edge:
    """One supersession relation, normalised so both header spellings produce one object."""

    superseded: str
    successor: str


def edges_from_fields(stem: str, fields: Mapping[str, str]) -> set[Edge]:
    """The edges one PEP's headers declare. `stem` is this PEP, zero-padded."""
    edges: set[Edge] = set()
    for number in re.findall(r"\d+", fields.get("Superseded-By", "")):
        edges.add(Edge(superseded=stem, successor=f"pep-{int(number):04d}"))
    for number in re.findall(r"\d+", fields.get("Replaces", "")):
        edges.add(Edge(superseded=f"pep-{int(number):04d}", successor=stem))
    return edges


def sentences(body: str) -> list[str]:
    """Sentences, with hard line wrapping undone first.

    RST wraps prose at column ~79, so a restatement routinely spans two lines. Splitting on
    newlines cut `"It has been\\nsuperseded by :pep:`287`."` in half and lost the reference,
    undercounting the recall ceiling — the one number this whole set exists to publish.
    """
    flat = re.sub(r"\s*\n\s*", " ", body)
    return re.findall(r"[^.!?]*[.!?]", flat)


def restates(body: str, partner: str) -> str | None:
    """The sentence stating this edge in prose, or None.

    Requires a closure marker AND the partner's reference in the SAME sentence. Whole-body
    co-occurrence — a marker in one section, the reference in another — is not a restatement;
    on this corpus it more than triples the count (8 to 26 of 47) by pairing text that has no
    relation, which is the failure `fix.py`'s docstring records for looser matching.
    """
    for sentence in sentences(body):
        if CLOSURE_MARKERS.search(sentence) and partner in pep_refs(sentence):
            return sentence
    return None
