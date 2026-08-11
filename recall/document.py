"""The single seam between a corpus file's bytes and the body every reader treats as evidence.

There were six production callers of `parse_frontmatter`, and five of them read the body as
evidence: the indexer, the S3 generation builder, the write-time check, the semantic lint, and
`lint --fix`. A derived block reaching any of those would let an extraction pass read its own
prior output back as evidence and amplify.

Stripping the block at each of the six would work exactly until a seventh appears, and a seventh
that forgets goes unnoticed — nothing turns red. So the strip lives HERE, and a contract test
(`tests/test_derived_block_contract.py`) asserts that no production module outside this file
imports `parse_frontmatter` at all. The seventh call site fails a test instead of silently
poisoning the corpus.

`recall/lint.py` is the one caller that also wants `derived_text`, because it is the thing that
reports a malformed or tampered block.
"""
from __future__ import annotations

from dataclasses import dataclass

from recall.derived_block import split_derived_block
from recall.frontmatter import parse_frontmatter


@dataclass(frozen=True)
class ParsedDocument:
    """A corpus file cut into its three parts.

    `human_body` is what a human wrote: frontmatter removed, derived block removed, rstripped.
    It is always a PREFIX of the body `parse_frontmatter` returned, which is what keeps
    `structure_chunks` offsets identical whether or not the file carries a block.
    """

    meta: dict[str, str]
    human_body: str
    derived_text: str


def parse_document(text: str) -> ParsedDocument:
    """Parse a corpus file. The only production path that should call `parse_frontmatter`."""
    meta, body = parse_frontmatter(text)
    split = split_derived_block(body)
    return ParsedDocument(meta, split.human, split.block_text)
