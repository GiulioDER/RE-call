"""A machine-owned, regenerable block at the end of a memo body.

Frontmatter recognises three keys — `supersedes`, `valid_from`, `valid_until`
(`recall/frontmatter.py:12`) — and the trust layer acts on those and nothing else. The relations
an extractor proposes, `contradicts` and `same_entity`, have nowhere to land. Putting them in
frontmatter would put a machine's inference into the namespace a human authors, where the trust
layer reads it as authored truth.

So they land here instead: a fenced block, last in the body, owned by the machine and stripped
from every path that reads a document as evidence.

**The hazard this module exists to prevent.** `content_hash` is over raw file bytes
(`recall/index.py:518`), so writing a block re-indexes the file. If the block were chunked, the
next extraction pass would read its own prior output back as evidence and amplify: a proposal
becomes a citation for the next proposal, and the corpus grows a self-referential belief no human
ever stated. `split_derived_block` is the only thing standing between the two, which is why it is
total — it never raises, on any input, including input no writer would ever produce.

**Placement is not a preference.** `structure_chunks` computes offsets with
`body.find(text, ...)` (`recall/context.py:197`). While `human` is a strict prefix of `body`,
every offset is identical with or without a block, so `text_start` / `text_end`
(`recall/index.py:600`) are invariant. Prepending would shift every offset in every chunk of every
file that gains a block. End placement also keeps the block out of `document_title`
(`recall/context.py:159`), which reads only frontmatter and the first H1.

**HTML comment fences**, so every markdown renderer hides the block and it can never be mistaken
for frontmatter.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Alone on a line. `.strip()` is used for the comparison, so an indented fence still counts —
#: the read path errs toward stripping MORE, never less.
OPEN_FENCE = "<!-- recall:derived v1 -->"
CLOSE_FENCE = "<!-- /recall:derived -->"


@dataclass(frozen=True)
class DerivedSplit:
    """A body cut into the part humans wrote and the part the machine owns."""

    human: str
    block_text: str
    fence_start: int | None


def _first_fence_offset(body: str) -> int | None:
    cursor = 0
    for line in body.splitlines(keepends=True):
        if line.strip() == OPEN_FENCE:
            return cursor
        cursor += len(line)
    return None


def split_derived_block(body: str) -> DerivedSplit:
    """Cut a body at its first derived-block fence. Total: never raises, on any input.

    `.human` is ALWAYS a prefix of `body` — block or not, well-formed or not — because the rule
    is one rule: strip from the first open fence to EOF. Two blocks, an unclosed fence, a human's
    prose appended after the close fence: all the same. A rejoin around each block would preserve
    that appended prose but would stop `.human` being a prefix, and then the offset invariance
    holds only on well-formed files. A qualified guarantee is one that stops being checked.

    The cost is real and is reported, not hidden: prose after a close fence is excluded from
    retrieval, and `derived-block-not-last` names the byte count it costs.

    **The `rstrip()` is on BOTH branches, and the no-fence branch is the load-bearing one.**
    Pre-write a body ends ``"...adopted.\\n"``. Post-write the block sits after one blank line, so
    ``body[:fence_start]`` ends ``"...adopted.\\n\\n"``. Both rstrip to ``"...adopted."``. Strip
    only when a fence is present and the two differ, the extraction cache key changes on the
    first write, and the fixed point fails on iteration one — indistinguishable, from the outside,
    from model nondeterminism. ``s[:n].rstrip()`` is still a prefix of ``s``, so this costs
    nothing.
    """
    start = _first_fence_offset(body)
    if start is None:
        return DerivedSplit(body.rstrip(), "", None)
    return DerivedSplit(body[:start].rstrip(), body[start:], start)
