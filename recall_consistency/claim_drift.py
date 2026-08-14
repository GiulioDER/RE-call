"""Find claims a corpus restated with different numbers.

The signal is deliberately narrow: the same sentence, the same words around the number, a
different number. That is what a retraction looks like when the author corrects a measurement in
place, and it is the case where an old revision stays a close semantic match to the question the
new revision answers. Rewritten claims are missed on purpose. A detector that guesses produces a
report that asserts something false about the corpus, and correctness is the entire point of the
report.
"""
from __future__ import annotations

import re

from recall_consistency.findings import ClaimDrift
from recall_consistency.history_corpus import Revision

#: Integers, decimals, signed values, percentages. `+0.0303`, `0.945`, `792`, `48%`.
#: The condition sits on the SIGN, not on the match. Only the sign can eat a separator, so
#: `2020-2024`, `45%-50%` and `45ms-50ms` keep their hyphens and stay one claim each, while a
#: bare digit run after a letter is still a number and `p95`, `v2`, `batch_32` and `x5` are
#: still read. Constraining the whole match instead loses all four, which is a large hole in
#: exactly the benchmark prose this tool is pointed at.
#:
#: The rule is general on purpose. Enumerating separators leaked twice, first through `%`, then
#: through unit suffixes, and a third leak was one unnoticed suffix away.
NUMBER = re.compile(r"(?:(?<![\w.%])[-+])?\d+(?:[.,]\d+)*%?")

#: Below this, the non-numeric remainder of a line is table punctuation rather than a claim.
MIN_SKELETON_CHARS = 12

#: An ordered list item's own number. `2.` becoming `3.` is a renumber, not a restated claim.
LIST_MARKER = re.compile(r"^\d+[.)]\s")


def claim_skeleton(line: str) -> str:
    """A line with every number replaced by `#`. Two claims are the same claim when these match.

    Public because it is the whole matching rule, and a rule that can only be tested through
    `drifts` cannot be tested at all: two skeletons have to collide before any of it is
    observable, and most correct behaviour here is the absence of a collision.
    """
    return NUMBER.sub("#", line.strip()).strip()


def _claim_lines(body: str) -> dict[str, str]:
    """Map each numeric line's skeleton to the line, dropping any skeleton that occurs twice.

    Skeletons are not unique. `recall@5 is 0.92` and `recall@10 is 0.88` both reduce to
    `recall@# is #`, so a document holding both gives one key two candidate lines. Keeping the
    first would pair whichever happened to come first in each revision, and report two unrelated
    claims as one claim restated. That is a fabricated citation in a report whose whole purpose
    is to show this tool does not fabricate.

    So ambiguity fails closed, which is the same choice `recall.frontmatter` already makes: a
    supersession edge whose target stem matches two files is refused rather than guessed at. The
    cost is a missed detection, and the report states that it counts a floor rather than a total.
    """
    seen: dict[str, str] = {}
    ambiguous: set[str] = set()
    for raw in body.splitlines():
        line = raw.strip()
        if not NUMBER.search(line):
            continue
        skeleton = claim_skeleton(line)
        if len(skeleton) < MIN_SKELETON_CHARS:
            continue
        if skeleton in seen:
            # Only a collision between DIFFERENT lines is ambiguous. The same sentence written
            # twice has one answer, so dropping it would be a false negative bought for nothing.
            if seen[skeleton] != line:
                ambiguous.add(skeleton)
            continue
        seen[skeleton] = line
    for skeleton in ambiguous:
        del seen[skeleton]
    return seen


def drifts(revs: list[Revision]) -> list[ClaimDrift]:
    """Every consecutive pair of revisions where one claim kept its words and changed its number."""
    found: list[ClaimDrift] = []
    for old, new in zip(revs, revs[1:]):
        old_claims = _claim_lines(old.body)
        new_claims = _claim_lines(new.body)
        for skeleton, old_line in old_claims.items():
            new_line = new_claims.get(skeleton)
            if new_line is None or new_line == old_line:
                continue
            # A restated claim changes exactly one number. Two changed numbers almost always
            # means two different subjects: `gpt3 scored 88` against `gpt4 scored 91` shares a
            # skeleton, and pairing them quotes two models as one retraction. No lexical rule
            # separates an identifier's digits from a measurement's, because `gpt3` and `gpt4`
            # are two subjects while `0.92` and `0.945` are one claim, and the characters do not
            # say which. Counting what changed does say, and it holds whatever the tokenizer did.
            old_nums = NUMBER.findall(old_line)
            new_nums = NUMBER.findall(new_line)
            if len(old_nums) != len(new_nums):
                # Equal skeletons do NOT guarantee equal counts, which an earlier version of
                # this comment claimed. The placeholder is `#`, and a literal `#` in the text (a
                # heading, an issue reference like `#281`) fills a slot a real number could fill.
                # Comparing anyway lets `zip` truncate, and a cross-subject pairing then passes
                # the single-change rule below. Refuse instead.
                continue
            changed = [i for i, (a, b) in enumerate(zip(old_nums, new_nums)) if a != b]
            if len(changed) != 1:
                continue
            if changed[0] == 0 and LIST_MARKER.match(old_line) and LIST_MARKER.match(new_line):
                # An ordered list renumbered because an item was inserted above it. Found on this
                # repository's own WRITEUP.md, where `2.` became `3.` and the sentence after it
                # was untouched. A reader spots that instantly, and one visible false positive
                # discredits every true finding standing next to it.
                continue
            found.append(
                ClaimDrift(
                    path=new.path,
                    old_sha=old.sha,
                    new_sha=new.sha,
                    old_date=old.date,
                    new_date=new.date,
                    old_line=old_line,
                    new_line=new_line,
                )
            )
    return found
