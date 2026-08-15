"""Supersession has a direction, and getting it backwards is the worst failure available.

`recall/fix.py` states the stake: declaring the edge the wrong way round marks the LIVE document
stale and ranks it beneath the one it replaced, which is the exact failure the trust layer exists
to prevent, caused by the tool meant to fix it.

Two defects, both measured on `python/peps` rather than imagined:

1. The prompt never said which end of the relation `superseded` holds. The semantic lived only in
   `SupersessionClaim`'s docstring, which the model never sees. Given "This PEP was superseded by
   :pep:`345`", the model filled the slot with the SUBJECT, and `pep-0262` declared itself its own
   predecessor.

2. The ladder's self-supersession guard could not fire outside markdown. It compared
   `supersedes_key(resolved)` against `supersedes_key(file)`, and `supersedes_key` strips `.md`
   and nothing else by a documented contract written for a memo corpus. So `pep-0262.rst` keyed to
   `pep-0262.rst`, never equal to the corpus key `pep-0262`, and the claim in (1) survived the
   ladder. A guard that cannot fail is worse than no guard: it reads as protection.
"""
from __future__ import annotations

import pytest

from recall.truth_extraction import (
    PROMPT_REVISION,
    build_extraction_prompt,
    normalize_extraction,
)

SELF = '{"claims":[{"kind":"supersession","superseded":"%s","quote":"x"}]}'


def test_the_prompt_states_which_end_superseded_holds() -> None:
    """The model is told, not left to infer from a field name."""
    user = build_extraction_prompt(file="a.md", human_body="b", corpus_names=("a", "c")).user
    assert "DIRECTION." in user
    assert "OLDER document that THIS document replaces" in user
    assert "Never name this document itself" in user


def test_the_prompt_tells_the_model_what_to_do_with_the_passive_voice() -> None:
    """The half a direction statement alone does not cover.

    "This document was superseded BY X" has no slot in the schema. Without an instruction the
    model guesses, and the guess it made was to name itself.
    """
    user = build_extraction_prompt(file="a.md", human_body="b", corpus_names=("a", "c")).user
    assert "superseded BY another, emit NO supersession claim" in user


def test_the_prompt_revision_moved_so_cached_answers_are_not_mixed() -> None:
    """`PROMPT_REVISION` is in the cache key. Wording changed under v1 would silently reuse it."""
    assert PROMPT_REVISION == "truth-extraction-prompt-v2"


@pytest.mark.parametrize(
    "file,target,corpus,label",
    [
        # ⚠️ The CORPUS NAME's shape varies here, not only the file's, and that is the point.
        # The first version of this test held the corpus at stem shape on every row and varied
        # only the suffix of `file`. That is the one shape in which a guard comparing a raw
        # corpus name against a stemmed file name works, so the parametrisation could not fail
        # for the defect it was written for — and the fix it certified had in fact BROKEN the
        # three rows below it. `corpus_names` is usually just the list of file names, so the
        # extensioned rows are the common case, not the exotic one.
        ("pep-0262.rst", "pep-0262.rst", ("pep-0262.rst", "other.rst"), "rst corpus, rst file"),
        ("pep-0262.rst", "pep-0262", ("pep-0262", "other"), "stem corpus, rst file"),
        ("docs/pep-0262.rst", "docs/pep-0262.rst", ("docs/pep-0262.rst", "docs/other.rst"),
         "directory-prefixed on both sides"),
        ("a.md", "a.md", ("a.md", "b.md"), "md corpus, the case that already worked"),
        ("a.md", "a", ("a", "b"), "stem corpus, md file"),
        # ⚠️ The MIRROR asymmetry: a stem-shaped `file` against a corpus carrying extensions.
        # Every rule this guard has had covered one direction and left this one open, so a
        # document was free to declare itself its own predecessor — which this file's own
        # docstring calls the worst failure available.
        ("pep-0262", "pep-0262.rst", ("pep-0262.rst", "other.rst"), "stem file, rst corpus"),
        ("minutes.2026-08-15", "minutes.2026-08-15.rst",
         ("minutes.2026-08-15.rst", "minutes.2026-08-14.rst"), "stem file, dated rst corpus"),
        ("notes.txt", "notes.txt", ("notes.txt", "other.txt"), "txt, any other extension"),
        ("a.b.md", "a.b.md", ("a.b.md", "other.md"), "multi-dot name"),
        ("plain", "plain", ("plain", "other"), "no extension at all"),
    ],
)
def test_a_document_cannot_supersede_itself_whatever_its_extension(
    file: str, target: str, corpus: tuple[str, ...], label: str
) -> None:
    """The guard, across every name shape a corpus can hold.

    The defect was that `supersedes_key` handles exactly one extension, so the guard was inert
    outside markdown. A test naming only `.rst` would go green again the moment someone measured
    a fourth corpus, and a test holding the corpus shape fixed goes green against a half-fix.
    """
    claims, rejections = normalize_extraction(
        SELF % target,
        file=file,
        human_body="x",
        corpus_names=corpus,
    )
    assert not claims, f"{label}: a self-supersession must not survive the ladder"
    assert [r.rung for r in rejections] == ["target_not_in_corpus"], label
    # The REASON, not just the rung. `target_not_in_corpus` is also what an unresolvable target
    # gets ("is not a file in the corpus", "matches N files"), so a row whose target simply
    # failed to resolve would otherwise pass this test while proving nothing about
    # self-supersession. Matched on the shared tail rather than on "names itself", because the
    # two forms of this refusal say different things on purpose: only an exact corpus-key match
    # is a FACT that the two names are one entry. Where the file and the corpus disagree about
    # extensions the code has inferred it, and `_own_entries` documents why it refuses anyway.
    assert "cannot supersede itself" in rejections[0].reason, (
        f"{label}: refused for the wrong reason: {rejections[0].reason}"
    )


@pytest.mark.parametrize(
    "file,target,corpus",
    [
        ("pep-0262.rst", "pep-0263.rst", ("pep-0262.rst", "pep-0263.rst")),
        ("a.md", "b.md", ("a.md", "b.md")),
        ("docs/a.md", "docs/b.md", ("docs/a.md", "docs/b.md")),
        # ⚠️ Same base name, different extension: two DIFFERENT documents. Comparing the two
        # names by stem read these as one and refused a real edge as "names itself", which is
        # the wrong answer AND the wrong reason. The three rows below are the ones that
        # survived the guard's first two versions and were broken by its third; without them
        # the parametrisation could not fail for it.
        ("guide.rst", "guide.txt", ("guide.rst", "guide.txt")),
        ("notes.txt", "notes.md", ("notes.txt", "notes.md")),
        ("minutes.2026-08-15", "minutes.2026-08-14", ("minutes.2026-08-15", "minutes.2026-08-14")),
    ],
)
def test_naming_a_different_document_is_not_read_as_naming_itself(
    file: str, target: str, corpus: tuple[str, ...]
) -> None:
    """The over-rejection half, at the same name shapes.

    Comparing stems discards the directory, so this is where a stem comparison would go too far
    if it ever went further than `supersedes_key` already does.
    """
    claims, rejections = normalize_extraction(
        SELF % target, file=file, human_body="x", corpus_names=corpus
    )
    assert [c.superseded for c in claims] == [target], f"{file} -> {target} was refused"
    assert not rejections


@pytest.mark.parametrize(
    "file,target,corpus",
    [
        ("pep-0262.rst", "pep-0262.rst", ("pep-0262.rst", "other.rst")),
        ("a.md", "a.md", ("a.md", "b.md")),
        ("docs/pep-0262.rst", "docs/pep-0262.rst", ("docs/pep-0262.rst", "docs/other.rst")),
    ],
)
def test_an_exact_corpus_match_says_so_rather_than_hedging(
    file: str, target: str, corpus: tuple[str, ...]
) -> None:
    """The other half of the message split, which the shared-tail assertion cannot see.

    When the target and the file resolve to ONE corpus entry, that they are the same document is
    a fact, and the refusal should say it plainly. Only the shapes where the file and the corpus
    disagree about extensions are inferences. Without this, softening every message to the
    hedged form would leave the whole suite green and tell a reader that a certainty was a guess.
    """
    _, rejections = normalize_extraction(
        SELF % target, file=file, human_body="x", corpus_names=corpus
    )
    assert "names itself" in rejections[0].reason, rejections[0].reason


@pytest.mark.parametrize(
    "file,target,corpus,label",
    [
        # A backup artifact beside the real entry. Trying the two fallbacks IN ORDER let this
        # `.orig` answer first, so the lookup that would have found `guide` never ran and the
        # self-edge was ACCEPTED. One unrelated name in a corpus flipped a correct refusal.
        ("guide.rst", "guide", ("guide", "guide.rst.orig"), "a .orig sibling must not shadow"),
        # The other side of the trade `_own_entries` documents. Structurally identical to the
        # row above and semantically the opposite, and no rule over names can separate them, so
        # the conservative answer is taken and the REASON says it was an inference.
        ("v1.2", "v1", ("v1.2.rst", "v1"), "a dotted segment that is not an extension"),
        ("guide.v2", "guide", ("guide.v2.rst", "guide"), "same, with a version suffix"),
    ],
)
def test_a_name_that_cannot_be_told_apart_from_this_file_is_refused(
    file: str, target: str, corpus: tuple[str, ...], label: str
) -> None:
    """Conservative, and honest about which of the two findings it made.

    The stakes are asymmetric and that is the whole argument: a wrongly ACCEPTED self-edge marks
    a live document stale beneath itself, which is the failure the trust layer exists to prevent
    and the one `recall/fix.py` measured; a wrongly refused edge costs one proposal.

    The reason must NOT claim the target "names itself" here, because that is a fact the code
    does not have — it inferred it from a suffix. Asserting the softer wording is what keeps a
    reader from re-checking a document that is fine.
    """
    claims, rejections = normalize_extraction(
        SELF % target, file=file, human_body="x", corpus_names=corpus
    )
    assert not claims, f"{label}: a self-edge must not survive the ladder"
    assert rejections[0].rung == "target_not_in_corpus", label
    assert "resolves to" in rejections[0].reason, (
        f"{label}: the refusal claims a fact it inferred: {rejections[0].reason}"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("pep-0262.rst", "pep-0262"),
        ("docs/pep-0262.rst", "pep-0262"),
        ("a.md", "a"),
        ("a.b.md", "a"),
        ("plain", "plain"),
        # ⚠️ The two shapes that make this OS-dependent under `Path`. `supersedes_key` splits on
        # `/` alone, so a backslash and a drive prefix are ORDINARY CHARACTERS in a name here.
        # `PureWindowsPath` treats both as path syntax and `PurePosixPath` does not, so the same
        # corpus produced a different self-supersession verdict on Windows and on Linux — inside
        # a module whose first docstring line promises no filesystem.
        ("docs\\pep-0262.rst", "docs\\pep-0262"),
        ("C:pep.rst", "C:pep"),
    ],
)
def test_the_base_name_rule_is_pure_string_handling(name: str, expected: str) -> None:
    from recall.truth_extraction._normalize import _base_name

    assert _base_name(name) == expected


def test_a_real_supersession_still_survives() -> None:
    """The non-over-rejection case. A guard that refuses everything passes every refusal test."""
    claims, rejections = normalize_extraction(
        SELF % "old",
        file="new.rst",
        human_body="x",
        corpus_names=("old", "new"),
    )
    assert [c.superseded for c in claims] == ["old"]
    assert not rejections
