"""One test per rule of the three deterministic context modes.

The modes build EMBEDDING text; they must never touch the text RE-call stores or the hash that
identifies a source file. That last property is the load-bearing one, so it gets a property-style
test over five corpus shapes rather than one example, and it is asserted again at the `Indexer`
level in `tests/test_context_modes_index.py`, where the stored rows actually exist.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from recall.context import (
    DEGRADATION_ORDER,
    DOCUMENT_DEGRADATION_ORDER,
    NEIGHBOR_MAX_CHARS,
    SECTION_DEGRADED_MAX_CHARS,
    SECTION_MAX_CHARS,
    SOURCE_MAX_CHARS,
    TITLE_MAX_CHARS,
    ContextPolicy,
    contextual_passages,
    _degradation_ladder,
    document_title,
    root_relative_source,
)
from recall.embedding_registry import REGISTERED_PROFILES, context_version_for
from recall.index import DEFAULT_MAX_CHARS, chunk_text

#: The three candidate profiles this session exists to make testable, and the mode each declares.
CANDIDATE_PROFILES = {
    "bge-small-context-document-v1": "document",
    "bge-small-context-section-v1": "section",
    "bge-small-context-neighbor-v1": "neighbor",
}
CONTEXT_MODES = tuple(CANDIDATE_PROFILES.values())


@dataclass
class CharTokenizer:
    """Exact by construction: one token per character, so a budget is a length."""

    def count_tokens(self, text: str) -> int:
        return len(text)


def field(passage: str, name: str) -> str | None:
    """The value of one rendered field, or None when the field was not rendered."""
    prefix = f"{name}: "
    for line in passage.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


# ---------------------------------------------------------------------------------------------
# Rule 1. Title precedence: frontmatter title, then first H1, then root-relative basename.
# ---------------------------------------------------------------------------------------------


def test_title_precedence_is_frontmatter_then_first_h1_then_basename() -> None:
    fm = "---\ntitle: From Frontmatter\n---\n\n# From H1\n\nbody text.\n"
    assert document_title(fm, "\n# From H1\n\nbody text.\n", "team/notes.md") == "From Frontmatter"

    h1 = "# From H1\n\n# Second H1\n\nbody text.\n"
    assert document_title(h1, h1, "team/notes.md") == "From H1"

    plain = "body text with no heading and no frontmatter.\n"
    assert document_title(plain, plain, "team/deep/notes.md") == "notes.md"


def test_the_basename_fallback_survives_a_path_longer_than_the_source_cap() -> None:
    """The basename comes from the WHOLE path, not from the path after the source cap.

    `root_relative_source` used to truncate to 256 characters, and this fallback split that
    truncated string. At 264 characters the cut landed on a `/`, the basename was empty and the
    `title:` field disappeared from the passage; at 261 the title was the fragment `not`. The
    document is unchanged and legal in both cases, so nothing announced the loss.
    """
    plain = "body text with no heading and no frontmatter.\n"
    for source in ("dir/" * 64 + "notes.md", "d" * 252 + "/notes.md", "x/" * 400 + "notes.md"):
        assert len(source) > 256
        assert document_title(plain, plain, source) == "notes.md", source

    # And the field really is rendered, rather than silently omitted.
    _, passages = contextual_passages(
        plain, plain, [plain.strip()], "dir/" * 64 + "notes.md", ContextPolicy(mode="document")
    )
    assert field(passages[0], "title") == "notes.md"


def test_an_indented_frontmatter_title_does_not_outrank_the_documents_own() -> None:
    """A nested `title:` is a sub-object's label, not the document's.

    The scan returns on its first hit, so a nested key appearing ABOVE the real one used to win.
    Indentation is the only thing that tells them apart.
    """
    raw = "---\nauthor: someone\nmeta:\n  title: Nested Label\ntitle: The Real Title\n---\n\nbody.\n"
    assert document_title(raw, "\nbody.\n", "team/notes.md") == "The Real Title"


def test_a_title_is_not_lifted_from_prose_between_two_thematic_breaks() -> None:
    """A leading ``---`` is also a horizontal rule, and this scan used to pair it with the next.

    Any ``title:`` line in the prose caught between the two rules was then read as the
    document's own title, so a memo could be labelled by a sentence a reader would never have
    taken for metadata. `frontmatter_span` refuses the pairing, and the H1 wins as it should.
    """
    raw = "---\n\nSome prose.\n\ntitle: not a real title\n\n---\n\n# Real Heading\n"
    assert document_title(raw, raw, "team/notes.md") == "Real Heading"


# ---------------------------------------------------------------------------------------------
# Rule 2. Root-relative paths only; strip control characters from structural fields.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "C:\\Users\\someone\\secret.md",
        "c:/Users/someone/secret.md",
        "../../../etc/shadow",
        "team/../../outside/notes.md",
        "//server/share/notes.md",
    ],
)
@pytest.mark.parametrize("mode", ("none", *CONTEXT_MODES))
def test_a_source_that_is_not_root_relative_is_refused_in_every_mode(
    hostile: str, mode: str
) -> None:
    """Including `none`, which returns before it would use the source.

    A guard reachable only on the expensive path is a guard the cheapest caller skips.
    """
    body = "first paragraph.\n\nsecond paragraph."
    with pytest.raises(ValueError, match="root-relative"):
        contextual_passages(
            body, body, ["first paragraph.", "second paragraph."], hostile,
            ContextPolicy(mode=mode),  # type: ignore[arg-type]
        )


def test_control_characters_are_stripped_from_every_structural_field() -> None:
    """Title, source and section hierarchy. The chunk itself is content and is left alone."""
    raw = "---\ntitle: Ti\x07tle\u200bHere\n---\n\n## Sec\x01tion\u2060Name\n\nthe chunk\x07 body.\n"
    body = "\n## Sec\x01tion\u2060Name\n\nthe chunk\x07 body.\n"
    structured, passages = contextual_passages(
        raw, body, ["## Sec\x01tion\u2060Name", "the chunk\x07 body."],
        "te\x02am/no\u200btes.md", ContextPolicy(mode="section"),
    )

    assert field(passages[1], "title") == "TitleHere"
    assert field(passages[1], "source") == "team/notes.md"
    assert field(passages[1], "section") == "SectionName"
    # The chunk is content, not a structural field: it reaches the embedder as stored.
    assert structured[1].text == "the chunk\x07 body."
    assert "the chunk\x07 body." in passages[1]


# ---------------------------------------------------------------------------------------------
# Rule 3. Caps: title 256, source 256, section hierarchy 512.
# ---------------------------------------------------------------------------------------------


def test_title_source_and_section_are_capped_at_256_256_and_512() -> None:
    """Asserted against the LITERAL numbers, not against the constants.

    Asserting `len(title) == TITLE_MAX_CHARS` compares the code with itself: raising the constant
    raises both sides and the test stays green while the published cap moves. A mutation sweep
    caught exactly that, on all three caps.
    """
    title = "T" * 600
    headings = "".join(f"{'#' * level} {'H' * 200}\n\n" for level in range(1, 7))
    raw = f"---\ntitle: {title}\n---\n\n{headings}leaf paragraph.\n"
    body = f"\n{headings}leaf paragraph.\n"
    source = "dir/" * 200 + "notes.md"

    _, passages = contextual_passages(
        raw, body, ["leaf paragraph."], source, ContextPolicy(mode="section")
    )

    assert len(field(passages[0], "title") or "") == 256
    assert len(field(passages[0], "source") or "") == 256
    assert len(field(passages[0], "section") or "") == 512
    # The uncapped inputs really were longer, so a cap is what produced those lengths rather than
    # the input happening to fit.
    assert len(title) > 256 and len(source) > 256
    assert len(" > ".join(("H" * 200,) * 6)) > 512
    # The named constants must BE those numbers, so a reader who reaches for them is not reading
    # a different rule from the one enforced above.
    assert (TITLE_MAX_CHARS, SOURCE_MAX_CHARS, SECTION_MAX_CHARS) == (256, 256, 512)
    assert SECTION_DEGRADED_MAX_CHARS == 256


def test_the_path_guard_holds_on_its_own_return_value() -> None:
    """`root_relative_source` validates and does NOT truncate, so its postcondition survives.

    It used to return `normalised[:256]`, applied after the checks, which could MANUFACTURE the
    traversal it had just refused: `"a" * 253 + "/..x"` passed the check and came back as a
    256-character path whose final segment is `..`. The cap now belongs to the rendered field.
    """
    for prefix in range(250, 260):
        source = "a" * prefix + "/..x/leaf.md"
        result = root_relative_source(source)
        assert ".." not in result.split("/"), f"guard returned a traversal at prefix {prefix}"
        assert not result.startswith("/")
        assert result == source, "the guard normalises, it does not shorten"

    # A real traversal is still refused, so the postcondition above is not held vacuously by a
    # guard that stopped checking.
    with pytest.raises(ValueError, match="traversal"):
        root_relative_source("a" * 253 + "/../leaf.md")

    # The rendered field is where the 256 applies.
    body = "para one.\n\npara two."
    _, passages = contextual_passages(
        body, body, ["para one.", "para two."], "d/" * 300 + "n.md",
        ContextPolicy(mode="document"),
    )
    assert len(field(passages[0], "source") or "") == 256


def test_a_single_letter_first_segment_is_not_mistaken_for_a_drive() -> None:
    """`^[A-Za-z]:` also refuses `a:b/notes.md`, a legal relative path on Linux and macOS.

    `Indexer` builds its source from `relative_to(root).as_posix()`, so that shape reaches this
    guard, and the refusal is raised inside the per-file loop after earlier batches have already
    been committed.
    """
    assert root_relative_source("a:b/notes.md") == "a:b/notes.md"
    assert root_relative_source("ab:c/notes.md") == "ab:c/notes.md"
    for drive in ("C:/x", "C:\\x", "c:/Users/x", "C:", "Z:"):
        with pytest.raises(ValueError, match="absolute"):
            root_relative_source(drive)


def test_the_refusal_message_does_not_echo_the_host_path() -> None:
    """The value this fires on IS an absolute host path, so echoing it is the disclosure."""
    for hostile in ("/home/someone/secret.md", "C:\\Users\\someone\\secret.md"):
        with pytest.raises(ValueError) as caught:
            root_relative_source(hostile)
        message = str(caught.value)
        assert "root-relative" in message
        assert "someone" not in message and "secret" not in message, message


# ---------------------------------------------------------------------------------------------
# Rule 4. Neighbor mode adds at most 200 characters from each adjacent chunk.
# ---------------------------------------------------------------------------------------------


def test_neighbor_mode_takes_at_most_200_characters_from_each_side() -> None:
    """The 200 is a literal, and the fixture can tell a head from a tail.

    A chunk of `"A" * 1000` has the same first 200 characters as its last 200, so a fixture built
    from repeated characters passes whether the code takes the tail or the head. Both weaknesses
    were found by mutating the implementation.
    """
    chunks = [
        "HEAD-A" + "a" * 1000 + "TAIL-A",
        "HEAD-B" + "b" * 1000 + "TAIL-B",
        "HEAD-C" + "c" * 1000 + "TAIL-C",
    ]
    body = "\n\n".join(chunks)
    _, passages = contextual_passages(body, body, chunks, "n.md", ContextPolicy(mode="neighbor"))

    previous, following = field(passages[1], "previous"), field(passages[1], "following")
    assert previous is not None and following is not None
    assert len(previous) == 200
    assert len(following) == 200
    assert NEIGHBOR_MAX_CHARS == 200
    # The TAIL of the one before and the HEAD of the one after, not arbitrary slices.
    assert previous == chunks[0][-200:] and previous.endswith("TAIL-A")
    assert following == chunks[2][:200] and following.startswith("HEAD-C")
    assert "HEAD-A" not in previous and "TAIL-C" not in following
    # At the boundaries there is no neighbour to add, and none is invented.
    assert field(passages[0], "previous") is None
    assert field(passages[2], "following") is None


def test_a_neighbour_excerpt_is_folded_to_one_line_before_the_200_is_counted() -> None:
    """The neighbour budget must count the same kind of character as every other cap.

    Two consequences, and the second is why it matters. A newline in an adjacent chunk used to be
    spent against the 200 while the other caps measure post-normalisation length; and because the
    rendered form is one `field: value` per line, an adjacent chunk containing `\\nsource: /etc/x`
    put a SECOND `source:` line into this chunk's passage.

    The chunk's own text is NOT folded: rule 5 preserves it exactly, so the rendered format
    remains something to embed and never something to parse.
    """
    # ⚠️ The forged line must land INSIDE the 200-character window, or the assertion below cannot
    # discriminate. A first version of this test used `"...\nsource: /etc/shadow\n" + "z" * 400`,
    # whose tail-200 is all `z`: the forgery never reached the field and the injection assertions
    # passed against the unfixed code. The hostile chunk is therefore SHORT.
    hostile = "lead in\nsource: /etc/shadow\ntitle: Trusted"
    chunks = [hostile, "the middle chunk.", "tail\n\nchunk\ttext"]
    body = "\n\n".join(chunks)
    _, passages = contextual_passages(
        body, body, chunks, "docs/notes.md", ContextPolicy(mode="neighbor")
    )

    # Exactly one of each structural field, whatever the neighbours contain.
    assert [ln for ln in passages[1].splitlines() if ln.startswith("source: ")] == [
        "source: docs/notes.md"
    ]
    assert len([ln for ln in passages[1].splitlines() if ln.startswith("title: ")]) == 1
    assert field(passages[1], "previous") == "lead in source: /etc/shadow title: Trusted"

    # The length half of the rule needs a neighbour longer than the cap, so it gets its own.
    long_chunks = ["HEAD" + "z" * 400, "the middle chunk.", "tail\n\nchunk\ttext"]
    long_body = "\n\n".join(long_chunks)
    _, passages = contextual_passages(
        long_body, long_body, long_chunks, "docs/notes.md", ContextPolicy(mode="neighbor")
    )
    previous, following = field(passages[1], "previous"), field(passages[1], "following")
    assert previous is not None and following is not None
    # Counted on the PASSAGE, not on `field()`'s return: `field` reads the first line of a value,
    # so `"\n" not in previous` could never have failed. Five fields, five lines: title, source,
    # previous, content, following (no headings in this corpus, so no section).
    assert len(passages[1].splitlines()) == 5, passages[1]
    assert len(previous) == 200 and len(following) <= 200
    # The 200 is now 200 NORMALISED characters, so a chunk of newlines does not spend the budget
    # on characters the other caps would never have counted.
    assert previous == "z" * 200
    assert following == "tail chunk text"

    # The current chunk is still preserved byte for byte, newlines and all.
    assert chunks[1] in passages[1]
    _, hostile_passages = contextual_passages(
        body, body, chunks, "docs/notes.md", ContextPolicy(mode="section")
    )
    assert chunks[0] in hostile_passages[0]


# ---------------------------------------------------------------------------------------------
# Rule 5. The complete current chunk is preserved before any optional context is added.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_the_complete_chunk_survives_every_rung_of_the_degradation_ladder(mode: str) -> None:
    """Context is what gets dropped under pressure. The chunk is never shortened to make room."""
    chunks = ["first paragraph here.", "second paragraph here.", "third paragraph here."]
    raw = "---\ntitle: A Title\n---\n\n# Head\n\n## Sub\n\n" + "\n\n".join(chunks)
    body = "\n# Head\n\n## Sub\n\n" + "\n\n".join(chunks)

    for budget in range(1, 400):
        _, passages = contextual_passages(
            raw, body, chunks, "team/notes.md",
            ContextPolicy(mode=mode, max_tokens=budget, tokenizer=CharTokenizer()),  # type: ignore[arg-type]
        )
        for chunk, passage in zip(chunks, passages):
            assert chunk in passage, f"chunk was truncated at budget {budget} in mode {mode}"
    # Even a budget below the chunk's own length yields the whole chunk, not a prefix of it.
    _, passages = contextual_passages(
        raw, body, chunks, "team/notes.md",
        ContextPolicy(mode=mode, max_tokens=1, tokenizer=CharTokenizer()),  # type: ignore[arg-type]
    )
    assert passages == chunks


# ---------------------------------------------------------------------------------------------
# Rule 6. Degradation order: neighbor first, section detail second, title detail last.
# ---------------------------------------------------------------------------------------------


def test_degradation_drops_neighbor_first_section_second_and_title_last() -> None:
    """Asserted as the sequence of surviving field sets as the budget shrinks monotonically.

    A budget sweep is the observable form of the claim: sampling one budget would show a single
    rung and say nothing about the ORDER, which is the whole rule.
    """
    chunks = ["first paragraph here.", "second paragraph here.", "third paragraph here."]
    raw = "---\ntitle: A Title\n---\n\n# Head\n\n## Sub\n\n" + "\n\n".join(chunks)
    body = "\n# Head\n\n## Sub\n\n" + "\n\n".join(chunks)

    observed: list[frozenset[str]] = []
    for budget in range(400, 0, -1):
        _, passages = contextual_passages(
            raw, body, chunks, "team/notes.md",
            ContextPolicy(mode="neighbor", max_tokens=budget, tokenizer=CharTokenizer()),
        )
        present = frozenset(
            name for name in ("title", "section", "source", "previous", "following")
            if field(passages[1], name) is not None
        )
        if not observed or observed[-1] != present:
            observed.append(present)

    assert observed == [
        frozenset({"title", "section", "source", "previous", "following"}),
        frozenset({"title", "section", "source"}),   # neighbours went first
        frozenset({"title", "source"}),              # then the section
        frozenset({"source"}),                       # then the title
        frozenset(),                                 # bare chunk last
    ]
    # The ladder's own declaration must agree with the sweep, so a reordering is caught even at a
    # budget the sweep never reaches.
    assert DEGRADATION_ORDER.index("drop-neighbor") < DEGRADATION_ORDER.index("shorten-section")
    assert DEGRADATION_ORDER.index("shorten-section") < DEGRADATION_ORDER.index("drop-section")
    assert DEGRADATION_ORDER.index("drop-section") < DEGRADATION_ORDER.index("drop-title")
    assert DEGRADATION_ORDER[-1] == "chunk-only"


#: Written out, not derived. The ladder now BUILDS itself from `DEGRADATION_ORDER`, so comparing
#: what it emits against that constant compares the code with itself: a mutation sweep reordered
#: the constant and the assertion followed it, green. The literal is the independent claim.
EXPECTED_RUNGS: dict[str, tuple[str, ...]] = {
    "section": ("full", "drop-neighbor", "shorten-section", "drop-section", "drop-title",
                "chunk-only"),
    "neighbor": ("full", "drop-neighbor", "shorten-section", "drop-section", "drop-title",
                 "chunk-only"),
    "document": ("drop-section", "drop-title", "chunk-only"),
}


@pytest.mark.parametrize("mode", CONTEXT_MODES)
def test_the_ladder_emits_exactly_the_declared_order(mode: str) -> None:
    """`DEGRADATION_ORDER` is the ladder's order, not a second copy of it.

    It used to be an independent literal that merely agreed with the rungs the implementation
    built: renaming the implementation's labels left every test green, so the constant was
    decorative and the claim that a test could assert it was false. The ladder is now emitted
    FROM the constant, which is what makes the two impossible to disagree — and is also why the
    expected order below is spelled out rather than read back from it.
    """
    rungs = [
        rung
        for rung, _ in _degradation_ladder(
            "CHUNK", mode=mode, title="T", section="S",  # type: ignore[arg-type]
            source="s.md", previous="p", following="f",
        )
    ]
    assert tuple(rungs) == EXPECTED_RUNGS[mode]
    assert rungs, "the ladder must never be empty; the last rung is the fallback"
    assert DEGRADATION_ORDER == EXPECTED_RUNGS["neighbor"]
    assert DOCUMENT_DEGRADATION_ORDER == EXPECTED_RUNGS["document"]
    # Document mode is a SUFFIX of the one order, never an order of its own.
    assert DOCUMENT_DEGRADATION_ORDER == DEGRADATION_ORDER[-len(DOCUMENT_DEGRADATION_ORDER):]


def test_section_detail_is_shortened_before_it_is_dropped() -> None:
    """The rung between "full section" and "no section" — otherwise "detail" means "presence"."""
    headings = "".join(f"{'#' * level} {'H' * 200}\n\n" for level in range(1, 7))
    chunk = "leaf paragraph."
    raw = f"---\ntitle: A Title\n---\n\n{headings}{chunk}\n"
    body = f"\n{headings}{chunk}\n"

    lengths = set()
    for budget in range(700, 0, -1):
        _, passages = contextual_passages(
            raw, body, [chunk], "n.md",
            ContextPolicy(mode="section", max_tokens=budget, tokenizer=CharTokenizer()),
        )
        section = field(passages[0], "section")
        if section is not None:
            lengths.add(len(section))
    assert lengths == {512, 256}, (
        "a shortened section rung must exist between the full one and dropping it"
    )


# ---------------------------------------------------------------------------------------------
# Rule 7. Context mode and version recorded in the profile identity.
#         (The chunk-metadata half needs stored rows: tests/test_context_modes_index.py.)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("profile_id", "mode"), sorted(CANDIDATE_PROFILES.items()))
def test_each_candidate_profile_carries_its_context_mode_and_version_in_its_identity(
    profile_id: str, mode: str
) -> None:
    entry = REGISTERED_PROFILES[profile_id]
    assert entry.context_mode == mode
    assert entry.context_version == context_version_for(mode)  # type: ignore[arg-type]
    assert entry.context_version == f"context-{mode}-v1"
    # The identity handed to the runtime carries it too, not only the registry row.
    identity = entry.identity(artifact_digest="a" * 64)
    assert identity.context_version == f"context-{mode}-v1"
    # And it is part of the cache fingerprint, so two modes cannot alias one cached vector.
    other = REGISTERED_PROFILES["bge-small-symmetric-v1"].identity(artifact_digest="a" * 64)
    assert identity.fingerprint() != other.fingerprint()


# ---------------------------------------------------------------------------------------------
# Rule 8. Raw chunk content and raw content hashes are byte-identical across all three modes.
# ---------------------------------------------------------------------------------------------

#: Five shapes, named by what each is meant to stress, each with the `max_chars` that makes the
#: real `chunk_text` split it. The boundaries are always the chunker's own: the short shapes get a
#: small budget rather than a hand-written chunk list, so nothing here can drift from what the
#: indexer would actually store.
CORPORA: dict[str, tuple[str, int]] = {
    "with-frontmatter": (
        "---\ntitle: A Document\nauthor: someone\n---\n\n"
        "# Heading One\n\nFirst paragraph.\n\nSecond paragraph.\n",
        40,
    ),
    "without-frontmatter": ("# Heading One\n\nFirst paragraph.\n\nSecond paragraph.\n", 40),
    "no-headings": ("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n", 40),
    "nested-headings": (
        "# One\n\ntop text.\n\n## Two\n\nmid text.\n\n### Three\n\ndeep text.\n\n"
        "## Two Again\n\nsibling text.\n",
        40,
    ),
    # Oversized paragraphs at the DEFAULT budget, so `chunk_text` force-splits with overlap and a
    # chunk begins mid-section while the next begins at a heading. This is the shape where offsets
    # and heading attribution are hardest, and it uses no special budget to get there.
    "chunk-boundaries": (
        "# Alpha\n\n" + "alpha body. " * 120 + "\n\n## Beta\n\n" + "beta body. " * 120
        + "\n\n### Gamma\n\n" + "gamma body. " * 120 + "\n",
        DEFAULT_MAX_CHARS,
    ),
}


def raw_hash(chunks: list[str]) -> str:
    """The identity of the STORED text: every chunk, in order, with an unambiguous separator."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@pytest.mark.parametrize("mode", CONTEXT_MODES)
@pytest.mark.parametrize("shape", sorted(CORPORA))
def test_raw_chunk_text_and_hash_match_the_symmetric_baseline_in_every_mode(
    mode: str, shape: str
) -> None:
    """The load-bearing invariant: a context mode changes embedding text and nothing else.

    `mode="none"` IS the symmetric baseline (`bge-small-symmetric-v1`), so the comparison is
    against a genuinely different code path rather than against a second call with the same
    arguments — the two disagree on the passages and must agree on everything else.
    """
    raw, max_chars = CORPORA[shape]
    body = raw.split("---\n", 2)[2] if raw.startswith("---\n") else raw
    chunks = chunk_text(body, max_chars=max_chars)
    assert len(chunks) >= 2, f"corpus {shape!r} must exercise more than one chunk"
    if shape == "chunk-boundaries":
        assert len(chunks) >= 3, "the boundary corpus must be split by the chunker, not by hand"

    baseline_structured, baseline_passages = contextual_passages(
        raw, body, chunks, "team/notes.md", ContextPolicy(mode="none")
    )
    structured, passages = contextual_passages(
        raw, body, chunks, "team/notes.md", ContextPolicy(mode=mode)  # type: ignore[arg-type]
    )

    baseline_text = [chunk.text for chunk in baseline_structured]
    mode_text = [chunk.text for chunk in structured]
    assert mode_text == baseline_text == chunks
    assert raw_hash(mode_text) == raw_hash(baseline_text) == raw_hash(chunks)
    # Offsets address the same bytes of the same document under either mode, AND those bytes are
    # the chunk's own. Comparing the two arms alone would let an offset defect present in both
    # pass unseen, so each arm is also checked against the document it claims to index. The
    # comparison is on visible tokens because `structure_chunks` maps a chunker that normalises
    # blank-line runs, which its docstring states.
    assert [(c.start, c.end) for c in structured] == [
        (c.start, c.end) for c in baseline_structured
    ]
    for piece in structured:
        assert body[piece.start:piece.end].split() == piece.text.split()
    # And the test is not vacuous: the embedding text really did change.
    assert passages != baseline_passages
    assert baseline_passages == chunks
