"""The derived block's contract, enforced rather than described.

A derived block is machine-owned inference written into a memo body. `content_hash` is over raw
file bytes (`recall/index.py:518`), so writing one re-indexes the file — and if the block were
chunked, the next extraction pass would read its own prior output back as evidence and amplify.
This suite pins the properties that prevent that, one test per property, each written so a
plausible wrong implementation fails it:

* `split_derived_block` is total and `.human` is ALWAYS a prefix of `body`, malformed input
  included, which is what makes the offset invariance unconditional;
* the `rstrip()` on the branch with NO fence is load bearing — without it the extraction cache
  key changes on the first write and the fixed point fails on iteration one;
* a rendered block round trips, and re-renders byte identical, so a re-run never churns;
* the digest is over the parsed structure, not raw bytes, so a CRLF checkout is not tampering;
* every grammar rule is a refusal and never a repair;
* a file with a block chunks identically to the same file without one, and its `text_start` /
  `text_end` offsets are unchanged;
* `status: superseded` inside a block does not read as a closed decision, and `recall check`
  does not offer the machine's own values back to the author.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from recall.derived_block import (
    CLOSE_FENCE,
    OPEN_FENCE,
    DerivedBlockError,
    DerivedDigestMismatch,
    DerivedEntry,
    derived_digest,
    diagnose_derived_block,
    parse_derived_block,
    render_derived_block,
    split_derived_block,
    verify_derived_block,
)
from recall.document import parse_document
from recall.lint import lint_corpus


def test_no_fence_yields_the_whole_body_rstripped() -> None:
    body = "# A\n\nplain prose.\n"
    split = split_derived_block(body)
    assert split.human == "# A\n\nplain prose."
    assert split.block_text == ""
    assert split.fence_start is None


def test_a_fence_splits_the_body_at_its_first_line() -> None:
    block = f"{OPEN_FENCE}\nstatus: adopted\n{CLOSE_FENCE}\n"
    body = f"# A\n\nplain prose.\n\n{block}"
    split = split_derived_block(body)
    assert split.human == "# A\n\nplain prose."
    assert split.block_text == block
    assert split.fence_start == len("# A\n\nplain prose.\n\n")


def test_rstrip_makes_human_body_a_fixed_point_on_the_first_write() -> None:
    """The rstrip on the NO-FENCE branch is the load-bearing half.

    Pre-write the body ends "...adopted.\\n". Post-write `body[:fence_start]` ends
    "...adopted.\\n\\n", because the block sits after one blank line. Both must rstrip to the
    same bytes. Without the rstrip on the no-fence branch the pre-write value keeps its trailing
    newline, the extraction cache key changes on the first write, and the fixed point fails on
    iteration one — which looks exactly like model nondeterminism.
    """
    before = "# Retention\n\nThe 90 day window was adopted.\n"
    first = split_derived_block(before).human

    block = f"{OPEN_FENCE}\nstatus: adopted\n{CLOSE_FENCE}\n"
    after = before + "\n" + block
    second = split_derived_block(after).human

    assert first == "# Retention\n\nThe 90 day window was adopted."
    assert second == first


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("# A\n\nprose.\n", id="no-block"),
        pytest.param(f"# A\n\nprose.\n\n{OPEN_FENCE}\nstatus: adopted\n{CLOSE_FENCE}\n",
                     id="one-block"),
        pytest.param(f"# A\n\nprose.\n\n{OPEN_FENCE}\nstatus: adopted\n{CLOSE_FENCE}\n"
                     f"\n{OPEN_FENCE}\nstatus: closed\n{CLOSE_FENCE}\n", id="two-blocks"),
        pytest.param(f"# A\n\nprose.\n\n{OPEN_FENCE}\nstatus: adopted\n{CLOSE_FENCE}\n"
                     f"\nA human appended this after the block.\n", id="block-not-last"),
        pytest.param(f"# A\n\nprose.\n\n{OPEN_FENCE}\nstatus: adopted\n", id="unclosed-fence"),
        pytest.param(f"{OPEN_FENCE}\nstatus: adopted\n{CLOSE_FENCE}\n", id="block-is-whole-body"),
        pytest.param("", id="empty"),
    ],
)
def test_human_body_is_always_a_prefix_of_body(body: str) -> None:
    """The invariant the whole design rests on.

    `structure_chunks` computes offsets with `body.find(text, ...)` (`recall/context.py:197`).
    While `human` is a prefix, every offset is identical with or without a block. This holds for
    MALFORMED input too, which is why the read path strips from the first fence to EOF rather
    than rejoining the text around each block.
    """
    human = split_derived_block(body).human
    assert body.startswith(human)


def test_split_never_raises_on_malformed_input() -> None:
    """index.py and generations.py do not run lint. A half-written file must not crash a build."""
    for body in (f"{OPEN_FENCE}", f"{CLOSE_FENCE}\nstray\n", f"{OPEN_FENCE}\n{OPEN_FENCE}\n"):
        split_derived_block(body)


_PROPOSAL_A = "a" * 64
_PROPOSAL_B = "b" * 64
_PROPOSAL_C = "c" * 64


def _entry(
    head: str = "contradicts",
    value: str = "project_alpha_2026-03-02",
    proposal: str = _PROPOSAL_A,
    note: str = "",
) -> DerivedEntry:
    return DerivedEntry(
        head=head,
        value=value,
        proposal=proposal,
        provider="recall.deterministic@session3-v1",
        reviewer="giulio",
        at="2026-08-11T09:14:22Z",
        note=note,
    )


def test_render_emits_the_documented_grammar() -> None:
    text = render_derived_block([_entry(note="both state a retention window")])
    assert text == (
        f"{OPEN_FENCE}\n"
        "contradicts: project_alpha_2026-03-02\n"
        f"  proposal: {_PROPOSAL_A}\n"
        "  provider: recall.deterministic@session3-v1\n"
        "  reviewer: giulio\n"
        "  at: 2026-08-11T09:14:22Z\n"
        "  note: both state a retention window\n"
        f"digest: {derived_digest([_entry(note='both state a retention window')])}\n"
        f"{CLOSE_FENCE}\n"
    )


def test_render_omits_an_absent_note() -> None:
    assert "note:" not in render_derived_block([_entry()])


def test_render_sorts_entries_by_head_then_value() -> None:
    """Sorted output is what makes a re-render byte identical, so a re-run never churns."""
    shuffled = [
        _entry("status", "adopted", _PROPOSAL_C),
        _entry("same_entity", "project_beta_2026-04-01", _PROPOSAL_B),
        _entry("contradicts", "project_alpha_2026-03-02", _PROPOSAL_A),
    ]
    heads = [
        line.split(":")[0]
        for line in render_derived_block(shuffled).split("\n")
        if line and not line.startswith((" ", "<", "digest:"))
    ]
    assert heads == ["contradicts", "same_entity", "status"]


def test_render_normalises_deprecated_and_obsolete_to_superseded() -> None:
    """The one place a repair is correct: a proposal's vocabulary arriving at the boundary.

    `deprecated` and `obsolete` are in CLOSURE_MARKERS (`recall/lint.py:36`). Written literally,
    the machine's own block would trip the linter built to find prose closure.
    """
    for alias in ("deprecated", "obsolete"):
        assert "status: superseded\n" in render_derived_block([_entry("status", alias)])


def test_render_refuses_an_empty_entry_list() -> None:
    with pytest.raises(DerivedBlockError, match="no entries"):
        render_derived_block([])


def test_render_refuses_duplicate_entries() -> None:
    """A duplicate (head, value) makes the sort non-total, so input order would leak into bytes."""
    with pytest.raises(DerivedBlockError, match="duplicate"):
        render_derived_block([_entry(proposal=_PROPOSAL_A), _entry(proposal=_PROPOSAL_B)])


def test_render_refuses_more_than_one_status() -> None:
    with pytest.raises(DerivedBlockError, match="status"):
        render_derived_block([_entry("status", "adopted"), _entry("status", "closed")])


def test_digest_covers_every_field() -> None:
    """A field outside the digest is a field an editor can change without detection."""
    base = _entry()
    for changed in (
        DerivedEntry("same_entity", base.value, base.proposal, base.provider, base.reviewer,
                     base.at, base.note),
        DerivedEntry(base.head, "other_memo_2026-05-05", base.proposal, base.provider,
                     base.reviewer, base.at, base.note),
        DerivedEntry(base.head, base.value, _PROPOSAL_B, base.provider, base.reviewer, base.at,
                     base.note),
        DerivedEntry(base.head, base.value, base.proposal, "other.provider", base.reviewer,
                     base.at, base.note),
        DerivedEntry(base.head, base.value, base.proposal, base.provider, "someone-else",
                     base.at, base.note),
        DerivedEntry(base.head, base.value, base.proposal, base.provider, base.reviewer,
                     "2020-01-01T00:00:00Z", base.note),
        DerivedEntry(base.head, base.value, base.proposal, base.provider, base.reviewer,
                     base.at, "a note that was not there"),
    ):
        assert derived_digest([changed]) != derived_digest([base])


def test_digest_carries_the_grammar_version() -> None:
    """A v2 grammar must not be able to collide with a v1 hash of the same entries."""
    from recall.derived_block import DERIVED_BLOCK_VERSION
    from recall.lineage import canonical_sha256

    entry = _entry()
    assert derived_digest([entry]) == canonical_sha256(
        {
            "v": DERIVED_BLOCK_VERSION,
            "entries": [
                {
                    "head": entry.head, "value": entry.value, "proposal": entry.proposal,
                    "provider": entry.provider, "reviewer": entry.reviewer, "at": entry.at,
                    "note": entry.note,
                }
            ],
        }
    )


def _block(*entries: DerivedEntry) -> str:
    return render_derived_block(list(entries))


def test_round_trip() -> None:
    entries = [
        _entry("contradicts", "project_alpha_2026-03-02", _PROPOSAL_A, note="they differ"),
        _entry("same_entity", "project_beta_2026-04-01", _PROPOSAL_B),
        _entry("status", "superseded", _PROPOSAL_C),
    ]
    assert list(parse_derived_block(render_derived_block(entries)).entries) == entries


def test_re_render_is_byte_identical() -> None:
    """A re-run over an unchanged file must not churn it."""
    once = _block(_entry("status", "adopted"), _entry("contradicts", "old_memo_2026-01-02"))
    assert render_derived_block(list(parse_derived_block(once).entries)) == once


def test_verify_accepts_a_rendered_block() -> None:
    text = _block(_entry())
    assert verify_derived_block(text).digest == derived_digest([_entry()])


def test_verify_detects_a_tampered_value() -> None:
    text = _block(_entry()).replace("project_alpha_2026-03-02", "project_gamma_2026-09-09")
    with pytest.raises(DerivedDigestMismatch, match="digest"):
        verify_derived_block(text)


def test_verify_detects_a_tampered_reviewer() -> None:
    """The field a forger would most want to change is inside the digest."""
    text = _block(_entry()).replace("reviewer: giulio", "reviewer: nobody")
    with pytest.raises(DerivedDigestMismatch, match="digest"):
        verify_derived_block(text)


def test_digest_is_over_structure_not_bytes() -> None:
    """A CRLF checkout and a BOM are not tampering. This repo lives on Windows and Linux."""
    text = _block(_entry())
    crlf = text.replace("\n", "\r\n")
    bom = "﻿" + text
    assert verify_derived_block(crlf).digest == verify_derived_block(text).digest
    assert verify_derived_block(bom).digest == verify_derived_block(text).digest


def test_a_fully_populated_well_formed_block_parses() -> None:
    """The non-over-rejection test. Every refusal below is worthless without this one."""
    entries = [
        _entry("contradicts", "project_alpha_2026-03-02", _PROPOSAL_A, note="a: colon; and ; too"),
        _entry("same_entity", "project_beta_2026-04-01", _PROPOSAL_B),
        _entry("status", "abandoned", _PROPOSAL_C),
    ]
    assert len(verify_derived_block(render_derived_block(entries)).entries) == 3


@pytest.mark.parametrize(
    ("text", "match"),
    [
        pytest.param(
            f"{OPEN_FENCE}\nsupersedes: other_memo_2026-01-01\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "second source of truth",
            id="forbidden-head-supersedes",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nvalid_until: 2026-01-01\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "second source of truth",
            id="forbidden-head-valid-until",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nrelates_to: x_2026-01-01\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "unknown head",
            id="unknown-head",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nstatus: deprecated\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "closure marker",
            id="status-deprecated",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nstatus: obsolete\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "closure marker",
            id="status-obsolete",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nstatus: pending\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "closed vocabulary",
            id="status-outside-vocabulary",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: [[project_alpha_2026-03-02]]\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "bare document stem",
            id="value-is-a-wikilink",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02.md\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "bare document stem",
            id="value-carries-an-extension",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: not-a-hash\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "proposal",
            id="proposal-not-hex",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "at",
            id="at-is-a-date-not-an-instant",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22+02:00\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "at",
            id="at-is-not-utc",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: \n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "provider",
            id="provider-empty",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: \n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "reviewer",
            id="reviewer-empty",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "reviewer",
            id="missing-required-subkey",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\n  confidence: 0.9\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "unknown sub-key",
            id="unknown-subkey",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"    proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "two spaces",
            id="subkey-over-indented",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nstatus: adopted\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\n"
            f"contradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_B}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n",
            "sorted",
            id="entries-out-of-order",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\n{CLOSE_FENCE}\n",
            "digest",
            id="missing-digest",
        ),
        pytest.param(
            f"{OPEN_FENCE}\ncontradicts: project_alpha_2026-03-02\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: nope\n{CLOSE_FENCE}\n",
            "digest",
            id="malformed-digest",
        ),
        pytest.param(f"{OPEN_FENCE}\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n", "no entries",
                     id="zero-entries"),
        pytest.param(f"{OPEN_FENCE}\nstatus: adopted\n", "close fence", id="unclosed-fence"),
        pytest.param(
            f"{OPEN_FENCE}\nstatus: adopted\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n"
            f"{OPEN_FENCE}\nstatus: closed\n{CLOSE_FENCE}\n",
            "second open fence",
            id="second-open-fence",
        ),
        pytest.param(
            f"{OPEN_FENCE}\nstatus: adopted\n"
            f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
            f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n"
            "a human appended this\n",
            "after the close fence",
            id="content-after-close-fence",
        ),
        pytest.param("no fence at all\n", "open fence", id="no-open-fence"),
    ],
)
def test_parse_refuses(text: str, match: str) -> None:
    with pytest.raises(DerivedBlockError, match=match):
        parse_derived_block(text)


def _codes(body: str) -> list[str]:
    return [code for code, _ in diagnose_derived_block(split_derived_block(body).block_text)]


def test_a_clean_file_diagnoses_nothing() -> None:
    assert _codes("# A\n\nprose.\n") == []
    assert _codes("# A\n\nprose.\n\n" + _block(_entry())) == []


def test_two_blocks_are_duplicated() -> None:
    body = "# A\n\nprose.\n\n" + _block(_entry()) + "\n" + _block(_entry("status", "adopted"))
    assert _codes(body) == ["derived-block-duplicated"]


def test_content_after_the_close_fence_is_not_last() -> None:
    body = "# A\n\nprose.\n\n" + _block(_entry()) + "\nA human appended this.\n"
    codes = diagnose_derived_block(split_derived_block(body).block_text)
    assert [code for code, _ in codes] == ["derived-block-not-last"]


def test_not_last_names_the_bytes_it_costs() -> None:
    """The error must state its own consequence: this text is excluded from retrieval."""
    tail = "A human appended this."
    body = "# A\n\nprose.\n\n" + _block(_entry()) + f"\n{tail}\n"
    ((_, message),) = diagnose_derived_block(split_derived_block(body).block_text)
    assert str(len(tail.encode("utf-8"))) in message
    assert "retrieval" in message


def test_a_tampered_digest_is_tampered_not_malformed() -> None:
    body = "# A\n\nprose.\n\n" + _block(_entry()).replace("reviewer: giulio", "reviewer: nobody")
    assert _codes(body) == ["derived-block-tampered"]


def test_a_half_written_block_is_malformed_not_tampered() -> None:
    """An unclosed fence has no digest to disagree with. Calling it tampering sends the reader
    looking for an attacker who is not there."""
    body = f"# A\n\nprose.\n\n{OPEN_FENCE}\nstatus: adopted\n"
    assert _codes(body) == ["derived-block-malformed"]


def test_a_forbidden_head_is_malformed() -> None:
    body = (
        f"# A\n\nprose.\n\n{OPEN_FENCE}\nsupersedes: other_memo_2026-01-01\n"
        f"  proposal: {_PROPOSAL_A}\n  provider: p\n  reviewer: r\n"
        f"  at: 2026-08-11T09:14:22Z\ndigest: {'0' * 64}\n{CLOSE_FENCE}\n"
    )
    assert _codes(body) == ["derived-block-malformed"]


def test_at_most_one_code_per_file() -> None:
    """A single mistake reported four ways is four times the noise and none of the signal."""
    body = (
        "# A\n\nprose.\n\n"
        + _block(_entry()).replace("reviewer: giulio", "reviewer: nobody")
        + "\n"
        + _block(_entry("status", "adopted"))
        + "\ntrailing prose\n"
    )
    assert len(diagnose_derived_block(split_derived_block(body).block_text)) == 1
    assert _codes(body) == ["derived-block-duplicated"]


#: The seam itself, and the one file allowed to import `parse_frontmatter`.
_SEAM = "recall/document.py"

#: Outside the production packages and on the allowlist for a stated reason: it discards the body
#: entirely and inspects only `meta`, so no evidence path runs through it.
_ALLOWED_OUTSIDE = {"benchmarks/check_temporal_inert.py"}


def test_parse_document_strips_frontmatter_and_the_block() -> None:
    text = (
        "---\nsupersedes: old_memo_2026-01-02\n---\n"
        "# New\n\nThe 90 day window was adopted.\n\n"
        + _block(_entry())
    )
    document = parse_document(text)
    assert document.meta == {"supersedes": "old_memo_2026-01-02"}
    assert document.human_body == "# New\n\nThe 90 day window was adopted."
    assert document.derived_text.startswith(OPEN_FENCE)
    assert OPEN_FENCE not in document.human_body


def test_parse_document_on_a_file_with_no_block() -> None:
    document = parse_document("---\nvalid_from: 2026-01-01\n---\n# New\n\nprose.\n")
    assert document.human_body == "# New\n\nprose."
    assert document.derived_text == ""


def _imports_parse_frontmatter(source: str) -> bool:
    """True if `source` imports `parse_frontmatter` directly out of `recall.frontmatter`."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "recall.frontmatter"
        and any(alias.name == "parse_frontmatter" for alias in node.names)
        for node in ast.walk(ast.parse(source))
    )


def test_only_the_seam_imports_parse_frontmatter() -> None:
    """The guard that makes a seventh call site impossible to add silently.

    Five of the six evidence readers must never see a derived block. Enforcing that at each call
    site means a NEW caller of `parse_frontmatter` silently reads machine output back as evidence
    and nothing goes red. Enforcing it here means the seventh site fails this test.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = [
        relative
        for package in ("recall", "recall_mcp", "recall_interop", "benchmarks")
        for path in sorted((root / package).rglob("*.py"))
        if (relative := path.relative_to(root).as_posix()) != _SEAM
        and relative not in _ALLOWED_OUTSIDE
        and _imports_parse_frontmatter(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        f"{offenders} import parse_frontmatter directly; go through {_SEAM} so the derived "
        f"block cannot reach an evidence path"
    )


def test_the_seam_guard_can_see_a_violation() -> None:
    """The canary, running on the SAME detector the test above uses.

    A guard that cannot fail is worse than no guard, because it reads as one. The mutation in
    this task's step 11 proves the guard once, by hand; this proves it on every CI run. Both
    directions are asserted, so a detector stuck on `True` fails here too.
    """
    assert _imports_parse_frontmatter(
        "from recall.frontmatter import parse_frontmatter, supersedes_key\n"
    )
    assert not _imports_parse_frontmatter(
        "from recall.frontmatter import supersedes_key, validity_bounds\n"
    )
    assert not _imports_parse_frontmatter("from recall.document import parse_document\n")


def _write(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8", newline="\n")


def test_lint_reports_a_tampered_block(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "memo_2026-06-01.md",
        "# Memo\n\nprose.\n\n" + _block(_entry()).replace("reviewer: giulio", "reviewer: nobody"),
    )
    issues = lint_corpus(tmp_path)
    assert [(i.level, i.code) for i in issues] == [("error", "derived-block-tampered")]


def test_lint_reports_a_duplicated_block(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "memo_2026-06-01.md",
        "# Memo\n\nprose.\n\n" + _block(_entry()) + "\n" + _block(_entry("status", "adopted")),
    )
    assert [i.code for i in lint_corpus(tmp_path)] == ["derived-block-duplicated"]


def test_lint_reports_a_block_that_is_not_last(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "memo_2026-06-01.md",
        "# Memo\n\nprose.\n\n" + _block(_entry()) + "\nappended by a human\n",
    )
    (issue,) = lint_corpus(tmp_path)
    assert issue.code == "derived-block-not-last"
    assert issue.level == "error"
    assert "retrieval" in issue.message


def test_lint_reports_a_malformed_block(tmp_path: Path) -> None:
    _write(tmp_path, "memo_2026-06-01.md", f"# Memo\n\nprose.\n\n{OPEN_FENCE}\nstatus: adopted\n")
    assert [i.code for i in lint_corpus(tmp_path)] == ["derived-block-malformed"]


def test_lint_is_clean_on_a_well_formed_block(tmp_path: Path) -> None:
    _write(tmp_path, "memo_2026-06-01.md", "# Memo\n\nprose.\n\n" + _block(_entry()))
    assert lint_corpus(tmp_path) == []


def test_a_block_does_not_trip_the_prose_closure_warning(tmp_path: Path) -> None:
    """A block's own text must be invisible to CLOSURE_MARKERS.

    The payload is a `note:` carrying "replaces", a word the regex really matches. NOT
    `status: superseded` — that looks like the obvious choice and is a guard that cannot fail:
    CLOSURE_MARKERS (`recall/lint.py:36`) alternates on "superseded by" and "supersedes", and
    the bare word "superseded" matches neither, so such a test stays green even if the strip
    regresses. A `note:` is free text and is never normalised, so it reaches the rendered block
    verbatim; unstripped, this file would earn a `closure-marker-unlinked` warning it does not
    deserve.
    """
    _write(
        tmp_path,
        "memo_2026-06-01.md",
        "# Memo\n\nA settled question.\n\n"
        + _block(_entry(note="replaces the earlier retention window")),
    )
    assert [i.code for i in lint_corpus(tmp_path)] == []


def test_the_closure_warning_still_fires_on_real_prose(tmp_path: Path) -> None:
    """Paired with the test above, so it cannot pass against a linter that warns about nothing."""
    _write(
        tmp_path,
        "memo_2026-06-01.md",
        "# Memo\n\nThis replaces the earlier retention window.\n",
    )
    assert [i.code for i in lint_corpus(tmp_path)] == ["closure-marker-unlinked"]
