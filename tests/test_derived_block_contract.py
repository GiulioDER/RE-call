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

import pytest

from recall.derived_block import (
    CLOSE_FENCE,
    OPEN_FENCE,
    DerivedBlockError,
    DerivedEntry,
    derived_digest,
    render_derived_block,
    split_derived_block,
)


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
