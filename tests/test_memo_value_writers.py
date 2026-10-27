"""Writers a person uses to change a memo's declared values: remove a supersedes reference, set or
clear a validity date, set or clear the status. Bytes in, bytes out, nothing else moved.

Red proof, 2026-10-06, each mutation alone, failing for the stated reason (JUnit XML), then restored;
all six green after, and tests/test_frontmatter.py, test_fix.py, test_corpus_rewrite_contract.py and
test_rewrite_routing_contract.py still pass (206 passed, 1 skipped). Against `recall/frontmatter.py`:
- W1 the last reference withdrawn leaves `supersedes:` empty: `supersedes` still in the metadata.
- W2 only the first of repeated keys withdrawn: two `supersedes` lines remain.
- W3 an undeclared reference "withdrawn" anyway: bytes returned instead of None.
- W4 a scalar appended instead of replaced: two `valid_until` lines.
- W5 a line break in a value accepted: DID NOT RAISE ValueError.
Against `recall/rewrite.py`:
- W6 the old status kept beside the new one: two `status:` entries.
- W7 a status outside the vocabulary accepted: DID NOT RAISE RewriteRefused.
"""

from __future__ import annotations

import pytest

from recall.frontmatter import parse_frontmatter, remove_supersedes_target, set_frontmatter_scalar, supersedes_targets
from recall.rewrite import RewriteRefused, _derived_value, set_derived_status


def _meta(raw: bytes) -> dict:
    return parse_frontmatter(raw.decode("utf-8"))[0]


BLOCK = b"---\nsupersedes:\n  - a.md\n  - b.md\nvalid_until: 2026-01-01\n---\nbody line\r\n"


def test_removing_one_reference_keeps_the_others_and_the_rest_of_the_file() -> None:
    out = remove_supersedes_target(BLOCK, "a.md")
    assert out is not None
    assert supersedes_targets(_meta(out)["supersedes"]) == ("b.md",)
    assert _meta(out)["valid_until"] == "2026-01-01"
    assert out.endswith(b"body line\r\n")


def test_removing_the_last_reference_removes_the_key_not_leaves_it_empty() -> None:
    out = remove_supersedes_target(b"---\nsupersedes: a.md\ntype: x\n---\nx\n", "a")
    assert out is not None and "supersedes" not in _meta(out)
    assert b"supersedes" not in out


def test_removing_a_reference_it_does_not_declare_changes_nothing() -> None:
    assert remove_supersedes_target(BLOCK, "zz.md") is None


def test_repeated_supersedes_keys_are_all_withdrawn() -> None:
    raw = b"---\nsupersedes: a.md\nsupersedes: b.md\n---\nx\n"
    out = remove_supersedes_target(raw, "a.md")
    assert out is not None and supersedes_targets(_meta(out)["supersedes"]) == ("b.md",)
    assert out.count(b"supersedes") == 1


def test_a_date_is_replaced_in_place_inserted_when_absent_and_removed_on_none() -> None:
    replaced = set_frontmatter_scalar(BLOCK, "valid_until", "2027-02-03")
    assert replaced is not None and _meta(replaced)["valid_until"] == "2027-02-03"
    assert replaced.count(b"valid_until") == 1
    assert set_frontmatter_scalar(BLOCK, "valid_until", "2026-01-01") is None  # unchanged
    cleared = set_frontmatter_scalar(BLOCK, "valid_until", None)
    assert cleared is not None and "valid_until" not in _meta(cleared)
    inserted = set_frontmatter_scalar(b"no block\n", "valid_from", "2026-01-01")
    assert inserted is not None and _meta(inserted)["valid_from"] == "2026-01-01"
    with pytest.raises(ValueError):
        set_frontmatter_scalar(BLOCK, "valid_until", "2026-01-01\nsupersedes: z.md")


def test_status_is_replaced_never_duplicated_and_only_from_the_vocabulary() -> None:
    once = set_derived_status(b"# T\n\nbody\n", "deprecated")
    assert once is not None and _derived_value(once, "status") == "deprecated"
    twice = set_derived_status(once, "active")
    assert twice is not None and _derived_value(twice, "status") == "active"
    assert twice.count(b"status:") == 1
    assert set_derived_status(twice, "active") is None
    cleared = set_derived_status(twice, None)
    assert cleared is not None and _derived_value(cleared, "status") is None
    with pytest.raises(RewriteRefused):
        set_derived_status(once, "archived")
