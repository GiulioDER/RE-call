"""`recall.source_identity`: one identity per document, whatever spelling a row carries (M6).

Invariant: every spelling of one memo in `tests/identity_spellings.same_document` has one
`SourceIdentity`, and every memo in `other_documents` has another. The failure mode is the
`identity-mismatch` class: a comparison site derives the root its own way, so one memo spelled two
ways reads as two roots, or two memos read as one.

Red proof (2026-11-04), each mutation restored afterwards:

- `source_identity` returning `root` without `nfc()`: `test_every_spelling_has_one_identity` failed
  on `nfd-root` in its equality assertion.
- `ReferenceIndex.__init__` (recall/supersession.py) rooting pairs with the pre-change
  `root_of(source, file)` instead of `source_identity(source, file).root`:
  `test_reference_index_keeps_one_memo_one_document` failed on `nfd-root` and
  `file-uri-decoded-file`, the memo reported in `collided`.
"""

from __future__ import annotations

import pytest

from recall.source_identity import SourceIdentity, source_identity
from recall.supersession import ReferenceIndex
from tests.identity_spellings import FILE, ROOT, other_documents, same_document

PLAIN_FILE, PLAIN_SOURCE = same_document()["plain"]
EXPECTED = source_identity(PLAIN_SOURCE, PLAIN_FILE)


def test_the_plain_spelling_is_the_root_and_the_file() -> None:
    assert EXPECTED == SourceIdentity(root=f"{ROOT}/", file=FILE)


@pytest.mark.parametrize("spelling", sorted(same_document()))
def test_every_spelling_has_one_identity(spelling: str) -> None:
    file, source = same_document()[spelling]
    assert source_identity(source, file) == EXPECTED


@pytest.mark.parametrize("other", sorted(other_documents()))
def test_other_documents_have_other_identities(other: str) -> None:
    file, source = other_documents()[other]
    assert source_identity(source, file) != EXPECTED


def test_a_row_without_a_source_has_no_root() -> None:
    assert source_identity(None, FILE) == SourceIdentity(root=None, file=FILE)
    assert source_identity("", FILE) == SourceIdentity(root=None, file=FILE)


@pytest.mark.parametrize("spelling", sorted(set(same_document()) - {"plain"}))
def test_reference_index_keeps_one_memo_one_document(spelling: str) -> None:
    """Two rows of one memo are one root, so the memo is not reported as collided copies."""
    assert ReferenceIndex([same_document()["plain"], same_document()[spelling]]).collided == frozenset()


def test_reference_index_keeps_two_roots_apart() -> None:
    """The negative control: a second root holding the same relative path IS a collision."""
    index = ReferenceIndex([same_document()["plain"], other_documents()["second-root"]])
    assert index.collided == frozenset({FILE})
