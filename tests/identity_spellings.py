"""Spellings of one memo, and of other memos that look like it, for identity tests (M6).

CCA's `identity-mismatch` class (42 confirmed RE-call findings, 11 P1, measured 2026-11-04) is one
document carried under several spellings and a check that compares only one. A test of a comparison
site runs every pair from `same_document` through it and expects one answer, and every pair from
`other_documents` and expects another. Each pair is `(file, source)`, the shape the store's rows
and `recall.supersession.ReferenceIndex` take.
"""

from __future__ import annotations

import os
import unicodedata
from urllib.parse import quote

# An absolute root on either platform, so a `file://` URI and the plain path name one place.
ROOT = ("C:" if os.name == "nt" else "") + "/srv/memos/café"
FILE = "notes/my plan.md"


def same_document(root: str = ROOT, file: str = FILE) -> dict[str, tuple[str, str]]:
    """Rows of the one memo `root`/`file`, each spelled the way some writer stores it."""
    nfd_root = unicodedata.normalize("NFD", root)
    nfd_file = unicodedata.normalize("NFD", file)
    uri = "file://" + ("" if root.startswith("/") else "/") + quote(f"{root}/{file}", safe="/:")
    return {
        "plain": (file, f"{root}/{file}"),
        "nfd-root": (file, f"{nfd_root}/{file}"),
        "nfd-file": (nfd_file, f"{root}/{nfd_file}"),
        "backslash-file": (file.replace("/", "\\"), f"{root}/{file}"),
        "file-uri-decoded-file": (file, uri),
        "file-uri-escaped-file": (quote(file), uri),
    }


def other_documents(root: str = ROOT, file: str = FILE) -> dict[str, tuple[str, str]]:
    """Rows of OTHER memos that share a spelling with `root`/`file`: never the same document."""
    head, _, name = file.rpartition("/")
    upper = f"{head}/{name.upper()}" if head else name.upper()
    return {
        "second-root": (file, f"{root}-other/{file}"),
        "upload-root": (file, f"{root}/uploads/acme/{file}"),
        "other-case": (upper, f"{root}/{upper}"),
        "same-stem-other-folder": (f"elsewhere/{name}", f"{root}/elsewhere/{name}"),
    }
