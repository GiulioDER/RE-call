"""Folder and facet scoping: a second retrieval dimension over the vectors already stored.

A dense corpus answers "what is near this question". It cannot answer "and only from the python
notes", because nearness has no place to put that. This module is the place: a `Scope` names a
region of the corpus by structure rather than by meaning, and every leg of retrieval applies the
same predicate to it.

**The folder dimension needs no re-index.** The indexer already records a root-relative posix path
per chunk (`recall/index.py`, ``metadata["file"]``), so a folder is a prefix of a value that is
already in the row. Adding folders to a corpus is `mkdir` plus a re-index of the moved files, not
a schema change. The FACET dimension is different and the difference is worth stating plainly: it
is read from frontmatter at index time, so it lands only on the next generation build. Until then
a facet filter matches nothing, which is why `Scope.facet` is refused against a corpus with no
facet rows rather than returning an empty result that reads like "no match".

Two rules this module exists to hold in one place:

⛔ **A scope value is never interpolated into SQL.** Folder matching is a `LIKE`, and a folder is
a caller-supplied string, so ``%`` and ``_`` in one would silently widen the scope from "this
folder" to "every folder that happens to match". `_like_prefix` escapes them and the predicate
binds the result as a parameter. A widened scope does not error: it answers confidently from the
wrong region, which is exactly the failure a scope layer is supposed to make impossible.

⛔ **A dimension is chosen from a fixed table, never built from a caller's string.** `GROUP_SQL`
maps the two dimension NAMES to the two SQL expressions this module is willing to emit. The
alternative, letting a caller pass the JSON key to group by, is an injection with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
from typing import Literal

#: The two structural dimensions. `folder` is derived from where the file sits, `facet` from what
#: the author declared it to be. They are independent on purpose: a memo in `python/` can be a
#: `reference`, and both facts are worth filtering on.
Dimension = Literal["folder", "facet"]

#: The frontmatter key carrying the facet. `recall.frontmatter.FACET_KEYS` decides what reaches
#: chunk metadata in the first place; this is the query-side half of the same agreement, and the
#: two must name the same key or a facet filter silently matches nothing.
FACET_METADATA_KEY = "type"

#: What a chunk's dimension value IS, in SQL, given the chunk table's alias.
#:
#: The folder expression returns ``''`` for a file at the corpus root rather than NULL, so "the
#: root folder" is a value that can be grouped and compared like any other. NULL would drop the
#: root from a GROUP BY that an operator reads as a complete inventory.
GROUP_SQL: dict[str, str] = {
    "folder": (
        "CASE WHEN position('/' in {a}.metadata->>'file') > 0 "
        "THEN regexp_replace({a}.metadata->>'file', '/[^/]*$', '') "
        "ELSE '' END"
    ),
    "facet": "lower({a}.metadata->>'" + FACET_METADATA_KEY + "')",
}


def group_expression(dimension: str, alias: str) -> str:
    """The SQL expression for `dimension`'s value, for the chunk table aliased as `alias`.

    `alias` is an identifier this package chooses (``c``, or a table name), never a caller's
    string, and it is checked anyway. `dimension` is looked up rather than formatted, so an
    unknown one raises here instead of reaching Postgres.
    """
    if not alias.isidentifier():
        raise ValueError(f"alias must be a bare identifier, got {alias!r}")
    try:
        template = GROUP_SQL[dimension]
    except KeyError:
        raise ValueError(
            f"unknown scope dimension {dimension!r}; expected one of {sorted(GROUP_SQL)}"
        ) from None
    return template.format(a=alias)


def folder_of(file: str) -> str:
    """The folder a root-relative `file` sits in: ``'python/notes.md'`` to ``'python'``.

    A file at the root returns ``''``, matching `GROUP_SQL`. Backslashes are normalized, because
    the indexer writes posix separators (`Path.as_posix`) while a caller on Windows may well have
    assembled a path by hand, and a mismatch there addresses a folder that does not exist.
    """
    normalized = file.replace("\\", "/").strip("/")
    return posixpath.dirname(normalized)


def _like_prefix(folder: str) -> str:
    """`folder` as a `LIKE` pattern matching everything beneath it, wildcards defused.

    ``%``, ``_`` and the escape character itself are escaped, so a folder literally named
    ``draft_1`` matches ``draft_1/...`` and NOT ``draftX1/...``. The predicate declares the escape
    character explicitly rather than relying on the server default.
    """
    escaped = folder.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "/%"


@dataclass(frozen=True)
class Scope:
    """Which region of the corpus a retrieval may draw from.

    Every field is a conjunction: a scope naming both a folder and a facet matches only chunks in
    that folder that also carry that facet. Empty (`Scope()`) matches the whole tenant, which is
    what every caller that predates scoping gets.

    `source` is the pre-existing exact-match filter, carried here so one object says what a query
    is scoped to, rather than three parameters that each say part of it.
    """

    source: str | None = None
    folder: str | None = None
    facet: str | None = None

    def __post_init__(self) -> None:
        for name in ("source", "folder", "facet"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str or None, got {type(value).__name__}")
            if not value.strip():
                # An empty string is not "no filter". Read as one it would silently widen the
                # query to the whole corpus, which is the direction that returns a confident
                # wrong answer instead of an error. `folder=""` meaning the corpus ROOT is a real
                # request and is spelled `folder="/"`, normalized below.
                raise ValueError(
                    f"{name} was empty or whitespace; pass None for 'no filter', or "
                    f"folder='/' for the corpus root"
                )

    @property
    def is_empty(self) -> bool:
        return self.source is None and self.folder is None and self.facet is None

    @property
    def normalized_folder(self) -> str | None:
        """`folder` with separators normalized and outer slashes removed; ``'/'`` becomes ``''``."""
        if self.folder is None:
            return None
        return self.folder.replace("\\", "/").strip("/")

    @property
    def normalized_facet(self) -> str | None:
        """`facet` lowercased, matching the ``lower(...)`` `GROUP_SQL` applies to the stored side."""
        return None if self.facet is None else self.facet.strip().lower()

    def predicate(
        self, alias: str, source_column: str = "source"
    ) -> tuple[str, dict[str, object]]:
        """``(sql, params)`` for this scope, as a fragment beginning with ``AND``.

        Returns ``("", {})`` when the scope is empty, so a caller can splice it unconditionally.
        Parameter names are prefixed ``scope_`` so they cannot collide with a caller's own.

        `source_column` exists because the two stores do not agree on the name: the legacy table
        calls it ``source`` and the generation table ``source_uri``. Only the SOURCE arm varies;
        the folder and facet arms read `metadata`, which both tables carry under the same name.
        Like `alias`, it is an identifier this package chooses and is checked as one anyway.
        """
        if not alias.isidentifier():
            raise ValueError(f"alias must be a bare identifier, got {alias!r}")
        if not source_column.isidentifier():
            raise ValueError(f"source_column must be a bare identifier, got {source_column!r}")
        clauses: list[str] = []
        params: dict[str, object] = {}

        if self.source is not None:
            # Match the caller-facing identifier: hits surface the root-relative
            # `metadata->>'file'`, never the absolute `source` column, so a `source=` filter
            # passed back from a hit must resolve against `file`. The `source` arm keeps legacy
            # rows (no `file` metadata) and absolute-path callers working. Same rule as
            # recall_forget.
            clauses.append(
                f"({alias}.metadata->>'file' = %(scope_source)s "
                f"OR {alias}.{source_column} = %(scope_source)s)"
            )
            params["scope_source"] = self.source

        folder = self.normalized_folder
        if folder is not None:
            if folder == "":
                # The corpus root, and ONLY the root: a file with no separator in its path. Not
                # "everything", which is what a bare prefix match on '' would have meant.
                clauses.append(f"position('/' in {alias}.metadata->>'file') = 0")
            else:
                # A folder is not itself a chunk, so this is "at or beneath it". The equality arm
                # covers a corpus whose whole relative path happens to equal the folder name.
                clauses.append(
                    f"({alias}.metadata->>'file' LIKE %(scope_folder_prefix)s ESCAPE '\\' "
                    f"OR {alias}.metadata->>'file' = %(scope_folder)s)"
                )
                params["scope_folder_prefix"] = _like_prefix(folder)
                params["scope_folder"] = folder

        facet = self.normalized_facet
        if facet is not None:
            clauses.append(f"{group_expression('facet', alias)} = %(scope_facet)s")
            params["scope_facet"] = facet

        if not clauses:
            return "", {}
        return "AND " + " AND ".join(clauses), params


def coerce_scope(scope: "Scope | None", source: str | None) -> Scope:
    """One scope from the two ways a caller can express one, refusing the ambiguous case.

    Every retrieval entry point kept its `source=` parameter when scoping arrived, so both forms
    reach the store. Passing both is refused rather than merged: a caller who sets
    ``scope=Scope(source='a')`` alongside ``source='b'`` has a bug, and silently picking one would
    answer from a region the caller never named.
    """
    if scope is None:
        return Scope(source=source) if source is not None else Scope()
    if source is not None:
        raise ValueError(
            "pass either scope= or source=, not both: they name the same filter, and merging "
            f"them would silently pick one (got scope.source={scope.source!r}, "
            f"source={source!r})"
        )
    return scope


__all__ = [
    "Dimension",
    "FACET_METADATA_KEY",
    "GROUP_SQL",
    "Scope",
    "coerce_scope",
    "folder_of",
    "group_expression",
]
