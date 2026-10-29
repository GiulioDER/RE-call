"""Audit records: the `recall_record` front matter object, admitted at index time and looked up
by exact key.

A memo may carry one `recall_record` key, a one-line JSON object following schema version 1
(`recall/schemas/recall_record.v1.json`). CCA writes it; RE-call stores it in chunk metadata and
returns it **exactly as authored**: no field is renamed, added, defaulted, normalised or dropped,
so the object a caller wrote and the object `lookup_records` returns compare equal as JSON values.
That equality is what lets CCA rebuild a ledger row from RE-call's answer
(`docs/AUDIT_MEMORY_DESIGN.md`, "Round-trip contract").

Three parts:

- `admit_record` runs where a document's front matter becomes chunk metadata (`recall.index` and
  `recall.generations`), and where `recall.records_files` reads a memo directory directly. A
  record that does not validate, or names a version this code does not read, is dropped from the
  metadata with a warning naming the file and the reason; the document is indexed without it.
  Never partially read.
- `match_candidates` is the match, best-of, reverse `caused` edge and ordering rule every records
  reader shares: `lookup_records` over a store's chunks, and
  `recall.records_files.match_memo_records` over a memo directory, so the two cannot disagree.
- `lookup_records` answers by key, with no embedding, no calibration and no abstention: an exact
  key match is not a relevance claim. Superseded and expired records are returned and marked,
  never hidden: `superseded_by` and `expired` are both computed at the lookup's `now`. In a code tenant a chunk carrying `code_anchor` (its file, symbol and hash) is a
  record of kind ``code`` and matches by the same rules; the chunks sharing one anchor (a split
  symbol, a module's module-level chunks) are one record, identified by its lowest-ordinal chunk,
  with every chunk in `chunk_ids`. A code anchor with a null `repo` never matches a `repo` query.

The schema is checked by a small interpreter of the keywords version 1 uses rather than by the
`jsonschema` package, which is not a dependency of the base install (it arrives only with the
`mcp` extra). `tests/test_frontmatter_recall_record.py` asserts this interpreter and `jsonschema`
agree on every golden record and on a set of mutations of them.
"""
from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib import resources
from typing import Any, Literal, NamedTuple

from recall.errors import RecallError
from recall.frontmatter import RECORD_KEY, RECORD_LABEL_KEYS, supersedes_targets, validity_bounds
from recall.observability import get_logger
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.supersession import store_supersession_graph
from recall.types import Chunk

_log = get_logger("records")

#: The only `recall_record` version this code reads. Any other is refused at index time.
RECORD_VERSION = 1
#: Chunk metadata key a code chunker writes (Session 3 of the audit memory plan): the
#: ``{repo, file, symbol, code_sha}`` of the symbol the chunk holds.
CODE_ANCHOR_KEY = "code_anchor"
#: Every kind a record can have, plus ``code`` for a code chunk's anchor.
RECORD_KINDS: frozenset[str] = frozenset(
    {"bug", "false_positive", "false_positive_retraction", "lesson", "lesson_retraction", "code"}
)
MAX_QUERY_ANCHORS = 200
MAX_RECORDS = 500
#: `superseded_by` when a successor exists that the caller's source policy does not let it read.
WITHHELD = "<withheld>"

#: Best first. `anchor_match` is the best of a record's `anchor_matches`.
MATCH_ORDER: tuple[str, ...] = ("exact", "symbol_changed", "file_only", "none")


class RecordsQueryError(ValueError, RecallError):
    """A `lookup_records` query that cannot be answered as asked (refused, never truncated)."""


# --- the schema ------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def record_schema() -> dict[str, Any]:
    """`recall_record` version 1, as shipped in the package."""
    text = resources.files("recall.schemas").joinpath("recall_record.v1.json").read_text(
        encoding="utf-8"
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


def _type_ok(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        # Draft 2020-12 reads any number with a zero fractional part as an integer, so `1.0` is one.
        if isinstance(value, float):
            return value.is_integer()
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise ValueError(f"schema type {name!r} is not one this validator reads")


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: ``1`` and ``True`` differ, as JSON Schema's `const` and `enum` require."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return bool(a == b)


def _pattern(pattern: str) -> re.Pattern[str]:
    # ECMA-262's `$` matches only at the end; Python's also before a final newline, which would
    # admit "2026-10-21\n" as a date. `\Z` is the ECMA meaning.
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1] + r"\Z"
    return re.compile(pattern)


_SUPPORTED = frozenset(
    {
        "$schema", "$id", "title", "description", "oneOf", "type", "const", "enum", "required",
        "properties", "additionalProperties", "items", "minItems", "uniqueItems", "minLength",
        "pattern", "minimum",
    }
)


def _errors(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    """Every way `value` fails `schema`, as ``path: reason`` strings. Empty when it validates."""
    unknown = set(schema) - _SUPPORTED
    if unknown:
        raise ValueError(f"schema keywords {sorted(unknown)} are not ones this validator reads")
    where = path or "record"
    if "oneOf" in schema:
        branches = [_errors(value, branch, path) for branch in schema["oneOf"]]
        passing = [b for b in branches if not b]
        if len(passing) == 1:
            return []
        if len(passing) > 1:
            return [f"{where}: matches more than one record shape"]
        if path:
            # Only the record itself chooses its shape by `kind`; a nested union (an anchor's
            # `code_sha`, a string or null) has no kind, and naming one would blame the wrong field.
            return [f"{where}: matches no allowed shape"]
        # Report the branch the record says it is, so the reason names a field, not "oneOf".
        kind = value.get("kind") if isinstance(value, dict) else None
        for branch, branch_errors in zip(schema["oneOf"], branches):
            accepted = branch.get("properties", {}).get("kind", {})
            if ("const" in accepted and _equal(accepted["const"], kind)) or any(
                _equal(k, kind) for k in accepted.get("enum", ())
            ):
                return branch_errors
        return [f"{where}: kind {kind!r} is not a record kind"]
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(value, name) for name in names):
            return [f"{where}: expected {' or '.join(names)}, got {type(value).__name__}"]
    if "const" in schema and not _equal(value, schema["const"]):
        return [f"{where}: must be {schema['const']!r}"]
    if "enum" in schema and not any(_equal(value, item) for item in schema["enum"]):
        return [f"{where}: must be one of {schema['enum']!r}"]
    errors: list[str] = []
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{where}: shorter than {schema['minLength']}")
        if "pattern" in schema and not _pattern(schema["pattern"]).search(value):
            errors.append(f"{where}: does not match {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        if value < schema["minimum"]:
            errors.append(f"{where}: less than {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{where}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems") and not _unique(value):
            errors.append(f"{where}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(_errors(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{where}: missing required field {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(_errors(item, properties[key], f"{path}.{key}" if path else key))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}: unknown field {key!r}")
    return errors


def _unique(items: Sequence[Any]) -> bool:
    """Whether no two of `items` are equal as `_equal` judges them, in linear time for scalars.

    A pairwise check was quadratic, and the schema caps no array: one memo with thousands of
    distinct `caused_by` ids held the index run (and the tenant's writer lock) for minutes. A
    scalar is keyed by its JSON type, so ``1`` and ``True`` stay distinct while ``1`` and ``1.0``
    collide, as `_equal` has them; only objects and arrays, which no version 1 array holds, are
    still compared pairwise, among themselves.
    """
    seen: set[tuple[str, object]] = set()
    containers: list[Any] = []
    for item in items:
        if isinstance(item, (dict, list)):
            if any(_equal(item, other) for other in containers):
                return False
            containers.append(item)
            continue
        key = ("bool" if isinstance(item, bool) else "scalar", item)
        if key in seen:
            return False
        seen.add(key)
    return True


def record_problem(value: Any) -> str | None:
    """Why `value` is not an admissible `recall_record`, or None when it is."""
    if isinstance(value, Mapping) and "__parse_error__" in value:
        return str(value["__parse_error__"])
    if not isinstance(value, dict):
        return "recall_record must be a JSON object"
    version = value.get("v")
    if not _equal(version, RECORD_VERSION):
        return (
            f"recall_record version {version!r} is not one this RE-call reads "
            f"(it reads version {RECORD_VERSION}); the record was refused, not partly read"
        )
    if _holds_unstorable(value):
        # JSON escapes decode to U+0000 and to unpaired surrogates, both of which PostgreSQL jsonb
        # refuses: stored, either would fail the whole index run or generation build on
        # PostgreSQL while a lite store accepted it. Refused here, so both stores index the memo
        # without it.
        return f"recall_record holds {_UNSTORABLE_TEXT}, which PostgreSQL cannot store"
    errors = _errors(value, record_schema(), "")
    return "; ".join(errors[:_MAX_REPORTED_ERRORS]) if errors else None


#: How many of a refused record's schema errors its warning names: enough to fix it, not a dump.
_MAX_REPORTED_ERRORS = 3


#: What `_holds_unstorable` looks for, as a refusal names it.
_UNSTORABLE_TEXT = "U+0000 (NUL) or an unpaired UTF-16 surrogate"
_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


def _holds_unstorable(value: object) -> bool:
    """Whether any string in `value`, key or value, holds a character PostgreSQL jsonb refuses:
    U+0000, or a lone surrogate (U+D800 to U+DFFF). Iterative: no depth limit."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _UNSTORABLE.search(item):
                return True
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def admit_record(meta: Mapping[str, object], source: str) -> dict[str, object]:
    """`meta` with its `recall_record` kept when valid, else removed with a warning naming `source`.

    The record's `name` and `description` labels (kept by the parser only beside a record) leave
    with it, so metadata never carries labels for a record it does not hold. A label is decoded
    from a JSON string, so it can hold the same unstorable text a record can; a record whose label
    does is refused with it, never kept without its label.
    """
    admitted = dict(meta)
    if RECORD_KEY not in admitted:
        return admitted
    problem = record_problem(admitted[RECORD_KEY])
    if problem is None and _holds_unstorable([admitted.get(key) for key in RECORD_LABEL_KEYS]):
        problem = f"its name or description holds {_UNSTORABLE_TEXT}, which PostgreSQL cannot store"
    if problem is None:
        return admitted
    _log.warning("%s: recall_record ignored, the document is indexed without it: %s", source, problem)
    for key in (RECORD_KEY, *RECORD_LABEL_KEYS):
        admitted.pop(key, None)
    return admitted


# --- the lookup ------------------------------------------------------------------------------


def record_parts(record: Mapping[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    """``(id, kind, anchors)`` of an admitted `recall_record`: the one reading every records reader
    uses, so a store's chunk and a memo file present a record identically."""
    anchors = record.get("anchors")
    return (
        str(record.get("id")),
        str(record.get("kind")),
        [a for a in anchors if isinstance(a, dict)] if isinstance(anchors, list) else [],
    )


def record_causes(record: Mapping[str, Any]) -> tuple[str, ...]:
    """The record ids a `recall_record`'s `caused_by` names, in order: the one reading of that
    edge `match_candidates` and `recall.related` share."""
    causes = record.get("caused_by")
    if not isinstance(causes, list):
        return ()
    return tuple(cause for cause in causes if isinstance(cause, str))


class _ParsedRecord(NamedTuple):
    """A record or code chunk as the lookup reads it."""

    id: str
    record: dict[str, Any]
    kind: str
    anchors: list[dict[str, Any]]


def _record_of(chunk: Chunk) -> _ParsedRecord | None:
    """The `_ParsedRecord` of a record or code chunk, else None."""
    meta = chunk.metadata
    record = meta.get(RECORD_KEY)
    if isinstance(record, dict) and isinstance(record.get("id"), str):
        record_id, kind, anchors = record_parts(record)
        return _ParsedRecord(record_id, record, kind, anchors)
    anchor = meta.get(CODE_ANCHOR_KEY)
    if isinstance(anchor, dict) and isinstance(anchor.get("file"), str):
        return _ParsedRecord(chunk.id, anchor, "code", [
            {"file": anchor.get("file"), "symbol": anchor.get("symbol"), "code_sha": anchor.get("code_sha")}
        ])
    return None


def _repo_of(record: Mapping[str, Any]) -> object:
    return record.get("repo")


def _match(query: Mapping[str, Any], anchor: Mapping[str, Any]) -> str:
    """How one query anchor meets one record anchor: exact, symbol_changed, file_only or none.

    `exact` is file, symbol and hash all equal; `symbol_changed` is file and symbol equal with the
    hash differing (a lesson's null hash differs from any real one); `file_only` is the file alone.
    """
    if query.get("file") != anchor.get("file"):
        return "none"
    if query.get("symbol") != anchor.get("symbol"):
        return "file_only"
    if query.get("code_sha") == anchor.get("code_sha"):
        return "exact"
    return "symbol_changed"


def _best(matches: Iterable[str]) -> str:
    return min(matches, key=MATCH_ORDER.index, default="none")


def _validated_anchors(anchors: object) -> list[dict[str, Any]]:
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes, bytearray)):
        raise RecordsQueryError("anchors must be a list of {file, symbol, code_sha} objects")
    if len(anchors) > MAX_QUERY_ANCHORS:
        raise RecordsQueryError(
            f"{len(anchors)} anchors asked for, at most {MAX_QUERY_ANCHORS} per query: split the "
            "query; nothing was truncated"
        )
    out: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping) or not isinstance(anchor.get("file"), str) or not anchor["file"]:
            raise RecordsQueryError(f"anchors[{index}] must be an object with a non-empty string `file`")
        symbol, sha = anchor.get("symbol"), anchor.get("code_sha")
        if symbol is not None and not isinstance(symbol, str):
            raise RecordsQueryError(f"anchors[{index}].symbol must be a string or null")
        if sha is not None and not isinstance(sha, str):
            raise RecordsQueryError(f"anchors[{index}].code_sha must be a string or null")
        out.append({"file": anchor["file"], "symbol": symbol, "code_sha": sha})
    return out


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RecordsQueryError(f"{name} must be a list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise RecordsQueryError(f"{name} must contain only non-empty strings")
    return list(value)


def validated_anchor_query(repo: object, anchors: object) -> list[dict[str, Any]]:
    """`anchors` checked as the anchors of a query in `repo`; raises `RecordsQueryError` otherwise.

    The one statement of what an anchor query accepts, so `lookup_records` and
    `recall.records_files.match_memo_records` refuse the same queries with the same words.
    """
    if repo is not None and (not isinstance(repo, str) or not repo):
        raise RecordsQueryError("repo must be a non-empty string")
    query_anchors = _validated_anchors(anchors)
    if repo is None:
        raise RecordsQueryError("an anchor query needs `repo`: anchors are paths inside one repository")
    return query_anchors


def validated_kinds(kinds: object) -> set[str] | None:
    """`kinds` as a set of known record kinds, or None for every kind; raises on an unknown one."""
    if kinds is None:
        return None
    wanted = set(_string_list(kinds, "kinds"))
    if unknown := wanted - RECORD_KINDS:
        raise RecordsQueryError(f"unknown kinds {sorted(unknown)}; known: {sorted(RECORD_KINDS)}")
    return wanted


def as_utc(now: datetime | None) -> datetime:
    """`now` (default: the current time) as an aware instant, a naive one read as UTC."""
    instant = now or datetime.now(UTC)
    return instant if instant.tzinfo is not None else instant.replace(tzinfo=UTC)


def _generation_id(store: object) -> str:
    reader = getattr(store, "generation_binding", None)
    if callable(reader):
        generation = reader().get("generation_id")
        if generation:
            return str(generation)
    return str(getattr(store, "generation_id", None) or "legacy")


def _scalar(value: object) -> object:
    """A front matter value as the envelope reports it: None when absent or empty."""
    return None if value in (None, "") else value


def _expired(meta: Mapping[str, Any], instant: datetime) -> bool:
    """Whether the document's `valid_until` has passed at `instant`, by the rule the trust gate and
    the indexer read it with (`validity_bounds`): a date-only value stays valid through the end of
    that day, UTC. Absent, never expired. A value that does not read as a date is expired: it is
    not proof the record is current, and the indexer refuses such a memo, so none reaches a store
    that way. Computed here once, so no consumer redoes the comparison and gets the day wrong.
    """
    value = meta.get("valid_until")
    if value in (None, ""):
        return False
    try:
        _start, end = validity_bounds({"valid_until": value})
    except ValueError:
        return True
    return end is not None and end < instant


@dataclass(frozen=True)
class RecordCandidate:
    """One record as the matcher sees it, read from a store's chunks or from a memo file.

    `meta` is the document's front matter as chunk metadata carries it; `file` is the key the
    supersession graph knows the document by. A memo read from disk has no chunk, so its
    `chunk_id` is None and `chunk_ids` is empty.
    """

    id: str
    record: dict[str, Any]
    kind: str
    anchors: list[dict[str, Any]]
    meta: Mapping[str, Any]
    source: str
    file: str
    chunk_id: str | None
    chunk_ids: list[str]


def match_candidates(
    candidates: Sequence[RecordCandidate],
    *,
    query_anchors: list[dict[str, Any]] | None,
    repo: str | None,
    wanted_ids: set[str] | None,
    wanted_kinds: set[str] | None,
    superseded_by: Callable[[RecordCandidate], str | None],
    now: datetime,
) -> list[dict[str, Any]]:
    """The C5 record entries for `candidates`, best match first: the rule every records reader shares.

    `lookup_records` calls it with a store's records and `recall.records_files.match_memo_records`
    with a memo directory's, so the two cannot disagree on a match, a best-of, the reverse
    `caused` edge or the order. `superseded_by` answers the supersession question for one
    candidate, because the two read their supersession graph from different places; `expired` is
    judged here, at `now` (aware), from the document's own `valid_until`. `caused` is built over
    every candidate before any filter applies, so a record's reverse edge does not depend on the
    query that returned it.
    """
    caused: dict[str, list[str]] = {}
    for candidate in candidates:
        for cause in record_causes(candidate.record):
            caused.setdefault(cause, []).append(candidate.id)

    # A query anchor for another file can only ever meet a record anchor as "none", which is
    # `_best`'s default, so each record anchor is compared with its own file's queries alone:
    # the same answer, without comparing every record anchor with every query anchor.
    by_file: dict[str, list[dict[str, Any]]] = {}
    for query in query_anchors or ():
        by_file.setdefault(query["file"], []).append(query)

    def queries_for(anchor: Mapping[str, Any]) -> list[dict[str, Any]]:
        file = anchor.get("file")
        return by_file.get(file, []) if isinstance(file, str) else []

    matched: list[tuple[int, str, dict[str, Any]]] = []
    for candidate in candidates:
        record_id, record, kind, meta = candidate.id, candidate.record, candidate.kind, candidate.meta
        if wanted_kinds is not None and kind not in wanted_kinds:
            continue
        if repo is not None and _repo_of(record) != repo:
            continue
        if wanted_ids is not None and record_id not in wanted_ids:
            continue
        if query_anchors is not None:
            anchor_matches: list[dict[str, Any]] = [
                {"anchor": anchor, "match": _best(_match(q, anchor) for q in queries_for(anchor))}
                for anchor in candidate.anchors
            ]
        else:
            anchor_matches = [{"anchor": anchor, "match": "none"} for anchor in candidate.anchors]
        best = _best(m["match"] for m in anchor_matches)
        if query_anchors is not None and best == "none":
            continue
        successor = superseded_by(candidate)
        targets = supersedes_targets(meta.get("supersedes"))
        matched.append(
            (
                MATCH_ORDER.index(best),
                record_id,
                {
                    "id": record_id,
                    "kind": kind,
                    "recall_record": record,
                    "name": _scalar(meta.get("name")) if kind != "code" else None,
                    "description": _scalar(meta.get("description")) if kind != "code" else None,
                    "source": candidate.source,
                    "chunk_id": candidate.chunk_id,
                    "chunk_ids": candidate.chunk_ids,
                    "valid_from": _scalar(meta.get("valid_from")),
                    "supersedes": (list(targets) if len(targets) > 1 else targets[0]) if targets else None,
                    "superseded_by": successor,
                    "valid_until": _scalar(meta.get("valid_until")),
                    "expired": _expired(meta, now),
                    "caused": sorted(caused.get(record_id, [])),
                    "anchor_matches": anchor_matches,
                    "anchor_match": best,
                },
            )
        )
    matched.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in matched]


def lookup_records(
    store: Any,
    *,
    repo: str | None = None,
    anchors: Sequence[Mapping[str, Any]] | None = None,
    ids: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    all_records: bool = False,
    limit: int | None = MAX_RECORDS,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Records matching `anchors` (with `repo`), or `ids`, or every record; the C5 envelope.

    Exactly one of `anchors`, `ids` and `all_records` is given. An anchor query returns each record
    with at least one anchor matching at `file_only` or better, best match first; an `ids` or
    `all_records` query marks every match `none`. `repo`, when given, restricts every form.

    More matches than `limit` refuses an anchor or id query (never a silent cut) and cuts an
    `all_records` query to its first `limit` records, with `truncated` true when more remain.
    `limit=None`, with `all_records` only, returns every record from the one snapshot: the full
    dump a round-trip check needs. There is no offset: separate paged calls cannot be pinned to
    one state on a store with no generation id (owner decision 2026-10-29, CCA CODE-007).
    Reads one snapshot of the store's active generation when it has one. A legacy `PgVectorStore`
    has no `snapshot()`, so it reads the records and the supersession graph in two statements, and
    an index run landing between them can pair one state's records with another's graph.

    `expired` is true when the document's `valid_until` has passed at `now` (a date-only value
    lasts through that day, UTC). `superseded_by` is what hides the record at `now` (default: the
    current time), read from the store's supersession graph the way the trust gate reads it: a successor hides its predecessor
    only inside its own validity window, a chain names its deepest valid member, and a path two
    roots hold is keyed by its source.

    With a `security_policy` (and its `access_context`), a record in a file the caller may not
    read is left out before any edge is derived, so it is never returned and never named in
    another record's `caused`; a successor the caller may not read is reported as ``"<withheld>"``
    (`WITHHELD`), never as None, which would present a superseded record as current.
    """
    forms = sum(x for x in (anchors is not None, ids is not None, bool(all_records)))
    if forms != 1:
        raise RecordsQueryError("give exactly one of anchors, ids or all_records")
    if limit is None:
        if not all_records:
            raise RecordsQueryError("an unbounded limit is only for an all_records dump")
    elif isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RECORDS:
        raise RecordsQueryError(f"limit must be an integer from 1 to {MAX_RECORDS}")
    if repo is not None and (not isinstance(repo, str) or not repo):
        raise RecordsQueryError("repo must be a non-empty string")
    query_anchors = validated_anchor_query(repo, anchors) if anchors is not None else None
    wanted_ids = set(_string_list(ids, "ids")) if ids is not None else None
    wanted_kinds = validated_kinds(kinds)
    policy: tuple[SourceSecurityPolicy, AccessContext] | None = None
    if security_policy is not None:
        if access_context is None:
            raise RecordsQueryError("access_context is required when security_policy is configured")
        policy = (security_policy, access_context)
    # Which code-anchor rows can match: none when `kinds` leaves code out, and only the asked
    # files' for an anchor query. Every chunk of a code tenant carries an anchor, so reading them
    # all for a query that can only match a few files read the whole tenant. Filtering by the
    # anchor's file keeps every chunk of a code record, so `chunk_ids` stays complete; an id query
    # still reads them all, because a code record's id is a chunk id, not a file.
    code_files: list[str] | None = None
    if wanted_kinds is not None and "code" not in wanted_kinds:
        code_files = []
    elif query_anchors is not None:
        code_files = sorted({query["file"] for query in query_anchors})

    snapshot = getattr(store, "snapshot", None)
    with snapshot() if callable(snapshot) else contextlib.nullcontext():
        generation_id = _generation_id(store)
        chunks = _record_chunks(store, code_files=code_files)
        graph = store_supersession_graph(store)
    instant = as_utc(now)

    # One record per document: a memo's chunks all carry its record. In a code tenant several
    # chunks share one anchor (a split symbol, every module-level chunk of a file), so a code
    # record is one per distinct (repo, file, symbol, code_sha). Either way the record is reported
    # from its lowest-ordinal chunk, and `chunk_ids` lists every chunk carrying it.
    groups: dict[object, list[tuple[Chunk, _ParsedRecord]]] = {}
    for chunk in sorted(chunks, key=lambda c: (_sort_ordinal(c), c.source, c.id)):
        parsed = _record_of(chunk)
        if parsed is None:
            continue
        if parsed.kind == "code":
            anchor = parsed.record
            key: object = ("code", *(json.dumps(anchor.get(f)) for f in ("repo", "file", "symbol", "code_sha")))
        else:
            key = ("memo", chunk.source)
        groups.setdefault(key, []).append((chunk, parsed))
    # The source policy applies BEFORE any edge is derived: a record the caller may not read must
    # not surface through another record's `caused` list or `superseded_by` either.
    def allowed(file: str) -> bool:
        return policy is None or policy[0].decide(file, policy[1]).allowed

    groups = {key: members for key, members in groups.items() if allowed(_file_of(members[0][0]))}
    by_source = {key: members[0] for key, members in groups.items()}
    chunk_ids = {key: [member.id for member, _parsed in members] for key, members in groups.items()}

    candidates = [
        RecordCandidate(
            id=parsed.id,
            record=parsed.record,
            kind=parsed.kind,
            anchors=parsed.anchors,
            meta=chunk.metadata,
            source=chunk.source,
            file=_file_of(chunk),
            chunk_id=chunk.id,
            chunk_ids=chunk_ids[group],
        )
        for group, (chunk, parsed) in by_source.items()
    ]

    # A path two roots hold is keyed in the graph by its absolute source, which no source policy
    # is written against. Read before the policy filter, so a denied successor still maps.
    file_of_source = {chunk.source: _file_of(chunk) for chunk in chunks}

    def successor_file(key: str) -> str | None:
        """The root-relative file a successor's graph key names, or None when it cannot be told."""
        file = file_of_source.get(key)
        if file is not None and file in graph.collided:
            return file
        posix = key.replace("\\", "/")
        if any(posix.endswith("/" + name.replace("\\", "/")) for name in graph.collided):
            # Either the source of a collided file this lookup holds no chunk of, or a file
            # nested under a collided name: which of the two cannot be told here.
            return None
        return key

    def superseded_by(candidate: RecordCandidate) -> str | None:
        successor = graph.successor(graph.key(candidate.file, candidate.source), now=instant)
        if successor is None:
            return None
        file = successor_file(successor)
        if file is None:
            # No policy can be asked about a key it cannot place, so it is withheld, as a denial is.
            return successor if policy is None else WITHHELD
        # A successor the caller may not read is withheld, never reported as None: None would
        # present a retracted record as current, and a current false positive is one CCA may
        # suppress a finding on. The policy decides on the root-relative file; the report keeps
        # the graph key, so a collided successor is not reported as the name both roots hold.
        return successor if allowed(file) else WITHHELD

    records = match_candidates(
        candidates,
        query_anchors=query_anchors,
        repo=repo,
        wanted_ids=wanted_ids,
        wanted_kinds=wanted_kinds,
        superseded_by=superseded_by,
        now=instant,
    )
    truncated = False
    if all_records:
        end = len(records) if limit is None else limit
        truncated = len(records) > end
        records = records[:end]
    elif limit is not None and len(records) > limit:
        raise RecordsQueryError(
            f"{len(records)} records match, more than the limit of {limit}: narrow the query "
            "(fewer anchors or ids, or a kinds filter); nothing was truncated"
        )
    return {
        "tenant_id": str(getattr(store, "tenant", "")),
        "generation_id": generation_id,
        "truncated": truncated,
        "records": records,
    }


def record_rows_condition(
    dialect: Literal["postgres", "sqlite"], code_files: Collection[str] | None = None
) -> tuple[str, tuple[object, ...]]:
    """The SQL condition selecting the chunks a records lookup reads, and its parameters.

    The one statement of which rows can be records (a chunk carrying `recall_record`, or a code
    chunk's `code_anchor`), shared by the legacy table's read below, `GenerationStore` and
    `LiteStore`, so the three cannot select different rows. `code_files`, when given, keeps only
    the code-anchor rows whose anchor names one of those files; empty, it keeps none.
    """
    files = list(code_files or ())
    parts: list[str] = []
    params: list[object] = []
    if dialect == "postgres":
        parts.append("metadata ? %s")
        params.append(RECORD_KEY)
        if code_files is None:
            parts.append("metadata ? %s")
            params.append(CODE_ANCHOR_KEY)
        elif files:
            parts.append("metadata -> %s ->> 'file' = ANY(%s)")
            params.extend([CODE_ANCHOR_KEY, files])
    else:
        parts.append("json_type(metadata, ?) IS NOT NULL")
        params.append(f"$.{RECORD_KEY}")
        if code_files is None:
            parts.append("json_type(metadata, ?) IS NOT NULL")
            params.append(f"$.{CODE_ANCHOR_KEY}")
        elif files:
            parts.append(f"json_extract(metadata, ?) IN ({', '.join('?' * len(files))})")
            params.extend([f"$.{CODE_ANCHOR_KEY}.file", *files])
    return "(" + " OR ".join(parts) + ")", tuple(params)


def _record_chunks(store: Any, *, code_files: Collection[str] | None = None) -> list[Chunk]:
    """Every chunk carrying an audit record or a code anchor, without passage text; with
    `code_files`, only the code anchors in those files (`record_rows_condition`).

    `LiteStore` and `GenerationStore` answer this themselves. The legacy PostgreSQL store is asked
    here, through its own connection helper, rather than by a method on `PgVectorStore`: that
    module is bound by the AML release manifest until the official runs end (2026-10-30), so it
    cannot change. An audit tenant holds hundreds of records, so matching happens in Python.
    """
    reader = getattr(store, "record_chunks", None)
    if callable(reader):
        return list(reader(code_files=code_files))
    from recall.store import PgVectorStore

    if not isinstance(store, PgVectorStore):
        raise RecordsQueryError(f"{type(store).__name__} cannot answer a records lookup")
    table, tenant = store.table, store.tenant
    condition, params = record_rows_condition("postgres", code_files)
    rows = store._with_retry(
        lambda conn: conn.execute(
            f"SELECT id, source, metadata FROM {table} "  # noqa: S608  # a validated identifier
            f"WHERE tenant_id = %s AND {condition} ORDER BY id",
            (tenant, *params),
        ).fetchall()
    )
    return [Chunk(id=row[0], source=row[1], text="", metadata=row[2] or {}) for row in rows]


def _file_of(chunk: Chunk) -> str:
    value = chunk.metadata.get("file")
    return value if isinstance(value, str) and value else chunk.source


def _sort_ordinal(chunk: Chunk) -> int:
    value = chunk.metadata.get("ord")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "CODE_ANCHOR_KEY",
    "MATCH_ORDER",
    "MAX_QUERY_ANCHORS",
    "MAX_RECORDS",
    "RECORD_KINDS",
    "RECORD_VERSION",
    "RecordCandidate",
    "RecordsQueryError",
    "WITHHELD",
    "admit_record",
    "as_utc",
    "lookup_records",
    "match_candidates",
    "record_causes",
    "record_parts",
    "record_problem",
    "record_rows_condition",
    "record_schema",
    "validated_anchor_query",
    "validated_kinds",
]
