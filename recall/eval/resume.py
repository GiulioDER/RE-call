"""The one resume mechanism for incremental evaluation runs: an append-only JSONL ledger.

Three incompatible mechanisms existed when this module was written, and the point of the module
is that there are now fewer, not one more:

============================  ================================================================
`benchmarks/ladder/run.py`    one JSONL line per scored instance, resume by `instance_id`,
                              truncated trailing line tolerated **and repaired**
`benchmarks/beam/run.py`      one JSONL line per scored question across N sidecars, resume by
                              `question_id`, any malformed line fatal
`recall/eval/gap_run.py`      one JSON **file** per corpus carrying a `status` marker
============================  ================================================================

The ladder mechanism is the one factored out here, because it is strictly the strongest of the
three: it survives the way an overnight run actually dies (SIGKILL or power loss mid-write, which
leaves a truncated final line) while still refusing corruption in the middle of a file, where
"truncated write" is not an available explanation. `benchmarks/beam/run.py` now delegates to it
and gains that tolerance; `benchmarks/ladder/run.py` delegates to it and keeps exactly the
behaviour it had.

`recall/eval/gap_run.py` is deliberately NOT migrated. Its unit of resume is a whole corpus
result *file* with a `status` marker, not a line per item, and its artifacts under `results/gap/`
are committed in that shape. Unifying it means rewriting a published artifact format for no
resume benefit, so it stays, and this docstring is where that decision is recorded rather than
left to be rediscovered as "a fourth mechanism appeared".

Two properties the callers depend on, and the tests pin:

- **Tolerating a torn tail and repairing it are separate decisions.** A reader that opens someone
  else's finished sidecar must not rewrite it; a writer about to append to its own file must,
  because otherwise the next row is glued onto the end of a partial line and the file stops being
  JSONL for every future reader. `repair_truncated_tail` is therefore explicit at every call
  site.
- **First occurrence of an id wins.** Resuming across several sidecars means the same question
  may appear more than once; they are the same scored item either way, and taking the first keeps
  the result independent of the order the paths are passed.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LedgerCorrupted(ValueError):
    """A malformed line that is NOT the last line of its file.

    Distinct from a truncated trailing write, which is expected and recoverable. This one means
    damage somewhere in the middle of an otherwise-flushed file, and treating it as "not yet
    recorded" would silently re-score an item whose real result is sitting unreadable on disk.
    """


@dataclass(frozen=True)
class LedgerRead:
    """Everything already on disk, plus what had to be discarded to read it."""

    #: One dict per recorded item, first occurrence of an id winning, in the order encountered.
    rows: tuple[dict[str, Any], ...]
    #: The ids `rows` covers. A `frozenset` because callers only ever ask membership of it.
    ids: frozenset[str]
    #: Paths whose final line was discarded as a truncated write. Non-empty means a previous run
    #: died mid-write; the caller may want to say so rather than resume silently.
    truncated: tuple[Path, ...]


def _identity(row: Mapping[str, Any], id_fields: tuple[str, ...]) -> str:
    """The resume key for one row.

    NUL-joined rather than concatenated: a composite key over ``("corpus", "question_id")`` must
    not let ``("ab", "c")`` and ``("a", "bc")`` collide, and two different questions sharing a
    resume key means one of them is silently never scored.
    """
    return "\x00".join(str(row[field]) for field in id_fields)


def _read_one(
    path: Path, *, id_field: tuple[str, ...], repair_truncated_tail: bool
) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    valid: list[str] = []
    truncated = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise LedgerCorrupted(
                    f"{path}: line {index + 1} of {len(lines)} is not valid JSON and is NOT the "
                    f"last line — this is corruption in the middle of the file, not a truncated "
                    f"trailing write, and must not be silently treated as 'not yet recorded'."
                ) from None
            truncated = True
            break
        if not isinstance(row, dict):
            raise LedgerCorrupted(
                f"{path}: line {index + 1} parsed as {type(row).__name__}, not a JSON object. "
                f"A ledger line is one record; this file is not a ledger."
            )
        absent = [field for field in id_field if field not in row]
        if absent:
            raise LedgerCorrupted(
                f"{path}: line {index + 1} has no {absent!r} field(s), so it cannot be resumed "
                f"from. Passing the wrong id_field for this artifact would otherwise re-run "
                f"every item while reporting a successful resume."
            )
        rows.append(row)
        valid.append(line)
    if truncated and repair_truncated_tail:
        # Without this rewrite the next appended row is glued onto the end of the partial line,
        # and the file stops being readable JSONL for every future reader, not just this run.
        #
        # newline="\n" for the same reason `append_rows` uses it: these ledgers are committed
        # artifacts, and Python's text-mode translation would turn a repair on Windows into a
        # whole-file CRLF rewrite of a file another platform wrote with "\n".
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(f"{line}\n" for line in valid))
    return rows, truncated


def ledger_id(row: Mapping[str, Any], id_field: str | Sequence[str]) -> str:
    """The resume key `read_ledger` would compute for `row`.

    Exported because a caller that wants to ask "is this item already done?" must key the question
    exactly the way the ledger keyed the answer. A caller rebuilding the key by hand is how two
    definitions of the same identity start to disagree.
    """
    return _identity(row, _normalise_id_field(id_field))


def _normalise_id_field(id_field: str | Sequence[str]) -> tuple[str, ...]:
    fields = (id_field,) if isinstance(id_field, str) else tuple(id_field)
    if not fields:
        raise ValueError("id_field must name at least one field")
    return fields


def read_ledger(
    paths: Path | Sequence[Path] | None,
    *,
    id_field: str | Sequence[str],
    repair_truncated_tail: bool = False,
) -> LedgerRead:
    """Load every recorded row from `paths`, keyed by `id_field`.

    Takes a sequence of paths because an interrupted-and-resumed run leaves one sidecar per
    attempt and the items already paid for are spread across all of them; resuming from only the
    newest re-pays for the earlier ones.

    `id_field` may name several fields, for an artifact whose items are only unique in
    combination — the promotion harness writes one ledger covering several corpora, where a
    question id is unique within its corpus and not across them.
    """
    fields = _normalise_id_field(id_field)
    if paths is None:
        candidates: list[Path] = []
    elif isinstance(paths, Path):
        candidates = [paths]
    else:
        candidates = list(paths)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    truncated: list[Path] = []
    for path in candidates:
        found, was_truncated = _read_one(
            path, id_field=fields, repair_truncated_tail=repair_truncated_tail
        )
        if was_truncated:
            truncated.append(path)
        for row in found:
            identity = _identity(row, fields)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    return LedgerRead(rows=tuple(rows), ids=frozenset(seen), truncated=tuple(truncated))


def append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows to a ledger, flushing each one. Returns how many were written.

    Flushed per row rather than per batch on purpose: an unflushed buffer is exactly the work a
    crash throws away, and this ledger exists so that a crash costs one item.

    newline="\n": text-mode translation would write CRLF on Windows, and these ledgers are
    committed artifacts. An artifact whose raw bytes depend on the OS that produced it cannot be
    compared across machines, which is the same reason `benchmarks/ladder/manifest.py` pins it.
    """
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    return written
