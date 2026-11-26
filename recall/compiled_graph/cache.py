"""The compile cache: accepted records per (provider, model, prompt, source, body).

A build stopped by the budget or a credit stop resumes without paying again for what it got, and
an unchanged source on Lite is never sent twice. The cache only saves money, so a damaged row or an
unreadable file is a miss and a failed write is counted rather than raised: neither may fail a build
or a write. Erasure is the exception: a forget must not leave a quote behind, so an unreadable file
is deleted and a locked one raises.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from recall.compiled_graph import anchors as anchors_module
from recall.compiled_graph import provider as provider_module
from recall.compiled_graph import records as records_module
from recall.compiled_graph.provider import ProviderConfig, url_without_credentials
from recall.compiled_graph.records import CompiledRecord, prompt_digest

#: Bumped whenever the shape a record is stored or validated in changes.
PAYLOAD_SCHEMA_VERSION = 1


def cache_key(config: ProviderConfig, source: str, body_sha256: str) -> str:
    """Everything that decides a compile's records, and nothing secret."""
    identity = {
        "url": url_without_credentials(config.base_url),
        "model": config.model,
        "prompt": prompt_digest(),
        "schema": PAYLOAD_SCHEMA_VERSION,
        "anchors": [anchors_module.ANCHOR_CHARS, anchors_module.ANCHOR_OVERLAP_CHARS],
        "validation": [
            records_module.MAX_RECORDS,
            records_module.MAX_ANCHORS_PER_RECORD,
            records_module.MAX_ENTITIES,
            records_module.BACKFILL_CHARS,
        ],
        "max_tokens": provider_module.MAX_OUTPUT_TOKENS,
        "source": source,
        "body": body_sha256,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def _encode(records: Sequence[CompiledRecord]) -> str:
    return json.dumps(
        [
            {
                "kind": record.kind,
                "fields": list(record.fields),
                "entities": list(record.entities),
                "anchor_ids": list(record.anchor_ids),
                "spans": [list(span) for span in record.spans],
            }
            for record in records
        ]
    )


def _is_strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _decode(text: str) -> list[CompiledRecord] | None:
    try:
        rows = json.loads(text)
    except ValueError:
        return None
    if not isinstance(rows, list):
        return None
    records: list[CompiledRecord] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
            return None
        spans = row.get("spans")
        if not (
            _is_strings(row.get("fields"))
            and _is_strings(row.get("entities"))
            and _is_strings(row.get("anchor_ids"))
            and isinstance(spans, list)
            and all(
                isinstance(span, list)
                and len(span) == 2
                and all(isinstance(n, int) and not isinstance(n, bool) for n in span)
                for span in spans
            )
        ):
            return None
        records.append(
            CompiledRecord(
                kind=row["kind"],
                fields=tuple(row["fields"]),
                entities=tuple(row["entities"]),
                anchor_ids=tuple(row["anchor_ids"]),
                spans=tuple((span[0], span[1]) for span in spans),
            )
        )
    return records


class CompileCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.write_failures = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create()
            self.usable = True
        except sqlite3.Error:
            self.usable = False

    def _create(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS compiled "
                "(key TEXT PRIMARY KEY, source TEXT NOT NULL, records TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS compiled_source ON compiled (source)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Commit on success and always close: Windows cannot delete a file a connection holds."""
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, key: str) -> list[CompiledRecord] | None:
        if not self.usable:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT records FROM compiled WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        return _decode(row[0]) if row is not None and isinstance(row[0], str) else None

    def put(self, key: str, source: str, records: Sequence[CompiledRecord]) -> None:
        if not self.usable:
            self.write_failures += 1
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO compiled (key, source, records) VALUES (?, ?, ?)",
                    (key, source, _encode(records)),
                )
        except sqlite3.Error:
            self.write_failures += 1

    def erase_sources(self, sources: Collection[str]) -> int:
        """Delete every row of ``sources``; a forget must not leave their quotes here.

        A file SQLite cannot read as a database is deleted whole, since its rows cannot be found
        one by one. A lock (``OperationalError``) raises: the file is sound and still holds them.
        """
        try:
            self._create()
            with self._connect() as conn:
                return sum(
                    conn.execute("DELETE FROM compiled WHERE source = ?", (source,)).rowcount
                    for source in sources
                )
        except sqlite3.OperationalError:
            raise
        except sqlite3.DatabaseError:
            self.path.unlink()
            self.usable = False
            return 0
