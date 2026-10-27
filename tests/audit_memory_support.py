"""Shared fixtures for the audit-memory record tests: the C4 golden memos in a lite store.

Contract C4 of `docs/AUDIT_MEMORY_PLAN.md`: `tests/fixtures/audit_memory/memos/` holds the exact
bytes cca-audit renders, and `expected_records.json` the `recall_record` RE-call must return for
each. Copied into the test's own folder so the stored sources are the test's, and indexed with the
offline `hashing` embedder: nothing here downloads a model or spends API credit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recall.embeddings import resolve_embedder
from recall.index import Indexer
from recall.lite import LiteStore

FIXTURES = Path(__file__).parent / "fixtures" / "audit_memory"
MEMOS = FIXTURES / "memos"
EXPECTED: dict[str, dict[str, Any]] = json.loads(
    (FIXTURES / "expected_records.json").read_text(encoding="utf-8")
)
REPO = "GiulioDER/RE-call"


def copy_memos(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for memo in MEMOS.glob("*.md"):
        # Bytes, so the copy is the golden file exactly, line endings included.
        (root / memo.name).write_bytes(memo.read_bytes())
    return root


def lite_dsn(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


def indexed_lite_store(tmp: Path, *, memos: Path | None = None) -> LiteStore:
    """A lite store holding the golden memos (or `memos`), indexed through the real `Indexer`."""
    folder = memos if memos is not None else copy_memos(tmp / "memos")
    embedder = resolve_embedder("hashing")
    store = LiteStore.from_dsn(lite_dsn(tmp / "memory.db"), dim=embedder.dim)
    Indexer(store, embedder, allow_prune=True, env={}).index_path(folder)
    return store


def memo_name(source: str) -> str:
    return Path(source).name
