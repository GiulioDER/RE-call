"""RE-call adapter for the Bench'd harness (github.com/benchdai/harness).

This file is the source of truth in the RE-call repository; at run time it is copied into a
harness clone as ``benchd_harness/adapters/recall_adapter.py`` and registered in
``benchd_harness/adapters/__init__.py``. It follows the same faithfulness rules as
``recall_interop.memory_benchmarks.RecallBackend`` (the mem0-harness seam), restated where they
apply because each one changes a number.

Bench'd calls, per benchmark item (``benchd_harness/runner.py``):

    reset() -> ingest(turns) -> recall(query) -> str

and the returned string is the ONLY thing their locked answerer sees. Their turn dicts carry
``role``, ``content`` and a ``metadata`` dict; the harness hands the same dicts to every adapter,
so using ``metadata.session_date_time`` (LoCoMo session dates) is fair game and is the only way
temporal questions are winnable. One field is deliberately refused: LongMemEval's
``metadata.has_answer`` is an oracle evidence flag, and reading it would be cheating, so this
adapter never touches it.

Configuration is via environment variables so a run's config is visible in its command line:

    RECALL_BENCHD_DSN          required, pgvector DSN
    RECALL_BENCHD_EMBEDDER     default "fastembed" (see recall_interop.resolve_embedder)
    RECALL_BENCHD_TOP_K        default "5"; chunks returned per recall
    RECALL_BENCHD_GRANULARITY  "session" (default) or "turn"; document unit at ingest
    RECALL_BENCHD_ABSTAIN      "honour" (default) or "suppress"; honoured abstention returns ""
    RECALL_BENCHD_INGEST_CACHE "1" (default) or "0"; skip re-ingest when the turn list is
                               byte-identical to what is already indexed (Bench'd re-ingests the
                               same conversation for every one of its questions; RE-call's
                               ingestion is deterministic, so the index state is identical)
    RECALL_BENCHD_TABLE        default "benchd_bench_chunks"; never recall_chunks, a benchmark
                               must not touch or be polluted by a real corpus
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter

TENANT = "benchd-bench"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class RecallAdapter(BaseAdapter):
    """RE-call behind Bench'd's ingest/recall interface."""

    def __init__(self, adapter_config: Optional[Dict[str, Any]] = None) -> None:
        self._config = adapter_config or {}
        self._store: Any = None
        self._embedder: Any = None
        self._workspace: Optional[Path] = None
        self._seq = 0
        self._indexed_hash: Optional[str] = None
        self._pending_reset = False

        self._top_k = int(_env("RECALL_BENCHD_TOP_K", "5"))
        self._granularity = _env("RECALL_BENCHD_GRANULARITY", "session")
        self._abstain = _env("RECALL_BENCHD_ABSTAIN", "honour")
        self._cache = _env("RECALL_BENCHD_INGEST_CACHE", "1") == "1"
        self._table = _env("RECALL_BENCHD_TABLE", "benchd_bench_chunks")

    @property
    def name(self) -> str:
        return "re-call"

    @property
    def version(self) -> Optional[str]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("recall-rag")
        except PackageNotFoundError:
            return "source"

    # ------------------------------------------------------------------ lifecycle

    def setup(self) -> None:
        dsn = self._config.get("dsn") or os.environ.get("RECALL_BENCHD_DSN")
        if not dsn:
            raise RuntimeError("RecallAdapter requires RECALL_BENCHD_DSN")

        from recall_interop.memory_benchmarks import resolve_embedder
        from recall.store import PgVectorStore

        self._embedder = resolve_embedder(_env("RECALL_BENCHD_EMBEDDER", "fastembed"))
        self._store = PgVectorStore(
            dsn, dim=self._embedder.dim, tenant=TENANT, table=self._table
        )
        self._store.__enter__()
        self._store.ensure_schema()
        self._wipe()  # a fresh run must not inherit a previous run's rows
        self._workspace = Path(tempfile.mkdtemp(prefix="benchd-recall-"))

    def teardown(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            finally:
                self._store = None
        if self._workspace is not None:
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    def reset(self) -> None:
        """Bench'd resets before every item. With the cache on, the wipe is deferred to
        ingest(), which skips it when the incoming turn list is identical to what is indexed."""
        if self._cache:
            self._pending_reset = True
        else:
            self._wipe()
            self._indexed_hash = None

    def _wipe(self) -> None:
        sources = sorted({c.source for c in self._store.iter_chunks()})
        if sources:
            self._store.delete_sources(sources)
        if self._workspace is not None:
            for f in self._workspace.glob("*.md"):
                f.unlink()
        self._seq = 0

    # ------------------------------------------------------------------ ingest

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        blob = json.dumps(
            [
                {
                    "role": t.get("role"),
                    "content": t.get("content"),
                    # has_answer is deliberately excluded from the hash AND from the documents:
                    # it is LongMemEval's oracle flag and no memory system may see it.
                    "session_index": (t.get("metadata") or {}).get("session_index"),
                    "session_date_time": (t.get("metadata") or {}).get("session_date_time"),
                    "speaker": (t.get("metadata") or {}).get("speaker"),
                }
                for t in turns
            ],
            sort_keys=True,
        )
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()

        if self._cache and self._pending_reset and digest == self._indexed_hash:
            self._pending_reset = False
            return
        if self._pending_reset:
            self._wipe()
            self._pending_reset = False

        docs = (
            self._docs_by_session(turns)
            if self._granularity == "session"
            else self._docs_by_turn(turns)
        )

        assert self._workspace is not None
        paths: List[Path] = []
        for body in docs:
            if not body.strip():
                continue
            self._seq += 1
            path = self._workspace / f"m{self._seq:06d}.md"
            path.write_text(body, encoding="utf-8")
            paths.append(path)

        if paths:
            from recall.index import Indexer

            # files= rather than a workspace re-glob: one pass, and _prune_vanished deletes
            # nothing that is still on disk.
            Indexer(self._store, self._embedder).index_path(self._workspace, files=paths)
        self._indexed_hash = digest

    @staticmethod
    def _line(t: Dict[str, Any]) -> str:
        meta = t.get("metadata") or {}
        who = meta.get("speaker") or t.get("role", "user")
        return f"{who}: {t.get('content', '')}"

    def _docs_by_session(self, turns: List[Dict[str, Any]]) -> List[str]:
        """One document per session, headed by the session date when the dataset provides one."""
        docs: List[str] = []
        current_key: Any = object()
        lines: List[str] = []
        header = ""
        for t in turns:
            meta = t.get("metadata") or {}
            key = meta.get("session_index")
            if key != current_key:
                if lines:
                    docs.append(header + "\n".join(lines))
                current_key = key
                lines = []
                date = meta.get("session_date_time")
                header = f"Session dated {date}:\n" if date else ""
            lines.append(self._line(t))
        if lines:
            docs.append(header + "\n".join(lines))
        return docs

    def _docs_by_turn(self, turns: List[Dict[str, Any]]) -> List[str]:
        """One document per turn, each carrying its own session date, so a retrieved chunk can
        never be separated from the date temporal questions need."""
        docs = []
        for t in turns:
            meta = t.get("metadata") or {}
            date = meta.get("session_date_time")
            prefix = f"[{date}] " if date else ""
            docs.append(prefix + self._line(t))
        return docs

    # ------------------------------------------------------------------ recall

    def recall(self, query: str) -> str:
        from recall.retriever import DEFAULT_CANDIDATE_K
        from recall.eval._research_trust import research_search
        from recall_interop.memory_benchmarks import _bench_calibration

        result = research_search(
            self._store,
            self._embedder,
            query,
            k=self._top_k,
            candidate_k=max(self._top_k, DEFAULT_CANDIDATE_K),
            calibration=_bench_calibration(self._embedder),
        )
        if result.abstained and self._abstain == "honour":
            # The answerer will say "Insufficient information in memory." and the locked judge
            # scores that INCORRECT. Honouring it anyway is the library's real behaviour; the
            # config knob exists so the choice is explicit either way.
            return ""
        return "\n\n".join(hit.chunk.text for hit in result.hits)
