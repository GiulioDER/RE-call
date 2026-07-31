"""RE-call as a mem-bench axis-3 (temporal correctness) system.

Prior-work status for this package is recorded once in `__init__.py`: relocation of an existing
adapter that has already produced a published figure, not a new experiment, and `docs_search` was
unavailable (VPS2 down).

SUBMITTER-SIDE, like the isolation adapter: it imports `recall`, so it stays out of `membench/`.

**The reference time is threaded, not ignored.** `trusted_search(..., now=<datetime>)` is a public
parameter and `evaluate()` is a pure function of it, so this adapter passes the question's reference
time through. What `now` gates in RE-call is the trust VERDICT -- `_verdict` marks a hit expired if
`now > end` or not-yet-valid if `now < start`, where start/end come from a chunk's validity
frontmatter. It does not re-RANK by covering interval.

The corpus states validity in prose (`"As of 2026-03-14, the deadline is ..."`) and carries no
frontmatter, so RE-call has no structured window to evaluate here. This adapter therefore reports
exactly what the shipped library returns for a query asked as of that time. That is the honest
measurement: whether RE-call, as it ships, serves the value that was true then.

⚠️ **`known_as_of` is NOT wired here, and that is deliberate.** RE-call 0.7.0 added bi-temporal
retrieval, and it is tempting to read axis 3's low covering rate as measuring a version that
predates its own fix. It does not: `known_as_of` is TRANSACTION time (when a memory was written) and
this axis scores VALID time (which assertion's `[effective_from, effective_to)` covers the reference
date). `recall/trust.py` says so itself -- it "does not rewind supersession". Wiring it in would
change what is measured without improving what is measured. Worth testing one day as its own
pre-registered arm; not worth smuggling in as a fix.

Run it:

    python -m membench.axes.temporal.run \\
        --system benchmarks.membench.recall_temporal:RecallTemporal \\
        --out artifact.jsonl --submitter github:you --system-name RE-call \\
        --run-started <ISO> --config '{"embedder": "BAAI/bge-small-en-v1.5", "dim": 384}'

`--system-version` is deliberately absent: `system_version` below reports the live package, and
mem-bench aborts if a flag contradicts it.
"""
from __future__ import annotations

import datetime as dt
import os

from membench.axes.temporal.adapter import TemporalResponse

from benchmarks.membench import _env
from recall.index import Indexer
from recall.store import PgVectorStore
from recall.trust import trusted_search

DSN = os.environ.get("MEMBENCH_DSN", "postgresql:///membench_tmp")
TABLE = os.environ.get("MEMBENCH_TABLE", "tmp_chunks")


class RecallTemporal:
    name = "RE-call"

    @property
    def system_version(self) -> str:
        """Read from the live `recall` package -- see `_env.system_version`."""
        return _env.system_version()

    def __init__(self) -> None:
        cfg = _env.from_env()
        self._embedder = cfg.embedder
        self._rr = cfg.reranker
        self._k = cfg.k
        self._ck = cfg.candidate_k
        self._store = PgVectorStore(DSN, dim=self._embedder.dim, tenant="temporal", table=TABLE)
        self._store.ensure_schema()

    def ingest(self, docs) -> None:
        import tempfile
        from pathlib import Path

        work = Path(tempfile.mkdtemp(prefix="membench-temporal-"))
        for d in docs:
            (work / f"{d.doc_id}.md").write_text(d.text, encoding="utf-8")
        Indexer(self._store, self._embedder).index_path(work)

    def indexed_doc_ids(self) -> frozenset[str]:
        out = set()
        for chunk in self._store.iter_chunks():
            f = chunk.metadata.get("file")
            if f:
                out.add(f.rsplit(".", 1)[0])
        return frozenset(out)

    def query(self, question: str, *, reference_time: str) -> TemporalResponse:
        now = dt.datetime.fromisoformat(reference_time).replace(tzinfo=dt.UTC)
        result = trusted_search(
            self._store, self._embedder, question, now=now,
            k=self._k, candidate_k=self._ck, reranker=self._rr,
        )
        cited: list[str] = []
        # `ok` only: a hit RE-call itself marks expired or not-yet-valid is not being SERVED as
        # current, and counting it would understate the system by scoring hits it already flagged.
        for hit in result.hits:
            if getattr(hit, "verdict", "ok") != "ok":
                continue
            f = hit.chunk.metadata.get("file")
            if f:
                did = f.rsplit(".", 1)[0]
                if did not in cited:
                    cited.append(did)
        answered = not result.abstained and bool(cited)
        tokens = sum(len(h.chunk.text.split()) for h in result.hits)
        return TemporalResponse(answered=answered, cited_ids=tuple(cited), tokens=tokens)
