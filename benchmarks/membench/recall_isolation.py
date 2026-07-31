"""RE-call as a mem-bench axis-2 (tenant isolation) system.

Prior-work status for this package is recorded once in `__init__.py`: relocation of an existing
adapter that has already produced a published figure, not a new experiment, and `docs_search` was
unavailable (VPS2 down).

SUBMITTER-SIDE CODE. It imports `recall`, so it deliberately does NOT live in `membench/` -- that
package is stdlib-only and the whole $0 claim rests on it staying that way. mem-bench's runner takes
`--system module:Factory`, which is the seam that lets a vendor's adapter live in the vendor's repo.

RE-call's isolation is real and it is in the DATABASE, not in application code: `PgVectorStore`
binds one tenant per store, sets a per-connection GUC that a Postgres RLS policy reads, carries
`tenant_id` in the primary key, and leads every hot-path predicate with it. So a foreign document is
not filtered out of the results -- it is never a candidate. That is the pre-filter shape axis 2's
T3 was built to distinguish from post-hoc scrubbing.

One store per tenant, which is what the library's own docstring says a multi-tenant server does.

Run it:

    python -m membench.axes.isolation.run \\
        --system benchmarks.membench.recall_isolation:RecallIsolation \\
        --out artifact.jsonl --submitter github:you --system-name RE-call \\
        --run-started <ISO> --config '{"embedder": "BAAI/bge-small-en-v1.5", "dim": 384}'

`--system-version` is deliberately absent: `system_version` below reports the live package, and
mem-bench aborts if a flag contradicts it.
"""
from __future__ import annotations

import os

from membench.axes.isolation.adapter import TenantResponse

from benchmarks.membench import _env
from recall.index import Indexer
from recall.store import PgVectorStore
from recall.trust import trusted_search

DSN = os.environ.get("MEMBENCH_DSN", "postgresql:///membench_iso")
TABLE = os.environ.get("MEMBENCH_TABLE", "iso_chunks")


class RecallIsolation:
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
        self._stores: dict[str, PgVectorStore] = {}
        self._workdirs: list[str] = []

    def _store(self, tenant: str) -> PgVectorStore:
        if tenant not in self._stores:
            s = PgVectorStore(DSN, dim=self._embedder.dim, tenant=tenant, table=TABLE)
            s.ensure_schema()
            self._stores[tenant] = s
        return self._stores[tenant]

    def ingest(self, docs, *, tenant: str) -> None:
        """ADD, never replace: axis 2 needs every tenant resident at once or there is nothing to
        leak from. One store per tenant means the RLS policy does the isolating."""
        import tempfile
        from pathlib import Path

        work = Path(tempfile.mkdtemp(prefix=f"membench-iso-{tenant}-"))
        self._workdirs.append(str(work))
        for d in docs:
            # The doc_id IS the filename, so `indexed_doc_ids` and `cited_ids` speak the same
            # vocabulary the manifest does. Opaque hex, so nothing here is parseable to a tenant.
            (work / f"{d.doc_id}.md").write_text(d.text, encoding="utf-8")
        Indexer(self._store(tenant), self._embedder).index_path(work)

    def indexed_doc_ids(self, *, tenant: str | None = None) -> frozenset[str]:
        tenants = [tenant] if tenant else list(self._stores)
        out: set[str] = set()
        for t in tenants:
            for chunk in self._store(t).iter_chunks():
                f = chunk.metadata.get("file")
                if f:
                    out.add(f.rsplit(".", 1)[0])
        return frozenset(out)

    def query(self, question: str, *, tenant: str) -> TenantResponse:
        result = trusted_search(
            self._store(tenant), self._embedder, question,
            k=self._k, candidate_k=self._ck, reranker=self._rr,
        )
        cited: list[str] = []
        for hit in result.hits:
            f = hit.chunk.metadata.get("file")
            if f:
                did = f.rsplit(".", 1)[0]
                if did not in cited:
                    cited.append(did)
        answered = not result.abstained and bool(result.hits)
        tokens = sum(len(h.chunk.text.split()) for h in result.hits)
        return TenantResponse(answered=answered, cited_ids=tuple(cited), tokens=tokens)
