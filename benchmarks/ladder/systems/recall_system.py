"""The RE-call reference adapter: `MemorySystem` implemented over this library's own stack.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Wiring mirrors `recall/eval/locomo.py:160-243` (`write_conversation_corpus`, `index_conversation`)
deliberately: documents are written to disk and routed through `Indexer.index_path`, the same
chunking/hashing/frontmatter path every real corpus takes. A bespoke in-memory loader would
measure a code path no user runs — that is the reason the sibling file writes files at all.

**RE-call runs at shipped defaults here.** No reranker, no tuned `candidate_k`, no non-default
embedder — `FastEmbedEmbedder` (local, no API key) and the library's own retrieval defaults. A
tuned variant is a separately labelled arm and never the headline (`SUITE-DESIGN.md` rule 4); this
file is the headline arm and must not quietly become the tuned one.

⚠️ **"Shipped defaults" cannot include the shipped TRUST policy, and that is not a loophole.**
`trusted_search` defaults to strict, which refuses any corpus without a published calibration
artifact — correct for a serving path and fatal for a benchmark, since an uncertified corpus is
what a benchmark measures. Read literally, "no overrides" made this arm refuse every question with
`TrustRefusal: INDEX_NOT_READY` rather than score at shipped defaults, so the rule defeated itself.
Retrieval therefore goes through `benchmarks._trust.bench_search`, which is development mode plus
an EXPLICIT calibration at the library's own untuned 0.50 floor. That is exactly the configuration
`benchmarks/ladder/report.py` already discloses as the one this arm's published numbers ran with
(`UNCALIBRATED_BGE_SMALL_FLOOR`), so it restores the documented behaviour rather than changing it.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from recall.cache import EmbeddingCache
from recall.embeddings import Embedder, FastEmbedEmbedder
from recall.index import Indexer
from recall.store import PgVectorStore

from benchmarks._trust import bench_search
from benchmarks.ladder.adapter import Document, Response

_log = logging.getLogger(__name__)

#: This adapter's own fixed table/tenant. One `RecallSystem` instance owns this table exclusively
#: for the life of the run — see the non-empty-table guard in `__init__` for why a table shared
#: with anything else (a stale table from a crashed prior run, a corpus another script indexed)
#: must be refused rather than silently reused.
DEFAULT_TABLE = "ladder_recall_chunks"
DEFAULT_TENANT = "ladder"

#: Characters that survive `_doc_id_to_filename` unescaped. Everything else — including `/` and
#: `:`, both of which appear in ladder doc ids (`"{cluster_id}/{dia_id}"`, and `dia_id` itself is
#: `"D1:3"`) — is percent-hex escaped. The escape character itself (`_`) is therefore ALWAYS
#: escaped too, which is what keeps the mapping invertible: an unescaped run can never contain the
#: prefix of an escape sequence. Mirrors `_dia_id_to_filename` in recall/eval/locomo.py:125,
#: generalised because doc ids here also carry the cluster segment that file lacks.
_SAFE_CHAR = re.compile(r"[A-Za-z0-9.\-]")
_ESCAPE = re.compile(r"_([0-9a-f]{2})")


def _doc_id_to_filename(doc_id: str) -> str:
    if not doc_id:
        raise ValueError("doc_id must be non-empty")
    out = []
    for ch in doc_id:
        if _SAFE_CHAR.fullmatch(ch):
            out.append(ch)
        else:
            code = ord(ch)
            if code > 0xFF:
                raise ValueError(f"doc_id contains a non-Latin-1 character, unsupported: {doc_id!r}")
            out.append(f"_{code:02x}")
    return "".join(out) + ".md"


def _filename_to_doc_id(filename: str) -> str:
    """Inverse of `_doc_id_to_filename`."""
    stem = Path(filename).stem
    return _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), stem)


class RecallSystem:
    """`MemorySystem` over RE-call's own `PgVectorStore` + `HybridRetriever` + trust layer.

    One instance owns `table` exclusively for its whole life. `ingest()` REPLACES the corpus each
    call — invariant 1 (Task 8) exists precisely to catch a system that merges instead, and this
    adapter is deliberately built so it cannot pass that check by accident: every call deletes the
    exact source rows the PREVIOUS call wrote, before the new ones are indexed.
    """

    name = "recall"

    def __init__(
        self,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        table: str = DEFAULT_TABLE,
        tenant: str = DEFAULT_TENANT,
        cache_path: str | Path | None = None,
    ) -> None:
        self._embedder = embedder or FastEmbedEmbedder()
        self._store = PgVectorStore(dsn, dim=self._embedder.dim, tenant=tenant, table=table)
        self._store.ensure_schema()
        # A non-empty table here is either a stale leftover from a crashed prior run or a table
        # this adapter is not the sole owner of. Either way, this ingest()'s "replace" contract
        # (and invariant 1, which depends on it) can only be trusted against a table THIS instance
        # started empty and has tracked every write to since.
        #
        # This check is inherently TOCTOU across genuinely concurrent processes sharing --dsn: it
        # cannot atomically claim the table, it can only observe it was empty a moment ago. The
        # fix for that is per-run isolation (a distinct --table/--tenant per run — both scope
        # `count()`/`delete_sources()`, see PgVectorStore), not a tighter check here.
        if self._store.count():
            raise RuntimeError(
                f"table {table!r} (tenant {tenant!r}) already holds "
                f"{self._store.count()} row(s). RecallSystem must own an empty table+tenant "
                f"pair: pass a fresh --table and/or --tenant (both are run.py flags), or drop "
                f"this one first. Reusing it silently would let a prior run's rows masquerade as "
                f"this run's ingest, defeating invariant 1."
            )

        # A fixed name in the shared OS temp dir is a predictable, world-writable path another
        # user on the same host can pre-create or symlink (CWE-377/CWE-59). `tenant` already
        # identifies this run (`run.py --tenant`, threaded above), so it is what gives the cache
        # a per-run identity the caller controls without inventing a new flag; a caller that wants
        # full control can still pass `cache_path` directly. The cache's value is surviving across
        # every `ingest()` call on THIS instance, so this directory is made once here, not per
        # call — `mkdtemp` is the same pattern `ingest()` already uses for its work directories,
        # and (on POSIX) creates it 0700, owner-only.
        self._owns_cache_dir = cache_path is None
        if cache_path is not None:
            self._cache_path = Path(cache_path)
        else:
            self._cache_dir = Path(tempfile.mkdtemp(prefix="ladder-recall-cache-"))
            self._cache_path = self._cache_dir / f"{tenant}_embed_cache.sqlite"
        self._cache = EmbeddingCache(self._cache_path)
        # Exactly the absolute source paths the last `ingest()` wrote — the "replace" list a fresh
        # `ingest()` deletes before writing its own. Not the same thing as "what's in the DB right
        # now": this is bookkeeping for the delete, not a cache of query results.
        self._prev_sources: list[str] = []
        #: Texts actually handed to the embedder (cache MISSES only — `embed_with_cache` never
        #: calls through on a hit). Exposed so a caller can verify the cache is doing its job
        #: instead of trusting that it is: a silently-missing cache looks exactly like a slow run.
        self.embed_calls_total = 0
        self.texts_embedded_total = 0
        #: Number of `ingest()` calls made on this instance — i.e. distinct corpus states. Not
        #: used by any correctness check; exposed purely so a caller (the smoke run, an operator
        #: watching an overnight job) can see the runner is really batching by state and not
        #: silently falling back to one ingest per instance.
        self.ingest_calls = 0
        real_embed = self._embedder.embed

        def _counting_embed(texts: list[str]) -> list[list[float]]:
            self.embed_calls_total += 1
            self.texts_embedded_total += len(texts)
            return real_embed(texts)

        self._embedder.embed = _counting_embed  # type: ignore[method-assign]

    def close(self) -> None:
        self._cache.close()
        self._store.close()
        if self._owns_cache_dir:
            shutil.rmtree(self._cache_dir, ignore_errors=True)

    def ingest(self, docs: Iterable[Document]) -> None:
        """Replace the corpus with `docs`: delete every row the previous call wrote, then index.

        Each call writes to a FRESH temp directory and discards it once indexed, so the delete
        list is exact — the rows to remove are exactly `self._prev_sources`, no directory diffing
        against disk state required. `Indexer.index_path`'s own same-path/same-hash skip cannot
        fire across calls (the path is new every time), so every call pays a database write per
        surviving document — cheap, and exactly what 1b in the brief says an adapter may cost.
        What must NOT be repaid is the embedding: `self._cache` is content-addressed
        (embedder name, dim, text), shared across every `ingest()` call on this instance, so an
        unchanged turn's vector is computed once across the whole run, not once per corpus state.

        Rows are deleted BEFORE the new ones are indexed, so a failure partway through leaves
        `self._prev_sources` stale relative to the store — acceptable here because any exception
        out of `ingest()` is meant to stop the whole run (Task 8's invariants), not be caught and
        retried with this instance.
        """
        docs = list(docs)
        self.ingest_calls += 1
        workdir = Path(tempfile.mkdtemp(prefix="ladder-recall-"))
        try:
            written_paths: list[str] = []
            for doc in docs:
                path = workdir / _doc_id_to_filename(doc.doc_id)
                path.write_text(doc.text, encoding="utf-8")
                written_paths.append(str(path))

            if self._prev_sources:
                self._store.delete_sources(self._prev_sources)

            before_embed = self.texts_embedded_total
            Indexer(self._store, self._embedder, cache=self._cache).index_path(workdir)
            _log.info(
                "ingested %d doc(s) into %r: %d text(s) embedded (cache miss), %d served from "
                "the embedding cache",
                len(docs), self._store.table, self.texts_embedded_total - before_embed,
                len(docs) - (self.texts_embedded_total - before_embed),
            )
            self._prev_sources = written_paths
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def indexed_doc_ids(self) -> frozenset[str]:
        ids = set()
        for chunk in self._store.iter_chunks():
            filename = chunk.metadata.get("file")
            if filename:
                ids.add(_filename_to_doc_id(filename))
        return frozenset(ids)

    def query(self, question: str) -> Response:
        """Ask RE-call's own agent-facing entry point, at its shipped retrieval defaults.

        The trust policy is the one documented exception, and the module docstring says why: the
        shipped policy refuses an uncertified corpus, which is the only kind a benchmark has.
        `bench_search` supplies development mode and the explicit 0.50 floor this arm's report
        already discloses; `k`, `candidate_k` and the reranker are still untouched defaults.
        """
        result = bench_search(self._store, self._embedder, question)
        # Read the score BEFORE the abstention branch. An abstained result still carries hits and
        # still has a top-1 cosine, and those are exactly the rows the threshold sweep needs.
        # Taking it only on the answered path would silently make the sweep blind to abstentions.
        top_cosine = max((h.cosine for h in result.hits), default=None)
        if result.abstained or not result.hits:
            return Response(answer=None, top_cosine=top_cosine)
        ok_hits = [h for h in result.hits if h.verdict == "ok"]
        top = ok_hits[0] if ok_hits else result.hits[0]
        cited: list[str] = []
        for hit in ok_hits:
            filename = hit.chunk.metadata.get("file")
            if not filename:
                continue
            doc_id = _filename_to_doc_id(filename)
            if doc_id not in cited:
                cited.append(doc_id)
        answer = top.chunk.text
        # RE-call has no LLM in this path (recall/eval/locomo.py's own header note) — `tokens` is
        # therefore not an API-reported count, and no tokenizer is a project dependency to reach
        # for one. A whitespace word count is a real, measured quantity (not a hardcoded 0) and is
        # what "measured" here can mean without inventing a token boundary this library never
        # draws.
        tokens = len(answer.split())
        return Response(answer=answer, cited_ids=tuple(cited), tokens=tokens, top_cosine=top_cosine)
