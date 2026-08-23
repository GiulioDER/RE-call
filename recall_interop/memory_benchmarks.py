"""`RecallBackend` — a `Mem0Client`-shaped façade over RE-call, for THEIR benchmark harness.

The seam
--------
`mem0ai/memory-benchmarks` talks to its memory system through exactly two async methods
(``benchmarks/locomo/run.py:417`` and ``benchmarks/beam/run.py:690``)::

    await mem0.add(messages, user_id, timestamp=<epoch>)   ->  {"results": [...]}   (ingest)
    await mem0.search(question, user_id, top_k=200)        ->  [{"memory", "created_at", ...}]

Everything downstream of ``search`` — the answerer prompt, the judge prompt, the 10/20/50/200
cutoffs, the metric — is theirs and is not touched. This class implements that pair (plus
``delete_user`` / ``close`` / the async-context protocol their runners use) on top of RE-call.

Four adapter decisions, stated up front because each one changes a number
------------------------------------------------------------------------
1. **One document per ``add()`` call, text verbatim.** Their runner chunks the transcript itself
   (LOCOMO: 1 turn per call; BEAM: 2) and hands each chunk to the memory system. Mem0 runs an LLM
   fact-extraction pass over that chunk; RE-call indexes the chunk as a document. The document
   body is the message content **exactly as Mem0 receives it** — in particular the session date is
   NOT injected into the text, because Mem0's extractor does not see it either. Both systems get
   the date the same way their harness gives it: as ``timestamp`` on ingest, surfaced as
   ``created_at`` on search, which their answerer prompt prints beside each memory.

2. **``created_at`` is carried in the filename.** RE-call's frontmatter parser keeps only its own
   validity keys (`recall.frontmatter.VALIDITY_KEYS`), so an arbitrary ``created_at:`` line would
   be silently dropped, and writing the ingest timestamp into ``valid_from`` instead would enrol
   every memory in the trust layer's validity window — changing what the trust layer *decides*,
   not merely what it reports. The timestamp therefore rides in the document filename, which
   `recall.index.Indexer` does preserve (``metadata["file"]``), and is decoded back on search.
   No RE-call library code is modified to make this benchmark work.

3. **``score`` preserves RE-call's ranking, and is not the cosine.** Their harness re-sorts every
   result list by ``score`` descending and then slices ``[:cutoff]``, so whatever goes in ``score``
   *is* the ranking. RE-call ranks by RRF fusion of a dense and a sparse leg while each hit carries
   its **true dense cosine** as its score (`recall.retriever.HybridRetriever.search`): handing them
   the cosine would let their sort silently discard the sparse leg and re-rank RE-call by dense
   similarity alone, which is not RE-call's ranking and would change which memories survive the
   top-10 cutoff. ``score`` is therefore a strictly-decreasing rank surrogate; the real numbers
   (cosine, confidence, trust verdict, RE-call's own rank) are reported verbatim in ``score_debug``,
   which their harness passes through into the per-question artifact.

4. **The candidate pool is widened to the requested ``top_k``.** RE-call's default per-leg pool is
   ``DEFAULT_CANDIDATE_K = 20``, so the fused pool holds at most ~40 distinct chunks and a request
   for ``top_k=200`` would return ~40 — RE-call would appear to lose at their budget for a reason
   that has nothing to do with retrieval quality. `recall.trust.trusted_search` exposes
   ``candidate_k`` precisely so a caller running at a different budget can hold the pool to it.
   The pool actually used is recorded in `describe`.

What is NOT changed to suit the benchmark
-----------------------------------------
Abstention. `trusted_search` is RE-call's agent-facing entry point and its ``abstained`` flag is
honoured: when RE-call abstains, ``search`` returns an **empty list**, their answerer sees
"(No relevant memories found)", and their prompt — which forbids saying "not specified" — will
usually produce a wrong answer that scores 0. On LOCOMO categories 1-4 (all answerable) abstention
can only ever cost points. It stays on because it is the library's real behaviour, and a benchmark
run that quietly disabled the one mechanism RE-call exists for would be measuring something this
repo does not ship.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Table this adapter writes. Never `recall_chunks`: a benchmark run must not be able to touch, or
#: be polluted by, a real corpus sharing the same database.
BENCH_TABLE = "their_bench_chunks"

#: Prefix of the per-conversation tenant. Mirrors the `user_id` their harness generates, so a stray
#: row is recognisable and one conversation's turns cannot answer another's questions.
TENANT_PREFIX = "their-bench-"

#: Encodes the ingest timestamp into the document filename — see decision 2 in the module docstring.
#: ``NONE`` when their runner supplied no timestamp, so the absence is explicit rather than a 0 epoch
#: silently dated 1970 (their answerer prints the date it is given, and 1970 would be a fabrication).
_TS_RE = re.compile(r"__ts(?P<ts>\d+|NONE)\.md$")

#: The one format their `benchmarks.locomo.prompts._to_human_date` parses without falling back to a
#: bare date slice. Written in UTC because their harness derives the epoch in UTC too.
_ISO_FMT = "%Y-%m-%dT%H:%M:%S"


def _iso_from_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(_ISO_FMT)


def resolve_embedder(name: str) -> Any:
    """Embedder by name. ``fastembed`` (default) is the free local bge-small — no API, no spend.

    Deliberately the same resolver shape `recall.eval.locomo._make_embedder` uses, so a
    their-harness run can be pinned to the identical embedder as a RE-call-harness run and the two
    differ only in the harness. ``hashing`` exists for tests: it needs no model download.
    """
    if name == "hashing":
        from recall.embeddings import HashingEmbedder

        return HashingEmbedder(dim=64)
    if name.startswith("st:"):
        from recall.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(name[3:])
    if name.startswith("fastembed:"):
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(model_name=name[len("fastembed:"):])
    if name == "fastembed":
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise ValueError(
        f"unknown embedder {name!r}: use 'fastembed', 'fastembed:<model>', 'st:<model>' or "
        "'hashing'. A paid embedder is deliberately not routable here — this arm is $0 by "
        "construction, and every published RE-call number was measured on a local model."
    )


class _UserScope:
    """Everything RE-call needs for one of their ``user_id``s: a tenant, a store, a workspace."""

    __slots__ = ("tenant", "workspace", "store", "seq")

    def __init__(self, tenant: str, workspace: Path, store: Any) -> None:
        self.tenant = tenant
        self.workspace = workspace
        self.store = store
        self.seq = 0


def _bench_calibration(embedder: Any) -> Any:
    """The benchmark's operating point, stated OUT LOUD instead of inherited by accident.

    This backend's abstention was previously produced by `_UNCALIBRATED` — the library's 0.50
    default — reached because no calibration was passed. That made every published abstention
    figure a measurement of an uncertified constant that nothing in the call site mentioned.

    Naming it here changes no number: it is the same threshold the backend has always used. What
    changes is that it is now a visible, deliberate choice a reader can question, and it survives
    the strict trust gate, which refuses an unstated default rather than silently supplying one.
    It is still NOT certified for any corpus, so results stay `trust_state=degraded` and
    `calibrated=False`.
    """
    from recall.calibration import Calibration
    from recall.guards import DEFAULT_GAP_THRESHOLD

    return Calibration(embedder=getattr(embedder, "name", "benchmark"),
                       threshold=DEFAULT_GAP_THRESHOLD)


class RecallBackend:
    """RE-call behind `Mem0Client`'s interface. Drop-in for their LOCOMO and BEAM runners.

    Args:
        dsn: PostgreSQL DSN for the pgvector store.
        embedder_name: see `resolve_embedder`. Local and free by default.
        table: benchmark-only table (default `BENCH_TABLE`).
        workspace_root: where per-user document workspaces are written. A temp dir by default;
            pass one to keep the exact corpus a run indexed.
        keep_workspace: leave the workspaces on disk after `close` (for auditing a run).
        candidate_k: per-leg candidate pool. ``None`` (default) means "match the requested
            ``top_k``" — see decision 4.
        **_ignored: their runners construct `Mem0Client` with `rpm`, `host`, `api_key`,
            `max_retries` etc. Swallowed so the swap needs no change at the call site; RE-call is
            a local library with no server and no rate limit, and pretending otherwise by raising
            would just force a call-site edit that has nothing to teach the reader.
    """

    #: Their runners read `mode` only for logging. Named for what it is.
    mode = "recall"

    def __init__(
        self,
        dsn: str,
        embedder_name: str = "fastembed",
        table: str = BENCH_TABLE,
        workspace_root: str | Path | None = None,
        keep_workspace: bool = False,
        candidate_k: int | None = None,
        **_ignored: Any,
    ) -> None:
        self._dsn = dsn
        self._embedder_name = embedder_name
        self._embedder = resolve_embedder(embedder_name)
        self._table = table
        self._keep_workspace = keep_workspace
        self._candidate_k = candidate_k
        self._own_root = workspace_root is None
        self._root = (
            Path(tempfile.mkdtemp(prefix="their-bench-"))
            if workspace_root is None
            else Path(workspace_root)
        )
        self._root.mkdir(parents=True, exist_ok=True)
        self._scopes: dict[str, _UserScope] = {}
        # ONE lock over every RE-call operation. RE-call is synchronous and CPU-bound (local
        # embedding, Postgres), so it is run in a worker thread to keep their event loop free for
        # the LLM calls that genuinely parallelise. Serialising it there is deliberate: their
        # runner drives up to 10 conversations concurrently, and a shared embedder plus one
        # long-lived connection per store are not safe to touch from two threads at once. The
        # measured axis is retrieval quality, not how many cores the local embedder can saturate.
        self._lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------ reporting

    def describe(self) -> dict[str, Any]:
        """This arm's configuration, for the results artifact. Carries no secret (the DSN may
        embed a password and is deliberately not reported)."""
        return {
            "system": "recall",
            "embedder": {"name": self._embedder_name, "model": self._embedder.name,
                         "dim": self._embedder.dim},
            "table": self._table,
            "candidate_k": self._candidate_k or "matches top_k",
            "reranker": None,
            "abstention": "honoured (empty result list)",
            "recall_version": _recall_version(),
            "users": sorted(self._scopes),
        }

    # ------------------------------------------------------------------ their interface

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Index one of their ingestion chunks. Returns their ``{"results": [...]}`` shape.

        ``custom_instructions`` and ``metadata`` are accepted and ignored: they steer Mem0's
        LLM extraction step, and RE-call has no LLM in its path to steer. Ignoring them is the
        honest behaviour — there is nothing to translate them into.
        """
        self._require_open()
        epoch = _resolve_epoch(timestamp, observation_date)
        body = _messages_to_document(messages)
        if not body:
            # Their runner already skips empty chunks; a chunk that is whitespace-only after
            # joining would otherwise create an empty document and an empty embedding row.
            return {"results": []}

        def work() -> dict:
            scope = self._scope(user_id)
            scope.seq += 1
            stamp = "NONE" if epoch is None else str(epoch)
            path = scope.workspace / f"m{scope.seq:06d}__ts{stamp}.md"
            path.write_text(body, encoding="utf-8")
            from recall.index import Indexer

            # `files=` rather than a re-glob of the whole workspace: their LOCOMO run issues one
            # `add` per turn (BEAM one per two turns), so re-walking would make ingestion
            # quadratic in turns — thousands of file reads per call on BEAM's 1M-token
            # conversations. Passing the single new file also means `_prune_vanished` finds every
            # older document still present on disk and deletes nothing.
            Indexer(scope.store, self._embedder).index_path(scope.workspace, files=[path])
            return {"results": [{"id": path.stem, "memory": body, "event": "ADD"}]}

        async with self._lock:
            return await asyncio.to_thread(work)

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Retrieve up to `top_k` memories for `query`, in RE-call's own ranked order.

        Returns ``[]`` when RE-call abstains (see the module docstring) and when the user has
        nothing indexed. ``rerank`` is refused rather than ignored: RE-call *has* a reranker seam,
        so silently dropping the flag would report a reranked run that never happened.
        """
        self._require_open()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if rerank:
            raise ValueError(
                "rerank=True is not supported by this adapter: RE-call's reranker is a paid "
                "cross-encoder, and this arm is $0 by construction. Silently ignoring the flag "
                "would label the run 'reranked' when it was not."
            )
        if user_id not in self._scopes:
            return []

        pool = self._candidate_k or top_k

        def work() -> list[dict]:
            from recall.retriever import DEFAULT_CANDIDATE_K
            # Benchmark backend: measures retrieval on uncertified corpora, so it opts
            # into development mode explicitly. See recall/eval/_research_trust.py.
            from recall.eval._research_trust import research_search

            scope = self._scopes[user_id]
            result = research_search(
                scope.store,
                self._embedder,
                query,
                k=top_k,
                candidate_k=max(pool, DEFAULT_CANDIDATE_K),
                calibration=_bench_calibration(self._embedder),
            )
            if result.abstained:
                return []
            n = len(result.hits)
            out: list[dict] = []
            for i, hit in enumerate(result.hits):
                entry: dict[str, Any] = {
                    "memory": hit.chunk.text,
                    # Strictly decreasing, so their `sort(key=score, reverse=True)` is a no-op on
                    # RE-call's order — see decision 3.
                    "score": (n - i) / n,
                    "id": hit.chunk.id,
                    "score_debug": {
                        "recall_rank": i,
                        "cosine": hit.cosine,
                        "confidence": hit.confidence,
                        "verdict": hit.verdict,
                    },
                }
                created = _created_at_of(hit)
                if created:
                    entry["created_at"] = created
                out.append(entry)
            return out

        async with self._lock:
            return await asyncio.to_thread(work)

    async def delete_user(self, user_id: str) -> bool:
        """Drop everything indexed for `user_id`. True on success, False if unknown."""

        def work() -> bool:
            # Look up first, remove-and-close only after the work succeeds. On a mid-delete
            # failure the scope stays in `self._scopes` with its store still OPEN, so a retry
            # re-runs `iter_chunks`/`delete_sources` against a live connection. Closing in a
            # `finally` (the previous shape) stranded a CLOSED store in the map, so every retry
            # failed instantly against the closed connection — the opposite of retryable. The
            # store is closed on the success path here and, for a scope never deleted, by
            # `close()` on the whole registry.
            scope = self._scopes.get(user_id)
            if scope is None:
                return False
            sources = sorted({c.source for c in scope.store.iter_chunks()})
            if sources:
                scope.store.delete_sources(sources)
            del self._scopes[user_id]
            scope.store.close()
            if not self._keep_workspace:
                shutil.rmtree(scope.workspace, ignore_errors=True)
            return True

        # Membership is decided INSIDE the lock: checked outside it, two concurrent deletes
        # of one user both passed the check and the second pop raised KeyError.
        async with self._lock:
            return await asyncio.to_thread(work)

    async def close(self) -> None:
        """Close every store. Workspaces are removed unless `keep_workspace` was set."""
        if self._closed:
            return
        self._closed = True

        def work() -> None:
            for scope in self._scopes.values():
                try:
                    scope.store.close()
                except Exception:  # pragma: no cover - a store that never opened
                    pass
            self._scopes.clear()
            if self._own_root and not self._keep_workspace:
                shutil.rmtree(self._root, ignore_errors=True)

        async with self._lock:
            await asyncio.to_thread(work)

    async def __aenter__(self) -> RecallBackend:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RecallBackend is closed")

    def _scope(self, user_id: str) -> _UserScope:
        """The tenant/store/workspace for `user_id`, created on first use."""
        scope = self._scopes.get(user_id)
        if scope is not None:
            return scope
        from recall.store import PgVectorStore

        tenant = f"{TENANT_PREFIX}{user_id}"
        # A directory named from a hash of the user id, not from the id itself: their BEAM ids
        # embed the conversation id verbatim and there is no guarantee it is a legal path segment
        # on every platform this runs on.
        workspace = self._root / hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        workspace.mkdir(parents=True, exist_ok=True)
        store = PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=tenant, table=self._table
        )
        store.__enter__()
        store.ensure_schema()
        # A fresh run must not inherit a previous one's rows. Their harness stamps `user_id` with a
        # per-run id, so a collision is unlikely — but "unlikely" is how a corpus silently doubles,
        # and a doubled corpus fills top-k with duplicates and makes the published number a
        # function of how many times the harness happened to be run.
        stale = sorted({c.source for c in store.iter_chunks()})
        if stale:
            store.delete_sources(stale)
        scope = _UserScope(tenant, workspace, store)
        self._scopes[user_id] = scope
        return scope


def _messages_to_document(messages: list[dict[str, str]]) -> str:
    """Their ingestion chunk as one markdown document body — content verbatim, in order.

    Only ``content`` is written. Their LOCOMO parser has already prefixed the speaker into the
    content (``"Caroline: ..."``), and the ``role`` it derives alongside is a Mem0-API artefact
    with no counterpart in a RE-call document. Writing a synthetic ``user:``/``assistant:`` line
    would hand RE-call a token Mem0's extractor never sees.
    """
    parts = [str(m.get("content", "")).strip() for m in messages]
    return "\n\n".join(p for p in parts if p)


def _resolve_epoch(timestamp: int | None, observation_date: str | None) -> int | None:
    """Their two ways of dating a chunk, reduced to one epoch. Mirrors `Mem0Client._add_oss`."""
    if timestamp is not None:
        return int(timestamp)
    if observation_date:
        try:
            d = datetime.strptime(observation_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return int(d.timestamp())
    return None


def _created_at_of(hit: Any) -> str | None:
    """Decode the ingest timestamp this chunk was written with, or None when there was none.

    Falls back to `provenance.indexed_at` ONLY when the filename carries no stamp, and never
    invents one: a fabricated date is worse than a missing one here, because their answerer prompt
    prints whatever date it is handed and reasons about it.
    """
    file_key = hit.chunk.metadata.get("file") or hit.chunk.source or ""
    m = _TS_RE.search(str(file_key).replace("\\", "/"))
    if m:
        raw = m.group("ts")
        return None if raw == "NONE" else _iso_from_epoch(int(raw))
    indexed_at = getattr(hit.provenance, "indexed_at", None)
    if isinstance(indexed_at, datetime):
        return indexed_at.astimezone(timezone.utc).strftime(_ISO_FMT)
    return None


def _recall_version() -> str | None:
    """The installed `recall-rag` version, for the results artifact."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("recall-rag")
    except PackageNotFoundError:  # pragma: no cover - source checkout without an install
        return None
