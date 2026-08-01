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

**The window is now supplied, and it was not before.** The corpus states validity in prose, but the
runner ALSO offers `load_metadata` with the intervals as structured data, precisely because "this
axis measures SELECTION, not extraction from prose (spec section 8)". This adapter did not
implement that hook, so RE-call's validity layer received nothing: measured on a v4 run, 0 of 420
stored chunks carried `valid_from` or `valid_until`. `now` was threaded into `trusted_search` with
nothing to compare against, so `expired` and `not_yet_valid` could not fire and the covering rate
was scoring plain semantic retrieval. `load_metadata` below writes `effective_from`/`effective_to`
into the frontmatter RE-call reads.

⚠️ **`known_as_of` is still NOT wired here, deliberately, and the reason has narrowed.** It is
TRANSACTION time; valid time is what tiers T1 to T5 score, and that is what `load_metadata` now
supplies. But manifest v4 added **T6-write-time**, which places the reference time strictly inside
the gap between `asserted_at` and `effective_from` -- a tier that cannot be answered from valid
time alone. So the old claim that this axis "scores valid time" and therefore cannot measure
`known_as_of` is no longer true of v4. Wiring it needs `asserted_at` mapped onto the store's write
column, which means backdating rows on purpose; that is its own pre-registered arm, and folding it
in here would make this a two-variable change.

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
import tempfile
import os
from collections.abc import Iterable

from membench.axes.temporal.adapter import TemporalDocument, TemporalResponse

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
        self._docs: dict[str, str] = {}
        self._work = tempfile.mkdtemp(prefix="membench-temporal-")

    def ingest(self, docs: Iterable[TemporalDocument]) -> None:
        self._docs = {d.doc_id: d.text for d in docs}
        self._index(self._docs)

    def load_metadata(self, meta: dict[str, dict]) -> None:
        """Take the intervals as structured data and put them where RE-call reads them.

        The runner offers this hook and its own comment says why: "the intervals as structured
        data, for systems that accept them ... this axis measures SELECTION, not extraction from
        prose (spec section 8)". This adapter never implemented it, so RE-call's validity layer
        received NOTHING. Measured on a v4 run: 0 of 420 stored chunks carried `valid_from`,
        `valid_until`, or any clock at all. `now=reference_time` was handed to `trusted_search`
        with nothing to compare against, so `expired` and `not_yet_valid` could not fire and the
        covering rate was scoring plain semantic retrieval.

        That is the failure this repo already keeps a script for: a capability present in the code
        path and structurally unable to run, reporting a clean number for something that never
        executed. See `benchmarks/check_temporal_inert.py`.

        Re-indexes rather than patching rows in place. Adding frontmatter changes the document, so
        the content hash changes and the Indexer re-reads it through the ordinary path; nothing
        here writes to the store behind the library's back.

        VALID time only. `asserted_at` is deliberately left alone: it is transaction time, it
        needs `known_as_of` plus a backdated write column, and folding it in here would make a
        two-variable change out of a one-variable one.
        """
        self._index(self._docs, meta)

    def _index(self, docs: dict[str, str], meta: dict[str, dict] | None = None) -> None:
        from pathlib import Path

        # ONE work directory for the life of the adapter, not a fresh mkdtemp per pass. The
        # Indexer keys `replace_sources` on the absolute source path, so a second pass under a new
        # tempdir INSERTS rather than replaces: measured, 840 rows for 420 documents, every one of
        # them present twice and half of them carrying no validity metadata at all. A retrieval
        # then sees two copies of every fact and the metadata-less one can win.
        work = Path(self._work)
        for doc_id, text in docs.items():
            body = text
            m = (meta or {}).get(doc_id) or {}
            # `effective_to` is None on half of v4's assertions (an open-ended interval), and an
            # empty `valid_until` would be a parse failure rather than "no upper bound", so the
            # key is omitted instead of emitted blank.
            lines = []
            if m.get("effective_from"):
                lines.append(f"valid_from: {m['effective_from']}")
            if m.get("effective_to"):
                # HALF-OPEN to INCLUSIVE. mem-bench's covering predicate is
                # `reference_time < effective_to` (corpus.py), while RE-call reads `valid_until`
                # as inclusive to end-of-day. Passing `effective_to` straight through therefore
                # kept an assertion alive for one extra day, and every T5-boundary instance —
                # whose reference time sits exactly on the edge — scored 0.0000. Converting the
                # exclusive end to the previous day is the translation between two interval
                # conventions, which is what an adapter is for; leaving it would have reported an
                # adapter artefact as a RE-call weakness.
                end = dt.date.fromisoformat(m["effective_to"]) - dt.timedelta(days=1)
                lines.append(f"valid_until: {end.isoformat()}")
            if lines:
                body = "---\n" + "\n".join(lines) + "\n---\n" + text
            (work / f"{doc_id}.md").write_text(body, encoding="utf-8")
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
