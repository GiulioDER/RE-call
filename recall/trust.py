"""The trust layer: confidence + provenance + validity over retrieval, with calibrated abstention.

Semantic similarity answers "which memory looks most like the query"; it cannot answer "should
this memory still be believed". This module adds that second judgment as a pure post-processing
step over `HybridRetriever.search()`:

- every hit gets a verdict — ``ok``, ``superseded``, ``expired``, ``not_yet_valid`` or
  ``low_confidence`` — plus a calibrated confidence, provenance, and its validity window;
- a memory that is superseded or outside its validity window loses even with a top cosine:
  valid hits are ordered first (so a retrieved successor outranks the stale memory it replaced),
  and when NO valid hit remains the result is an explicit abstention with a reason;
- the abstention threshold comes from `recall.calibration` when available; otherwise the default
  gap threshold is used and the result is flagged ``calibrated=False``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from recall.observability import METRICS
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle: entailment imports trust's abstain wording
    from recall.entailment import EntailmentJudge

from recall.calibration import Calibration
from recall.observability import get_logger
from recall.embeddings import Embedder
from recall.frontmatter import validity_bounds
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.rerank import Reranker
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever
from recall.store import PgVectorStore
from recall.types import (
    Provenance,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,
    Validity,
    Verdict,
)

_log = get_logger("trust")

_UNCALIBRATED = Calibration(embedder="uncalibrated", threshold=DEFAULT_GAP_THRESHOLD)

#: Embedder names already warned about, so a long-running process says this once per model rather
#: than once per query. Not thread-guarded: a duplicate warning under a race is harmless, a lock in
#: the retrieval hot path is not.
_WARNED_UNCALIBRATED: set[str] = set()

#: Store types already warned about for a point-in-time query they can only half-answer. Same
#: once-per-process, unguarded rationale as `_WARNED_UNCALIBRATED`.
_WARNED_NO_EDGE_DATES: set[str] = set()


def _as_utc(value: datetime) -> datetime:
    """A naive datetime read as UTC. One helper, because the three comparison sites in this
    module (`now`, `known_as_of`, and each `edge_dates` value) all put a caller-supplied instant
    against a TIMESTAMPTZ from the store, and normalising some of them raises where normalising
    none of them did not."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _warn_no_edge_dates(store_type: str) -> None:
    """A `known_as_of` query on a store that cannot date its supersession edges.

    The caller asked what was true at a past instant and gets a HALF answer: hits are rewound by
    write time, but supersession is not, so a memory can read as `superseded` by a document that
    did not exist at that instant. Degrading silently would be worse than the AttributeError it
    replaced, because a wrong answer arrives looking like a right one.
    """
    if store_type in _WARNED_NO_EDGE_DATES:
        return
    _WARNED_NO_EDGE_DATES.add(store_type)
    _log.warning(
        "%s exposes no supersession_all(), so known_as_of rewinds hits but NOT supersession "
        "edges: a hit may read as superseded by a document that did not exist at that instant.",
        store_type,
    )


def _warn_uncalibrated(embedder_name: str) -> None:
    """Say plainly that an untuned constant is about to gate abstention for this model.

    `DEFAULT_GAP_THRESHOLD` is an ABSOLUTE cosine, and cosine levels are a property of how a model
    was trained, not a comparable quantity across models. Measured over two corpora and three
    embedders (2 x 3 conditions, n=100 and n=775): the default sits at the **0th percentile** of
    five of those six top-1 distributions and at the **16th** of the sixth. It is therefore inert
    for most models and, for one, discards roughly a sixth of all queries into empty retrieval —
    which the answerer turns into a refusal and the caller reads as a wrong answer, with nothing
    anywhere reporting a misconfiguration.

    Worse, the failure is invisible on the DEFAULT embedder: bge-small's observed minimum is 0.63,
    well above the 0.50 floor, so the shipped combination never trips it. Only a user who changes
    embedder is exposed, and they get no signal at all.

    Four replacements were measured and none survived (per-query percentile, score gap, corpus
    quantile on real queries, corpus quantile derived from self-queries at index time). Every one
    is a monotone function of the same score, and the score does not always order the classes
    correctly. So this does not silently pick a different constant — it says which model is
    unmeasured and points at `recall calibrate`, and lets the caller decide.
    """
    if embedder_name in _WARNED_UNCALIBRATED:
        return
    _WARNED_UNCALIBRATED.add(embedder_name)
    _log.warning(
        "no calibration found for embedder %r — abstention will use the UNTUNED default cosine "
        "floor %.2f. That constant is not comparable across embedders (measured: 0th percentile "
        "of five top-1 distributions, 16th of a sixth), so on some models it never fires and on "
        "others it discards a sixth of queries as empty retrieval. Run `recall calibrate` for "
        "this embedder, or pass calibration= explicitly, to replace the guess with a measurement.",
        embedder_name,
        DEFAULT_GAP_THRESHOLD,
    )


#: ANSI escape sequences a terminal ACTS on: CSI (`\x1b[…`), OSC (`\x1b]…` up to BEL or ST), and
#: the two-character forms. Matched as whole sequences so nothing is left behind to re-arm.
_ANSI_SEQUENCE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"        # CSI — cursor moves, erases, colours
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC — window title, hyperlinks
    r"|\x1b[@-Z\\-_]"                 # two-character escapes
)


def resolve_successor(
    file: str,
    supersession: dict[str, str],
    edge_dates: dict[str, datetime] | None = None,
    known_as_of: datetime | None = None,
) -> str | None:
    """Terminal successor of `file` in the supersession chain, or None if it has none.

    A cycle (a.md -> b.md -> a.md) cannot loop: the walk stops on the first revisit and the
    cycle member resolves to its direct successor. A self-claim (`supersedes:` the file's own
    name — an authoring mistake) is ignored: a document cannot supersede itself.

    With `known_as_of`, the walk sees only the edges asserted by that instant, so a point-in-time
    replay resolves to the successor that was current *then*. The filter is applied per STEP
    rather than to the final answer: in a chain a -> b -> c where only the first edge predates the
    instant, the answer is `b`. Gating the terminal successor instead would answer `c`, a document
    that had not yet superseded anything.

    An edge with no recorded date applies, which is the inverse of the rule for hits with no
    `indexed_at` and deliberately so. Both are fail-closed: hiding a hit of unknown age would
    silently empty result sets for stores predating the column, while ignoring an edge of unknown
    age would serve a memory the corpus explicitly marks as stale.
    """

    def step(cur: str) -> str | None:
        """The next file in the chain, or None if there is no edge or it is not yet asserted."""
        nxt = supersession.get(cur)
        if nxt is None:
            return None
        if known_as_of is not None and edge_dates is not None:
            asserted = edge_dates.get(cur)
            if asserted is not None and asserted > known_as_of:
                return None
        return nxt

    first = step(file)
    if first in (None, file):
        return None
    seen = {file}
    cur = file
    while True:
        nxt = step(cur)
        if nxt is None:
            return cur
        if nxt in seen:
            return first
        seen.add(nxt)
        cur = nxt


def _verdict(
    hit: ScoredChunk,
    supersession: dict[str, str],
    threshold: float,
    now: datetime,
    unresolved: frozenset[str] = frozenset(),
    known_as_of: datetime | None = None,
    edge_dates: dict[str, datetime] | None = None,
) -> tuple[Verdict, Validity]:
    meta = hit.chunk.metadata
    file = meta.get("file")
    try:
        start, end = validity_bounds(meta)
    except ValueError:
        # Malformed validity metadata (reachable via direct store.upsert, which bypasses the
        # Indexer's fail-fast). Fail CLOSED per hit: an unparseable window must not read as
        # trustworthy, and one bad row must not crash every search that retrieves it.
        return "invalid_metadata", Validity(valid_from=None, valid_until=None, superseded_by=None)
    if known_as_of is not None and hit.indexed_at is not None and hit.indexed_at > known_as_of:
        # TRANSACTION time, checked BEFORE supersession on purpose: a memory written after the
        # as-of instant did not exist yet, and whether it was later superseded is not a question
        # that can be asked about something that had not been written. Ordering it after
        # `superseded` would report the fate of a memory the caller cannot see.
        #
        # A row with NO `indexed_at` is left alone rather than hidden. It reaches here only from a
        # store that did not record one, and defaulting an unknown write time to "after the as-of"
        # would silently empty a result set for callers whose data predates the column.
        return (
            "not_yet_known",
            Validity(valid_from=start, valid_until=end, superseded_by=None),
        )
    if file is not None and file in unresolved:
        # A supersession edge points at this memory's basename, but several documents carry it.
        # Serving it as ``ok`` because the edge could not be resolved would be the silent
        # wrong-answer this whole layer exists to prevent.
        return (
            "ambiguous_supersession",
            Validity(valid_from=start, valid_until=end, superseded_by=None),
        )
    successor = (
        resolve_successor(file, supersession, edge_dates, known_as_of) if file else None
    )
    validity = Validity(valid_from=start, valid_until=end, superseded_by=successor)
    if successor is not None:
        return "superseded", validity
    if end is not None and now > end:
        return "expired", validity
    if start is not None and now < start:
        return "not_yet_valid", validity
    if hit.score < threshold:
        return "low_confidence", validity
    return "ok", validity


#: Longest corpus-controlled identifier rendered into agent-facing prose. Real memo paths sit far
#: below this; the bound exists so a hostile 10 KB file name cannot bury the sentence it appears
#: in under its own payload.
MAX_REF_CHARS = 120


def terminal_safe(value: str | None) -> str:
    """Strip anything a terminal would interpret rather than display.

    The sibling of `safe_ref` for the OTHER interpreter this library writes to. `recall.cli`
    prints corpus-controlled strings — file names, successor names, chunk previews — and a
    terminal executes ANSI escapes: `\\x1b[2K\\r` erases the line just written, `\\x1b[1A` moves
    up over it. A corpus can therefore make `recall lint` render a clean report while scrolling
    away the issues it just found.

    A filter, not a renderer: unlike `safe_ref` it adds no quotes and no length bound, because
    the CLI's own format strings already do the layout and an operator reading their own corpus
    should see the name they wrote.

    Whole SEQUENCES go, not just the `\\x1b` that introduces them. Dropping the escape byte alone
    is the tempting one-liner and it is wrong twice over: it leaves `[2K` behind as literal
    garbage in the operator's output, and it leaves a payload that becomes live again the moment
    anything upstream reinserts an escape byte. Removing the sequence removes both.
    """
    if value is None:
        return ""
    cleaned = _ANSI_SEQUENCE.sub("", str(value))
    return "".join(
        ch for ch in cleaned
        if not unicodedata.category(ch).startswith(("Cc", "Cf", "Zl", "Zp"))
    )


def safe_ref(value: str | None) -> str:
    """Render a corpus-controlled identifier for inclusion in agent-facing prose.

    `provenance.file` and `validity.superseded_by` are `metadata['file']` — a path chosen by
    whoever can write a file into the corpus — and they are interpolated into `reason`, which
    `recall_mcp.service.search_memory` folds into `advice`, the field `recall_search` tells the
    model to obey. That is untrusted input reaching an instruction channel, so it is treated as
    data being quoted rather than as prose being continued.

    No attempt is made to recognise malicious wording; that is unwinnable and would fail exactly
    when it mattered. What is removed instead are the three properties that let corpus text
    impersonate this library's own voice:

    - **Control characters and line breaks.** A newline is what turns an identifier into what
      looks like a fresh instruction on its own line. Dropped by Unicode category, which also
      takes the bidirectional overrides (`Cf`) that can visually reorder a line into something
      other than what it says.
    - **Length.** Bounded, so a payload cannot push the real message out of view.
    - **Undelimited placement.** Quoted, so it reads as a quoted name — and the delimiter is
      stripped from the body, because a quote that the value can close is not a delimiter.

    `None` renders as ``unknown`` rather than the string ``"None"``: a hit whose provenance is
    missing should say so, not name a file that does not exist.
    """
    if value is None:
        return "unknown"
    cleaned = "".join(
        " " if ch.isspace() else ch
        for ch in str(value)
        if not unicodedata.category(ch).startswith(("Cc", "Cf", "Zl", "Zp"))
    )
    # The delimiter is REMOVED from the body, not escaped: an escape (`\"`) still puts the
    # closing character in the string, and the reader that matters here is a language model
    # rather than a parser that honours backslashes.
    cleaned = " ".join(cleaned.replace('"', "'").split())
    if not cleaned:
        return "unknown"
    if len(cleaned) > MAX_REF_CHARS:
        cleaned = cleaned[:MAX_REF_CHARS] + "…"
    return f'"{cleaned}"'


def is_trusted(hit: TrustedHit) -> bool:
    """Whether a hit may be relied on — the single definition of ``ok`` the adapters share."""
    return hit.verdict == "ok"


def marked_text(hit: TrustedHit) -> str:
    """The hit's text, carrying an IN-BAND warning when its verdict is not ``ok``.

    Exists because out-of-band metadata is exactly what failed at the framework boundary. Both
    adapters attach `recall_verdict` to a Document/Node's `metadata`, but LangChain's stock
    `stuff_documents_chain` and LlamaIndex's default node handling render `page_content` / `text`
    alone into the prompt — so the memory arrived and the warning did not. A caller who
    deliberately opts into untrusted hits gets the warning where the model will actually see it.

    The successor file is corpus-controlled, so it goes through `safe_ref` for the same reason
    `abstain_reason` does.
    """
    if is_trusted(hit):
        return hit.chunk.text
    detail = ""
    if hit.verdict == "superseded" and hit.validity.superseded_by:
        detail = f" by {safe_ref(hit.validity.superseded_by)}"
    return (
        f"[RE-CALL WARNING — this memory is {hit.verdict}{detail}. It did NOT pass the trust "
        f"layer; treat it as untrusted context and do not rely on it.]\n{hit.chunk.text}"
    )


def servable_hits(hits: list[TrustedHit], *, include_untrusted: bool = False) -> list[TrustedHit]:
    """The hits an adapter may hand to a chain.

    `TrustedResult.hits` is `ok + rest`, so a result with ONE valid hit does not abstain and
    carries every superseded/expired/invalid hit along with it. Serving that list wholesale made
    the adapters inconsistent with themselves: the same superseded memo was withheld when nothing
    else matched (abstention -> empty) and forwarded when something unrelated did. Nobody chose
    that rule; it fell out of reusing the list.
    """
    return list(hits) if include_untrusted else [h for h in hits if is_trusted(h)]


def abstain_reason(hits: list[TrustedHit]) -> str:
    if not hits:
        return "no memory retrieved at all"
    best = max(hits, key=lambda h: h.cosine)
    file = safe_ref(best.provenance.file)
    if best.verdict == "superseded":
        return (
            f"best candidate ({file}) is superseded by "
            f"{safe_ref(best.validity.superseded_by)} — consult the successor, not this memory"
        )
    if best.verdict == "expired":
        return f"best candidate ({file}) is outside its validity window (expired)"
    if best.verdict == "not_yet_valid":
        return f"best candidate ({file}) is not yet valid"
    if best.verdict == "not_yet_known":
        # Distinct wording from `not_yet_valid` on purpose. That one means the fact was not true
        # yet; this one means the memory had not been WRITTEN yet. A caller replaying a past
        # decision needs to tell "we had no such memory then" apart from "we had it and it did not
        # apply", because only the first exonerates the decision.
        return (
            f"best candidate ({file}) was written after the as-of instant — it did not exist "
            f"yet, so it cannot explain a decision made then"
        )
    if best.verdict == "invalid_metadata":
        return f"best candidate ({file}) carries malformed validity metadata"
    if best.verdict == "ambiguous_supersession":
        return (
            f"a supersession edge points at the best candidate ({file}) by a "
            f"basename that more than one document carries, so the edge cannot be resolved — "
            f"disambiguate the corpus rather than trusting either copy"
        )
    return "no hit above the calibrated confidence threshold (probable corpus gap)"


def evaluate(
    result: RetrievalResult,
    supersession: dict[str, str],
    calibration: Calibration | None,
    now: datetime,
    unresolved: frozenset[str] = frozenset(),
    known_as_of: datetime | None = None,
    edge_dates: dict[str, datetime] | None = None,
) -> TrustedResult:
    """Pure trust evaluation of a retrieval result (no DB access, no clock reads).

    Two independent time axes, which is what makes this bi-temporal:

    - ``now`` is **valid time**: when a fact was true. It drives ``expired`` and ``not_yet_valid``
      from the memory's own declared `valid_from` / `valid_until`.
    - ``known_as_of`` is **transaction time**: when the memory was written. It drives
      ``not_yet_known`` from the store's `indexed_at`, and answers a different question, *what did
      we know at that moment*, rather than *what was true*.

    They compose: ``evaluate(..., now=june, known_as_of=tuesday)`` asks what we believed on Tuesday
    about the state of the world in June. Passing neither leaves behaviour exactly as before.

    **``edge_dates`` rewinds supersession too**, when supplied. `trusted_search` gets it from
    `PgVectorStore.supersession_all()` whenever `known_as_of` is set, the store exposes that
    method and the retrieval returned hits; a store without it logs a warning and rewinds hits
    only. An edge becomes
    assertable when the superseding document is written, so its `indexed_at` dates the edge; a
    chain resolves per step, to the successor current at the instant. Point-in-time replay is
    then honest about which memories existed *and* about which were current.

    🔴 **Not merge-ready.** `indexed_at` is the LAST write, not the first, so re-indexing a
    superseding document moves its edge forward and a past replay can serve the superseded memory
    as ``ok`` where the pre-change code correctly said ``superseded``. See
    `PgVectorStore.supersession_all` for the mechanism and the fix it needs. Callers passing
    ``known_as_of`` on a corpus that is ever re-indexed should not rely on the rewind yet.

    Narrower residues, both fail-closed: an edge whose superseding file has no recorded date
    applies unconditionally, and ``unresolved`` is not rewound, so an ambiguous claim written
    after the instant still forces an abstention at it.

    Omitting ``edge_dates`` keeps the old behaviour exactly, so no existing caller changes.

    Verdict precedence per hit: invalid_metadata > not_yet_known > superseded > expired /
    not_yet_valid > low_confidence > ok.
    Successor promotion: when a superseded hit scored above the threshold (it would have been
    the confident answer), its retrieved successor is promoted from ``low_confidence`` to
    ``ok`` even if its own wording scores lower — the explicit supersession edge transfers the
    topical relevance the stale memory proved. Hits are then reordered valid-first so the
    successor outranks the stale memory. `abstained` is True when no hit earned verdict ``ok``.
    A tz-naive `now` is interpreted as UTC.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if known_as_of is not None:
        # BOTH operands, and NOT only when `known_as_of` happened to be naive. Nesting the
        # `edge_dates` half inside that branch left the mirror combination raising: an AWARE
        # `known_as_of` against a naive hand-built date map still hit the comparison, so the more
        # careful caller was the one that crashed. Two rounds of review to get one comment to
        # match its own code.
        known_as_of = _as_utc(known_as_of)
        if edge_dates:
            edge_dates = {file: _as_utc(when) for file, when in edge_dates.items()}
    cal = calibration or _UNCALIBRATED
    trusted: list[TrustedHit] = []
    for hit in result.hits:
        verdict, validity = _verdict(
            hit, supersession, cal.threshold, now, unresolved, known_as_of, edge_dates
        )
        meta = hit.chunk.metadata
        trusted.append(
            TrustedHit(
                chunk=hit.chunk,
                cosine=hit.score,
                confidence=cal.confidence(hit.score),
                verdict=verdict,
                provenance=Provenance(
                    source=hit.chunk.source,
                    file=meta.get("file"),
                    ord=meta.get("ord"),
                    indexed_at=hit.indexed_at,
                ),
                validity=validity,
            )
        )
    promoted_files = {
        h.validity.superseded_by
        for h in trusted
        if h.verdict == "superseded" and h.cosine >= cal.threshold
    }
    trusted = [
        replace(h, verdict="ok")
        if h.verdict == "low_confidence" and h.provenance.file in promoted_files
        else h
        for h in trusted
    ]
    ok = [h for h in trusted if h.verdict == "ok"]
    rest = [h for h in trusted if h.verdict != "ok"]
    abstained = not ok
    # The operational questions this library exists to answer — how often does it abstain, and
    # what is it demoting — are answerable only if they are counted where the decision is made.
    METRICS.increment("recall_searches_total")
    if abstained:
        METRICS.increment("recall_abstentions_total")
    if result.gap_warning:
        METRICS.increment("recall_gap_warnings_total")
    if result.staleness.stale:
        METRICS.increment("recall_stale_results_total")
    for trusted_hit in trusted:  # not `hit`: that name is bound to a ScoredChunk above
        METRICS.increment("recall_verdicts_total", verdict=trusted_hit.verdict)
    return TrustedResult(
        query=result.query,
        hits=ok + rest,
        abstained=abstained,
        reason=abstain_reason(rest) if abstained else "",
        calibrated=calibration is not None,
        gap_warning=result.gap_warning,
        staleness=result.staleness,
    )


def trusted_search(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    k: int = 5,
    source: str | None = None,
    calibration: Calibration | None = None,
    reranker: Reranker | None = None,
    now: datetime | None = None,
    known_as_of: datetime | None = None,
    entailment: "EntailmentJudge | None" = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> TrustedResult:
    """Hybrid search + trust evaluation in one call — the recommended agent-facing entry point.

    `entailment` is OFF by default: when a judge is passed, verdict-ok hits that do not entail
    an answer to the query are demoted to ``not_entailed`` (see `recall.entailment`) — the
    near-miss guard the cosine threshold cannot provide. Costs one judge pass per ok hit.

    `candidate_k` is the per-leg pool size handed to the retriever (default the library's own
    ``DEFAULT_CANDIDATE_K``). It is exposed so a caller that widened the pool for its other
    retrievals — e.g. an eval sweep — can hold this call to the SAME pool, rather than silently
    reverting to the default here.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    # single fallback resolution: the retriever's gap threshold and the verdict threshold must
    # always come from the same calibration (or the same uncalibrated default)
    #
    # A calibration on disk is LOADED here when the caller passes none. It used to be read only
    # by `recall.cli`, so a user who ran `recall calibrate`, saw calibration.json appear, and then
    # used the library API got the uncalibrated default anyway — silently, while this module's own
    # docstring promised the threshold "comes from recall.calibration when available". It was
    # available; nothing fetched it.
    #
    # The cost of that gap is measurable, not theoretical: DEFAULT_GAP_THRESHOLD is 0.50, tuned
    # for bge-small whose cosines run high. On text-embedding-3-small, whose top-1 cosines were
    # measured between 0.41 and 0.76, it silently dropped 18 of 300 BEAM questions to EMPTY
    # retrieval — a guaranteed zero, no warning, on a model the library advertises support for.
    #
    # Explicit `calibration=` still wins, and a corpus with no calibration file behaves exactly as
    # before, so this cannot change a result for anyone who had not already calibrated.
    if calibration is None:
        from recall.calibration import load_for

        # `name` is part of the Embedder protocol, but this reads it DEFENSIVELY: auto-loading is
        # a convenience, and a convenience must never turn a search that worked into an
        # AttributeError. A stub or a partial implementation simply does not get a calibration,
        # which is exactly the behaviour it had before this feature existed.
        name = getattr(embedder, "name", None)
        if isinstance(name, str):
            calibration = load_for(name)
            if calibration is None:
                _warn_uncalibrated(name)
    cal = calibration or _UNCALIBRATED
    retriever = HybridRetriever(
        store, embedder, reranker=reranker, gap_threshold=cal.threshold, candidate_k=candidate_k
    )
    result = retriever.search(query, k=k, source=source)
    # ONE call when the edge dates are needed. `supersession()` and `supersession_dates()` each
    # run their own cache validation, so calling them in sequence lets a concurrent index hand
    # back edges from one scan dated by the next. Read defensively: `store` is duck-typed in
    # several places (tests and downstream adapters implement only the read surface they need),
    # and a store without the method degrades to exactly the pre-change behaviour.
    supersession: dict[str, str] = {}
    unresolved: frozenset[str] = frozenset()
    edge_dates: dict[str, datetime] | None = None
    if result.hits:
        fetch_all = getattr(store, "supersession_all", None) if known_as_of is not None else None
        if fetch_all is not None:
            supersession, unresolved, edge_dates = fetch_all()
        else:
            if known_as_of is not None:
                _warn_no_edge_dates(type(store).__name__)
            supersession, unresolved = store.supersession()
    trusted = evaluate(
        result,
        supersession,
        calibration,
        now or datetime.now(timezone.utc),
        unresolved,
        known_as_of,
        edge_dates,
    )
    if entailment is not None:
        from recall.entailment import apply_entailment

        trusted = apply_entailment(trusted, entailment)
    return trusted
