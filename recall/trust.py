"""The trust layer: confidence + provenance + validity over retrieval, with calibrated abstention.

Semantic similarity answers "which memory looks most like the query"; it cannot answer "should
this memory still be believed". This module adds that second judgment as a pure post-processing
step over `HybridRetriever.search()`:

- every hit gets a verdict — ``ok``, ``superseded``, ``expired``, ``not_yet_valid`` or
  ``low_confidence`` — plus a calibrated confidence, provenance, and its validity window;
- a memory that is superseded or outside its validity window loses even with a top cosine:
  valid hits are ordered first (so a retrieved successor outranks the stale memory it replaced),
  and when NO valid hit remains the result is an explicit abstention with a reason;
- the abstention threshold comes from an exact v2 generation binding when available; otherwise
  the default gap threshold is used and the result is flagged ``calibrated=False``.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from recall.observability import METRICS
from recall.promotion import reviewed_promotion_is_trusted_metadata

if TYPE_CHECKING:  # avoid a runtime import cycle: entailment imports trust's abstain wording
    from recall.entailment import EntailmentJudge

from recall.calibration import Calibration
from recall.embeddings import Embedder, embedding_profile_id
from recall.frontmatter import validity_bounds
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.observability import get_logger
from recall.rerank import Reranker
from recall.retriever import (
    DEFAULT_CANDIDATE_K,
    DocumentExpansionPolicy,
    HybridRetriever,
    StructuralExpansionPolicy,
    expand_retrieval_by_source,
    expand_retrieval_by_structure,
)
from recall.store import EdgeCandidates, PgVectorStore
from recall.trust_policy import (
    TrustFailureCode,
    TrustPolicy,
    TrustRefusal,
    TrustState,
    code_for_status,
)
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
    module (`now`, `known_as_of`, and each candidate's date) all put a caller-supplied instant
    against a TIMESTAMPTZ from the store, and normalising some of them raises where normalising
    none of them did not."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


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
    unmeasured and points at generation-bound calibration, and lets the caller decide.
    """
    if embedder_name in _WARNED_UNCALIBRATED:
        return
    _WARNED_UNCALIBRATED.add(embedder_name)
    _log.warning(
        "no calibration found for embedder %r — abstention will use the UNTUNED default cosine "
        "floor %.2f. That constant is not comparable across embedders (measured: 0th percentile "
        "of five top-1 distributions, 16th of a sixth), so on some models it never fires and on "
        "others it discards a sixth of queries as empty retrieval. Run `recall calibration "
        "calibrate --generation G --queries FILE --publish` for the active tenant generation to "
        "replace the guess with an exact measurement.",
        embedder_name,
        DEFAULT_GAP_THRESHOLD,
    )


#: ANSI escape sequences a terminal ACTS on: CSI (`\x1b[…`), OSC (`\x1b]…` up to BEL or ST), and
#: the two-character forms. Matched as whole sequences so nothing is left behind to re-arm.
_ANSI_SEQUENCE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI — cursor moves, erases, colours
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC — window title, hyperlinks
    r"|\x1b[@-Z\\-_]"  # two-character escapes
)


def resolve_successor(
    file: str,
    supersession: dict[str, str],
    edge_candidates: EdgeCandidates | None = None,
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
    recorded write time at all and deliberately so. Both are fail-closed: hiding a hit of unknown age would
    silently empty result sets for stores predating the column, while ignoring an edge of unknown
    age would serve a memory the corpus explicitly marks as stale.
    """

    def step(cur: str) -> str | None:
        """The next file in the chain, as it stood at `known_as_of`.

        With no instant to replay, this is `supersession[cur]` and nothing else happens. With one,
        it is the LIVE claim: of every document claiming to supersede `cur`, those asserted at or
        before the instant, and among them the one asserted LAST.

        Choosing among candidates is the whole point. `supersession` keeps a single winner per
        target, picked by scan order, which is a time-independent rule and therefore answers a
        different question: where two documents supersede one target, the winner today is often
        not the one live at a past instant, and gating that single winner drops a real edge and
        reports the stale memory as current.

        ⚠️ **So asking about NOW two ways can give two answers, deliberately.** With fan-in where
        the later-asserted claim is not the last scan row, a plain search follows `supersession`
        and answers the scan-order winner, while `known_as_of=<any instant after both>` answers
        the latest-ASSERTED one. By this function's own argument the replay answer is the better
        one, but `supersession()` is not changed to match: it is what every existing caller
        already gets, and silently re-answering it would be the behaviour change this whole branch
        has been avoiding. Pinned by `test_replay_at_now_may_differ_from_a_plain_search`.
        """
        if known_as_of is None or edge_candidates is None:
            return supersession.get(cur)
        claims = edge_candidates.get(cur)
        if claims is None:
            # ABSENT from the map. Candidates and `supersession` come from one pass, so this means
            # hand-built, inconsistent input. Fail closed: keep demoting.
            return supersession.get(cur)
        if not claims:
            # PRESENT but empty is a different signal, and conflating the two silently ignored
            # `known_as_of` for that step. `_resolve_rows` never emits an empty list, so this only
            # reaches a caller who built one, and the only thing it can mean is "no claim".
            return None
        live = [(f, when) for f, when in claims if when is None or when <= known_as_of]
        if not live:
            return None
        best, best_when = live[0]
        for f, when in live[1:]:
            # An undated claim is of unknown age and loses to any dated one, so a known assertion
            # decides the answer where one exists. Ties go to the later scan position, which is
            # the same rule `supersession`'s single winner uses.
            if best_when is None or (when is not None and when >= best_when):
                best, best_when = f, when
        return best

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
    edge_candidates: EdgeCandidates | None = None,
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
    # Read INSIDE the `known_as_of` guard, not before it. The original condition short-circuited
    # on `known_as_of is not None`, so a hit carrying no write time at all never had the attribute
    # touched; hoisting the lookup out broke duck-typed hits that had never needed one. And
    # `getattr` on both, because adding a field to `ScoredChunk` must not turn a working search
    # into an AttributeError — the same rule the `embedder.name` lookup below follows.
    first_known = None
    if known_as_of is not None:
        # Explicit None test, not `or`: truthiness cannot tell "this store has no such column"
        # from "something between the store and here dropped the field", and the second is what
        # actually happened once — the retriever rebuilt hits from a hand-written field list and
        # silently lost it, so the fallback below became the ONLY branch production ever took.
        first_known = getattr(hit, "first_indexed_at", None)
        if first_known is None:
            first_known = getattr(hit, "indexed_at", None)
    if known_as_of is not None and first_known is not None and first_known > known_as_of:
        # TRANSACTION time, checked BEFORE supersession on purpose: a memory written after the
        # as-of instant did not exist yet, and whether it was later superseded is not a question
        # that can be asked about something that had not been written. Ordering it after
        # `superseded` would report the fate of a memory the caller cannot see.
        #
        # FIRST write, not last. `indexed_at` moves forward every time a document is re-indexed,
        # so using it here claimed a memo edited today did not exist last month, and every replay
        # before the edit reported an empty store. The fallback to `indexed_at` is for stores
        # predating the column, where it is the only evidence available and is exactly what the
        # migration backfills.
        #
        # A row with NEITHER is left alone rather than hidden: defaulting an unknown write time to
        # "after the as-of" would silently empty a result set for callers whose data predates both.
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
        resolve_successor(file, supersession, edge_candidates, known_as_of) if file else None
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
        ch for ch in cleaned if not unicodedata.category(ch).startswith(("Cc", "Cf", "Zl", "Zp"))
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


def metadata_is_trusted(value: object) -> bool:
    """Only reviewed promotions may be treated as trusted inferred metadata."""

    return reviewed_promotion_is_trusted_metadata(value)


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
    if hit.verdict == "unverified":
        # A different sentence on purpose. Every other verdict means the trust layer ran and
        # judged this memory; `unverified` means it never ran, so "did NOT pass the trust layer"
        # would be a false statement about a check that was skipped, not failed. That is
        # requirement 13's distinction, and it has to survive into the prompt, because the prompt
        # is the only place a model will ever see it.
        return (
            "[RE-CALL WARNING — the trust layer did not run for this memory: no certified "
            "calibration was available, so it was NOT judged at all. This is a degraded "
            "development-mode result; treat it as unverified context and do not rely on it.]\n"
            f"{hit.chunk.text}"
        )
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
    edge_candidates: EdgeCandidates | None = None,
    calibration_id: str | None = None,
    calibration_status: str | None = None,
    generation_binding: dict[str, str] | None = None,
    query_set_digest: str | None = None,
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

    **``edge_candidates`` rewinds supersession too**, when supplied. `trusted_search` gets it from
    `PgVectorStore.supersession_all()` whenever `known_as_of` is set, the store exposes that
    method and the retrieval returned hits; a store without it logs a warning and rewinds hits
    only. An edge becomes
    assertable when the superseding document is written, so its `indexed_at` dates the edge; a
    chain resolves per step, to the successor current at the instant. Point-in-time replay is
    then honest about which memories existed *and* about which were current.

    Transaction time comes from `first_indexed_at`, the FIRST write, preserved across
    re-indexing. `indexed_at` is the LAST write and using it here claimed a memo edited today had
    never existed before the edit, so every replay of an earlier instant reported an empty store.
    A hit carrying neither is left visible rather than hidden.

    Narrower residues, both fail-closed: an edge whose superseding file has no recorded date
    applies unconditionally, and ``unresolved`` is not rewound, so an ambiguous claim written
    after the instant still forces an abstention at it.

    Omitting ``edge_candidates`` keeps the old behaviour exactly, so no caller changes.

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
        now = now.replace(tzinfo=UTC)
    if known_as_of is not None:
        # BOTH operands, and NOT only when `known_as_of` happened to be naive. Nesting the
        # `edge_candidates` half inside that branch left the mirror combination raising: an AWARE
        # `known_as_of` against a naive hand-built date map still hit the comparison, so the more
        # careful caller was the one that crashed. Two rounds of review to get one comment to
        # match its own code.
        known_as_of = _as_utc(known_as_of)
        if edge_candidates:
            edge_candidates = {
                target: [(f, None if w is None else _as_utc(w)) for f, w in claims]
                for target, claims in edge_candidates.items()
            }
    cal = calibration or _UNCALIBRATED
    trusted: list[TrustedHit] = []
    for hit in result.hits:
        verdict, validity = _verdict(
            hit, supersession, cal.threshold, now, unresolved, known_as_of, edge_candidates
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
        gap_warning=result.gap_warning,
        staleness=result.staleness,
        diagnostics=result.diagnostics,
        calibration_id=calibration_id,
        calibration_status=(
            calibration_status
            if calibration_status is not None
            else ("legacy_unbound" if calibration is not None else "missing")
        ),
        tenant_id=(generation_binding or {}).get("tenant_id"),
        generation_id=(generation_binding or {}).get("generation_id"),
        pipeline_fingerprint=(generation_binding or {}).get("pipeline_fingerprint"),
        corpus_fingerprint=(generation_binding or {}).get("corpus_fingerprint"),
        query_set_digest=query_set_digest,
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
    entailment: EntailmentJudge | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    retrieval_profile: str = "legacy",
    index_generation: str = "legacy",
    policy: TrustPolicy | None = None,
    document_expansion: DocumentExpansionPolicy | None = None,
    structural_expansion: StructuralExpansionPolicy | None = None,
    _generation_snapshot: bool = True,
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
    snapshot = getattr(store, "snapshot", None)
    if _generation_snapshot and callable(snapshot):
        with snapshot():
            return trusted_search(
                store,
                embedder,
                query,
                k=k,
                source=source,
                calibration=calibration,
                now=now,
                known_as_of=known_as_of,
                reranker=reranker,
                entailment=entailment,
                candidate_k=candidate_k,
                retrieval_profile=retrieval_profile,
                index_generation=index_generation,
                policy=policy,
                document_expansion=document_expansion,
                structural_expansion=structural_expansion,
                _generation_snapshot=False,
            )
    # single fallback resolution: the retriever's gap threshold and the verdict threshold must
    # always come from the same calibration (or the same uncalibrated default)
    #
    # Generation stores resolve through the tenant-scoped database repository. Legacy JSON is
    # deliberately never auto-loaded: it has no tenant, generation, pipeline, corpus, or labelled
    # query-set binding and therefore cannot establish that its threshold applies here.
    active_policy = policy or TrustPolicy()
    calibration_id: str | None = None
    calibration_status = "legacy_unbound" if calibration is not None else "missing"
    query_set_digest: str | None = None
    generation_binding: dict[str, str] | None = None
    # A dependency fault must be distinguishable from a calibration verdict. Reading the binding
    # and resolving the artifact both touch the database, so a failure in either is
    # DEPENDENCY_UNAVAILABLE ("the gate could not run") and never CALIBRATION_MISSING ("the gate
    # ran and found nothing"). Collapsing the two would tell an operator to go and calibrate
    # while the real fault was an unreachable control plane.
    try:
        binding_reader = getattr(store, "generation_binding", None)
        if callable(binding_reader):
            generation_binding = binding_reader()
        resolver = getattr(store, "resolve_calibration", None)
        if calibration is None and callable(resolver):
            resolution = resolver()
            calibration_status = resolution.status.value
            artifact = resolution.artifact
            if artifact is not None:
                calibration = artifact.runtime
                calibration_id = artifact.calibration_id
                query_set_digest = artifact.query_set_digest
        elif calibration is None:
            _warn_uncalibrated(embedding_profile_id(embedder))
    except TrustRefusal:
        raise
    except Exception as exc:
        binding = generation_binding or {}
        raise TrustRefusal(
            code=TrustFailureCode.DEPENDENCY_UNAVAILABLE,
            calibration_status=calibration_status,
            tenant_id=binding.get("tenant_id") or getattr(store, "tenant", None),
            generation_id=binding.get("generation_id"),
        ) from exc

    # THE GATE. It sits here, above `retriever.search(...)`, and that position is the whole
    # guarantee: a refusal raised before any `query_dense` call cannot leak corpus bytes, because
    # none were ever fetched. Placing it after retrieval and filtering the payload would have
    # left a sanitiser that someone eventually forgets to call.
    failure_code = code_for_status(calibration_status)
    if failure_code is not None:
        binding = generation_binding or {}
        # No active generation outranks any calibration verdict, and for the same reason
        # `tenant_readiness` checks it first: telling an operator to recalibrate against a
        # generation that does not exist is useless advice. The two call sites must agree, or the
        # readiness endpoint and the search path would name different faults for one cause.
        if not binding.get("generation_id"):
            failure_code = TrustFailureCode.INDEX_NOT_READY
        if active_policy.strict:
            raise TrustRefusal(
                code=failure_code,
                calibration_status=calibration_status,
                tenant_id=binding.get("tenant_id") or getattr(store, "tenant", None),
                generation_id=binding.get("generation_id"),
                calibration_id=calibration_id,
                pipeline_fingerprint=binding.get("pipeline_fingerprint"),
                corpus_fingerprint=binding.get("corpus_fingerprint"),
                query_set_digest=query_set_digest,
            )
        METRICS.increment("recall_degraded_searches_total", code=failure_code.value)
    cal = calibration or _UNCALIBRATED
    retriever = HybridRetriever(
        store,
        embedder,
        reranker=reranker,
        gap_threshold=cal.threshold,
        candidate_k=candidate_k,
        retrieval_profile=retrieval_profile,
        index_generation=index_generation,
    )
    result = retriever.search(query, k=k, source=source)
    if document_expansion is not None:
        result = expand_retrieval_by_source(result, retriever.search, document_expansion)
    if structural_expansion is not None:
        result = expand_retrieval_by_structure(result, retriever.search, structural_expansion)
    # ONE call when the candidates are needed: `supersession_all()` returns edges and their
    # dates from a single validated scan, so they cannot describe different scans. Read
    # defensively: `store` is duck-typed in
    # several places (tests and downstream adapters implement only the read surface they need),
    # and a store without the method degrades to exactly the pre-change behaviour.
    supersession: dict[str, str] = {}
    unresolved: frozenset[str] = frozenset()
    edge_candidates: EdgeCandidates | None = None
    if result.hits:
        fetch_all = getattr(store, "supersession_all", None) if known_as_of is not None else None
        if fetch_all is not None:
            supersession, unresolved, edge_candidates = fetch_all()
        else:
            if known_as_of is not None:
                _warn_no_edge_dates(type(store).__name__)
            supersession, unresolved = store.supersession()
    trust_started = time.perf_counter()
    trusted = evaluate(
        result,
        supersession,
        calibration,
        now or datetime.now(UTC),
        unresolved,
        known_as_of,
        edge_candidates,
        calibration_id,
        calibration_status,
        generation_binding,
        query_set_digest,
    )
    stage_ms = dict(trusted.diagnostics.stage_ms)
    stage_ms["trust_evaluation"] = round((time.perf_counter() - trust_started) * 1000.0, 3)
    trusted = replace(
        trusted,
        diagnostics=replace(trusted.diagnostics, stage_ms=stage_ms),
    )
    if failure_code is not None:
        # Development degradation. Reached only when the policy explicitly allows it, since
        # strict already raised above. The result is ALWAYS marked degraded and is never
        # `calibrated`, whichever branch below runs.
        trusted = replace(
            trusted,
            trust_state=TrustState.DEGRADED.value,
            failure_code=failure_code.value,
        )
        if calibration is None:
            # No threshold exists at all: the trust system is genuinely unavailable. Every
            # verdict is overwritten rather than adjusted, because the ones `evaluate` just
            # computed came from `_UNCALIBRATED`'s 0.50 floor, so `ok` would mean "cleared a
            # threshold nobody chose". `abstained` is forced False for the mirror reason:
            # abstaining is itself a trustworthy decision, and no gate licensed it.
            trusted = replace(
                trusted,
                hits=[replace(hit, verdict="unverified") for hit in trusted.hits],
                abstained=False,
                reason="",
            )
        # The other branch: the CALLER passed an explicit `Calibration`. A threshold exists and
        # the caller chose it deliberately, so the verdicts `evaluate` produced are meaningful
        # and are left alone — this is the path every abstention benchmark measures, and blanking
        # it would make "does RE-call abstain correctly" unmeasurable. What the caller does NOT
        # get is a trust claim: the artifact is not certified-bound to this tenant and
        # generation, so `trust_state` stays `degraded` and `calibrated` stays False. Strict mode
        # refuses this case outright, so it can never reach a production serving path.
    if entailment is not None:
        from recall.entailment import apply_entailment

        trusted = apply_entailment(trusted, entailment)
    return trusted
