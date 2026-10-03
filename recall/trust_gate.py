"""The trust gate: verdicts over one retrieval result.

``evaluate`` decides, hit by hit, whether a retrieved chunk may be served (supersession, validity
windows, dependency invalidation, the calibrated score floor), and ``order_promoted`` orders what
it promoted. ``recall.trust`` runs a whole trusted search and calls this; ``recall.related`` and the
MCP graph cache call it directly. Kept apart so those callers do not import the retriever, the
reranker and the embedders that a full search needs. Its public names, and ``_verdict``, are
re-exported from ``recall.trust``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from recall.calibration import Calibration
from recall.dependency_invalidation import (
    DependencyProjection,
    authority_from_metadata,
    dependencies_from_metadata,
    source_file,
)
from recall.frontmatter import validity_bounds
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.observability import METRICS
from recall.trust_verdicts import abstain_reason, decision_state_for
from recall.types import (
    Authority,
    Provenance,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,
    Validity,
    Verdict,
)

if TYPE_CHECKING:
    from recall.store import EdgeCandidates

_UNCALIBRATED = Calibration(embedder="uncalibrated", threshold=DEFAULT_GAP_THRESHOLD)


def _as_utc(value: datetime) -> datetime:
    """A naive datetime read as UTC. One helper, because the three comparison sites in this
    module (`now`, `known_as_of`, and each candidate's date) all put a caller-supplied instant
    against a TIMESTAMPTZ from the store, and normalising some of them raises where normalising
    none of them did not."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


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
    if getattr(hit, "score_kind", "dense_cosine") == "dense_cosine" and hit.score < threshold:
        return "low_confidence", validity
    return "ok", validity


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
    dependency_projection: DependencyProjection | None = None,
    dependency_mode: str = "off",
    record_metrics: bool = True,
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

    ``record_metrics=False`` supports a private diagnostic pass over already fetched candidates.
    It changes only counters. Verdicts, ordering, abstention, and returned metadata stay identical.

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
        authority: Authority = "unknown"
        dependencies: tuple[str, ...] = ()
        metadata_error = False
        try:
            authority = authority_from_metadata(meta)
            dependencies = dependencies_from_metadata(meta)
        except ValueError:
            metadata_error = True
            verdict = "invalid_metadata"
        invalidation = None
        if (
            not metadata_error
            and dependency_mode == "enforce"
            and dependency_projection is not None
        ):
            invalidation = dependency_projection.reason_for(source_file(hit.chunk))
            if invalidation is not None and verdict == "ok":
                verdict = "dependency_invalidated"
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
                    first_indexed_at=getattr(hit, "first_indexed_at", None),
                ),
                validity=validity,
                authority=authority,
                dependencies=dependencies,
                invalidation=invalidation,
            )
        )
    invalidated_files = {
        h.provenance.file for h in trusted if h.verdict == "dependency_invalidated"
    }
    promoted_files = {
        h.validity.superseded_by
        for h in trusted
        if h.verdict == "superseded" and h.cosine >= cal.threshold
    } - invalidated_files
    trusted = [
        replace(h, verdict="ok")
        if h.verdict == "low_confidence" and h.provenance.file in promoted_files
        else h
        for h in trusted
    ]
    ok = [h for h in trusted if h.verdict == "ok"]
    rest = [h for h in trusted if h.verdict != "ok"]
    abstained = not ok
    decision_state = decision_state_for(trusted, gap_warning=result.gap_warning)
    # The operational questions this library exists to answer — how often does it abstain, and
    # what is it demoting — are answerable only if they are counted where the decision is made.
    if record_metrics:
        METRICS.increment("recall_searches_total")
        if abstained:
            METRICS.increment("recall_abstentions_total")
        if result.gap_warning:
            METRICS.increment("recall_gap_warnings_total")
        if result.staleness.stale:
            METRICS.increment("recall_stale_results_total")
        for trusted_hit in trusted:  # not `hit`: that name is bound to a ScoredChunk above
            METRICS.increment("recall_verdicts_total", verdict=trusted_hit.verdict)
            METRICS.increment("recall_authority_records_total", authority=trusted_hit.authority)
            if (
                trusted_hit.invalidation is not None
                and trusted_hit.verdict == "dependency_invalidated"
            ):
                METRICS.increment(
                    "recall_dependency_invalidations_total",
                    cause=trusted_hit.invalidation.cause,
                )
                if len(trusted_hit.invalidation.path) > 2:
                    METRICS.increment("recall_dependency_transitive_invalidations_total")
    return TrustedResult(
        query=result.query,
        hits=ok + rest,
        abstained=abstained,
        reason=abstain_reason(rest) if abstained else "",
        decision_state=decision_state,
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


def order_promoted(
    trusted: TrustedResult, pool_index: dict[str, int], ordering: str
) -> TrustedResult:
    """Reorder the verdict-`ok` hits according to `SuccessorExpansionPolicy.ordering`.

    `evaluate` returns ``ok + rest`` with pool position preserved inside each group, so a fetched
    successor, appended last by the expander, is last among `ok` however relevant it is. Measured
    over 6 absent-successor queries: promoted 6 of 6, then ranked 5, 5, 5, 5 and 2.

    `pool_index` must be built from the RetrievalResult BEFORE `evaluate`, because ``ok + rest``
    has already destroyed the interleaving this needs: the predecessor sits in `rest` and its
    position relative to the `ok` hits is not recoverable afterwards.

    A successor inherits the LOWEST pool index among the superseded hits naming it. A document is
    several chunks, and the rank it earned is the best one any of them reached, not the last.

    `rest` is never reordered. It is the demoted material, and its order is not a claim.
    """
    if ordering == "pool":
        return trusted
    ok = [hit for hit in trusted.hits if hit.verdict == "ok"]
    rest = [hit for hit in trusted.hits if hit.verdict != "ok"]
    if not ok:
        return trusted
    inherited: dict[str, int] = {}
    for hit in trusted.hits:
        target = hit.validity.superseded_by
        index = pool_index.get(hit.chunk.id)
        if not target or index is None:
            continue
        if index < inherited.get(target, index + 1):
            inherited[target] = index
    # A hit the pool never saw sorts last rather than first. Reachable only if a caller hands in a
    # partial index, and defaulting to 0 there would silently promote an unknown to the top.
    last = len(pool_index) + 1

    def _own(hit: TrustedHit) -> int:
        return pool_index.get(hit.chunk.id, last)

    if ordering == "promoted_first":
        ok.sort(key=lambda hit: (0 if hit.provenance.file in inherited else 1, _own(hit)))
    else:  # "inherit"
        ok.sort(key=lambda hit: inherited.get(hit.provenance.file or "", _own(hit)))
    return replace(trusted, hits=ok + rest)
