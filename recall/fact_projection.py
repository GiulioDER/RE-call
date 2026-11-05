"""What a fact ledger's history says, computed without the ledger or the controller.

It holds the request-id forms the ledger mints for its own events (`reserved_request_id`),
retraction and withdrawal (`applications`, `retracted_keys`, `withdrawn_event_ids`), validity
windows and the interval helpers (`fact_window`), the supersession index (`fact_supersession`),
the write-ordering helper `ordered_after` and the conflict-key hash `fact_conflict_key`.

Moved verbatim out of `recall.provenance_controller`, which re-exports every name, so both
import paths give the same objects. It imports the event vocabulary (`recall.fact_events`) and
never the controller.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from recall.fact_events import DecisionCode, FactEvent, _as_utc, canonical_json, fact_identity
from recall.types import AtomicFact


def asserts_fact(event: FactEvent) -> bool:
    """Whether an event is an application of its fact by its own record: `asserted`, and not a
    DUPLICATE. Projections decide over the whole history instead (`applications`)."""
    return event.event_type == "asserted" and event.decision_code != DecisionCode.DUPLICATE


def applications(events: Iterable[FactEvent]) -> dict[str, FactEvent]:
    """The assertions projections count as applications of their fact, by event id, in ledger
    order, each as it is counted.

    A fact applied again under a new request records a second `asserted` event coded DUPLICATE,
    written while an application of the same fact is a candidate: it repeats the latest earlier
    application of its fact, and the two are one episode. It is skipped, so the fact is listed
    once, unless that application was withdrawn. Then the DUPLICATE is counted in its place AS
    it: with its supersession (a DUPLICATE is stored with none) and at its point in the ledger,
    so it hides exactly what the withdrawn application hid, and a value another memo still states
    keeps its supersession and all, which is what `plan_withdrawals` promises. A retracted
    application needs no stand-in: a retraction takes every application of the fact together,
    and a retracted successor still hides (`FactSupersession`). Before this, withdrawing the
    APPLIED memo of a fact whose DUPLICATE's memo had been forgotten made the value the fact
    superseded current again, while the other order kept it retired (2026-10-27 audit of #921,
    STAKES-001).
    """
    history = tuple(events)
    withdrawn = withdrawn_event_ids(history)
    counted: dict[str, FactEvent] = {}
    latest: dict[str, FactEvent] = {}
    for event in history:
        fact_id = event.fact_id
        if event.event_type != "asserted" or fact_id is None:
            continue
        if not asserts_fact(event):  # a DUPLICATE
            repeated = latest.get(fact_id)
            if repeated is None or repeated.event_id not in withdrawn:
                continue
            event = replace(
                event, supersedes_fact_ids=repeated.supersedes_fact_ids, created_at=repeated.created_at
            )
        counted[event.event_id] = event
        latest[fact_id] = event
    return counted


RetractionKey = tuple[str | None, str, frozenset[str]]


def retraction_key(event: FactEvent) -> RetractionKey:
    """What an `asserted` event and the `retracted` event that retracts it share.

    The fact, the generation and the exact set of evidence cards: a retraction hides the
    applications it was written for and no other, so the same fact applied again later from a new
    memo (new cards) is listed, and nothing depends on how two events with one timestamp sort.
    """
    return event.fact_id, event.generation_id, frozenset(card.card_id for card in event.evidence_cards)


def retracted_keys(events: Iterable[FactEvent]) -> frozenset[RetractionKey]:
    """The `retraction_key` of every assertion that a `retracted` or `withdrawn` event in `events`
    takes away.

    A withdrawal names its assertion by event id (`withdrawn_event_ids`) and carries no cards, so
    no copy of a value is written to withdraw it; its key is read from the assertion itself, which
    the same history holds. Every projection hides a withdrawn assertion as it hides a retracted
    one; `fact_supersession` is where the two differ.
    """
    history = tuple(events)
    withdrawn = withdrawn_event_ids(history)
    keys: set[RetractionKey] = {
        (fact_id, event.generation_id, frozenset(card.card_id for card in event.evidence_cards))
        for event in history
        if event.event_type == "retracted"
        for fact_id in event.supersedes_fact_ids
    }
    if withdrawn:
        keys.update(retraction_key(event) for event in history if event.event_id in withdrawn)
    return frozenset(keys)


def withdrawn_event_ids(events: Iterable[FactEvent]) -> frozenset[str]:
    """The ids of the assertions a `withdrawn` event in `events` names (`WITHDRAWAL_REQUEST_PREFIX`)."""
    width = len(WITHDRAWAL_REQUEST_PREFIX)
    return frozenset(
        event.request_id[width:]
        for event in events
        if event.event_type == "withdrawn" and event.request_id.startswith(WITHDRAWAL_REQUEST_PREFIX)
    )


#: A fact's validity as the ledger reads it: half-open, ``[valid_from, valid_until)``, either end
#: None when open. A successor's window decides when it hides what it supersedes.
FactWindow = tuple[datetime | None, datetime | None]


_EARLIEST = datetime.min.replace(tzinfo=UTC)


_LATEST = datetime.max.replace(tzinfo=UTC)


#: The request-id suffix of the `superseded` event that follows an applied assertion.
SUPERSESSION_REQUEST_SUFFIX = ":supersession"


#: The request-id prefix of a `retracted` event (`recall.fact_ledger.retraction_ids`).
RETRACTION_REQUEST_PREFIX = "retraction:"


#: The request-id prefix of a `withdrawn` event: the rest is the withdrawn assertion's event id.
WITHDRAWAL_REQUEST_PREFIX = "withdrawal:"


#: The request-id prefix of a memo's own statement (`recall.stated_facts.stated_request_id`).
STATED_REQUEST_PREFIX = "stated:"


def reserved_request_id(request_id: str) -> bool:
    """Whether `request_id` has a form the ledger mints for its own events, so no client may send it.

    A request id is unique per tenant, so a client request under `<id>:supersession` or
    `retraction:<event id>` took the server's slot: PostgreSQL, SQLite and lite answered it
    DUPLICATE and recorded nothing, while the in-memory ledger, which never registered the
    supersession id, appended a second row under it (2026-10-21 supersession audit). A `stated:`
    id is a memo's own statement (`recall.stated_facts`), which a client request would pre-empt.
    """
    return request_id.endswith(SUPERSESSION_REQUEST_SUFFIX) or request_id.startswith(
        (RETRACTION_REQUEST_PREFIX, WITHDRAWAL_REQUEST_PREFIX, STATED_REQUEST_PREFIX)
    )


def fact_window(fact: AtomicFact) -> FactWindow:
    return (
        _as_utc(fact.valid_from) if fact.valid_from else None,
        _as_utc(fact.valid_until) if fact.valid_until else None,
    )


def _holds(window: FactWindow | None, instant: datetime) -> bool:
    """Whether `instant` lies in `window`; an unknown window (None) holds at every instant."""
    if window is None:
        return True
    start, end = window
    return (start is None or instant >= start) and (end is None or instant < end)


def _without(
    intervals: list[tuple[datetime, datetime]], window: FactWindow | None
) -> list[tuple[datetime, datetime]]:
    """`intervals` less `window` (None, unknown, removes everything)."""
    if window is None:
        return []
    start, end = window[0] or _EARLIEST, window[1] or _LATEST
    out: list[tuple[datetime, datetime]] = []
    for low, high in intervals:
        if end <= low or high <= start:
            out.append((low, high))
            continue
        if low < start:
            out.append((low, start))
        if end < high:
            out.append((end, high))
    return out


#: A closed-open interval on the whole time line, ``[low, high)``.
Interval = tuple[datetime, datetime]


def _span(window: FactWindow | None) -> Interval:
    """`window` as an interval; an unknown window (None) is the whole line."""
    if window is None:
        return (_EARLIEST, _LATEST)
    return (window[0] or _EARLIEST, window[1] or _LATEST)


def _merge(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """The union of `intervals` as sorted disjoint intervals; an empty one (inverted) adds nothing."""
    merged: list[Interval] = []
    for low, high in sorted(intervals):
        if low >= high:
            continue
        if merged and low <= merged[-1][1]:
            if high > merged[-1][1]:
                merged[-1] = (merged[-1][0], high)
        else:
            merged.append((low, high))
    return tuple(merged)


@dataclass(frozen=True)
class FactSupersession:
    """Which assertion events a ledger's successors hide, and inside which windows.

    **Assertion events, not fact ids.** A successor hides the assertions of the facts it names
    that were recorded BEFORE it, never a later one: Paris, then Rome superseding it, then Paris
    asserted again leaves the third assertion current, where hiding the fact id hid Paris for
    good. **The window rule.** A successor hides only while its own fact holds (half-open
    `valid_from` to `valid_until`), so a successor dated from next quarter leaves today's value
    current, an expired bounded successor gives the old value back, and a 2020 to 2022
    correction leaves the open-ended present alone. **Chains**: what hides a successor hides what
    it hid, each inside its own window, so a value superseded by a temporary one and then by its
    replacement stays hidden while either holds. Whether a successor is still live does not
    matter: forgetting a successor removes its value and gives nothing back. **Withdrawal** is
    the one way back (owner decision, 2026-10-22): a withdrawn successor (`withdrawn_event_ids`)
    hides nothing in its own window but still passes on what hides IT, so in A, then B superseding
    A, then C superseding B, withdrawing B leaves A hidden by C alone (the chain is spliced), and
    withdrawing the head B of A, B makes A current again.

    `hidden` holds, per assertion event, the union of every such window as sorted disjoint
    intervals. A union, not a list: a value asserted again keeps its fact id, so each later
    successor hides every earlier assertion of it, and listing the windows grew by a factor of
    about 1.6 per flip (2026-10-21 supersession audit, SEC-002).
    """

    hidden: Mapping[str, tuple[Interval, ...]]

    def hidden_at(self, event: FactEvent, instant: datetime) -> bool:
        moment = _as_utc(instant)
        return any(low <= moment < high for low, high in self.hidden.get(event.event_id, ()))

    def in_force(self, event: FactEvent) -> list[tuple[datetime, datetime]]:
        """The intervals in which `event`'s fact holds and no successor hides it."""
        if event.fact is None:
            return []
        start, end = fact_window(event.fact)
        intervals = [(start or _EARLIEST, end or _LATEST)]
        for window in self.hidden.get(event.event_id, ()):
            intervals = _without(intervals, window)
            if not intervals:
                break
        return intervals

    def overlaps(self, event: FactEvent, fact: AtomicFact) -> bool:
        """Whether `event` is in force at some instant inside `fact`'s window."""
        start, end = fact_window(fact)
        low, high = start or _EARLIEST, end or _LATEST
        return any(max(a, low) < min(b, high) for a, b in self.in_force(event))


class _FactSuccessorIndex:
    """One fact identity's successors, without assertion/successor pairs.

    Navigation follows original successor order and skips assertions already entered by the
    DFS. The second tree is in timestamp order: it aggregates each successor's own window,
    plus inherited windows only AFTER that successor resolves. A visiting cycle member still
    contributes its own window, as in the original walk, but not unfinished inheritance.
    Timestamp suffix queries exclude the assertion's own unique ledger event.
    """

    def __init__(
        self,
        claims: Sequence[int],
        successors: Sequence[tuple[FactEvent, FactWindow | None]],
        children: Sequence[FactEvent | None],
        own_windows: Sequence[tuple[Interval, ...]],
    ) -> None:
        self.claims = tuple(claims)
        self.position = {claim: position for position, claim in enumerate(claims)}
        ordered = sorted(claims, key=lambda claim: (_as_utc(successors[claim][0].created_at), claim))
        self.times = [_as_utc(successors[claim][0].created_at) for claim in ordered]
        self.time_position = {claim: position for position, claim in enumerate(ordered)}
        # Two original-order claims per timestamp suffix suffice to exclude the assertion itself
        # when establishing the old root order, including legacy receipts.
        self.first: list[tuple[int, ...]] = [()] * (len(ordered) + 1)
        for position in range(len(ordered) - 1, -1, -1):
            self.first[position] = tuple(sorted((*self.first[position + 1], ordered[position]))[:2])
        self.size = 1 << (len(claims) - 1).bit_length()
        self.active: list[datetime | None] = [None] * (2 * self.size)
        self.unions: list[tuple[Interval, ...]] = [()] * (2 * self.size)
        for position, claim in enumerate(claims):
            event = successors[claim][0]
            if children[claim] is not None:
                self.active[self.size + position] = _as_utc(event.created_at)
        for position, claim in enumerate(ordered):
            self.unions[self.size + position] = own_windows[claim]
        for node in range(self.size - 1, 0, -1):
            self._refresh_active(node)
            self.unions[node] = _merge((*self.unions[2 * node], *self.unions[2 * node + 1]))

    def _refresh_active(self, node: int) -> None:
        left, right = self.active[2 * node], self.active[2 * node + 1]
        self.active[node] = right if left is None else left if right is None else max(left, right)

    def first_claim(self, written: datetime, own: int | None) -> int | None:
        return next(
            (claim for claim in self.first[bisect_left(self.times, written)] if claim != own),
            None,
        )

    def next_assertion(self, written: datetime) -> int | None:
        latest = self.active[1]
        if latest is None or latest < written:
            return None
        node = 1
        while node < self.size:
            left = self.active[2 * node]
            node = 2 * node if left is not None and left >= written else 2 * node + 1
        return self.claims[node - self.size]

    def enter(self, claim: int) -> None:
        node = self.size + self.position[claim]
        self.active[node] = None
        while node > 1:
            node //= 2
            self._refresh_active(node)

    def publish(self, claim: int, impact: tuple[Interval, ...]) -> None:
        node = self.size + self.time_position[claim]
        self.unions[node] = impact
        while node > 1:
            node //= 2
            self.unions[node] = _merge((*self.unions[2 * node], *self.unions[2 * node + 1]))

    def _range(self, low: int, high: int) -> tuple[Interval, ...]:
        low, high = low + self.size, high + self.size
        spans: list[Interval] = []
        while low < high:
            if low & 1:
                spans.extend(self.unions[low])
                low += 1
            if high & 1:
                high -= 1
                spans.extend(self.unions[high])
            low, high = low // 2, high // 2
        return _merge(spans)

    def hidden(self, written: datetime, own: int | None) -> tuple[Interval, ...]:
        low = bisect_left(self.times, written)
        own_position = self.time_position.get(own) if own is not None else None
        if own_position is not None and own_position >= low:
            return _merge((
                *self._range(low, own_position),
                *self._range(own_position + 1, len(self.times)),
            ))
        return self._range(low, len(self.times))


def fact_supersession(events: Sequence[FactEvent]) -> FactSupersession:
    """The successors of every assertion in events, in original ledger traversal order.

    Applied assertions name the facts they supersede; an unpaired legacy superseded receipt
    contributes its own window (unknown when it has no fact). Equal recorded times count as
    before, while self-reference never hides the assertion itself.

    The original iterative DFS order, including equal-time legacy cycles, is retained using
    per-identity indexes rather than expanding assertion/successor pairs. With compact interval
    unions, navigation, updates and queries take logarithmic work per recorded claim; disjoint
    interval output can itself grow. A legacy tie never selects a dense fallback.
    """
    withdrawn = withdrawn_event_ids(events)

    def withdrawn_for(successor: FactEvent) -> str | None:
        """The withdrawn assertion `successor` is, or speaks for, else None.

        A `superseded` event speaks for its assertion (`<assertion id>_sup`). It is only ever a
        successor of its own when that assertion carries no fact, which is what erasing a value
        leaves (PR 950's redaction); withdrawn, it must then be withdrawn with it.
        """
        if successor.event_id in withdrawn:
            return successor.event_id
        if successor.event_type == "superseded" and successor.event_id.endswith("_sup"):
            holder = successor.event_id[: -len("_sup")]
            if holder in withdrawn:
                return holder
        return None

    by_fact: dict[str, list[FactEvent]] = {}
    successors: list[tuple[FactEvent, FactWindow | None]] = []
    paired: set[str] = set()
    for event in applications(events).values():
        if event.fact_id is not None:
            by_fact.setdefault(event.fact_id, []).append(event)
            if event.supersedes_fact_ids and event.fact is not None:
                successors.append((event, fact_window(event.fact)))
                paired.add(event.request_id + SUPERSESSION_REQUEST_SUFFIX)
            elif event.supersedes_fact_ids and not asserts_fact(event):
                # A DUPLICATE standing in for a withdrawn application whose value a forget or
                # withdraw erased: it has no `superseded` event of its own, so it hides what that
                # application superseded, from its point in the ledger and with an unknown window,
                # as the erased application's own `superseded` event did.
                successors.append((event, None))
    for event in events:
        if event.event_type == "superseded" and event.request_id not in paired:
            successors.append((event, fact_window(event.fact) if event.fact is not None else None))

    # A withdrawn assertion contributes no window of its own, but passes on the windows
    # hiding it. An unpaired receipt for an erased assertion inherits through its holder too.
    # Index claims by that logical child so all aliases enter and publish together; no dense
    # assertion/successor fallback is needed when a history contains withdrawals.
    events_by_id = {event.event_id: event for event in events}
    children: list[FactEvent | None] = []
    own_windows: list[tuple[Interval, ...]] = []
    claims_by_child: dict[str, list[int]] = {}
    for claim, (successor, window) in enumerate(successors):
        holder = withdrawn_for(successor)
        child = successor if successor.event_type == "asserted" else events_by_id.get(holder or "")
        children.append(child)
        own_windows.append((_span(window),) if holder is None else ())
        if child is not None:
            claims_by_child.setdefault(child.event_id, []).append(claim)

    claims: dict[str, list[int]] = {}
    target_positions: list[dict[str, int]] = []
    successor_by_event = {event.event_id: claim for claim, (event, _window) in enumerate(successors)}
    for claim, (successor, _window) in enumerate(successors):
        positions: dict[str, int] = {}
        for position, fact_id in enumerate(successor.supersedes_fact_ids):
            if fact_id not in positions:
                positions[fact_id] = position
                claims.setdefault(fact_id, []).append(claim)
        target_positions.append(positions)
    indexes = {
        fact_id: _FactSuccessorIndex(group, successors, children, own_windows)
        for fact_id, group in claims.items()
        if fact_id in by_fact
    }
    # The old nested loop inserted a root at its first eligible pair: successor position,
    # target position within that successor, then assertion position in that fact's bucket.
    roots: list[tuple[int, int, int, FactEvent]] = []
    for fact_id, assertions in by_fact.items():
        index = indexes.get(fact_id)
        if index is None:
            continue
        for position, assertion in enumerate(assertions):
            first_eligible = index.first_claim(
                _as_utc(assertion.created_at), successor_by_event.get(assertion.event_id)
            )
            if first_eligible is not None:
                roots.append((first_eligible, target_positions[first_eligible][fact_id], position, assertion))

    def enter(event: FactEvent) -> None:
        for claim in claims_by_child.get(event.event_id, ()):
            for fact_id in target_positions[claim]:
                if fact_id in indexes:
                    indexes[fact_id].enter(claim)

    resolved: dict[str, tuple[Interval, ...]] = {}
    for _claim, _target_position, _assertion_position, root in sorted(roots, key=lambda item: item[:3]):
        if root.event_id in resolved:
            continue
        stack = [root]
        enter(root)
        while stack:
            event = stack[-1]
            index = indexes.get(event.fact_id) if event.fact_id is not None else None
            written = _as_utc(event.created_at)
            next_claim = index.next_assertion(written) if index is not None else None
            if next_claim is not None:
                next_event = children[next_claim]
                assert next_event is not None  # only claims with a child are navigable
                enter(next_event)
                stack.append(next_event)
                continue
            own = successor_by_event.get(event.event_id)
            hidden = index.hidden(written, own) if index is not None else ()
            resolved[event.event_id] = hidden
            stack.pop()
            for claim in claims_by_child.get(event.event_id, ()):
                impact = _merge((*own_windows[claim], *hidden))
                for fact_id in target_positions[claim]:
                    if fact_id in indexes:
                        indexes[fact_id].publish(claim, impact)
    return FactSupersession(resolved)


def ordered_after(moment: datetime, history: Sequence[FactEvent], fact: AtomicFact) -> datetime:
    """`moment`, or one microsecond after the latest recorded event of `fact`'s conflict key.

    Which assertion a successor hides is decided by ledger order (`fact_supersession`), and
    every ledger orders by `created_at` first. Two applications of one key at the same clock
    reading (an evaluation clock, a fixed test clock, a clock stepped back) would otherwise be
    ordered by event id, a hash, so a value asserted again after its successor could sort before
    it and stay hidden. Every writer of a key holds that key's lock while it reads `history`, so
    this keeps one key's events in the order they were written.
    """
    instant = _as_utc(moment)
    key = fact_conflict_key(fact)
    # The key is a hash of subject, predicate, namespace and context, so only an event that
    # shares the first three is hashed: hashing the whole history on every write, under the
    # key's lock, made writes six times slower at a thousand events (2026-10-21 audit, PERF-002).
    members = [
        event
        for event in history
        if event.fact is not None
        and event.fact.subject == fact.subject
        and event.fact.predicate == fact.predicate
        and event.fact.namespace == fact.namespace
        and fact_conflict_key(event.fact) == key
    ]
    # A `superseded` receipt carries no fact on the durable ledgers, and one whose successor a
    # forget erased pairs with nothing: it hides what it names over the whole line. It belongs
    # to the key of the facts it names, so it orders a new assertion of them too; without it a
    # value asserted again after its successor was forgotten took the receipt's own time under a
    # fixed clock and stayed hidden while the ledger answered APPLIED (2026-11-04 hunt, NUM-001).
    key_ids = {event.fact_id for event in members} | {fact_identity(fact)}
    members += [
        event
        for event in history
        if event.fact is None and key_ids.intersection(event.supersedes_fact_ids)
    ]
    latest = max((_as_utc(event.created_at) for event in members), default=None)
    if latest is not None and instant <= latest:
        return latest + timedelta(microseconds=1)
    return instant


def fact_conflict_key(fact: AtomicFact) -> str:
    """Identity used to serialize all competing values of one predicate context."""
    payload = {
        "namespace": fact.namespace,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "context": fact.context,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"conflict_{digest[:32]}"
