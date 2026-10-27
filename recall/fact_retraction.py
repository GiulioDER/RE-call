"""The facts whose memos are gone: retracted by `recall_forget`, hidden while a memo is absent.

A fact applied with `recall_apply_fact` cites evidence cards, and each card names the memo it came
from. Two things take a memo away, and they are treated differently on purpose:

- `recall_forget` is a DECISION. The memo is erased from every generation, so the facts it was
  the only evidence for get a permanent `retracted` event (`prepare_fact_erasure`, rule in
  `recall.fact_ledger.plan_retractions`), and the values it supplied are then erased from the
  ledger (`recall.fact_redaction`).
- A memo deleted from disk, pruned by the indexer, or left out of a promoted generation is a
  STATE. A rollback, a restored file or a repaired sync brings it back, so nothing is written: the
  projections hide a fact none of whose memos the store serves now (`served_sources`, rule in
  `recall.provenance_controller.absent_fact_ids`), and show it again when a memo returns. A lite
  or PostgreSQL store also answers with which content each memo is served and which memos its supersession
  graph retires (`recall.provenance_controller.ServedState`).
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import cache
import hashlib
import json
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from recall.observability import get_logger
from recall.supersession import SupersessionGraph, store_supersession_graph, window_contains

if TYPE_CHECKING:
    from recall.provenance_controller import ServedSources
    from recall.types import EvidenceCard

_log = get_logger("fact_retraction")
_P = ParamSpec("_P")
_V = TypeVar("_V")

#: The variable naming the controller role's DSN; `recall_mcp.provenance` reads the same one.
FACT_WRITE_DSN_ENV = "RECALL_FACT_WRITE_DSN"


def fact_ledger_for(store: object, env: Mapping[str, str] | None = None) -> Any | None:
    """The fact ledger `recall_apply_fact` writes for `store`, or None for a store that has none.

    A lite store's ledger is in its own file. A PostgreSQL store's is written through the
    controller role (`RECALL_FACT_WRITE_DSN` in `env`), else the store's own DSN, as
    `recall_mcp.provenance` resolves it.
    """
    import os

    from recall.lite import LiteStore
    from recall.store import PgVectorStore

    if isinstance(store, LiteStore):
        from recall.lite.facts import LiteFactLedger

        return LiteFactLedger(store)
    if isinstance(store, PgVectorStore):
        from recall.fact_ledger import PostgresFactLedger

        configured = (os.environ if env is None else env).get(FACT_WRITE_DSN_ENV, "").strip()
        return PostgresFactLedger(configured or store.dsn, tenant_id=store.tenant)
    return None


def prepare_retraction(
    ledger: Any,
    store: Any,
    shadow_store: Any | None,
    *,
    named: Collection[str],
    erased: Collection[str],
    withdraw: bool = False,
) -> Callable[[], int]:
    """The retraction half of `prepare_fact_erasure`: a callable answering how many facts it retracted."""
    return prepare_fact_erasure(
        ledger, store, shadow_store, named=named, erased=erased, withdraw=withdraw
    ).retract


@dataclass
class FactErasure:
    """What a deletion owes the fact ledger, prepared while the memos' rows still exist.

    `retract()` appends the retractions and answers how many facts it retracted; `redact()`, run
    after it, erases the values those memos supplied (`recall.fact_redaction`) and answers how many
    fact values it erased. Both run after the chunks are deleted, and both are no-ops when there
    is nothing left to do, so a forget re-run with the same names completes either.
    """

    retract: Callable[[], int]
    redact: Callable[[], int]


def prepare_fact_erasure(
    ledger: Any,
    store: Any,
    shadow_store: Any | None,
    *,
    named: Collection[str],
    erased: Collection[str],
    redactor: Any | None = None,
    withdraw: bool = False,
) -> FactErasure:
    """Prepare, before any chunk is deleted, the retraction and redaction a deletion owes the ledger.

    With `withdraw` it is a withdrawal instead (`recall_withdraw`): the facts are taken back with
    `withdrawn` events, which also undo the supersessions they made
    (`recall.fact_ledger.plan_withdrawals`), on the same names. The redaction is the same either
    way, since a withdrawn memo is deleted like a forgotten one.

    A card names its memo by the root-relative file a search hit shows, while deletion keys on the
    absolute source, so every card's name is resolved here, while its rows still exist: the cards
    every ledger event cites and, with a `redactor`, the card table's. `erased` is the absolute
    sources about to be deleted; `named` is every name the caller used for them (a forget's
    requested identifiers, or the erased sources themselves).

    A card is GONE when its name resolves to nothing outside `erased` (nothing at all when an
    earlier deletion took it), and TOUCHED when `named` holds its name or this deletion erases its
    memo, so a forget re-run after a failed retraction or redaction, which deletes nothing,
    completes it. A card whose name was not known here (applied while this ran) counts as
    surviving: a retraction may fail to happen, never remove a fact that still has evidence. A
    redacted card (`recall.types.RedactedEvidenceCard`) is gone: its memo was forgotten, and a
    namesake is another memo.
    """
    from recall.types import RedactedEvidenceCard

    cited = {
        card.source
        for event in ledger.events
        for card in event.evidence_cards
        if not isinstance(card, RedactedEvidenceCard)
    }
    if redactor is not None:
        cited |= set(redactor.card_sources())
    names = sorted(cited)
    resolved: dict[str, list[str]] = dict(store.sources_for_identifiers(names)) if names else {}
    if shadow_store is not None and names:
        for name, values in shadow_store.sources_for_identifiers(names).items():
            bucket = resolved.setdefault(name, [])
            bucket.extend(value for value in values if value not in bucket)
    known = set(names)
    gone_sources = set(erased)
    touched_names = set(named) | gone_sources
    touched_names.update(name for name, values in resolved.items() if set(values) & gone_sources)
    for target in (store, shadow_store):
        recover = getattr(target, "forgotten_source_aliases", None)
        if callable(recover):
            touched_names.update(recover(sorted(named)))
    # Keep only source names, never memo text. The mapping must commit before the chunks vanish,
    # so a failed append can be retried with the original absolute source.
    append = getattr(store, "append_audit_event", None)
    if callable(append):
        for source in sorted(gone_sources):
            aliases = sorted(name for name, values in resolved.items() if source in values)
            if aliases:
                identity = json.dumps([store.tenant, source, aliases], ensure_ascii=False)
                append(
                    "forget_source_aliases", {"aliases": aliases}, source_uri=source,
                    event_id="forget_alias_" + hashlib.sha256(identity.encode()).hexdigest(),
                )

    def touched(card: EvidenceCard) -> bool:
        return card.source in touched_names

    remaining_cache: list[dict[str, list[str]]] = []

    def remaining() -> dict[str, list[str]]:
        # Look again after deletion. Concurrent forgets may each have observed the other's memo
        # before either deletion; the last committer must see that both are now gone.
        if not remaining_cache:
            found = dict(store.sources_for_identifiers(names)) if names else {}
            if shadow_store is not None and names:
                for name, values in shadow_store.sources_for_identifiers(names).items():
                    found.setdefault(name, []).extend(values)
            remaining_cache.append(found)
        return remaining_cache[0]

    def gone_name(name: str) -> bool:
        return name in known and not remaining().get(name)

    def gone(card: EvidenceCard) -> bool:
        return isinstance(card, RedactedEvidenceCard) or gone_name(card.source)

    def retract() -> int:
        append = ledger.withdraw_forgotten if withdraw else ledger.retract_forgotten
        written = append(tenant_id=store.tenant, touched=touched, gone=gone)
        return len({fact_id for event in written for fact_id in event.supersedes_fact_ids})

    def redact() -> int:
        if redactor is None:
            return 0
        report = redactor.redact(
            erased=lambda name: name in touched_names and gone_name(name),
            names=sorted(touched_names),
        )
        return int(report.facts_redacted)

    return FactErasure(retract=retract, redact=redact)


def _settled(fn: Callable[_P, _V]) -> Callable[_P, _V]:
    """`fn` run once per argument tuple, its failure remembered like its answer.

    `functools.cache` keeps no exception, so a lookup that timed out would be run, and time out,
    again for every check that reads it; here the second caller gets the same error at once.
    """
    outcomes: dict[tuple[object, ...], tuple[_V | None, Exception | None]] = {}

    def call(*args: _P.args, **kwargs: _P.kwargs) -> _V:
        if args not in outcomes:
            try:
                outcomes[args] = (fn(*args, **kwargs), None)
            except Exception as exc:  # BROAD-CATCH: error-translation  # kept and raised to every caller
                outcomes[args] = (None, exc)
        value, error = outcomes[args]
        if error is not None:
            raise error
        return value  # type: ignore[return-value]

    return call


def served_sources(store: object, *, failures: list[str] | None = None) -> ServedSources | None:
    """The memo names `store` serves now, with their content, supersession and stating rule on a
    lite or PostgreSQL store; None without a lookup. On a generation store every answer is the
    SERVED generation's, and ONE generation's: the first lookup reads which generation the store
    serves (pinned, fixed, else active) and every lookup is pinned to it, so a promote between two
    lookups cannot pair one generation's names with another's content. Names resolve through
    `served_sources_for_identifiers`, not `sources_for_identifiers`, which spans every generation
    for erasure, so a stated card lives only while that generation holds the content it recorded.
    Presence is the keys of the same resolution, so one query answers both.

    Presence and supersession fail open: when those lookups error every name counts as served and
    unsuperseded, so an outage shows applied facts as the ledger alone would rather than hiding
    all of them. A stated card's two checks, admission and content, fail CLOSED: stated facts are
    recorded with no review, so one whose content cannot be checked is not shown (review of PR 935,
    where a timed-out hash lookup brought an edited memo's old value back beside its new one).
    Either way the answer is incomplete, so each failed lookup is named in `failures` (`presence`,
    `supersession`, `admission`, `content`) for the caller to report (`recall_current_facts`'s
    `served_state`).
    """
    lookup = getattr(store, "served_identifiers", None)
    if not callable(lookup):
        return None

    def failed(what: str) -> None:
        if failures is not None and what not in failures:
            failures.append(what)

    from recall.lite import LiteStore
    from recall.store import PgVectorStore

    if not isinstance(store, (LiteStore, PgVectorStore)):

        @cache
        def plain(names: frozenset[str]) -> Collection[str]:
            try:
                return {str(name) for name in lookup(sorted(names))}
            except Exception:  # BROAD-CATCH: fail-open  # the ledger's own answer, without hiding
                _log.exception("serving lookup for %d memo name(s) failed; hiding no fact", len(names))
                failed("presence")
                return names

        return plain
    from recall.provenance_controller import ServedState
    from recall.stated_facts import stating_rule

    # The served generation only on a generation store; a lite file or the legacy table has one
    # set of rows, so its erasure resolver already answers for what it serves, and its
    # `served_identifiers` is that resolver's keys.
    resolve = getattr(store, "served_sources_for_identifiers", None) or store.sources_for_identifiers
    pin = getattr(store, "pin_generation", None)

    @_settled
    def generation() -> str:
        return str(store.generation_id)

    def pinned() -> AbstractContextManager[object]:
        return pin(generation()) if callable(pin) else nullcontext()

    @_settled
    def resolved_names(names: frozenset[str]) -> dict[str, list[str]]:
        with pinned():
            return dict(resolve(sorted(names)))

    @_settled
    def raw_hashes() -> dict[str, str]:
        with pinned():
            return store.source_raw_hashes()

    @_settled
    def supersession() -> SupersessionGraph:
        with pinned():
            return store_supersession_graph(store)

    # Once per read: the rule resolves every prefix on disk.
    rule = _settled(lambda: stating_rule(store))

    @cache
    def served(names: frozenset[str]) -> Collection[str]:
        try:
            return set(resolved_names(names))
        except Exception:  # BROAD-CATCH: fail-open  # the ledger's own answer, without hiding
            _log.exception("serving lookup for %d memo name(s) failed; hiding no fact", len(names))
            failed("presence")
            return names

    @cache
    def content_hashes(names: frozenset[str]) -> dict[str, frozenset[str]] | None:
        try:
            admits = rule()
            if isinstance(admits, str):
                return {}
            resolved = resolved_names(names)
            raw = raw_hashes()
        except Exception:  # BROAD-CATCH: fail-closed  # no stated fact is current on an error
            # Not None, which `card_liveness` reads as "no content check": that brought an edited
            # memo's old value back as current whenever this lookup timed out while admission,
            # computed from the same cached names, still passed.
            _log.exception("content lookup for %d memo name(s) failed; listing no stated fact", len(names))
            failed("content")
            return {}
        return {
            # Admission and content must come from the same absolute source. A relative
            # alias can also resolve to an excluded namesake with a different hash.
            name: frozenset(raw[source] for source in sources if raw.get(source) and admits.admits(name, source))
            for name, sources in resolved.items()
        }

    @cache
    def stated_admitted(names: frozenset[str]) -> frozenset[str]:
        """The names whose memos the operator's rule lets state facts now. Fails CLOSED: a rule
        that cannot be computed admits nothing, so no memo states facts on an error."""
        try:
            admits = rule()
            if isinstance(admits, str) or not names:
                return frozenset()
            resolved = resolved_names(names)
        except Exception:  # BROAD-CATCH: fail-closed  # no stated fact is current on an error
            _log.exception("stating rule lookup failed; listing no stated fact")
            failed("admission")
            return frozenset()
        return frozenset(
            name for name, sources in resolved.items() if any(admits.admits(name, s) for s in sources)
        )

    @cache
    def superseded(names: frozenset[str], now: datetime | None = None) -> frozenset[str]:
        """Names a served memo the operator admits supersedes at valid time `now`. A memo that
        may not state facts (an upload, a file outside the prefixes) cannot retire another memo's
        facts either, and an ambiguous reference retires nothing; search's own verdicts are not
        changed by either.

        Every claimant of a target counts, not the edge map's last-row winner: otherwise an upload
        sorting after an admitted claimant would displace it and keep the old value current. A
        claimant retires a memo only while the claimant itself is valid at `now` (the window rule,
        as search applies it); with no instant, every claim applies. A memo is also retired
        through any branch whose every member is admitted: in a, b (expired) superseding a, c
        superseding b, c retires a even when an excluded sibling wins search's single chain.
        Expired intermediates are traversed, excluded intermediates are not. A name two
        roots hold is keyed in the graph by each source (`SupersessionGraph.collided`), and is
        retired when any of them is: looked up by name, it never was (audit SEC-005).
        """
        try:
            # Only admitted memos can retire facts, so with stating off nothing can be retired, and
            # the store's supersession graph is not asked at all: that lookup would otherwise run on
            # every read of every PostgreSQL deployment that never turned stating on.
            admits = rule()
            if isinstance(admits, str):
                return frozenset()
            graph = supersession()
            collided = frozenset(name for name in names if name in graph.collided)
            copies = resolved_names(collided) if collided else {}
            keys = {
                name: tuple(copies.get(name, ())) if name in collided else (name,)
                for name in names
            }
            # Resolve reachable claimants once. This structural walk does not grant authority;
            # the retirement walk below stops at every excluded claimant.
            claims: dict[str, tuple[str, ...]] = {}
            pending = [key for group in keys.values() for key in group]
            while pending:
                key = pending.pop()
                if key in claims:
                    continue
                claims[key] = graph.live_claimants(key)
                pending.extend(claimant for claimant in claims[key] if claimant not in claims)
            claimants = resolved_names(
                frozenset(claimant for group in claims.values() for claimant in group)
            )
            # Keep the graph's canonical keys, including source-qualified copies. Admission
            # must hold for every source a key resolves to, so ambiguous authority retires nothing.
            admitted = frozenset(
                name for name, sources in claimants.items()
                if sources and all(admits.admits(name, source) for source in sources)
            )

            def retires(key: str) -> bool:
                # Each node is visited once per starting copy, without enumerating paths. The
                # starting copy is already seen, so a cycle cannot make it retire itself.
                seen = {key}
                pending = [key]
                while pending:
                    for claimant in claims.get(pending.pop(), ()):
                        if claimant in seen:
                            continue
                        seen.add(claimant)
                        if claimant not in admitted:
                            continue
                        if now is None or window_contains(graph.windows.get(claimant), now):
                            return True
                        pending.append(claimant)
                return False

            return frozenset(name for name, group in keys.items() if any(retires(key) for key in group))
        except Exception:  # BROAD-CATCH: fail-open  # no memo's facts are retired
            _log.exception("supersession lookup failed; retiring no memo's facts")
            failed("supersession")
            return frozenset()

    return ServedState(
        served, content_hashes=content_hashes, superseded_at=superseded, stated_admitted=stated_admitted
    )


__all__ = [
    "FACT_WRITE_DSN_ENV",
    "FactErasure",
    "fact_ledger_for",
    "prepare_fact_erasure",
    "prepare_retraction",
    "served_sources",
]
