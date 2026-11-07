"""The facts whose memos are gone: retracted by `recall_forget`, hidden while a memo is absent.

A fact applied with `recall_apply_fact` cites evidence cards, and each card names the memo it came
from. Two things take a memo away, and they are treated differently on purpose:

- `recall_forget` is a DECISION. The memo is erased from every generation, so the facts it was
  the only evidence for get a permanent `retracted` event (`prepare_fact_erasure`, rule in
  `recall.fact_ledger.plan_retractions`), and the values it supplied are then erased from the
  ledger (`recall.fact_redaction`).
- A memo deleted from disk, pruned by the indexer, or left out of a promoted generation is a
  STATE. A rollback, a restored file or a repaired sync brings it back, so nothing is written to
  the fact ledger: the projections hide a fact none of whose memos the store serves now
  (`served_sources`, rule in `recall.provenance_controller.absent_fact_ids`), and show it again
  when a memo returns. The indexer's prune leaves one audit record of the chunks it removed
  (`prune_source_chunks`), so a later forget of another memo keeps a pruned card as evidence
  rather than reading its memo as forgotten; a memo left out of a promoted generation has no such
  record, and a forget still reads it by name. A lite
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

from recall.chunk_records import FORGET_SOURCE_CHUNKS, PRUNE_SOURCE_CHUNKS, ChunkRecord
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
    `recall_mcp.provenance` resolves it when `env` is the MCP runtime environment. With no `env`
    it reads the process environment, which lacks a DSN the server resolved from a secret, so
    an MCP caller passes its runtime environment.
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


def chunk_targets(store: Any, shadow_store: Any | None) -> list[Any]:
    """The stores whose chunks a forget erases and a prune records, by chunk id: `store` and the
    shadow, never a generation store, which reads chunks one generation at a time and keeps the
    name rule. One choice for both (CCA CODE-004 of run 20261106T212835Z)."""
    from recall.generation_store import GenerationStore

    return [
        target for target in (store, shadow_store)
        if target is not None and not isinstance(target, GenerationStore)
    ]


def _pruned_and_decided(
    cited_cards: list[Any], targets: list[Any], erased_chunks: set[str], asked: set[str], gone_sources: set[str],
    resolved: Mapping[str, list[str]],
) -> tuple[dict[str, ChunkRecord], dict[str, set[str]]]:
    """The cited chunks whose latest record is the indexer's prune, and of those the ones this call
    decides by naming their memo, grouped by its absolute source (`prepare_fact_erasure`).

    A call names a pruned memo by its absolute source, or by its root-relative name when that
    name names this memo and no other: no memo the store serves under it, no other pruned memo
    carrying it, and no memo a forget recorded under it (`forget_source_aliases`). A namesake
    under an unrelated root is never forgotten by a bare name, a forget run again after its memo
    is gone included (owner decision 2026-11-07); `resolved` is what each card name served before
    the deletion."""
    candidates = sorted({card.chunk_id for card in cited_cards} - erased_chunks)
    # One read: the stores of one tenant share its audit records, and the prune and the forget
    # write theirs to the primary store (CCA PERF-003 of run 20261106T212835Z).
    reader = next((target for target in targets if callable(getattr(target, "latest_chunk_records", None))), None)
    latest: dict[str, ChunkRecord] = reader.latest_chunk_records(candidates) if reader is not None and candidates else {}
    pruned = {chunk: record for chunk, record in latest.items() if record.kind == PRUNE_SOURCE_CHUNKS}
    # Every memo each card name stood for: what it serves now, the memos pruned under it, and
    # those a forget recorded under it, whose cards are redacted and so cite nothing.
    under: dict[str, set[str]] = {}
    for card in cited_cards:
        record = pruned.get(card.chunk_id)
        if record is not None:
            under.setdefault(card.source, set()).add(record.source)
    forgotten_as = getattr(reader, "sources_forgotten_as", None)
    if callable(forgotten_as) and under:
        for name, sources in forgotten_as(sorted(under)).items():
            under.setdefault(name, set()).update(sources)

    def names_it(card: Any, record: ChunkRecord) -> bool:
        if record.source in asked | gone_sources:
            return True
        return card.source in asked and set(resolved.get(card.source, ())) | under[card.source] == {record.source}

    decided: dict[str, set[str]] = {}
    for card in cited_cards:
        record = pruned.get(card.chunk_id)
        if record is not None and names_it(card, record):
            decided.setdefault(record.source, set()).add(card.chunk_id)
    return pruned, decided


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
    namesake is another memo. So is a card whose own chunk this deletion erases, on a lite store
    or the legacy table, though another root still serves a memo under its name; its value is then
    redacted as well when no other card supports it. A card whose chunk the indexer last recorded
    pruning (`prune_source_chunks`) survives unless this call names its memo: a memo deleted from
    disk is a state, and only that positive record tells it from a forgotten one. Naming a pruned
    memo records it as forgotten (`forget_source_chunks`) here, before anything is retracted, so
    preparing writes as well as reads. A generation store has no prune records and keeps the name
    rule.
    """
    from recall.types import RedactedEvidenceCard

    cited_cards = [
        card
        for event in ledger.events
        for card in event.evidence_cards
        if not isinstance(card, RedactedEvidenceCard)
    ]
    cited = {card.source for card in cited_cards}
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

    def record_aliases(source: str, aliases: list[str]) -> None:
        if aliases and callable(append):
            identity = json.dumps([store.tenant, source, aliases], ensure_ascii=False)
            append(
                "forget_source_aliases", {"aliases": aliases}, source_uri=source,
                event_id="forget_alias_" + hashlib.sha256(identity.encode()).hexdigest(),
            )

    for source in sorted(gone_sources):
        record_aliases(source, sorted(name for name, values in resolved.items() if source in values))

    # The chunks this deletion erases, so a card citing one is gone although a memo another root
    # holds under the same root-relative name is still served (owner decision 2026-11-06, CCA
    # SEC-001). Chunk ids hash the absolute source, so they never name another root's chunk. A
    # generation store reads chunks per generation only, so there it stays the name rule: a
    # retraction may fail to happen, never remove a fact that still has evidence.
    targets = chunk_targets(store, shadow_store)
    erased_chunks: set[str] = set()
    for source in sorted(gone_sources):
        ids = sorted({chunk.id for target in targets for chunk in target.chunks_for_source(source)})
        erased_chunks.update(ids)
        # Recorded before the rows vanish, like the aliases, so a forget re-run after a failed
        # retraction or redaction, which deletes nothing, still knows them (CCA BUG-001). Ids only.
        # One record per forget, never merged with an earlier one: the latest record of a chunk
        # decides whether it was forgotten or only pruned.
        if ids and callable(append):
            append(FORGET_SOURCE_CHUNKS, {"chunk_ids": ids}, source_uri=source)
    for target in targets:
        recover_chunks = getattr(target, "forgotten_source_chunks", None)
        if callable(recover_chunks):
            erased_chunks.update(recover_chunks(sorted(named)))
    # A memo pruned from disk is a state, not a decision (owner decision 2026-11-06, CCA STAKES-001
    # of run 20261105T225204Z): a card whose chunk's latest record is the indexer's prune survives,
    # so a fact it also states is hidden while the memo is away instead of retracted for good. No
    # record means gone, as before: a memo forgotten before these records existed stays forgotten.
    # A pruned memo this call names, by its source or by a root-relative name that stands for it
    # alone (`_pruned_and_decided`), is a decision; it is recorded as forgotten first, with the
    # names it was decided under, so a later forget reads it so even when this call's redaction
    # fails and a bare name stays ambiguous once it stood for it (CCA STAKES-001 of run
    # 20261107T124651Z, DOC-001). Only
    # what the call names counts, never `touched_names`: that also holds every name a memo this
    # call erases is served under, so forgetting /A/x.md would decide a pruned /B/x.md (CCA BUG-001
    # of run 20261106T212835Z).
    pruned, decided = _pruned_and_decided(cited_cards, targets, erased_chunks, set(named), gone_sources, resolved)
    for source, chunk_ids in sorted(decided.items()):
        record_aliases(source, sorted({card.source for card in cited_cards if card.chunk_id in chunk_ids}))
        if callable(append):
            append(FORGET_SOURCE_CHUNKS, {"chunk_ids": sorted(chunk_ids)}, source_uri=source)
        erased_chunks.update(chunk_ids)
        for chunk_id in chunk_ids:
            pruned.pop(chunk_id, None)
    chunk_cache: dict[str, bool] = {}

    def chunk_gone(chunk_id: str) -> bool:
        # Look again after deletion, as `remaining` does: a chunk indexed again is evidence again.
        if chunk_id not in erased_chunks:
            return False
        if chunk_id not in chunk_cache:
            chunk_cache[chunk_id] = all(target.chunk_by_id(chunk_id) is None for target in targets)
        return chunk_cache[chunk_id]

    def touched(card: EvidenceCard) -> bool:
        # A card whose own chunk this deletion erases is touched by it even when its name, stamped
        # under another root, no longer resolves (CCA SEC-001 of run 20261106T121445Z).
        return card.source in touched_names or chunk_gone(card.chunk_id)

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
        if isinstance(card, RedactedEvidenceCard) or chunk_gone(card.chunk_id):
            return True
        return gone_name(card.source) and card.chunk_id not in pruned

    def retract() -> int:
        append = ledger.withdraw_forgotten if withdraw else ledger.retract_forgotten
        written = append(tenant_id=store.tenant, touched=touched, gone=gone)
        return len({fact_id for event in written for fact_id in event.supersedes_fact_ids})

    def redact() -> int:
        if redactor is None:
            return 0
        # A name a kept (only pruned) card still carries is not erased with a card of another root.
        kept_names = {card.source for card in cited_cards if card.chunk_id in pruned}
        report = redactor.redact(
            erased=lambda name: name in touched_names and gone_name(name),
            names=sorted(touched_names),
            erased_chunk=chunk_gone,
            chunk_ids=sorted(erased_chunks),
            kept_chunk=lambda chunk_id: chunk_id in pruned,
            name_gone=lambda name: gone_name(name) and name not in kept_names,
        )
        return int(report.facts_redacted)

    return FactErasure(retract=retract, redact=redact)


def _settled(fn: Callable[_P, _V]) -> Callable[_P, _V]:
    """`fn` run once per argument tuple, its failure remembered like its answer.

    `functools.cache` keeps no exception, so a lookup that timed out would be run, and time out,
    again for every check that reads it; here the second caller gets the same error at once.
    """
    answers: dict[tuple[object, ...], _V] = {}
    errors: dict[tuple[object, ...], Exception] = {}

    def call(*args: _P.args, **kwargs: _P.kwargs) -> _V:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in answers and key not in errors:
            try:
                answers[key] = fn(*args, **kwargs)
            except Exception as exc:  # BROAD-CATCH: error-translation  # kept and raised to every caller
                errors[key] = exc
        if key in errors:
            raise errors[key]
        return answers[key]

    return call


def served_sources(
    store: object, *, failures: list[str] | None = None, strict: bool = False,
) -> ServedSources | None:
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

    `strict` is for authorization (`recall_apply_fact`): a failed admission or content lookup
    RAISES rather than hiding stated facts, because a hidden fact cannot conflict with anything,
    so the write would be admitted against a value it contradicts. The controller's conflict read
    then refuses the write as `LEDGER_UNAVAILABLE`. Presence and supersession still fail open,
    which on a write only keeps more facts in force.
    """
    lookup = getattr(store, "served_identifiers", None)
    if not callable(lookup):
        return None

    def failed(what: str) -> None:
        if failures is not None and what not in failures:
            failures.append(what)

    def present(names: frozenset[str], look: Callable[[frozenset[str]], Collection[str]]) -> Collection[str]:
        try:
            return look(names)
        except Exception:  # BROAD-CATCH: fail-open  # the ledger's own answer, without hiding
            _log.exception("serving lookup for %d memo name(s) failed; hiding no fact", len(names))
            failed("presence")
            return names

    from recall.generation_store import GenerationStore
    from recall.lite import LiteStore
    from recall.store import PgVectorStore

    if not isinstance(store, (LiteStore, PgVectorStore)):

        @cache
        def plain(names: frozenset[str]) -> Collection[str]:
            return present(names, lambda wanted: {str(name) for name in lookup(sorted(wanted))})

        return plain
    from recall.provenance_controller import ServedState
    from recall.stated_facts import StatingRule, stating_rule

    # The served generation only on a generation store; a lite file or the legacy table has one
    # set of rows, so its erasure resolver already answers for what it serves, and its
    # `served_identifiers` is that resolver's keys.
    resolve = store.served_sources_for_identifiers
    pin = store.pin_generation if isinstance(store, GenerationStore) else None

    @_settled
    def generation() -> str:
        return str(store.generation_id)

    def pinned() -> AbstractContextManager[object]:
        return pin(generation()) if pin is not None else nullcontext()

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

    def admitting_rule() -> StatingRule | None:
        """The operator's stating rule, or None while stating is off (the rule is then a reason)."""
        admits = rule()
        return None if isinstance(admits, str) else admits

    @cache
    def served(names: frozenset[str]) -> Collection[str]:
        return present(names, lambda wanted: set(resolved_names(wanted)))

    @cache
    def content_hashes(names: frozenset[str]) -> dict[str, frozenset[str]]:
        try:
            admits = admitting_rule()
            if admits is None:
                return {}
            resolved = resolved_names(names)
            raw = raw_hashes()
        except Exception:  # BROAD-CATCH: fail-closed  # no stated fact is current on an error
            # Not None, which `card_liveness` reads as "no content check": that brought an edited
            # memo's old value back as current whenever this lookup timed out while admission,
            # computed from the same cached names, still passed.
            _log.exception("content lookup for %d memo name(s) failed; listing no stated fact", len(names))
            failed("content")
            if strict:
                raise
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
            admits = admitting_rule()
            if admits is None or not names:
                return frozenset()
            resolved = resolved_names(names)
        except Exception:  # BROAD-CATCH: fail-closed  # no stated fact is current on an error
            _log.exception("stating rule lookup failed; listing no stated fact")
            failed("admission")
            if strict:
                raise
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
        retired when any of them is: looked up by name, it never was (audit SEC-005). A claimant
        forgotten or pruned still retires through its standing record, admitted by the source the
        record names, as the document layer keeps it retiring; a withdrawn one never claims.
        """
        try:
            # Only admitted memos can retire facts, so with stating off nothing can be retired, and
            # the store's supersession graph is not asked at all: that lookup would otherwise run on
            # every read of every PostgreSQL deployment that never turned stating on.
            admits = admitting_rule()
            if admits is None:
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
            wanted = frozenset(claimant for group in claims.values() for claimant in group)
            claimants = resolved_names(wanted)
            # A claimant the store no longer serves is a memo forgotten or pruned whose record
            # still stands in the graph; forget keeps retired what it retired (owner decision
            # 2026-10-22), so it is admitted by its record's source (CCA BUG-001).
            unserved = sorted(name for name in wanted if not claimants.get(name))
            if unserved:
                with pinned():
                    recorded = store.recorded_sources_for_identifiers(unserved)
                claimants = {**claimants, **{name: list(values) for name, values in recorded.items() if values}}
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
