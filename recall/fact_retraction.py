"""The facts whose memos are gone: retracted by `recall_forget`, hidden while a memo is absent.

A fact applied with `recall_apply_fact` cites evidence cards, and each card names the memo it came
from. Two things take a memo away, and they are treated differently on purpose:

- `recall_forget` is a DECISION. The memo is erased from every generation and nothing brings it
  back, so the facts it was the only evidence for get a permanent `retracted` event
  (`prepare_retraction`, rule in `recall.fact_ledger.plan_retractions`).
- A memo deleted from disk, pruned by the indexer, or left out of a promoted generation is a
  STATE. A rollback, a restored file or a repaired sync brings it back, so nothing is written: the
  projections hide a fact none of whose memos the store serves now (`served_sources`, rule in
  `recall.provenance_controller.absent_fact_ids`), and show it again when a memo returns. A lite
  or PostgreSQL store also answers with which content each memo is served and which memos its supersession
  graph retires (`recall.provenance_controller.ServedState`).
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any

from recall.observability import get_logger

if TYPE_CHECKING:
    from recall.provenance_controller import ServedSources
    from recall.types import EvidenceCard

_log = get_logger("fact_retraction")

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
) -> Callable[[], int]:
    """Prepare, before any chunk is deleted, the retraction a deletion owes the fact ledger.

    Returns a callable that appends it and answers how many facts it retracted. A card names its
    memo by the root-relative file a search hit shows, while deletion keys on the absolute
    source, so every card's name is resolved here, while its rows still exist. `erased` is the
    absolute sources about to be deleted; `named` is every name the caller used for them (a
    forget's requested identifiers, or the erased sources themselves).

    A card is GONE when its name resolves to nothing outside `erased` (nothing at all when an
    earlier deletion took it), and TOUCHED when `named` holds its name or this deletion erases its
    memo, so a forget re-run after a failed retraction, which deletes nothing, completes it. A
    card whose name was not known here (applied while this ran) counts as surviving: a retraction
    may fail to happen, never remove a fact that still has evidence.
    """
    names = sorted({
        card.source
        for event in ledger.events
        if event.event_type == "asserted"
        for card in event.evidence_cards
    })
    resolved: dict[str, list[str]] = dict(store.sources_for_identifiers(names)) if names else {}
    if shadow_store is not None and names:
        for name, values in shadow_store.sources_for_identifiers(names).items():
            bucket = resolved.setdefault(name, [])
            bucket.extend(value for value in values if value not in bucket)
    known = set(names)
    gone_sources = set(erased)
    touched_names = set(named) | gone_sources

    def gone(card: EvidenceCard) -> bool:
        return card.source in known and set(resolved.get(card.source, ())) <= gone_sources

    def touched(card: EvidenceCard) -> bool:
        return card.source in touched_names or (bool(resolved.get(card.source)) and gone(card))

    def retract() -> int:
        written = ledger.retract_forgotten(tenant_id=store.tenant, touched=touched, gone=gone)
        return len({fact_id for event in written for fact_id in event.supersedes_fact_ids})

    return retract


def served_sources(store: object) -> ServedSources | None:
    """The memo names `store` serves now, with their content, supersession and stating rule on a
    lite or PostgreSQL store; None without a lookup. On a generation store the content is the
    SERVED generation's: `sources_for_identifiers` spans every generation, `source_raw_hashes` the
    served one, so a stated card lives only while that generation holds the content it recorded.

    Fails open: when a lookup errors every name counts as served, unedited and unsuperseded, so an
    outage shows facts as the ledger alone would rather than hiding all of them.
    """
    lookup = getattr(store, "served_identifiers", None)
    if not callable(lookup):
        return None

    def served(names: frozenset[str]) -> Collection[str]:
        try:
            return {str(name) for name in lookup(sorted(names))}
        except Exception:  # BROAD-CATCH: fail-open  # the ledger's own answer, without hiding
            _log.exception("serving lookup for %d memo name(s) failed; hiding no fact", len(names))
            return names

    from recall.lite import LiteStore
    from recall.store import PgVectorStore

    if not isinstance(store, (LiteStore, PgVectorStore)):
        return served
    from recall.provenance_controller import ServedState

    def content_hashes(names: frozenset[str]) -> dict[str, frozenset[str]] | None:
        try:
            resolved = store.sources_for_identifiers(sorted(names))
            raw = store.source_raw_hashes()
        except Exception:  # BROAD-CATCH: fail-open  # no stated card is checked against content
            _log.exception("content lookup for %d memo name(s) failed; checking no stated card", len(names))
            return None
        return {
            name: frozenset(raw[source] for source in sources if raw.get(source))
            for name, sources in resolved.items()
        }

    def stated_admitted(names: frozenset[str]) -> frozenset[str]:
        """The names whose memos the operator's rule lets state facts now. Fails CLOSED: a rule
        that cannot be computed admits nothing, so no memo states facts on an error."""
        from recall.stated_facts import stating_rule

        try:
            rule = stating_rule(store)
            if isinstance(rule, str) or not names:
                return frozenset()
            resolved = store.sources_for_identifiers(sorted(names))
        except Exception:  # BROAD-CATCH: fail-closed  # no stated fact is current on an error
            _log.exception("stating rule lookup failed; listing no stated fact")
            return frozenset()
        return frozenset(
            name for name, sources in resolved.items() if any(rule.admits(name, s) for s in sources)
        )

    def superseded(names: frozenset[str]) -> frozenset[str]:
        """Names a served memo the operator admits supersedes. A memo that may not state facts
        (an upload, a file outside the prefixes) cannot retire another memo's facts either, and an
        ambiguous reference retires nothing; search's own verdicts are not changed by either.

        Every claimant of a target counts, not the edge map's last-row winner: otherwise an upload
        sorting after an admitted claimant would displace it and keep the old value current.
        """
        try:
            edges, _unresolved, candidates = store.supersession_all()
            claims = {
                name: {claimant for claimant, _when in candidates.get(name, ())} | {edges[name]}
                for name in names
                if name in edges
            }
            admitted = stated_admitted(frozenset(c for group in claims.values() for c in group))
        except Exception:  # BROAD-CATCH: fail-open  # no memo's facts are retired
            _log.exception("supersession lookup failed; retiring no memo's facts")
            return frozenset()
        return frozenset(name for name, group in claims.items() if group & admitted)

    return ServedState(
        served, content_hashes=content_hashes, superseded=superseded, stated_admitted=stated_admitted
    )


__all__ = ["FACT_WRITE_DSN_ENV", "fact_ledger_for", "prepare_retraction", "served_sources"]
