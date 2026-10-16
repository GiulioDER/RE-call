"""Facts a memo states in its own frontmatter reach the fact ledger when the store indexes it.

A memo may carry ``recall_graph: {"facts": [{"subject": ..., "predicate": ..., "object": ...}]}``.
Until this module the only way such a fact became current was `recall_apply_fact`, which needs
evidence cards from a trusted search that carry exactly that fact: a chain agents were measured
not to complete. Here the server records the fact itself, after the write that indexed the memo,
with a card built from the memo's own chunk (`stated_card`).

Recording is an observation ("memo M, with content hash h, stated F"), appended once per
(memo, content, fact) (`stated_request_id`). Whether the fact is CURRENT is decided at read time
from what the store serves (`recall.provenance_controller.card_liveness`): an edit, a supersession,
an absence and a disagreement are states that reverse with the store's content; only forget and
gc retract.

Who may state facts is the operator's decision, by location, never the memo's (`StatingRule`):
the memo is written by whoever's trust is in question. Lite stores only, so far; PostgreSQL
follows in a later change. Design: recall-lab
`research/designs/2026-10-15-stated-facts-reach-the-ledger.md`.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from recall.evidence import _evidence_links
from recall.frontmatter import GRAPH_KEY, validity_bounds
from recall.observability import get_logger
from recall.provenance_controller import (
    STATED_POLICY_VERSION,
    STATED_TRUST_STATE,
    STATED_WRITER,
    DecisionCode,
    ProvenanceController,
    fact_identity,
)
from recall.types import AtomicFact, Chunk, EvidenceCard

__all__ = [
    "STATED_FACT_SOURCES_ENV",
    "StatedFactsReport",
    "StatingRule",
    "memo_facts",
    "reconcile_stated_facts",
    "stated_card",
    "stated_marker",
    "stated_request_id",
    "stating_rule",
]

_log = get_logger("stated_facts")

#: Root-relative prefixes whose memos may state facts, comma or space separated; `none` turns
#: stating off. Unset, a lite store states from the folders its refresh follows.
STATED_FACT_SOURCES_ENV = "RECALL_STATED_FACT_SOURCES"


def _under(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _runs_through_uploads(parts: Sequence[str], tenant: str) -> bool:
    """Whether a path passes through `uploads/<tenant>/`, where every `recall_ingest` upload is
    staged below an index root (`recall.uploads._tenant_uploads_root`).

    Compared without case: on a case-insensitive filesystem staging writes into an `Uploads/`
    that already exists, and the indexed path carries that case. It only ever fails closed.
    """
    folded = tenant.casefold()
    return any(
        parts[i].casefold() == "uploads" and parts[i + 1].casefold() == folded
        for i in range(len(parts) - 1)
    )


@dataclass(frozen=True)
class StatingRule:
    """Which memos may state facts: the operator's prefixes, else the folders a refresh follows.

    `prefixes` match a memo's stored name (`metadata.file`, root-relative) on a path boundary, so
    `memos` admits `memos/x.md` and not `memos2/x.md`. `roots` match its absolute source. An upload
    staged by `recall_ingest` never states, whatever either says: its text came from outside.

    An upload is recognised by where it is staged, structurally: a memo whose stored name or
    absolute source runs through `uploads/<tenant>/` is one. The test needs no environment on
    purpose. The first version derived the uploads folder from RECALL_INDEX_ROOT, and a refresh run
    without the server's environment, or a caller passing an env of its own, did not recognise it,
    so an upload's facts became current (security review, 2026-10-15). A memo folder a person
    happens to keep at `uploads/<tenant>/` is treated as uploads too: it fails closed.
    """

    prefixes: tuple[str, ...] = ()
    roots: tuple[Path, ...] = ()
    tenant: str | None = None

    def admits(self, name: str, source: str) -> bool:
        path = Path(source)
        if self.tenant is not None and (
            _runs_through_uploads(Path(name).parts, self.tenant)
            or _runs_through_uploads(path.parts, self.tenant)
        ):
            return False
        if self.prefixes:
            return any(name == prefix or name.startswith(prefix + "/") for prefix in self.prefixes)
        return any(_under(path, root) for root in self.roots)


def stating_rule(
    store: object,
    *,
    roots: Sequence[Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> StatingRule | str:
    """The rule for `store`, or the reason none of its memos states facts."""
    from recall.lite import LiteStore

    if not isinstance(store, LiteStore):
        return "stated facts are recorded on a lite store only, so far"
    values = os.environ if env is None else env
    raw = values.get(STATED_FACT_SOURCES_ENV, "").strip()
    if raw.lower() == "none":
        return f"{STATED_FACT_SOURCES_ENV}=none"
    if raw:
        prefixes = tuple(dict.fromkeys(
            part.replace("\\", "/").strip("/")
            for part in raw.replace(",", " ").split()
            if part.replace("\\", "/").strip("/")
        ))
        if prefixes:
            return StatingRule(prefixes=prefixes, tenant=store.tenant)
    if roots is None:
        from recall.lite.refresh import read_state

        roots = [Path(str(root)) for root in read_state(store.path).get("roots") or ()]
    if not roots:
        return f"no memo folder is refreshed and {STATED_FACT_SOURCES_ENV} is unset"
    return StatingRule(roots=tuple(Path(root).resolve() for root in roots), tenant=store.tenant)


def memo_facts(chunk: Chunk) -> tuple[tuple[AtomicFact, ...], int]:
    """The facts `chunk`'s memo states, each given its memo's window where it names none, and
    how many could not be read.

    A fact that names no `valid_from` or `valid_until` takes its memo's, so the ledger agrees with
    search's `expired` and `not_yet_valid` verdicts. A malformed `recall_graph` line counts once;
    each fact that does not parse, or whose value canonical JSON refuses (NaN), counts once.
    """
    metadata = chunk.metadata or {}
    graph = metadata.get(GRAPH_KEY)
    if not isinstance(graph, Mapping) or "__parse_error__" in graph:
        return (), 1
    raw = graph.get("facts", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return (), 1
    try:
        start, end = validity_bounds(metadata)
    except ValueError:
        return (), max(len(raw), 1)
    facts: dict[str, AtomicFact] = {}
    malformed = 0
    for item in raw:
        if not isinstance(item, Mapping):
            malformed += 1
            continue
        try:
            fact = AtomicFact.from_payload(item)
            fact = replace(fact, valid_from=fact.valid_from or start, valid_until=fact.valid_until or end)
            identity = fact_identity(fact)  # canonical JSON refuses NaN and unsupported values
        except (TypeError, ValueError, KeyError):
            malformed += 1
            continue
        facts.setdefault(identity, fact)
    return tuple(facts.values()), malformed


def stated_card(
    tenant: str, generation_id: str, chunk: Chunk, name: str, facts: tuple[AtomicFact, ...]
) -> EvidenceCard | None:
    """The card a memo's statement cites: its first chunk, its content hash, its own links.

    None when the chunk carries no `content_hash`, since liveness keys on it. Raises ValueError
    for a memo whose window is inverted (the card refuses it).
    """
    metadata = chunk.metadata or {}
    digest = metadata.get("content_hash")
    if not isinstance(digest, str) or not digest:
        return None
    raw_graph = metadata.get(GRAPH_KEY)
    graph: Mapping[str, object] = raw_graph if isinstance(raw_graph, Mapping) else {}
    start, end = validity_bounds(metadata)
    return EvidenceCard(
        card_id="",
        chunk_id=chunk.id,
        source=name,
        source_digest=digest,
        valid_from=start,
        valid_until=end,
        first_indexed_at=None,
        indexed_at=None,
        tenant_id=tenant,
        generation_id=generation_id,
        pipeline_fingerprint=None,
        corpus_fingerprint=None,
        calibration_id=None,
        calibration_status="not_applicable",
        trust_state=STATED_TRUST_STATE,
        verdict="unverified",
        confidence=1.0,
        rank=1,
        supersession_links=_evidence_links(graph, metadata, "authored_supersedes"),
        contradiction_links=_evidence_links(graph, metadata, "authored_contradicts"),
        support_refs=_evidence_links(graph, metadata, "support_refs"),
        structured_facts=facts,
    )


def stated_marker(
    store: object,
    *,
    roots: Sequence[Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """What a refresh records about the rule its reconcile ran under.

    The lite refresh skips an unchanged folder without running anything, so without this a store
    upgraded with its memos in place, or a rule widened, would record nothing until some memo
    changed. A refresh that finds a different marker treats the folder as changed.
    """
    rule = stating_rule(store, roots=roots, env=env)
    digest = hashlib.sha256(repr(rule).encode("utf-8")).hexdigest()[:16]
    return f"{STATED_POLICY_VERSION}:{digest}"


def stated_request_id(tenant: str, name: str, content_hash: str, fact: AtomicFact) -> str:
    """One request per (memo, content, fact), ever: the generation is left out on purpose, so a
    re-run, an unchanged re-index and a new generation of the same content append nothing."""
    key = f"{tenant}\0{name}\0{content_hash}\0{fact_identity(fact)}"
    return "stated:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


@dataclass
class StatedFactsReport:
    """One reconcile: whether it ran, and what it appended. `facts_stated` -1 is a failed append."""

    enabled: bool
    reason: str
    memos_read: int = 0
    facts_stated: int = 0
    facts_duplicate: int = 0
    facts_malformed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if not self.enabled:
            return f"off: {self.reason}"
        if self.facts_stated < 0:
            return f"FAILED: {self.reason}"
        return (
            f"{self.facts_stated} new, {self.facts_duplicate} repeated, "
            f"{self.facts_malformed} malformed, from {self.memos_read} memo(s)"
        )

    def render(self) -> str:
        return f"stated facts {self.summary()}"


def _admitted_memos(store: Any, rule: StatingRule) -> list[tuple[str, Chunk]]:
    """`(name, first chunk)` of every served memo with a `recall_graph` the rule admits, by name."""
    chosen: dict[str, tuple[int, str, Chunk]] = {}
    for chunk in store.iter_chunks():
        metadata = chunk.metadata or {}
        graph = metadata.get(GRAPH_KEY)
        if graph is None or (
            isinstance(graph, Mapping) and "facts" not in graph and "__parse_error__" not in graph
        ):
            continue
        file = metadata.get("file")
        name = file if isinstance(file, str) and file else chunk.source
        if not rule.admits(name, chunk.source):
            continue
        ordinal = metadata.get("ord")
        position = ordinal if isinstance(ordinal, int) else 0
        held = chosen.get(chunk.source)
        if held is None or position < held[0]:
            chosen[chunk.source] = (position, name, chunk)
    return sorted(
        ((name, chunk) for _position, name, chunk in chosen.values()),
        key=lambda item: (item[0], item[1].source),
    )


def reconcile_stated_facts(
    store: Any,
    *,
    roots: Sequence[Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> StatedFactsReport:
    """Record, once, every fact an admitted memo of `store` states. Never raises: the write that
    called it has already succeeded, and a failure is reported in the result instead."""
    try:
        rule = stating_rule(store, roots=roots, env=env)
        if isinstance(rule, str):
            return StatedFactsReport(False, rule)
        return _reconcile(store, rule)
    except Exception as exc:  # BROAD-CATCH: fail-open  # the index that called this succeeded
        _log.warning(
            "recording stated facts for tenant %r failed: %s", getattr(store, "tenant", "?"), exc
        )
        return StatedFactsReport(True, f"{type(exc).__name__}: {exc}", facts_stated=-1)


def _reconcile(store: Any, rule: StatingRule) -> StatedFactsReport:
    from recall.fact_retraction import served_sources
    from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger

    report = StatedFactsReport(True, "")
    memos = _admitted_memos(store, rule)
    report.memos_read = len(memos)
    generation_id = store.generation_id
    cards: list[EvidenceCard] = []
    pending: list[tuple[AtomicFact, EvidenceCard, str]] = []
    for name, chunk in memos:
        # One memo's failure is that memo's, never the store's: an error nobody anticipated (a
        # RecursionError from JSON nested deeper than a recursive walk survives) counts it
        # malformed and the rest are still recorded.
        try:
            facts, malformed = memo_facts(chunk)
            report.facts_malformed += malformed
            if not facts:
                continue
            try:
                card = stated_card(store.tenant, generation_id, chunk, name, facts)
            except ValueError:
                card = None
            if card is None:
                report.facts_malformed += len(facts)
                continue
            requests = [
                (fact, card, stated_request_id(store.tenant, name, card.source_digest, fact))
                for fact in facts
            ]
        except Exception as exc:  # BROAD-CATCH: fail-open  # this memo is skipped and counted
            _log.warning("skipping the facts memo %r states: %s: %s", name, type(exc).__name__, exc)
            report.facts_malformed += 1
            continue
        cards.append(card)
        pending.extend(requests)
    ledger = LiteFactLedger(store)
    known = {event.request_id for event in ledger.events}
    todo = [item for item in pending if item[2] not in known]
    if not todo:
        report.reason = "every stated fact is already recorded"
        return report
    card_store = LiteEvidenceCardStore(store)
    card_store.put(cards)
    controller = ProvenanceController(
        tenant_id=store.tenant,
        generation_id=generation_id,
        cards=card_store,
        ledger=ledger,
        writer=STATED_WRITER,
        served_sources=served_sources(store),
    )
    for fact, card, request_id in todo:
        decision = controller.apply_stated(fact, card, request_id)
        if decision.code == DecisionCode.APPLIED:
            report.facts_stated += 1
        elif decision.code == DecisionCode.DUPLICATE:
            report.facts_duplicate += 1
        else:
            report.facts_stated = -1
            report.reason = f"{decision.code} {decision.detail}".strip()
            _log.warning(
                "recording stated facts for tenant %r failed: %s; nothing more was recorded",
                store.tenant, report.reason,
            )
            return report
    report.reason = f"recorded {len(todo)} statement(s)"
    return report
