"""The ``recall provenance`` command group."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.provenance_controller import (
    FactApplicationRequest,
    ProvenanceController,
    evidence_card_from_payload,
    source_digest,
)
from recall.types import AtomicFact, EvidenceCard


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "provenance",
        help="apply structured facts through the deterministic provenance controller",
        description=(
            "Apply structured facts through server-created evidence cards, or inspect the current "
            "tenant-scoped deterministic provenance projection."
        ),
    )
    parser.set_defaults(_opens_db=True, func=_cmd_provenance)
    commands = parser.add_subparsers(dest="provenance_cmd", required=True)

    apply = commands.add_parser("apply", help="apply a claim using server-created evidence cards")
    apply.add_argument("--claim", required=True, help="JSON file containing one atomic fact")
    apply.add_argument("--cards", required=True, help="JSON file containing evidence cards")
    apply.add_argument("--request-id", required=True)
    apply.add_argument("--generation", required=True)
    apply.add_argument("--writer", default="cli")
    apply.add_argument("--sqlite-path", default=None, metavar="PATH")
    apply.add_argument("--source-root", default=None, metavar="PATH")

    current = commands.add_parser("current", help="read the current fact projection")
    current.add_argument("--as-of", default=None)
    current.add_argument("--sqlite-path", default=None, metavar="PATH")


def _read_cards(path: str) -> tuple[EvidenceCard, ...]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cards", [])
    if not isinstance(payload, list):
        raise SystemExit("--cards must contain a JSON array or an evidence result object")
    try:
        return tuple(evidence_card_from_payload(item) for item in payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid evidence card payload: {exc}") from exc


def _decision_payload(decision: Any) -> dict[str, object]:
    return {
        "allowed": decision.allowed,
        "decision_code": str(decision.code),
        "request_id": decision.request_id,
        "fact_id": decision.fact_id,
        "retried": decision.retried,
        "detail": decision.detail,
        "event_id": decision.event.event_id if decision.event else None,
        "evidence_card_ids": [card.card_id for card in decision.cards],
    }


def _cmd_provenance(args: argparse.Namespace) -> None:
    from recall.fact_ledger import PostgresFactLedger, SQLiteFactLedger

    if args.provenance_cmd == "current":
        instant = (
            datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
            if args.as_of
            else datetime.now(UTC)
        )
        if args.sqlite_path:
            with SQLiteFactLedger(args.sqlite_path, tenant_id=args.tenant) as ledger:
                events = ledger.current(tenant_id=args.tenant, now=instant)
        else:
            events = PostgresFactLedger(args.dsn, tenant_id=args.tenant).current(
                tenant_id=args.tenant, now=instant
            )
        print(json.dumps({
            "tenant_id": args.tenant,
            "as_of": instant.isoformat(),
            "facts": [
                {
                    "event_id": event.event_id,
                    "fact_id": event.fact_id,
                    "fact": event.fact.to_payload() if event.fact else None,
                    "evidence_card_ids": [card.card_id for card in event.evidence_cards],
                    "generation_id": event.generation_id,
                    "writer": event.writer,
                    "asserted_at": event.created_at.isoformat(),
                }
                for event in events
            ],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.provenance_cmd != "apply":
        raise SystemExit(f"unknown provenance command {args.provenance_cmd!r}")
    claim_payload = json.loads(Path(args.claim).read_text(encoding="utf-8"))
    cards = _read_cards(args.cards)
    claim = AtomicFact.from_payload(claim_payload)
    request = FactApplicationRequest(claim, tuple(card.card_id for card in cards), args.request_id)

    if args.sqlite_path:
        if not args.source_root:
            raise SystemExit("--source-root is required with --sqlite-path")
        root = Path(args.source_root).resolve()

        def current_digest(card: EvidenceCard) -> str | None:
            path = (root / card.source).resolve()
            try:
                path.relative_to(root)
                return source_digest(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                return None

        from recall.provenance_cards import SQLiteEvidenceCardStore

        with (
            SQLiteEvidenceCardStore(args.sqlite_path, tenant_id=args.tenant) as card_store,
            SQLiteFactLedger(args.sqlite_path, tenant_id=args.tenant) as ledger,
        ):
            card_store.put(cards)
            decision = ProvenanceController(
                tenant_id=args.tenant,
                generation_id=args.generation,
                cards=card_store,
                ledger=ledger,
                source_digest_for=current_digest,
                writer=args.writer,
            ).apply_fact(request)
        print(json.dumps(_decision_payload(decision), ensure_ascii=False, indent=2, sort_keys=True))
        return

    from recall_mcp.service import apply_fact_memory
    from recall.cli_commands._shared import _make_embedder
    from recall.store import PgVectorStore

    embedder = _make_embedder(args.embedder)
    with PgVectorStore(
        args.dsn,
        dim=embedder.dim,
        table=args.table,
        tenant=args.tenant,
        generation_id=args.generation,
    ) as store:
        print(json.dumps(apply_fact_memory(
            store,
            embedder,
            claim=claim_payload,
            evidence_card_ids=[card.card_id for card in cards],
            request_id=args.request_id,
            writer=args.writer,
        ), ensure_ascii=False, indent=2, sort_keys=True))
