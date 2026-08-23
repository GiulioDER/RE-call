"""`recall reasoning`: opt-in reasoning projection, proposals, query, trace and audit tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from recall.store import PgVectorStore

from recall.cli_commands._shared import _cli_trust, _make_embedder
from recall._env import env_is_production

if TYPE_CHECKING:
    from recall.reasoning import ReasoningResponse
    from recall.trust_policy import TrustPolicy


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_reasoning = sub.add_parser(
        "reasoning",
        help="explicit opt-in reasoning projection, proposals, query, trace and audit tools",
    )
    p_reasoning.set_defaults(_opens_db=True, func=_cmd_reasoning)
    reasoning_sub = p_reasoning.add_subparsers(dest="reasoning_cmd", required=True)
    p_reasoning_projection = reasoning_sub.add_parser(
        "projection", help="build and inspect the derived reasoning projection"
    )
    p_reasoning_projection.add_argument(
        "--include-text",
        action="store_true",
        help="include evidence text in the projection summary input. Defaults off for privacy.",
    )
    p_reasoning_proposals = reasoning_sub.add_parser(
        "proposals", help="inspect deterministic inference proposals"
    )
    p_reasoning_proposals.add_argument(
        "--include-extracted",
        action="store_true",
        help="also list proposals replayed from prose extraction recorded at ingest. Refuses "
        "if nothing was recorded: extraction never runs on the query path.",
    )
    p_reasoning_query = reasoning_sub.add_parser("query", help="run a bounded reasoning query")
    p_reasoning_query.add_argument("query")
    p_reasoning_query.add_argument("-k", type=int, default=5)
    p_reasoning_query.add_argument("--source")
    p_reasoning_query.add_argument(
        "--mode",
        choices=["evidence_assembly", "proposal_assisted", "review_required", "retrieval_only"],
        default="proposal_assisted",
    )
    p_reasoning_query.add_argument("--max-steps", type=int, default=12)
    p_reasoning_query.add_argument("--max-graph-nodes", type=int, default=32)
    p_reasoning_query.add_argument("--max-evidence-tokens", type=int, default=2048)
    p_reasoning_trace = reasoning_sub.add_parser(
        "trace", help="run a bounded query and export only the reasoning trace"
    )
    p_reasoning_trace.add_argument("query")
    p_reasoning_trace.add_argument("--output", required=True)
    p_reasoning_trace.add_argument("-k", type=int, default=5)
    p_reasoning_trace.add_argument("--source")
    p_reasoning_trace.add_argument("--max-steps", type=int, default=12)
    p_reasoning_trace.add_argument("--max-graph-nodes", type=int, default=32)
    p_reasoning_trace.add_argument("--max-evidence-tokens", type=int, default=2048)
    p_reasoning_audit = reasoning_sub.add_parser(
        "audit", help="run the reasoning integration audit"
    )
    p_reasoning_audit.add_argument("--query", default="reasoning audit sentinel")


def _refuse_untrusted_reasoning_inspection(trust_state: str, policy: "TrustPolicy") -> None:
    if trust_state != "trusted" and policy.strict:
        raise SystemExit(
            "reasoning inspection refused in strict mode: generation identity or calibration is "
            "missing. Set RECALL_TRUST_MODE=development to inspect degraded artifacts."
        )


def _reasoning_trace_export(response: "ReasoningResponse") -> dict[str, object]:
    trace = response.to_dict()["reasoning_trace"]
    if trace is None:
        reason = (
            response.refusal_reason or response.trusted_evidence.failure_code or response.outcome
        )
        raise SystemExit(f"reasoning trace unavailable: {reason}")
    assert isinstance(trace, dict)
    initial = trace.get("initial_retrieval")
    if isinstance(initial, dict):
        initial.pop("reason", None)
    return trace


def _cmd_reasoning(args: argparse.Namespace) -> None:
    # Legacy process-global calibration is deliberately never auto-loaded here. See the longer
    # note in `_cmd_search` (recall/cli_commands/index_search.py), kept beside the search path
    # where the design question originated.
    calibration = None

    embedder = _make_embedder(args.embedder)
    from recall.generation_store import GenerationStore
    from recall_mcp.service import (
        reasoning_audit,
        reasoning_projection,
        reasoning_proposals,
        reasoning_query,
    )

    if env_is_production():
        reasoning_store_context: PgVectorStore = GenerationStore(
            args.dsn, embedder.dim, tenant=args.tenant
        )
    else:
        reasoning_store_context = PgVectorStore(
            args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
        )
    with reasoning_store_context as store:
        store.check_schema()
        _reasoning_policy, _reasoning_calibration = _cli_trust(embedder, calibration)
        if args.reasoning_cmd == "projection":
            projection = reasoning_projection(store, include_text=args.include_text)
            _refuse_untrusted_reasoning_inspection(projection.trust_state, _reasoning_policy)
            print(projection.model_dump_json(indent=2))
            return
        if args.reasoning_cmd == "proposals":
            try:
                proposal_result = reasoning_proposals(
                    store, include_extracted=args.include_extracted
                )
            except ValueError as exc:
                # `--include-extracted` refuses when nothing was recorded at ingest. Left
                # raw it was the flag's only reachable outcome AND a traceback, where every
                # neighbouring refusal in this CLI prints one line and exits 2.
                print(f"recall reasoning: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
            trust_state = (
                "trusted" if proposal_result.generation_id != "legacy" else "degraded"
            )
            _refuse_untrusted_reasoning_inspection(trust_state, _reasoning_policy)
            print(proposal_result.model_dump_json(indent=2))
            return

        if args.reasoning_cmd in {"query", "trace"}:
            response = reasoning_query(
                store,
                embedder,
                args.query,
                source=args.source,
                k=args.k,
                mode=getattr(args, "mode", "proposal_assisted"),
                max_steps=args.max_steps,
                max_graph_nodes=args.max_graph_nodes,
                max_evidence_tokens=args.max_evidence_tokens,
                policy=_reasoning_policy,
                calibration=_reasoning_calibration,
            )
            if args.reasoning_cmd == "trace":
                payload = _reasoning_trace_export(response)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, default=str),
                    encoding="utf-8",
                )
                print(f"trace: {args.output}")
                return
            print(json.dumps(response.to_dict(), indent=2, default=str))
            return
        if args.reasoning_cmd == "audit":
            print(
                reasoning_audit(
                    store,
                    embedder,
                    query=args.query,
                    policy=_reasoning_policy,
                    calibration=_reasoning_calibration,
                ).model_dump_json(indent=2)
            )
            return
        raise SystemExit(f"unknown reasoning subcommand: {args.reasoning_cmd}")
