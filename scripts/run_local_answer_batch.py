"""Run the local answer provider over serialized graph precision responses."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall.answer_provider import resolve_answer_provider
from recall.evidence import parse_answer_envelope, render_evidence_prompt, validate_answer
from recall.reasoning import reasoning_response_from_dict


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_local_answer_batch.py INPUT_JSON_OR_B64 OUTPUT_JSON")
    raw_input = Path(sys.argv[1]).read_bytes()
    try:
        payloads = json.loads(raw_input.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payloads = json.loads(base64.b64decode(raw_input).decode("utf-8"))
    provider = resolve_answer_provider(
        {
            "RECALL_REASONING_ANSWER_ENABLED": "1",
            "RECALL_REASONING_ANSWER_MODEL": "qwen3:4b",
            "RECALL_REASONING_ANSWER_BASE_URL": "http://127.0.0.1:11434/v1",
            "RECALL_REASONING_ANSWER_TIMEOUT": "120",
            "RECALL_REASONING_ANSWER_MAX_TOKENS": "512",
        }
    )
    assert provider is not None
    rows: list[dict[str, object]] = []
    for index, item in enumerate(payloads, start=1):
        print(
            f"answer {index}/{len(payloads)} {item['variant']}/{item['relation_control']} "
            f"{item['arm']} {item['query']['query']}",
            flush=True,
        )
        response = reasoning_response_from_dict(json.loads(item["payload"]))
        bundle = response.trusted_evidence
        evidence = [
            {
                "chunk_id": evidence_item.chunk_id,
                "source": evidence_item.source,
                "ordinal": evidence_item.ordinal,
                "text": evidence_item.text,
                "confidence": evidence_item.confidence,
                "cosine": evidence_item.cosine,
            }
            for evidence_item in bundle.items
        ]
        gold = set(item["query"].get("relevant_ids", []))
        evidence_keys = [
            f"{evidence_item.source}:{evidence_item.ordinal}"
            for evidence_item in bundle.items
            if evidence_item.ordinal is not None
        ]
        matched = sum(key in gold for key in evidence_keys)
        diagnostics = response.diagnostics
        record: dict[str, object] = {
            "query": item["query"]["query"],
            "answerable": item["query"]["answerable"],
            "gold_ids": item["query"].get("relevant_ids", []),
            "arm": item["arm"],
            "variant": item["variant"],
            "relation_control": item["relation_control"],
            "relation_control_seed": item["relation_control_seed"],
            "hub_threshold": item.get("hub_threshold", 32),
            "cosine_margin": item.get("cosine_margin", 0.10),
            "evidence": evidence,
            "evidence_ids": [evidence_item.chunk_id for evidence_item in bundle.items],
            "evidence_keys": evidence_keys,
            "evidence_recall": matched / len(gold) if gold else None,
            "evidence_precision": matched / len(evidence_keys) if evidence_keys else None,
            "trusted_items": len(bundle.items),
            "retrieval_latency_ms": diagnostics.latency_ms,
            "graph_latency_ms": diagnostics.graph_expansion_latency_ms,
            "graph_entities": diagnostics.graph_entities_inspected,
            "graph_relations": diagnostics.graph_relations_inspected,
            "graph_candidates": diagnostics.graph_candidates_discovered,
            "graph_rejected": diagnostics.graph_candidates_rejected,
            "graph_diagnostics": diagnostics.graph_diagnostics_encountered,
            "graph_admission_rejections": dict(diagnostics.graph_admission_rejections),
            "graph_expansion_refusals": dict(diagnostics.graph_expansion_refusals),
            "graph_gate_reason": diagnostics.graph_gate_reason,
            "graph_policy_fingerprint": diagnostics.graph_policy_fingerprint,
            "response_refusal_reason": response.refusal_reason,
            "bundle_decision": bundle.decision,
            "answer_provider_invoked": False,
        }
        if bundle.decision == "answer" and bundle.items:
            record["answer_provider_invoked"] = True
            system, user = render_evidence_prompt(bundle)
            try:
                raw = provider(system, user)
                envelope = parse_answer_envelope(raw)
                validation = validate_answer(envelope, bundle)
                record.update(
                    {
                        "answer": envelope.answer,
                        "citations": list(envelope.citations),
                        "insufficient_evidence": envelope.insufficient_evidence,
                        "answer_valid": validation.valid,
                        "answer_errors": list(validation.errors),
                        "provider": provider.provider_metadata().to_dict(),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "answer": None,
                        "citations": [],
                        "insufficient_evidence": None,
                        "answer_valid": False,
                        "answer_errors": [f"{type(exc).__name__}: {exc}"],
                        "provider": provider.provider_metadata().to_dict(),
                    }
                )
        else:
            record.update(
                {
                    "answer": None,
                    "citations": [],
                    "insufficient_evidence": True,
                    "answer_valid": True,
                    "answer_errors": [],
                    "provider": None,
                }
            )
        rows.append(record)
    Path(sys.argv[2]).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": sys.argv[2]}))


if __name__ == "__main__":
    main()
