"""The supersession probe: does annotating evidence with successor markers change the answer?

Pre-registered at `results/enterprise_rag/PREREGISTRATION-supersession-annotation.md`, committed
at `4f0a8c8` before this existed. Reads the frozen evidence fixture, so the index is out of the
loop and both arms see byte-identical evidence.

Two arms, same frozen bundles, same model, temperature 0, same unmodified `SYSTEM_PROMPT`:

- **C**, control: the evidence as it renders today.
- **S**, supersession: each item carries a library-authored annotation naming whether the document
  it came from declares itself the current version or is superseded by a sibling in the bundle.

⚠️ **The annotation is prepended to the item TEXT, and that is a probe shortcut, not the design.**
`recall/evidence.py` is frozen: `render_evidence_prompt`'s body is pinned as source text by
`tests/test_evidence_contract.py`, so a probe must not add a field to the item payload. The shipped
version would add a real field. What matters for the measurement is preserved either way: the
annotation is library-authored, it is a selection among module-level constants, and it lands inside
the delimited data region rather than the instruction channel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Phrases by which a document declares ITSELF the newer one. Drawn from step 0, which read the
#: eight gold documents behind the four supersession rows and found the successor announcing
#: itself in every pair: "intended to supersede the older Confluence page", "no longer a direct
#: embedded blob in v1", "includes the v2 scoring thresholds", "Updated classifier thresholds".
#:
#: ⚠️ Deliberately NOT the nine-word status vocabulary the shipped detector uses. That one matched
#: 8 of 8 documents positively, so it separates nothing on this corpus.
SUCCESSOR_MARKERS = (
    r"\bsupersed\w*",
    r"\bno longer\b",
    r"\bupdated\b",
    r"\breplaces\b",
    r"\bdeprecat\w*",
    # ⚠️ NOT `\bv2\b`. Underscore is a word character, so `\b` finds no boundary in
    # `retention_policy_v2.md` and the marker misses every versioned FILENAME, which is where
    # this signal most often lives. Apparatus check A4 caught exactly that.
    r"(?<![a-z0-9])v2(?![a-z0-9])",
    r"\bversion 2\b",
    r"\brevised\b",
)
_MARKER_RE = re.compile("|".join(SUCCESSOR_MARKERS), re.IGNORECASE)

#: The two annotations, as module-level literals. No corpus byte reaches the instruction channel
#: and no annotation is interpolated from document text: the marker phrase is quoted back, but it
#: sits inside the data region with everything else the document contributed.
CURRENT = "[supersession: this document declares itself the current version]"
SUPERSEDED = "[supersession: a sibling document in this evidence set declares itself newer]"


def marker_hits(text: str) -> tuple[str, ...]:
    """Every distinct successor marker in `text`, lowercased, in first-appearance order."""
    seen: dict[str, None] = {}
    for match in _MARKER_RE.finditer(text):
        seen.setdefault(match.group(0).lower(), None)
    return tuple(seen)


def annotate(hits: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Map chunk_id to an annotation, or an empty map when the rule does not fire.

    The rule is COMPARATIVE, not absolute, which is what makes it directional. Marker presence
    alone is not enough: step 0 found the OLDER document in `qst_0419` also carrying "legacy" and
    "replace". So each source document is scored by how many distinct successor markers it
    carries, and the annotation fires only when one document strictly beats every other.

    Does not fire when the bundle holds a single source document: there is no sibling to be newer
    than, which is pre-registered apparatus check A5.
    """
    by_doc: dict[str, list[str]] = {}
    for hit in hits:
        doc = str(hit.get("doc_id") or hit.get("source") or "")
        text = f"{hit.get('source', '')} {hit.get('title', '')} {hit.get('text', '')}"
        by_doc.setdefault(doc, [])
        by_doc[doc].extend(marker_hits(text))

    if len(by_doc) < 2:
        return {}
    scores = {doc: len(set(markers)) for doc, markers in by_doc.items()}
    best = max(scores.values())
    winners = [doc for doc, score in scores.items() if score == best]
    if best == 0 or len(winners) != 1:
        return {}

    winner = winners[0]
    out: dict[str, str] = {}
    for hit in hits:
        doc = str(hit.get("doc_id") or hit.get("source") or "")
        chunk_id = str(hit.get("chunk_id"))
        out[chunk_id] = CURRENT if doc == winner else SUPERSEDED
    return out


def build_bundle_text(hits: Sequence[Mapping[str, object]], annotations: Mapping[str, str]) -> list[str]:
    """Item texts for one arm. Empty `annotations` yields the control arm's texts unchanged."""
    texts = []
    for hit in hits:
        note = annotations.get(str(hit.get("chunk_id")))
        body = str(hit.get("text", ""))
        texts.append(f"{note}\n{body}" if note else body)
    return texts


def _trusted_result(
    question: str, hits: Sequence[Mapping[str, object]], texts: Sequence[str]
) -> Any:
    from datetime import timedelta

    from recall.types import (
        Chunk,
        Provenance,
        StalenessReport,
        TrustedHit,
        TrustedResult,
        Validity,
    )

    trusted = [
        TrustedHit(
            chunk=Chunk(
                id=str(hit["chunk_id"]),
                source=str(hit.get("source", "")),
                text=text,
                metadata={"doc_id": str(hit.get("doc_id", ""))},
            ),
            cosine=float(hit.get("score") or 0.0),  # type: ignore[arg-type]
            confidence=float(hit.get("score") or 0.0),  # type: ignore[arg-type]
            verdict="ok",
            provenance=Provenance(
                source=str(hit.get("source", "")),
                file=str(hit.get("doc_id", "")),
                ord=None,
                indexed_at=None,
            ),
            validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
        )
        for hit, text in zip(hits, texts, strict=True)
    ]
    return TrustedResult(
        query=question,
        hits=trusted,
        abstained=not trusted,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(
            stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=3650)
        ),
    )


def run_arm(
    question: str,
    hits: Sequence[Mapping[str, object]],
    annotations: Mapping[str, str],
    provider: Any,
    max_items: int,
) -> dict[str, object]:
    from recall.evidence import EvidencePolicy, EvidenceValidationError, generate_from_evidence

    texts = build_bundle_text(hits, annotations)
    result = _trusted_result(question, hits, texts)
    try:
        generated = generate_from_evidence(result, provider, EvidencePolicy(max_items=max_items))
    except EvidenceValidationError as exc:
        return {"answer": "", "validation_error": str(exc)}
    envelope = generated.envelope
    return {
        "answer": envelope.answer or "",
        "insufficient_evidence": envelope.insufficient_evidence,
        "citations": len(envelope.citations),
        "evidence_items": len(generated.evidence.items),
    }


def main(argv: list[str] | None = None) -> int:
    from recall._env import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="annotate only; call no model")
    args = parser.parse_args(argv)
    load_dotenv()

    from benchmarks.fact_anchors import anchors_digest, load_anchors, score_row
    from benchmarks.freeze_supersession_evidence import evidence_digest

    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    evidence = payload["evidence"]
    digest = evidence_digest(evidence)
    if digest != payload["_provenance"]["evidence_sha256"]:
        raise SystemExit("evidence fixture does not match its own digest; refusing to run")
    anchors = load_anchors(args.anchors)
    a_digest = anchors_digest(json.loads(args.anchors.read_text(encoding="utf-8")))

    provider = None
    if not args.dry_run:
        from recall.answer_provider import OpenAIAnswerProvider, _client_from_env

        env = {**os.environ, "RECALL_REASONING_MODEL": args.model}
        provider = OpenAIAnswerProvider(_client_from_env(env), model_id=args.model)

    rows: dict[str, object] = {}
    for qid in sorted(evidence):
        row = evidence[qid]
        hits = row["hits"]
        annotations = annotate(hits)
        record: dict[str, object] = {
            "group": row["group"],
            "annotator_fired": bool(annotations),
            "annotated_current": sorted(
                {str(h.get("doc_id")) for h in hits
                 if annotations.get(str(h.get("chunk_id"))) == CURRENT}
            ),
            "n_docs_in_bundle": len({str(h.get("doc_id")) for h in hits}),
        }
        if not args.dry_run:
            for arm, ann in (("C", {}), ("S", annotations)):
                out = run_arm(row["question"], hits, ann, provider, args.max_items)
                score = score_row(str(out["answer"]), anchors[qid])
                record[arm] = {
                    **out,
                    "hits": score.hits,
                    "total": score.total,
                    "rate": round(score.rate, 4),
                    "missed": list(score.missed),
                    "violated": list(score.violated),
                    "answer_chars": len(str(out["answer"])),
                }
        rows[qid] = record
        fired = "FIRED" if record["annotator_fired"] else "-"
        extra = ""
        if not args.dry_run:
            extra = f"  C={record['C']['rate']:.2f} S={record['S']['rate']:.2f}"  # type: ignore[index]
        print(f"{qid} [{row['group']:16}] {fired:5}{extra}", flush=True)

    artifact = {
        "_provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "preregistration": "results/enterprise_rag/PREREGISTRATION-supersession-annotation.md",
            "registration_commit": "4f0a8c83a199367f1db9eb4ffd257902a7eb8573",
            "evidence_sha256": digest,
            "anchors_sha256": a_digest,
            "model": args.model,
            "temperature": 0.0,
            "max_items": args.max_items,
            "judge": "none; mechanical fact-anchor scoring",
            "dry_run": args.dry_run,
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
