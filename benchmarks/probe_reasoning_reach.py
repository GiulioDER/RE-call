"""Does the reasoning layer's contradiction detection reach EnterpriseRAG's conflicts?

**No model calls. No API spend.** The deterministic proposal path and the planner are pure
functions over a graph projection, so the question "does the reasoning fire, and what does it do
when it fires" is answerable offline, in a second, for nothing.

Motivation. `results/enterprise_rag/FINDING-where-the-deficit-actually-is.md` locates five
`conflicting_info` rows where the gold documents were fully retrieved and the answer was still
wrong. Those are the rows a reasoning layer is supposed to win. Before paying a model to attempt
them, this probe asks whether the shipped mechanism can engage with them at all.

The mechanism, read from `recall/reasoning_proposals/_deterministic.py`, is three conjuncts:

1. two claims must share a `subject`, which is a decision-subject regex over the first 1000
   characters, falling back to the FILE STEM with a trailing version suffix stripped;
2. their validity windows must overlap, which is trivially true when neither has frontmatter,
   because an absent bound becomes `datetime.min` / `datetime.max`;
3. their texts must carry OPPOSING STATUS VOCABULARY, from two closed nine-word lists:
   `enabled|active|valid|approved|ship|on` against `disabled|inactive|invalid|rejected|blocked|off`.

And when it does fire, `_check_contradiction` calls `_fail_closed(..., "ambiguous_evidence")`.
That is the fact worth knowing before budgeting anything: **the reasoning layer's response to a
contradiction is to ABSTAIN, not to resolve it.** It is a safety mechanism, not an accuracy one.

Run: `python -m benchmarks.probe_reasoning_reach`
"""

from __future__ import annotations

from recall.reasoning_graph import build_reasoning_graph
from recall.reasoning_proposals._deterministic import deterministic_inference_proposals
from recall.types import Chunk

#: An EnterpriseRAG shaped conflict: two different documents, factual disagreement, no status
#: vocabulary and no shared subject. This is what `conflicting_info` questions are made of.
ENTERPRISE_SHAPED = [
    Chunk(
        id="c1",
        source="dsid_aaa__multipart-upload-limits-2025.txt",
        text=(
            "Multipart upload limits are 10 MiB per file and 50 MiB total per request. "
            "These limits apply to all tenants on the shared ingest path."
        ),
        metadata={"file": "dsid_aaa__multipart-upload-limits-2025.txt", "doc_id": "dsid_aaa"},
    ),
    Chunk(
        id="c2",
        source="dsid_bbb__ingest-runbook-2026.txt",
        text=(
            "The ingest runbook raises multipart upload limits to 25 MiB per file and "
            "100 MiB total per request for enterprise tenants."
        ),
        metadata={"file": "dsid_bbb__ingest-runbook-2026.txt", "doc_id": "dsid_bbb"},
    ),
]

#: A memo shaped conflict: same subject via the version-stripped file stem, and the nine-word
#: status vocabulary on both sides. This is the corpus the mechanism was designed for.
MEMO_SHAPED = [
    Chunk(
        id="m1",
        source="retention_policy_v1.md",
        text="The retention policy for cold storage is approved and active for all tenants.",
        metadata={"file": "retention_policy_v1.md"},
    ),
    Chunk(
        id="m2",
        source="retention_policy_v2.md",
        text="The retention policy for cold storage is rejected and disabled pending review.",
        metadata={"file": "retention_policy_v2.md"},
    ),
]


def _contradictions(chunks: list[Chunk], label: str) -> int:
    graph = build_reasoning_graph(
        chunks,
        tenant_id="probe",
        generation_id="probe-gen",
        include_text=True,
    )
    proposals = deterministic_inference_proposals(graph, pipeline_id="probe-pipeline")
    contradictions = [p for p in proposals if p.proposed_relation == "contradicts"]
    print(f"  {label}")
    print(f"    proposals of any kind : {len(proposals)}")
    print(f"    contradiction proposals: {len(contradictions)}")
    for p in contradictions:
        print(f"      -> {p.subject_id} vs {p.object_id}: {p.explanation}")
    for p in proposals:
        if p.proposed_relation != "contradicts":
            print(f"      (other: {p.proposed_relation} via {p.rule_id})")
    return len(contradictions)


def main() -> int:
    print("Does the shipped contradiction detector engage EnterpriseRAG's conflicts?")
    print()
    enterprise = _contradictions(ENTERPRISE_SHAPED, "EnterpriseRAG shaped (factual, numeric):")
    print()
    memo = _contradictions(MEMO_SHAPED, "Memo shaped (shared subject, status vocabulary):")
    print()
    print("Reading:")
    if enterprise == 0 and memo > 0:
        print("  The detector fires on the corpus it was designed for and NOT on this benchmark's")
        print("  conflicts. It matches a nine-word status vocabulary between claims sharing a")
        print("  subject; EnterpriseRAG conflicts are numeric disagreements across documents whose")
        print("  file stems differ, so neither conjunct holds.")
        print()
        print("  And when it DOES fire, `_check_contradiction` calls `_fail_closed` with")
        print("  `ambiguous_evidence`. The layer abstains rather than resolving, so on a benchmark")
        print("  scored for correctness it cannot win a row; at best it converts a wrong answer")
        print("  into a refusal, which scores the same.")
        return 0
    print(f"  UNEXPECTED: enterprise={enterprise}, memo={memo}. Re-read the predicates before")
    print("  trusting either number; this probe's whole value is that it is cheap to re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
