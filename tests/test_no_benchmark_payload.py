"""No tracked file may carry a benchmark's questions or its gold labels.

`docs/ENTERPRISE_RAG_SUBMISSION.md` states that this repository deliberately carries no benchmark
questions, gold answers or document text. That is a promise to the people who publish those
benchmarks, and until now it was kept by memory alone.

Memory was not enough. On 2026-08-20, four committed result files were found carrying 1,013 ATM
Bench question texts with their gold evidence ids, between 0.18 and 29 MB each, on a branch that
was one command from being pushed. For the 139 `list_recall` questions those gold evidence ids ARE
the gold answer, so what had been committed was the dataset rather than a log of a run. A fifth
file, 29 MB, was initially cleared by a hand-written scan that stopped four levels deep while the
records sat five levels down.

⛔ The failure mode this guards is not carelessness, it is that the payload arrives inside an
artifact nobody thinks of as data. A retrieval-results dump looks like a measurement. It is a
measurement that quotes the whole question set.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: A record carrying one of these keys is quoting the benchmark rather than summarising it.
PAYLOAD_KEYS = frozenset(
    {"question", "answer", "ground_truth", "gt_evidence_ids", "evidence_ids", "gold_answer"}
)

#: Below this many records a file is a fixture or an example, not a dataset. Set low on purpose:
#: the point is to catch a dump, and a legitimate file rarely needs even this many.
RECORD_LIMIT = 5

#: Files that are FINE, each for a stated reason. Keep this list short, and never add a results
#: artifact to it.
EXEMPT = {
    # Ids and a selection rule, deliberately answer-free: the whole point of that file is that it
    # identifies questions without carrying anything that can be answered.
    "benchmarks/atm_subset300_ids.json",
    # OUR predictions, and only ours: question_id, answer, document_ids. No question text and no
    # gold. It is the submission artifact ENTERPRISE_RAG_SUBMISSION.md deliberately publishes, and
    # publishing what our own system said is the opposite of leaking what it was asked.
    "benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.answers.jsonl",
    # recall's OWN evaluation fixtures. The questions are written for this project, about a
    # fictional API, and belong to nobody else.
    "recall/eval/reasoning_session1.json",
    "recall/eval/reasoning_session6.json",
}

#: ⛔ KNOWN VIOLATIONS THAT PREDATE THIS GATE. These are NOT fine. They are listed so the suite is
#: green while the decision they need is taken, and listed HERE rather than in EXEMPT so that
#: nobody can mistake one for the other.
#:
#: Both were already pushed to the public master before this test existed, so removing them from
#: the tree does not unpublish them and needs a deliberate call about history, not a quiet delete.
#:
#: `locomo_errors.json` carries 156 LoCoMo records, each with the benchmark's `question` and its
#: `golden_answer`. `beam_9l_temporal.json` carries 7 records with a question, a `gold_interval`
#: and a `gold_rubric`.
#:
#: Neither was found by a person. Both were found by this gate on the first run it ever made, which
#: is the argument for the gate.
KNOWN_UNFIXED = {
    "benchmarks/audit_data/locomo_errors.json",
    "results/beam_9l_temporal.json",
}


def tracked_json_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.json", "*.jsonl"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def count_payload_records(value: object, depth: int = 0) -> int:
    """Walk the WHOLE structure, to any depth.

    The scan that missed a 29 MB file stopped at depth four while its records were at depth five,
    and reported the file clean. Depth limits are how a dataset hides inside a results artifact.
    """
    if depth > 40:  # cycle guard only; real data is far shallower
        return 0
    if isinstance(value, dict):
        if PAYLOAD_KEYS & set(value):
            scalars = [
                key for key in PAYLOAD_KEYS & set(value)
                if not isinstance(value[key], (dict,))
            ]
            if scalars:
                return 1
        return sum(count_payload_records(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return sum(count_payload_records(v, depth + 1) for v in value)
    return 0


def load_any(path: Path) -> object:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return rows  # not JSON at all; whatever parsed is enough to judge
        return rows


@pytest.mark.parametrize("relative", tracked_json_files())
def test_no_tracked_file_carries_a_benchmark_payload(relative: str) -> None:
    if relative in EXEMPT:
        pytest.skip(f"{relative} is exempt and says why in EXEMPT")
    if relative in KNOWN_UNFIXED:
        pytest.skip(
            f"{relative} is a KNOWN violation that predates this gate and is already on the "
            f"public master. It needs a decision about history, not a quiet delete."
        )
    path = REPO / relative
    if not path.exists():
        pytest.skip(f"{relative} is tracked but absent from this worktree")
    found = count_payload_records(load_any(path))
    assert found < RECORD_LIMIT, (
        f"{relative} carries {found} records with benchmark question or gold fields "
        f"({', '.join(sorted(PAYLOAD_KEYS))}). This repository states that it holds no benchmark "
        f"questions, gold answers or document text. Publish aggregates, or ids without answers, "
        f"and keep the payload out of the tree."
    )


def test_the_guard_actually_catches_a_dump() -> None:
    """A gate that cannot fail is not a gate. This is the shape that was nearly published."""
    dump = {"arms": {"dense": {"details": [
        {"id": f"q{i}", "question": "when was it", "gt_evidence_ids": ["e1"]} for i in range(10)
    ]}}}
    assert count_payload_records(dump) == 10


def test_the_guard_does_not_fire_on_an_id_list() -> None:
    """The exemption has to be earned by shape, not only by being named in EXEMPT."""
    ids = {"selection": "sha256 order", "ids": {"number": ["a", "b"], "open_end": ["c"]}}
    assert count_payload_records(ids) == 0
