"""RC-1 harness: the arms render what they claim, and a rejected envelope scores as a refusal.
Each test names its red proof."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recall.evidence import EVIDENCE_CLOSE, EVIDENCE_OPEN, SYSTEM_PROMPT, render_evidence_prompt  # noqa: E402
from reasoning_reader_contract_lme import (  # noqa: E402
    BENCHMARK_CONTRACT,
    GENERIC_CONTRACT,
    REFUSAL,
    TOP_K,
    build_bundle,
    hypothesis,
    judge_prompt,
    paired,
    render,
)

ITEMS = [
    {"id": f"raw_{n}", "content": f"[2023-05-{10 + n:02d} 09:00 UTC] note {n}", "created_at": f"2023-05-{10 + n:02d}T09:00:00Z",
     "score": "0.5", "source": "aml://session/x"}
    for n in range(12)
]


def _payload(user: str) -> dict:
    assert user.startswith(EVIDENCE_OPEN) and user.endswith(EVIDENCE_CLOSE)
    return json.loads(user[len(EVIDENCE_OPEN):-len(EVIDENCE_CLOSE)])


def test_bundle_keeps_top_k_strips_the_header_and_dates_items_by_session() -> None:
    """Invariant: the evidence is the top `TOP_K` items in rank order, their text without C9's date
    header (so R does not get dates in its text that the shipped library would not show), and
    `indexed_at` is the session timestamp.

    Red proof: dropping the header substitution (`text=str(item["content"])`) leaves
    "[2023-05-10 09:00 UTC] note 0" and fails the text equality.
    """
    bundle = build_bundle("q", ITEMS)
    assert len(bundle.items) == TOP_K
    assert bundle.items[0].text == "note 0" and bundle.items[0].chunk_id == "raw_0"
    assert bundle.items[3].indexed_at is not None and bundle.items[3].indexed_at.isoformat() == "2023-05-13T09:00:00+00:00"


def test_r_is_the_library_rendering_byte_for_byte() -> None:
    """Invariant: the control arm is the shipped boundary itself, not a re-implementation.

    Red proof: rendering R through the Q branch (adding `question_date`) makes the user message
    differ and fails the equality.
    """
    bundle = build_bundle("what did I buy?", ITEMS)
    assert render("R", bundle, "2023/05/30 (Tue) 23:40") == render_evidence_prompt(bundle)
    assert render("R2", bundle, "2023/05/30 (Tue) 23:40") == render_evidence_prompt(bundle)


def test_q_adds_only_the_question_date() -> None:
    """Invariant: Q differs from R by exactly one payload key, `question_date`, and keeps the
    shipped system prompt, so Q minus R measures the date alone.

    Red proof: appending `GENERIC_CONTRACT` to Q's system prompt fails the system equality.
    """
    bundle = build_bundle("q", ITEMS)
    system, user = render("Q", bundle, "2023/05/30 (Tue) 23:40")
    base = _payload(render_evidence_prompt(bundle)[1])
    data = _payload(user)
    assert system == SYSTEM_PROMPT
    assert data.pop("question_date") == "2023/05/30 (Tue) 23:40"
    assert data == base


def test_h_keeps_the_safety_prompt_first_and_names_each_date() -> None:
    """Invariant: H's system prompt STARTS with the unchanged safety contract and adds only the
    generic reader contract; each item carries `date` equal to its `indexed_at`; HB adds the
    benchmark rules and H does not.

    Red proof: putting the contract BEFORE the safety prompt fails `startswith`; omitting
    `payload["date"]` fails the date equality.
    """
    bundle = build_bundle("q", ITEMS)
    system, user = render("H", bundle, "2023/05/30 (Tue) 23:40")
    assert system.startswith(SYSTEM_PROMPT) and system == SYSTEM_PROMPT + GENERIC_CONTRACT
    item = _payload(user)["evidence"][0]
    assert item["date"] == item["indexed_at"] == "2023-05-10T09:00:00+00:00"
    hb_system, _ = render("HB", bundle, "2023/05/30 (Tue) 23:40")
    assert hb_system == SYSTEM_PROMPT + GENERIC_CONTRACT + BENCHMARK_CONTRACT and BENCHMARK_CONTRACT not in system


def test_a_rejected_envelope_scores_as_the_tools_refusal() -> None:
    """Invariant: an answer the library's validator rejects (here, no citation) is judged as the
    refusal `recall_reasoning_query` would return, never as the rejected text; a cited answer is
    judged as written.

    Red proof: returning the answer text when validation fails (skipping the `.valid` check) makes
    the uncited answer reach the judge and fails the first equality.
    """
    bundle = build_bundle("q", ITEMS)
    uncited = json.dumps({"answer": "a blue bike", "citations": [], "insufficient_evidence": False})
    cited = json.dumps({"answer": "a blue bike", "citations": ["raw_1"], "insufficient_evidence": False})
    unknown = json.dumps({"answer": "a blue bike", "citations": ["raw_99"], "insufficient_evidence": False})
    none = json.dumps({"answer": None, "citations": [], "insufficient_evidence": True})
    assert hypothesis(uncited, bundle) == (REFUSAL, "invalid")
    assert hypothesis(unknown, bundle) == (REFUSAL, "invalid")
    assert hypothesis(cited, bundle) == ("a blue bike", "answered")
    assert hypothesis(none, bundle) == (REFUSAL, "insufficient")
    assert hypothesis("not json", bundle) == (REFUSAL, "unparseable")
    assert hypothesis(None, bundle) == (REFUSAL, "provider_error")


def test_judge_uses_the_abstention_prompt_for_abs_ids_and_the_type_prompt_otherwise() -> None:
    """Invariant: LongMemEval decides the prompt by `_abs` in the id FIRST, then by type; the
    temporal prompt carries the off-by-one allowance and the preference prompt a rubric.

    Red proof: testing the type before `_abs` judges an unanswerable temporal question with the
    temporal prompt and fails the "unanswerable" assertion.
    """
    abs_q = {"question_id": "gpt4_x_abs", "question_type": "temporal-reasoning", "question": "q", "answer": "a"}
    tmp_q = {"question_id": "gpt4_x", "question_type": "temporal-reasoning", "question": "q", "answer": "a"}
    pref_q = {"question_id": "p1", "question_type": "single-session-preference", "question": "q", "answer": "r"}
    assert "correctly identify the question as unanswerable" in judge_prompt(abs_q, "resp")
    assert "off-by-one" in judge_prompt(tmp_q, "resp") and "unanswerable" not in judge_prompt(tmp_q, "resp")
    assert "Rubric: r" in judge_prompt(pref_q, "resp")


def test_paired_difference_and_wins() -> None:
    """Invariant: the paired difference is arm minus base over questions both answered.

    Red proof: swapping the operands (`base - arm`) reports -0.5 and fails.
    """
    correct = {"H": {"a": True, "b": True}, "R": {"a": False, "b": True}}
    out = paired(correct, "H", "R", ["a", "b"])
    assert out["diff"] == 0.5 and out["wins"] == 1 and out["losses"] == 0
