"""RC-2 LoCoMo harness: arms render with the shipped code, the question date is the LAST session's,
and a rejected envelope scores as a refusal. Each test names its red proof."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmarks.pipeline import JUDGE_SYSTEM_PROMPT  # noqa: E402
from recall.evidence import (  # noqa: E402
    DATED_SYSTEM_PROMPT,
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    SYSTEM_PROMPT,
    render_evidence_prompt,
)
from reasoning_reader_contract_locomo import (  # noqa: E402
    REFUSAL,
    TOP_K,
    build_bundle,
    hypothesis,
    judge_user,
    last_session_time,
    render,
)

ITEMS = [{"id": f"raw_{n}", "content": f"Caroline: note {n}", "created_at": f"2023-05-{10 + n:02d}T13:56:00Z",
          "session_id": "conv-26:session_1"} for n in range(12)]


def test_the_question_date_is_the_numerically_last_session() -> None:
    """Invariant: LoCoMo sessions are numbered; session_10 is later than session_9 though it sorts
    before it as text, and the question is asked after the last one.

    Red proof: sorting the keys as text (``sorted(key for key in ...)``) picks session_9 and fails.
    """
    conversation = {"session_9_date_time": "2:31 pm on 17 July, 2023",
                    "session_10_date_time": "8:56 am on 4 August, 2023", "session_9": []}
    assert last_session_time(conversation) == datetime(2023, 8, 4, 8, 56, tzinfo=timezone.utc)


def test_r_is_the_shipped_plain_rendering_and_h_the_shipped_dated_one() -> None:
    """Invariant: the arms ARE the product code: R is `render_evidence_prompt`, H is
    `render_dated_evidence_prompt` with the ISO question date, top `TOP_K` items in served order.

    Red proof: rendering H with the plain renderer fails the system identity check.
    """
    bundle = build_bundle("When did Caroline go?", ITEMS)
    assert len(bundle.items) == TOP_K and bundle.items[0].chunk_id == "raw_0"
    assert render("R", bundle, datetime(2023, 8, 4, tzinfo=timezone.utc)) == render_evidence_prompt(bundle)
    system, user = render("H", bundle, datetime(2023, 8, 4, 8, 56, tzinfo=timezone.utc))
    assert system is DATED_SYSTEM_PROMPT and system != SYSTEM_PROMPT
    data = json.loads(user[len(EVIDENCE_OPEN):-len(EVIDENCE_CLOSE)])
    assert data["question_date"] == "2023-08-04T08:56:00+00:00"
    assert data["evidence"][0]["date"] == "2023-05-10T13:56:00+00:00"


def test_a_rejected_envelope_is_the_refusal_and_the_judge_prompt_is_the_benchmarks() -> None:
    """Invariant: an uncited answer is judged as the refusal the tool returns; the judge reads
    `benchmarks.pipeline`'s user message format.

    Red proof: skipping the `validate_answer` check lets the uncited text through and fails.
    """
    bundle = build_bundle("q", ITEMS)
    uncited = json.dumps({"answer": "7 May 2023", "citations": [], "insufficient_evidence": False})
    cited = json.dumps({"answer": "7 May 2023", "citations": ["raw_2"], "insufficient_evidence": False})
    assert hypothesis(uncited, bundle) == (REFUSAL, "invalid")
    assert hypothesis(cited, bundle) == ("7 May 2023", "answered")
    assert judge_user("q", "g", "a") == "Question: q\nGold answer: g\nPredicted answer: a\nCorrect?"
    assert "Reply with exactly YES" in JUDGE_SYSTEM_PROMPT
