"""X-1 Stage C (amendment 5) builds each source's prompt and scores as registered.

Invariants: PersonaMem-v2's chat history is replaced by ONE system message of retrieved memories,
followed by the query with AML's recall suffix and AML's MCQ message; an answer is right only when
the chosen option's text is the correct answer, and no letter is wrong; a CLBench judge reply is
parsed as AML parses it (code fences stripped), and one that does not parse is ``None`` so the
caller retries; a question answered twice by one arm stops the score.

The AML pipelines are stand-ins with AML's interfaces (the real ones live in the pinned checkout on
the testbench); what is tested is this harness's own mapping and parsing.

Red proof, 2026-09-27, each against ``scripts/aml_x1_stagec.py`` with this file unchanged
(``PYTHONDONTWRITEBYTECODE=1``):
- ``personamem_messages`` dropping ``RECALL_SUFFIX``:
  ``test_personamem_replaces_the_history_with_one_memory_message`` failed on the query equality.
- ``personamem_correct`` indexing with ``ord(letter) - 64``:
  ``test_personamem_scores_the_chosen_options_text`` failed on its first ``is True``.
- ``clbench_judged`` not stripping a leading fence:
  ``test_clbench_judge_replies_parse_as_aml_parses_them`` failed on the fenced reply's score.
- the duplicate check removed from ``score``:
  ``test_score_refuses_an_arm_answering_twice`` failed on ``DID NOT RAISE``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aml_x1_stagec", Path(__file__).parents[1] / "scripts" / "aml_x1_stagec.py"
)
assert SPEC and SPEC.loader
stagec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stagec
SPEC.loader.exec_module(stagec)

SUFFIX = " Please recall my related preferences from our conversation history to give personalized responses."


def _letter(text: str) -> str:
    match = re.search(r"Final Answer:\s*([A-Z])", str(text))
    return match.group(1) if match else ""


def _statuses(value: object) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


PIPELINE = SimpleNamespace(
    RECALL_SUFFIX=SUFFIX,
    MCQ_PROMPT_TEMPLATE="Please choose the best answer from the following options:\n\n{options}\n",
    extract_final_letter=_letter,
    _coerce_status_list=_statuses,
    _coerce_score=lambda payload: 1 if int(payload.get("Overall Score", 0)) == 1 else 0,
    _compute_requirement_ratio=lambda s: sum(x == "yes" for x in s) / len(s) if s else 0.0,
)
SCORER = {"user_query": "Which gift suits me?", "options": ["a scarf", "a book", "a mug"],
          "correct_letter": "B", "correct_answer": "a book"}
ITEMS = [{"id": "1", "content": "I love reading novels"}, {"id": "2", "content": "  "},
         {"id": "3", "content": "Mugs clutter my desk"}]


def test_personamem_replaces_the_history_with_one_memory_message() -> None:
    messages = stagec.personamem_messages(PIPELINE, SCORER, ITEMS)
    assert [m["role"] for m in messages] == ["system", "user", "system"]
    assert messages[0]["content"] == (stagec.MEMORY_PREAMBLE + "\n\n- I love reading novels\n- Mugs clutter my desk")
    assert messages[1]["content"] == "Which gift suits me?" + SUFFIX
    assert "A. a scarf\nB. a book\nC. a mug" in messages[2]["content"]


def test_personamem_scores_the_chosen_options_text() -> None:
    assert stagec.personamem_correct(PIPELINE, SCORER, "Reasoning.\nFinal Answer: B") == (True, "B")
    assert stagec.personamem_correct(PIPELINE, SCORER, "Final Answer: A") == (False, "A")
    assert stagec.personamem_correct(PIPELINE, SCORER, "I cannot tell.") == (False, "")
    assert stagec.personamem_correct(PIPELINE, SCORER, "Final Answer: Z") == (False, "Z")


def test_clbench_judge_replies_parse_as_aml_parses_them() -> None:
    body = {"Grading Rationale": "ok", "List of Requirement Satisfaction Status": ["yes", "no"], "Overall Score": 0}
    fenced = "```json\n" + json.dumps({**body, "Overall Score": 1}) + "\n```"
    assert stagec.clbench_judged(PIPELINE, fenced) == {"score": 1.0, "ratio": 0.5, "statuses": 2}
    assert stagec.clbench_judged(PIPELINE, json.dumps(body))["score"] == 0.0
    assert stagec.clbench_judged(PIPELINE, "not json") is None
    assert stagec.clbench_judged(PIPELINE, json.dumps({"Grading Rationale": "x"})) is None
    assert stagec.clbench_judged(PIPELINE, None) is None


def test_score_refuses_an_arm_answering_twice(tmp_path: Path) -> None:
    row = {"id": "personamem_v2:row1", "source": "personamem_v2", "arm": "B", "category": "c",
           "correct": 1.0, "letter": "B"}
    answers = tmp_path / "a.jsonl"
    answers.write_text(json.dumps(row) + "\n" + json.dumps({**row, "correct": 0.0}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="twice"):
        stagec.score(SimpleNamespace(answers=answers, out=tmp_path / "s.json"))
