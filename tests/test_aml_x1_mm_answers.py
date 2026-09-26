"""The MM-1/MM-3 held-out answer harness builds each source's prompt and scores as registered.

Invariants: B and B2 are Stage B's own items and D/Dt the recorded arms; the image cap keeps the
longest ranked prefix with at most 30 images; MobileMem's question carries its options one per line
and switches to the multimodal prompt, images in place, when any image is retrieved; MemLens marks
each image item and attaches the image after the text; a judge reply without a label is unscored,
never counted as wrong.

Red proof, 2026-09-26, each against ``scripts/aml_x1_mm_answers.py`` with this file unchanged
(``PYTHONDONTWRITEBYTECODE=1``):
- ``items_for`` answering B2 from the D arm: ``test_b_and_b2_are_stage_b_items`` failed on ``==``.
- ``capped`` counting ``total >= limit``: ``test_the_cap_keeps_the_longest_prefix_within_30``
  failed on ``len(kept) == 3``.
- ``mobilemem_question`` dropping the options: ``test_mobilemem_question_lists_its_options``
  failed on ``in``.
- ``judge_label`` defaulting to WRONG: ``test_a_reply_without_a_label_is_unscored`` failed on
  ``is None``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "aml_x1_mm_answers", Path(__file__).parents[1] / "scripts" / "aml_x1_mm_answers.py"
)
assert SPEC and SPEC.loader
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)

PROMPTS = """Prompt for Question Answering with Text Memory

Question:
[Question]
Please answer the question based on the following memories:
[Retrieved Memories]
Answer Rules:
1. text rules


Prompt for Question Answering with Mutimodal Memory

Question:
[Question]
Please answer the question based on the following memories:
[Retrieved Memories]
The following images are retrieved as potentially relevant visual memories. The timestamps, in order, are:
[Timestamps]
[Images]
Answer Rules:
1. image rules


Prompt for LLM-as-a-Judge Evaluation

Question: [Question]
Gold answer: [Gold Answer]
Evidence: [Evidences]
Generated answer: [Generated Answer]
"""
SECTIONS = harness.mobilemem_sections(PROMPTS)
EV = {
    "mm_text": SECTIONS["Prompt for Question Answering with Text Memory"],
    "mm_multimodal": SECTIONS["Prompt for Question Answering with Mutimodal Memory"],
    "mm_judge": SECTIONS["Prompt for LLM-as-a-Judge Evaluation"],
    "prompt_builders": SimpleNamespace(build_mem0_answer_messages=lambda q, memories, date: [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "\n".join(f"- {m['memory']}" for m in memories) + f"\nQuestion: {q}"},
    ]),
}
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
IMAGES = {"d1": IMAGE, "d2": IMAGE}


def text_item(text: str) -> dict:
    return {"id": text, "content": text, "created_at": "2024-01-01T00:00:00Z"}


def image_item(digest: str, when: str = "2024-02-02T00:00:00Z") -> dict:
    return {"id": digest, "content": [{"type": "image_ref", "sha256": digest}], "created_at": when}


def test_the_three_mobilemem_prompts_are_split_out() -> None:
    assert set(SECTIONS) == {"Prompt for Question Answering with Text Memory",
                             "Prompt for Question Answering with Mutimodal Memory",
                             "Prompt for LLM-as-a-Judge Evaluation"}
    assert "[Images]" in EV["mm_multimodal"] and "[Images]" not in EV["mm_text"]


def test_b_and_b2_are_stage_b_items() -> None:
    search = {"items": [text_item("b")], "arms": {"D": {"items": [text_item("d")]},
                                                  "Dt": {"items": [text_item("dt")]}}}
    assert harness.items_for("B", search) == harness.items_for("B2", search) == [text_item("b")]
    assert harness.items_for("D", search) == [text_item("d")]
    assert harness.items_for("Dt", search) == [text_item("dt")]


def test_the_cap_keeps_the_longest_prefix_within_30() -> None:
    many = {"id": "m", "content": [{"type": "image_ref", "sha256": f"x{i}"} for i in range(10)]}
    items = [many, many, many, many]
    kept, cut = harness.capped(items)
    assert len(kept) == 3 and cut
    assert harness.capped(items[:3]) == (items[:3], False)


def test_mobilemem_question_lists_its_options() -> None:
    question = harness.mobilemem_question({"question": "Which city?"}, ["A. Rome", "B. Oslo"])
    assert question == "Which city?\nA. Rome\nB. Oslo"
    assert "A. Rome" in question


def test_mobilemem_uses_the_multimodal_prompt_only_when_an_image_is_retrieved() -> None:
    scorer = {"question": "Where was I?"}
    text_only = harness.mobilemem_messages(EV, scorer, None, [text_item("in Rome")], IMAGES)
    assert isinstance(text_only[0]["content"], str) and "- in Rome" in text_only[0]["content"]
    mixed = harness.mobilemem_messages(EV, scorer, None, [text_item("in Rome"), image_item("d1")], IMAGES)
    parts = mixed[0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text"]
    assert "2024-02-02T00:00:00Z" in parts[0]["text"] and "image rules" in parts[2]["text"]


def test_memlens_marks_image_items_and_attaches_the_images_after_the_text() -> None:
    scorer = {"question": "What colour?", "question_date": "2024/03/03"}
    messages = harness.memlens_messages(EV, scorer, [text_item("a note"), image_item("d2")], IMAGES)
    content = messages[-1]["content"]
    assert content[0]["type"] == "text" and "[image 1]" in content[0]["text"]
    assert content[1] == IMAGE


def test_a_reply_without_a_label_is_unscored() -> None:
    assert harness.judge_label('Reason. {"label": "CORRECT"}') == "CORRECT"
    assert harness.judge_label('{"label": "wrong"}') == "WRONG"
    assert harness.judge_label("I cannot decide.") is None
