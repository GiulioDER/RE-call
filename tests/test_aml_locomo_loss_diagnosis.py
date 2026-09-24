"""Rule buckets of the C9 LoCoMo loss diagnosis.

Red proof, 2026-09-24, both by mutating ``scripts/aml_locomo_loss_diagnosis.py``:

* ``test_a_retrieval_miss_outranks_an_abstention``: swapping the two checks in
  ``deterministic_bucket`` made it return ``ABSTAINED`` for a missed question that also refused.
* ``test_refusals_are_abstentions_and_answers_are_not``: deleting the ``not (?:mentioned|...)``
  alternative from ``ABSTENTION`` made "It is not mentioned in the memories." return ``None``.
* ``test_memories_render_in_rank_order_with_their_timestamp``: replacing ``if text:`` in
  ``render_memories`` with ``if True:`` rendered the blank item as a line and failed the equality.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aml_locomo_loss_diagnosis import deterministic_bucket, render_memories  # noqa: E402


def _row(hit_at_100: int) -> dict[str, object]:
    return {"hits": {"turn_hit@100": hit_at_100}}


def test_a_retrieval_miss_outranks_an_abstention() -> None:
    assert deterministic_bucket(_row(0), "Not mentioned.") == "RETRIEVAL_MISS"


def test_refusals_are_abstentions_and_answers_are_not() -> None:
    for refusal in (
        "It is not mentioned in the memories.",
        "Unknown",
        "There is no information about that.",
        "Cannot be determined from the memories.",
    ):
        assert deterministic_bucket(_row(1), refusal) == "ABSTAINED", refusal
    for answer in ("7 May 2023", "Sweden", "She knows about the adoption"):
        assert deterministic_bucket(_row(1), answer) is None, answer


def test_memories_render_in_rank_order_with_their_timestamp() -> None:
    rendered = render_memories(
        [
            {"content": "Caroline: first", "created_at": "2023-05-08T13:56:00Z"},
            {"content": "  ", "created_at": "2023-05-08T13:56:00Z"},
            {"content": "Melanie: second", "created_at": None},
        ]
    )
    assert rendered == "- [2023-05-08T13:56:00Z] Caroline: first\n- Melanie: second"
