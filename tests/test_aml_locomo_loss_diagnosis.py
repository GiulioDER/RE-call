"""Rule buckets of the C9 LoCoMo loss diagnosis.

Red proof, 2026-09-24, both by mutating ``scripts/aml_locomo_loss_diagnosis.py``:

* ``test_a_retrieval_miss_outranks_an_abstention``: swapping the two checks in
  ``deterministic_bucket`` made it return ``ABSTAINED`` for a missed question that also refused.
* ``test_refusals_are_abstentions_and_answers_are_not``: deleting the ``not (?:mentioned|...)``
  alternative from ``ABSTENTION`` made "It is not mentioned in the memories." return ``None``.
* ``test_memories_render_in_rank_order_with_their_timestamp``: replacing ``if text:`` in
  ``render_memories`` with ``if True:`` rendered the blank item as a line and failed the equality.
* ``test_the_counterfactual_arm_keeps_only_raw_windows_in_served_order``: making
  ``served_items`` return every item kept the compiled records and failed the id list.
* ``test_the_route_filter_uses_the_served_router``: making ``route_of`` return ``"code"`` failed
  the ``context`` assertion.
* ``test_paired_counts_treatment_minus_control``: reversing the subtraction in ``paired`` gave
  -25.0 and failed.

Added for docs/preregistrations/2026-09-24-aml-c9-reader-dates.md, red proof by mutating the
same script:

* ``test_the_content_view_never_shows_a_timestamp``: making the stamp ignore ``dated`` (``stamp =
  str(item.get("created_at") or "").strip()``) rendered ``- [2023-05-08T13:56:00Z] Caroline:
  first`` and failed the equality.
* ``test_timestamped_windows_changes_only_the_renderer_flag``: making ``timestamped_windows``
  return ``behavior`` unchanged left ``content_only_windows`` True and failed the first assertion.
* ``test_parallel_add_lanes_keep_each_users_sessions_in_order``: making ``adds_by_user`` prepend
  (``lanes.setdefault(...).insert(0, request)``) reversed a user's sessions and failed the lane
  equality.

Added for docs/preregistrations/2026-09-25-aml-c9-window-format.md, same method:

* ``test_the_product_dated_view_is_what_the_service_returns``: making ``product_dated`` return
  its input unchanged dropped the ``[2023-05-08 13:56 UTC]`` prefix and failed the equality.
* ``test_the_stage_model_follows_the_environment_and_defaults_to_gpt_4o_mini``: making
  ``stage_model`` return its default unconditionally ignored ``AML_DIAG_ANSWER_MODEL`` and failed
  the first assertion.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aml_locomo_loss_diagnosis import (  # noqa: E402
    adds_by_user,
    product_dated,
    stage_model,
    deterministic_bucket,
    paired,
    render_memories,
    route_of,
    served_items,
    timestamped_windows,
)
from recall_aml.variants import variant  # noqa: E402


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


def test_the_counterfactual_arm_keeps_only_raw_windows_in_served_order() -> None:
    items = [
        {"id": "a", "kind": "raw"},
        {"id": "b", "kind": "successful repair"},
        {"id": "c", "kind": "raw"},
        {"id": "d", "kind": "repository fact"},
    ]
    assert [i["id"] for i in served_items(items, drop_compiled=True)] == ["a", "c"]
    assert served_items(items, drop_compiled=False) == items


def test_the_route_filter_uses_the_served_router() -> None:
    assert route_of("When did Melanie paint a sunrise?") == "context"
    assert route_of("What is Caroline's identity?") == "code"
    assert route_of("What was in the photo Caroline shared?") == "multimodal"


def test_paired_counts_treatment_minus_control() -> None:
    result = paired([0, 1, 1, 0], [1, 1, 0, 1])
    assert result["delta_points"] == 25.0
    assert (result["wrong_to_right"], result["right_to_wrong"]) == (2, 1)


def test_the_content_view_never_shows_a_timestamp() -> None:
    rendered = render_memories(
        [
            {"content": "Caroline: first", "created_at": "2023-05-08T13:56:00Z"},
            {"content": "  ", "created_at": "2023-05-08T13:56:00Z"},
            {"content": "Melanie: second", "created_at": None},
        ],
        dated=False,
    )
    assert rendered == "- Caroline: first\n- Melanie: second"


def test_timestamped_windows_changes_only_the_renderer_flag() -> None:
    served = variant("C9_routed_specialists_grounded_graph_atomic")
    assert served.content_only_windows is True
    changed = timestamped_windows(served)
    assert changed.content_only_windows is False
    assert dataclasses.replace(changed, content_only_windows=True) == served


def test_parallel_add_lanes_keep_each_users_sessions_in_order() -> None:
    adds = [
        {"user_id": "u1", "session_id": "s1"},
        {"user_id": "u2", "session_id": "s1"},
        {"user_id": "u1", "session_id": "s2"},
        {"user_id": "u1", "session_id": "s3"},
        {"user_id": "u2", "session_id": "s2"},
    ]
    lanes = adds_by_user(adds)
    assert [[(a["user_id"], a["session_id"]) for a in lane] for lane in lanes] == [
        [("u1", "s1"), ("u1", "s2"), ("u1", "s3")],
        [("u2", "s1"), ("u2", "s2")],
    ]


def test_the_product_dated_view_is_what_the_service_returns() -> None:
    items = [
        {"id": "a", "content": "Caroline: first", "created_at": "2023-05-08T13:56:00+00:00",
         "session_id": "s", "kind": "raw"},
        {"id": "b", "content": "Melanie: second", "created_at": None, "session_id": "s",
         "kind": "raw"},
    ]
    rendered = render_memories(product_dated(items), dated=False)
    assert rendered == "- [2023-05-08 13:56 UTC] Caroline: first\n- Melanie: second"


def test_the_stage_model_follows_the_environment_and_defaults_to_gpt_4o_mini(monkeypatch) -> None:
    monkeypatch.setenv("AML_DIAG_ANSWER_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.delenv("AML_DIAG_JUDGE_MODEL", raising=False)
    assert stage_model("answer") == "deepseek/deepseek-v4-flash-0731"
    assert stage_model("judge") == "openai/gpt-4o-mini"
    monkeypatch.setenv("AML_DIAG_JUDGE_MODEL", "  ")
    assert stage_model("judge") == "openai/gpt-4o-mini"
