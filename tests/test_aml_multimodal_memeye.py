"""Deterministic checks for the frozen public MemEye admission apparatus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recall_aml.models import AddRequest

from scripts.aml_multimodal_memeye import (
    HttpResult,
    JsonClient,
    aggregate_questions,
    answer_rotation,
    build_add_requests,
    pack_answer_content,
    retrieval_metrics,
)
from scripts.audit_aml_multimodal_memeye import audit
from scripts.select_aml_multimodal_memeye import ARMS, decide


def _dataset() -> dict:
    return {
        "multi_session_dialogues": [
            {
                "session_id": "S1",
                "date": "2024-01-05",
                "dialogues": [
                    {
                        "round": "S1:R1",
                        "user": "compare the visual",
                        "assistant": "noted",
                        "input_image": ["Brand_Memory_Test/one.png"],
                        "image_caption": ["forbidden annotation"],
                    },
                    {"round": "S1:R2", "user": "text only", "assistant": "noted"},
                ],
            }
        ],
        "human-annotated QAs": [],
    }


def test_add_translation_preserves_order_and_excludes_annotation(tmp_path: Path) -> None:
    """Image bytes follow user text, while the annotation never reaches Add.

    Red proof receipt ``memeye-add-shape-01`` targets
    ``scripts.aml_multimodal_memeye.build_add_requests``. Replacing the image part's Data URI with
    the dataset ``image_caption`` made the exact ordered content assertion fail because the second
    part became text instead of ``image_url``. The same node was observed red on frozen commit
    ``9b1cd324`` because the runner emitted an ISO string where the AML wire contract requires Unix
    milliseconds. The assertion received ``2024-01-05T12:00:00Z`` instead of
    ``1704456000000``. Restoring source-image encoding and integer milliseconds made it green.
    A fresh live attempt at frozen commit ``f5275670`` then returned HTTP 422 at the same first
    Add because ``image_url`` was a string instead of the AML object ``{"url": DATA_URI}``.
    Extending this boundary test with ``AddRequest.model_validate`` reproduced that exact failure
    before the repair; the nested URL shape makes the same request validate.
    """
    image = tmp_path / "Brand_Memory_Test" / "one.png"
    image.parent.mkdir()
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    requests = build_add_requests(_dataset(), tmp_path, arm="MM2_dual", user_id="u")
    assert len(requests) == 2
    assert requests[0]["session_id"] == "S1:R1"
    content = requests[0]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[0]["text"] == "compare the visual"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "forbidden annotation" not in json.dumps(requests)
    assert requests[0]["messages"][0]["timestamp"] == 1_704_456_000_000
    assert isinstance(requests[0]["messages"][0]["timestamp"], int)
    assert requests[1]["messages"][0]["content"] == "text only"
    AddRequest.model_validate(requests[0])


def test_vps2_wrapper_migrates_fresh_table_before_service_start() -> None:
    """Every immutable table is migrated with the owner role before the worker can boot.

    Red proof receipt ``memeye-schema-preflight-01`` targets
    ``scripts/aml_multimodal_memeye_vps2.sh``. Frozen commit ``6dfead31`` created the fresh table
    name and started the worker without applying migrations; the worker refused startup with
    ``SchemaTooOld`` listing migrations 0001 through 0007. This assertion was red because the
    wrapper contained no migration DSN or ``schema apply`` invocation. The repair loads the owner
    DSN and applies the 1,024-dimensional Voyage context schema before ``systemctl restart``.
    """
    script = Path("scripts/aml_multimodal_memeye_vps2.sh").read_text(encoding="utf-8")
    assert 'migration_dsn="${RECALL_MIGRATION_DSN:-}"' in script
    assert '--migration-dsn "$migration_dsn"' in script
    assert '--embedder "voyage-context"' in script
    assert "aml-multimodal-memeye-brand-v2" in script
    assert script.index("schema apply") < script.index('systemctl --user restart "$unit_name"')


def test_retrieval_metrics_use_exact_clue_round_ids() -> None:
    """Clue recall is session-ID membership, not text or substring matching.

    Red proof receipt ``memeye-clue-rank-01`` targets
    ``scripts.aml_multimodal_memeye.retrieval_metrics``. Changing the comparison to substring
    membership admitted ``S1:R10`` as clue ``S1:R1`` and failed Recall at 5 at its intended zero
    assertion. Restoring exact set membership made this node green.
    """
    items = [
        {"session_id": "S1:R10"},
        {"session_id": "S2:R1"},
        {"session_id": "S1:R1"},
        {"session_id": "S3:R1"},
        {"session_id": "S4:R1"},
        {"session_id": "S5:R1"},
        {"session_id": "S1:R2"},
    ]
    metrics = retrieval_metrics(items, ["S1:R1", "S1:R2"])
    assert metrics["first_clue_rank"] == 3
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["any_recall_at_5"] == 1
    assert metrics["complete_recall_at_5"] == 0
    assert metrics["complete_recall_at_10"] == 1
    assert metrics["clue_fraction_at_5"] == 0.5


def test_answer_packing_keeps_a_ranked_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget boundary stops packing instead of skipping to a later small item.

    Red proof receipt ``memeye-ranked-prefix-01`` targets
    ``scripts.aml_multimodal_memeye.pack_answer_content``. Replacing ``break`` with ``continue``
    admitted rank three after rejecting rank two, so ``admitted_items`` became two and this node
    failed at the registered prefix assertion. Restoring ``break`` made it green.
    """
    import scripts.aml_multimodal_memeye as runner

    monkeypatch.setattr(runner, "ANSWER_TOKEN_BUDGET", 250)
    items = [
        {"session_id": "S1", "content": "a" * 100},
        {"session_id": "S2", "content": "b" * 1000},
        {"session_id": "S3", "content": "c"},
    ]
    parts, packing = pack_answer_content(
        items, "question", {"A": "one", "B": "two", "C": "three", "D": "four"}
    )
    rendered = json.dumps(parts)
    assert packing["admitted_items"] == 1
    assert packing["truncated"] is True
    assert "session S1" in rendered
    assert "session S3" not in rendered


def test_answer_cost_has_conservative_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing provider cost still consumes the registered experiment ledger.

    Red proof receipt ``memeye-answer-cost-fallback-01`` targets
    ``scripts.aml_multimodal_memeye.answer_rotation``. Returning only OpenRouter's absent zero cost
    made ``cost_usd`` zero and failed the intended positive fallback assertion. Restoring the
    conservative token estimate made this node green. Frozen v2 commit ``2cc83bae`` later completed
    48 MM1 rotations, then the forty-ninth Answer request exceeded its explicit 180-second read
    timeout after Search returned HTTP 200. Capturing the constructed Answer client made this node
    red at 180 seconds. The apparatus repair raises only that timeout to 600 seconds.
    """
    captured: list[dict] = []
    captured_timeouts: list[float] = []

    def fake_call(self: JsonClient, path: str, payload: dict | None = None) -> HttpResult:
        assert path == "/chat/completions"
        assert payload is not None
        captured.append(payload)
        captured_timeouts.append(self.timeout)
        return HttpResult(
            200,
            {
                "choices": [{"message": {"content": "A"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 10},
            },
            1.0,
            10,
            1,
        )

    monkeypatch.setattr(JsonClient, "call", fake_call)
    _, usage = answer_rotation(
        JsonClient("https://example.invalid", "token", sleep=lambda _: None),
        api_key="answer-key",
        system_prompt="system",
        parts=[{"type": "text", "text": "question"}],
    )
    assert captured[0]["usage"] == {"include": True}
    assert captured_timeouts == [600]
    assert usage["reported_cost_usd"] == 0
    assert usage["cost_usd"] == pytest.approx(1.00004)


def test_answer_timeout_retries_identical_payload_and_reserves_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read timeout is retried without changing the request or hiding possible spend.

    Red proof receipt ``memeye-answer-timeout-retry-01`` targets
    ``scripts.aml_multimodal_memeye.answer_rotation``. On frozen commit ``d633de57``, the fake
    first read timeout escaped directly. This node caught it and failed at the intended
    ``error is None`` assertion, before either a second identical request or a cost reservation
    could occur. The repair makes at most three total Answer attempts, reuses the exact payload,
    and reserves the registered maximum request cost immediately after each ambiguous timeout.
    A deliberate mutation that skipped the ``reserve_timeout_cost`` callback kept the retry green
    but made this node fail at the intended reservation assertion. Restoring the callback made the
    same node green.
    """
    captured: list[dict] = []
    reserved: list[float] = []

    def fake_call(self: JsonClient, path: str, payload: dict | None = None) -> HttpResult:
        assert path == "/chat/completions"
        assert payload is not None
        captured.append(payload)
        if len(captured) == 1:
            raise TimeoutError("ambiguous provider timeout")
        return HttpResult(
            200,
            {
                "choices": [{"message": {"content": "A"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "cost": 0.001,
                },
            },
            1.0,
            10,
            1,
        )

    monkeypatch.setattr(JsonClient, "call", fake_call)
    error: TimeoutError | None = None
    usage: dict = {}
    try:
        _, usage = answer_rotation(
            JsonClient("https://example.invalid", "token", sleep=lambda _: None),
            api_key="answer-key",
            system_prompt="system",
            parts=[{"type": "text", "text": "question"}],
            reserve_timeout_cost=reserved.append,
        )
    except TimeoutError as exc:
        error = exc

    assert error is None, "the bounded Answer retry policy did not handle the read timeout"
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert usage["attempts"] == 2
    assert usage["timeout_retry_count"] == 1
    assert usage["timeout_reserved_cost_usd"] == pytest.approx(0.117824)
    assert reserved == [pytest.approx(0.117824)]


def test_memory_api_timeout_covers_large_multimodal_responses() -> None:
    """The hosted-memory client permits the registered 30 MiB response budget to arrive.

    Red proof receipt ``memeye-memory-timeout-01`` targets
    ``scripts.aml_multimodal_memeye.JsonClient``. Frozen commit ``a64cacab`` completed 24 MM1
    answer rotations, then the twenty-fifth Search exceeded the default 120-second read timeout.
    This assertion was red at 120 seconds. Raising only the hosted-memory client default to 600
    seconds leaves the explicitly bounded 180-second Answer client unchanged.
    """
    assert JsonClient("https://example.invalid", "token").timeout == 600


def _rotation(em: float, any_10: float, any_100: float = 1.0) -> dict:
    return {
        "selected_position": "A",
        "valid_choice": True,
        "em": em,
        "retrieval": {
            "clue_count": 1,
            "first_clue_rank": 1,
            "mrr": 1.0,
            "any_recall_at_5": any_10,
            "any_recall_at_10": any_10,
            "any_recall_at_100": any_100,
            "complete_recall_at_5": any_10,
            "complete_recall_at_10": any_10,
            "complete_recall_at_100": any_100,
            "clue_fraction_at_5": any_10,
            "clue_fraction_at_10": any_10,
            "clue_fraction_at_100": any_100,
        },
    }


def test_aggregate_uses_all_rotations_and_axis_membership() -> None:
    """Debiased EM averages rotations before questions and keeps both axis labels.

    Red proof receipt ``memeye-debiased-aggregate-01`` targets
    ``scripts.aml_multimodal_memeye.aggregate_questions``. Aggregating only the final rotation
    changed mean debiased EM from 0.5 to 1.0 and failed the intended equality. Restoring all four
    rotations made this node green.
    """
    rows = [
        {
            "axes": ["X2", "Y1"],
            "rotations": [_rotation(0, 0), _rotation(0, 0), _rotation(1, 1), _rotation(1, 1)],
            "debiased_em": 0.5,
            "strict_exact_match": False,
        }
    ]
    aggregate = aggregate_questions(rows)
    assert aggregate["rotation_count"] == 4
    assert aggregate["mean_debiased_em"] == 0.5
    assert aggregate["any_recall_at_10"] == 0.5
    assert aggregate["by_axis"]["X2"]["mean_debiased_em"] == 0.5
    assert aggregate["by_axis"]["Y1"]["question_count"] == 1


def _arm(name: str, answer: float, recall_10: float, recall_100: float = 1.0) -> dict:
    return {
        "arm": name,
        "complete": True,
        "identity": {
            "dataset_revision": "r",
            "dataset_json": "j",
            "dataset_sha256": "d",
            "memeye_commit": "m",
            "prompt_path": "p",
            "prompt_sha256": "h",
            "answer_model": "a",
            "answer_temperature": 0,
            "answer_token_budget": 117760,
            "top_k": 100,
        },
        "add": {"count": 72, "replay_passed": True},
        "aggregate": {
            "question_count": 29,
            "rotation_count": 116,
            "mean_debiased_em": answer,
            "any_recall_at_10": recall_10,
            "any_recall_at_100": recall_100,
        },
        "provider_spend_usd": 1,
        "cleanup": {"passed": True},
    }


def test_selector_requires_answer_and_retrieval_gain() -> None:
    """The full admission verdict cannot be earned by a retrieval proxy alone.

    Red proof receipt ``memeye-selector-answer-gate-01`` targets
    ``scripts.select_aml_multimodal_memeye.decide``. Removing ``dual_answer_gain`` from the full
    conjunction promoted this fixture to ``ADMIT_FULL_MEMEYE`` and failed the expected
    ``RETRIEVAL_ONLY`` verdict. Restoring the conjunction made this node green.
    """
    arms = {
        "MM0_caption": _arm("MM0_caption", 0.40, 0.40),
        "MM1_preserve": _arm("MM1_preserve", 0.50, 0.40),
        "MM2_dual": _arm("MM2_dual", 0.50, 0.50),
    }
    verdict = decide(arms)
    assert verdict["verdict"] == "RETRIEVAL_ONLY"
    assert verdict["conditions"]["dual_answer_gain"] is False


def test_selector_applies_cost_ceiling_across_all_arms() -> None:
    """Three individually cheap arms cannot exceed the one experiment-wide ceiling.

    Red proof receipt ``memeye-selector-global-cost-01`` targets
    ``scripts.select_aml_multimodal_memeye.decide``. The v1 proof showed that checking each arm
    against its ceiling while omitting their sum admitted an over-budget experiment. For v2, this
    node was red when the old USD 10 ceiling rejected three USD 4 arms. The repaired selector admits
    their USD 12 total and still rejects three USD 9 arms above the newly frozen USD 25 ceiling.
    """
    arms = {
        "MM0_caption": _arm("MM0_caption", 0.40, 0.40),
        "MM1_preserve": _arm("MM1_preserve", 0.50, 0.40),
        "MM2_dual": _arm("MM2_dual", 0.55, 0.50),
    }
    for payload in arms.values():
        payload["provider_spend_usd"] = 4
    assert decide(arms)["verdict"] != "INVALID"
    for payload in arms.values():
        payload["provider_spend_usd"] = 9
    verdict = decide(arms)
    assert verdict["verdict"] == "INVALID"
    assert "the experiment exceeded the registered provider cost ceiling" in verdict["reasons"]


def test_v2_runtime_uses_the_frozen_cost_ceiling() -> None:
    """Runtime enforcement and the preregistered v2 amount cannot drift.

    Red proof receipt ``memeye-runtime-cost-v2-01`` targets
    ``scripts/aml_multimodal_memeye.py``. The test was red while the runner retained its v1 literal
    USD 10 checks. Naming the USD 25 constant and using it at both stop boundaries makes the
    contract reviewable and keeps the carried-forward spend ledger authoritative.
    """
    script = Path("scripts/aml_multimodal_memeye.py").read_text(encoding="utf-8")
    assert "EXPERIMENT_COST_CEILING_USD = 25.0" in script
    assert script.count("EXPERIMENT_COST_CEILING_USD") == 3


def test_selector_rejects_a_missing_arm_cost() -> None:
    """A missing cost is invalid even when the v2 ceiling exceeds the old fallback literal."""
    arms = {
        "MM0_caption": _arm("MM0_caption", 0.40, 0.40),
        "MM1_preserve": _arm("MM1_preserve", 0.50, 0.40),
        "MM2_dual": _arm("MM2_dual", 0.55, 0.50),
    }
    del arms["MM2_dual"]["provider_spend_usd"]
    verdict = decide(arms)
    assert verdict["verdict"] == "INVALID"
    assert "MM2_dual provider cost is missing or invalid" in verdict["reasons"]


def test_independent_audit_rejects_tampered_aggregate(tmp_path: Path) -> None:
    """The audit recomputes scores instead of trusting the arm summary.

    Red proof receipt ``memeye-audit-recompute-01`` targets
    ``scripts.audit_aml_multimodal_memeye.audit``. Replacing the recomputed EM comparison with a
    self-comparison let the tampered 0.99 aggregate pass and failed this node at ``passed is
    False``. Restoring the independent comparison made it green.
    """
    arm_hashes = {}
    for name in ARMS:
        rotations = [_rotation(1, 1) for _ in range(4)]
        questions = [
            {
                "rotations": rotations,
                "debiased_em": 1.0,
                "strict_exact_match": True,
            }
            for _ in range(29)
        ]
        aggregate = {
            "question_count": 29,
            "rotation_count": 116,
            "mean_debiased_em": 0.99 if name == "MM2_dual" else 1.0,
            "strict_question_accuracy": 1.0,
            "valid_choice_rate": 1.0,
            "any_recall_at_10": 1.0,
            "any_recall_at_100": 1.0,
            "complete_recall_at_10": 1.0,
            "clue_fraction_at_10": 1.0,
            "mrr": 1.0,
        }
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "arm": name,
                    "complete": True,
                    "cleanup": {"passed": True},
                    "questions": questions,
                    "aggregate": aggregate,
                }
            ),
            encoding="utf-8",
        )
        arm_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "selection.json").write_text(
        json.dumps({"arm_sha256": arm_hashes}), encoding="utf-8"
    )
    result = audit(tmp_path)
    assert result["passed"] is False
    assert "MM2_dual aggregate mismatch for mean_debiased_em" in result["failures"]
