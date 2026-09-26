"""``compare`` in ``scripts/aml_c9_compile_cost_replay.py`` reports the pre-registered measures.

Red proof, 2026-09-26: counting ``A_truncated_then_compiled`` over every Add instead of over
``truncated_first`` (``sum(a[i]["compiled"] for i in ids)``) failed
``out["A_truncated_then_compiled"] == 0`` with 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.aml_c9_compile_cost_replay import compare


def _row(ident: str, arm: str, calls: list[dict], compiled: bool, error: str | None) -> dict:
    cost = sum((c.get("prompt_tokens", 0) * 0.15 + c.get("completion_tokens", 0) * 0.6) / 1e6 for c in calls)
    return {"id": ident, "stratum": [0, 1], "arm": arm, "calls": calls, "cost_usd": cost,
            "accepted_records": 3 if compiled else 0, "error": error, "compiled": compiled}


def test_compare_reports_cost_ratio_truncation_and_skips(tmp_path: Path, capsys) -> None:
    cut = {"prompt_tokens": 45_000, "completion_tokens": 2_400, "finish_reason": "length"}
    ok = {"prompt_tokens": 3_000, "completion_tokens": 900, "finish_reason": "stop"}
    a = [_row("x", "A", [cut, cut, cut], False, "JSONDecodeError"), _row("y", "A", [ok], True, None)]
    b = [_row("x", "B", [], False, "CompilerInputTooLarge"), _row("y", "B", [ok], True, None)]
    (tmp_path / "A.jsonl").write_text("\n".join(map(json.dumps, a)))
    (tmp_path / "B.jsonl").write_text("\n".join(map(json.dumps, b)))

    compare(argparse.Namespace(a=tmp_path / "A.jsonl", b=tmp_path / "B.jsonl"))
    out = json.loads(capsys.readouterr().out)

    assert out["A_truncated_first_answers"] == 1
    assert out["A_truncated_then_compiled"] == 0
    assert out["compiled_kept_B_over_A"] == 1.0
    assert out["by_first_call_prompt_tokens"]["40000"]["B_skipped"] == 1
    assert 0 < out["cost_ratio_B_over_A"] < 0.1
