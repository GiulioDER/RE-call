"""RC-3 harness: each arm is the SHIPPED dated prompt plus exactly its rule. Each test names its
red proof."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recall.evidence import DATED_SYSTEM_PROMPT, render_dated_evidence_prompt  # noqa: E402
from reasoning_hb_rules import (  # noqa: E402
    BENCHMARK_CONTRACT,
    SINGLE_RULES,
    build_bundle,
    render,
    system_prompt,
)

ITEMS = [{"id": f"raw_{n}", "content": f"[2023-05-{10 + n:02d} 09:00 UTC] note {n}",
          "created_at": f"2023-05-{10 + n:02d}T09:00:00Z"} for n in range(12)]


def test_d_is_the_shipped_default_byte_for_byte() -> None:
    """Invariant: the base arm is master's dated rendering itself, system and user.

    Red proof: returning `DATED_SYSTEM_PROMPT + BENCHMARK_CONTRACT` for D fails the equality.
    """
    bundle = build_bundle("q", ITEMS, strip_header=True)
    assert render("D", bundle, "2023/05/30") == render_dated_evidence_prompt(bundle, "2023/05/30")
    assert render("D2", bundle, "2023/05/30") == render_dated_evidence_prompt(bundle, "2023/05/30")


def test_each_single_rule_arm_adds_exactly_its_own_rule() -> None:
    """Invariant: a single-rule arm is the shipped prompt plus that rule and no other, so a
    difference from D is that rule's; the user message is identical in every arm.

    Red proof: mapping R11 to R10's text (`SINGLE_RULES["R10"]`) fails the "detailed" check.
    """
    bundle = build_bundle("q", ITEMS, strip_header=True)
    _, base_user = render("D", bundle, "d")
    for arm, rule in SINGLE_RULES.items():
        system, user = render(arm, bundle, "d")
        assert system == DATED_SYSTEM_PROMPT + rule and user == base_user
        assert sum(other in system for other in SINGLE_RULES.values()) == 1
    assert "Make answers detailed" in system_prompt("R11")
    assert "most recent value" in system_prompt("R10")


def test_hb_is_rc1s_contract_and_contains_every_rule_word_for_word() -> None:
    """Invariant: HB reproduces RC-1's four rules byte for byte, and each single rule carries the
    same words (only its number differs), so a single arm tests the rule HB contained.

    Red proof: editing a word of rule 9 in `SINGLE_RULES["R9"]` fails the containment check.
    """
    assert system_prompt("HB") == DATED_SYSTEM_PROMPT + BENCHMARK_CONTRACT
    for rule in SINGLE_RULES.values():
        body = rule.split("8. ", 1)[1]
        assert body in BENCHMARK_CONTRACT


def test_header_stripping_follows_the_dataset() -> None:
    """Invariant: LongMemEval items lose C9's date header (as in RC-1), LoCoMo items have none to
    lose (as in RC-2); either way the item's date comes from `created_at`.

    Red proof: stripping unconditionally is invisible here, so the check is on the LME side:
    dropping the substitution leaves the header and fails the text equality.
    """
    stripped = build_bundle("q", ITEMS, strip_header=True)
    kept = build_bundle("q", ITEMS, strip_header=False)
    assert stripped.items[0].text == "note 0" and kept.items[0].text.startswith("[2023-05-10")
    assert stripped.items[0].indexed_at == datetime(2023, 5, 10, 9, 0, tzinfo=timezone.utc)
