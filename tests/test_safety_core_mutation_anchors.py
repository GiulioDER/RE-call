"""Every safety-core mutation still finds the code it mutates.

`scripts/safety_core_mutations.py` proves the trust, auth, generation and provenance guards are
covered by breaking each one and watching its tests fail. It is not run in CI, and a mutation whose
anchor no longer matches is only reported as STALE and skipped, so moving or rewording guarded code
silently drops that guard from the sweep. On 2026-10-03 nine of its 21 mutations were stale: six
after S4 and S7c moved trust code into `recall/trust_verdicts.py` and `recall/trust_gate.py`, and
three after earlier rewrites of the MCP tool guard and the evidence-card trust check.

This runs no mutation and no other test; it only checks that each anchor matches exactly once in its
module (zero means stale, two means the mutation could hit the wrong place) and that each
replacement differs from its anchor.

Red proof, recorded 2026-10-03 on a Linux test host: restoring the pre-fix anchor
`b"            limiter.check(tenant, _SCOPE_BUDGETS[scope])"` for "per tenant rate limit check is
removed" fails `test_every_mutation_anchor_matches_its_module_exactly_once` naming that label with
count 0. Unmodified, it passes.
"""

from __future__ import annotations

from scripts.safety_core_mutations import MUTATIONS, ROOT


def test_every_mutation_anchor_matches_its_module_exactly_once() -> None:
    wrong = [
        f"{mutation.module.relative_to(ROOT).as_posix()}: {mutation.label} "
        f"(anchor found {mutation.module.read_bytes().count(mutation.find)} times)"
        for mutation in MUTATIONS
        if mutation.module.read_bytes().count(mutation.find) != 1
    ]
    assert wrong == [], "safety-core mutations that no longer reach their code:\n" + "\n".join(wrong)


def test_every_mutation_changes_its_anchor() -> None:
    assert [mutation.label for mutation in MUTATIONS if mutation.find == mutation.replace] == []
